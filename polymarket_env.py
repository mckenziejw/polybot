"""
Gymnasium environment for Polymarket 5-minute BTC Up/Down binary markets.

Each episode = one resolved market window.
The agent trades Yes and No tokens, but cannot hold both simultaneously.
Buying one side forces a flatten of the other side first.

Action space (Discrete 6, maskable):
    0  Hold
    1  Buy Yes small  ($50 or min order, whichever is greater)
    2  Buy Yes large  ($100)
    3  Buy No small
    4  Buy No large
    5  Sell           (flatten current position)

Observation space:
    - Contract features: Yes book (5 levels bid/ask, mid, spread, imbalance)
                         No book  (5 levels bid/ask, mid, spread, imbalance)
    - Position features: yes_position, no_position, yes_avg_cost, no_avg_cost,
                         unrealized_pnl, capital_at_risk_pct
    - Time feature:      time_remaining_pct (1.0 at open, 0.0 at close)
    - BTC features:      mid_price (normalized), log_return, ret_5s, ret_15s,
                         ret_60s, rvol_30s, rvol_60s, rvol_300s
    - Staleness:         yes_staleness_norm, no_staleness_norm
                         ms since last real tick / MAX_STALENESS_MS, clipped [0,1]
                         0.0 = fresh data this bar, 1.0 = stale >= 5 seconds

Fill model: mid-price fill, immediate (no slippage simulation yet).

Reward:
    - Step:     alpha * realized_pnl_delta * exposure_penalty_multiplier
                nonzero only when a position is closed (ACTION_SELL)
                exposure_penalty = 1 - sigmoid(k * (exposure - midpoint))
                  ~1.0 at <3% exposure, ~0.50 at 6%, ~0.08 at 10%
                sum of step rewards targets ~20-30% of terminal magnitude
    - Terminal: full episode PnL settled at binary resolution (1.0 or 0.0)
                unscaled — remains the dominant training signal
                markets without a known resolution are excluded at init

Usage:
    from env.polymarket_env import PolymarketEnv
    env = PolymarketEnv(
        book_dir="data/telonex_book_snapshots",
        btc_path="data/btc_quotes/btcusdt_quotes.parquet",
    )
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(action)
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

from market_analysis import ResolutionStore

log = logging.getLogger(__name__)


# Book health thresholds (from spread/size distribution analysis)
BOOK_MAX_SPREAD = 0.05   # spreads above 5 cents flag a thin/unreliable book (p95)
BOOK_MIN_SIZE   = 20.0   # top-of-book size below $20 is thinner than 75% of normal obs


def _book_mid(row) -> tuple[float, bool]:
    """
    Compute mid price from a single book row and return a health flag.

    Returns (mid, is_healthy) where is_healthy=True means the book has
    sufficient depth and tight spread to be trusted for pricing.
    """
    bid    = float(row.get('bid_price_1', 0) or 0)
    ask    = float(row.get('ask_price_1', 0) or 0)
    bid_sz = float(row.get('bid_size_1',  0) or 0)
    ask_sz = float(row.get('ask_size_1',  0) or 0)

    has_bid = bid > 0 and bid_sz >= BOOK_MIN_SIZE
    has_ask = ask > 0 and ask_sz >= BOOK_MIN_SIZE and ask <= 1.0

    if has_bid and has_ask:
        spread = ask - bid
        if spread <= BOOK_MAX_SPREAD:
            return (bid + ask) / 2.0, True   # healthy two-sided book
        else:
            return (bid + ask) / 2.0, False  # wide spread — use mid but flag unhealthy
    elif has_ask and not has_bid:
        return ask, False   # ask-only: near-certain winner, but book is one-sided
    elif has_bid and not has_ask:
        return bid, False   # bid-only: near-certain loser, but book is one-sided
    else:
        stored = float(row.get('mid_price', 0) or 0)
        return (stored if 0 < stored < 1 else 0.5), False


def _robust_mid_pair(yes_row, no_row) -> tuple[float, float]:
    """
    Compute manipulation-resistant mid prices for both tokens simultaneously.

    On a binary market P(Yes) + P(No) = 1 at resolution. When one token's
    book is unhealthy (thin, wide spread, or one-sided), anchor its price
    to the complement of the other token's price rather than trusting the
    thin book — this prevents spoofed bids from distorting our fill prices.

    If both books are unhealthy (e.g. illiquid market overall), fall back
    to each token's own best estimate independently.
    """
    yes_mid, yes_healthy = _book_mid(yes_row)
    no_mid,  no_healthy  = _book_mid(no_row)

    if yes_healthy and not no_healthy:
        no_mid = 1.0 - yes_mid          # anchor No to complement of healthy Yes
    elif no_healthy and not yes_healthy:
        yes_mid = 1.0 - no_mid          # anchor Yes to complement of healthy No
    # both healthy or both unhealthy: use each token's own mid

    return float(np.clip(yes_mid, 0.0, 1.0)), float(np.clip(no_mid, 0.0, 1.0))


def _robust_mid(yes_row, no_row=None) -> float:
    """Single-token wrapper for call sites that only need one price.
    When no_row is provided, uses complement anchoring. Otherwise solo estimate."""
    if no_row is not None:
        yes_mid, _ = _robust_mid_pair(yes_row, no_row)
        return yes_mid
    mid, _ = _book_mid(yes_row)
    return float(np.clip(mid, 0.0, 1.0))





# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STARTING_BANKROLL  = 10_000.0
MIN_ORDER          = 5.0
SMALL_ORDER_PCT    = 0.005   # 0.5% of bankroll
LARGE_ORDER_PCT    = 0.010   # 1.0% of bankroll
MAX_POSITION_PCT   = 0.10    # 10.0% of bankroll — hard mask cap
TAKER_FEE_PCT      = 0.02    # 2% taker fee on notional, both sides of each trade
MIN_BUY_PRICE      = 0.10    # don't buy tokens below this price (near-certain loser)
MAX_BUY_PRICE      = 0.90    # don't buy tokens above this price (near-certain winner, fee-drag)

# ---------------------------------------------------------------------------
# Reward shaping
# ---------------------------------------------------------------------------
# Step reward: alpha * realized_pnl_delta (net of fees)
# Terminal reward: full episode PnL (unscaled) — dominant signal
#
# Fees (TAKER_FEE_PCT) are the primary deterrent against churning.
# The exposure penalty has been removed — with fees a round trip costs
# ~4% of notional and is self-deterring without additional shaping.
# STEP_REWARD_ALPHA removed — mark-to-market reward needs no scaling

N_BOOK_LEVELS      = 5
MARKET_DURATION_MS = 900_000  # 5 minutes in milliseconds
RESAMPLE_MS        = 100       # 100ms bars → 3,000 steps/episode

# Action indices
ACTION_HOLD          = 0
ACTION_BUY_YES_SMALL = 1
ACTION_BUY_YES_LARGE = 2
ACTION_BUY_NO_SMALL  = 3
ACTION_BUY_NO_LARGE  = 4
ACTION_SELL          = 5
N_ACTIONS            = 6

# Observation dimensions
# Per token: mid, spread, imbalance, bid_price x5, bid_size x5, ask_price x5, ask_size x5 = 23
OBS_PER_TOKEN  = 3 + N_BOOK_LEVELS * 4   # 23
OBS_CONTRACT   = OBS_PER_TOKEN * 2        # 46  (Yes + No)
OBS_POSITION   = 6                        # yes_pos, no_pos, yes_avg, no_avg, upnl, risk_pct
OBS_TIME       = 1                        # time_remaining_pct
OBS_BTC        = 8                        # mid_norm, log_ret, ret_5s, ret_15s, ret_60s, rvol_30s, rvol_60s, rvol_300s
OBS_STALENESS  = 2                        # yes_staleness_norm, no_staleness_norm
OBS_DIM        = OBS_CONTRACT + OBS_POSITION + OBS_TIME + OBS_BTC + OBS_STALENESS  # 63

# Staleness normalisation cap: staleness >= this value maps to 1.0
# 5 seconds covers the longest quiet periods; beyond this the book is
# effectively dead and the exact duration carries no additional signal.
MAX_STALENESS_MS = 5_000


def _resample_book(
    df: pd.DataFrame,
    open_ms: int,
    close_ms: int,
    freq_ms: int,
) -> pd.DataFrame:
    """
    Resample a single token's book data to a uniform fixed-frequency grid.

    - Grid spans [open_ms, close_ms) in steps of freq_ms
    - Each bar takes the last observed value within that window
    - Bars with no data are forward-filled from the previous bar,
      then backward-filled for any leading gaps at market open
    - Adds a `staleness_ms` column: milliseconds since the last real tick
      at each bar. Forward-filled bars accumulate staleness; bars with
      fresh data have staleness == 0.
    """
    # Build the target grid
    grid = np.arange(open_ms, close_ms, freq_ms)

    if df.empty:
        result = pd.DataFrame(index=grid)
        result["staleness_ms"] = MAX_STALENESS_MS  # fully stale if no data at all
        return result

    df = df.set_index("exchange_timestamp").sort_index()

    # Assign each tick to its 100ms bar (floor division)
    df["bar_ts"] = (df.index // freq_ms) * freq_ms

    # Last-value within each bar; also record the actual tick timestamp
    # so we can compute staleness after reindexing.
    df["last_tick_ts"] = df.index  # the real exchange_timestamp of the last tick
    resampled = df.groupby("bar_ts").last()

    # Reindex to full grid — NaN where no tick fell in the bar
    resampled = resampled.reindex(grid)

    # Forward-fill book data and last_tick_ts, then backward-fill leading gaps.
    # After ffill, last_tick_ts at every bar is the most recent real tick ts.
    resampled = resampled.ffill().bfill()

    # staleness_ms: how long ago the last real tick arrived relative to this bar.
    # Clipped to 0 — ticks that land after bar_start but within the bar produce
    # a negative raw value (bar_ts < last_tick_ts), which we treat as 0 (fresh).
    resampled["staleness_ms"] = (resampled.index - resampled["last_tick_ts"]).clip(lower=0)

    # Drop the helper column — not needed in the env
    resampled = resampled.drop(columns=["last_tick_ts"])

    return resampled


class PolymarketEnv(gym.Env):
    """
    Polymarket 5-minute BTC binary market trading environment.

    Parameters:
        book_dir:     directory containing per-market parquet files
        btc_path:     path to btcusdt_quotes.parquet (1s bars with features)
        bankroll:     starting capital per episode
        deterministic: if True, episodes cycle through markets in order;
                       if False, sampled randomly (default for training)
        market_slugs: optional list of slugs to restrict episode sampling to
        seed:         random seed
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        book_dir: str = "data/telonex_book_snapshots",
        btc_path: str = "data/btc_quotes/btcusdt_quotes.parquet",
        resolution_store: Optional[ResolutionStore] = None,
        resolution_path: str = "data/resolutions.json",
        bankroll: float = STARTING_BANKROLL,
        deterministic: bool = False,
        market_slugs: Optional[list[str]] = None,
        seed: Optional[int] = None,
    ):
        super().__init__()

        self.book_dir         = Path(book_dir)
        self.btc_path         = Path(btc_path)
        self.initial_bankroll = bankroll
        self.deterministic    = deterministic

        # Resolution store — accepts an already-constructed ResolutionStore
        # (preferred for training, where you pre-populate via fetch_batch before
        # initialising the env) or constructs one from resolution_path if not
        # supplied (convenient for quick single-env usage).
        #
        # The store is used READ-ONLY during training (get() only, no API calls).
        # Call PolymarketEnv.prefetch_resolutions() or ResolutionStore.fetch_batch()
        # externally before training to ensure the cache is fully populated.
        if resolution_store is not None:
            self._resolution_store = resolution_store
        else:
            self._resolution_store = ResolutionStore(
                cache_path=Path(resolution_path)
            )

        n_cached = len(self._resolution_store._cache)
        if n_cached == 0:
            log.warning(
                "ResolutionStore cache is empty. Terminal positions will be "
                "settled at mid-price (incorrect). Call "
                "env.prefetch_resolutions() before training."
            )
        else:
            log.info(f"ResolutionStore: {n_cached} resolutions available")

        # Discover available markets
        all_files = sorted(self.book_dir.glob("btc-updown-15m-*.parquet"))
        if market_slugs:
            slug_set = set(market_slugs)
            self.market_files = [f for f in all_files if f.stem in slug_set]
        else:
            self.market_files = all_files

        if not self.market_files:
            raise ValueError(f"No market files found in {book_dir}")

        # Filter markets: must have sufficient data AND a known resolution.
        # Markets without resolution cannot produce a correct terminal reward
        # and must be excluded — falling back to mid-price settlement would
        # corrupt the training signal for those episodes.
        MIN_ROWS = 500
        kept, dropped_thin, dropped_no_res = [], [], []

        import pyarrow.parquet as pq
        for f in self.market_files:
            slug   = f.stem
            # Count rows without loading data — use parquet metadata
            n_rows = pq.read_metadata(f).num_rows
            if n_rows < MIN_ROWS:
                dropped_thin.append(slug)
                continue
            if self._resolution_store.get(slug) is None:
                dropped_no_res.append(slug)
                continue
            kept.append(f)

        self.market_files = kept

        if dropped_thin:
            log.info(f"Dropped {len(dropped_thin)} markets with < {MIN_ROWS} rows")
        if dropped_no_res:
            total = len(kept) + len(dropped_thin) + len(dropped_no_res)
            log.info(
                f"Dropped {len(dropped_no_res)} markets with no resolution "
                f"({100*len(dropped_no_res)/total:.1f}% of total)"
            )
        if not self.market_files:
            raise ValueError(
                f"No usable markets in {book_dir} after filtering. "
                f"Thin: {len(dropped_thin)}, no resolution: {len(dropped_no_res)}. "
                f"Run env.prefetch_resolutions() to populate the cache."
            )

        log.info(f"PolymarketEnv: {len(self.market_files)} markets ready for training")

        # Load BTC quotes index (timestamps only) for fast lookup
        log.info("Loading BTC quotes...")
        self._btc_df = pd.read_parquet(self.btc_path).set_index("timestamp_ms")

        # Action and observation spaces
        self.action_space = spaces.Discrete(N_ACTIONS)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
        )

        # Episode state — initialized in reset()
        self._yes_df:         pd.DataFrame | None = None
        self._no_df:          pd.DataFrame | None = None
        self._timestamps:     np.ndarray | None   = None
        self._step_idx:       int                 = 0
        self._market_open_ms: int                 = 0
        self._market_close_ms: int                = 0
        self._episode_idx:    int                 = 0

        # Resolution values for current episode: 1.0 if token won, 0.0 if lost, None if unknown
        self._yes_resolved:   float | None        = None
        self._no_resolved:    float | None        = None

        self._bankroll:       float = bankroll
        self._yes_position:   float = 0.0   # dollar value of Yes tokens held
        self._no_position:    float = 0.0   # dollar value of No tokens held
        self._yes_avg_cost:   float = 0.0   # average cost per dollar invested in Yes
        self._no_avg_cost:    float = 0.0   # average cost per dollar invested in No
        self._realized_pnl:   float = 0.0
        self._fees_paid:      float = 0.0

        self._rng = np.random.default_rng(seed)

    def prefetch_resolutions(self) -> tuple[int, int]:
        """
        Fetch and cache resolutions for all market files known to this env.

        Hits the Gamma API for any slug not already in the cache. Call this
        once before training — not during, since API calls during rollouts
        would stall the training loop.

        Returns (n_resolved, n_total).

        Example:
            env = PolymarketEnv(...)
            resolved, total = env.prefetch_resolutions()
            print(f"Resolved {resolved}/{total} markets")
        """
        slugs   = [f.stem for f in self.market_files]
        results = self._resolution_store.fetch_batch(slugs)
        return len(results), len(slugs)

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Select market for this episode
        if self.deterministic:
            idx = self._episode_idx % len(self.market_files)
        else:
            idx = self._rng.integers(0, len(self.market_files))

        self._episode_idx += 1
        market_file = self.market_files[idx]
        self._load_market(market_file)

        # Reset position state
        self._bankroll      = self.initial_bankroll
        self._yes_position  = 0.0
        self._no_position   = 0.0
        self._yes_avg_cost  = 0.0
        self._no_avg_cost   = 0.0
        self._realized_pnl  = 0.0
        self._fees_paid     = 0.0
        self._step_idx      = 0

        obs  = self._get_obs()
        info = self._get_info()
        return obs, info

    def _portfolio_value(self) -> float:
        """Current mark-to-market portfolio value: bankroll + unrealized PnL."""
        idx     = min(self._step_idx, len(self._timestamps) - 1)
        yes_mid, no_mid = _robust_mid_pair(self._yes_df.iloc[idx], self._no_df.iloc[idx])
        return self._bankroll + self._unrealized_pnl(yes_mid, no_mid)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        assert self._yes_df is not None, "Call reset() before step()"

        terminated = False
        truncated  = False

        # Snapshot portfolio value before action for mark-to-market delta
        value_before = self._portfolio_value()

        # Execute action
        if action != ACTION_HOLD:
            self._execute_action(action)

        self._step_idx += 1

        # -----------------------------------------------------------------
        # Reward: mark-to-market portfolio value delta
        # -----------------------------------------------------------------
        # reward = change in (bankroll + unrealized_pnl) this step.
        #
        # This rewards holding winners and penalises holding losers on every
        # step, not just when positions are closed. Fees are baked in — a
        # buy immediately reduces bankroll by size+fee, so the reward is
        # negative unless the position is already worth more than it cost.
        #
        # Terminal step settles at binary resolution (0 or 1), which gives
        # the true final signal regardless of mid-price accuracy.
        # -----------------------------------------------------------------
        if self._step_idx >= len(self._timestamps):
            terminated = True
            reward     = self._close_episode()
        else:
            value_after = self._portfolio_value()
            reward      = value_after - value_before

        obs  = self._get_obs() if not terminated else np.zeros(OBS_DIM, dtype=np.float32)
        info = self._get_info()
        return obs, reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        """
        Returns boolean mask of valid actions for current state.
        Used by sb3-contrib MaskablePPO.

        Three independent buy-side constraints (all must pass):
          1. Exposure cap  — capital-at-risk < 10% of initial bankroll
          2. Affordability — available bankroll >= order cost
          3. Side conflict — can't hold Yes and No simultaneously
             (buying the opposite side first flattens the current one,
              but that flatten must itself be affordable)

        Note: order sizes are fixed off initial_bankroll, not current
        bankroll, so affordability degrades gracefully as the agent loses.
        """
        masks = np.ones(N_ACTIONS, dtype=bool)

        # Fixed order costs (same as _execute_action)
        small_cost = max(MIN_ORDER, self.initial_bankroll * SMALL_ORDER_PCT)
        large_cost = self.initial_bankroll * LARGE_ORDER_PCT

        capital_at_risk = self._yes_position + self._no_position
        hard_cap        = self.initial_bankroll * MAX_POSITION_PCT
        at_cap          = capital_at_risk >= hard_cap
        flat            = (self._yes_position == 0.0 and self._no_position == 0.0)
        bankroll        = self._bankroll

        # Current mid prices for price range filtering
        idx     = min(self._step_idx, len(self._timestamps) - 1)
        yes_mid, no_mid = _robust_mid_pair(self._yes_df.iloc[idx], self._no_df.iloc[idx])

        # Price range: only allow buys when mid is within [MIN_BUY_PRICE, MAX_BUY_PRICE].
        # Contracts near 0 or 1 are nearly resolved — buying them is low-edge,
        # high-fee-drag, and signals the agent is chasing near-certain outcomes.
        yes_in_range = MIN_BUY_PRICE <= yes_mid <= MAX_BUY_PRICE
        no_in_range  = MIN_BUY_PRICE <= no_mid  <= MAX_BUY_PRICE

        holding_yes = self._yes_position > 0
        holding_no  = self._no_position > 0

        # --- Buy Yes ---
        # Masked if: at cap, can't afford, price OOB, or holding No (must sell first)
        if at_cap or bankroll < small_cost or not yes_in_range or holding_no:
            masks[ACTION_BUY_YES_SMALL] = False
        if at_cap or bankroll < large_cost or not yes_in_range or holding_no:
            masks[ACTION_BUY_YES_LARGE] = False

        # --- Buy No ---
        # Masked if: at cap, can't afford, price OOB, or holding Yes (must sell first)
        if at_cap or bankroll < small_cost or not no_in_range or holding_yes:
            masks[ACTION_BUY_NO_SMALL] = False
        if at_cap or bankroll < large_cost or not no_in_range or holding_yes:
            masks[ACTION_BUY_NO_LARGE] = False

        # --- Sell ---
        if flat:
            masks[ACTION_SELL] = False

        return masks

    # ------------------------------------------------------------------
    # Internal — market loading
    # ------------------------------------------------------------------

    def _load_market(self, path: Path):
        """
        Load a market parquet, resample to RESAMPLE_MS bars (last-value),
        and align Yes/No tokens on a fixed uniform timestamp grid.
        Also resolves terminal settlement values from the resolution cache.
        """
        slug = path.stem
        try:
            ts = int(slug.split("-")[-1])
        except ValueError:
            raise ValueError(f"Cannot parse timestamp from slug: {slug}")

        self._market_open_ms  = ts * 1000
        self._market_close_ms = self._market_open_ms + MARKET_DURATION_MS

        df = pd.read_parquet(path)
        df = df.sort_values("exchange_timestamp").reset_index(drop=True)

        # Separate Yes (Up) and No (Down) tokens
        yes_raw = df[df["token_label"] == "Up"].copy()
        no_raw  = df[df["token_label"] == "Down"].copy()

        # ------------------------------------------------------------------
        # Resolution lookup
        # ------------------------------------------------------------------
        # Preprocessed files (from preprocess_markets.py) embed resolved_value
        # directly as a column — no ResolutionStore lookup needed at episode init.
        # Raw Telonex files fall back to the ResolutionStore cache.
        self._yes_resolved = None
        self._no_resolved  = None

        if "resolved_value" in df.columns:
            # Preprocessed format: resolved_value is per-row, constant per token
            if not yes_raw.empty:
                self._yes_resolved = float(yes_raw["resolved_value"].iloc[0])
            if not no_raw.empty:
                self._no_resolved = float(no_raw["resolved_value"].iloc[0])
        else:
            # Raw format: look up via ResolutionStore (cache only, no API calls)
            yes_asset_id = yes_raw["asset_id"].iloc[0] if not yes_raw.empty else None
            no_asset_id  = no_raw["asset_id"].iloc[0]  if not no_raw.empty  else None
            resolution   = self._resolution_store.get(slug)
            if resolution is None:
                log.warning(f"{slug}: no resolution in cache — should have been filtered at init")
            else:
                if yes_asset_id and yes_asset_id in resolution.token_outcomes:
                    self._yes_resolved = resolution.token_outcomes[yes_asset_id]
                if no_asset_id and no_asset_id in resolution.token_outcomes:
                    self._no_resolved = resolution.token_outcomes[no_asset_id]
                if self._yes_resolved is None or self._no_resolved is None:
                    log.warning(
                        f"{slug}: resolution found but asset_id mismatch. "
                        f"yes_asset_id={yes_asset_id}, no_asset_id={no_asset_id}, "
                        f"resolution keys={list(resolution.token_outcomes.keys())}"
                    )

        # ------------------------------------------------------------------
        # Resample or pass through
        # ------------------------------------------------------------------
        # Preprocessed files already contain the 100ms grid with staleness_ms.
        # Raw files need resampling. Detected by presence of staleness_ms column.
        if "staleness_ms" in yes_raw.columns:
            # Preprocessed: set index to exchange_timestamp, already on grid
            yes = yes_raw.set_index("exchange_timestamp").sort_index()
            no  = no_raw.set_index("exchange_timestamp").sort_index()
        else:
            # Raw: resample to 100ms grid
            yes = _resample_book(yes_raw, self._market_open_ms, self._market_close_ms, RESAMPLE_MS)
            no  = _resample_book(no_raw,  self._market_open_ms, self._market_close_ms, RESAMPLE_MS)

        expected_bars = MARKET_DURATION_MS // RESAMPLE_MS
        if len(yes) != expected_bars or len(no) != expected_bars:
            raise ValueError(
                f"{slug}: unexpected bar count — yes={len(yes)}, no={len(no)}, "
                f"expected={expected_bars}. File may be corrupt or truncated."
            )

        self._yes_df  = yes
        self._no_df   = no
        self._timestamps = yes.index.values  # uniform grid, same for both

        log.debug(
            f"Loaded {slug}: {len(self._timestamps)} steps ({RESAMPLE_MS}ms bars), "
            f"yes_resolved={self._yes_resolved}, no_resolved={self._no_resolved}"
        )

    def _get_book_row(self, df: pd.DataFrame, idx: int) -> pd.Series:
        return df.iloc[idx]

    # ------------------------------------------------------------------
    # Internal — BTC feature lookup
    # ------------------------------------------------------------------

    def _get_btc_features(self, timestamp_ms: int) -> np.ndarray:
        """
        Get BTC features for the 1s bar containing timestamp_ms.
        Falls back to nearest available bar if exact match missing.
        """
        bar_ts = (timestamp_ms // 1000) * 1000

        try:
            row = self._btc_df.loc[bar_ts]
        except KeyError:
            # Find nearest bar
            idx = self._btc_df.index.searchsorted(bar_ts)
            idx = min(idx, len(self._btc_df) - 1)
            row = self._btc_df.iloc[idx]

        # Normalize mid_price by dividing by 100k (rough BTC scale)
        mid_norm = row["mid_price"] / 100_000.0

        return np.array([
            mid_norm,
            _safe(row, "log_return"),
            _safe(row, "ret_5s"),
            _safe(row, "ret_15s"),
            _safe(row, "ret_60s"),
            _safe(row, "rvol_30s"),
            _safe(row, "rvol_60s"),
            _safe(row, "rvol_300s"),
        ], dtype=np.float32)

    # ------------------------------------------------------------------
    # Internal — observation
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        idx = min(self._step_idx, len(self._timestamps) - 1)
        ts  = self._timestamps[idx]

        yes_row = self._yes_df.iloc[idx]
        no_row  = self._no_df.iloc[idx]

        yes_features = _book_features(yes_row, N_BOOK_LEVELS)
        no_features  = _book_features(no_row,  N_BOOK_LEVELS)

        # Position features
        capital_at_risk = self._yes_position + self._no_position
        yes_mid, no_mid = _robust_mid_pair(yes_row, no_row)
        upnl    = self._unrealized_pnl(yes_mid, no_mid)

        position_features = np.array([
            self._yes_position / self.initial_bankroll,
            self._no_position  / self.initial_bankroll,
            self._yes_avg_cost,
            self._no_avg_cost,
            upnl / self.initial_bankroll,
            capital_at_risk / self.initial_bankroll,
        ], dtype=np.float32)

        # Time remaining
        elapsed = ts - self._market_open_ms
        time_remaining = max(0.0, 1.0 - elapsed / MARKET_DURATION_MS)

        btc_features = self._get_btc_features(int(ts))

        # Staleness: ms since last real tick, normalised and clipped to [0, 1]
        yes_stale = float(yes_row.get("staleness_ms", MAX_STALENESS_MS))
        no_stale  = float(no_row.get("staleness_ms",  MAX_STALENESS_MS))
        staleness_features = np.array([
            min(yes_stale / MAX_STALENESS_MS, 1.0),
            min(no_stale  / MAX_STALENESS_MS, 1.0),
        ], dtype=np.float32)

        obs = np.concatenate([
            yes_features,
            no_features,
            position_features,
            np.array([time_remaining], dtype=np.float32),
            btc_features,
            staleness_features,
        ])

        return obs.astype(np.float32)

    # ------------------------------------------------------------------
    # Internal — action execution
    # ------------------------------------------------------------------

    def _execute_action(self, action: int):
        idx     = min(self._step_idx, len(self._timestamps) - 1)
        yes_row = self._yes_df.iloc[idx]
        no_row  = self._no_df.iloc[idx]
        yes_mid, no_mid = _robust_mid_pair(yes_row, no_row)

        small = max(MIN_ORDER, self.initial_bankroll * SMALL_ORDER_PCT)
        large = self.initial_bankroll * LARGE_ORDER_PCT
        max_pos = self.initial_bankroll * MAX_POSITION_PCT

        if action == ACTION_SELL:
            self._flatten(yes_mid, no_mid)

        elif action in (ACTION_BUY_YES_SMALL, ACTION_BUY_YES_LARGE):
            size = small if action == ACTION_BUY_YES_SMALL else large
            # Flatten No first if held
            if self._no_position > 0:
                self._flatten_no(no_mid)
            # Cap at max position
            capital_at_risk = self._yes_position + self._no_position
            size = min(size, max_pos - capital_at_risk)
            if size >= MIN_ORDER:
                self._buy_yes(size, yes_mid)

        elif action in (ACTION_BUY_NO_SMALL, ACTION_BUY_NO_LARGE):
            size = small if action == ACTION_BUY_NO_SMALL else large
            # No auto-flatten: agent must explicitly sell before switching sides.
            if self._yes_position == 0:
                capital_at_risk = self._no_position
                size = min(size, max_pos - capital_at_risk)
                if size >= MIN_ORDER:
                    self._buy_no(size, no_mid)

    def _buy_yes(self, size: float, mid: float):
        """Buy `size` dollars of Yes token at mid-price, less taker fee."""
        fee    = size * TAKER_FEE_PCT
        tokens = size / mid
        total_cost = self._yes_position + size
        self._yes_avg_cost = total_cost / (self._yes_position / self._yes_avg_cost + tokens) \
            if self._yes_position > 0 else mid
        self._yes_position += size
        self._bankroll     -= (size + fee)
        self._fees_paid    += fee

    def _buy_no(self, size: float, mid: float):
        """Buy `size` dollars of No token at mid-price, less taker fee."""
        fee    = size * TAKER_FEE_PCT
        tokens = size / mid
        total_cost = self._no_position + size
        self._no_avg_cost = total_cost / (self._no_position / self._no_avg_cost + tokens) \
            if self._no_position > 0 else mid
        self._no_position += size
        self._bankroll    -= (size + fee)
        self._fees_paid   += fee

    def _flatten_yes(self, mid: float):
        """Close Yes position at mid-price, less taker fee on proceeds."""
        if self._yes_position <= 0:
            return
        tokens         = self._yes_position / self._yes_avg_cost
        gross_proceeds = tokens * mid
        net_proceeds   = gross_proceeds * (1.0 - TAKER_FEE_PCT)
        self._fees_paid    += gross_proceeds - net_proceeds
        self._realized_pnl += net_proceeds - self._yes_position
        self._bankroll     += net_proceeds
        self._yes_position  = 0.0
        self._yes_avg_cost  = 0.0

    def _flatten_no(self, mid: float):
        """Close No position at mid-price, less taker fee on proceeds."""
        if self._no_position <= 0:
            return
        tokens         = self._no_position / self._no_avg_cost
        gross_proceeds = tokens * mid
        net_proceeds   = gross_proceeds * (1.0 - TAKER_FEE_PCT)
        self._fees_paid    += gross_proceeds - net_proceeds
        self._realized_pnl += net_proceeds - self._no_position
        self._bankroll     += net_proceeds
        self._no_position   = 0.0
        self._no_avg_cost   = 0.0

    def _flatten(self, yes_mid: float, no_mid: float):
        self._flatten_yes(yes_mid)
        self._flatten_no(no_mid)

    def _unrealized_pnl(self, yes_mid: float, no_mid: float) -> float:
        upnl = 0.0
        if self._yes_position > 0 and self._yes_avg_cost > 0:
            tokens = self._yes_position / self._yes_avg_cost
            upnl  += tokens * yes_mid - self._yes_position
        if self._no_position > 0 and self._no_avg_cost > 0:
            tokens = self._no_position / self._no_avg_cost
            upnl  += tokens * no_mid - self._no_position
        return upnl

    def _close_episode(self) -> float:
        """
        Settle all open positions at market resolution and return episode PnL.

        Binary contracts resolve to exactly 1.0 (winner) or 0.0 (loser).
        Resolution values are guaranteed to be present — the env filters out
        any market without a known resolution at init time.
        """
        # Settle Yes position
        if self._yes_position > 0:
            assert self._yes_resolved is not None, (
                "_yes_resolved is None at episode close — this market should "
                "have been filtered out at init. Check market_files filtering."
            )
            tokens   = self._yes_position / self._yes_avg_cost
            proceeds = tokens * self._yes_resolved
            self._realized_pnl += proceeds - self._yes_position
            self._bankroll     += proceeds
            self._yes_position  = 0.0
            self._yes_avg_cost  = 0.0

        # Settle No position
        if self._no_position > 0:
            assert self._no_resolved is not None, (
                "_no_resolved is None at episode close — this market should "
                "have been filtered out at init. Check market_files filtering."
            )
            tokens   = self._no_position / self._no_avg_cost
            proceeds = tokens * self._no_resolved
            self._realized_pnl += proceeds - self._no_position
            self._bankroll     += proceeds
            self._no_position   = 0.0
            self._no_avg_cost   = 0.0

        return self._realized_pnl

    # ------------------------------------------------------------------
    # Internal — info dict
    # ------------------------------------------------------------------

    def _get_info(self) -> dict:
        idx     = min(self._step_idx, len(self._timestamps) - 1)
        yes_mid, no_mid = _robust_mid_pair(self._yes_df.iloc[idx], self._no_df.iloc[idx])
        return {
            "step":          self._step_idx,
            "yes_position":  self._yes_position,
            "no_position":   self._no_position,
            "yes_avg_cost":  self._yes_avg_cost,
            "no_avg_cost":   self._no_avg_cost,
            "unrealized_pnl": self._unrealized_pnl(yes_mid, no_mid),
            "realized_pnl":  self._realized_pnl,
                "fees_paid":     self._fees_paid,
            "bankroll":      self._bankroll,
            "action_masks":  self.action_masks(),
        }


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------

def _book_features(row: pd.Series, n_levels: int) -> np.ndarray:
    """
    Extract normalized book features from a parquet row.
    Returns: [mid, spread, imbalance, bid_p1..n, bid_s1..n, ask_p1..n, ask_s1..n]
    Prices are raw (already 0-1 for binary markets).
    Sizes are normalized by dividing by 1000 (rough scale).
    """
    SIZE_SCALE = 1000.0

    mid      = _robust_mid(row)
    spread   = float(row.get("spread",         0.0))
    imbal    = float(row.get("book_imbalance", 0.0))

    bid_prices = [float(row.get(f"bid_price_{i+1}", 0.0)) for i in range(n_levels)]
    bid_sizes  = [float(row.get(f"bid_size_{i+1}",  0.0)) / SIZE_SCALE for i in range(n_levels)]
    ask_prices = [float(row.get(f"ask_price_{i+1}", 0.0)) for i in range(n_levels)]
    ask_sizes  = [float(row.get(f"ask_size_{i+1}",  0.0)) / SIZE_SCALE for i in range(n_levels)]

    return np.array(
        [mid, spread, imbal] + bid_prices + bid_sizes + ask_prices + ask_sizes,
        dtype=np.float32,
    )


def _safe(row: pd.Series, col: str, default: float = 0.0) -> float:
    """Return float value from series, replacing NaN with default."""
    v = row.get(col, default)
    return default if pd.isna(v) else float(v)