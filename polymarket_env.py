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

Fill model: mid-price fill, immediate (no slippage simulation yet).

Reward:
    - Step: 0 (no shaping — terminal reward only, keeps it clean for now)
    - Terminal: realized PnL of the episode
                position forcibly closed at final mid-price

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

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STARTING_BANKROLL  = 10_000.0
MIN_ORDER          = 5.0
SMALL_ORDER_PCT    = 0.005   # 0.5% of bankroll
LARGE_ORDER_PCT    = 0.010   # 1.0% of bankroll
MAX_POSITION_PCT   = 0.05    # 5.0% of bankroll

N_BOOK_LEVELS      = 5
MARKET_DURATION_MS = 300_000  # 5 minutes in milliseconds
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
OBS_DIM        = OBS_CONTRACT + OBS_POSITION + OBS_TIME + OBS_BTC  # 61


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
    """
    # Build the target grid
    grid = np.arange(open_ms, close_ms, freq_ms)

    if df.empty:
        return pd.DataFrame(index=grid)

    df = df.set_index("exchange_timestamp").sort_index()

    # Assign each tick to its 100ms bar (floor division)
    df["bar_ts"] = (df.index // freq_ms) * freq_ms

    # Last-value within each bar
    resampled = df.groupby("bar_ts").last()

    # Reindex to full grid, forward-fill then backward-fill
    resampled = resampled.reindex(grid).ffill().bfill()

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
        bankroll: float = STARTING_BANKROLL,
        deterministic: bool = False,
        market_slugs: Optional[list[str]] = None,
        seed: Optional[int] = None,
    ):
        super().__init__()

        self.book_dir      = Path(book_dir)
        self.btc_path      = Path(btc_path)
        self.initial_bankroll = bankroll
        self.deterministic = deterministic

        # Discover available markets
        all_files = sorted(self.book_dir.glob("btc-updown-5m-*.parquet"))
        if market_slugs:
            slug_set = set(market_slugs)
            self.market_files = [f for f in all_files if f.stem in slug_set]
        else:
            self.market_files = all_files

        if not self.market_files:
            raise ValueError(f"No market files found in {book_dir}")

        # Filter out thin markets (first few minutes of 2/12, book barely active)
        MIN_ROWS = 500
        self.market_files = [
            f for f in self.market_files
            if len(pd.read_parquet(f, columns=["exchange_timestamp"])) >= MIN_ROWS
        ]
        if not self.market_files:
            raise ValueError(f"No markets with >= {MIN_ROWS} rows found in {book_dir}")

        log.info(f"PolymarketEnv: {len(self.market_files)} markets available ({MIN_ROWS}+ rows)")

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

        self._bankroll:       float = bankroll
        self._yes_position:   float = 0.0   # dollar value of Yes tokens held
        self._no_position:    float = 0.0   # dollar value of No tokens held
        self._yes_avg_cost:   float = 0.0   # average cost per dollar invested in Yes
        self._no_avg_cost:    float = 0.0   # average cost per dollar invested in No
        self._realized_pnl:   float = 0.0

        self._rng = np.random.default_rng(seed)

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
            # Currently does not prevent resampling the same market
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
        self._step_idx      = 0

        obs  = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        assert self._yes_df is not None, "Call reset() before step()"

        reward     = 0.0
        terminated = False
        truncated  = False

        # Execute action
        if action != ACTION_HOLD:
            self._execute_action(action)

        self._step_idx += 1

        # Check termination
        if self._step_idx >= len(self._timestamps):
            terminated = True
            reward = self._close_episode()

        obs  = self._get_obs() if not terminated else np.zeros(OBS_DIM, dtype=np.float32)
        info = self._get_info()
        return obs, reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        """
        Returns boolean mask of valid actions for current state.
        Used by sb3-contrib MaskablePPO.
        """
        masks = np.ones(N_ACTIONS, dtype=bool)

        capital_at_risk = self._yes_position + self._no_position
        max_position    = self.initial_bankroll * MAX_POSITION_PCT
        at_max          = capital_at_risk >= max_position
        flat            = (self._yes_position == 0.0 and self._no_position == 0.0)

        # Can't buy if at max position
        if at_max:
            masks[ACTION_BUY_YES_SMALL] = False
            masks[ACTION_BUY_YES_LARGE] = False
            masks[ACTION_BUY_NO_SMALL]  = False
            masks[ACTION_BUY_NO_LARGE]  = False

        # Can't sell if flat
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

        # Separate Yes and No tokens
        yes_raw = df[df["token_label"] == "Up"].copy()
        no_raw  = df[df["token_label"] == "Down"].copy()

        # Resample each token to fixed RESAMPLE_MS grid using last-value
        yes = _resample_book(yes_raw, self._market_open_ms, self._market_close_ms, RESAMPLE_MS)
        no  = _resample_book(no_raw,  self._market_open_ms, self._market_close_ms, RESAMPLE_MS)

        self._yes_df  = yes
        self._no_df   = no
        self._timestamps = yes.index.values  # uniform grid, same for both

        log.debug(f"Loaded {slug}: {len(self._timestamps)} steps ({RESAMPLE_MS}ms bars)")

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
        yes_mid = float(yes_row.get("mid_price", 0.5))
        no_mid  = float(no_row.get("mid_price",  0.5))
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

        obs = np.concatenate([
            yes_features,
            no_features,
            position_features,
            np.array([time_remaining], dtype=np.float32),
            btc_features,
        ])

        return obs.astype(np.float32)

    # ------------------------------------------------------------------
    # Internal — action execution
    # ------------------------------------------------------------------

    def _execute_action(self, action: int):
        idx     = min(self._step_idx, len(self._timestamps) - 1)
        yes_row = self._yes_df.iloc[idx]
        no_row  = self._no_df.iloc[idx]
        yes_mid = float(yes_row.get("mid_price", 0.5))
        no_mid  = float(no_row.get("mid_price",  0.5))

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
            # Flatten Yes first if held
            if self._yes_position > 0:
                self._flatten_yes(yes_mid)
            # Cap at max position
            capital_at_risk = self._yes_position + self._no_position
            size = min(size, max_pos - capital_at_risk)
            if size >= MIN_ORDER:
                self._buy_no(size, no_mid)

    def _buy_yes(self, size: float, mid: float):
        """Buy `size` dollars of Yes token at mid-price."""
        tokens = size / mid  # number of tokens
        # Update average cost
        total_cost = self._yes_position + size
        self._yes_avg_cost = total_cost / (self._yes_position / self._yes_avg_cost + tokens) \
            if self._yes_position > 0 else mid
        self._yes_position += size
        self._bankroll     -= size

    def _buy_no(self, size: float, mid: float):
        """Buy `size` dollars of No token at mid-price."""
        tokens = size / mid
        total_cost = self._no_position + size
        self._no_avg_cost = total_cost / (self._no_position / self._no_avg_cost + tokens) \
            if self._no_position > 0 else mid
        self._no_position += size
        self._bankroll    -= size

    def _flatten_yes(self, mid: float):
        """Close Yes position at mid-price."""
        if self._yes_position <= 0:
            return
        tokens   = self._yes_position / self._yes_avg_cost
        proceeds = tokens * mid
        self._realized_pnl += proceeds - self._yes_position
        self._bankroll     += proceeds
        self._yes_position  = 0.0
        self._yes_avg_cost  = 0.0

    def _flatten_no(self, mid: float):
        """Close No position at mid-price."""
        if self._no_position <= 0:
            return
        tokens   = self._no_position / self._no_avg_cost
        proceeds = tokens * mid
        self._realized_pnl += proceeds - self._no_position
        self._bankroll     += proceeds
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
        """Force-flatten at final mid-price and return episode PnL."""
        last_idx = len(self._timestamps) - 1
        yes_mid  = float(self._yes_df.iloc[last_idx].get("mid_price", 0.5))
        no_mid   = float(self._no_df.iloc[last_idx].get("mid_price",  0.5))
        self._flatten(yes_mid, no_mid)
        return self._realized_pnl

    # ------------------------------------------------------------------
    # Internal — info dict
    # ------------------------------------------------------------------

    def _get_info(self) -> dict:
        idx     = min(self._step_idx, len(self._timestamps) - 1)
        yes_mid = float(self._yes_df.iloc[idx].get("mid_price", 0.5))
        no_mid  = float(self._no_df.iloc[idx].get("mid_price",  0.5))
        return {
            "step":          self._step_idx,
            "yes_position":  self._yes_position,
            "no_position":   self._no_position,
            "yes_avg_cost":  self._yes_avg_cost,
            "no_avg_cost":   self._no_avg_cost,
            "unrealized_pnl": self._unrealized_pnl(yes_mid, no_mid),
            "realized_pnl":  self._realized_pnl,
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

    mid      = float(row.get("mid_price",      0.5))
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