"""
Polymarket BTC 5-minute market analysis pipeline.

Three components:
  1. MarketLoader       — loads trade + book parquet files with per-slug eviction
  2. ResolutionStore    — fetches and caches resolution outcomes from Gamma API
  3. CalibrationAnalyser — builds calibration curves and screens for mispricing

Usage:
    from market_analysis import MarketLoader, ResolutionStore, CalibrationAnalyser

    loader = MarketLoader(data_dir=Path("data"))
    store  = ResolutionStore(cache_path=Path("data/resolutions.json"))

    slugs       = loader.available_slugs()
    resolutions = store.fetch_batch(slugs)

    analyser    = CalibrationAnalyser(loader, store)
    obs         = analyser.build_observations(slugs)
    curve       = analyser.calibration_curve(obs)
    mispricings = analyser.screen_mispricings(obs)

Data source rationale:
    build_observations uses book snapshots (~6.7k rows/market, 3.8GB total)
    rather than trade events (~95k rows/market, 27GB total). The 17x size
    reduction makes batched processing viable on 32GB RAM. Book snapshots
    also give a cleaner implied probability signal — each row is the full
    order book state rather than an individual fill that may be part of a
    multi-level sweep.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GAMMA_API = "https://gamma-api.polymarket.com"
REQUEST_DELAY = 0.2   # seconds between API calls

# Price buckets for calibration curve — (low_inclusive, high_exclusive)
# Finer resolution at the tails where mispricing is hypothesised
PRICE_BUCKETS = [
    (0.01, 0.03),
    (0.03, 0.05),
    (0.05, 0.08),
    (0.08, 0.12),
    (0.12, 0.20),
    (0.20, 0.35),
    (0.35, 0.50),
    (0.50, 0.65),
    (0.65, 0.80),
    (0.80, 0.88),
    (0.88, 0.92),
    (0.92, 0.95),
    (0.95, 0.97),
    (0.97, 0.99),
    (0.99, 1.01),
]

# Time windows (seconds before close) for stratified analysis
TIME_BUCKETS = [
    ("full_market", 0,   300),
    ("first_half",  150, 300),
    ("second_half", 0,   150),
    ("final_90s",   0,   90),
    ("final_60s",   0,   60),
    ("final_30s",   0,   30),
]

# PyArrow schema for the observations parquet.
# Pinned explicitly so ParquetWriter never hits type mismatches between batches
# (e.g. an empty batch where pandas infers object dtype for a float column).
OBS_SCHEMA = pa.schema([
    ("exchange_timestamp",   pa.int64()),
    ("asset_id",             pa.string()),
    ("mid_price",            pa.float64()),
    ("best_bid",             pa.float64()),
    ("best_ask",             pa.float64()),
    ("spread",               pa.float64()),
    ("book_imbalance",       pa.float64()),
    ("slug",                 pa.string()),
    ("seconds_before_close", pa.float64()),
    ("resolved_outcome",     pa.float64()),
    ("won",                  pa.bool_()),
])


# ---------------------------------------------------------------------------
# 1. MarketLoader
# ---------------------------------------------------------------------------

class MarketLoader:
    """
    Loads trade event and book snapshot parquet files with optional in-memory
    caching. Call evict(slug) after processing each market during the
    build_observations pass to prevent unbounded RAM growth.
    """

    def __init__(self, data_dir: Path = Path("data")):
        self.trade_dir = data_dir / "trade_events"
        self.book_dir  = data_dir / "book_snapshots"
        self._trade_cache: dict[str, pd.DataFrame] = {}
        self._book_cache:  dict[str, pd.DataFrame] = {}

    def available_slugs(self) -> list[str]:
        """Return slugs for all book snapshot files present on disk."""
        return [p.stem for p in sorted(self.book_dir.glob("*.parquet"))]

    def trades(self, slug: str) -> pd.DataFrame | None:
        """
        Load trade events for a slug. Returns None if file missing.
        WARNING: trade files are ~25MB each (~95k rows). Do not load
        many simultaneously — use book snapshots for bulk analysis.
        """
        if slug in self._trade_cache:
            return self._trade_cache[slug]

        path = self.trade_dir / f"{slug}.parquet"
        if not path.exists():
            logger.warning(f"Trade file not found: {path}")
            return None

        try:
            df = pq.read_table(path).to_pandas()
        except Exception as e:
            logger.error(f"Failed to read {path}: {e}")
            return None

        df["exchange_dt"] = pd.to_datetime(
            df["exchange_timestamp"], unit="ms", utc=True
        )
        df = df.sort_values("exchange_timestamp").reset_index(drop=True)
        self._trade_cache[slug] = df
        return df

    def books(self, slug: str) -> pd.DataFrame | None:
        """
        Load book snapshots for a slug. Returns None if file missing.
        Uses top-10 flattened bid/ask columns. full_bids/full_asks are
        retained as JSON strings for deep-book analysis if needed.
        """
        if slug in self._book_cache:
            return self._book_cache[slug]

        path = self.book_dir / f"{slug}.parquet"
        if not path.exists():
            return None

        try:
            df = pq.read_table(path).to_pandas()
        except Exception as e:
            logger.error(f"Failed to read {path}: {e}")
            return None

        df["exchange_dt"] = pd.to_datetime(
            df["exchange_timestamp"], unit="ms", utc=True
        )
        df = df.sort_values("exchange_timestamp").reset_index(drop=True)
        self._book_cache[slug] = df
        return df

    def evict(self, slug: str):
        """
        Drop a single slug from both in-memory caches.
        Call this immediately after processing each slug during bulk passes
        to prevent the cache growing to the size of the full dataset.
        """
        self._trade_cache.pop(slug, None)
        self._book_cache.pop(slug, None)

    def clear_cache(self):
        """Drop all cached DataFrames."""
        self._trade_cache.clear()
        self._book_cache.clear()


# ---------------------------------------------------------------------------
# 2. ResolutionStore
# ---------------------------------------------------------------------------

@dataclass
class Resolution:
    slug:           str
    condition_id:   str
    closed_time:    str    # ISO string from Gamma API
    winning_token:  str
    token_outcomes: dict   # {token_id: 1.0 or 0.0}


class ResolutionStore:
    """
    Fetches market resolutions from Gamma API and caches to a JSON file.
    Only fully resolved markets are cached — unresolved fetches are not
    stored, so they will be retried on the next run.
    """

    def __init__(self, cache_path: Path = Path("data/resolutions.json")):
        self.cache_path = cache_path
        self._cache: dict[str, dict] = self._load()

    def _load(self) -> dict:
        if self.cache_path.exists():
            with open(self.cache_path) as f:
                data = json.load(f)
            logger.info(
                f"Loaded {len(data)} cached resolutions from {self.cache_path}"
            )
            return data
        return {}

    def _save(self):
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "w") as f:
            json.dump(self._cache, f, indent=2)

    def get(self, slug: str) -> Resolution | None:
        """Return cached resolution without hitting the API. None if absent."""
        raw = self._cache.get(slug)
        return Resolution(**raw) if raw else None

    def fetch(self, slug: str) -> Resolution | None:
        """
        Fetch resolution, using cache if available.
        Returns None for unresolved markets or API failures.
        """
        if slug in self._cache:
            return Resolution(**self._cache[slug])

        try:
            resp = requests.get(
                f"{GAMMA_API}/markets/slug/{slug}", timeout=10
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.error(f"API request failed for {slug}: {e}")
            return None

        if data.get("umaResolutionStatus") != "resolved":
            logger.debug(f"{slug}: not yet resolved")
            return None

        try:
            token_ids      = json.loads(data["clobTokenIds"])
            outcome_prices = json.loads(data["outcomePrices"])

            # Robust: handles "1", "1.0", and 1.0 from the API
            winning_idx = next(
                i for i, p in enumerate(outcome_prices)
                if float(p) == 1.0
            )
        except Exception as e:
            logger.error(f"Failed to parse resolution for {slug}: {e}")
            return None

        resolution = Resolution(
            slug=slug,
            condition_id=data["conditionId"],
            closed_time=data.get("closedTime", ""),
            winning_token=token_ids[winning_idx],
            token_outcomes={
                token_ids[0]: float(outcome_prices[0]),
                token_ids[1]: float(outcome_prices[1]),
            },
        )

        from dataclasses import asdict
        self._cache[slug] = asdict(resolution)
        self._save()
        time.sleep(REQUEST_DELAY)
        return resolution

    def fetch_batch(self, slugs: list[str]) -> dict[str, Resolution]:
        """
        Fetch resolutions for multiple slugs.
        Returns {slug: Resolution} for all successfully resolved markets.
        """
        uncached     = [s for s in slugs if s not in self._cache]
        cached_count = len(slugs) - len(uncached)

        if uncached:
            logger.info(
                f"Fetching {len(uncached)} resolutions "
                f"({cached_count} already cached)"
            )

        results = {}
        for i, slug in enumerate(slugs):
            resolution = self.fetch(slug)
            if resolution:
                results[slug] = resolution
            if uncached and slug in uncached and (i + 1) % 50 == 0:
                logger.info(f"  Progress: {i + 1}/{len(slugs)}")

        logger.info(f"Resolved {len(results)}/{len(slugs)} markets")
        return results


# ---------------------------------------------------------------------------
# 3. CalibrationAnalyser
# ---------------------------------------------------------------------------

class CalibrationAnalyser:
    """
    Builds calibration curves and screens for mispriced contracts.

    The calibration curve answers: for snapshots where a contract's implied
    probability was P, what fraction actually resolved as winners?

    If contracts priced at 0.04 win 8% of the time, they are systematically
    underpriced and represent a positive EV buy.
    """

    def __init__(self, loader: MarketLoader, store: ResolutionStore):
        self.loader = loader
        self.store  = store

    def build_observations(
        self,
        slugs: list[str],
        min_snapshots: int = 50,
        batch_size: int = 100,
        obs_cache_path: Path = Path("data/observations.parquet"),
        rebuild: bool = False,
    ) -> pd.DataFrame:
        """
        Build calibration observations from book snapshots.

        Each output row is one book snapshot for one token, tagged with the
        token's resolution outcome. This gives implied probability at a known
        point in time with ground truth.

        Streams to parquet in batches. Peak RAM is bounded to roughly:
            batch_size * median_book_size * 2 ≈ 100 * 13MB * 2 ≈ 2.6GB
        Reduce batch_size if OOM persists.

        On subsequent calls, returns cached parquet unless rebuild=True.

        Args:
            slugs:          Market slugs to process.
            min_snapshots:  Skip markets with fewer snapshots than this.
            batch_size:     Markets per batch before flushing to disk.
            obs_cache_path: Parquet output path.
            rebuild:        Reprocess even if cache exists.

        Returns:
            DataFrame with columns per OBS_SCHEMA:
                exchange_timestamp, asset_id, mid_price, best_bid, best_ask,
                spread, book_imbalance, slug, seconds_before_close,
                resolved_outcome, won
        """
        if obs_cache_path.exists() and not rebuild:
            logger.info(f"Loading cached observations from {obs_cache_path}")
            df = pq.read_table(obs_cache_path).to_pandas()
            logger.info(
                f"Loaded {len(df):,} observations from "
                f"{df['slug'].nunique()} markets"
            )
            return df

        obs_cache_path.parent.mkdir(parents=True, exist_ok=True)

        writer       = None   # opened on first non-empty batch
        total_rows   = 0
        skipped      = 0
        slug_batches = [
            slugs[i : i + batch_size]
            for i in range(0, len(slugs), batch_size)
        ]

        logger.info(
            f"Building observations: {len(slugs)} markets in "
            f"{len(slug_batches)} batches of {batch_size}"
        )

        try:
            for batch_num, batch in enumerate(slug_batches, 1):
                frames = []

                for slug in batch:
                    resolution = self.store.get(slug)
                    if resolution is None:
                        skipped += 1
                        continue

                    books = self.loader.books(slug)
                    if books is None or len(books) < min_snapshots:
                        skipped += 1
                        continue

                    # Filter to the two primary resolution tokens.
                    # Each market has ~68-row ghost tokens (secondary contracts)
                    # that carry no resolution outcome — drop them early.
                    valid_tokens = set(resolution.token_outcomes.keys())
                    books = books[books["asset_id"].isin(valid_tokens)]

                    if books.empty:
                        skipped += 1
                        continue

                    close_ts = books["exchange_timestamp"].max()

                    df = books[[
                        "exchange_timestamp", "asset_id",
                        "mid_price", "bid_price_1", "ask_price_1",
                        "spread", "book_imbalance",
                    ]].copy()

                    # bid_price_1 / ask_price_1 are the best bid and ask.
                    # Rename to best_bid / best_ask to match OBS_SCHEMA and
                    # downstream analysis which uses these names consistently.
                    df = df.rename(columns={
                        "bid_price_1": "best_bid",
                        "ask_price_1": "best_ask",
                    })

                    df["slug"]                 = slug
                    df["seconds_before_close"] = (
                        (close_ts - df["exchange_timestamp"]) / 1000.0
                    )
                    df["resolved_outcome"] = df["asset_id"].map(
                        resolution.token_outcomes
                    )
                    df = df.dropna(subset=["resolved_outcome"])
                    df["won"] = df["resolved_outcome"] == 1.0

                    frames.append(df)

                    # Evict immediately — keeping loaded books in cache
                    # defeats the purpose of batching
                    self.loader.evict(slug)

                if not frames:
                    logger.info(
                        f"  Batch {batch_num}/{len(slug_batches)} — "
                        f"no valid markets"
                    )
                    continue

                batch_df = pd.concat(frames, ignore_index=True)

                # Cast to pinned schema — prevents type mismatch errors
                # when an edge-case batch produces different inferred dtypes
                table = pa.Table.from_pandas(
                    batch_df, schema=OBS_SCHEMA, preserve_index=False
                )

                if writer is None:
                    writer = pq.ParquetWriter(obs_cache_path, OBS_SCHEMA)

                writer.write_table(table)
                total_rows += len(batch_df)

                logger.info(
                    f"  Batch {batch_num}/{len(slug_batches)} — "
                    f"{len(batch_df):,} rows written "
                    f"(running total: {total_rows:,})"
                )

                del frames, batch_df, table

        finally:
            # Always close — an unclosed ParquetWriter leaves a corrupt footer
            if writer is not None:
                writer.close()

        if skipped:
            logger.info(
                f"Skipped {skipped}/{len(slugs)} markets "
                f"(no resolution or insufficient snapshots)"
            )

        if total_rows == 0:
            logger.warning("No observations written — check resolutions are loaded")
            return pd.DataFrame()

        logger.info(
            f"Done. {total_rows:,} observations written to {obs_cache_path}"
        )
        return pq.read_table(obs_cache_path).to_pandas()

    # -------------------------------------------------------------------------
    # ITERATION HISTORY — kept for reference
    #
    # v1 — iterrows: every cell becomes a Python object, 134M dicts in RAM
    #   for _, row in trades.iterrows():
    #       records.append({...})
    #   return pd.DataFrame(records)   # OOM + slow
    #
    # v2 — vectorized, no batching: correct logic, still OOM
    #   for slug in slugs:
    #       frames.append(tagged_df)   # all 1405 DataFrames in RAM at once
    #   return pd.concat(frames)       # one 27GB allocation
    #
    # v3 — batched, trade events: ~95k rows/market, batch_size=50 → 1.25GB/batch
    #   trades = self.loader.trades(slug)   # 25MB per market, OOM before flush
    #
    # v4 (current) — batched, book snapshots: ~6.7k rows/market, ~13MB each
    #   books = self.loader.books(slug)     # batch_size=100 → ~2.6GB peak
    #   writer.write_table(table)           # flushes before next batch loads
    # -------------------------------------------------------------------------

    def calibration_curve(
        self,
        obs: pd.DataFrame,
        time_window: tuple[float, float] = (0, 300),
        price_col: str = "mid_price",
        min_observations: int = 30,
    ) -> pd.DataFrame:
        """
        Calibration curve: implied probability vs actual win rate.

        Args:
            obs:              DataFrame from build_observations().
            time_window:      (min_secs, max_secs) before close.
                              Default (0, 300) = full 5-minute market.
            price_col:        'mid_price' for theoretical analysis,
                              'best_ask' for realistic buy-side cost.
            min_observations: Buckets below this sample size are omitted.

        Returns:
            DataFrame: bucket_label, price_low, price_high, implied_prob,
                       actual_win_rate, edge, n_observations, n_markets,
                       avg_spread
        """
        min_s, max_s = time_window
        filtered = obs[
            (obs["seconds_before_close"] >= min_s) &
            (obs["seconds_before_close"] <= max_s)
        ]

        rows = []
        for low, high in PRICE_BUCKETS:
            bucket = filtered[
                (filtered[price_col] >= low) &
                (filtered[price_col] <  high)
            ]
            if len(bucket) < min_observations:
                continue

            implied  = (low + high) / 2
            win_rate = bucket["won"].mean()

            rows.append({
                "bucket_label":    f"{low:.2f}-{high:.2f}",
                "price_low":       low,
                "price_high":      high,
                "implied_prob":    implied,
                "actual_win_rate": win_rate,
                "edge":            win_rate - implied,
                "n_observations":  len(bucket),
                "n_markets":       bucket["slug"].nunique(),
                "avg_spread":      bucket["spread"].mean(),
            })

        return pd.DataFrame(rows)

    def calibration_by_time(
        self,
        obs: pd.DataFrame,
        price_col: str = "mid_price",
    ) -> pd.DataFrame:
        """
        Calibration curves across all TIME_BUCKETS in one DataFrame.
        Use this to see whether mispricing concentrates at specific
        points in the market window.
        """
        frames = []
        for label, min_s, max_s in TIME_BUCKETS:
            curve = self.calibration_curve(
                obs, time_window=(min_s, max_s), price_col=price_col
            )
            if not curve.empty:
                curve["time_window"] = label
                frames.append(curve)

        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def screen_mispricings(
        self,
        obs: pd.DataFrame,
        min_edge: float = 0.02,
        max_price: float = 0.20,
        time_window: tuple[float, float] = (0, 90),
        min_observations: int = 30,
        price_col: str = "mid_price",
    ) -> pd.DataFrame:
        """
        Screen for systematically underpriced contracts.

        Defaults target the tail mispricing hypothesis:
          contracts below 0.20 in the final 90 seconds.

        Returns calibration rows where edge >= min_edge, sorted descending.
        """
        curve = self.calibration_curve(
            obs,
            time_window=time_window,
            price_col=price_col,
            min_observations=min_observations,
        )
        if curve.empty:
            return curve

        return curve[
            (curve["price_high"] <= max_price + 0.01) &
            (curve["edge"] >= min_edge)
        ].sort_values("edge", ascending=False)

    def ev_at_ask(
        self,
        obs: pd.DataFrame,
        time_window: tuple[float, float] = (0, 90),
        max_price: float = 0.20,
        min_observations: int = 30,
    ) -> pd.DataFrame:
        """
        Expected value of buying at the ask price.

        mid_price calibration will show edge that may not survive the spread.
        This is the honest estimate — positive EV means profitable after spread.

        EV         = actual_win_rate - avg_ask_paid
        spread_cost = avg_ask_paid - implied_prob  (how much spread erodes edge)
        """
        min_s, max_s = time_window
        filtered = obs[
            (obs["seconds_before_close"] >= min_s) &
            (obs["seconds_before_close"] <= max_s) &
            (obs["mid_price"] < max_price)
        ]

        rows = []
        for low, high in PRICE_BUCKETS:
            if high > max_price + 0.01:
                continue

            bucket = filtered[
                (filtered["mid_price"] >= low) &
                (filtered["mid_price"] <  high)
            ]
            if len(bucket) < min_observations:
                continue

            win_rate = bucket["won"].mean()
            avg_ask  = bucket["best_ask"].mean()
            implied  = (low + high) / 2

            rows.append({
                "bucket_label":    f"{low:.2f}-{high:.2f}",
                "implied_prob":    implied,
                "actual_win_rate": win_rate,
                "avg_ask_paid":    avg_ask,
                "ev_at_ask":       win_rate - avg_ask,
                "edge_vs_mid":     win_rate - implied,
                "spread_cost":     avg_ask - implied,
                "n_observations":  len(bucket),
                "n_markets":       bucket["slug"].nunique(),
            })

        df = pd.DataFrame(rows)
        return df.sort_values("ev_at_ask", ascending=False) if not df.empty else df


# ---------------------------------------------------------------------------
# Convenience runner
# ---------------------------------------------------------------------------

def run_analysis(
    data_dir: Path = Path("data"),
    cache_path: Path = Path("data/resolutions.json"),
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Full pipeline: load slugs → fetch resolutions → build observations →
    calibration curve → mispricing screen.

    Returns (obs, curve, mispricings).
    """
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )

    loader = MarketLoader(data_dir=data_dir)
    store  = ResolutionStore(cache_path=cache_path)

    slugs       = loader.available_slugs()
    logger.info(f"Found {len(slugs)} market files")

    resolutions = store.fetch_batch(slugs)
    logger.info(f"Resolutions available: {len(resolutions)}/{len(slugs)}")

    analyser = CalibrationAnalyser(loader, store)
    obs      = analyser.build_observations(list(resolutions.keys()))

    if obs.empty:
        logger.error("No observations — cannot proceed")
        return obs, pd.DataFrame(), pd.DataFrame()

    obs_clean = obs[
        (obs['spread'] > 0) &
        (obs['seconds_before_close'] > 10)   # only cut the worst 10s
    ].copy()

    print(f"Rows removed: {len(obs) - len(obs_clean):,} ({(1 - len(obs_clean)/len(obs)):.1%})")

    # Now run calibration across time windows to see where patterns actually live
    by_time = analyser.calibration_by_time(obs_clean)
    print(by_time.to_string(index=False))

    curve       = analyser.calibration_curve(obs)
    mispricings = analyser.screen_mispricings(obs)

    if verbose:
        print(f"\n{'='*60}")
        print(
            f"OBSERVATIONS: {len(obs):,} snapshots across "
            f"{obs['slug'].nunique()} markets"
        )
        print(f"{'='*60}")

        print(f"\n--- Calibration Curve (full market, mid_price) ---")
        print(curve.to_string(index=False))

        print(f"\n--- Tail Mispricing Screen (final 90s, price < 0.20) ---")
        if mispricings.empty:
            print("No significant mispricings found at current thresholds")
        else:
            print(mispricings.to_string(index=False))

        print(f"\n--- EV at Ask (final 90s, price < 0.20) ---")
        ev = analyser.ev_at_ask(obs)
        if ev.empty:
            print("Insufficient data at current thresholds")
        else:
            print(ev.to_string(index=False))

        # 1. EV at ask for high-probability contracts in final 30s
        ev_high = analyser.ev_at_ask(
            obs_clean,
            time_window=(0, 30),
            max_price=1.0,     # include the full range
            min_observations=100,
        )
        print(ev_high.to_string(index=False))

        # 2. Does edge scale with seconds remaining?
        # If edge grows as close approaches, that's a clear entry signal
        for window_label, min_s, max_s in [
            ("90-60s", 60, 90),
            ("60-30s", 30, 60),
            ("30-10s", 10, 30),
        ]:
            curve = analyser.calibration_curve(
                obs_clean,
                time_window=(min_s, max_s),
                price_col="best_ask",
                min_observations=200,
            )
            high_prob = curve[curve["price_low"] >= 0.80]
            if not high_prob.empty:
                print(f"\n{window_label}:")
                print(high_prob[["bucket_label","implied_prob","actual_win_rate","edge","n_observations"]].to_string(index=False))
    return obs, curve, mispricings, by_time
