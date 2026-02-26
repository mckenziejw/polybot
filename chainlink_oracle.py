"""
Chainlink BTC/USD oracle round fetcher for Polygon.

Fetches all historical rounds in the range covered by your Polymarket
market data and caches to parquet. Exposes utilities for computing
oracle lag and joining oracle state to market observations.

No API key required — uses public drpc.org Polygon RPC.
Swap RPC_URL for Polymarket's Chainlink Data Streams endpoint when
that API key arrives.

Usage:
    from chainlink_oracle import OracleStore

    store = OracleStore(
        data_dir=Path("data"),
        cache_path=Path("data/chainlink_rounds.parquet"),
    )
    store.build()

    # Oracle price at a specific timestamp
    price = store.price_at(timestamp_s=1771419302)

    # Lag between last oracle update and market close
    lag = store.lag_at_close(market_close_s=1771419600)

    # Join oracle features to observations DataFrame
    obs_with_oracle = store.join_to_observations(obs)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from web3 import Web3

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RPC_URL                   = "https://polygon.drpc.org"
CHAINLINK_BTC_USD_POLYGON = "0xc907E116054Ad103354f2D350FD2514433D57F6F"
PHASE_ID                  = 3
DECIMALS                  = 8
LATEST_AGG_ROUND          = 3_326_818   # update if rebuilding much later

# Conservative rate — public endpoint, no account
REQUESTS_PER_SECOND       = 8
REQUEST_DELAY             = 1.0 / REQUESTS_PER_SECOND

# How many rounds before first market to include as buffer
ROUND_BUFFER              = 50

ROUND_SCHEMA = pa.schema([
    ("agg_round",  pa.int64()),
    ("price",      pa.float64()),
    ("started_at", pa.int64()),    # Unix seconds
    ("updated_at", pa.int64()),    # Unix seconds — use this for lag calculation
])

ABI = [
    {
        "inputs": [],
        "name": "latestRoundData",
        "outputs": [
            {"name": "roundId",         "type": "uint80"},
            {"name": "answer",          "type": "int256"},
            {"name": "startedAt",       "type": "uint256"},
            {"name": "updatedAt",       "type": "uint256"},
            {"name": "answeredInRound", "type": "uint80"},
        ],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [{"name": "_roundId", "type": "uint80"}],
        "name": "getRoundData",
        "outputs": [
            {"name": "roundId",         "type": "uint80"},
            {"name": "answer",          "type": "int256"},
            {"name": "startedAt",       "type": "uint256"},
            {"name": "updatedAt",       "type": "uint256"},
            {"name": "answeredInRound", "type": "uint80"},
        ],
        "stateMutability": "view",
        "type": "function",
    },
]


# ---------------------------------------------------------------------------
# OracleStore
# ---------------------------------------------------------------------------

class OracleStore:
    """
    Fetches and caches Chainlink BTC/USD oracle rounds from Polygon.

    Exposes three main interfaces:
      - price_at(timestamp_s)          — oracle price as of a timestamp
      - lag_at_close(market_close_s)   — seconds since last update at close
      - join_to_observations(obs)      — attach oracle features to obs DataFrame
    """

    def __init__(
        self,
        data_dir: Path = Path("data"),
        cache_path: Path = Path("data/chainlink_rounds.parquet"),
    ):
        self.data_dir   = data_dir
        self.cache_path = cache_path
        self._rounds: pd.DataFrame | None = None
        self._w3   = Web3(Web3.HTTPProvider(RPC_URL))
        self._feed = self._w3.eth.contract(
            address=Web3.to_checksum_address(CHAINLINK_BTC_USD_POLYGON),
            abi=ABI,
        )

    # ------------------------------------------------------------------
    # Build / load
    # ------------------------------------------------------------------

    def build(self, rebuild: bool = False) -> None:
        """
        Fetch all rounds in the market data range and cache to parquet.
        No-op if cache exists and rebuild=False.
        """
        if self.cache_path.exists() and not rebuild:
            logger.info(f"Oracle cache exists at {self.cache_path} — loading")
            self._load()
            return

        start_agg, end_agg = self._round_range()
        # Buffer: include rounds before first market for lag context
        start_agg = max(1, start_agg - ROUND_BUFFER)

        total = end_agg - start_agg + 1
        logger.info(
            f"Fetching {total:,} oracle rounds "
            f"(aggRound {start_agg} → {end_agg})"
        )

        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        writer     = None
        fetched    = 0
        errors     = 0
        batch      = []
        BATCH_SIZE = 500   # write to parquet every N rounds

        try:
            for agg_round in range(start_agg, end_agg + 1):
                row = self._fetch_round(agg_round)

                if row is not None:
                    batch.append(row)
                    fetched += 1
                else:
                    errors += 1

                # Flush batch to parquet
                if len(batch) >= BATCH_SIZE:
                    table = pa.Table.from_pylist(batch, schema=ROUND_SCHEMA)
                    if writer is None:
                        writer = pq.ParquetWriter(self.cache_path, ROUND_SCHEMA)
                    writer.write_table(table)
                    batch = []

                if fetched % 1000 == 0 and fetched > 0:
                    logger.info(
                        f"  {fetched:,}/{total:,} rounds fetched "
                        f"({errors} errors)"
                    )

                time.sleep(REQUEST_DELAY)

        finally:
            # Flush remaining rows
            if batch:
                table = pa.Table.from_pylist(batch, schema=ROUND_SCHEMA)
                if writer is None:
                    writer = pq.ParquetWriter(self.cache_path, ROUND_SCHEMA)
                writer.write_table(table)
            if writer is not None:
                writer.close()

        logger.info(
            f"Done. {fetched:,} rounds cached, {errors} errors. "
            f"File: {self.cache_path} "
            f"({self.cache_path.stat().st_size / 1e6:.1f} MB)"
        )
        self._load()

    def _load(self) -> None:
        self._rounds = (
            pq.read_table(self.cache_path)
            .to_pandas()
            .sort_values("updated_at")
            .reset_index(drop=True)
        )
        logger.info(
            f"Loaded {len(self._rounds):,} oracle rounds "
            f"({pd.Timestamp(self._rounds['updated_at'].min(), unit='s', tz='UTC')} → "
            f"{pd.Timestamp(self._rounds['updated_at'].max(), unit='s', tz='UTC')})"
        )

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def price_at(self, timestamp_s: int) -> float | None:
        """
        Return the oracle price that was active at timestamp_s.
        This is the price of the most recent update at or before the timestamp.
        Returns None if timestamp is before our earliest cached round.
        """
        if self._rounds is None:
            raise RuntimeError("Call build() first")

        idx = self._rounds["updated_at"].searchsorted(timestamp_s, side="right") - 1
        if idx < 0:
            return None
        return float(self._rounds["price"].iloc[idx])

    def last_update_at(self, timestamp_s: int) -> dict | None:
        """
        Return the most recent oracle round at or before timestamp_s.
        Returns dict with agg_round, price, updated_at, or None.
        """
        if self._rounds is None:
            raise RuntimeError("Call build() first")

        idx = self._rounds["updated_at"].searchsorted(timestamp_s, side="right") - 1
        if idx < 0:
            return None

        row = self._rounds.iloc[idx]
        return {
            "agg_round":  int(row["agg_round"]),
            "price":      float(row["price"]),
            "updated_at": int(row["updated_at"]),
        }

    def lag_at_close(self, market_close_s: int) -> float | None:
        """
        Seconds elapsed since the last oracle update as of market close.
        This is the oracle lag — how stale the resolution price is.
        Returns None if insufficient data.
        """
        last = self.last_update_at(market_close_s)
        if last is None:
            return None
        return float(market_close_s - last["updated_at"])

    # ------------------------------------------------------------------
    # DataFrame join
    # ------------------------------------------------------------------

    def join_to_observations(self, obs: pd.DataFrame) -> pd.DataFrame:
        """
        Attach oracle features to the observations DataFrame.

        Computes features once per market (at close timestamp, inferred
        from slug) and broadcasts to all rows in that market.

        Columns added:
          oracle_price_at_close   — last oracle price before market close
          oracle_lag_at_close_s   — seconds since last oracle update at close
          oracle_price_vs_btc     — oracle price minus BTC close price
                                    (requires btc_close column from KlineStore)
          oracle_updates_in_5m    — number of oracle updates in the 5min window
          oracle_updates_in_30s   — number of oracle updates in final 30s
        """
        if self._rounds is None:
            raise RuntimeError("Call build() first")

        slugs = obs["slug"].unique()
        logger.info(
            f"Computing oracle features for {len(slugs):,} markets"
        )

        rows    = []
        missing = 0

        for slug in slugs:
            close_s = self._slug_to_close_s(slug)
            if close_s is None:
                missing += 1
                continue

            open_s = close_s - 300   # 5-minute market

            last = self.last_update_at(close_s)
            if last is None:
                missing += 1
                continue

            # Count updates within the market window
            window = self._rounds[
                (self._rounds["updated_at"] >= open_s) &
                (self._rounds["updated_at"] <= close_s)
            ]
            final_30s = window[window["updated_at"] >= close_s - 30]

            rows.append({
                "slug":                    slug,
                "oracle_price_at_close":   last["price"],
                "oracle_lag_at_close_s":   float(close_s - last["updated_at"]),
                "oracle_updates_in_5m":    len(window),
                "oracle_updates_in_30s":   len(final_30s),
            })

        if missing:
            logger.warning(
                f"Could not compute oracle features for "
                f"{missing}/{len(slugs)} markets"
            )

        oracle_df = pd.DataFrame(rows)
        result    = obs.merge(oracle_df, on="slug", how="inner")

        # If BTC close price is present, compute oracle divergence
        if "candle_close" in result.columns:
            result["oracle_price_vs_btc"] = (
                result["oracle_price_at_close"] - result["candle_close"]
            )

        dropped = len(obs) - len(result)
        if dropped:
            logger.info(
                f"Dropped {dropped:,} rows from markets with no oracle features"
            )

        logger.info(
            f"Joined oracle features to {len(result):,} observations"
        )
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_round(self, agg_round: int) -> dict | None:
        """Fetch a single round from the contract. Returns None on failure."""
        round_id = (PHASE_ID << 64) | agg_round
        try:
            r = self._feed.functions.getRoundData(round_id).call()
            # r = (roundId, answer, startedAt, updatedAt, answeredInRound)
            if r[3] == 0:
                # updatedAt == 0 means the round was never completed
                return None
            return {
                "agg_round":  agg_round,
                "price":      r[1] / 10**DECIMALS,
                "started_at": r[2],
                "updated_at": r[3],
            }
        except Exception as e:
            logger.debug(f"Round {agg_round} failed: {e}")
            return None

    def _slug_to_close_s(self, slug: str) -> int | None:
        """
        Extract market close Unix timestamp (seconds) from slug name.
        Format: btc-updown-5m-{unix_seconds}
        The embedded timestamp is market OPEN time — add 300s for close.
        """
        try:
            return int(slug.split("-")[-1]) + 300
        except (ValueError, IndexError):
            return None

    def _round_range(self) -> tuple[int, int]:
        """
        Find the starting and ending aggregator rounds for the market
        data range using binary search on updated_at.
        """
        start_s, end_s = self._market_time_range()

        logger.info(
            f"Binary searching for round range: "
            f"{pd.Timestamp(start_s, unit='s', tz='UTC')} → "
            f"{pd.Timestamp(end_s, unit='s', tz='UTC')}"
        )

        start_agg = self._binary_search_round(start_s)
        end_agg   = self._binary_search_round(end_s)

        logger.info(
            f"Round range: aggRound {start_agg} → {end_agg} "
            f"({end_agg - start_agg:,} rounds)"
        )
        return start_agg, end_agg

    def _binary_search_round(
        self,
        target_s: int,
        lo: int = 1,
        hi: int = LATEST_AGG_ROUND,
    ) -> int:
        """
        Binary search for the aggregator round whose updated_at is
        closest to target_s without exceeding it.
        """
        while hi - lo > 10:
            mid = (lo + hi) // 2
            row = self._fetch_round(mid)
            time.sleep(REQUEST_DELAY)
            if row is None:
                # Round doesn't exist — search lower half
                hi = mid
                continue
            if row["updated_at"] < target_s:
                lo = mid
            else:
                hi = mid

        # Fine scan the landing zone
        best = lo
        for agg in range(lo, min(hi + 5, LATEST_AGG_ROUND + 1)):
            row = self._fetch_round(agg)
            time.sleep(REQUEST_DELAY)
            if row is None:
                continue
            if row["updated_at"] <= target_s:
                best = agg
            else:
                break

        return best

    def _market_time_range(self) -> tuple[int, int]:
        """Infer Unix second range from slug names in book_snapshots."""
        book_dir = self.data_dir / "book_snapshots"
        slugs    = [p.stem for p in book_dir.glob("*.parquet")]

        timestamps = []
        for slug in slugs:
            try:
                timestamps.append(int(slug.split("-")[-1]))
            except (ValueError, IndexError):
                continue

        if not timestamps:
            raise RuntimeError(f"No parquet files found in {book_dir}")

        return min(timestamps), max(timestamps) + 300
# Convenience runner
# ---------------------------------------------------------------------------

def build_oracle_store(
    data_dir: Path = Path("data"),
    cache_path: Path = Path("data/chainlink_rounds.parquet"),
    rebuild: bool = False,
    verbose: bool = True,
) -> OracleStore:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )
    store = OracleStore(data_dir=data_dir, cache_path=cache_path)
    store.build(rebuild=rebuild)
    return store


if __name__ == "__main__":
    store = build_oracle_store()

    rounds = store._rounds
    print(f"\nOracle round summary:")
    print(f"  Total rounds: {len(rounds):,}")
    print(f"  Range: {pd.Timestamp(rounds['updated_at'].min(), unit='s', tz='UTC')} "
          f"→ {pd.Timestamp(rounds['updated_at'].max(), unit='s', tz='UTC')}")

    # Distribution of update intervals
    intervals = rounds["updated_at"].diff().dropna()
    print(f"\nUpdate interval (seconds):")
    print(intervals.describe())

    # Lag distribution at market close
    book_dir = Path("data/book_snapshots")
    slugs    = [p.stem for p in book_dir.glob("*.parquet")]
    lags     = []
    for slug in slugs[:100]:   # sample first 100 markets
        close_s = int(slug.split("-")[-1]) + 300  # slug is open time
        lag     = store.lag_at_close(close_s)
        if lag is not None:
            lags.append(lag)

    lag_series = pd.Series(lags)
    print(f"\nOracle lag at market close (seconds) — sample of 100 markets:")
    print(lag_series.describe())
    print(f"Lag > 60s: {(lag_series > 60).mean():.1%}")
    print(f"Lag > 30s: {(lag_series > 30).mean():.1%}")
    print(f"Lag > 10s: {(lag_series > 10).mean():.1%}")