"""
Binance BTC/USDT kline fetcher with local parquet cache.

No authentication required — uses the public Binance REST API.

Fetches 1-minute OHLCV candles for the date range covered by your
Polymarket market files, caches to parquet, and exposes lookup
functions for joining BTC features to market observations.

Usage:
    from btc_klines import KlineStore

    store = KlineStore(
        data_dir=Path("data"),
        cache_path=Path("data/btc_klines.parquet"),
    )

    # Fetch and cache all klines covering your market data range
    # (no-op on subsequent calls unless rebuild=True)
    store.build(rebuild=False)

    # Look up BTC state at a specific timestamp
    features = store.features_at(timestamp_ms=1771419302695)

    # Attach BTC features to your observations DataFrame
    obs_with_btc = store.join_to_observations(obs)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
SYMBOL             = "BTCUSDT"
INTERVAL           = "1m"
MAX_PER_REQUEST    = 1000     # Binance hard limit
REQUEST_DELAY      = 0.05     # seconds between requests — well under rate limit
LOOKBACK_CANDLES   = 30       # how many 1m candles before market open to include
                               # in pre-market feature window (30m lookback)

KLINE_SCHEMA = pa.schema([
    ("open_time",    pa.timestamp("ms", tz="UTC")),
    ("close_time",   pa.timestamp("ms", tz="UTC")),
    ("open",         pa.float64()),
    ("high",         pa.float64()),
    ("low",          pa.float64()),
    ("close",        pa.float64()),
    ("volume",       pa.float64()),
    ("n_trades",     pa.int64()),
])


# ---------------------------------------------------------------------------
# KlineStore
# ---------------------------------------------------------------------------

class KlineStore:
    """
    Fetches, caches, and serves Binance 1-minute BTC/USDT klines.

    The cache is a single parquet file sorted by open_time. On first run,
    fetches all candles covering the range of your market data plus a
    LOOKBACK_CANDLES buffer before the first market. On subsequent runs,
    reads directly from parquet.

    Exposes two main interfaces:
      - features_at(timestamp_ms): BTC state at a specific point in time
      - join_to_observations(obs): attach BTC features to full obs DataFrame
    """

    def __init__(
        self,
        data_dir: Path = Path("data"),
        cache_path: Path = Path("data/btc_klines.parquet"),
    ):
        self.data_dir   = data_dir
        self.cache_path = cache_path
        self._klines: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Build / load
    # ------------------------------------------------------------------

    def build(self, rebuild: bool = False) -> None:
        """
        Fetch klines from Binance covering the full range of market data
        and write to parquet cache. No-op if cache exists and rebuild=False.
        """
        if self.cache_path.exists() and not rebuild:
            logger.info(f"Kline cache exists at {self.cache_path} — skipping fetch")
            self._load()
            return

        start_ms, end_ms = self._market_time_range()
        if start_ms is None:
            raise RuntimeError(
                "No market files found — cannot determine date range. "
                "Check data_dir is correct."
            )

        # Buffer: fetch LOOKBACK_CANDLES extra minutes before first market
        # so pre-market features are available for every market in the dataset
        buffered_start_ms = start_ms - (LOOKBACK_CANDLES * 60 * 1000)

        logger.info(
            f"Fetching {SYMBOL} {INTERVAL} klines: "
            f"{pd.Timestamp(buffered_start_ms, unit='ms', tz='UTC')} → "
            f"{pd.Timestamp(end_ms, unit='ms', tz='UTC')}"
        )

        frames = []
        batch_start = buffered_start_ms
        batch_num   = 0

        while batch_start < end_ms:
            batch_end = min(
                batch_start + MAX_PER_REQUEST * 60 * 1000,
                end_ms,
            )

            df = self._fetch_batch(batch_start, batch_end)
            if df.empty:
                logger.warning(
                    f"Empty batch at {pd.Timestamp(batch_start, unit='ms', tz='UTC')}"
                    f" — stopping early"
                )
                break

            frames.append(df)
            batch_num += 1

            # Advance to the open_time of the last candle + 1 minute
            # (not batch_end, since Binance may return fewer than requested)
            last_open_ms = int(df["open_time"].iloc[-1].timestamp() * 1000)
            batch_start  = last_open_ms + 60_000

            if batch_num % 10 == 0:
                logger.info(
                    f"  Fetched {batch_num} batches "
                    f"({sum(len(f) for f in frames):,} candles so far)"
                )

            time.sleep(REQUEST_DELAY)

        if not frames:
            raise RuntimeError("No klines fetched — check Binance API connectivity")

        klines = pd.concat(frames, ignore_index=True)
        klines = klines.drop_duplicates("open_time").sort_values("open_time")

        # Write to parquet
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pandas(klines, schema=KLINE_SCHEMA, preserve_index=False)
        pq.write_table(table, self.cache_path, compression="snappy")

        logger.info(
            f"Cached {len(klines):,} candles to {self.cache_path} "
            f"({self.cache_path.stat().st_size / 1e6:.1f} MB)"
        )
        self._klines = klines

    def _load(self) -> None:
        """Load klines from parquet cache into memory."""
        self._klines = (
            pq.read_table(self.cache_path)
            .to_pandas()
            .sort_values("open_time")
            .reset_index(drop=True)
        )
        logger.info(
            f"Loaded {len(self._klines):,} cached klines "
            f"({self._klines['open_time'].min()} → {self._klines['open_time'].max()})"
        )

    # ------------------------------------------------------------------
    # Feature lookup
    # ------------------------------------------------------------------

    def features_at(self, timestamp_ms: int) -> dict | None:
        """
        Return BTC features for the 1-minute candle containing timestamp_ms,
        plus derived features from the preceding LOOKBACK_CANDLES candles.

        Returns None if insufficient history exists for this timestamp.

        Features returned:
          candle_*        — OHLCV of the current 1m candle
          ret_1m          — 1-minute return (close/prev_close - 1)
          ret_5m          — 5-minute return
          ret_15m         — 15-minute return
          ret_30m         — 30-minute return
          vol_5m          — realised volatility over last 5 candles (std of returns)
          vol_15m         — realised volatility over last 15 candles
          vol_30m         — realised volatility over last 30 candles
          volume_ratio_5m — current candle volume / mean of last 5 candles
          high_low_range  — (high - low) / close of current candle
        """
        if self._klines is None:
            raise RuntimeError("Call build() before features_at()")

        ts = pd.Timestamp(timestamp_ms, unit="ms", tz="UTC")

        # Find the candle whose open_time <= ts < close_time
        idx = self._klines["open_time"].searchsorted(ts, side="right") - 1
        if idx < LOOKBACK_CANDLES or idx >= len(self._klines):
            return None

        window = self._klines.iloc[idx - LOOKBACK_CANDLES : idx + 1]
        candle = window.iloc[-1]
        closes = window["close"].values

        # Return series
        returns = pd.Series(closes).pct_change().dropna().values

        def vol(n: int) -> float:
            return float(returns[-n:].std()) if len(returns) >= n else float("nan")

        def ret(n: int) -> float:
            if len(closes) <= n:
                return float("nan")
            return float(closes[-1] / closes[-(n + 1)] - 1)

        prev_volume_mean = window["volume"].iloc[:-1].tail(5).mean()

        return {
            "candle_open":      float(candle["open"]),
            "candle_high":      float(candle["high"]),
            "candle_low":       float(candle["low"]),
            "candle_close":     float(candle["close"]),
            "candle_volume":    float(candle["volume"]),
            "candle_n_trades":  int(candle["n_trades"]),
            "ret_1m":           ret(1),
            "ret_5m":           ret(5),
            "ret_15m":          ret(15),
            "ret_30m":          ret(30),
            "vol_5m":           vol(4),
            "vol_15m":          vol(14),
            "vol_30m":          vol(29),
            "volume_ratio_5m":  float(candle["volume"] / prev_volume_mean)
                                 if prev_volume_mean > 0 else float("nan"),
            "high_low_range":   float((candle["high"] - candle["low"]) / candle["close"])
                                 if candle["close"] > 0 else float("nan"),
        }

    def join_to_observations(
        self,
        obs: pd.DataFrame,
        at: str = "market_open",
    ) -> pd.DataFrame:
        """
        Attach BTC features to an observations DataFrame.

        Args:
            obs: DataFrame from CalibrationAnalyser.build_observations().
            at:  When to sample BTC features:
                 'market_open'  — BTC state at the start of each market
                                  (same features for all rows in a market)
                 'snapshot'     — BTC state at each individual snapshot timestamp
                                  (more granular, higher memory use)

        Returns:
            obs with BTC feature columns appended. Rows where BTC features
            could not be computed (insufficient history) are dropped.
        """
        if self._klines is None:
            raise RuntimeError("Call build() before join_to_observations()")

        if at == "market_open":
            return self._join_at_market_open(obs)
        elif at == "snapshot":
            return self._join_at_snapshot(obs)
        else:
            raise ValueError(f"at must be 'market_open' or 'snapshot', got {at!r}")

    def _join_at_market_open(self, obs: pd.DataFrame) -> pd.DataFrame:
        """
        Compute BTC features once per market (at open timestamp) and broadcast
        to all rows. Much faster than per-snapshot lookup.

        Market open timestamp is inferred from the slug name:
        btc-updown-5m-1771419300 → Unix seconds 1771419300 → ms * 1000
        """
        # Extract open timestamp from slug (Unix seconds embedded in slug name)
        def slug_to_open_ms(slug: str) -> int | None:
            try:
                return int(slug.split("-")[-1]) * 1000
            except (ValueError, IndexError):
                return None

        slugs = obs["slug"].unique()
        logger.info(f"Computing BTC features at market open for {len(slugs):,} markets")

        btc_rows = []
        missing  = 0
        for slug in slugs:
            open_ms = slug_to_open_ms(slug)
            if open_ms is None:
                missing += 1
                continue

            features = self.features_at(open_ms)
            if features is None:
                missing += 1
                continue

            btc_rows.append({"slug": slug, **features})

        if missing:
            logger.warning(
                f"Could not compute BTC features for {missing}/{len(slugs)} markets "
                f"(insufficient kline history or unparseable slug)"
            )

        if not btc_rows:
            raise RuntimeError("No BTC features computed — check kline coverage")

        btc_df = pd.DataFrame(btc_rows)
        result = obs.merge(btc_df, on="slug", how="inner")

        dropped = len(obs) - len(result)
        if dropped:
            logger.info(f"Dropped {dropped:,} rows from markets with no BTC features")

        logger.info(f"Joined BTC features to {len(result):,} observations")
        return result

    def _join_at_snapshot(self, obs: pd.DataFrame) -> pd.DataFrame:
        """
        Compute BTC features at each individual snapshot timestamp.
        Slower but gives intra-market BTC state for each observation.
        Uses vectorized candle lookup rather than row-by-row features_at().
        """
        logger.info(
            f"Computing BTC features at snapshot level for "
            f"{len(obs):,} observations — this may take a while"
        )

        klines = self._klines.copy()
        klines["open_time_ms"] = klines["open_time"].astype("int64") // 1_000_000

        obs = obs.copy()
        obs["_candle_idx"] = klines["open_time_ms"].searchsorted(
            obs["exchange_timestamp"].values, side="right"
        ) - 1

        # Only keep rows where we have enough lookback
        obs = obs[obs["_candle_idx"] >= LOOKBACK_CANDLES].copy()

        # For per-snapshot features we compute the most useful subset:
        # returns and volatility at the snapshot moment
        closes = klines["close"].values

        def safe_ret(idx_arr, n):
            results = []
            for idx in idx_arr:
                if idx - n < 0:
                    results.append(float("nan"))
                else:
                    results.append(closes[idx] / closes[idx - n] - 1)
            return results

        def safe_vol(idx_arr, n):
            results = []
            for idx in idx_arr:
                window = closes[max(0, idx - n): idx + 1]
                if len(window) < 2:
                    results.append(float("nan"))
                else:
                    rets = pd.Series(window).pct_change().dropna()
                    results.append(float(rets.std()))
            return results

        idxs = obs["_candle_idx"].values
        obs["btc_close"]   = closes[idxs]
        obs["btc_ret_1m"]  = safe_ret(idxs, 1)
        obs["btc_ret_5m"]  = safe_ret(idxs, 5)
        obs["btc_ret_15m"] = safe_ret(idxs, 15)
        obs["btc_vol_5m"]  = safe_vol(idxs, 4)
        obs["btc_vol_15m"] = safe_vol(idxs, 14)

        obs = obs.drop(columns=["_candle_idx"])
        logger.info(f"Snapshot-level join complete: {len(obs):,} rows")
        return obs

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_batch(self, start_ms: int, end_ms: int) -> pd.DataFrame:
        """Fetch one batch of klines from Binance."""
        try:
            resp = requests.get(
                BINANCE_KLINES_URL,
                params={
                    "symbol":    SYMBOL,
                    "interval":  INTERVAL,
                    "startTime": start_ms,
                    "endTime":   end_ms,
                    "limit":     MAX_PER_REQUEST,
                },
                timeout=15,
            )
            resp.raise_for_status()
        except requests.RequestException as e:
            logger.error(f"Binance request failed: {e}")
            return pd.DataFrame()

        raw = resp.json()
        if not raw:
            return pd.DataFrame()

        cols = [
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "n_trades",
            "taker_buy_base", "taker_buy_quote", "_ignore",
        ]
        df = pd.DataFrame(raw, columns=cols)
        df["open_time"]  = pd.to_datetime(df["open_time"],  unit="ms", utc=True)
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        df["n_trades"] = df["n_trades"].astype(int)

        return df[["open_time", "close_time", "open", "high", "low",
                   "close", "volume", "n_trades"]]

    def _market_time_range(self) -> tuple[int | None, int | None]:
        """
        Infer start and end Unix milliseconds from slug names in the
        book_snapshots directory.

        Slug format: btc-updown-5m-{unix_seconds}
        """
        book_dir = self.data_dir / "book_snapshots"
        slugs = [p.stem for p in book_dir.glob("*.parquet")]
        if not slugs:
            return None, None

        timestamps = []
        for slug in slugs:
            try:
                timestamps.append(int(slug.split("-")[-1]))
            except (ValueError, IndexError):
                continue

        if not timestamps:
            return None, None

        start_s = min(timestamps)
        end_s   = max(timestamps) + 300   # add 5 minutes for the last market window

        return start_s * 1000, end_s * 1000


# ---------------------------------------------------------------------------
# Convenience runner
# ---------------------------------------------------------------------------

def build_kline_store(
    data_dir: Path = Path("data"),
    cache_path: Path = Path("data/btc_klines.parquet"),
    rebuild: bool = False,
    verbose: bool = True,
) -> KlineStore:
    """
    Build and return a ready-to-use KlineStore.
    Fetches from Binance if cache doesn't exist, otherwise loads from disk.
    """
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )
    store = KlineStore(data_dir=data_dir, cache_path=cache_path)
    store.build(rebuild=rebuild)
    return store


# if __name__ == "__main__":
#     store = build_kline_store()

#     # Quick sanity check
#     klines = store._klines
#     print(f"\nKline summary:")
#     print(f"  Candles: {len(klines):,}")
#     print(f"  Range:   {klines['open_time'].min()} → {klines['open_time'].max()}")
#     print(f"  BTC price range: ${klines['low'].min():,.0f} – ${klines['high'].max():,.0f}")

#     # Test feature lookup at the first candle with enough lookback
#     test_ts = int(klines["open_time"].iloc[LOOKBACK_CANDLES + 1].timestamp() * 1000)
#     features = store.features_at(test_ts)
#     print(f"\nSample features at {pd.Timestamp(test_ts, unit='ms', tz='UTC')}:")
#     for k, v in features.items():
#         print(f"  {k:20s}: {v:.6f}" if isinstance(v, float) else f"  {k:20s}: {v}")