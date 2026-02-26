"""
Binance BTC/USDT quotes downloader and normalizer.

Downloads daily quote files from Telonex and normalizes to a canonical
parquet with derived features suitable for use as RL environment context.

Steps:
  1. download  — fetch raw daily parquets to ./datasets/telonex_raw/binance/
  2. normalize — resample to 1s bars, compute derived features, write to
                 ./data/btc_quotes/btcusdt_quotes.parquet (single file)

Run:
  python btc_pipeline.py download
  python btc_pipeline.py normalize
  python btc_pipeline.py all
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from telonex import download_async

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_KEY  = os.environ["TELONEX_API_KEY"]
RAW_DIR  = Path("./datasets/telonex_raw/binance")
OUT_DIR  = Path("./data/btc_quotes")
OUT_FILE = OUT_DIR / "btcusdt_quotes.parquet"

FROM_DATE = "2026-02-12"
TO_DATE   = "2026-02-25"  # exclusive

# Derived feature windows (in seconds, applied to 1s resampled data)
VOL_WINDOWS    = [30, 60, 300]   # rolling realized volatility lookbacks
RETURN_WINDOWS = [5, 15, 60]     # trailing log return windows

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

QUOTE_SCHEMA = pa.schema([
    pa.field("timestamp_ms",   pa.int64()),    # 1s bar open timestamp, milliseconds
    pa.field("mid_price",      pa.float64()),  # (bid + ask) / 2
    pa.field("spread",         pa.float64()),  # ask - bid
    pa.field("bid_price",      pa.float64()),  # last bid in bar
    pa.field("ask_price",      pa.float64()),  # last ask in bar
    pa.field("log_return",     pa.float64()),  # log(mid_t / mid_{t-1})
    # Rolling realized volatility: std of log returns over window
    pa.field("rvol_30s",       pa.float64()),
    pa.field("rvol_60s",       pa.float64()),
    pa.field("rvol_300s",      pa.float64()),
    # Trailing log returns
    pa.field("ret_5s",         pa.float64()),  # log(mid_t / mid_{t-5s})
    pa.field("ret_15s",        pa.float64()),
    pa.field("ret_60s",        pa.float64()),
])

# ---------------------------------------------------------------------------
# Step 1: Download
# ---------------------------------------------------------------------------

async def run_download():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    log.info(f"Downloading Binance BTC quotes {FROM_DATE} -> {TO_DATE}...")
    files = await download_async(
        api_key=API_KEY,
        exchange="binance",
        channel="quotes",
        from_date=FROM_DATE,
        to_date=TO_DATE,
        slug="btcusdt",
        download_dir=str(RAW_DIR),
        force_download=False,
    )
    log.info(f"Downloaded {len(files)} files")
    return files

# ---------------------------------------------------------------------------
# Step 2: Normalize
# ---------------------------------------------------------------------------

def load_and_clean(path: Path) -> pd.DataFrame:
    """Load one raw daily file, cast types, return clean DataFrame."""
    df = pd.read_parquet(path)
    df["bid_price"] = pd.to_numeric(df["bid_price"], errors="coerce")
    df["ask_price"] = pd.to_numeric(df["ask_price"], errors="coerce")
    df["bid_size"]  = pd.to_numeric(df["bid_size"],  errors="coerce")
    df["ask_size"]  = pd.to_numeric(df["ask_size"],  errors="coerce")
    df = df.dropna(subset=["bid_price", "ask_price"])
    df["mid_price"]    = (df["bid_price"] + df["ask_price"]) / 2
    df["spread"]       = df["ask_price"] - df["bid_price"]
    df["timestamp_ms"] = df["timestamp_us"] // 1000
    return df[["timestamp_ms", "mid_price", "spread", "bid_price", "ask_price"]]


def resample_to_1s(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample tick quotes to 1-second bars using last-value-in-bar.
    Timestamp is the bar open (floor to second in ms).
    Uses a separate bar_ts column to avoid timestamp_ms appearing in both
    the groupby key and the remaining columns after reset_index.
    """
    df = df.copy()
    df["bar_ts"] = (df["timestamp_ms"] // 1000) * 1000
    resampled = (df.drop(columns=["timestamp_ms"])
                   .groupby("bar_ts")
                   .last()
                   .reset_index()
                   .rename(columns={"bar_ts": "timestamp_ms"}))
    return resampled.sort_values("timestamp_ms").reset_index(drop=True)


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add log returns and rolling realized volatility to a 1s-resampled DataFrame.
    All windows are in seconds = rows (since data is 1s bars).
    """
    df = df.copy()

    # Log returns
    df["log_return"] = np.log(df["mid_price"] / df["mid_price"].shift(1))

    # Trailing log returns over N seconds
    for w in RETURN_WINDOWS:
        df[f"ret_{w}s"] = np.log(df["mid_price"] / df["mid_price"].shift(w))

    # Rolling realized volatility: std of log returns over window
    for w in VOL_WINDOWS:
        df[f"rvol_{w}s"] = df["log_return"].rolling(w).std()

    return df


def run_normalize():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    raw_files = sorted(RAW_DIR.glob("*.parquet"))
    if not raw_files:
        log.error(f"No raw files in {RAW_DIR}")
        return

    log.info(f"Loading {len(raw_files)} daily files...")
    daily = [load_and_clean(f) for f in raw_files]
    combined = pd.concat(daily, ignore_index=True).sort_values("timestamp_ms")

    log.info(f"Loaded {len(combined):,} raw ticks, resampling to 1s bars...")
    resampled = resample_to_1s(combined)
    # Dedup after resampling — overlapping daily files can produce duplicate seconds
    resampled = resampled.drop_duplicates(subset=["timestamp_ms"]).reset_index(drop=True)
    log.info(f"Resampled to {len(resampled):,} 1s bars")

    log.info("Computing derived features...")
    featured = add_derived_features(resampled)

    # Reorder columns to match schema
    featured = featured[[f.name for f in QUOTE_SCHEMA]]

    table = pa.Table.from_pandas(featured, schema=QUOTE_SCHEMA, preserve_index=False)
    pq.write_table(table, OUT_FILE, compression="snappy")
    log.info(f"Wrote {len(featured):,} rows to {OUT_FILE}")

    # Sanity check
    log.info(f"Timestamp range: {featured['timestamp_ms'].min()} -> {featured['timestamp_ms'].max()}")
    log.info(f"Mid price range: {featured['mid_price'].min():.2f} -> {featured['mid_price'].max():.2f}")
    log.info(f"Null counts:\n{featured.isnull().sum().to_string()}")

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"

    if cmd in ("download", "all"):
        asyncio.run(run_download())

    if cmd in ("normalize", "all"):
        run_normalize()