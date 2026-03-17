"""
Binance spot quotes downloader and normalizer — generalized for any asset.

Downloads daily quote files from Telonex (Binance exchange) and normalizes
to a canonical parquet with derived features (volatility, returns) suitable
for use as spot price context in the XGBoost pipeline.

Usage:
  python spot_pipeline.py btcusdt download
  python spot_pipeline.py ethusdt normalize
  python spot_pipeline.py solusdt all
  python spot_pipeline.py xrpusdt all
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

API_KEY = os.environ.get("TELONEX_API_KEY", "")
if not API_KEY:
    import json
    _cfg = json.loads(Path("config.json").read_text()) if Path("config.json").exists() else {}
    API_KEY = _cfg.get("telonex", {}).get("api_key", "")
if not API_KEY:
    raise RuntimeError("Set TELONEX_API_KEY env var or add to config.json")

FROM_DATE = "2025-10-01"
TO_DATE = "2026-03-14"  # exclusive

VOL_WINDOWS = [30, 60, 300]
RETURN_WINDOWS = [5, 15, 60]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

QUOTE_SCHEMA = pa.schema([
    pa.field("timestamp_ms", pa.int64()),
    pa.field("mid_price", pa.float64()),
    pa.field("spread", pa.float64()),
    pa.field("bid_price", pa.float64()),
    pa.field("ask_price", pa.float64()),
    pa.field("log_return", pa.float64()),
    pa.field("rvol_30s", pa.float64()),
    pa.field("rvol_60s", pa.float64()),
    pa.field("rvol_300s", pa.float64()),
    pa.field("ret_5s", pa.float64()),
    pa.field("ret_15s", pa.float64()),
    pa.field("ret_60s", pa.float64()),
])

# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

async def run_download(symbol: str, raw_dir: Path):
    raw_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Downloading Binance {symbol} quotes {FROM_DATE} -> {TO_DATE}...")
    files = await download_async(
        api_key=API_KEY,
        exchange="binance",
        channel="quotes",
        from_date=FROM_DATE,
        to_date=TO_DATE,
        slug=symbol,
        download_dir=str(raw_dir),
        force_download=False,
    )
    log.info(f"Downloaded {len(files)} files")
    return files

# ---------------------------------------------------------------------------
# Normalize
# ---------------------------------------------------------------------------

def load_and_clean(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["bid_price"] = pd.to_numeric(df["bid_price"], errors="coerce")
    df["ask_price"] = pd.to_numeric(df["ask_price"], errors="coerce")
    df["bid_size"] = pd.to_numeric(df["bid_size"], errors="coerce")
    df["ask_size"] = pd.to_numeric(df["ask_size"], errors="coerce")
    df = df.dropna(subset=["bid_price", "ask_price"])
    df["mid_price"] = (df["bid_price"] + df["ask_price"]) / 2
    df["spread"] = df["ask_price"] - df["bid_price"]
    df["timestamp_ms"] = df["timestamp_us"] // 1000
    return df[["timestamp_ms", "mid_price", "spread", "bid_price", "ask_price"]]


def resample_to_1s(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["bar_ts"] = (df["timestamp_ms"] // 1000) * 1000
    resampled = (df.drop(columns=["timestamp_ms"])
                   .groupby("bar_ts")
                   .last()
                   .reset_index()
                   .rename(columns={"bar_ts": "timestamp_ms"}))
    return resampled.sort_values("timestamp_ms").reset_index(drop=True)


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["log_return"] = np.log(df["mid_price"] / df["mid_price"].shift(1))
    for w in RETURN_WINDOWS:
        df[f"ret_{w}s"] = np.log(df["mid_price"] / df["mid_price"].shift(w))
    for w in VOL_WINDOWS:
        df[f"rvol_{w}s"] = df["log_return"].rolling(w).std()
    return df


def run_normalize(raw_dir: Path, out_file: Path):
    out_file.parent.mkdir(parents=True, exist_ok=True)
    raw_files = sorted(raw_dir.glob("*.parquet"))
    if not raw_files:
        log.error(f"No raw files in {raw_dir}")
        return

    log.info(f"Loading {len(raw_files)} daily files...")
    daily = [load_and_clean(f) for f in raw_files]
    combined = pd.concat(daily, ignore_index=True).sort_values("timestamp_ms")

    log.info(f"Loaded {len(combined):,} raw ticks, resampling to 1s bars...")
    resampled = resample_to_1s(combined)
    resampled = resampled.drop_duplicates(subset=["timestamp_ms"]).reset_index(drop=True)
    log.info(f"Resampled to {len(resampled):,} 1s bars")

    log.info("Computing derived features...")
    featured = add_derived_features(resampled)
    featured = featured[[f.name for f in QUOTE_SCHEMA]]

    table = pa.Table.from_pandas(featured, schema=QUOTE_SCHEMA, preserve_index=False)
    pq.write_table(table, out_file, compression="snappy")
    log.info(f"Wrote {len(featured):,} rows to {out_file}")

    log.info(f"Timestamp range: {featured['timestamp_ms'].min()} -> {featured['timestamp_ms'].max()}")
    log.info(f"Mid price range: {featured['mid_price'].min():.2f} -> {featured['mid_price'].max():.2f}")

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python spot_pipeline.py <symbol> [download|normalize|all]")
        print("  symbol: btcusdt, ethusdt, solusdt, xrpusdt")
        sys.exit(1)

    symbol = sys.argv[1].lower()
    cmd = sys.argv[2] if len(sys.argv) > 2 else "all"

    # Derive asset name from symbol (ethusdt -> eth)
    asset = symbol.replace("usdt", "")

    raw_dir = Path(f"./datasets/telonex_raw/binance_{symbol}")
    out_dir = Path(f"./data/{asset}_quotes")
    out_file = out_dir / f"{symbol}_quotes.parquet"

    log.info(f"Symbol: {symbol}, Raw: {raw_dir}, Out: {out_file}")

    if cmd in ("download", "all"):
        asyncio.run(run_download(symbol, raw_dir))

    if cmd in ("normalize", "all"):
        run_normalize(raw_dir, out_file)
