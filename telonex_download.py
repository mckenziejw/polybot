"""
Generic Telonex downloader and normalizer for any asset/timeframe.

Usage:
  python telonex_download.py btc 15m download
  python telonex_download.py btc 15m normalize
  python telonex_download.py btc 15m all
  python telonex_download.py sol 5m all
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

API_KEY     = os.environ["TELONEX_API_KEY"]
MARKETS_URL = "https://api.telonex.io/v1/datasets/polymarket/markets"
CONCURRENCY = 10
N_LEVELS    = 5

TIMEFRAME_SECONDS = {
    "5m":  300,
    "15m": 900,
    "1h":  3600,
    "4h":  14400,
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

BOOK_SCHEMA = pa.schema([
    pa.field("exchange_timestamp", pa.int64()),
    pa.field("received_timestamp", pa.int64()),
    pa.field("slug",               pa.string()),
    pa.field("market",             pa.string()),
    pa.field("asset_id",           pa.string()),
    pa.field("token_label",        pa.string()),
    pa.field("mid_price",          pa.float64()),
    pa.field("spread",             pa.float64()),
    pa.field("book_imbalance",     pa.float64()),
    *[pa.field(f"bid_price_{i+1}", pa.float64()) for i in range(N_LEVELS)],
    *[pa.field(f"bid_size_{i+1}",  pa.float64()) for i in range(N_LEVELS)],
    *[pa.field(f"ask_price_{i+1}", pa.float64()) for i in range(N_LEVELS)],
    *[pa.field(f"ask_size_{i+1}",  pa.float64()) for i in range(N_LEVELS)],
])

# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

LOCAL_INDEX_CACHE = Path("./datasets/telonex_markets_index.parquet")

def load_markets(asset: str, timeframe: str) -> pd.DataFrame:
    slug_pattern = f"{asset}-updown-{timeframe}"
    log.info(f"Loading markets matching '{slug_pattern}'...")
    if LOCAL_INDEX_CACHE.exists():
        log.info(f"Using cached market index: {LOCAL_INDEX_CACHE}")
        df = pd.read_parquet(LOCAL_INDEX_CACHE)
    else:
        log.info("Fetching market index from Telonex API (may be slow)...")
        df = pd.read_parquet(MARKETS_URL)
    matched = df[
        df["slug"].str.contains(slug_pattern, na=False) &
        (df["book_snapshot_5_from"] != "")
    ].copy()
    log.info(f"Found {len(matched)} markets with book_snapshot_5 data")
    return matched


async def download_one(slug, asset_id, from_date, to_date, semaphore):
    to_date_excl = (pd.Timestamp(to_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    async with semaphore:
        try:
            return await download_async(
                api_key=API_KEY,
                exchange="polymarket",
                channel="book_snapshot_5",
                from_date=from_date,
                to_date=to_date_excl,
                asset_id=asset_id,
                download_dir=str(RAW_DIR),
                force_download=False,
            )
        except Exception as e:
            log.error(f"Failed {slug}: {type(e).__name__}: {e}")
            return []


async def run_downloads(markets: pd.DataFrame):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    tasks = []
    for _, row in markets.iterrows():
        for asset_col in ("asset_id_0", "asset_id_1"):
            tasks.append((
                row["slug"], row[asset_col],
                row["book_snapshot_5_from"], row["book_snapshot_5_to"],
            ))

    log.info(f"Downloading {len(tasks)} token datasets ({CONCURRENCY} concurrent)...")
    semaphore = asyncio.Semaphore(CONCURRENCY)
    coros = [download_one(*t, semaphore) for t in tasks]

    all_files, done = [], 0
    for coro in asyncio.as_completed(coros):
        all_files.extend(await coro)
        done += 1
        if done % 50 == 0 or done == len(tasks):
            log.info(f"  {done}/{len(tasks)} done, {len(all_files)} files")

    log.info(f"Download complete: {len(all_files)} files in {RAW_DIR}")

# ---------------------------------------------------------------------------
# Normalize
# ---------------------------------------------------------------------------

def normalize_raw_file(path: Path, market_open_ms: int, market_close_ms: int) -> pd.DataFrame | None:
    try:
        df = pd.read_parquet(path)
    except Exception as e:
        log.error(f"Failed to read {path}: {e}")
        return None

    if df.empty:
        return None

    open_us  = market_open_ms  * 1000
    close_us = market_close_ms * 1000
    df = df[(df["timestamp_us"] >= open_us) & (df["timestamp_us"] < close_us)]

    if df.empty:
        return None

    df = df.drop_duplicates(subset=["timestamp_us", "asset_id"])

    for side in ("bid", "ask"):
        for i in range(N_LEVELS):
            for field in ("price", "size"):
                col = f"{side}_{field}_{i}"
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    bid_price_arr = np.column_stack([df[f"bid_price_{i}"].values for i in range(N_LEVELS)])
    bid_size_arr  = np.column_stack([df[f"bid_size_{i}"].values  for i in range(N_LEVELS)])
    bid_sort_idx  = np.argsort(-bid_price_arr, axis=1)
    bid_price_arr = np.take_along_axis(bid_price_arr, bid_sort_idx, axis=1)
    bid_size_arr  = np.take_along_axis(bid_size_arr,  bid_sort_idx, axis=1)

    ask_price_arr = np.column_stack([df[f"ask_price_{i}"].values for i in range(N_LEVELS)])
    ask_size_arr  = np.column_stack([df[f"ask_size_{i}"].values  for i in range(N_LEVELS)])

    best_bid  = bid_price_arr[:, 0]
    best_ask  = ask_price_arr[:, 0]
    mid_price = (best_bid + best_ask) / 2
    spread    = best_ask - best_bid

    total_bid = bid_size_arr.sum(axis=1)
    total_ask = ask_size_arr.sum(axis=1)
    denom     = total_bid + total_ask
    imbalance = np.where(denom > 0, (total_bid - total_ask) / denom, 0.0)

    return pd.DataFrame({
        "exchange_timestamp": df["timestamp_us"] // 1000,
        "received_timestamp": 0,
        "slug":               df["slug"],
        "market":             df["market_id"],
        "asset_id":           df["asset_id"],
        "token_label":        df["outcome"],
        "mid_price":          mid_price,
        "spread":             spread,
        "book_imbalance":     imbalance,
        **{f"bid_price_{i+1}": bid_price_arr[:, i] for i in range(N_LEVELS)},
        **{f"bid_size_{i+1}":  bid_size_arr[:, i]  for i in range(N_LEVELS)},
        **{f"ask_price_{i+1}": ask_price_arr[:, i]  for i in range(N_LEVELS)},
        **{f"ask_size_{i+1}":  ask_size_arr[:, i]   for i in range(N_LEVELS)},
    })


def build_asset_slug_map(asset: str, timeframe: str) -> dict[str, str]:
    slug_pattern = f"{asset}-updown-{timeframe}"
    log.info(f"Building asset_id -> slug lookup for '{slug_pattern}'...")
    if LOCAL_INDEX_CACHE.exists():
        df = pd.read_parquet(LOCAL_INDEX_CACHE)
    else:
        df = pd.read_parquet(MARKETS_URL)
    matched = df[df["slug"].str.contains(slug_pattern, na=False)]
    mapping = {}
    for _, row in matched.iterrows():
        mapping[row["asset_id_0"]] = row["slug"]
        mapping[row["asset_id_1"]] = row["slug"]
    log.info(f"Loaded {len(mapping)} asset_id -> slug mappings")
    return mapping


def run_normalize(asset: str, timeframe: str, duration_ms: int):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_files = list(RAW_DIR.glob("*.parquet"))
    if not raw_files:
        log.error(f"No raw files found in {RAW_DIR}")
        return

    asset_slug = build_asset_slug_map(asset, timeframe)

    slug_files: dict[str, list[Path]] = {}
    unmapped = 0
    for f in raw_files:
        parts = f.stem.split("_")
        try:
            date_idx = next(i for i, p in enumerate(parts) if p.startswith("202"))
            asset_id = parts[date_idx + 1]
        except StopIteration:
            continue
        slug = asset_slug.get(asset_id)
        if not slug:
            unmapped += 1
            continue
        slug_files.setdefault(slug, []).append(f)

    if unmapped:
        log.warning(f"{unmapped} files had no slug mapping — skipped")
    log.info(f"Normalizing {len(slug_files)} slugs from {len(raw_files)} raw files...")

    written = skipped = errors = 0
    for i, (slug, files) in enumerate(slug_files.items()):
        out_path = OUT_DIR / f"{slug}.parquet"
        if out_path.exists():
            skipped += 1
            continue

        try:
            market_open_ms  = int(slug.split("-")[-1]) * 1000
            market_close_ms = market_open_ms + duration_ms
        except (ValueError, IndexError):
            errors += 1
            continue

        dfs = [n for f in files if (n := normalize_raw_file(f, market_open_ms, market_close_ms)) is not None]
        if not dfs:
            errors += 1
            continue

        combined = pd.concat(dfs, ignore_index=True)
        combined = (combined
                    .drop_duplicates(subset=["exchange_timestamp", "asset_id"])
                    .sort_values("exchange_timestamp")
                    .reset_index(drop=True))

        try:
            table = pa.Table.from_pandas(combined, schema=BOOK_SCHEMA, preserve_index=False)
            pq.write_table(table, out_path, compression="snappy")
            written += 1
        except Exception as e:
            log.error(f"Failed writing {slug}: {e}")
            errors += 1

        if (i + 1) % 500 == 0:
            log.info(f"  {i+1}/{len(slug_files)} slugs processed")

    log.info(f"Normalize complete: {written} written, {skipped} skipped, {errors} errors")

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python telonex_download.py <asset> <timeframe> [download|normalize|all]")
        print("  asset: btc, eth, sol, xrp")
        print("  timeframe: 5m, 15m, 1h, 4h")
        sys.exit(1)

    asset = sys.argv[1].lower()
    timeframe = sys.argv[2].lower()
    cmd = sys.argv[3] if len(sys.argv) > 3 else "all"

    if timeframe not in TIMEFRAME_SECONDS:
        print(f"Unknown timeframe: {timeframe}. Use: {list(TIMEFRAME_SECONDS.keys())}")
        sys.exit(1)

    duration_ms = TIMEFRAME_SECONDS[timeframe] * 1000

    RAW_DIR = Path(f"./datasets/telonex_{asset}_{timeframe}_raw")
    # Match legacy directory naming conventions:
    #   BTC 5m  -> data/telonex_book_snapshots/
    #   BTC 4h  -> data/telonex_btc_4h_book_snapshots/
    #   ALT 5m  -> data/telonex_book_snapshots_{asset}/
    #   Other   -> data/telonex_{asset}_{timeframe}_book_snapshots/
    if asset == "btc" and timeframe == "5m":
        OUT_DIR = Path("./data/telonex_book_snapshots")
    elif asset != "btc" and timeframe == "5m":
        OUT_DIR = Path(f"./data/telonex_book_snapshots_{asset}")
    else:
        OUT_DIR = Path(f"./data/telonex_{asset}_{timeframe}_book_snapshots")

    log.info(f"Asset: {asset}, Timeframe: {timeframe}, Duration: {duration_ms}ms")
    log.info(f"Raw: {RAW_DIR}, Out: {OUT_DIR}")

    if cmd in ("download", "all"):
        markets = load_markets(asset, timeframe)
        asyncio.run(run_downloads(markets))

    if cmd in ("normalize", "all"):
        run_normalize(asset, timeframe, duration_ms)
