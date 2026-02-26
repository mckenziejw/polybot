"""
Telonex BTC 5m market downloader and normalizer.

Two steps:
  1. download  — fetch raw parquets from Telonex into ./datasets/telonex_raw/
  2. normalize — convert raw files to canonical schema, write to data/telonex_book_snapshots/

Run:
  python telonex_pipeline.py download
  python telonex_pipeline.py normalize
  python telonex_pipeline.py all
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from telonex import download_async

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_KEY     = os.environ["TELONEX_API_KEY"]
RAW_DIR     = Path("./datasets/telonex_raw")
OUT_DIR     = Path("./data/telonex_book_snapshots")
MARKETS_URL = "https://api.telonex.io/v1/datasets/polymarket/markets"
CONCURRENCY = 10
N_LEVELS    = 5  # book_snapshot_5

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Canonical schema — 5-level version
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
    pa.field("bid_price_1",        pa.float64()),
    pa.field("bid_size_1",         pa.float64()),
    pa.field("bid_price_2",        pa.float64()),
    pa.field("bid_size_2",         pa.float64()),
    pa.field("bid_price_3",        pa.float64()),
    pa.field("bid_size_3",         pa.float64()),
    pa.field("bid_price_4",        pa.float64()),
    pa.field("bid_size_4",         pa.float64()),
    pa.field("bid_price_5",        pa.float64()),
    pa.field("bid_size_5",         pa.float64()),
    pa.field("ask_price_1",        pa.float64()),
    pa.field("ask_size_1",         pa.float64()),
    pa.field("ask_price_2",        pa.float64()),
    pa.field("ask_size_2",         pa.float64()),
    pa.field("ask_price_3",        pa.float64()),
    pa.field("ask_size_3",         pa.float64()),
    pa.field("ask_price_4",        pa.float64()),
    pa.field("ask_size_4",         pa.float64()),
    pa.field("ask_price_5",        pa.float64()),
    pa.field("ask_size_5",         pa.float64()),
])

# ---------------------------------------------------------------------------
# Step 1: Download
# ---------------------------------------------------------------------------

def load_btc_markets() -> pd.DataFrame:
    log.info("Loading markets dataset...")
    df = pd.read_parquet(MARKETS_URL)
    btc = df[
        df["slug"].str.contains("btc-updown-5m", na=False) &
        (df["book_snapshot_5_from"] != "")
    ].copy()
    log.info(f"Found {len(btc)} BTC 5m markets with book_snapshot_5 data")
    return btc


async def download_one(
    slug: str,
    asset_id: str,
    from_date: str,
    to_date: str,
    semaphore: asyncio.Semaphore,
) -> list[str]:
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
# Step 2: Normalize
# ---------------------------------------------------------------------------

def normalize_raw_file(path: Path, market_open_ms: int, market_close_ms: int) -> pd.DataFrame | None:
    """
    Vectorized conversion of one raw Telonex file to canonical schema.
    Filters to [market_open_ms, market_close_ms) before any processing.
    """
    try:
        df = pd.read_parquet(path)
    except Exception as e:
        log.error(f"Failed to read {path}: {e}")
        return None

    if df.empty:
        return None

    # Filter to market window before any processing — drops pre/post market data
    open_us  = market_open_ms  * 1000
    close_us = market_close_ms * 1000
    df = df[(df["timestamp_us"] >= open_us) & (df["timestamp_us"] < close_us)]

    if df.empty:
        return None

    df = df.drop_duplicates(subset=["timestamp_us", "asset_id"])

    # Cast all price/size columns in one pass
    for side in ("bid", "ask"):
        for i in range(N_LEVELS):
            for field in ("price", "size"):
                col = f"{side}_{field}_{i}"
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # Telonex bid levels are left-packed ascending with zero-padding on the right:
    #   [worst_bid, ..., best_bid, 0, 0]
    # We need descending with zero-padding on the right:
    #   [best_bid, ..., worst_bid, 0, 0]
    # Strategy: stack into a 2D array, sort each row descending, unpack back.
    import numpy as np

    bid_price_arr = np.column_stack([df[f"bid_price_{i}"].values for i in range(N_LEVELS)])
    bid_size_arr  = np.column_stack([df[f"bid_size_{i}"].values  for i in range(N_LEVELS)])

    # Sort by price descending per row — argsort on negated prices
    bid_sort_idx  = np.argsort(-bid_price_arr, axis=1)
    bid_price_arr = np.take_along_axis(bid_price_arr, bid_sort_idx, axis=1)
    bid_size_arr  = np.take_along_axis(bid_size_arr,  bid_sort_idx, axis=1)

    # Asks are already ascending (best-first), just stack for consistency
    ask_price_arr = np.column_stack([df[f"ask_price_{i}"].values for i in range(N_LEVELS)])
    ask_size_arr  = np.column_stack([df[f"ask_size_{i}"].values  for i in range(N_LEVELS)])

    best_bid  = pd.Series(bid_price_arr[:, 0], index=df.index)
    best_ask  = pd.Series(ask_price_arr[:, 0], index=df.index)
    mid_price = (best_bid + best_ask) / 2
    spread    = best_ask - best_bid

    total_bid = bid_size_arr.sum(axis=1)
    total_ask = ask_size_arr.sum(axis=1)
    denom     = total_bid + total_ask
    imbalance = np.where(denom > 0, (total_bid - total_ask) / denom, 0.0)

    out = pd.DataFrame({
        "exchange_timestamp": df["timestamp_us"] // 1000,
        "received_timestamp": 0,
        "slug":               df["slug"],
        "market":             df["market_id"],
        "asset_id":           df["asset_id"],
        "token_label":        df["outcome"],
        "mid_price":          mid_price.values,
        "spread":             spread.values,
        "book_imbalance":     imbalance,
        **{f"bid_price_{i+1}": bid_price_arr[:, i] for i in range(N_LEVELS)},
        **{f"bid_size_{i+1}":  bid_size_arr[:, i]  for i in range(N_LEVELS)},
        **{f"ask_price_{i+1}": ask_price_arr[:, i]  for i in range(N_LEVELS)},
        **{f"ask_size_{i+1}":  ask_size_arr[:, i]   for i in range(N_LEVELS)},
    })

    return out


def build_asset_slug_map() -> dict[str, str]:
    """Load markets dataset and return {asset_id: slug} for all BTC 5m markets."""
    log.info("Building asset_id -> slug lookup...")
    df = pd.read_parquet(MARKETS_URL)
    btc = df[df["slug"].str.contains("btc-updown-5m", na=False)]
    mapping = {}
    for _, row in btc.iterrows():
        mapping[row["asset_id_0"]] = row["slug"]
        mapping[row["asset_id_1"]] = row["slug"]
    log.info(f"Loaded {len(mapping)} asset_id -> slug mappings")
    return mapping


def run_normalize():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_files = list(RAW_DIR.glob("*.parquet"))
    if not raw_files:
        log.error(f"No raw files found in {RAW_DIR}")
        return

    asset_slug = build_asset_slug_map()

    # Group files by slug using asset_id extracted from filename
    # Filename: polymarket_book_snapshot_5_2026-02-12_<asset_id>.parquet
    slug_files: dict[str, list[Path]] = {}
    unmapped = 0
    for f in raw_files:
        parts = f.stem.split("_")
        try:
            date_idx = next(i for i, p in enumerate(parts) if p.startswith("202"))
            asset_id = parts[date_idx + 1]
        except StopIteration:
            log.warning(f"Could not parse asset_id from: {f.name}")
            continue
        slug = asset_slug.get(asset_id)
        if not slug:
            log.warning(f"No slug for asset_id {asset_id[:20]}...")
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

        # Extract open timestamp from slug: btc-updown-5m-{ts}
        try:
            market_open_ms  = int(slug.split("-")[-1]) * 1000
            market_close_ms = market_open_ms + 300_000  # 5 minutes
        except (ValueError, IndexError):
            log.warning(f"Could not parse timestamp from slug: {slug}")
            errors += 1
            continue

        dfs = [n for f in files if (n := normalize_raw_file(f, market_open_ms, market_close_ms)) is not None]
        if not dfs:
            log.warning(f"No data for {slug}")
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
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"

    if cmd in ("download", "all"):
        markets = load_btc_markets()
        asyncio.run(run_downloads(markets))

    if cmd in ("normalize", "all"):
        run_normalize()