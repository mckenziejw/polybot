"""
preprocess_markets.py — Resample Telonex book snapshots to 100ms bars.

Reads raw tick-level parquet files from `src_dir`, resamples each token
(Up/Down) to a fixed 100ms grid, computes staleness_ms, and writes a
compact parquet to `dst_dir`.

Key behaviours:
  - Fetches resolutions via ResolutionStore.fetch_batch() before processing.
    Only markets with confirmed resolutions are written. This also updates
    the local resolution cache with any newly resolved markets.
  - Skips output files that already exist (resumable).
  - Drops columns not needed by the environment: received_timestamp,
    slug, market. These add ~15% file size and are unused at training time.
  - Output schema per row:
      exchange_timestamp  int64   (100ms bar timestamp, uniform grid)
      token_label         str     ("Up" or "Down")
      asset_id            str
      mid_price           float64
      spread              float64
      book_imbalance      float64
      bid_price_{1..5}    float64
      bid_size_{1..5}     float64
      ask_price_{1..5}    float64
      ask_size_{1..5}     float64
      staleness_ms        float64 (ms since last real tick at this bar)

  Output files contain exactly 2 × 3,000 rows (Yes + No × 3,000 bars).
  Files with fewer rows (sparse/broken markets) are skipped with a warning.

Usage:
    python preprocess_markets.py [--src ...] [--dst ...] [--cache ...] [--workers N]

Defaults:
    --src     data/telonex_book_snapshots
    --dst     data/telonex_100ms
    --cache   data/resolutions.json
    --workers 4   (parallel file processing; resolution fetch is serial)
"""

import argparse
import logging
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd

from market_analysis import ResolutionStore

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MARKET_DURATION_MS = 900_000   # 5 minutes
RESAMPLE_MS        = 100       # target bar width
EXPECTED_BARS      = MARKET_DURATION_MS // RESAMPLE_MS  # 3,000
MAX_STALENESS_MS   = 5_000

KEEP_COLUMNS = [
    "exchange_timestamp",
    "token_label",
    "asset_id",
    "mid_price",
    "spread",
    "book_imbalance",
    "bid_price_1", "bid_size_1",
    "bid_price_2", "bid_size_2",
    "bid_price_3", "bid_size_3",
    "bid_price_4", "bid_size_4",
    "bid_price_5", "bid_size_5",
    "ask_price_1", "ask_size_1",
    "ask_price_2", "ask_size_2",
    "ask_price_3", "ask_size_3",
    "ask_price_4", "ask_size_4",
    "ask_price_5", "ask_size_5",
]

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Resampling (mirrors polymarket_env._resample_book)
# ---------------------------------------------------------------------------

def _resample_token(df: pd.DataFrame, open_ms: int, close_ms: int) -> pd.DataFrame:
    """
    Resample one token's tick data to a uniform RESAMPLE_MS grid.

    Returns a DataFrame with exchange_timestamp as a regular column (not index),
    containing KEEP_COLUMNS (minus token_label) plus staleness_ms.
    """
    grid = np.arange(open_ms, close_ms, RESAMPLE_MS)

    data_cols = [c for c in KEEP_COLUMNS if c not in ("exchange_timestamp", "token_label")]

    if df.empty:
        result = pd.DataFrame({"exchange_timestamp": grid})
        for c in data_cols:
            result[c] = np.nan
        result["staleness_ms"] = float(MAX_STALENESS_MS)
        return result

    df = df.set_index("exchange_timestamp").sort_index()
    df = df[[c for c in data_cols if c in df.columns]]

    # Track tick timestamp for staleness computation after forward-fill
    df["last_tick_ts"] = df.index.astype("int64")

    # Assign each tick to its 100ms bar and take last value per bar
    bar_index = (df.index // RESAMPLE_MS) * RESAMPLE_MS
    resampled = df.groupby(bar_index).last()
    resampled.index.name = "exchange_timestamp"  # name explicitly before reindex

    # Reindex to full uniform grid, forward-fill then backward-fill gaps
    resampled = resampled.reindex(grid).ffill().bfill()
    resampled.index.name = "exchange_timestamp"

    # staleness_ms: ms since last real tick at this bar, clipped to >= 0
    resampled["staleness_ms"] = (resampled.index - resampled["last_tick_ts"]).clip(lower=0)
    resampled = resampled.drop(columns=["last_tick_ts"])

    # Return with exchange_timestamp as a regular column, not the index
    return resampled.reset_index()


# ---------------------------------------------------------------------------
# Per-file processing
# ---------------------------------------------------------------------------

def _parse_open_ms(path: Path) -> int | None:
    """Extract market open timestamp from filename slug."""
    try:
        return int(path.stem.split("-")[-1]) * 1000
    except (ValueError, IndexError):
        return None


def process_file(args: tuple) -> tuple[str, str]:
    """
    Process a single market file. Returns (slug, status) where status is
    one of: "written", "skipped_exists", "skipped_no_resolution",
    "skipped_sparse", "error".

    Designed to run in a worker process — takes a plain tuple so it's
    picklable without importing the resolution dict into each worker.
    Instead, resolution outcomes are passed in directly.
    """
    src_path, dst_path, yes_resolved, no_resolved = args
    slug = src_path.stem

    if dst_path.exists():
        return slug, "skipped_exists"

    try:
        open_ms  = _parse_open_ms(src_path)
        close_ms = open_ms + MARKET_DURATION_MS

        # Load only the columns we need — parquet skips the rest at read time
        df = pd.read_parquet(src_path, columns=[c for c in KEEP_COLUMNS if c != "staleness_ms"])

        yes_raw = df[df["token_label"] == "Up"].copy()
        no_raw  = df[df["token_label"] == "Down"].copy()

        yes_resampled = _resample_token(yes_raw, open_ms, close_ms)
        no_resampled  = _resample_token(no_raw,  open_ms, close_ms)

        if len(yes_resampled) < EXPECTED_BARS or len(no_resampled) < EXPECTED_BARS:
            log.warning(
                f"{slug}: sparse data — yes={len(yes_resampled)} "
                f"no={len(no_resampled)} bars (expected {EXPECTED_BARS}), skipping"
            )
            return slug, "skipped_sparse"

        # Add token_label and resolved_value columns
        # exchange_timestamp is already a regular column from _resample_token
        yes_resampled["token_label"]    = "Up"
        yes_resampled["resolved_value"] = yes_resolved
        no_resampled["token_label"]     = "Down"
        no_resampled["resolved_value"]  = no_resolved

        yes_out = yes_resampled
        no_out  = no_resampled

        out = pd.concat([yes_out, no_out], ignore_index=True)

        # Ensure column order matches KEEP_COLUMNS + staleness_ms + resolved_value
        final_cols = KEEP_COLUMNS + ["staleness_ms", "resolved_value"]
        out = out[[c for c in final_cols if c in out.columns]]

        dst_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(dst_path, index=False)

        return slug, "written"

    except Exception as e:
        log.error(f"{slug}: error — {e}")
        return slug, f"error: {e}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src",     default="data/telonex_book_snapshots",
                        help="Source directory of raw Telonex parquet files")
    parser.add_argument("--dst",     default="data/telonex_100ms",
                        help="Destination directory for resampled parquet files")
    parser.add_argument("--cache",   default="data/resolutions.json",
                        help="Path to ResolutionStore JSON cache")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel workers for file processing (default: 4)")
    args = parser.parse_args()

    src_dir = Path(args.src)
    dst_dir = Path(args.dst)
    dst_dir.mkdir(parents=True, exist_ok=True)

    src_files = sorted(src_dir.glob("*.parquet"))
    if not src_files:
        log.error(f"No parquet files found in {src_dir}")
        return

    log.info(f"Found {len(src_files)} source files in {src_dir}")

    # ------------------------------------------------------------------
    # Step 1: Resolve all markets (cache-first, API for new ones)
    # ------------------------------------------------------------------
    log.info("Fetching resolutions (cache-first, API for uncached)...")
    store = ResolutionStore(cache_path=Path(args.cache))
    slugs = [f.stem for f in src_files]
    resolutions = store.fetch_batch(slugs)
    log.info(f"Resolved {len(resolutions)}/{len(src_files)} markets")

    # ------------------------------------------------------------------
    # Step 2: Build work list — skip already-done and unresolved
    # ------------------------------------------------------------------
    work = []
    skipped_no_res = 0

    for src_path in src_files:
        slug = src_path.stem
        dst_path = dst_dir / src_path.name

        if dst_path.exists():
            # Count as skipped without adding to work queue
            continue

        resolution = resolutions.get(slug)
        if resolution is None:
            skipped_no_res += 1
            continue

        # Extract resolution values for each token — pass directly to worker
        # so workers don't need to re-instantiate ResolutionStore
        yes_asset_id = None
        no_asset_id  = None

        # We need asset_ids to look up token outcomes. Read minimally.
        try:
            ids_df = pd.read_parquet(src_path, columns=["token_label", "asset_id"])
            yes_rows = ids_df[ids_df["token_label"] == "Up"]["asset_id"]
            no_rows  = ids_df[ids_df["token_label"] == "Down"]["asset_id"]
            yes_asset_id = yes_rows.iloc[0] if not yes_rows.empty else None
            no_asset_id  = no_rows.iloc[0]  if not no_rows.empty  else None
        except Exception as e:
            log.warning(f"{slug}: could not read asset_ids — {e}, skipping")
            continue

        if yes_asset_id not in resolution.token_outcomes or \
           no_asset_id  not in resolution.token_outcomes:
            log.warning(f"{slug}: asset_id mismatch in resolution, skipping")
            skipped_no_res += 1
            continue

        yes_resolved = resolution.token_outcomes[yes_asset_id]
        no_resolved  = resolution.token_outcomes[no_asset_id]

        work.append((src_path, dst_path, yes_resolved, no_resolved))

    already_done = len(src_files) - len(work) - skipped_no_res
    log.info(
        f"Work queue: {len(work)} to process, "
        f"{already_done} already done, "
        f"{skipped_no_res} skipped (no resolution)"
    )

    if not work:
        log.info("Nothing to do.")
        return

    # ------------------------------------------------------------------
    # Step 3: Process files in parallel
    # ------------------------------------------------------------------
    n_workers = min(args.workers, len(work))
    log.info(f"Processing with {n_workers} workers...")

    counts = {"written": 0, "skipped_exists": 0, "skipped_sparse": 0, "error": 0}

    with mp.Pool(processes=n_workers) as pool:
        for i, (slug, status) in enumerate(pool.imap_unordered(process_file, work), 1):
            key = status if status in counts else "error"
            counts[key] += 1
            if i % 100 == 0 or i == len(work):
                log.info(
                    f"  {i}/{len(work)} — "
                    f"written={counts['written']} "
                    f"sparse={counts['skipped_sparse']} "
                    f"errors={counts['error']}"
                )
            if key == "error":
                log.error(f"  {slug}: {status}")

    log.info(
        f"\nDone. written={counts['written']}, "
        f"already_existed={already_done + counts['skipped_exists']}, "
        f"no_resolution={skipped_no_res}, "
        f"sparse={counts['skipped_sparse']}, "
        f"errors={counts['error']}"
    )


if __name__ == "__main__":
    main()