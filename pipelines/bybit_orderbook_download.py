"""Download and process Bybit BTCUSDT orderbook data (200ms snapshots).

Downloads historical orderbook data from Bybit's public data portal,
reconstructs full book state from snapshot+delta streams, extracts top
N levels, and saves as daily Parquet files. Optionally applies
Savitzky-Golay smoothing to price columns.

Data source: https://quote-saver.bycsi.com (Bybit public historical data)
Format: JSON lines, snapshot + delta model, 200 levels, ~200ms updates

Usage:
    python -m pipelines.bybit_orderbook_download 2025-05-01 2026-03-15
    python -m pipelines.bybit_orderbook_download 2025-05-01 2026-03-15 --no-smooth
    python -m pipelines.bybit_orderbook_download --resume  # continue from last downloaded date
"""

import argparse
import gzip
import io
import json
import sys
import time
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

try:
    import requests
except ImportError:
    print("ERROR: requests not installed. Run: pip install requests")
    sys.exit(1)

try:
    from scipy.signal import savgol_filter
except ImportError:
    savgol_filter = None

# ── Constants ─────────────────────────────────────────────────────────

SYMBOL = "BTCUSDT"
CATEGORY = "linear"
N_LEVELS = 20        # extract top 20 of 200 available
SG_WINDOW = 21       # Savitzky-Golay window (21 × 200ms = 4.2s)
SG_POLY = 3          # cubic polynomial

# Bybit public data base URLs
# orderbook: https://quote-saver.bycsi.com/orderbook/linear/BTCUSDT/
# Format: {date}_{symbol}_ob500.data.zip (pre ~Aug 2025) or ob200.data.zip (post ~Aug 2025)
QUOTE_SAVER_BASE = (
    "https://quote-saver.bycsi.com/orderbook/{category}/{symbol}/"
)

MAX_RETRIES = 5
RATE_LIMIT_SLEEP = 0.2  # seconds between requests

# Parquet schema for output
def make_schema(n_levels: int) -> pa.Schema:
    fields = [pa.field("timestamp_ms", pa.int64())]
    for side in ("bid", "ask"):
        for i in range(1, n_levels + 1):
            fields.append(pa.field(f"{side}_price_{i}", pa.float64()))
            fields.append(pa.field(f"{side}_size_{i}", pa.float64()))
    return pa.schema(fields)


# ── Download ──────────────────────────────────────────────────────────

def download_to_file(url: str, dest: Path, max_retries: int = MAX_RETRIES) -> bool:
    """Stream download to file with exponential backoff. Returns True on success."""
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, timeout=600, stream=True)
            if resp.status_code == 404:
                return False
            resp.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1_048_576):
                    f.write(chunk)
            return True
        except (requests.RequestException, ValueError) as e:
            wait = 2 ** attempt
            if attempt < max_retries - 1:
                print(f"    Retry {attempt + 1}/{max_retries} after {wait}s: {e}")
                time.sleep(wait)
            else:
                print(f"    Failed after {max_retries} attempts: {e}")
                return False
    return False


def discover_file_url(date_str: str) -> str | None:
    """Try known URL patterns for a given date.

    Bybit quote-saver uses .data.zip files:
      - ob500 (pre ~Aug 2025): 500-level orderbook
      - ob200 (post ~Aug 2025): 200-level orderbook
    """
    base = QUOTE_SAVER_BASE.format(category=CATEGORY, symbol=SYMBOL)
    # Try ob200 first (newer, 200 levels), then ob500 (older, 500 levels)
    patterns = [
        f"{base}{date_str}_{SYMBOL}_ob200.data.zip",
        f"{base}{date_str}_{SYMBOL}_ob500.data.zip",
    ]

    for url in patterns:
        try:
            resp = requests.head(url, timeout=10, allow_redirects=True)
            if resp.status_code == 200:
                return url
        except requests.RequestException:
            continue

    return None


def list_available_dates(start_date: str, end_date: str) -> list[str]:
    """Generate date strings between start and end (exclusive)."""
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    dates = []
    d = start
    while d < end:
        dates.append(d.strftime("%Y-%m-%d"))
        d += timedelta(days=1)
    return dates


# ── Parsing ───────────────────────────────────────────────────────────

def _decompress(data: bytes) -> bytes:
    """Decompress bytes from zip, gzip, or return raw."""
    # Try zip first (Bybit quote-saver uses .data.zip)
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            names = zf.namelist()
            if names:
                return zf.read(names[0])
    except zipfile.BadZipFile:
        pass

    # Try gzip
    try:
        return gzip.decompress(data)
    except gzip.BadGzipFile:
        pass

    return data


def parse_csv_orderbook(data: bytes, n_levels: int = N_LEVELS) -> pd.DataFrame:
    """Parse Bybit orderbook dump into structured DataFrame.

    Handles multiple formats:
    - JSON lines (quote-saver .data.zip — most common)
    - CSV row-per-level (timestamp, price, size, side)
    - CSV snapshot (timestamp, bids, asks as JSON arrays)
    """
    raw = _decompress(data)

    # Peek at first bytes to detect format
    first_bytes = raw[:200].strip()
    if first_bytes.startswith(b"{"):
        # JSON lines format (quote-saver .data files)
        return _parse_json_lines(raw, n_levels)

    # Try parsing as CSV
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as e:
        print(f"    Failed to parse CSV: {e}")
        # Fall back to JSON lines in case header confused CSV parser
        return _parse_json_lines(raw, n_levels)

    if df.empty:
        return pd.DataFrame()

    # Normalize column names
    cols_lower = {c: c.lower().strip() for c in df.columns}
    df = df.rename(columns=cols_lower)

    # Detect format
    if "timestamp" in df.columns and "price" in df.columns and "size" in df.columns:
        return _parse_row_format(df, n_levels)
    elif "timestamp" in df.columns and "bids" in df.columns:
        return _parse_snapshot_format(df, n_levels)
    else:
        # Try JSON lines format
        return _parse_json_lines(raw, n_levels)


def _parse_row_format(
    df: pd.DataFrame, n_levels: int
) -> pd.DataFrame:
    """Parse row-per-level format: timestamp, price, size, side."""
    # Ensure numeric
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["size"] = pd.to_numeric(df["size"], errors="coerce")
    df = df.dropna(subset=["timestamp", "price", "size"])

    # Convert to milliseconds — detect unit by magnitude
    first_ts = df["timestamp"].iloc[0]
    if first_ts < 1e10:          # seconds (< 2286)
        df["timestamp_ms"] = (df["timestamp"] * 1000).astype(np.int64)
    elif first_ts < 1e13:        # milliseconds (< 2286)
        df["timestamp_ms"] = df["timestamp"].astype(np.int64)
    elif first_ts < 1e16:        # microseconds
        df["timestamp_ms"] = (df["timestamp"] / 1000).astype(np.int64)
    else:                        # nanoseconds
        df["timestamp_ms"] = (df["timestamp"] / 1e6).astype(np.int64)

    # Detect side column
    side_col = None
    for c in ("side", "direction", "type"):
        if c in df.columns:
            side_col = c
            break

    if side_col is None:
        print("    Warning: no side column found, inferring from order")
        return pd.DataFrame()

    # Vectorized pivot to wide format
    side_lower = df[side_col].astype(str).str.lower()
    df["_is_bid"] = side_lower.str.startswith("b")
    df["_is_ask"] = side_lower.str.startswith("a") | side_lower.str.startswith("s")

    frames = []
    for is_bid, ascending, prefix in [(True, False, "bid"), (False, True, "ask")]:
        col_filter = "_is_bid" if is_bid else "_is_ask"
        side_df = df[df[col_filter]].copy()
        if side_df.empty:
            continue
        # Deduplicate (same timestamp + price), keep last
        side_df = side_df.drop_duplicates(
            subset=["timestamp_ms", "price"], keep="last"
        )
        # Rank within each timestamp group
        side_df = side_df.sort_values(["timestamp_ms", "price"], ascending=[True, ascending])
        side_df["_rank"] = side_df.groupby("timestamp_ms").cumcount() + 1
        side_df = side_df[side_df["_rank"] <= n_levels]

        # Pivot price and size
        for value_col, suffix in [("price", "price"), ("size", "size")]:
            pivoted = side_df.pivot(
                index="timestamp_ms", columns="_rank", values=value_col
            )
            pivoted.columns = [f"{prefix}_{suffix}_{int(c)}" for c in pivoted.columns]
            frames.append(pivoted)

    if not frames:
        return pd.DataFrame()

    result = pd.concat(frames, axis=1).reset_index()

    # Fill missing levels with 0
    for side in ("bid", "ask"):
        for i in range(1, n_levels + 1):
            for suffix in ("price", "size"):
                col = f"{side}_{suffix}_{i}"
                if col not in result.columns:
                    result[col] = 0.0

    # Order columns
    cols = ["timestamp_ms"]
    for side in ("bid", "ask"):
        for i in range(1, n_levels + 1):
            cols.extend([f"{side}_price_{i}", f"{side}_size_{i}"])
    result = result[cols].sort_values("timestamp_ms").reset_index(drop=True)

    return result


def _parse_json_lines(raw: bytes, n_levels: int) -> pd.DataFrame:
    """Parse JSON-lines orderbook format (quote-saver style).

    Each line is a JSON object with snapshot or delta type.
    Snapshots contain full book; deltas contain changes.
    We reconstruct the book by applying deltas to last snapshot.
    """
    lines = raw.decode("utf-8", errors="ignore").strip().split("\n")
    if not lines:
        return pd.DataFrame()

    # Current book state
    bids = {}  # price → size
    asks = {}

    snapshots = []
    last_snapshot_ts = 0
    target_interval_ms = 200

    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue

        # Extract timestamp
        ts = obj.get("ts", obj.get("T", obj.get("timestamp", 0)))
        if isinstance(ts, str):
            ts = int(ts)
        if ts < 1e10:
            ts = int(ts * 1000)       # seconds → ms
        elif ts >= 1e16:
            ts = int(ts / 1e6)        # nanoseconds → ms
        elif ts >= 1e13:
            ts = int(ts / 1000)       # microseconds → ms
        else:
            ts = int(ts)              # already ms

        # Determine if snapshot or delta
        msg_type = obj.get("type", "").lower()

        if msg_type == "snapshot":
            # Full snapshot — reset book state
            bids.clear()
            asks.clear()

        # Bybit WebSocket format nests bid/ask under "data" key
        inner = obj.get("data", obj)

        # Apply bid/ask updates (works for both snapshot and delta)
        bid_data = inner.get("b", inner.get("bids", []))
        ask_data = inner.get("a", inner.get("asks", []))

        if bid_data or ask_data:
            for p, s in bid_data:
                p, s = float(p), float(s)
                if s > 0:
                    bids[p] = s
                else:
                    bids.pop(p, None)

            for p, s in ask_data:
                p, s = float(p), float(s)
                if s > 0:
                    asks[p] = s
                else:
                    asks.pop(p, None)

        # Emit snapshot at target interval
        if ts - last_snapshot_ts >= target_interval_ms and bids and asks:
            row = {"timestamp_ms": ts}
            sorted_bids = sorted(bids.items(), key=lambda x: -x[0])[:n_levels]
            sorted_asks = sorted(asks.items(), key=lambda x: x[0])[:n_levels]

            for i, (p, s) in enumerate(sorted_bids, 1):
                row[f"bid_price_{i}"] = p
                row[f"bid_size_{i}"] = s
            for i, (p, s) in enumerate(sorted_asks, 1):
                row[f"ask_price_{i}"] = p
                row[f"ask_size_{i}"] = s

            # Fill remaining levels
            for side, data in [("bid", sorted_bids), ("ask", sorted_asks)]:
                for i in range(len(data) + 1, n_levels + 1):
                    row[f"{side}_price_{i}"] = 0.0
                    row[f"{side}_size_{i}"] = 0.0

            snapshots.append(row)
            last_snapshot_ts = ts

    if not snapshots:
        return pd.DataFrame()

    result = pd.DataFrame(snapshots)
    cols = ["timestamp_ms"]
    for side in ("bid", "ask"):
        for i in range(1, n_levels + 1):
            cols.extend([f"{side}_price_{i}", f"{side}_size_{i}"])
    result = result[cols].sort_values("timestamp_ms").reset_index(drop=True)
    return result


def _parse_snapshot_format(
    df: pd.DataFrame, n_levels: int
) -> pd.DataFrame:
    """Parse wide snapshot format with bids/asks as JSON strings."""
    # Vectorized: parse JSON columns into lists, then extract levels
    ts = pd.to_numeric(df["timestamp"], errors="coerce")
    # Detect unit: <1e10 = seconds, <1e13 = ms, <1e16 = us, else ns
    first_ts = ts.iloc[0] if len(ts) > 0 else 0
    if first_ts < 1e10:
        ts = (ts * 1000).astype(np.int64)
    elif first_ts < 1e13:
        ts = ts.astype(np.int64)
    elif first_ts < 1e16:
        ts = (ts / 1000).astype(np.int64)
    else:
        ts = (ts / 1e6).astype(np.int64)

    def _safe_json(val):
        if isinstance(val, str):
            try:
                return json.loads(val)
            except (json.JSONDecodeError, TypeError):
                return []
        return val if val is not None else []

    bids_parsed = df["bids"].map(_safe_json)
    asks_parsed = df["asks"].map(_safe_json)

    # Build columns
    data = {"timestamp_ms": ts}
    for side, parsed in [("bid", bids_parsed), ("ask", asks_parsed)]:
        for i in range(1, n_levels + 1):
            idx = i - 1
            data[f"{side}_price_{i}"] = parsed.map(
                lambda x, j=idx: float(x[j][0]) if j < len(x) else 0.0
            )
            data[f"{side}_size_{i}"] = parsed.map(
                lambda x, j=idx: float(x[j][1]) if j < len(x) else 0.0
            )

    result = pd.DataFrame(data)

    cols = ["timestamp_ms"]
    for side in ("bid", "ask"):
        for i in range(1, n_levels + 1):
            cols.extend([f"{side}_price_{i}", f"{side}_size_{i}"])

    for c in cols:
        if c not in result.columns:
            result[c] = 0.0

    return result[cols].sort_values("timestamp_ms").reset_index(drop=True)


# ── SG Smoothing ──────────────────────────────────────────────────────

def apply_sg_to_prices(
    df: pd.DataFrame,
    n_levels: int = N_LEVELS,
    window: int = SG_WINDOW,
    poly: int = SG_POLY,
) -> pd.DataFrame:
    """Apply Savitzky-Golay smoothing to price columns (in-place copy)."""
    if savgol_filter is None:
        print("  Warning: scipy not available, skipping SG smoothing")
        return df

    result = df.copy()
    if len(result) < window:
        return result

    for side in ("bid", "ask"):
        for i in range(1, n_levels + 1):
            col = f"{side}_price_{i}"
            if col in result.columns:
                vals = result[col].values.copy()
                mask = vals > 0
                if mask.sum() < window:
                    continue
                # Find contiguous non-zero runs and smooth each independently
                # to avoid SG filter being corrupted by zero-padded boundaries
                changes = np.diff(mask.astype(np.int8))
                starts = np.where(changes == 1)[0] + 1
                ends = np.where(changes == -1)[0] + 1
                # Handle edge: starts at 0
                if mask[0]:
                    starts = np.concatenate([[0], starts])
                if mask[-1]:
                    ends = np.concatenate([ends, [len(vals)]])
                for s, e in zip(starts, ends):
                    run_len = e - s
                    if run_len >= window:
                        vals[s:e] = savgol_filter(vals[s:e], window, poly)
                result[col] = vals

    return result


# ── Main Pipeline ─────────────────────────────────────────────────────

def process_day(
    date_str: str,
    out_dir: Path,
    raw_dir: Path | None = None,
    smooth: bool = True,
    n_levels: int = N_LEVELS,
) -> bool:
    """Download and process one day of orderbook data.

    Streams download to temp file, extracts zip, parses, saves parquet.
    Returns True if successful.
    """
    import tempfile

    out_path = out_dir / f"{date_str}.parquet"
    if out_path.exists():
        return True  # already done

    print(f"  {date_str}: discovering URL...")
    url = discover_file_url(date_str)
    if url is None:
        print(f"  {date_str}: no data available")
        return False

    # Stream download to temp file (files are 250-400 MB)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_zip = Path(tmpdir) / "download.zip"
        print(f"  {date_str}: downloading...")
        if not download_to_file(url, tmp_zip):
            return False

        size_mb = tmp_zip.stat().st_size / 1e6
        print(f"  {date_str}: downloaded {size_mb:.1f} MB, extracting...")

        # Extract and read the data file from the zip
        try:
            raw = _decompress(tmp_zip.read_bytes())
        except Exception as e:
            print(f"  {date_str}: decompression failed: {e}")
            return False

    print(f"  {date_str}: parsing {len(raw)/1e6:.0f} MB...")

    # Detect format and parse
    first_bytes = raw[:200].strip()
    if first_bytes.startswith(b"{"):
        df = _parse_json_lines(raw, n_levels)
    else:
        df = parse_csv_orderbook(raw, n_levels)

    if df.empty:
        print(f"  {date_str}: no valid snapshots parsed")
        return False

    n_snapshots = len(df)
    print(f"  {date_str}: {n_snapshots:,} snapshots")

    # Save raw (unsmoothed)
    if raw_dir is not None:
        raw_dir.mkdir(parents=True, exist_ok=True)
        raw_path = raw_dir / f"{date_str}.parquet"
        if not raw_path.exists():
            table = pa.Table.from_pandas(df, preserve_index=False)
            pq.write_table(table, raw_path, compression="snappy")

    # Apply SG smoothing
    if smooth:
        df = apply_sg_to_prices(df, n_levels)

    # Save
    out_dir.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, out_path, compression="snappy")

    file_mb = out_path.stat().st_size / 1e6
    print(f"  {date_str}: saved {file_mb:.1f} MB ({n_snapshots:,} snapshots)")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Download Bybit BTCUSDT orderbook data (200ms snapshots)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "start_date", nargs="?", help="Start date YYYY-MM-DD (default: 2025-05-01)",
        default="2025-05-01",
    )
    parser.add_argument(
        "end_date", nargs="?", help="End date YYYY-MM-DD exclusive (default: today)",
        default=None,
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("data/bybit_orderbook"),
        help="Output directory for smoothed parquet files",
    )
    parser.add_argument(
        "--raw-dir", type=Path, default=Path("data/bybit_orderbook_raw"),
        help="Output directory for raw (unsmoothed) parquet files",
    )
    parser.add_argument(
        "--no-smooth", action="store_true",
        help="Skip Savitzky-Golay smoothing",
    )
    parser.add_argument(
        "--no-raw", action="store_true",
        help="Don't save raw (unsmoothed) copies",
    )
    parser.add_argument(
        "--levels", type=int, default=N_LEVELS,
        help=f"Number of price levels to extract (default: {N_LEVELS})",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from last downloaded date",
    )
    args = parser.parse_args()

    if args.end_date is None:
        args.end_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    out_dir = args.out_dir
    raw_dir = None if args.no_raw else args.raw_dir

    # Resume: find latest existing file
    if args.resume and out_dir.exists():
        existing = sorted(out_dir.glob("*.parquet"))
        if existing:
            last = existing[-1].stem
            # Start from day after last
            last_dt = datetime.strptime(last, "%Y-%m-%d") + timedelta(days=1)
            args.start_date = last_dt.strftime("%Y-%m-%d")
            print(f"Resuming from {args.start_date} (last: {last})")

    dates = list_available_dates(args.start_date, args.end_date)
    print(f"Downloading {SYMBOL} orderbook: {args.start_date} → {args.end_date}")
    print(f"  {len(dates)} days, {args.levels} levels, "
          f"smooth={'no' if args.no_smooth else 'yes'}")
    print(f"  Output: {out_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    success = 0
    failed = 0
    for i, date_str in enumerate(dates):
        ok = process_day(
            date_str, out_dir, raw_dir,
            smooth=not args.no_smooth,
            n_levels=args.levels,
        )
        if ok:
            success += 1
        else:
            failed += 1

        time.sleep(RATE_LIMIT_SLEEP)

        # Progress every 10 days
        if (i + 1) % 10 == 0 or i == len(dates) - 1:
            pct = (i + 1) / len(dates) * 100
            print(f"\nProgress: {i + 1}/{len(dates)} ({pct:.0f}%) — "
                  f"{success} ok, {failed} failed\n")

    print(f"\n{'='*60}")
    print(f"Done: {success} days downloaded, {failed} failed")
    if out_dir.exists():
        total_files = len(list(out_dir.glob("*.parquet")))
        total_mb = sum(f.stat().st_size for f in out_dir.glob("*.parquet")) / 1e6
        print(f"Total: {total_files} files, {total_mb:.0f} MB")


if __name__ == "__main__":
    main()
