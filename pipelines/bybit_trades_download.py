"""Download and aggregate Bybit BTCUSDT tick trades into 200ms bins.

Downloads tick-level trade data from Bybit's public archive, aggregates
into 200ms bins aligned with orderbook snapshots, and saves as daily
Parquet files.

Source: https://public.bybit.com/trading/BTCUSDT/
Format: CSV.gz, tick-level with exact buy/sell side

Usage:
    python -m pipelines.bybit_trades_download 2025-05-01 2026-03-15
    python -m pipelines.bybit_trades_download --resume
"""

import argparse
import gzip
import io
import sys
import time
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

# ── Constants ─────────────────────────────────────────────────────────

SYMBOL = "BTCUSDT"
BASE_URL = "https://public.bybit.com/trading/{symbol}/"
BIN_MS = 200  # 200ms aggregation bins

MAX_RETRIES = 5
RATE_LIMIT_SLEEP = 0.2

# Output schema
OUTPUT_SCHEMA = pa.schema([
    pa.field("timestamp_ms", pa.int64()),
    pa.field("buy_volume", pa.float64()),
    pa.field("sell_volume", pa.float64()),
    pa.field("n_trades", pa.int32()),
    pa.field("n_buy_trades", pa.int32()),
    pa.field("n_sell_trades", pa.int32()),
    pa.field("vwap", pa.float64()),
    pa.field("max_trade_size", pa.float64()),
])


# ── Download ──────────────────────────────────────────────────────────

def download_with_retry(url: str, max_retries: int = MAX_RETRIES) -> bytes | None:
    """Download URL with exponential backoff."""
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, timeout=60)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return resp.content
        except (requests.RequestException, ValueError) as e:
            wait = 2 ** attempt
            if attempt < max_retries - 1:
                print(f"    Retry {attempt + 1}/{max_retries} after {wait}s: {e}")
                time.sleep(wait)
            else:
                print(f"    Failed after {max_retries} attempts: {e}")
                return None
    return None


def discover_trade_url(date_str: str) -> str | None:
    """Find the download URL for a given date's trade data."""
    patterns = [
        f"https://public.bybit.com/trading/{SYMBOL}/{SYMBOL}{date_str}.csv.gz",
        f"https://public.bybit.com/trading/{SYMBOL}/{SYMBOL}_{date_str}.csv.gz",
    ]
    for url in patterns:
        try:
            resp = requests.head(url, timeout=10)
            if resp.status_code == 200:
                return url
        except requests.RequestException:
            continue
    return None


# ── Parsing & Aggregation ─────────────────────────────────────────────

def parse_and_aggregate(data: bytes) -> pd.DataFrame:
    """Parse tick trades CSV and aggregate into 200ms bins.

    Bybit trade CSV format:
    timestamp, symbol, side, size, price, tickDirection, trdMatchID, grossValue, homeNotional, foreignNotional
    """
    try:
        raw = gzip.decompress(data)
    except gzip.BadGzipFile:
        raw = data

    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as e:
        print(f"    Failed to parse CSV: {e}")
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    # Normalize columns
    cols_lower = {c: c.lower().strip() for c in df.columns}
    df = df.rename(columns=cols_lower)

    # Find timestamp column
    ts_col = None
    for c in ("timestamp", "time", "ts"):
        if c in df.columns:
            ts_col = c
            break
    if ts_col is None:
        print("    No timestamp column found")
        return pd.DataFrame()

    df["ts"] = pd.to_numeric(df[ts_col], errors="coerce")
    df = df.dropna(subset=["ts"])

    # Convert to ms — detect unit by magnitude
    first_ts = df["ts"].iloc[0]
    if first_ts < 1e10:          # seconds
        df["timestamp_ms"] = (df["ts"] * 1000).astype(np.int64)
    elif first_ts < 1e13:        # milliseconds
        df["timestamp_ms"] = df["ts"].astype(np.int64)
    elif first_ts < 1e16:        # microseconds
        df["timestamp_ms"] = (df["ts"] / 1000).astype(np.int64)
    else:                        # nanoseconds
        df["timestamp_ms"] = (df["ts"] / 1e6).astype(np.int64)

    # Parse side
    side_col = None
    for c in ("side", "direction"):
        if c in df.columns:
            side_col = c
            break

    if side_col is None:
        print("    No side column found")
        return pd.DataFrame()

    # Coerce side to string to handle NaN/numeric values safely
    df["is_buy"] = df[side_col].astype(str).str.lower().str.startswith("b")

    # Parse size and price — drop rows with unparseable values (not fillna(0))
    for c in ("size", "qty", "amount"):
        if c in df.columns:
            df["trade_size"] = pd.to_numeric(df[c], errors="coerce")
            break
    else:
        df["trade_size"] = np.nan

    for c in ("price", "px"):
        if c in df.columns:
            df["trade_price"] = pd.to_numeric(df[c], errors="coerce")
            break
    else:
        df["trade_price"] = np.nan

    # Drop rows with missing/zero price or size (unparseable data)
    df = df.dropna(subset=["trade_size", "trade_price"])
    df = df[(df["trade_size"] > 0) & (df["trade_price"] > 0)]

    # Notional value
    df["notional"] = df["trade_size"] * df["trade_price"]

    # Bin to 200ms
    df["bin"] = (df["timestamp_ms"] // BIN_MS) * BIN_MS

    # Aggregate per bin
    buy_mask = df["is_buy"]
    df["buy_vol"] = np.where(buy_mask, df["trade_size"], 0.0)
    df["sell_vol"] = np.where(~buy_mask, df["trade_size"], 0.0)
    df["buy_count"] = buy_mask.astype(np.int32)
    df["sell_count"] = (~buy_mask).astype(np.int32)

    agg = df.groupby("bin").agg(
        buy_volume=("buy_vol", "sum"),
        sell_volume=("sell_vol", "sum"),
        n_trades=("trade_size", "count"),
        n_buy_trades=("buy_count", "sum"),
        n_sell_trades=("sell_count", "sum"),
        total_notional=("notional", "sum"),
        total_size=("trade_size", "sum"),
        max_trade_size=("trade_size", "max"),
    ).reset_index()

    agg["timestamp_ms"] = agg["bin"].astype(np.int64)
    agg["vwap"] = np.where(
        agg["total_size"] > 0,
        agg["total_notional"] / agg["total_size"],
        0.0,
    )

    result = agg[
        ["timestamp_ms", "buy_volume", "sell_volume", "n_trades",
         "n_buy_trades", "n_sell_trades", "vwap", "max_trade_size"]
    ].sort_values("timestamp_ms").reset_index(drop=True)

    # Cast types
    result["n_trades"] = result["n_trades"].astype(np.int32)
    result["n_buy_trades"] = result["n_buy_trades"].astype(np.int32)
    result["n_sell_trades"] = result["n_sell_trades"].astype(np.int32)

    return result


# ── Main Pipeline ─────────────────────────────────────────────────────

def process_day(date_str: str, out_dir: Path) -> bool:
    """Download and process one day of trade data. Returns True if ok."""
    out_path = out_dir / f"{date_str}.parquet"
    if out_path.exists():
        return True

    print(f"  {date_str}: discovering URL...")
    url = discover_trade_url(date_str)
    if url is None:
        print(f"  {date_str}: no data available")
        return False

    print(f"  {date_str}: downloading...")
    data = download_with_retry(url)
    if data is None:
        return False

    size_mb = len(data) / 1e6
    print(f"  {date_str}: downloaded {size_mb:.1f} MB, parsing...")

    df = parse_and_aggregate(data)
    if df.empty:
        print(f"  {date_str}: no valid trades parsed")
        return False

    n_bins = len(df)
    n_trades = df["n_trades"].sum()
    print(f"  {date_str}: {n_trades:,} trades → {n_bins:,} bins")

    out_dir.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, schema=OUTPUT_SCHEMA, preserve_index=False)
    pq.write_table(table, out_path, compression="snappy")

    file_mb = out_path.stat().st_size / 1e6
    print(f"  {date_str}: saved {file_mb:.2f} MB")

    return True


def list_dates(start: str, end: str) -> list[str]:
    """Generate date strings start → end (exclusive)."""
    s = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end, "%Y-%m-%d")
    dates = []
    d = s
    while d < e:
        dates.append(d.strftime("%Y-%m-%d"))
        d += timedelta(days=1)
    return dates


def main():
    parser = argparse.ArgumentParser(
        description="Download Bybit BTCUSDT tick trades → 200ms bins",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "start_date", nargs="?", default="2025-05-01",
        help="Start date YYYY-MM-DD (default: 2025-05-01)",
    )
    parser.add_argument(
        "end_date", nargs="?", default=None,
        help="End date YYYY-MM-DD exclusive (default: today)",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("data/bybit_trades_200ms"),
        help="Output directory",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from last downloaded date",
    )
    args = parser.parse_args()

    if args.end_date is None:
        args.end_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    out_dir = args.out_dir

    if args.resume and out_dir.exists():
        existing = sorted(out_dir.glob("*.parquet"))
        if existing:
            last = existing[-1].stem
            last_dt = datetime.strptime(last, "%Y-%m-%d") + timedelta(days=1)
            args.start_date = last_dt.strftime("%Y-%m-%d")
            print(f"Resuming from {args.start_date} (last: {last})")

    dates = list_dates(args.start_date, args.end_date)
    print(f"Downloading {SYMBOL} trades: {args.start_date} → {args.end_date}")
    print(f"  {len(dates)} days → 200ms bins")
    print(f"  Output: {out_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    success = 0
    failed = 0
    for i, date_str in enumerate(dates):
        ok = process_day(date_str, out_dir)
        if ok:
            success += 1
        else:
            failed += 1

        time.sleep(RATE_LIMIT_SLEEP)

        if (i + 1) % 10 == 0 or i == len(dates) - 1:
            pct = (i + 1) / len(dates) * 100
            print(f"\nProgress: {i + 1}/{len(dates)} ({pct:.0f}%) — "
                  f"{success} ok, {failed} failed\n")

    print(f"\n{'='*60}")
    print(f"Done: {success} days downloaded, {failed} failed")
    if out_dir.exists():
        total_files = len(list(out_dir.glob("*.parquet")))
        total_mb = sum(f.stat().st_size for f in out_dir.glob("*.parquet")) / 1e6
        print(f"Total: {total_files} files, {total_mb:.1f} MB")


if __name__ == "__main__":
    main()
