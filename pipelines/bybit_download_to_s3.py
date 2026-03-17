"""Download Bybit data and upload directly to S3.

Runs on an EC2 instance to avoid local bandwidth constraints.
Downloads orderbook + trades from Bybit, processes to parquet,
uploads to S3, then deletes local temp files.

Usage (on EC2):
    python -m pipelines.bybit_download_to_s3 2025-05-01 2026-03-15
    python -m pipelines.bybit_download_to_s3 --resume          # skip dates already in S3
    python -m pipelines.bybit_download_to_s3 --trades-only     # just trades
    python -m pipelines.bybit_download_to_s3 --orderbook-only  # just orderbook
"""

import argparse
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Re-use existing download/parse logic
from pipelines.bybit_orderbook_download import (
    discover_file_url,
    download_to_file,
    _decompress,
    _parse_json_lines,
    parse_csv_orderbook,
    apply_sg_to_prices,
    N_LEVELS,
)
from pipelines.bybit_trades_download import (
    discover_trade_url,
    download_with_retry,
    parse_and_aggregate,
    OUTPUT_SCHEMA,
)

import pyarrow as pa
import pyarrow.parquet as pq

DEFAULT_BUCKET = "polybot-swarm-data"
RATE_LIMIT_SLEEP = 0.2


def s3_key_exists(bucket: str, key: str) -> bool:
    """Check if an S3 key exists using AWS CLI."""
    result = subprocess.run(
        ["aws", "s3api", "head-object", "--bucket", bucket, "--key", key],
        capture_output=True,
    )
    return result.returncode == 0


def upload_to_s3(local_path: Path, bucket: str, s3_key: str) -> bool:
    """Upload a file to S3 using AWS CLI. Returns True on success."""
    result = subprocess.run(
        ["aws", "s3", "cp", str(local_path), f"s3://{bucket}/{s3_key}", "--quiet"],
        capture_output=True,
    )
    if result.returncode != 0:
        print(f"    S3 upload failed: {result.stderr.decode()}")
        return False
    return True


def list_dates(start: str, end: str) -> list[str]:
    """Generate date strings start -> end (exclusive)."""
    s = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end, "%Y-%m-%d")
    dates = []
    d = s
    while d < e:
        dates.append(d.strftime("%Y-%m-%d"))
        d += timedelta(days=1)
    return dates


def process_orderbook_day(
    date_str: str,
    bucket: str,
    tmp_dir: Path,
    resume: bool,
    n_levels: int = N_LEVELS,
) -> bool:
    """Download one day of orderbook, process, upload smoothed + raw to S3."""
    smooth_key = f"bybit_orderbook/{date_str}.parquet"
    raw_key = f"bybit_orderbook_raw/{date_str}.parquet"

    if resume:
        if s3_key_exists(bucket, smooth_key) and s3_key_exists(bucket, raw_key):
            return True  # both already in S3

    url = discover_file_url(date_str)
    if url is None:
        print(f"  {date_str}: no orderbook data available")
        return False

    # Download to temp file
    tmp_zip = tmp_dir / "ob_download.zip"
    print(f"  {date_str}: downloading orderbook...")
    if not download_to_file(url, tmp_zip):
        return False

    size_mb = tmp_zip.stat().st_size / 1e6
    print(f"  {date_str}: downloaded {size_mb:.1f} MB, extracting...")

    try:
        raw_data = _decompress(tmp_zip.read_bytes())
    except Exception as e:
        print(f"  {date_str}: decompression failed: {e}")
        tmp_zip.unlink(missing_ok=True)
        return False

    tmp_zip.unlink(missing_ok=True)

    print(f"  {date_str}: parsing {len(raw_data)/1e6:.0f} MB...")
    first_bytes = raw_data[:200].strip()
    if first_bytes.startswith(b"{"):
        df = _parse_json_lines(raw_data, n_levels)
    else:
        df = parse_csv_orderbook(raw_data, n_levels)

    del raw_data  # free memory

    if df.empty:
        print(f"  {date_str}: no valid snapshots parsed")
        return False

    n_snapshots = len(df)
    print(f"  {date_str}: {n_snapshots:,} snapshots")

    # Save and upload raw
    raw_path = tmp_dir / "raw.parquet"
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, raw_path, compression="snappy")
    if not upload_to_s3(raw_path, bucket, raw_key):
        raw_path.unlink(missing_ok=True)
        return False
    raw_mb = raw_path.stat().st_size / 1e6
    raw_path.unlink(missing_ok=True)

    # Apply SG smoothing and upload
    df_smooth = apply_sg_to_prices(df, n_levels)
    del df

    smooth_path = tmp_dir / "smooth.parquet"
    table = pa.Table.from_pandas(df_smooth, preserve_index=False)
    pq.write_table(table, smooth_path, compression="snappy")
    del df_smooth

    if not upload_to_s3(smooth_path, bucket, smooth_key):
        smooth_path.unlink(missing_ok=True)
        return False
    smooth_mb = smooth_path.stat().st_size / 1e6
    smooth_path.unlink(missing_ok=True)

    print(f"  {date_str}: uploaded raw={raw_mb:.1f}MB smooth={smooth_mb:.1f}MB ({n_snapshots:,} snapshots)")
    return True


def process_trades_day(
    date_str: str,
    bucket: str,
    tmp_dir: Path,
    resume: bool,
) -> bool:
    """Download one day of trades, aggregate to 200ms, upload to S3."""
    s3_key = f"bybit_trades_200ms/{date_str}.parquet"

    if resume and s3_key_exists(bucket, s3_key):
        return True

    url = discover_trade_url(date_str)
    if url is None:
        print(f"  {date_str}: no trade data available")
        return False

    print(f"  {date_str}: downloading trades...")
    data = download_with_retry(url)
    if data is None:
        return False

    size_mb = len(data) / 1e6
    print(f"  {date_str}: downloaded {size_mb:.1f} MB, parsing...")

    df = parse_and_aggregate(data)
    del data

    if df.empty:
        print(f"  {date_str}: no valid trades parsed")
        return False

    n_bins = len(df)
    n_trades = df["n_trades"].sum()

    out_path = tmp_dir / "trades.parquet"
    table = pa.Table.from_pandas(df, schema=OUTPUT_SCHEMA, preserve_index=False)
    pq.write_table(table, out_path, compression="snappy")
    del df

    if not upload_to_s3(out_path, bucket, s3_key):
        out_path.unlink(missing_ok=True)
        return False

    file_mb = out_path.stat().st_size / 1e6
    out_path.unlink(missing_ok=True)
    print(f"  {date_str}: {n_trades:,} trades -> {n_bins:,} bins ({file_mb:.2f} MB)")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Download Bybit data and upload directly to S3",
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
        "--bucket", default=DEFAULT_BUCKET,
        help=f"S3 bucket (default: {DEFAULT_BUCKET})",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip dates that already exist in S3",
    )
    parser.add_argument(
        "--orderbook-only", action="store_true",
        help="Only download orderbook data",
    )
    parser.add_argument(
        "--trades-only", action="store_true",
        help="Only download trade data",
    )
    parser.add_argument(
        "--tmp-dir", type=Path, default=Path("/tmp/bybit_download"),
        help="Temporary directory for processing (default: /tmp/bybit_download)",
    )
    args = parser.parse_args()

    if args.end_date is None:
        args.end_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    do_orderbook = not args.trades_only
    do_trades = not args.orderbook_only

    # Verify AWS CLI works
    result = subprocess.run(
        ["aws", "s3api", "head-bucket", "--bucket", args.bucket],
        capture_output=True,
    )
    if result.returncode != 0:
        print(f"ERROR: Cannot access S3 bucket '{args.bucket}'. Check AWS credentials.")
        sys.exit(1)

    dates = list_dates(args.start_date, args.end_date)
    print(f"Bybit -> S3 download: {args.start_date} -> {args.end_date}")
    print(f"  {len(dates)} days, bucket={args.bucket}")
    print(f"  orderbook={'yes' if do_orderbook else 'no'}, trades={'yes' if do_trades else 'no'}")
    print(f"  resume={'yes' if args.resume else 'no'}")

    args.tmp_dir.mkdir(parents=True, exist_ok=True)

    ob_ok = ob_fail = ob_skip = 0
    tr_ok = tr_fail = tr_skip = 0

    for i, date_str in enumerate(dates):
        if do_orderbook:
            if args.resume and s3_key_exists(args.bucket, f"bybit_orderbook/{date_str}.parquet") \
               and s3_key_exists(args.bucket, f"bybit_orderbook_raw/{date_str}.parquet"):
                ob_skip += 1
            else:
                if process_orderbook_day(date_str, args.bucket, args.tmp_dir, resume=False):
                    ob_ok += 1
                else:
                    ob_fail += 1

        if do_trades:
            if args.resume and s3_key_exists(args.bucket, f"bybit_trades_200ms/{date_str}.parquet"):
                tr_skip += 1
            else:
                if process_trades_day(date_str, args.bucket, args.tmp_dir, resume=False):
                    tr_ok += 1
                else:
                    tr_fail += 1

        time.sleep(RATE_LIMIT_SLEEP)

        if (i + 1) % 10 == 0 or i == len(dates) - 1:
            pct = (i + 1) / len(dates) * 100
            parts = [f"Progress: {i + 1}/{len(dates)} ({pct:.0f}%)"]
            if do_orderbook:
                parts.append(f"OB: {ob_ok} ok, {ob_skip} skip, {ob_fail} fail")
            if do_trades:
                parts.append(f"TR: {tr_ok} ok, {tr_skip} skip, {tr_fail} fail")
            print(f"\n{' — '.join(parts)}\n")

    print(f"\n{'='*60}")
    print("Done!")
    if do_orderbook:
        print(f"  Orderbook: {ob_ok} uploaded, {ob_skip} skipped, {ob_fail} failed")
    if do_trades:
        print(f"  Trades: {tr_ok} uploaded, {tr_skip} skipped, {tr_fail} failed")


if __name__ == "__main__":
    main()
