"""Download derivatives data from Crypto Lake S3.

Downloads funding rate, open interest, and liquidation data for
Binance Futures BTC-USDT-PERP (and optionally other assets).

Uses the 'cryptolake' AWS profile for authentication.

Usage:
    python -m pipelines.download_cryptolake
    python -m pipelines.download_cryptolake --start 2025-06-01 --end 2026-03-16
    python -m pipelines.download_cryptolake --symbols BTC-USDT-PERP ETH-USDT-PERP
    python -m pipelines.download_cryptolake --data-types funding open_interest
"""

import argparse
import os
from datetime import datetime, timedelta
from pathlib import Path

import boto3
import pandas as pd
import pyarrow.parquet as pq

BUCKET = "qnt.data"
PREFIX = "market-data/cryptofeed"
EXCHANGE = "BINANCE_FUTURES"
REGION = "eu-west-1"
PROFILE = "cryptolake"

DATA_TYPES = ["funding", "open_interest", "liquidations", "candles"]
DEFAULT_SYMBOLS = ["BTC-USDT-PERP"]


def download_data_type(
    s3_client,
    data_type: str,
    symbol: str,
    start: str,
    end: str,
    output_dir: Path,
    resume: bool = True,
) -> int:
    """Download one data type for one symbol over a date range.

    Returns number of days downloaded.
    """
    out_dir = output_dir / data_type / f"exchange={EXCHANGE}" / f"symbol={symbol}"
    out_dir.mkdir(parents=True, exist_ok=True)

    s3_prefix = f"{PREFIX}/{data_type}/exchange={EXCHANGE}/symbol={symbol}/"

    # List available dates
    paginator = s3_client.get_paginator("list_objects_v2")
    all_dates = []
    for page in paginator.paginate(Bucket=BUCKET, Prefix=s3_prefix, Delimiter="/"):
        if "CommonPrefixes" in page:
            all_dates.extend(
                p["Prefix"].split("=")[-1].rstrip("/") for p in page["CommonPrefixes"]
            )

    # Filter to requested range
    dates = sorted(d for d in all_dates if start <= d <= end)

    if not dates:
        print(f"  {data_type}/{symbol}: no data in range {start} → {end}")
        return 0

    downloaded = 0
    skipped = 0

    for date_str in dates:
        out_path = out_dir / f"{date_str}.parquet"

        if resume and out_path.exists():
            skipped += 1
            continue

        day_prefix = f"{s3_prefix}dt={date_str}/"
        resp = s3_client.list_objects_v2(Bucket=BUCKET, Prefix=day_prefix)

        if "Contents" not in resp:
            continue

        # Download and concatenate all files for this day
        frames = []
        for obj in resp["Contents"]:
            key = obj["Key"]
            if not key.endswith(".parquet"):
                continue
            local_tmp = out_dir / f"_tmp_{date_str}.parquet"
            s3_client.download_file(BUCKET, key, str(local_tmp))
            df = pd.read_parquet(local_tmp)
            frames.append(df)
            local_tmp.unlink()

        if frames:
            combined = pd.concat(frames, ignore_index=True)
            combined.to_parquet(out_path, index=False)
            downloaded += 1

    print(
        f"  {data_type}/{symbol}: {downloaded} downloaded, {skipped} skipped "
        f"({dates[0]} → {dates[-1]}, {len(dates)} total)"
    )
    return downloaded


def main():
    parser = argparse.ArgumentParser(description="Download Crypto Lake derivatives data")
    parser.add_argument("--start", default="2025-06-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default="2026-03-16", help="End date YYYY-MM-DD")
    parser.add_argument(
        "--symbols", nargs="+", default=DEFAULT_SYMBOLS, help="Symbols to download"
    )
    parser.add_argument(
        "--data-types", nargs="+", default=DATA_TYPES, help="Data types to download"
    )
    parser.add_argument(
        "--output-dir",
        default="data/cryptolake",
        help="Output directory (default: data/cryptolake)",
    )
    parser.add_argument("--resume", action="store_true", default=True, help="Skip existing files")
    parser.add_argument("--no-resume", action="store_false", dest="resume")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    session = boto3.Session(profile_name=PROFILE, region_name=REGION)
    s3 = session.client("s3")

    print(f"Downloading from Crypto Lake → {output_dir}")
    print(f"Range: {args.start} → {args.end}")
    print(f"Symbols: {args.symbols}")
    print(f"Data types: {args.data_types}")
    print()

    total = 0
    for symbol in args.symbols:
        for data_type in args.data_types:
            n = download_data_type(
                s3, data_type, symbol, args.start, args.end, output_dir, args.resume
            )
            total += n

    print(f"\nDone. {total} day-files downloaded to {output_dir}")

    # Print disk usage summary
    for dt in args.data_types:
        dt_dir = output_dir / dt
        if dt_dir.exists():
            size = sum(f.stat().st_size for f in dt_dir.rglob("*.parquet"))
            print(f"  {dt}: {size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
