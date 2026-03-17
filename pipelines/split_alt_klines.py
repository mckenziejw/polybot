"""Split monolithic alt kline parquets into daily 1-minute bar parquets."""

import pandas as pd
from pathlib import Path

BASE = Path("data")
ASSETS = ["eth", "sol", "xrp"]

for asset in ASSETS:
    src = BASE / f"{asset}_quotes" / f"{asset}usdt_quotes.parquet"
    out_dir = BASE / f"{asset}_quotes_1m"
    out_dir.mkdir(exist_ok=True)

    df = pd.read_parquet(src)
    print(f"\n=== {asset.upper()} ===")
    print(f"  Source rows: {len(df):,}")

    # Resample to 1-minute bars: floor to minute boundary, take last quote per minute
    df["minute_ms"] = df["timestamp_ms"] // 60_000 * 60_000
    df = (
        df.sort_values("timestamp_ms")
        .groupby("minute_ms", sort=False)
        .agg({"bid_price": "last", "ask_price": "last"})
        .reset_index()
        .rename(columns={"minute_ms": "timestamp_ms"})
    )
    print(f"  After 1m resample: {len(df):,} bars")

    # Compute date from timestamp_ms (UTC)
    df["date"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True).dt.date

    date_min = df["date"].min()
    date_max = df["date"].max()
    print(f"  Date range: {date_min} to {date_max}")

    total_written = 0
    for date, group in df.groupby("date"):
        day_df = group.drop(columns="date").sort_values("timestamp_ms").reset_index(drop=True)
        out_path = out_dir / f"{date}.parquet"
        day_df.to_parquet(out_path, engine="pyarrow", index=False)
        total_written += len(day_df)

    n_days = df["date"].nunique()
    print(f"  Written: {n_days} daily files, {total_written:,} total rows -> {out_dir}/")
