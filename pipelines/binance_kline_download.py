"""
Download Binance 1-minute klines and convert to quote-tick parquet format.

Downloads historical 1m OHLCV candles from the Binance public API (no key needed)
and synthesizes bid/ask quote ticks compatible with the evolution backtesting engine.

Usage:
    python pipelines/binance_kline_download.py btcusdt 2025-06-01 2026-03-13
    python pipelines/binance_kline_download.py ethusdt 2025-06-01 2026-03-13
    python pipelines/binance_kline_download.py --all 2025-06-01 2026-03-13
"""

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests

BINANCE_API = "https://api.binance.com/api/v3/klines"
MAX_LIMIT = 1000  # Binance max per request

SYMBOLS = {
    "btcusdt": {"precision": 2},
    "ethusdt": {"precision": 2},
    "solusdt": {"precision": 2},
    "xrpusdt": {"precision": 4},
}

QUOTE_SCHEMA = pa.schema([
    pa.field("timestamp_ms", pa.int64()),
    pa.field("bid_price", pa.float64()),
    pa.field("ask_price", pa.float64()),
])


def date_to_ms(date_str: str) -> int:
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def download_klines(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """Download 1m klines from Binance API in batches."""
    all_rows = []
    current_ms = start_ms
    total_expected = (end_ms - start_ms) / 60_000
    fetched = 0

    while current_ms < end_ms:
        params = {
            "symbol": symbol.upper(),
            "interval": "1m",
            "startTime": current_ms,
            "endTime": end_ms,
            "limit": MAX_LIMIT,
        }

        for attempt in range(5):
            try:
                resp = requests.get(BINANCE_API, params=params, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                break
            except (requests.RequestException, ValueError) as e:
                if attempt < 4:
                    wait = 2 ** attempt
                    print(f"  Retry {attempt+1}/5 after {wait}s: {e}")
                    time.sleep(wait)
                else:
                    raise

        if not data:
            break

        for row in data:
            all_rows.append({
                "open_time": row[0],
                "open": float(row[1]),
                "high": float(row[2]),
                "low": float(row[3]),
                "close": float(row[4]),
                "volume": float(row[5]),
                "close_time": row[6],
            })

        fetched += len(data)
        pct = min(100, fetched / total_expected * 100)
        last_ts = data[-1][0]
        last_dt = datetime.fromtimestamp(last_ts / 1000, tz=timezone.utc)
        print(f"  {fetched:>8,} klines ({pct:.0f}%) — up to {last_dt.strftime('%Y-%m-%d %H:%M')}")

        # Next batch starts after last candle
        current_ms = data[-1][6] + 1

        if len(data) < MAX_LIMIT:
            break

        # Rate limit: 1200 req/min, be conservative
        time.sleep(0.1)

    df = pd.DataFrame(all_rows)
    df = df.drop_duplicates(subset=["open_time"]).sort_values("open_time").reset_index(drop=True)
    return df


def klines_to_quotes(df: pd.DataFrame, precision: int) -> pd.DataFrame:
    """Convert 1m OHLCV klines to synthetic bid/ask quotes.

    Strategy: use close price as mid, estimate spread from high-low range.
    Each 1m candle produces one quote tick at candle close time.
    """
    mid = df["close"]

    # Spread estimate: use 20% of the high-low range as typical spread,
    # with a floor of 1 tick at the given precision
    hl_range = df["high"] - df["low"]
    min_tick = 10 ** (-precision)
    spread = np.maximum(hl_range * 0.2, min_tick)

    half_spread = spread / 2
    quotes = pd.DataFrame({
        "timestamp_ms": df["open_time"],
        "bid_price": np.round(mid - half_spread, precision),
        "ask_price": np.round(mid + half_spread, precision),
    })
    return quotes


def download_and_save(symbol: str, start_date: str, end_date: str, out_dir: Path,
                      merge_existing: bool = True):
    """Download klines for a symbol and save as quote parquet.

    If merge_existing=True and a Telonex quotes file exists, kline data is used
    only for timestamps not covered by the higher-resolution Telonex data.
    The merged result overwrites the existing file.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{symbol}_quotes.parquet"

    start_ms = date_to_ms(start_date)
    end_ms = date_to_ms(end_date)

    print(f"Downloading {symbol.upper()} 1m klines: {start_date} → {end_date}")
    klines = download_klines(symbol, start_ms, end_ms)

    if klines.empty:
        print(f"  No data returned for {symbol}")
        return

    days = (klines["open_time"].max() - klines["open_time"].min()) / 86_400_000
    print(f"  {len(klines):,} klines over {days:.0f} days")

    precision = SYMBOLS.get(symbol, {}).get("precision", 2)
    kline_quotes = klines_to_quotes(klines, precision)

    # Merge with existing Telonex data if available
    if merge_existing and out_file.exists():
        existing = pd.read_parquet(out_file, columns=["timestamp_ms", "bid_price", "ask_price"])
        existing_min = existing["timestamp_ms"].min()
        existing_max = existing["timestamp_ms"].max()
        print(f"  Existing data: {len(existing):,} rows, "
              f"{datetime.fromtimestamp(existing_min/1000, tz=timezone.utc).strftime('%Y-%m-%d')} → "
              f"{datetime.fromtimestamp(existing_max/1000, tz=timezone.utc).strftime('%Y-%m-%d')}")

        # Keep kline data only for timestamps outside existing coverage
        kline_before = kline_quotes[kline_quotes["timestamp_ms"] < existing_min]
        kline_after = kline_quotes[kline_quotes["timestamp_ms"] > existing_max]
        print(f"  Kline fill: {len(kline_before):,} before + {len(kline_after):,} after existing")

        merged = pd.concat([kline_before, existing, kline_after], ignore_index=True)
        merged = merged.sort_values("timestamp_ms").reset_index(drop=True)
        quotes = merged
    else:
        quotes = kline_quotes

    table = pa.Table.from_pandas(quotes, schema=QUOTE_SCHEMA, preserve_index=False)
    pq.write_table(table, out_file, compression="snappy")

    start_dt = datetime.fromtimestamp(quotes["timestamp_ms"].iloc[0] / 1000, tz=timezone.utc)
    end_dt = datetime.fromtimestamp(quotes["timestamp_ms"].iloc[-1] / 1000, tz=timezone.utc)
    file_mb = out_file.stat().st_size / 1_048_576
    print(f"  Wrote {len(quotes):,} quotes ({file_mb:.0f} MB) to {out_file}")
    print(f"  Range: {start_dt.strftime('%Y-%m-%d')} → {end_dt.strftime('%Y-%m-%d')}")
    print(f"  Price: ${quotes['bid_price'].iloc[0]:,.2f} → ${quotes['bid_price'].iloc[-1]:,.2f}")

    return out_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Binance klines as quote parquet")
    parser.add_argument("symbol", help="Symbol (btcusdt, ethusdt, etc.) or --all")
    parser.add_argument("start_date", help="Start date (YYYY-MM-DD)")
    parser.add_argument("end_date", help="End date exclusive (YYYY-MM-DD)")
    parser.add_argument("--out-dir", type=Path, default=Path("data"),
                        help="Output directory (default: data/)")
    args = parser.parse_args()

    if args.symbol == "--all":
        for sym in SYMBOLS:
            asset = sym.replace("usdt", "")
            download_and_save(sym, args.start_date, args.end_date,
                              args.out_dir / f"{asset}_quotes")
            print()
    else:
        asset = args.symbol.lower().replace("usdt", "")
        download_and_save(args.symbol.lower(), args.start_date, args.end_date,
                          args.out_dir / f"{asset}_quotes")
