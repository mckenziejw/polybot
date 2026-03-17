"""Download on-chain data for BTC directional classifier.

Sources:
  1. Etherscan v2 API — USDT Issue/Redeem + USDC Mint events on Ethereum
  2. Coin Metrics Community API — BTC exchange flows, MVRV, active addresses, etc.

All data saved to data/onchain/ as parquet files.
"""

import json
import time
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

# ── Config ───────────────────────────────────────────────────────────

DATA_DIR = Path("data/onchain")

# USDT on Ethereum
USDT_CONTRACT = "0xdac17f958d2ee523a2206206994597c13d831ec7"
USDT_ISSUE_TOPIC = "0xcb8241adb0c3fdb35b70c24ce35c5eb0c17af7431c99f827d44a445ca624176a"
USDT_REDEEM_TOPIC = "0x702d5967f45f6513a38ffc42d6ba9bf230bd40e8f53b16363c7eb4fd2deb9a44"

# USDC on Ethereum
USDC_CONTRACT = "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"
USDC_MINT_TOPIC = "0xab8530f87dc9b59234c4623bf917212bb2536d647574c8e7e5da92c2ede0c9f8"

ETHERSCAN_BASE = "https://api.etherscan.io/v2/api"
COINMETRICS_BASE = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"

# Both USDT and USDC use 6 decimals
STABLECOIN_DECIMALS = 6


def load_etherscan_key() -> str:
    """Load Etherscan API key from config.json."""
    cfg_path = Path("config.json")
    if not cfg_path.exists():
        raise FileNotFoundError("config.json not found")
    with open(cfg_path) as f:
        cfg = json.load(f)
    return cfg["etherscan"]["api_key"]


# ── Etherscan Helpers ────────────────────────────────────────────────


def _etherscan_get(params: dict, api_key: str) -> list[dict]:
    """Make an Etherscan v2 API call with rate limiting."""
    params["apikey"] = api_key
    params["chainid"] = "1"

    resp = requests.get(ETHERSCAN_BASE, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    if data.get("status") != "1" and data.get("message") != "No records found":
        msg = str(data.get("message", "")) + " " + str(data.get("result", ""))
        raise RuntimeError(f"Etherscan error: {msg}")

    result = data.get("result", [])
    if isinstance(result, str):
        return []
    return result


def get_block_by_timestamp(ts: int, api_key: str, closest: str = "before") -> int:
    """Get block number closest to a Unix timestamp."""
    params = {
        "module": "block",
        "action": "getblocknobytime",
        "timestamp": str(ts),
        "closest": closest,
        "apikey": api_key,
        "chainid": "1",
    }
    resp = requests.get(ETHERSCAN_BASE, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return int(data["result"])


def get_logs_paginated(
    contract: str,
    topic0: str,
    from_block: int,
    to_block: int,
    api_key: str,
    chunk_size: int = 500_000,
) -> list[dict]:
    """Fetch all logs matching topic0, chunking by block range."""
    all_logs = []
    current = from_block

    while current <= to_block:
        end = min(current + chunk_size - 1, to_block)

        page = 1
        while True:
            params = {
                "module": "logs",
                "action": "getLogs",
                "address": contract,
                "fromBlock": str(current),
                "toBlock": str(end),
                "topic0": topic0,
                "page": str(page),
                "offset": "1000",
            }
            time.sleep(0.35)  # 3 calls/sec rate limit
            logs = _etherscan_get(params, api_key)
            all_logs.extend(logs)

            if len(logs) < 1000:
                break
            page += 1

        current = end + 1

    return all_logs


def decode_uint256_from_data(data_hex: str) -> int:
    """Decode a uint256 from log data field."""
    # Remove 0x prefix, take first 32 bytes (64 hex chars)
    clean = data_hex.replace("0x", "")
    if len(clean) == 0:
        return 0
    return int(clean[:64], 16)


def parse_mint_events(logs: list[dict], token: str) -> pd.DataFrame:
    """Parse Etherscan log entries into a DataFrame of mint/burn events."""
    records = []
    for log in logs:
        amount_raw = decode_uint256_from_data(log["data"])
        amount = amount_raw / (10 ** STABLECOIN_DECIMALS)

        block = int(log["blockNumber"], 16)
        ts = int(log["timeStamp"], 16)

        records.append({
            "timestamp": ts,
            "block_number": block,
            "tx_hash": log["transactionHash"],
            "amount_usd": amount,
            "token": token,
        })

    df = pd.DataFrame(records)
    if len(df) > 0:
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
        df = df.sort_values("timestamp").reset_index(drop=True)
    return df


# ── Etherscan Download ───────────────────────────────────────────────


def download_stablecoin_mints(
    start_date: str = "2025-06-01",
    end_date: str = "2026-03-17",
    min_amount: float = 1_000_000,
) -> pd.DataFrame:
    """Download USDT Issue/Redeem + USDC Mint events from Etherscan.

    Parameters
    ----------
    start_date, end_date : str  'YYYY-MM-DD'
    min_amount : float  minimum event size in USD to keep

    Returns
    -------
    DataFrame with columns: timestamp, block_number, tx_hash, amount_usd,
                           token, event_type, datetime
    """
    api_key = load_etherscan_key()

    # Get block boundaries
    start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())
    end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())

    print(f"Getting block numbers for {start_date} to {end_date}...")
    time.sleep(0.35)
    from_block = get_block_by_timestamp(start_ts, api_key, "after")
    time.sleep(0.35)
    to_block = get_block_by_timestamp(end_ts, api_key, "before")
    print(f"  Block range: {from_block:,} to {to_block:,} ({to_block - from_block:,} blocks)")

    all_events = []

    # USDT Issue events
    print("Fetching USDT Issue events...")
    usdt_issue_logs = get_logs_paginated(USDT_CONTRACT, USDT_ISSUE_TOPIC, from_block, to_block, api_key)
    if usdt_issue_logs:
        df = parse_mint_events(usdt_issue_logs, "USDT")
        df["event_type"] = "mint"
        all_events.append(df)
        print(f"  {len(df)} USDT Issue events")

    # USDT Redeem events
    print("Fetching USDT Redeem events...")
    usdt_redeem_logs = get_logs_paginated(USDT_CONTRACT, USDT_REDEEM_TOPIC, from_block, to_block, api_key)
    if usdt_redeem_logs:
        df = parse_mint_events(usdt_redeem_logs, "USDT")
        df["event_type"] = "burn"
        df["amount_usd"] = -df["amount_usd"]  # negative for burns
        all_events.append(df)
        print(f"  {len(df)} USDT Redeem events")

    # USDC has too many Mint events (every USDC mint, thousands per day).
    # Use Coin Metrics daily supply delta for USDC instead.
    # Only fetch USDC if explicitly requested with very small chunks.
    print("Skipping USDC Mint events (too numerous — use Coin Metrics daily supply delta)")

    if not all_events:
        print("No events found!")
        return pd.DataFrame()

    result = pd.concat(all_events, ignore_index=True)
    result = result.sort_values("timestamp").reset_index(drop=True)

    # Filter by minimum amount
    large = result[result["amount_usd"].abs() >= min_amount].reset_index(drop=True)
    print(f"\nTotal events: {len(result)}, >= ${min_amount/1e6:.0f}M: {len(large)}")

    return result


# ── Coin Metrics Download ────────────────────────────────────────────


def download_coinmetrics_daily(
    start_date: str = "2025-06-01",
    end_date: str = "2026-03-17",
) -> dict[str, pd.DataFrame]:
    """Download daily on-chain metrics from Coin Metrics Community API.

    Returns dict of DataFrames: 'btc_onchain', 'usdt_supply', 'usdc_supply'
    """
    results = {}

    # BTC on-chain metrics
    btc_metrics = [
        "FlowInExNtv", "FlowOutExNtv",  # exchange in/out flows (BTC)
        "FlowInExUSD", "FlowOutExUSD",  # exchange in/out flows (USD)
        "SplyExNtv",                     # BTC held on exchanges
        "AdrActCnt",                     # active addresses
        "TxCnt",                         # transaction count
        "FeeTotNtv",                     # total fees
        "CapMVRVCur",                    # MVRV ratio
        "HashRate",                      # hash rate
    ]

    print("Downloading BTC on-chain metrics from Coin Metrics...")
    btc_df = _coinmetrics_fetch("btc", btc_metrics, start_date, end_date)
    if btc_df is not None:
        results["btc_onchain"] = btc_df
        print(f"  {len(btc_df)} days, {len(btc_df.columns)} columns")

    # USDT supply (for minting proxy)
    print("Downloading USDT supply from Coin Metrics...")
    usdt_df = _coinmetrics_fetch("usdt", ["SplyCur", "IssTotNtv"], start_date, end_date)
    if usdt_df is not None:
        results["usdt_supply"] = usdt_df
        print(f"  {len(usdt_df)} days")

    # USDC supply
    print("Downloading USDC supply from Coin Metrics...")
    usdc_df = _coinmetrics_fetch("usdc", ["SplyCur", "IssTotNtv"], start_date, end_date)
    if usdc_df is not None:
        results["usdc_supply"] = usdc_df
        print(f"  {len(usdc_df)} days")

    return results


def _coinmetrics_fetch(
    asset: str,
    metrics: list[str],
    start: str,
    end: str,
) -> pd.DataFrame | None:
    """Fetch metrics from Coin Metrics community API with pagination."""
    all_rows = []
    next_page = None
    metrics_str = ",".join(metrics)

    while True:
        params = {
            "assets": asset,
            "metrics": metrics_str,
            "start_time": start,
            "end_time": end,
            "frequency": "1d",
            "page_size": "10000",
        }
        if next_page:
            params["next_page_token"] = next_page

        resp = requests.get(COINMETRICS_BASE, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        rows = data.get("data", [])
        all_rows.extend(rows)

        next_page = data.get("next_page_token")
        if not next_page or not rows:
            break
        time.sleep(0.7)  # rate limit: 10 req / 6s

    if not all_rows:
        return None

    df = pd.DataFrame(all_rows)
    # Convert time column
    df["date"] = pd.to_datetime(df["time"]).dt.date
    df = df.drop(columns=["time", "asset"], errors="ignore")

    # Convert numeric columns
    for col in df.columns:
        if col != "date":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.sort_values("date").reset_index(drop=True)


# ── Main ─────────────────────────────────────────────────────────────


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Download on-chain data")
    parser.add_argument("--start", default="2025-06-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default="2026-03-17", help="End date YYYY-MM-DD")
    parser.add_argument("--source", default="all", choices=["all", "etherscan", "coinmetrics"],
                        help="Which source to download")
    parser.add_argument("--min-amount", type=float, default=1_000_000,
                        help="Min stablecoin event size in USD (default 1M)")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if args.source in ("all", "etherscan"):
        print("=" * 60)
        print("ETHERSCAN: Stablecoin mint/burn events")
        print("=" * 60)
        events = download_stablecoin_mints(args.start, args.end, args.min_amount)
        if len(events) > 0:
            out_path = DATA_DIR / "stablecoin_events.parquet"
            events.to_parquet(out_path, index=False)
            print(f"Saved {len(events)} events to {out_path}")

            # Print summary
            print("\n=== Event Summary ===")
            for token in events["token"].unique():
                t = events[events["token"] == token]
                mints = t[t["event_type"] == "mint"]
                burns = t[t["event_type"] == "burn"]
                print(f"  {token}: {len(mints)} mints (${mints['amount_usd'].sum()/1e9:.1f}B), "
                      f"{len(burns)} burns (${burns['amount_usd'].abs().sum()/1e9:.1f}B)")

            # Show large events
            large = events[events["amount_usd"].abs() >= 100_000_000].copy()
            if len(large) > 0:
                print(f"\n=== Events >= $100M ({len(large)}) ===")
                for _, row in large.iterrows():
                    print(f"  {row['datetime'].strftime('%Y-%m-%d %H:%M')} "
                          f"{row['token']} {row['event_type']} "
                          f"${row['amount_usd']/1e6:+,.0f}M")
        print()

    if args.source in ("all", "coinmetrics"):
        print("=" * 60)
        print("COIN METRICS: Daily on-chain data")
        print("=" * 60)
        cm_data = download_coinmetrics_daily(args.start, args.end)
        for name, df in cm_data.items():
            out_path = DATA_DIR / f"{name}.parquet"
            df.to_parquet(out_path, index=False)
            print(f"Saved {name}: {len(df)} rows to {out_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
