"""
batch_cadence.py

Pulls your personal trade history from the authenticated CLOB API,
resolves each transaction_hash to a block timestamp via Polygon RPC,
and measures the match_time → block_time gap (the nonce race window).

Also checks whether last_update tracks block time or CLOB-side confirmation.

Usage:
    python my_trade_latency.py
    python my_trade_latency.py --market 0xABC...  # filter to one market

Requires config.json with:
    {
      "polymarket": {
          "private_key": "0x...",
          "api_key": "...",
          "api_secret": "...",
          "api_passphrase": "..."
      }
    }
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import ApiCreds, TradeParams

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--market", default=None,
                    help="Optional condition ID to filter to one market")
parser.add_argument("--out", default="data/my_trade_latency.parquet")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Clients
# ---------------------------------------------------------------------------

with open("config.json") as f:
    config = json.load(f)
    pm = config["polymarket"]

client = ClobClient(
    host="https://clob.polymarket.com",
    chain_id=137,
    key=pm["private_key"],
    creds=ApiCreds(
        api_key=pm["api_key"],
        api_secret=pm["api_secret"],
        api_passphrase=pm["api_passphrase"],
    ),
)
client.set_api_creds(client.create_or_derive_api_creds())
ETHERSCAN_API = "https://api.etherscan.io/v2/api"
ETHERSCAN_KEY = config["etherscan"]["api_key"]

_session = requests.Session()
_session.mount("https://", HTTPAdapter(max_retries=Retry(
    total=5, backoff_factor=1.0,
    status_forcelist=[429, 500, 502, 503, 504],
)))

# ---------------------------------------------------------------------------
# Block timestamp via Etherscan — tx endpoint returns timeStamp directly
# ---------------------------------------------------------------------------

_tx_cache: dict[str, int] = {}

def get_block_timestamp(tx_hash: str) -> int | None:
    if tx_hash in _tx_cache:
        return _tx_cache[tx_hash]
    try:
        r = _session.get(ETHERSCAN_API, params={
            "chainid": 137,
            "module":  "proxy",
            "action":  "eth_getTransactionByHash",
            "txhash":  tx_hash,
            "apikey":  ETHERSCAN_KEY,
        }, timeout=15)
        block_hex = r.json()["result"]["blockNumber"]
        block_num = int(block_hex, 16)

        # Now fetch block to get timestamp
        r2 = _session.get(ETHERSCAN_API, params={
            "chainid": 137,
            "module":  "proxy",
            "action":  "eth_getBlockByNumber",
            "tag":     hex(block_num),
            "boolean": "false",
            "apikey":  ETHERSCAN_KEY,
        }, timeout=15)
        ts = int(r2.json()["result"]["timestamp"], 16)
        _tx_cache[tx_hash] = ts
        time.sleep(0.25)  # ~4 req/s — stay under free tier limit
        return ts
    except Exception as e:
        print(f"    Etherscan error {tx_hash[:16]}...: {e}")
        return None

# ---------------------------------------------------------------------------
# Fetch all trades via cursor pagination
# ---------------------------------------------------------------------------

def fetch_all_trades(market: str | None = None) -> list[dict]:
    trades      = []
    next_cursor = None
    page        = 0

    while True:
        params = TradeParams(market=market) if market else TradeParams()
        if next_cursor:
            params.next_cursor = next_cursor

        try:
            resp = client.get_trades(params)
        except Exception as e:
            print(f"  API error on page {page}: {e}")
            break

        # py-clob-client may return a list or a dict with 'data'
        if isinstance(resp, list):
            batch       = resp
            next_cursor = None
        else:
            batch       = resp.get("data", [])
            next_cursor = resp.get("next_cursor")

        trades.extend(batch)
        page += 1

        # "LTE=" is the sentinel for last page
        if not next_cursor or next_cursor == "LTE=":
            break

        time.sleep(0.1)

    return trades

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_ts(raw) -> float | None:
    """Parse unix timestamp — API returns string seconds."""
    if not raw:
        return None
    try:
        v = float(raw)
        # If milliseconds (>1e12), convert
        return v / 1000.0 if v > 1e12 else v
    except (ValueError, TypeError):
        return None

def main():
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("Fetching trade history...")
    trades = fetch_all_trades(args.market)
    print(f"  {len(trades)} trades returned")

    if not trades:
        print("No trades found. Check auth credentials.")
        return

    # Show raw field names from first trade so we know what we're working with
    print(f"\nFields in trade object: {list(trades[0].keys())}")
    print(f"Sample trade:")
    for k, v in trades[0].items():
        print(f"  {k}: {v}")

    rows = []
    confirmed = [t for t in trades if t.get("transaction_hash")]
    print(f"\n{len(confirmed)} trades have a transaction_hash (confirmed on-chain)")
    print(f"Resolving block timestamps via RPC...")

    for i, t in enumerate(confirmed):
        tx_hash  = t["transaction_hash"]
        match_ts = parse_ts(t.get("match_time"))
        last_ts  = parse_ts(t.get("last_update"))

        if not match_ts:
            continue

        block_ts = get_block_timestamp(tx_hash)
        if block_ts is None:
            continue

        rows.append({
            "trade_id":        t.get("id"),
            "market":          t.get("market"),
            "status":          t.get("status"),
            "side":            t.get("side"),
            "trader_side":     t.get("trader_side"),
            "price":           float(t.get("price", 0)),
            "size":            float(t.get("size", 0)),
            "match_time":      match_ts,
            "last_update":     last_ts,
            "block_time":      float(block_ts),
            "tx_hash":         tx_hash,
            # Key measurements
            "match_to_block":  block_ts - match_ts,
            "last_to_block":   (block_ts - last_ts) if last_ts else None,
            "match_to_last":   (last_ts - match_ts) if last_ts else None,
        })

        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(confirmed)} resolved...")
        time.sleep(0.05)

    if not rows:
        print("No rows with resolvable timestamps.")
        return

    df = pd.DataFrame(rows)

    # Filter obvious garbage (negative = clock skew, >300s = stale/anomaly)
    clean = df[
        (df["match_to_block"] >= 0) &
        (df["match_to_block"] < 300)
    ].copy()

    print(f"\n{'='*55}")
    print(f"Total confirmed trades:  {len(df):,}")
    print(f"Clean sample:            {len(clean):,}")

    print(f"\n--- match_time → block_time (race window) ---")
    print(clean["match_to_block"].describe(
        percentiles=[.25, .5, .75, .90, .95, .99]
    ).round(2))

    print(f"\n% settling within:")
    for t in [2, 5, 10, 20, 30, 60]:
        pct = (clean["match_to_block"] <= t).mean() * 100
        print(f"  {t:3d}s: {pct:.1f}%")

    if df["last_to_block"].notna().any():
        last_clean = clean[clean["last_to_block"].notna()]
        print(f"\n--- last_update → block_time ---")
        print(f"(negative = last_update is AFTER block time, i.e. CLOB confirmation lag)")
        print(last_clean["last_to_block"].describe(
            percentiles=[.25, .5, .75, .90, .95]
        ).round(2))

        print(f"\n--- match_time → last_update ---")
        print(f"(how long after match did CLOB update the record)")
        print(last_clean["match_to_last"].describe(
            percentiles=[.25, .5, .75, .90, .95]
        ).round(2))

    # Taker vs maker — do you settle faster as taker?
    if "trader_side" in df.columns:
        print(f"\nMedian match→block by trader side:")
        print(clean.groupby("trader_side")["match_to_block"].median().round(2))

    df.to_parquet(out_path)
    print(f"\nSaved {len(df):,} rows → {out_path}")


if __name__ == "__main__":
    main()