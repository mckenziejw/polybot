"""
settlement_latency.py

Measures the gap between CLOB match time (exchange_timestamp from WS capture)
and on-chain block timestamp for BTC up/down markets.

This gap is the nonce race exploit window — the time between when the CLOB
matches a trade and when it's actually settled on-chain.

Usage:
    python settlement_latency.py --trade-dir data/trade_events --n-markets 50
"""

import json
import time
import argparse
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--trade-dir",  default="data/trade_events")
parser.add_argument("--n-markets",  type=int, default=10)
parser.add_argument("--out",        default="data/settlement_latency.parquet")
parser.add_argument("--checkpoint", default="data/settlement_latency_checkpoint.parquet")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

with open("config.json") as f:
    config = json.load(f)

ETHERSCAN_KEY  = config["etherscan"]["api_key"]
ETHERSCAN_API  = "https://api.etherscan.io/v2/api"
CTF_EXCHANGE   = "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E"
SEARCH_WINDOW_S = 120  # max seconds to look for settlement tx after match

# ---------------------------------------------------------------------------
# HTTP session with retries
# ---------------------------------------------------------------------------

def make_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session

SESSION = make_session()

# ---------------------------------------------------------------------------
# Etherscan helpers
# ---------------------------------------------------------------------------

def ts_to_block(ts_s: int) -> int:
    r = SESSION.get(ETHERSCAN_API, params={
        "chainid":   137,
        "module":    "block",
        "action":    "getblocknobytime",
        "timestamp": ts_s,
        "closest":   "before",
        "apikey":    ETHERSCAN_KEY,
    }, timeout=30)
    result = r.json()["result"]
    return int(result)


def fetch_ctf_txs(start_ts_s: int, end_ts_s: int) -> list[dict]:
    start_block = ts_to_block(start_ts_s)
    end_block   = ts_to_block(end_ts_s)

    r = SESSION.get(ETHERSCAN_API, params={
        "chainid":    137,
        "module":     "account",
        "action":     "txlist",
        "address":    CTF_EXCHANGE,
        "startblock": start_block,
        "endblock":   end_block,
        "sort":       "asc",
        "apikey":     ETHERSCAN_KEY,
    }, timeout=30)

    result = r.json().get("result", [])
    if not isinstance(result, list):
        return []

    return [
        {
            "tx_hash":      tx["hash"],
            "block_ts":     int(tx["timeStamp"]),
            "block_number": int(tx["blockNumber"]),
            "method_id":    tx["input"][:10] if tx.get("input") else "",
        }
        for tx in result
    ]

# ---------------------------------------------------------------------------
# Trade loading
# ---------------------------------------------------------------------------

def load_trade_events(trade_dir: str, n: int) -> pd.DataFrame:
    files  = sorted(Path(trade_dir).glob("btc-updown-*.parquet"))
    step   = max(1, len(files) // n)
    sample = files[::step][:n]
    print(f"Loading {len(sample)} files from {len(files)} available...")

    dfs = []
    for f in sample:
        df = pd.read_parquet(f)
        df["slug"] = f.stem
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    deduped  = combined.drop_duplicates(subset=["slug", "hash"])
    print(f"Loaded {len(combined):,} rows → {len(deduped):,} unique trades")
    return deduped

# ---------------------------------------------------------------------------
# Match trades to on-chain txs by timing
# ---------------------------------------------------------------------------

def match_trades_to_chain(
    trades: pd.DataFrame,
    ctf_txs: list[dict],
) -> list[dict]:
    if not ctf_txs:
        return []

    tx_df  = pd.DataFrame(ctf_txs).sort_values("block_ts")
    trades = trades.sort_values("exchange_timestamp")
    rows   = []

    for _, trade in trades.iterrows():
        match_ts_s = trade["exchange_timestamp"] / 1000.0

        after = tx_df[
            (tx_df["block_ts"] >= match_ts_s) &
            (tx_df["block_ts"] <= match_ts_s + SEARCH_WINDOW_S)
        ]
        if after.empty:
            continue

        settlement = after.iloc[0]

        rows.append({
            "slug":        trade["slug"],
            "hash":        trade["hash"],
            "exchange_ts": match_ts_s,
            "block_ts":    float(settlement["block_ts"]),
            "latency_s":   settlement["block_ts"] - match_ts_s,
            "tx_hash":     settlement["tx_hash"],
            "method_id":   settlement["method_id"],
            "price":       trade["trade_price"],
            "size":        trade["size"],
            "side":        trade["side"],
        })

    return rows

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    out_path        = Path(args.out)
    checkpoint_path = Path(args.checkpoint)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load trade events
    trades = load_trade_events(args.trade_dir, args.n_markets)
    trades["window_60s"] = (trades["exchange_timestamp"] // 1000 // 60).astype(int)
    windows = sorted(trades["window_60s"].unique())

    # Resume from checkpoint if available
    all_rows          = []
    completed_windows = set()

    if checkpoint_path.exists():
        prev = pd.read_parquet(checkpoint_path)
        completed_windows = set((prev["exchange_ts"] // 60).astype(int).unique())
        all_rows = prev.to_dict("records")
        print(f"Resuming from checkpoint: {len(all_rows):,} rows, "
              f"{len(completed_windows)}/{len(windows)} windows done")

    remaining = [w for w in windows if w not in completed_windows]
    print(f"Processing {len(remaining)} remaining windows "
          f"across {trades['slug'].nunique()} markets...")

    api_calls = 0

    for i, window in enumerate(remaining):
        window_trades = trades[trades["window_60s"] == window]
        start_ts      = window * 60
        end_ts        = start_ts + 60 + SEARCH_WINDOW_S

        try:
            ctf_txs    = fetch_ctf_txs(start_ts, end_ts)
            api_calls += 3
        except Exception as e:
            print(f"  Window {window} fetch failed: {e} — skipping")
            time.sleep(2)
            continue

        rows = match_trades_to_chain(window_trades, ctf_txs)
        all_rows.extend(rows)

        # Rate limit: 3 calls per window, free tier is 5/s
        time.sleep(0.6)

        # Checkpoint every 50 windows
        if (i + 1) % 50 == 0:
            pd.DataFrame(all_rows).to_parquet(checkpoint_path)
            print(f"  [{i+1}/{len(remaining)}] {api_calls} API calls, "
                  f"{len(all_rows):,} matches — checkpointed")

    if not all_rows:
        print("No matches found.")
        return

    df    = pd.DataFrame(all_rows)
    clean = df[(df["latency_s"] >= 0) & (df["latency_s"] < 300)].copy()

    print(f"\n{'='*55}")
    print(f"Markets:         {df['slug'].nunique()}")
    print(f"Total matched:   {len(df):,}")
    print(f"Clean sample:    {len(clean):,}  "
          f"(dropped {len(df) - len(clean)} outliers)")

    print(f"\nRace window — exchange_ts → block_ts (seconds):")
    print(clean["latency_s"].describe(
        percentiles=[.25, .5, .75, .90, .95, .99]
    ).round(2))

    print(f"\n% of trades where race window is less than:")
    for t in [2, 5, 10, 20, 30, 60]:
        pct = (clean["latency_s"] <= t).mean() * 100
        print(f"  {t:3d}s: {pct:.1f}%")

    tx_sizes = clean.groupby("tx_hash").size()
    print(f"\nSettlement batch sizes (trades per tx):")
    print(tx_sizes.describe(percentiles=[.5, .75, .95]).round(1))

    print(f"\nCTF method IDs at settlement:")
    print(clean["method_id"].value_counts().head(10).to_string())

    clean.to_parquet(out_path)
    # Clean up checkpoint on successful completion
    if checkpoint_path.exists():
        checkpoint_path.unlink()
    print(f"\nSaved {len(clean):,} rows → {out_path}")

    # Add to your analysis after loading clean df
    # Look at batch submission timing directly

    batch_times = clean.groupby("tx_hash")["block_ts"].first().sort_values()
    batch_intervals = batch_times.diff().dropna()

    print(f"\nOperator batch submission intervals (seconds):")
    print(batch_intervals.describe(percentiles=[.25, .5, .75, .90, .95, .99]).round(2))

    print(f"\n% of batches submitted within:")
    for t in [5, 10, 30, 60, 120]:
        pct = (batch_intervals <= t).mean() * 100
        print(f"  {t:3d}s: {pct:.1f}%")

    # incrementNonce calls specifically
    nonce_txs = clean[clean["method_id"] == "0x627cdcb9"]
    print(f"\nincrementNonce calls: {len(nonce_txs):,}")
    print(f"Unique wallets calling incrementNonce: {nonce_txs['tx_hash'].nunique()}")

    # Time distribution of incrementNonce within market windows
    # Extract market open timestamp from slug
    nonce_txs = nonce_txs.copy()
    nonce_txs["market_open"] = nonce_txs["slug"].str.extract(r"-(\d+)$")[0].astype(float)
    nonce_txs["time_into_market_s"] = nonce_txs["exchange_ts"] - nonce_txs["market_open"]

    print(f"\nincrementNonce timing within market window:")
    print(nonce_txs["time_into_market_s"].describe(
        percentiles=[.25, .5, .75, .90, .95]
    ).round(1))


if __name__ == "__main__":
    main()