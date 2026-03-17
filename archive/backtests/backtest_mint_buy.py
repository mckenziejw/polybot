"""
Backtest: Buy both sides below 0.50.

Place resting BUY orders for both tokens at a few ticks below 0.50.
If both fill, you paid < $1 total for a pair that's guaranteed to pay $1.
If one fills, you hold a directional bet you got at a discount.

Uses both raw trade_events and Telonex book_snapshots (checking if bid ever drops to our level).
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

TRADE_DIR = Path("data/trade_events")
TELONEX_DIR = Path("data/telonex_book_snapshots")
RESOLUTION_CACHE = Path("data/resolution_cache.json")

with open(RESOLUTION_CACHE) as f:
    raw_cache = json.load(f)

resolutions = {}
for slug, info in raw_cache.items():
    winner = info.get("winningTokenId") or info.get("winning_token", "")
    if winner:
        resolutions[slug] = winner

print(f"Loaded {len(resolutions)} resolutions")

BUY_PRICES = [0.40, 0.42, 0.44, 0.46, 0.48, 0.49]

results = {bp: {"both_fill": 0, "one_fill_win": 0, "one_fill_lose": 0,
                "neither": 0, "total_pnl": 0.0, "n": 0,
                "fill_times": []} for bp in BUY_PRICES}


def process_market(slug, df, source_label):
    """Process one market's data. df must have columns we can check fills against."""
    winner = resolutions.get(slug)
    if not winner:
        return

    try:
        open_ts = int(slug.split("-")[-1])
    except ValueError:
        return

    open_ms = open_ts * 1000
    close_ms = open_ms + 300_000

    if df.empty:
        return

    assets = df["asset_id"].unique()
    if len(assets) < 2:
        return

    asset_a, asset_b = assets[0], assets[1]

    for buy_price in BUY_PRICES:
        r = results[buy_price]
        r["n"] += 1

        if source_label == "raw":
            # Check if SELL trades exist at <= our buy price (someone selling into our bid)
            a_sells = df[(df["asset_id"] == asset_a) & (df["side"] == "SELL") & (df["trade_price"] <= buy_price)]
            b_sells = df[(df["asset_id"] == asset_b) & (df["side"] == "SELL") & (df["trade_price"] <= buy_price)]
            a_filled = len(a_sells) > 0
            b_filled = len(b_sells) > 0
            if a_filled and b_filled:
                a_time = a_sells["exchange_timestamp"].min()
                b_time = b_sells["exchange_timestamp"].min()
                fill_time_s = (max(a_time, b_time) - open_ms) / 1000
                r["fill_times"].append(fill_time_s)
        else:
            # Telonex: check if ask_price_1 ever drops to our buy level
            # Actually for buys, we need to check if there are sellers at our price.
            # In book data: if ask_price_1 <= buy_price, our resting buy would fill.
            # But we have two assets interleaved — split by asset_id
            a_df = df[df["asset_id"] == asset_a]
            b_df = df[df["asset_id"] == asset_b]
            a_filled = (a_df["ask_price_1"] <= buy_price).any() if len(a_df) > 0 and "ask_price_1" in a_df.columns else False
            b_filled = (b_df["ask_price_1"] <= buy_price).any() if len(b_df) > 0 and "ask_price_1" in b_df.columns else False

        if a_filled and b_filled:
            # Both bought — paid 2*buy_price, get $1 back guaranteed
            pnl = 1.0 - 2 * buy_price
            r["both_fill"] += 1
        elif a_filled:
            # Bought A at buy_price
            a_won = (winner == asset_a)
            pnl = (1.0 if a_won else 0.0) - buy_price
            if a_won:
                r["one_fill_win"] += 1
            else:
                r["one_fill_lose"] += 1
        elif b_filled:
            b_won = (winner == asset_b)
            pnl = (1.0 if b_won else 0.0) - buy_price
            if b_won:
                r["one_fill_win"] += 1
            else:
                r["one_fill_lose"] += 1
        else:
            pnl = 0.0
            r["neither"] += 1

        r["total_pnl"] += pnl


# Process raw trade events
trade_files = sorted(TRADE_DIR.glob("btc-updown-5m-*.parquet"))
print(f"Raw trade event files: {len(trade_files)}")
raw_slugs = set()

for i, tf in enumerate(trade_files):
    slug = tf.stem
    raw_slugs.add(slug)
    try:
        open_ts = int(slug.split("-")[-1])
    except ValueError:
        continue
    open_ms = open_ts * 1000
    close_ms = open_ms + 300_000

    try:
        df = pd.read_parquet(tf)
    except Exception:
        continue

    df = df[(df["exchange_timestamp"] >= open_ms) & (df["exchange_timestamp"] < close_ms)]
    process_market(slug, df, "raw")

    if (i + 1) % 1000 == 0:
        print(f"  raw: {i+1}/{len(trade_files)}...", file=sys.stderr)

# Process Telonex book snapshots (only those not already in raw)
telonex_files = sorted(TELONEX_DIR.glob("btc-updown-5m-*.parquet"))
print(f"Telonex book snapshot files: {len(telonex_files)}")
telonex_added = 0

for i, tf in enumerate(telonex_files):
    slug = tf.stem
    if slug in raw_slugs:
        continue  # already processed from raw

    try:
        df = pd.read_parquet(tf)
    except Exception:
        continue

    if df.empty:
        continue

    process_market(slug, df, "telonex")
    telonex_added += 1

    if (i + 1) % 1000 == 0:
        print(f"  telonex: {i+1}/{len(telonex_files)}...", file=sys.stderr)

print(f"Telonex-only markets added: {telonex_added}")

print("\n" + "=" * 80)
print("BUY BOTH SIDES — BACKTEST RESULTS (per market)")
print("=" * 80)

for bp in BUY_PRICES:
    r = results[bp]
    n = r["n"]
    if n == 0:
        continue

    both_pct = r["both_fill"] / n * 100
    one_win_pct = r["one_fill_win"] / n * 100
    one_lose_pct = r["one_fill_lose"] / n * 100
    neither_pct = r["neither"] / n * 100
    avg_pnl = r["total_pnl"] / n

    print(f"\n--- Buy @ {bp:.2f} ---")
    print(f"  Markets: {n}")
    print(f"  Both fill:       {r['both_fill']:>5} ({both_pct:>5.1f}%)  PnL each: +${1 - 2*bp:.2f}")
    print(f"  One fill + win:  {r['one_fill_win']:>5} ({one_win_pct:>5.1f}%)  PnL each: +${1-bp:.2f}")
    print(f"  One fill + lose: {r['one_fill_lose']:>5} ({one_lose_pct:>5.1f}%)  PnL each: -${bp:.2f}")
    print(f"  Neither fill:    {r['neither']:>5} ({neither_pct:>5.1f}%)  PnL each: $0.00")
    print(f"  Avg PnL/market:  ${avg_pnl:+.4f}")
    print(f"  Total PnL ({n} mkts): ${r['total_pnl']:+.2f}")

    if r["fill_times"]:
        ft = np.array(r["fill_times"])
        print(f"  Both-fill time: median {np.median(ft):.0f}s, mean {np.mean(ft):.0f}s")

    # Capital efficiency: how much capital tied up per market?
    # Both fill: 2*bp deployed. One fill: bp deployed. Neither: 0.
    deployed = r["both_fill"] * 2 * bp + (r["one_fill_win"] + r["one_fill_lose"]) * bp
    if deployed > 0:
        roi = r["total_pnl"] / deployed * 100
        print(f"  ROI on deployed capital: {roi:+.2f}%")
