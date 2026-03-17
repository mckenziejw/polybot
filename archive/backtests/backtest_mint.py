"""
Backtest: Mint & Capture spread.

At market open, mint N token pairs ($1 each → 1 Up + 1 Down).
Place resting sells for both tokens at `sell_price` (a few ticks above 0.50).
A sell fills when a BUY trade occurs at >= our sell_price for that asset.

PnL scenarios per pair:
  Both fill:     +2*sell_price - 1.00 (guaranteed profit)
  One fill:      +sell_price + (1.0 if other wins else 0.0) - 1.00
  Neither fill:  +1.00 - 1.00 = 0 (one token always wins, get $1 back)
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

TRADE_DIR = Path("data/trade_events")
RESOLUTION_CACHE = Path("data/resolution_cache.json")

with open(RESOLUTION_CACHE) as f:
    raw_cache = json.load(f)

resolutions = {}
for slug, info in raw_cache.items():
    winner = info.get("winningTokenId") or info.get("winning_token", "")
    if winner:
        resolutions[slug] = winner

print(f"Loaded {len(resolutions)} resolutions")

trade_files = sorted(TRADE_DIR.glob("btc-updown-5m-*.parquet"))
print(f"Trade event files: {len(trade_files)}")

SELL_PRICES = [0.51, 0.52, 0.53, 0.54, 0.55, 0.57, 0.60]

results = {sp: {"both_fill": 0, "one_fill_win": 0, "one_fill_lose": 0,
                "neither": 0, "total_pnl": 0.0, "n": 0,
                "fill_times": []} for sp in SELL_PRICES}

processed = 0

for tf in trade_files:
    slug = tf.stem
    try:
        open_ts = int(slug.split("-")[-1])
    except ValueError:
        continue

    open_ms = open_ts * 1000
    close_ms = open_ms + 300_000  # 5 minutes

    winner = resolutions.get(slug)
    if not winner:
        continue

    try:
        df = pd.read_parquet(tf)
    except Exception:
        continue

    if df.empty:
        continue

    # Filter to market window
    df = df[(df["exchange_timestamp"] >= open_ms) & (df["exchange_timestamp"] < close_ms)]
    if df.empty:
        continue

    assets = df["asset_id"].unique()
    if len(assets) < 2:
        continue

    asset_a, asset_b = assets[0], assets[1]

    for sell_price in SELL_PRICES:
        r = results[sell_price]
        r["n"] += 1

        # Check if each asset has a BUY trade at >= sell_price
        a_buys = df[(df["asset_id"] == asset_a) & (df["side"] == "BUY") & (df["trade_price"] >= sell_price)]
        b_buys = df[(df["asset_id"] == asset_b) & (df["side"] == "BUY") & (df["trade_price"] >= sell_price)]

        a_filled = len(a_buys) > 0
        b_filled = len(b_buys) > 0

        if a_filled and b_filled:
            # Both sides sold — guaranteed profit
            pnl = 2 * sell_price - 1.0
            r["both_fill"] += 1
            # Time to complete (when second fill happens)
            a_time = a_buys["exchange_timestamp"].min()
            b_time = b_buys["exchange_timestamp"].min()
            fill_time_s = (max(a_time, b_time) - open_ms) / 1000
            r["fill_times"].append(fill_time_s)
        elif a_filled:
            # Only A sold — hold B to resolution
            b_won = (winner == asset_b)
            pnl = sell_price + (1.0 if b_won else 0.0) - 1.0
            if b_won:
                r["one_fill_win"] += 1
            else:
                r["one_fill_lose"] += 1
        elif b_filled:
            # Only B sold — hold A to resolution
            a_won = (winner == asset_a)
            pnl = sell_price + (1.0 if a_won else 0.0) - 1.0
            if a_won:
                r["one_fill_win"] += 1
            else:
                r["one_fill_lose"] += 1
        else:
            # Neither sold — one token wins, get $1 back, net 0
            pnl = 0.0
            r["neither"] += 1

        r["total_pnl"] += pnl

    processed += 1
    if processed % 1000 == 0:
        print(f"  {processed}/{len(trade_files)}...", file=sys.stderr)

print(f"\nProcessed: {processed} markets")

print("\n" + "=" * 80)
print("MINT & CAPTURE — BACKTEST RESULTS (per 1 token pair per market)")
print("=" * 80)

for sp in SELL_PRICES:
    r = results[sp]
    n = r["n"]
    if n == 0:
        continue

    both_pct = r["both_fill"] / n * 100
    one_win_pct = r["one_fill_win"] / n * 100
    one_lose_pct = r["one_fill_lose"] / n * 100
    neither_pct = r["neither"] / n * 100
    avg_pnl = r["total_pnl"] / n

    print(f"\n--- Sell @ {sp:.2f} ---")
    print(f"  Markets: {n}")
    print(f"  Both fill:       {r['both_fill']:>5} ({both_pct:>5.1f}%)  PnL each: +${2*sp - 1:.2f}")
    print(f"  One fill + win:  {r['one_fill_win']:>5} ({one_win_pct:>5.1f}%)  PnL each: +${sp:.2f}")
    print(f"  One fill + lose: {r['one_fill_lose']:>5} ({one_lose_pct:>5.1f}%)  PnL each: -${1-sp:.2f}")
    print(f"  Neither fill:    {r['neither']:>5} ({neither_pct:>5.1f}%)  PnL each: $0.00")
    print(f"  Avg PnL/pair:    ${avg_pnl:+.4f}")
    print(f"  Total PnL ({n} pairs): ${r['total_pnl']:+.2f}")

    if r["fill_times"]:
        ft = np.array(r["fill_times"])
        print(f"  Both-fill time: median {np.median(ft):.0f}s, mean {np.mean(ft):.0f}s, p90 {np.percentile(ft, 90):.0f}s")

# Gas cost analysis
print("\n" + "=" * 80)
print("GAS COST SENSITIVITY")
print("=" * 80)
print("Minting requires splitPosition() on CTF — ~150k gas on Polygon")
print("At ~30 gwei, that's ~0.0045 MATIC ≈ $0.003 per mint")
print("Selling via CLOB is off-chain (no gas for limit orders)")
print("Redemption: redeemPositions() ~100k gas ≈ $0.002")
print("Total gas per round-trip: ~$0.005")
print()
for sp in SELL_PRICES:
    r = results[sp]
    n = r["n"]
    if n == 0:
        continue
    avg_pnl = r["total_pnl"] / n
    net_after_gas = avg_pnl - 0.005
    print(f"  Sell @ {sp:.2f}: avg PnL ${avg_pnl:+.4f}, after gas ${net_after_gas:+.4f}")
