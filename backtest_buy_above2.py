"""
Backtest: Buy both sides above 0.50 (corrected).

Place stop-buy orders for both tokens at buy_price.
Fill condition: BUY trade at >= buy_price (someone paid that price, confirming demand).
Same criterion as the sell backtest — consistent "price reached X" definition.

When only one side reaches buy_price, it's almost always the winner.
When both reach it, we buy both and lose the spread above $1.

Uses raw trade_events only (Telonex has no trade data).
"""

import json
import sys
from pathlib import Path

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

BUY_PRICES = [0.51, 0.52, 0.53, 0.54, 0.55, 0.57, 0.60, 0.65]

# Collect vectorized data: for each market, max BUY-trade price per asset
market_rows = []

trade_files = sorted(TRADE_DIR.glob("btc-updown-5m-*.parquet"))
print(f"Trade event files: {len(trade_files)}")

for i, tf in enumerate(trade_files):
    slug = tf.stem
    winner = resolutions.get(slug)
    if not winner:
        continue

    try:
        open_ts = int(slug.split("-")[-1])
    except ValueError:
        continue

    open_ms = open_ts * 1000
    close_ms = open_ms + 300_000

    try:
        df = pd.read_parquet(tf, columns=["exchange_timestamp", "asset_id", "side", "trade_price"])
    except Exception:
        continue

    df = df[(df["exchange_timestamp"] >= open_ms) & (df["exchange_timestamp"] < close_ms)]
    if df.empty:
        continue

    # Only BUY trades — someone actively paying this price
    buys = df[df["side"] == "BUY"]
    if buys.empty:
        continue

    max_buy = buys.groupby("asset_id")["trade_price"].max()
    if len(max_buy) < 2:
        continue

    assets = max_buy.index.tolist()
    market_rows.append((slug, assets[0], assets[1], winner, max_buy.iloc[0], max_buy.iloc[1]))

    if (i + 1) % 1000 == 0:
        print(f"  {i+1}/{len(trade_files)}...", file=sys.stderr)

print(f"Markets with data: {len(market_rows)}")

# Vectorize
asset_a = np.array([r[1] for r in market_rows])
asset_b = np.array([r[2] for r in market_rows])
winners = np.array([r[3] for r in market_rows])
max_buy_a = np.array([r[4] for r in market_rows])
max_buy_b = np.array([r[5] for r in market_rows])

a_wins = winners == asset_a
b_wins = winners == asset_b
n = len(market_rows)

print(f"\n{'='*80}")
print("BUY ABOVE 0.50 — CORRECTED (BUY-trade criterion)")
print(f"{'='*80}")

for bp in BUY_PRICES:
    a_fills = max_buy_a >= bp
    b_fills = max_buy_b >= bp

    both = a_fills & b_fills
    only_a = a_fills & ~b_fills
    only_b = ~a_fills & b_fills
    neither = ~a_fills & ~b_fills

    n_both = both.sum()
    n_only_a = only_a.sum()
    n_only_b = only_b.sum()
    n_neither = neither.sum()

    # PnL
    # Both fill: paid 2*bp, get $1 back
    pnl_both = n_both * (1.0 - 2 * bp)
    # Only A fills: paid bp, get $1 if A wins
    pnl_only_a = (only_a & a_wins).sum() * (1.0 - bp) + (only_a & ~a_wins).sum() * (-bp)
    # Only B fills
    pnl_only_b = (only_b & b_wins).sum() * (1.0 - bp) + (only_b & ~b_wins).sum() * (-bp)
    total_pnl = pnl_both + pnl_only_a + pnl_only_b

    single_wins = (only_a & a_wins).sum() + (only_b & b_wins).sum()
    single_losses = (only_a & ~a_wins).sum() + (only_b & ~b_wins).sum()
    single_total = single_wins + single_losses

    avg_pnl = total_pnl / n

    print(f"\n--- Buy @ {bp:.2f} ---")
    print(f"  Markets: {n}")
    print(f"  Both fill:       {n_both:>5} ({n_both/n*100:>5.1f}%)  PnL each: ${1-2*bp:+.2f}")
    print(f"  Only one fills:  {single_total:>5} ({single_total/n*100:>5.1f}%)")
    if single_total > 0:
        print(f"    → winner fills: {single_wins:>5} ({single_wins/single_total*100:>5.1f}%)  PnL: +${1-bp:.2f}")
        print(f"    → loser fills:  {single_losses:>5} ({single_losses/single_total*100:>5.1f}%)  PnL: -${bp:.2f}")
    print(f"  Neither fills:   {n_neither:>5} ({n_neither/n*100:>5.1f}%)")
    print(f"  Avg PnL/market:  ${avg_pnl:+.4f}")
    print(f"  Total PnL:       ${total_pnl:+.2f}")

    deployed = n_both * 2 * bp + single_total * bp
    if deployed > 0:
        print(f"  Capital deployed: ${deployed:.0f}, ROI: {total_pnl/deployed*100:+.2f}%")
