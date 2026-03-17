"""
Backtest: Buy both sides above 0.50.

Place resting buy orders at buy_price (e.g. 0.52) for BOTH tokens.
In a trending market, only the winning token's price reaches 0.52+.
The losing token drops below 0.50 and never hits our buy.

Fill logic: token's max trade price >= buy_price means our buy would fill.
(Price went up to our level, we got filled.)

Vectorized — loads all trades per file in one shot.
"""

import json
import sys
from pathlib import Path

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

BUY_PRICES = [0.51, 0.52, 0.53, 0.54, 0.55, 0.57, 0.60, 0.65]

# Collect per-market results as arrays for vectorized analysis
market_data = []  # list of (slug, asset_a, asset_b, winner, max_price_a, max_price_b)


def extract_max_prices_raw(tf):
    slug = tf.stem
    winner = resolutions.get(slug)
    if not winner:
        return None
    try:
        open_ts = int(slug.split("-")[-1])
    except ValueError:
        return None

    open_ms = open_ts * 1000
    close_ms = open_ms + 300_000

    try:
        df = pd.read_parquet(tf, columns=["exchange_timestamp", "asset_id", "trade_price"])
    except Exception:
        return None

    df = df[(df["exchange_timestamp"] >= open_ms) & (df["exchange_timestamp"] < close_ms)]
    if df.empty:
        return None

    # Max trade price per asset — fully vectorized
    max_prices = df.groupby("asset_id")["trade_price"].max()
    if len(max_prices) < 2:
        return None

    assets = max_prices.index.tolist()
    return (slug, assets[0], assets[1], winner, max_prices.iloc[0], max_prices.iloc[1])


def extract_max_prices_telonex(tf):
    slug = tf.stem
    winner = resolutions.get(slug)
    if not winner:
        return None

    try:
        df = pd.read_parquet(tf, columns=["asset_id", "ask_price_1", "bid_price_1"])
    except Exception:
        return None

    if df.empty:
        return None

    # In book snapshots, the highest price a token trades at is approximated
    # by max(ask_price_1) — the highest ask seen means the price was at least that high.
    # Actually better: max of mid_price or bid_price_1 for "price reached this level"
    # Use bid_price_1: if someone's bidding 0.55, the price is at least 0.55
    max_prices = df.groupby("asset_id")["bid_price_1"].max()
    if len(max_prices) < 2:
        return None

    assets = max_prices.index.tolist()
    return (slug, assets[0], assets[1], winner, max_prices.iloc[0], max_prices.iloc[1])


# Process raw trade events
trade_files = sorted(TRADE_DIR.glob("btc-updown-5m-*.parquet"))
print(f"Raw trade event files: {len(trade_files)}")
raw_slugs = set()

for i, tf in enumerate(trade_files):
    raw_slugs.add(tf.stem)
    result = extract_max_prices_raw(tf)
    if result:
        market_data.append(result)
    if (i + 1) % 1000 == 0:
        print(f"  raw: {i+1}/{len(trade_files)}...", file=sys.stderr)

# Process Telonex (non-overlapping only)
telonex_files = sorted(TELONEX_DIR.glob("btc-updown-5m-*.parquet"))
print(f"Telonex files: {len(telonex_files)}")

for i, tf in enumerate(telonex_files):
    if tf.stem in raw_slugs:
        continue
    result = extract_max_prices_telonex(tf)
    if result:
        market_data.append(result)
    if (i + 1) % 1000 == 0:
        print(f"  telonex: {i+1}/{len(telonex_files)}...", file=sys.stderr)

print(f"\nTotal markets with data: {len(market_data)}")

# Convert to numpy for vectorized analysis
slugs = [m[0] for m in market_data]
asset_a = np.array([m[1] for m in market_data])
asset_b = np.array([m[2] for m in market_data])
winners = np.array([m[3] for m in market_data])
max_a = np.array([m[4] for m in market_data], dtype=np.float64)
max_b = np.array([m[5] for m in market_data], dtype=np.float64)

a_wins = (winners == asset_a)
b_wins = (winners == asset_b)

print("\n" + "=" * 80)
print("BUY ABOVE 0.50 — BACKTEST RESULTS")
print("=" * 80)

for bp in BUY_PRICES:
    a_fills = max_a >= bp
    b_fills = max_b >= bp

    both = a_fills & b_fills
    only_a = a_fills & ~b_fills
    only_b = ~a_fills & b_fills
    neither = ~a_fills & ~b_fills

    n = len(market_data)
    n_both = both.sum()
    n_only_a = only_a.sum()
    n_only_b = only_b.sum()
    n_neither = neither.sum()

    # PnL calculations
    # Both fill: paid 2*bp, get $1 back → PnL = 1 - 2*bp
    pnl_both = n_both * (1.0 - 2 * bp)

    # Only A fills: paid bp, get $1 if A wins, $0 if A loses
    pnl_only_a = (only_a & a_wins).sum() * (1.0 - bp) + (only_a & ~a_wins).sum() * (-bp)

    # Only B fills: paid bp, get $1 if B wins, $0 if B loses
    pnl_only_b = (only_b & b_wins).sum() * (1.0 - bp) + (only_b & ~b_wins).sum() * (-bp)

    # Neither: $0
    total_pnl = pnl_both + pnl_only_a + pnl_only_b

    # Win/loss for single fills
    single_fill_wins = (only_a & a_wins).sum() + (only_b & b_wins).sum()
    single_fill_losses = (only_a & ~a_wins).sum() + (only_b & ~b_wins).sum()
    single_total = single_fill_wins + single_fill_losses

    avg_pnl = total_pnl / n

    print(f"\n--- Buy @ {bp:.2f} ---")
    print(f"  Markets: {n}")
    print(f"  Both fill:       {n_both:>5} ({n_both/n*100:>5.1f}%)  PnL each: ${1-2*bp:+.2f}")
    print(f"  Only one fills:  {single_total:>5} ({single_total/n*100:>5.1f}%)")
    if single_total > 0:
        print(f"    → winner fills: {single_fill_wins:>5} ({single_fill_wins/single_total*100:>5.1f}% of single fills)  PnL: +${1-bp:.2f}")
        print(f"    → loser fills:  {single_fill_losses:>5} ({single_fill_losses/single_total*100:>5.1f}% of single fills)  PnL: -${bp:.2f}")
    print(f"  Neither fills:   {n_neither:>5} ({n_neither/n*100:>5.1f}%)")
    print(f"  Avg PnL/market:  ${avg_pnl:+.4f}")
    print(f"  Total PnL:       ${total_pnl:+.2f}")

    # Capital deployed
    deployed = n_both * 2 * bp + single_total * bp
    if deployed > 0:
        print(f"  Capital deployed: ${deployed:.0f}, ROI: {total_pnl/deployed*100:+.2f}%")

    # What if we cancel the second buy after the first fills?
    # Then "both fill" becomes "one fill" — we only buy the first one to reach bp
    # The first to reach bp is more likely the winner (it's going up)
    # This is harder to backtest without timestamps, but let's estimate
