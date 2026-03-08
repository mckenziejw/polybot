"""
Backtest: Fade the dumb money.

For each 5-min market, measure net dollar flow in the first N seconds.
If retail is net buying Up, we buy Down (and vice versa).
Check resolution rates.

Trade flow signal: sum(size * price) for BUY trades minus sum(size * price) for SELL trades,
per asset. Net positive flow into an asset = dumb money buying it.
We fade by buying the OTHER asset.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

TRADE_DIR = Path("data/trade_events")
BOOK_DIR = Path("data/book_snapshots")
RESOLUTION_CACHE = Path("data/resolution_cache.json")

# Load resolutions
with open(RESOLUTION_CACHE) as f:
    raw_cache = json.load(f)

resolutions = {}
for cid, info in raw_cache.items():
    winner = info.get("winningTokenId") or info.get("winning_token", "")
    if winner:
        resolutions[cid] = winner

print(f"Loaded {len(resolutions)} resolutions")

# Get all trade event files
trade_files = sorted(TRADE_DIR.glob("btc-updown-5m-*.parquet"))
print(f"Trade event files: {len(trade_files)}")

# We need to know which asset_id is "Up" vs "Down" for each market.
# Strategy: the asset trading near the higher price early on is the "favorite".
# But actually we don't need labels — we just need to know which asset won.
# We fade whichever asset has more net buying flow.

# Analysis parameters
LOOKBACK_WINDOWS = [15, 30, 60, 90, 120]  # seconds into the market

results_by_window = {w: [] for w in LOOKBACK_WINDOWS}

processed = 0
skipped_no_resolution = 0
skipped_no_trades = 0

for tf in trade_files:
    slug = tf.stem
    # Extract market open timestamp from slug
    try:
        open_ts = int(slug.split("-")[-1])
    except ValueError:
        continue

    open_ms = open_ts * 1000

    # Read trades
    try:
        df = pd.read_parquet(tf)
    except Exception:
        continue

    if df.empty:
        skipped_no_trades += 1
        continue

    # Look up winner by slug
    winner = resolutions.get(slug)
    if not winner:
        skipped_no_resolution += 1
        continue

    # Get the two asset IDs
    assets = df["asset_id"].unique()
    if len(assets) < 2:
        skipped_no_trades += 1
        continue

    asset_a, asset_b = assets[0], assets[1]

    # For each lookback window
    for window_s in LOOKBACK_WINDOWS:
        window_ms = window_s * 1000
        cutoff = open_ms + window_ms

        window_df = df[df["exchange_timestamp"] < cutoff]
        if window_df.empty:
            continue

        # Net dollar flow per asset: BUY adds flow, SELL subtracts
        # (from taker perspective: BUY = someone aggressively buying, SELL = someone aggressively selling)
        flow = {}
        for asset in [asset_a, asset_b]:
            asset_trades = window_df[window_df["asset_id"] == asset]
            buys = asset_trades[asset_trades["side"] == "BUY"]
            sells = asset_trades[asset_trades["side"] == "SELL"]
            buy_flow = (buys["size"] * buys["trade_price"]).sum()
            sell_flow = (sells["size"] * sells["trade_price"]).sum()
            flow[asset] = buy_flow - sell_flow

        # Which asset has more net buying pressure?
        if abs(flow[asset_a] - flow[asset_b]) < 1.0:
            # Flow too close to call — skip
            continue

        if flow[asset_a] > flow[asset_b]:
            dumb_money_asset = asset_a
            fade_asset = asset_b
        else:
            dumb_money_asset = asset_b
            fade_asset = asset_a

        # Did the fade asset win?
        fade_won = (winner == fade_asset)

        # What price would we enter at? Use the book snapshot at the signal time.
        # For simplicity, use the last trade price of the fade asset in the window
        fade_trades = window_df[window_df["asset_id"] == fade_asset]
        if not fade_trades.empty:
            entry_price = fade_trades["trade_price"].iloc[-1]
        else:
            # No trades for fade asset in window — use complement
            dumb_trades = window_df[window_df["asset_id"] == dumb_money_asset]
            if dumb_trades.empty:
                continue
            entry_price = 1.0 - dumb_trades["trade_price"].iloc[-1]

        # Only trade if entry price is in a reasonable range
        if entry_price < 0.10 or entry_price > 0.90:
            continue

        net_flow_diff = abs(flow[asset_a] - flow[asset_b])

        results_by_window[window_s].append({
            "slug": slug,
            "fade_won": fade_won,
            "entry_price": entry_price,
            "flow_diff": net_flow_diff,
            "pnl": (1.0 - entry_price) if fade_won else -entry_price,
        })

    processed += 1
    if processed % 1000 == 0:
        print(f"  {processed}/{len(trade_files)}...", file=sys.stderr)

print(f"\nProcessed: {processed}, No resolution: {skipped_no_resolution}, No trades: {skipped_no_trades}")

print("\n" + "=" * 70)
print("FADE DUMB MONEY — BACKTEST RESULTS")
print("=" * 70)

for window_s in LOOKBACK_WINDOWS:
    res = results_by_window[window_s]
    if not res:
        continue

    df_res = pd.DataFrame(res)
    n = len(df_res)
    wins = df_res["fade_won"].sum()
    wr = wins / n * 100
    avg_entry = df_res["entry_price"].mean()
    total_pnl = df_res["pnl"].sum()
    avg_pnl = df_res["pnl"].mean()
    ev_per_dollar = avg_pnl / avg_entry if avg_entry > 0 else 0

    print(f"\n--- Lookback: {window_s}s ---")
    print(f"  Signals: {n}, Wins: {wins}, WR: {wr:.1f}%")
    print(f"  Avg entry: {avg_entry:.3f}")
    print(f"  Avg PnL/trade: ${avg_pnl:.4f}")
    print(f"  EV per $1: ${ev_per_dollar:.4f}")
    print(f"  Total PnL ($1/trade): ${total_pnl:.2f}")

    # Breakdown by flow strength
    quartiles = df_res["flow_diff"].quantile([0.25, 0.5, 0.75])
    print(f"  Flow diff quartiles: {quartiles.values}")

    # WR by flow strength tercile
    tercile_edges = df_res["flow_diff"].quantile([0, 0.33, 0.66, 1.0]).values
    for i in range(3):
        lo, hi = tercile_edges[i], tercile_edges[i + 1]
        mask = (df_res["flow_diff"] >= lo) & (df_res["flow_diff"] <= hi)
        sub = df_res[mask]
        if len(sub) > 0:
            sub_wr = sub["fade_won"].mean() * 100
            sub_avg_entry = sub["entry_price"].mean()
            label = ["weak", "medium", "strong"][i]
            print(f"    {label} flow (${lo:.0f}-${hi:.0f}): {len(sub)} trades, {sub_wr:.1f}% WR, avg entry {sub_avg_entry:.3f}")

    # WR by entry price bucket
    print(f"  By entry price:")
    for lo, hi in [(0.1, 0.3), (0.3, 0.45), (0.45, 0.55), (0.55, 0.7), (0.7, 0.9)]:
        mask = (df_res["entry_price"] >= lo) & (df_res["entry_price"] < hi)
        sub = df_res[mask]
        if len(sub) > 10:
            sub_wr = sub["fade_won"].mean() * 100
            print(f"    [{lo:.2f}-{hi:.2f}): {len(sub)} trades, {sub_wr:.1f}% WR")
