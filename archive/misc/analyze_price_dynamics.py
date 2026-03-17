"""
Analyze price dynamics across different timeframes.

Key questions:
1. How much do prices oscillate vs trend directionally?
2. What fraction of time does the mid price spend near 0.50?
3. How many times does the price cross 0.50?
4. What's the spread distribution?

These tell us whether market making is viable in each timeframe.

Usage:
  python analyze_price_dynamics.py data/telonex_btc_15m_book_snapshots
  python analyze_price_dynamics.py data/telonex_btc_4h_book_snapshots
  python analyze_price_dynamics.py data/telonex_book_snapshots  # existing 5m data
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

if len(sys.argv) < 2:
    print("Usage: python analyze_price_dynamics.py <data_dir>")
    sys.exit(1)

DATA_DIR = Path(sys.argv[1])
files = sorted(DATA_DIR.glob("*.parquet"))
print(f"Data dir: {DATA_DIR}")
print(f"Files: {len(files)}")

if not files:
    print("No files found!")
    sys.exit(1)

# Collect stats per market
stats = []

for i, f in enumerate(files):
    try:
        df = pd.read_parquet(f, columns=["exchange_timestamp", "asset_id", "mid_price", "spread",
                                          "bid_price_1", "ask_price_1"])
    except Exception:
        continue

    if df.empty or len(df) < 10:
        continue

    # Use one asset only (the "Up" token — pick the one with higher avg mid)
    assets = df["asset_id"].unique()
    if len(assets) < 2:
        continue

    # Pick the asset — use first one, doesn't matter which for dynamics
    a0 = df[df["asset_id"] == assets[0]]
    a1 = df[df["asset_id"] == assets[1]]

    # Use whichever has more data
    asset_df = a0 if len(a0) >= len(a1) else a1

    if len(asset_df) < 5:
        continue

    mid = asset_df["mid_price"].values
    spread = asset_df["spread"].values
    ts = asset_df["exchange_timestamp"].values

    # Duration in seconds
    duration_s = (ts[-1] - ts[0]) / 1000 if len(ts) > 1 else 0

    # Price range
    price_min = mid.min()
    price_max = mid.max()
    price_range = price_max - price_min

    # Time near 0.50 (within 0.05)
    near_50 = np.mean((mid >= 0.45) & (mid <= 0.55))

    # Number of times mid crosses 0.50
    above = mid > 0.50
    crossings = np.sum(np.diff(above.astype(int)) != 0)

    # Average spread
    valid_spread = spread[spread > 0]
    avg_spread = valid_spread.mean() if len(valid_spread) > 0 else 0

    # Final price (how directional was it?)
    final_mid = mid[-1]
    initial_mid = mid[0]

    # Max drawdown from starting position (either direction)
    max_up = price_max - initial_mid
    max_down = initial_mid - price_min

    stats.append({
        "slug": f.stem,
        "n_ticks": len(asset_df),
        "duration_s": duration_s,
        "initial_mid": initial_mid,
        "final_mid": final_mid,
        "price_min": price_min,
        "price_max": price_max,
        "price_range": price_range,
        "pct_near_50": near_50,
        "crossings_50": crossings,
        "avg_spread": avg_spread,
    })

    if (i + 1) % 2000 == 0:
        print(f"  {i+1}/{len(files)}...", file=sys.stderr)

df_stats = pd.DataFrame(stats)
n = len(df_stats)
print(f"\nAnalyzed {n} markets\n")

print("=" * 70)
print("PRICE DYNAMICS SUMMARY")
print("=" * 70)

print(f"\nTicks per market: mean={df_stats['n_ticks'].mean():.0f}, median={df_stats['n_ticks'].median():.0f}")
print(f"Duration: mean={df_stats['duration_s'].mean():.0f}s, median={df_stats['duration_s'].median():.0f}s")

print(f"\n--- Price Range ---")
print(f"  Mean range:  {df_stats['price_range'].mean():.4f}")
print(f"  Median range: {df_stats['price_range'].median():.4f}")
print(f"  P25/P75:     {df_stats['price_range'].quantile(0.25):.4f} / {df_stats['price_range'].quantile(0.75):.4f}")

print(f"\n--- Time Near 0.50 (within ±0.05) ---")
print(f"  Mean:   {df_stats['pct_near_50'].mean()*100:.1f}%")
print(f"  Median: {df_stats['pct_near_50'].median()*100:.1f}%")
pct_mostly_near = (df_stats["pct_near_50"] > 0.5).mean() * 100
print(f"  Markets spending >50% of time near 0.50: {pct_mostly_near:.1f}%")

print(f"\n--- 0.50 Crossings ---")
print(f"  Mean:   {df_stats['crossings_50'].mean():.1f}")
print(f"  Median: {df_stats['crossings_50'].median():.0f}")
print(f"  Markets with 0 crossings: {(df_stats['crossings_50'] == 0).mean()*100:.1f}%")
print(f"  Markets with 5+ crossings: {(df_stats['crossings_50'] >= 5).mean()*100:.1f}%")
print(f"  Markets with 10+ crossings: {(df_stats['crossings_50'] >= 10).mean()*100:.1f}%")

print(f"\n--- Spread ---")
print(f"  Mean:   {df_stats['avg_spread'].mean():.4f}")
print(f"  Median: {df_stats['avg_spread'].median():.4f}")

print(f"\n--- Directionality ---")
# How often does price end up far from 0.50?
final = df_stats["final_mid"]
print(f"  Final mid <0.20 or >0.80: {((final < 0.20) | (final > 0.80)).mean()*100:.1f}%")
print(f"  Final mid <0.10 or >0.90: {((final < 0.10) | (final > 0.90)).mean()*100:.1f}%")
print(f"  Final mid 0.40-0.60:       {((final >= 0.40) & (final <= 0.60)).mean()*100:.1f}%")

# Market making viability score:
# High crossings + high time near 0.50 + wide spread = good for MM
print(f"\n--- Market Making Viability ---")
viable = (df_stats["crossings_50"] >= 3) & (df_stats["pct_near_50"] > 0.3) & (df_stats["avg_spread"] >= 0.02)
print(f"  Markets with 3+ crossings, >30% near 0.50, spread>=2c: {viable.sum()} ({viable.mean()*100:.1f}%)")

# Distribution of crossings
print(f"\n--- Crossing Distribution ---")
for threshold in [0, 1, 2, 3, 5, 10, 20, 50]:
    pct = (df_stats["crossings_50"] >= threshold).mean() * 100
    print(f"  >= {threshold:>2} crossings: {pct:>5.1f}%")
