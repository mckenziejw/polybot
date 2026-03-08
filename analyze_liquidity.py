#!/usr/bin/env python3
"""
Orderbook liquidity analysis at t=120s for position sizing.

Answers: how large an order can reliably get filled at the best ask
in BTC Up/Down 5-minute markets?
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

DATA_DIR = Path("data/telonex_book_snapshots")
N_LEVELS = 5  # parquet has 5 levels


def main():
    # Load a sample of parquet files (not all 6k+ — that's ~50GB in memory)
    all_files = sorted(DATA_DIR.glob("*.parquet"), key=lambda f: f.name)
    print(f"Found {len(all_files)} parquet files total")

    # Use most recent 500 markets (representative of current liquidity)
    N_SAMPLE = 500
    files = all_files[-N_SAMPLE:]
    print(f"Loading {len(files)} most recent files...")

    dfs = []
    for f in files:
        dfs.append(pd.read_parquet(f))
    df = pd.concat(dfs, ignore_index=True)
    print(f"Total snapshots: {len(df):,}")

    # Parse market open time from slug
    df["market_open_ms"] = df["slug"].str.extract(r"(\d+)$").astype(float) * 1000
    df["t_since_open_s"] = (df["exchange_timestamp"] - df["market_open_ms"]) / 1000

    # Filter to leader token snapshots around t=110-130s
    leader = df[df["mid_price"] >= 0.50].copy()
    window = leader[(leader["t_since_open_s"] >= 110) & (leader["t_since_open_s"] <= 130)]
    print(f"Leader snapshots in [110s, 130s]: {len(window):,}")

    # Pick closest to t=120s per market
    window["t_dist"] = abs(window["t_since_open_s"] - 120)
    best = window.loc[window.groupby("slug")["t_dist"].idxmin()].copy()
    n_markets = len(best)
    print(f"Unique markets with leader snapshot near 120s: {n_markets:,}")

    # Compute ask-side depth metrics
    best["ask_depth_1_tokens"] = best["ask_size_1"]
    best["ask_depth_1_dollars"] = best["ask_size_1"] * best["ask_price_1"]

    # Cumulative depth at levels 1-3 and 1-5
    for n_levels in [3, 5]:
        tokens = np.zeros(len(best))
        dollars = np.zeros(len(best))
        for lv in range(1, n_levels + 1):
            sz = best[f"ask_size_{lv}"].fillna(0).values
            px = best[f"ask_price_{lv}"].fillna(0).values
            tokens += sz
            dollars += sz * px
        best[f"ask_depth_{n_levels}_tokens"] = tokens
        best[f"ask_depth_{n_levels}_dollars"] = dollars

    # Slippage analysis: how much within 1, 2, 3 cents of best ask
    best_ask = best["ask_price_1"].values
    for slip_cents in [1, 2, 3]:
        slip = slip_cents / 100
        tokens = np.zeros(len(best))
        dollars = np.zeros(len(best))
        for lv in range(1, N_LEVELS + 1):
            px = best[f"ask_price_{lv}"].fillna(0).values
            sz = best[f"ask_size_{lv}"].fillna(0).values
            within = (px > 0) & (px <= best_ask + slip)
            tokens += np.where(within, sz, 0)
            dollars += np.where(within, sz * px, 0)
        best[f"slip_{slip_cents}c_tokens"] = tokens
        best[f"slip_{slip_cents}c_dollars"] = dollars

    # ===== RESULTS =====
    print(f"\n{'='*70}")
    print("ASK-SIDE LIQUIDITY AT t=120s (leader token)")
    print(f"{'='*70}")

    print(f"\n--- Token depth distribution ---")
    print(f"  {'Metric':<25s}  {'p10':>8s}  {'p25':>8s}  {'p50':>8s}  {'p75':>8s}  {'p90':>8s}  {'p95':>8s}")
    print("  " + "-" * 70)

    for col, label in [
        ("ask_depth_1_tokens", "Level 1 (best ask)"),
        ("ask_depth_3_tokens", "Levels 1-3"),
        ("ask_depth_5_tokens", "Levels 1-5"),
        ("slip_1c_tokens", "Within 1c of best ask"),
        ("slip_2c_tokens", "Within 2c of best ask"),
        ("slip_3c_tokens", "Within 3c of best ask"),
    ]:
        vals = best[col].values
        pcts = np.percentile(vals, [10, 25, 50, 75, 90, 95])
        print(f"  {label:<25s}  {pcts[0]:>8.0f}  {pcts[1]:>8.0f}  {pcts[2]:>8.0f}  "
              f"{pcts[3]:>8.0f}  {pcts[4]:>8.0f}  {pcts[5]:>8.0f}")

    print(f"\n--- Dollar depth distribution ---")
    print(f"  {'Metric':<25s}  {'p10':>8s}  {'p25':>8s}  {'p50':>8s}  {'p75':>8s}  {'p90':>8s}  {'p95':>8s}")
    print("  " + "-" * 70)

    for col, label in [
        ("ask_depth_1_dollars", "Level 1 (best ask)"),
        ("ask_depth_3_dollars", "Levels 1-3"),
        ("ask_depth_5_dollars", "Levels 1-5"),
        ("slip_1c_dollars", "Within 1c"),
        ("slip_2c_dollars", "Within 2c"),
        ("slip_3c_dollars", "Within 3c"),
    ]:
        vals = best[col].values
        pcts = np.percentile(vals, [10, 25, 50, 75, 90, 95])
        print(f"  {label:<25s}  ${pcts[0]:>7.0f}  ${pcts[1]:>7.0f}  ${pcts[2]:>7.0f}  "
              f"${pcts[3]:>7.0f}  ${pcts[4]:>7.0f}  ${pcts[5]:>7.0f}")

    # Fill probability at various bet sizes
    print(f"\n{'='*70}")
    print("FILL PROBABILITY BY BET SIZE")
    print(f"{'='*70}")

    print(f"\n  {'Bet $':>7s}  {'L1 fill %':>10s}  {'2c slip fill %':>15s}  {'3c slip fill %':>15s}")
    print("  " + "-" * 55)

    for bet in [5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200]:
        l1_fill = (best["ask_depth_1_dollars"] >= bet).mean()
        s2_fill = (best["slip_2c_dollars"] >= bet).mean()
        s3_fill = (best["slip_3c_dollars"] >= bet).mean()
        print(f"  ${bet:>5d}  {l1_fill:>9.1%}  {s2_fill:>14.1%}  {s3_fill:>14.1%}")

    # Max bet sizes at various fill rates
    print(f"\n{'='*70}")
    print("MAX BET SIZE AT GUARANTEED FILL RATES")
    print(f"{'='*70}")

    print(f"\n  {'Fill target':>12s}  {'L1 max $':>10s}  {'2c slip max $':>14s}  {'3c slip max $':>14s}")
    print("  " + "-" * 55)

    for fill_pct in [0.50, 0.75, 0.90, 0.95, 0.99]:
        l1_max = np.percentile(best["ask_depth_1_dollars"].values, (1 - fill_pct) * 100)
        s2_max = np.percentile(best["slip_2c_dollars"].values, (1 - fill_pct) * 100)
        s3_max = np.percentile(best["slip_3c_dollars"].values, (1 - fill_pct) * 100)
        print(f"  {fill_pct:>11.0%}  ${l1_max:>8.0f}  ${s2_max:>12.0f}  ${s3_max:>12.0f}")

    # Depth by mid price bucket
    print(f"\n{'='*70}")
    print("LIQUIDITY BY LEADER MID PRICE (Level 1 dollar depth)")
    print(f"{'='*70}")

    print(f"\n  {'Mid range':<14s}  {'N':>6s}  {'p10':>8s}  {'p25':>8s}  {'p50':>8s}  {'p75':>8s}  {'p90':>8s}")
    print("  " + "-" * 60)

    for lo, hi in [(0.50, 0.55), (0.55, 0.60), (0.60, 0.65), (0.65, 0.70),
                   (0.70, 0.75), (0.75, 0.80), (0.80, 0.85), (0.85, 0.90)]:
        mask = (best["mid_price"] >= lo) & (best["mid_price"] < hi)
        n = mask.sum()
        if n < 10:
            continue
        vals = best.loc[mask, "ask_depth_1_dollars"].values
        pcts = np.percentile(vals, [10, 25, 50, 75, 90])
        print(f"  [{lo:.2f}, {hi:.2f})  {n:>6d}  ${pcts[0]:>7.0f}  ${pcts[1]:>7.0f}  "
              f"${pcts[2]:>7.0f}  ${pcts[3]:>7.0f}  ${pcts[4]:>7.0f}")

    # Depth by mid price bucket — 2c slippage
    print(f"\n{'='*70}")
    print("LIQUIDITY BY LEADER MID PRICE (within 2c slippage, dollars)")
    print(f"{'='*70}")

    print(f"\n  {'Mid range':<14s}  {'N':>6s}  {'p10':>8s}  {'p25':>8s}  {'p50':>8s}  {'p75':>8s}  {'p90':>8s}")
    print("  " + "-" * 60)

    for lo, hi in [(0.50, 0.55), (0.55, 0.60), (0.60, 0.65), (0.65, 0.70),
                   (0.70, 0.75), (0.75, 0.80), (0.80, 0.85), (0.85, 0.90)]:
        mask = (best["mid_price"] >= lo) & (best["mid_price"] < hi)
        n = mask.sum()
        if n < 10:
            continue
        vals = best.loc[mask, "slip_2c_dollars"].values
        pcts = np.percentile(vals, [10, 25, 50, 75, 90])
        print(f"  [{lo:.2f}, {hi:.2f})  {n:>6d}  ${pcts[0]:>7.0f}  ${pcts[1]:>7.0f}  "
              f"${pcts[2]:>7.0f}  ${pcts[3]:>7.0f}  ${pcts[4]:>7.0f}")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    med_l1 = np.median(best["ask_depth_1_dollars"])
    med_s2 = np.median(best["slip_2c_dollars"])
    p10_l1 = np.percentile(best["ask_depth_1_dollars"], 10)
    print(f"  Median level-1 ask depth: ${med_l1:.0f}")
    print(f"  Median 2c-slippage depth: ${med_s2:.0f}")
    print(f"  10th percentile level-1 depth: ${p10_l1:.0f}")
    print(f"  Current bet ($5): fills at L1 in {(best['ask_depth_1_dollars'] >= 5).mean():.1%} of markets")
    print(f"  $20 bet: fills at L1 in {(best['ask_depth_1_dollars'] >= 20).mean():.1%} of markets")
    print(f"  $50 bet: fills at L1 in {(best['ask_depth_1_dollars'] >= 50).mean():.1%} of markets")
    print(f"  $100 bet: fills at L1 in {(best['ask_depth_1_dollars'] >= 100).mean():.1%} of markets")


if __name__ == "__main__":
    main()
