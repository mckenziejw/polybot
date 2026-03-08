#!/usr/bin/env python3
"""
Unconventional signal exploration for Polymarket BTC Up/Down 5-min markets.

Tests 4 creative signal hypotheses:
  1. Psychological BTC price levels (round numbers)
  2. End-of-market liquidity drain (seconds 240-300)
  3. BTC-vs-Polymarket disagreement (contrarian)
  4. Orderbook entropy as predictability regime detector

Uses the feature cache + raw parquet files + BTC quotes.
"""

import json
import sys
import warnings
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

warnings.filterwarnings("ignore")

BASE = Path("/home/josh/Documents/projects/wss_test")
FEATURE_CACHE = BASE / "data/xgb_features_v2.parquet"
RESOLUTION_CACHE = BASE / "data/resolution_cache.json"
BTC_QUOTES = BASE / "data/btc_quotes/btcusdt_quotes.parquet"
BOOK_DIR = BASE / "data/telonex_book_snapshots"
MARKET_DURATION_MS = 300_000

# ============================================================================
# Utility functions
# ============================================================================

def load_resolutions():
    with open(RESOLUTION_CACHE) as f:
        raw = json.load(f)
    res = {}
    for slug, info in raw.items():
        winner = info.get("winning_token") or info.get("winningTokenId", "")
        if winner:
            res[slug] = winner
    return res

def load_feature_cache():
    df = pd.read_parquet(FEATURE_CACHE)
    print(f"Feature cache: {len(df)} rows, {df['slug'].nunique()} unique markets")
    print(f"Columns: {df.columns.tolist()[:20]}...")
    print(f"Label distribution: {df['label'].value_counts().to_dict()}")
    return df

def load_btc_quotes():
    df = pd.read_parquet(BTC_QUOTES)
    df = df.sort_values("timestamp_ms").reset_index(drop=True)
    print(f"BTC quotes: {len(df)} rows")
    print(f"  Range: {pd.to_datetime(df['timestamp_ms'].min(), unit='ms')} to "
          f"{pd.to_datetime(df['timestamp_ms'].max(), unit='ms')}")
    print(f"  Price range: ${df['mid_price'].min():.0f} - ${df['mid_price'].max():.0f}")
    return df

def significance_test(group_wr, group_n, baseline_wr, baseline_n):
    """Two-proportion z-test."""
    if group_n < 10 or baseline_n < 10:
        return float('nan'), float('nan')
    p_pool = (group_wr * group_n + baseline_wr * baseline_n) / (group_n + baseline_n)
    if p_pool <= 0 or p_pool >= 1:
        return float('nan'), float('nan')
    se = np.sqrt(p_pool * (1 - p_pool) * (1/group_n + 1/baseline_n))
    if se == 0:
        return float('nan'), float('nan')
    z = (group_wr - baseline_wr) / se
    p_val = 2 * (1 - scipy_stats.norm.cdf(abs(z)))
    return z, p_val


# ============================================================================
# SIGNAL 1: Psychological BTC Price Levels
# ============================================================================

def test_psychological_levels(features_df, btc_df):
    """
    Hypothesis: Markets opening near round BTC prices ($X000, $X500, $X00)
    behave differently. Round numbers act as psychological support/resistance,
    which may make price movement (and thus market outcomes) more predictable.
    """
    print("\n" + "="*80)
    print("SIGNAL 1: PSYCHOLOGICAL BTC PRICE LEVELS")
    print("="*80)

    # Get one row per market (use observe_time_s == 60 to deduplicate)
    mkt = features_df[features_df["observe_time_s"] == 60].copy()
    if len(mkt) == 0:
        mkt = features_df.groupby("slug").first().reset_index()

    # Get BTC price at market open
    btc_ts = btc_df["timestamp_ms"].values
    btc_mids = btc_df["mid_price"].values

    btc_at_open = []
    for _, row in mkt.iterrows():
        open_ms = int(row["open_ts"]) * 1000
        idx = np.searchsorted(btc_ts, open_ms, side="right") - 1
        if idx >= 0:
            btc_at_open.append(btc_mids[idx])
        else:
            btc_at_open.append(np.nan)

    mkt["btc_price"] = btc_at_open
    mkt = mkt.dropna(subset=["btc_price"])

    baseline_wr = mkt["label"].mean()
    baseline_n = len(mkt)
    print(f"\nBaseline: {baseline_wr:.4f} win rate ({baseline_n} markets)")

    # Test different round number thresholds
    for level_name, modulo, threshold in [
        ("$1000 levels", 1000, 50),
        ("$500 levels", 500, 25),
        ("$100 levels", 100, 10),
        ("$1000 levels (tight)", 1000, 20),
    ]:
        dist = mkt["btc_price"] % modulo
        # Distance from nearest round number
        dist_from_round = np.minimum(dist, modulo - dist)
        near = mkt[dist_from_round <= threshold]
        far = mkt[dist_from_round > threshold]

        if len(near) < 20:
            print(f"\n  {level_name} (within ${threshold}): too few samples ({len(near)})")
            continue

        near_wr = near["label"].mean()
        far_wr = far["label"].mean()
        z, p = significance_test(near_wr, len(near), far_wr, len(far))

        print(f"\n  {level_name} (within ${threshold}):")
        print(f"    Near round: {near_wr:.4f} WR ({len(near)} mkts)")
        print(f"    Far:        {far_wr:.4f} WR ({len(far)} mkts)")
        print(f"    Diff:       {(near_wr - far_wr)*100:+.2f}pp  z={z:.2f}  p={p:.4f}")

    # Test: does BTC proximity to round number predict Up vs Down winner?
    print("\n  --- Up vs Down conditional on proximity ---")
    mkt["dist_1000"] = np.minimum(mkt["btc_price"] % 1000, 1000 - mkt["btc_price"] % 1000)
    mkt["dist_100"] = np.minimum(mkt["btc_price"] % 100, 100 - mkt["btc_price"] % 100)

    # Are markets near $X000 from below more likely to go UP (resistance break)?
    mkt["below_1000"] = (mkt["btc_price"] % 1000) > 500  # within top half approaching next $1000
    for label, mask in [("Approaching $X000 from below", mkt["below_1000"]),
                         ("Approaching $X000 from above", ~mkt["below_1000"])]:
        sub = mkt[mask & (mkt["dist_1000"] < 100)]
        if len(sub) >= 20:
            wr = sub["label"].mean()
            print(f"    {label}: {wr:.4f} WR ({len(sub)} mkts)")

    # Leader wins by BTC price quartile distance to $1000
    print("\n  --- Leader WR by distance to nearest $1000 ---")
    mkt["dist_1000_bin"] = pd.qcut(mkt["dist_1000"], 5, labels=False, duplicates='drop')
    for b in sorted(mkt["dist_1000_bin"].unique()):
        sub = mkt[mkt["dist_1000_bin"] == b]
        dist_range = f"${sub['dist_1000'].min():.0f}-${sub['dist_1000'].max():.0f}"
        wr = sub["label"].mean()
        z, p = significance_test(wr, len(sub), baseline_wr, baseline_n)
        print(f"    Q{b}: {dist_range:>15s}  WR={wr:.4f} ({len(sub):4d})  z={z:+.2f}  p={p:.3f}")


# ============================================================================
# SIGNAL 2: End-of-Market Liquidity Drain
# ============================================================================

def test_end_of_market_drain(features_df, btc_df):
    """
    Hypothesis: In the last 60 seconds of each 5-minute market, market makers
    withdraw liquidity (widen spreads, reduce depth). This creates predictable
    price patterns. Markets where early drift stalls in the final minute may
    indicate the drift was artificial (MM-driven), not real.

    We test: does the spread/depth trajectory from 120s to 240s predict outcomes
    better than snapshot features?
    """
    print("\n" + "="*80)
    print("SIGNAL 2: END-OF-MARKET DYNAMICS (last 60s)")
    print("="*80)

    # We need raw parquet data for late-market features
    resolutions = load_resolutions()
    parquet_files = sorted(BOOK_DIR.glob("btc-updown-5m-*.parquet"))

    btc_ts = btc_df["timestamp_ms"].values
    btc_mids = btc_df["mid_price"].values

    results = []
    sampled_files = parquet_files  # use all

    for i, pf in enumerate(sampled_files):
        slug = pf.stem
        if slug not in resolutions:
            continue

        try:
            open_ts = int(slug.split("-")[-1])
        except ValueError:
            continue

        open_ms = open_ts * 1000
        close_ms = open_ms + MARKET_DURATION_MS

        try:
            df = pd.read_parquet(pf)
        except Exception:
            continue

        if "exchange_timestamp" not in df.columns or "token_label" not in df.columns:
            continue

        # Get Up token data
        up = df[df["token_label"] == "Up"].sort_values("exchange_timestamp")
        down = df[df["token_label"] == "Down"].sort_values("exchange_timestamp")

        if up.empty or down.empty:
            continue

        # Filter to market window
        up = up[(up["exchange_timestamp"] >= open_ms) & (up["exchange_timestamp"] < close_ms)]
        down = down[(down["exchange_timestamp"] >= open_ms) & (down["exchange_timestamp"] < close_ms)]

        if len(up) < 10 or len(down) < 10:
            continue

        up["sec"] = ((up["exchange_timestamp"] - open_ms) / 1000).astype(int)
        down["sec"] = ((down["exchange_timestamp"] - open_ms) / 1000).astype(int)

        # Determine winner
        winning_token = resolutions[slug]
        up_asset = up["asset_id"].iloc[0]
        up_wins = 1 if up_asset == winning_token else 0

        # Extract features at different time windows
        row = {"slug": slug, "open_ts": open_ts, "up_wins": up_wins}

        for token_name, token_df in [("up", up), ("down", down)]:
            for t_start, t_end, window_name in [
                (0, 60, "early"),
                (60, 180, "mid"),
                (180, 240, "late_pre"),
                (240, 300, "final"),
            ]:
                window = token_df[(token_df["sec"] >= t_start) & (token_df["sec"] < t_end)]
                if len(window) >= 2:
                    row[f"{token_name}_{window_name}_spread_mean"] = window["spread"].mean()
                    row[f"{token_name}_{window_name}_spread_std"] = window["spread"].std()
                    row[f"{token_name}_{window_name}_mid_start"] = window["mid_price"].iloc[0]
                    row[f"{token_name}_{window_name}_mid_end"] = window["mid_price"].iloc[-1]
                    row[f"{token_name}_{window_name}_mid_range"] = window["mid_price"].max() - window["mid_price"].min()
                    row[f"{token_name}_{window_name}_n_updates"] = len(window)

                    if "book_imbalance" in window.columns:
                        row[f"{token_name}_{window_name}_imb_mean"] = window["book_imbalance"].mean()

                    # Depth features
                    for side in ["bid", "ask"]:
                        depth_cols = [f"{side}_size_{i}" for i in range(1, 6) if f"{side}_size_{i}" in window.columns]
                        if depth_cols:
                            row[f"{token_name}_{window_name}_{side}_depth"] = window[depth_cols].sum(axis=1).mean()

        results.append(row)

        if (i + 1) % 500 == 0:
            print(f"  Processed {i+1}/{len(sampled_files)}...")

    eom = pd.DataFrame(results)
    print(f"\nProcessed {len(eom)} markets with end-of-market data")

    # Analysis 1: Spread widening in final minute
    if "up_final_spread_mean" in eom.columns and "up_late_pre_spread_mean" in eom.columns:
        eom["spread_widen_up"] = eom["up_final_spread_mean"] - eom["up_late_pre_spread_mean"]
        eom["spread_widen_down"] = eom.get("down_final_spread_mean", 0) - eom.get("down_late_pre_spread_mean", 0)

        valid = eom.dropna(subset=["spread_widen_up"])
        print(f"\n  Spread widening (final vs pre-final minute):")
        print(f"    Up token:   mean={valid['spread_widen_up'].mean():.6f}, "
              f"median={valid['spread_widen_up'].median():.6f}")
        if "spread_widen_down" in valid.columns:
            valid2 = valid.dropna(subset=["spread_widen_down"])
            print(f"    Down token: mean={valid2['spread_widen_down'].mean():.6f}, "
                  f"median={valid2['spread_widen_down'].median():.6f}")

    # Analysis 2: Does spread widening predict leader reversal?
    # If spread widens dramatically, it might mean the leader's support is thin
    if "up_early_mid_end" in eom.columns and "up_final_mid_end" in eom.columns:
        eom["up_drift_early"] = eom["up_early_mid_end"] - 0.50
        eom["up_drift_final"] = eom["up_final_mid_end"] - 0.50
        eom["drift_reversal"] = np.sign(eom["up_drift_early"]) != np.sign(eom["up_drift_final"])

        valid = eom.dropna(subset=["drift_reversal"])
        reversal_rate = valid["drift_reversal"].mean()
        print(f"\n  Drift reversal rate (early vs final): {reversal_rate:.4f} ({len(valid)} mkts)")

    # Analysis 3: Update frequency drop in final minute
    if "up_final_n_updates" in eom.columns and "up_mid_n_updates" in eom.columns:
        eom["update_ratio_final"] = eom["up_final_n_updates"] / eom["up_mid_n_updates"].clip(lower=1)
        valid = eom.dropna(subset=["update_ratio_final"])

        print(f"\n  Update frequency ratio (final/mid):")
        print(f"    Mean: {valid['update_ratio_final'].mean():.3f}")
        print(f"    Markets with >50% drop: {(valid['update_ratio_final'] < 0.5).sum()} "
              f"({(valid['update_ratio_final'] < 0.5).mean()*100:.1f}%)")

        # Does low update frequency in final minute predict anything?
        baseline_wr = valid["up_wins"].mean()
        low_activity = valid[valid["update_ratio_final"] < 0.5]
        high_activity = valid[valid["update_ratio_final"] >= 0.5]

        if len(low_activity) >= 20 and len(high_activity) >= 20:
            low_wr = low_activity["up_wins"].mean()
            high_wr = high_activity["up_wins"].mean()
            z, p = significance_test(low_wr, len(low_activity), high_wr, len(high_activity))
            print(f"    Low activity final: Up WR={low_wr:.4f} ({len(low_activity)})")
            print(f"    High activity final: Up WR={high_wr:.4f} ({len(high_activity)})")
            print(f"    z={z:.2f}, p={p:.4f}")

    # Analysis 4: Depth drain in final minute
    if "up_final_bid_depth" in eom.columns and "up_mid_bid_depth" in eom.columns:
        eom["depth_drain_bid"] = eom["up_final_bid_depth"] / eom["up_mid_bid_depth"].clip(lower=0.01)
        eom["depth_drain_ask"] = eom.get("up_final_ask_depth", 0) / eom.get("up_mid_ask_depth", pd.Series([0.01])).clip(lower=0.01)

        valid = eom.dropna(subset=["depth_drain_bid"])
        print(f"\n  Depth drain (final/mid bid depth ratio):")
        print(f"    Mean: {valid['depth_drain_bid'].mean():.3f}")
        print(f"    Median: {valid['depth_drain_bid'].median():.3f}")

        # Quintile analysis
        valid["depth_drain_q"] = pd.qcut(valid["depth_drain_bid"], 5, labels=False, duplicates='drop')
        baseline_wr = valid["up_wins"].mean()
        print(f"\n  Up WR by depth drain quintile (baseline={baseline_wr:.4f}):")
        for q in sorted(valid["depth_drain_q"].unique()):
            sub = valid[valid["depth_drain_q"] == q]
            wr = sub["up_wins"].mean()
            ratio_range = f"{sub['depth_drain_bid'].min():.2f}-{sub['depth_drain_bid'].max():.2f}"
            z, p_val = significance_test(wr, len(sub), baseline_wr, len(valid))
            print(f"    Q{q} ({ratio_range:>12s}): WR={wr:.4f} ({len(sub):4d})  z={z:+.2f}  p={p_val:.3f}")

    # Analysis 5: Mid-price movement in final minute vs prior drift direction
    # Does the final-minute price confirm or deny the prior trend?
    if "up_late_pre_mid_end" in eom.columns and "up_final_mid_end" in eom.columns:
        eom["prior_drift"] = eom["up_late_pre_mid_end"] - 0.50  # drift direction before final minute
        eom["final_move"] = eom["up_final_mid_end"] - eom["up_late_pre_mid_end"]  # movement in final minute
        eom["final_confirms"] = np.sign(eom["prior_drift"]) == np.sign(eom["final_move"])

        valid = eom.dropna(subset=["final_confirms", "prior_drift"])
        # Only look at markets with meaningful prior drift
        drifted = valid[valid["prior_drift"].abs() > 0.03]
        if len(drifted) >= 30:
            confirms = drifted[drifted["final_confirms"]]
            denies = drifted[~drifted["final_confirms"]]
            print(f"\n  Final-minute trend confirmation (for |drift|>0.03):")
            print(f"    Confirmation rate: {drifted['final_confirms'].mean():.4f}")
            if len(confirms) >= 10 and len(denies) >= 10:
                c_wr = confirms["up_wins"].mean()
                d_wr = denies["up_wins"].mean()
                print(f"    Confirms + Up WR: {c_wr:.4f} ({len(confirms)})")
                print(f"    Denies + Up WR:   {d_wr:.4f} ({len(denies)})")

    return eom


# ============================================================================
# SIGNAL 3: BTC-Polymarket Disagreement (Contrarian)
# ============================================================================

def test_btc_polymarket_disagreement(features_df, btc_df):
    """
    Hypothesis: When BTC is clearly trending up but the Polymarket mid for
    "Up" is < 0.50 (or vice versa), there's a mispricing. Who's right?

    Also tests the 1-second lead-lag window.
    """
    print("\n" + "="*80)
    print("SIGNAL 3: BTC vs POLYMARKET DISAGREEMENT")
    print("="*80)

    mkt = features_df[features_df["observe_time_s"] == 60].copy()
    if len(mkt) == 0:
        mkt = features_df.groupby("slug").first().reset_index()

    # btc_ret_since_open: BTC return during the market's first 60s
    if "btc_ret_since_open" not in mkt.columns:
        print("  No btc_ret_since_open in features, skipping")
        return

    # drift_from_50: mid - 0.50 of the leader token
    # If drift > 0, leader is ahead (Polymarket thinks leader wins)
    # If btc_ret_since_open > 0, BTC went up (should favor Up token)

    # We need to know which token is the leader
    # In the feature cache, label=1 means leader won, label=0 means leader lost
    # drift_from_50 is always positive for the leader by construction

    baseline_wr = mkt["label"].mean()
    print(f"\n  Baseline leader WR: {baseline_wr:.4f} ({len(mkt)} markets)")

    # Test 1: Does strong BTC move improve predictability?
    btc_ret = mkt["btc_ret_since_open"]
    print(f"\n  BTC return since open stats: mean={btc_ret.mean():.6f}, std={btc_ret.std():.6f}")

    for threshold in [0.0005, 0.001, 0.002, 0.005]:
        strong_up = mkt[btc_ret > threshold]
        strong_down = mkt[btc_ret < -threshold]
        calm = mkt[btc_ret.abs() <= threshold]

        print(f"\n  BTC return threshold: {threshold:.4f}")
        for name, sub in [("BTC strong up", strong_up), ("BTC strong down", strong_down), ("BTC calm", calm)]:
            if len(sub) >= 20:
                wr = sub["label"].mean()
                z, p = significance_test(wr, len(sub), baseline_wr, len(mkt))
                print(f"    {name:20s}: WR={wr:.4f} ({len(sub):4d})  z={z:+.2f}  p={p:.3f}")

    # Test 2: Disagreement signal
    # When mid is near 0.50 but BTC has moved strongly, there might be a lag
    mkt["abs_drift"] = mkt["drift_from_50"].abs() if "drift_from_50" in mkt.columns else 0
    mkt["btc_abs_ret"] = btc_ret.abs()

    # "Disagreement": strong BTC move but flat Polymarket
    if "drift_from_50" in mkt.columns:
        disagree = mkt[(mkt["btc_abs_ret"] > 0.001) & (mkt["abs_drift"] < 0.02)]
        agree = mkt[(mkt["btc_abs_ret"] > 0.001) & (mkt["abs_drift"] >= 0.02)]

        print(f"\n  Disagreement analysis (BTC moved >0.1% but Poly flat <2c):")
        if len(disagree) >= 20:
            wr = disagree["label"].mean()
            z, p = significance_test(wr, len(disagree), baseline_wr, len(mkt))
            print(f"    Disagree: WR={wr:.4f} ({len(disagree):4d})  z={z:+.2f}  p={p:.3f}")
        if len(agree) >= 20:
            wr = agree["label"].mean()
            z, p = significance_test(wr, len(agree), baseline_wr, len(mkt))
            print(f"    Agree:    WR={wr:.4f} ({len(agree):4d})  z={z:+.2f}  p={p:.3f}")

    # Test 3: BTC volatility regimes
    if "btc_window_vol" in mkt.columns:
        vol = mkt["btc_window_vol"]
        mkt["vol_quintile"] = pd.qcut(vol, 5, labels=False, duplicates='drop')
        print(f"\n  Leader WR by BTC volatility quintile:")
        for q in sorted(mkt["vol_quintile"].dropna().unique()):
            sub = mkt[mkt["vol_quintile"] == q]
            wr = sub["label"].mean()
            vol_range = f"{sub['btc_window_vol'].min():.6f}-{sub['btc_window_vol'].max():.6f}"
            z, p = significance_test(wr, len(sub), baseline_wr, len(mkt))
            print(f"    Q{q} (vol {vol_range:>22s}): WR={wr:.4f} ({len(sub):4d})  z={z:+.2f}  p={p:.3f}")


# ============================================================================
# SIGNAL 4: Orderbook Entropy
# ============================================================================

def test_orderbook_entropy(features_df, btc_df):
    """
    Hypothesis: The distribution of depth across price levels contains
    information. High entropy (diffuse depth) = uncertain market = less
    predictable. Low entropy (concentrated) = confident market = more predictable.

    We compute entropy from the bid/ask size distributions at observation time.
    """
    print("\n" + "="*80)
    print("SIGNAL 4: ORDERBOOK ENTROPY AS REGIME DETECTOR")
    print("="*80)

    resolutions = load_resolutions()
    parquet_files = sorted(BOOK_DIR.glob("btc-updown-5m-*.parquet"))

    results = []
    for i, pf in enumerate(parquet_files):
        slug = pf.stem
        if slug not in resolutions:
            continue

        try:
            open_ts = int(slug.split("-")[-1])
        except ValueError:
            continue

        open_ms = open_ts * 1000
        close_ms = open_ms + MARKET_DURATION_MS
        winning_token = resolutions[slug]

        try:
            df = pd.read_parquet(pf)
        except Exception:
            continue

        if "exchange_timestamp" not in df.columns:
            continue

        up = df[df["token_label"] == "Up"].sort_values("exchange_timestamp")
        down = df[df["token_label"] == "Down"].sort_values("exchange_timestamp")

        if up.empty or down.empty:
            continue

        up_asset = up["asset_id"].iloc[0]
        up_wins = 1 if up_asset == winning_token else 0

        # Filter to market window
        up = up[(up["exchange_timestamp"] >= open_ms) & (up["exchange_timestamp"] < close_ms)]
        down = down[(down["exchange_timestamp"] >= open_ms) & (down["exchange_timestamp"] < close_ms)]

        row = {"slug": slug, "open_ts": open_ts, "up_wins": up_wins}

        # Compute entropy at different time points
        for token_name, token_df in [("up", up), ("down", down)]:
            for obs_s, window_label in [(30, "30s"), (60, "60s"), (120, "120s")]:
                obs_ms = open_ms + obs_s * 1000
                pre = token_df[token_df["exchange_timestamp"] <= obs_ms]
                if pre.empty:
                    continue

                snap = pre.iloc[-1]

                # Bid side entropy
                bid_sizes = np.array([snap.get(f"bid_size_{i}", 0) or 0 for i in range(1, 6)])
                ask_sizes = np.array([snap.get(f"ask_size_{i}", 0) or 0 for i in range(1, 6)])

                for side, sizes in [("bid", bid_sizes), ("ask", ask_sizes)]:
                    total = sizes.sum()
                    if total > 0:
                        probs = sizes / total
                        probs = probs[probs > 0]
                        entropy = -np.sum(probs * np.log2(probs))
                        # Max entropy for 5 levels = log2(5) = 2.32
                        norm_entropy = entropy / np.log2(5)
                    else:
                        entropy = 0
                        norm_entropy = 0

                    row[f"{token_name}_{side}_entropy_{window_label}"] = entropy
                    row[f"{token_name}_{side}_norm_entropy_{window_label}"] = norm_entropy

                # Combined entropy (all 10 levels)
                all_sizes = np.concatenate([bid_sizes, ask_sizes])
                total = all_sizes.sum()
                if total > 0:
                    probs = all_sizes / total
                    probs = probs[probs > 0]
                    combined_entropy = -np.sum(probs * np.log2(probs))
                    norm_combined = combined_entropy / np.log2(10)
                else:
                    combined_entropy = 0
                    norm_combined = 0

                row[f"{token_name}_combined_entropy_{window_label}"] = combined_entropy
                row[f"{token_name}_combined_norm_entropy_{window_label}"] = norm_combined

                # Concentration: fraction of depth at best level
                if bid_sizes.sum() > 0:
                    row[f"{token_name}_bid_concentration_{window_label}"] = bid_sizes[0] / bid_sizes.sum()
                if ask_sizes.sum() > 0:
                    row[f"{token_name}_ask_concentration_{window_label}"] = ask_sizes[0] / ask_sizes.sum()

                # Bid-ask depth ratio
                if ask_sizes.sum() > 0:
                    row[f"{token_name}_depth_ratio_{window_label}"] = bid_sizes.sum() / ask_sizes.sum()

        # Entropy change over time (dynamic)
        for token_name in ["up", "down"]:
            e30 = row.get(f"{token_name}_combined_entropy_30s")
            e60 = row.get(f"{token_name}_combined_entropy_60s")
            e120 = row.get(f"{token_name}_combined_entropy_120s")
            if e30 is not None and e60 is not None:
                row[f"{token_name}_entropy_delta_30_60"] = e60 - e30
            if e60 is not None and e120 is not None:
                row[f"{token_name}_entropy_delta_60_120"] = e120 - e60

        results.append(row)

        if (i + 1) % 500 == 0:
            print(f"  Processed {i+1}/{len(parquet_files)}...")

    ent_df = pd.DataFrame(results)
    print(f"\nProcessed {len(ent_df)} markets with entropy data")

    baseline_wr = ent_df["up_wins"].mean()
    print(f"Baseline Up WR: {baseline_wr:.4f}")

    # Analysis 1: Entropy quintiles and predictability
    for token in ["up"]:
        for obs in ["60s"]:
            col = f"{token}_combined_norm_entropy_{obs}"
            if col not in ent_df.columns:
                continue

            valid = ent_df.dropna(subset=[col])
            valid["entropy_q"] = pd.qcut(valid[col], 5, labels=False, duplicates='drop')

            print(f"\n  {token.upper()} token combined normalized entropy at {obs}:")
            print(f"  (Low entropy = concentrated depth, High = diffuse)")

            for q in sorted(valid["entropy_q"].unique()):
                sub = valid[valid["entropy_q"] == q]
                wr = sub["up_wins"].mean()
                ent_range = f"{sub[col].min():.3f}-{sub[col].max():.3f}"
                z, p = significance_test(wr, len(sub), baseline_wr, len(valid))
                print(f"    Q{q} ({ent_range:>12s}): WR={wr:.4f} ({len(sub):4d})  z={z:+.2f}  p={p:.3f}")

    # Analysis 2: Bid concentration (fraction at best level)
    for col_name, display in [
        ("up_bid_concentration_60s", "Up bid concentration at 60s"),
        ("up_ask_concentration_60s", "Up ask concentration at 60s"),
    ]:
        if col_name not in ent_df.columns:
            continue

        valid = ent_df.dropna(subset=[col_name])
        valid["conc_q"] = pd.qcut(valid[col_name], 5, labels=False, duplicates='drop')

        print(f"\n  {display}:")
        for q in sorted(valid["conc_q"].unique()):
            sub = valid[valid["conc_q"] == q]
            wr = sub["up_wins"].mean()
            conc_range = f"{sub[col_name].min():.3f}-{sub[col_name].max():.3f}"
            z, p = significance_test(wr, len(sub), baseline_wr, len(valid))
            print(f"    Q{q} ({conc_range:>12s}): WR={wr:.4f} ({len(sub):4d})  z={z:+.2f}  p={p:.3f}")

    # Analysis 3: Entropy dynamics (change over time)
    delta_col = "up_entropy_delta_30_60"
    if delta_col in ent_df.columns:
        valid = ent_df.dropna(subset=[delta_col])
        valid["delta_q"] = pd.qcut(valid[delta_col], 5, labels=False, duplicates='drop')

        print(f"\n  Entropy change 30s->60s (Up token):")
        print(f"  (Negative = entropy decreasing = depth concentrating)")
        for q in sorted(valid["delta_q"].unique()):
            sub = valid[valid["delta_q"] == q]
            wr = sub["up_wins"].mean()
            delta_range = f"{sub[delta_col].min():.4f} to {sub[delta_col].max():.4f}"
            z, p = significance_test(wr, len(sub), baseline_wr, len(valid))
            print(f"    Q{q} ({delta_range:>22s}): WR={wr:.4f} ({len(sub):4d})  z={z:+.2f}  p={p:.3f}")

    # Analysis 4: Asymmetric entropy (bid vs ask entropy difference)
    for obs in ["60s"]:
        bid_col = f"up_bid_norm_entropy_{obs}"
        ask_col = f"up_ask_norm_entropy_{obs}"
        if bid_col in ent_df.columns and ask_col in ent_df.columns:
            valid = ent_df.dropna(subset=[bid_col, ask_col])
            valid["entropy_asym"] = valid[bid_col] - valid[ask_col]

            valid["asym_q"] = pd.qcut(valid["entropy_asym"], 5, labels=False, duplicates='drop')
            print(f"\n  Bid-Ask entropy asymmetry at {obs} (Up token):")
            print(f"  (Positive = bids more diffuse than asks)")
            for q in sorted(valid["asym_q"].unique()):
                sub = valid[valid["asym_q"] == q]
                wr = sub["up_wins"].mean()
                asym_range = f"{sub['entropy_asym'].min():.3f} to {sub['entropy_asym'].max():.3f}"
                z, p = significance_test(wr, len(sub), baseline_wr, len(valid))
                print(f"    Q{q} ({asym_range:>18s}): WR={wr:.4f} ({len(sub):4d})  z={z:+.2f}  p={p:.3f}")

    return ent_df


# ============================================================================
# BONUS SIGNAL 5: Calendar/Timing Patterns
# ============================================================================

def test_timing_patterns(features_df, btc_df):
    """
    Test time-of-day, day-of-week, and hour patterns.
    """
    print("\n" + "="*80)
    print("SIGNAL 5: CALENDAR & TIMING PATTERNS")
    print("="*80)

    mkt = features_df[features_df["observe_time_s"] == 60].copy()
    if len(mkt) == 0:
        mkt = features_df.groupby("slug").first().reset_index()

    # Convert open_ts to datetime
    mkt["dt"] = pd.to_datetime(mkt["open_ts"], unit="s", utc=True)
    mkt["hour"] = mkt["dt"].dt.hour
    mkt["minute"] = mkt["dt"].dt.minute
    mkt["dow"] = mkt["dt"].dt.dayofweek  # 0=Monday
    mkt["hour_min"] = mkt["hour"] + mkt["minute"] / 60.0

    # Which market in the 5-min cycle? (markets every 300s)
    mkt["cycle_pos"] = mkt["open_ts"] % 3600  # position within the hour

    baseline_wr = mkt["label"].mean()
    baseline_n = len(mkt)
    print(f"\n  Baseline: WR={baseline_wr:.4f} ({baseline_n} markets)")

    # Test 1: Hour of day
    print(f"\n  Leader WR by hour of day (UTC):")
    for h in range(24):
        sub = mkt[mkt["hour"] == h]
        if len(sub) >= 20:
            wr = sub["label"].mean()
            z, p = significance_test(wr, len(sub), baseline_wr, baseline_n)
            marker = " ***" if p < 0.05 else " *" if p < 0.10 else ""
            print(f"    {h:02d}:00  WR={wr:.4f} ({len(sub):4d})  z={z:+.2f}  p={p:.3f}{marker}")

    # Test 2: Day of week
    print(f"\n  Leader WR by day of week:")
    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    for d in range(7):
        sub = mkt[mkt["dow"] == d]
        if len(sub) >= 20:
            wr = sub["label"].mean()
            z, p = significance_test(wr, len(sub), baseline_wr, baseline_n)
            marker = " ***" if p < 0.05 else " *" if p < 0.10 else ""
            print(f"    {dow_names[d]}  WR={wr:.4f} ({len(sub):4d})  z={z:+.2f}  p={p:.3f}{marker}")

    # Test 3: Round hour/half-hour effects
    print(f"\n  Round time effects:")
    for window_name, condition in [
        ("Within 5min of hour", mkt["minute"] < 5),
        ("Within 5min of half-hour", (mkt["minute"] >= 25) & (mkt["minute"] < 35)),
        ("Last 5min of hour", mkt["minute"] >= 55),
        ("Mid-hour (15-45)", (mkt["minute"] >= 15) & (mkt["minute"] < 45)),
    ]:
        sub = mkt[condition]
        if len(sub) >= 20:
            wr = sub["label"].mean()
            z, p = significance_test(wr, len(sub), baseline_wr, baseline_n)
            print(f"    {window_name:35s}: WR={wr:.4f} ({len(sub):4d})  z={z:+.2f}  p={p:.3f}")

    # Test 4: Weekend vs weekday
    mkt["is_weekend"] = mkt["dow"] >= 5
    for name, mask in [("Weekday", ~mkt["is_weekend"]), ("Weekend", mkt["is_weekend"])]:
        sub = mkt[mask]
        if len(sub) >= 20:
            wr = sub["label"].mean()
            z, p = significance_test(wr, len(sub), baseline_wr, baseline_n)
            print(f"    {name:35s}: WR={wr:.4f} ({len(sub):4d})  z={z:+.2f}  p={p:.3f}")

    # Test 5: Trading session effects (US market hours, Asian hours, etc.)
    print(f"\n  Trading session effects (UTC):")
    sessions = {
        "Asia (00-08 UTC)": (0, 8),
        "Europe (08-14 UTC)": (8, 14),
        "US overlap (14-17 UTC)": (14, 17),
        "US afternoon (17-21 UTC)": (17, 21),
        "Late night (21-00 UTC)": (21, 24),
    }
    for name, (h_start, h_end) in sessions.items():
        sub = mkt[(mkt["hour"] >= h_start) & (mkt["hour"] < h_end)]
        if len(sub) >= 20:
            wr = sub["label"].mean()
            z, p = significance_test(wr, len(sub), baseline_wr, baseline_n)
            marker = " ***" if p < 0.05 else " *" if p < 0.10 else ""
            print(f"    {name:30s}: WR={wr:.4f} ({len(sub):4d})  z={z:+.2f}  p={p:.3f}{marker}")


# ============================================================================
# BONUS SIGNAL 6: Minimum Spread / Tick Size Exploitation
# ============================================================================

def test_minimum_spread(features_df):
    """
    When spread is at minimum tick (0.01), depth is the only differentiator.
    Does minimum spread predict higher or lower leader win rate?
    """
    print("\n" + "="*80)
    print("SIGNAL 6: MINIMUM SPREAD / TICK SIZE PATTERNS")
    print("="*80)

    mkt = features_df[features_df["observe_time_s"] == 60].copy()
    if len(mkt) == 0:
        mkt = features_df.groupby("slug").first().reset_index()

    # Find spread column
    spread_col = None
    for c in ["w60_spread", "w45_spread", "w30_spread", "spread"]:
        if c in mkt.columns:
            spread_col = c
            break

    if spread_col is None:
        print("  No spread column found, skipping")
        return

    baseline_wr = mkt["label"].mean()
    print(f"\n  Baseline: WR={baseline_wr:.4f} ({len(mkt)} markets)")
    print(f"\n  Spread stats ({spread_col}):")
    print(f"    Mean: {mkt[spread_col].mean():.4f}")
    print(f"    Median: {mkt[spread_col].median():.4f}")
    print(f"    Min: {mkt[spread_col].min():.4f}")
    print(f"    Max: {mkt[spread_col].max():.4f}")

    # Minimum tick on Polymarket is 0.01
    min_spread = mkt[mkt[spread_col] <= 0.011]
    wide_spread = mkt[mkt[spread_col] > 0.03]
    mid_spread = mkt[(mkt[spread_col] > 0.011) & (mkt[spread_col] <= 0.03)]

    print(f"\n  WR by spread regime:")
    for name, sub in [("Min spread (<=0.011)", min_spread),
                       ("Mid spread (0.011-0.03)", mid_spread),
                       ("Wide spread (>0.03)", wide_spread)]:
        if len(sub) >= 20:
            wr = sub["label"].mean()
            z, p = significance_test(wr, len(sub), baseline_wr, len(mkt))
            print(f"    {name:30s}: WR={wr:.4f} ({len(sub):4d})  z={z:+.2f}  p={p:.3f}")

    # Quintile analysis
    mkt["spread_q"] = pd.qcut(mkt[spread_col], 5, labels=False, duplicates='drop')
    print(f"\n  WR by spread quintile:")
    for q in sorted(mkt["spread_q"].dropna().unique()):
        sub = mkt[mkt["spread_q"] == q]
        wr = sub["label"].mean()
        spread_range = f"{sub[spread_col].min():.4f}-{sub[spread_col].max():.4f}"
        z, p = significance_test(wr, len(sub), baseline_wr, len(mkt))
        print(f"    Q{q} ({spread_range:>15s}): WR={wr:.4f} ({len(sub):4d})  z={z:+.2f}  p={p:.3f}")

    # Cross with imbalance: at min spread, is imbalance more predictive?
    imb_col = None
    for c in ["w60_imbalance", "w45_imbalance", "w30_imbalance", "imbalance"]:
        if c in mkt.columns:
            imb_col = c
            break

    if imb_col and len(min_spread) >= 40:
        print(f"\n  At minimum spread, imbalance ({imb_col}) analysis:")
        ms = min_spread.copy()
        ms["imb_q"] = pd.qcut(ms[imb_col], 3, labels=["low", "mid", "high"], duplicates='drop')
        for q in ["low", "mid", "high"]:
            sub = ms[ms["imb_q"] == q]
            if len(sub) >= 10:
                wr = sub["label"].mean()
                z, p = significance_test(wr, len(sub), baseline_wr, len(mkt))
                print(f"    Imb {q:4s}: WR={wr:.4f} ({len(sub):4d})  z={z:+.2f}  p={p:.3f}")


# ============================================================================
# Main
# ============================================================================

def main():
    print("UNCONVENTIONAL SIGNAL EXPLORATION")
    print("=" * 80)

    # Load data
    features_df = load_feature_cache()
    btc_df = load_btc_quotes()

    # Run all tests
    test_psychological_levels(features_df, btc_df)
    test_timing_patterns(features_df, btc_df)
    test_minimum_spread(features_df)
    test_btc_polymarket_disagreement(features_df, btc_df)

    # These two are slower (read raw parquet files)
    print("\n\n--- Running raw parquet analyses (slower) ---")
    test_end_of_market_drain(features_df, btc_df)
    test_orderbook_entropy(features_df, btc_df)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
