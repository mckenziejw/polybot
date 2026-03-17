#!/usr/bin/env python3
"""
Analyze what makes certain trading days unprofitable for XGBoost v3.1.

Bad days: 2/6, 2/23, 3/03, 3/04 (2026)
Uses walk-forward methodology from analyze_vol_filter_full.py.
"""

import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

warnings.filterwarnings("ignore")

FEATURE_CACHE = Path("data/xgb_features_v3.parquet")
MODEL_PATH = Path("data/xgb_model.json")
META_PATH = Path("data/xgb_model_meta.json")

BAD_DATES = {"2026-02-06", "2026-02-23", "2026-03-03", "2026-03-04"}


def polymarket_fee(price: float) -> float:
    return 100 * price * 0.25 * (price * (1 - price)) ** 2 / 10000


def main():
    # Load metadata
    with open(META_PATH) as f:
        meta = json.load(f)
    feature_cols = meta["feature_cols"]
    vol_threshold = meta["vol_threshold"]
    conf_threshold = 0.72  # v3.1 uses 0.72

    print("=" * 90)
    print("BAD DAY ANALYSIS: XGBoost v3.1 Strategy")
    print(f"Bad dates: {', '.join(sorted(BAD_DATES))}")
    print(f"Vol threshold: {vol_threshold:.6f}, Conf threshold: {conf_threshold}")
    print("=" * 90)

    # Load data
    df = pd.read_parquet(FEATURE_CACHE)

    # Sort markets chronologically
    slug_order = df.groupby("slug")["open_ts"].first().sort_values()
    ordered_slugs = slug_order.index.tolist()
    n_markets = len(ordered_slugs)

    print(f"Total: {len(df)} samples, {n_markets} markets")

    # ── Walk-forward predictions ──────────────────────────────────────────
    print("\nRunning walk-forward backtest...")

    min_train = 1000
    step = 200
    val_frac = 0.15

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": 3,
        "learning_rate": 0.01,
        "subsample": 0.7,
        "colsample_bytree": 0.6,
        "min_child_weight": 15,
        "lambda": 2.0,
        "alpha": 0.5,
        "seed": 42,
        "verbosity": 0,
    }

    all_preds = []
    fold = 0
    train_end = min_train

    while train_end < n_markets:
        test_end = min(train_end + step, n_markets)
        val_start = int(train_end * (1 - val_frac))

        actual_train = set(ordered_slugs[:val_start])
        actual_val = set(ordered_slugs[val_start:train_end])
        test_slugs_fold = set(ordered_slugs[train_end:test_end])

        train_data = df[df["slug"].isin(actual_train)]
        val_data = df[df["slug"].isin(actual_val)]
        test_data = df[df["slug"].isin(test_slugs_fold)]

        X_train = train_data[feature_cols].values.astype(np.float32)
        y_train = train_data["label"].values
        X_val = val_data[feature_cols].values.astype(np.float32)
        y_val = val_data["label"].values
        X_test = test_data[feature_cols].values.astype(np.float32)
        y_test = test_data["label"].values

        for X in [X_train, X_val, X_test]:
            X[~np.isfinite(X)] = 0.0

        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_cols)
        dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_cols)
        dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_cols)

        model = xgb.train(params, dtrain, num_boost_round=2000,
                          evals=[(dval, "val")], early_stopping_rounds=50,
                          verbose_eval=False)

        preds = model.predict(dtest)

        for i, (_, row) in enumerate(test_data.iterrows()):
            entry_price = row.get("w120_mid", 0.70)
            if pd.isna(entry_price):
                entry_price = 0.70

            rec = {
                "slug": row["slug"],
                "open_ts": row["open_ts"],
                "observe_time_s": row["observe_time_s"],
                "pred": preds[i],
                "label": row["label"],
                "btc_vol_5m": row.get("btc_vol_5m", np.nan),
                "entry_price": entry_price,
            }
            # Carry over key features for analysis
            for col in ["btc_ret_60s", "btc_ret_300s", "btc_ret_900s",
                        "btc_rvol_60s", "btc_rvol_300s",
                        "w120_spread", "w120_bid_depth_3", "w120_ask_depth_3",
                        "w120_imbalance", "w120_mid",
                        "btc_mid_zscore", "btc_range_5m", "btc_vol_15m",
                        "abs_drift", "drift_speed", "r_squared",
                        "realized_vol", "choppiness", "consistency"]:
                rec[col] = row.get(col, np.nan)
            all_preds.append(rec)

        fold += 1
        if fold % 10 == 0:
            test_dt = datetime.fromtimestamp(slug_order.iloc[train_end], tz=timezone.utc)
            print(f"  Fold {fold}: trained on {train_end}, date ~{test_dt.date()}, OOS preds: {len(all_preds)}")

        train_end = test_end

    pdf = pd.DataFrame(all_preds)
    print(f"Total OOS predictions: {len(pdf)}, {pdf['slug'].nunique()} markets")

    # Filter to 120s window + vol filter + conf threshold
    w120 = pdf[pdf["observe_time_s"] == 120].copy().sort_values("open_ts").reset_index(drop=True)
    vol_mask = w120["btc_vol_5m"] > vol_threshold
    conf_mask = w120["pred"] >= conf_threshold
    trades = w120[vol_mask & conf_mask].copy()

    # Add date column
    trades["date"] = trades["open_ts"].apply(
        lambda ts: datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d"))

    # Compute PnL
    fees = trades["entry_price"].apply(lambda p: polymarket_fee(p) * 10.0)
    trades["pnl"] = np.where(
        trades["label"] == 1,
        (1.0 - trades["entry_price"]) * 10.0 - fees,
        -trades["entry_price"] * 10.0 - fees,
    )
    trades["fee"] = fees

    trades["is_bad_day"] = trades["date"].isin(BAD_DATES)

    n_trades = len(trades)
    n_bad = trades["is_bad_day"].sum()
    print(f"\nFiltered trades (conf>={conf_threshold}, vol>{vol_threshold:.6f}): {n_trades}")
    print(f"Trades on bad days: {n_bad}, good days: {n_trades - n_bad}")

    # ══════════════════════════════════════════════════════════════════════
    # 1. DAILY BREAKDOWN
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 90}")
    print("1. DAILY BREAKDOWN")
    print(f"{'=' * 90}")

    daily = trades.groupby("date").agg(
        n_trades=("pnl", "count"),
        wins=("label", "sum"),
        wr=("label", "mean"),
        pnl=("pnl", "sum"),
        avg_conf=("pred", "mean"),
        avg_entry=("entry_price", "mean"),
        avg_vol=("btc_vol_5m", "mean"),
        avg_spread=("w120_spread", "mean"),
        avg_bid_depth=("w120_bid_depth_3", "mean"),
        avg_ask_depth=("w120_ask_depth_3", "mean"),
        avg_mid=("w120_mid", "mean"),
        avg_abs_drift=("abs_drift", "mean"),
        avg_choppiness=("choppiness", "mean"),
        avg_ret_300s=("btc_ret_300s", "mean"),
    ).reset_index()
    daily["cum_pnl"] = daily["pnl"].cumsum()
    daily["is_bad"] = daily["date"].isin(BAD_DATES)
    daily["ev_per_trade"] = daily["pnl"] / daily["n_trades"]

    print(f"\n  {'Date':<12s} {'Bad':>4s} {'Trd':>4s} {'Win':>4s} {'WR':>6s} "
          f"{'PnL':>8s} {'CumPnL':>8s} {'$/Trd':>7s} {'Conf':>6s} {'Entry':>6s} "
          f"{'Vol5m':>8s} {'Spread':>7s}")
    print("  " + "-" * 95)
    for _, row in daily.iterrows():
        marker = " ***" if row["is_bad"] else ""
        print(f"  {row['date']:<12s} {'YES' if row['is_bad'] else '':>4s} "
              f"{row['n_trades']:>4.0f} {row['wins']:>4.0f} {row['wr']:>5.1%} "
              f"${row['pnl']:>+6.1f} ${row['cum_pnl']:>+6.1f} ${row['ev_per_trade']:>+5.2f} "
              f"{row['avg_conf']:>5.3f} {row['avg_entry']:>5.3f} "
              f"{row['avg_vol']:>7.5f} {row['avg_spread']:>6.4f}{marker}")

    # ══════════════════════════════════════════════════════════════════════
    # 2. BAD vs GOOD DAYS COMPARISON
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 90}")
    print("2. BAD DAYS vs GOOD DAYS — FEATURE COMPARISON")
    print(f"{'=' * 90}")

    bad_trades = trades[trades["is_bad_day"]]
    good_trades = trades[~trades["is_bad_day"]]

    compare_cols = [
        ("btc_vol_5m", "BTC Vol 5m"),
        ("btc_vol_15m", "BTC Vol 15m"),
        ("btc_range_5m", "BTC Range 5m"),
        ("btc_ret_60s", "BTC Ret 60s"),
        ("btc_ret_300s", "BTC Ret 300s"),
        ("btc_ret_900s", "BTC Ret 900s"),
        ("btc_rvol_60s", "BTC RVol 60s"),
        ("btc_rvol_300s", "BTC RVol 300s"),
        ("btc_mid_zscore", "BTC Mid Z-Score"),
        ("w120_spread", "Spread @120s"),
        ("w120_bid_depth_3", "Bid Depth 3 @120s"),
        ("w120_ask_depth_3", "Ask Depth 3 @120s"),
        ("w120_imbalance", "Imbalance @120s"),
        ("w120_mid", "Mid Price @120s"),
        ("entry_price", "Entry Price"),
        ("pred", "Model Confidence"),
        ("abs_drift", "Abs Drift"),
        ("drift_speed", "Drift Speed"),
        ("r_squared", "R-Squared"),
        ("realized_vol", "Realized Vol"),
        ("choppiness", "Choppiness"),
        ("consistency", "Consistency"),
    ]

    print(f"\n  {'Feature':<22s} {'Bad Mean':>10s} {'Good Mean':>10s} {'Diff':>10s} {'Bad Med':>10s} {'Good Med':>10s}")
    print("  " + "-" * 78)
    for col, label in compare_cols:
        if col not in trades.columns:
            continue
        bad_vals = bad_trades[col].dropna()
        good_vals = good_trades[col].dropna()
        if len(bad_vals) == 0 or len(good_vals) == 0:
            continue
        bm, gm = bad_vals.mean(), good_vals.mean()
        bmed, gmed = bad_vals.median(), good_vals.median()
        diff = bm - gm
        pct = diff / abs(gm) * 100 if gm != 0 else 0
        print(f"  {label:<22s} {bm:>10.5f} {gm:>10.5f} {diff:>+9.5f} {bmed:>10.5f} {gmed:>10.5f}  ({pct:+.1f}%)")

    # ══════════════════════════════════════════════════════════════════════
    # 3. DAILY FEATURE CORRELATIONS WITH EV
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 90}")
    print("3. CORRELATION: Daily EV vs Daily Feature Averages")
    print(f"{'=' * 90}")

    # Build daily feature averages
    feature_agg_cols = ["btc_vol_5m", "btc_vol_15m", "btc_range_5m",
                        "btc_ret_60s", "btc_ret_300s", "btc_ret_900s",
                        "btc_rvol_60s", "btc_rvol_300s", "btc_mid_zscore",
                        "w120_spread", "w120_bid_depth_3", "w120_ask_depth_3",
                        "w120_imbalance", "w120_mid", "entry_price", "pred",
                        "abs_drift", "drift_speed", "r_squared",
                        "realized_vol", "choppiness", "consistency"]

    daily_feats = trades.groupby("date").agg(
        ev=("pnl", "mean"),
        n_trades=("pnl", "count"),
        wr=("label", "mean"),
        **{col: (col, "mean") for col in feature_agg_cols if col in trades.columns}
    ).reset_index()

    print(f"\n  {'Feature':<22s} {'Corr w/ EV':>12s} {'Corr w/ WR':>12s}")
    print("  " + "-" * 50)
    corr_results = []
    for col in feature_agg_cols:
        if col not in daily_feats.columns:
            continue
        valid = daily_feats[[col, "ev", "wr"]].dropna()
        if len(valid) < 5:
            continue
        corr_ev = valid[col].corr(valid["ev"])
        corr_wr = valid[col].corr(valid["wr"])
        corr_results.append((col, corr_ev, corr_wr))

    # Sort by absolute correlation with EV
    corr_results.sort(key=lambda x: abs(x[1]), reverse=True)
    for col, corr_ev, corr_wr in corr_results:
        marker = " <--" if abs(corr_ev) > 0.3 else ""
        print(f"  {col:<22s} {corr_ev:>+11.3f} {corr_wr:>+11.3f}{marker}")

    # ══════════════════════════════════════════════════════════════════════
    # 4. CONFIDENCE DISTRIBUTION
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 90}")
    print("4. CONFIDENCE DISTRIBUTION: Bad vs Good Days")
    print(f"{'=' * 90}")

    for label, subset in [("Bad days", bad_trades), ("Good days", good_trades)]:
        confs = subset["pred"]
        print(f"\n  {label} (n={len(subset)}):")
        print(f"    Mean: {confs.mean():.4f}, Median: {confs.median():.4f}, Std: {confs.std():.4f}")
        for pct in [0.72, 0.75, 0.80, 0.85, 0.90]:
            n_above = (confs >= pct).sum()
            wr_above = subset[confs >= pct]["label"].mean() if n_above > 0 else 0
            print(f"    conf >= {pct:.2f}: {n_above:>4d} trades, WR={wr_above:.1%}")

    # ══════════════════════════════════════════════════════════════════════
    # 5. ENTRY PRICE DISTRIBUTION
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 90}")
    print("5. ENTRY PRICE DISTRIBUTION: Bad vs Good Days")
    print(f"{'=' * 90}")

    for label, subset in [("Bad days", bad_trades), ("Good days", good_trades)]:
        prices = subset["entry_price"]
        print(f"\n  {label} (n={len(subset)}):")
        print(f"    Mean: {prices.mean():.4f}, Median: {prices.median():.4f}, Std: {prices.std():.4f}")
        for lo, hi in [(0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]:
            mask = (prices >= lo) & (prices < hi)
            n = mask.sum()
            wr = subset[mask]["label"].mean() if n > 0 else 0
            pnl = subset[mask]["pnl"].sum() if n > 0 else 0
            print(f"    [{lo:.1f}-{hi:.1f}): {n:>4d} trades, WR={wr:.1%}, PnL=${pnl:+.1f}")

    # ══════════════════════════════════════════════════════════════════════
    # 6. CANDIDATE DAILY FILTERS
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 90}")
    print("6. CANDIDATE FILTERS TO AVOID BAD DAYS")
    print(f"{'=' * 90}")

    # For each candidate filter, compute stats with and without
    # We apply filters per-trade (not per-day) since some features vary intraday

    filters = []

    # BTC vol filters
    for q in [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
        thresh = trades["btc_vol_5m"].quantile(q)
        mask = trades["btc_vol_5m"] > thresh
        filters.append((f"btc_vol_5m > p{int(q*100)} ({thresh:.6f})", mask))

    # BTC vol upper bound (avoid extreme vol?)
    for q in [0.90, 0.95]:
        thresh = trades["btc_vol_5m"].quantile(q)
        mask = trades["btc_vol_5m"] < thresh
        filters.append((f"btc_vol_5m < p{int(q*100)} ({thresh:.6f})", mask))

    # Spread filters
    for q in [0.75, 0.80, 0.90]:
        thresh = trades["w120_spread"].quantile(q)
        mask = trades["w120_spread"] < thresh
        filters.append((f"spread < p{int(q*100)} ({thresh:.4f})", mask))

    # Choppiness filters
    if "choppiness" in trades.columns:
        for q in [0.75, 0.80, 0.90]:
            thresh = trades["choppiness"].dropna().quantile(q)
            mask = (trades["choppiness"] < thresh) | trades["choppiness"].isna()
            filters.append((f"choppiness < p{int(q*100)} ({thresh:.4f})", mask))

    # BTC ret_300s abs (avoid strong trends?)
    for q in [0.80, 0.90]:
        thresh = trades["btc_ret_300s"].abs().quantile(q)
        mask = trades["btc_ret_300s"].abs() < thresh
        filters.append((f"|btc_ret_300s| < p{int(q*100)} ({thresh:.6f})", mask))

    # Consistency filter
    if "consistency" in trades.columns:
        for q in [0.25, 0.33]:
            thresh = trades["consistency"].dropna().quantile(q)
            mask = (trades["consistency"] > thresh) | trades["consistency"].isna()
            filters.append((f"consistency > p{int(q*100)} ({thresh:.4f})", mask))

    # R-squared filter
    if "r_squared" in trades.columns:
        for q in [0.25, 0.33]:
            thresh = trades["r_squared"].dropna().quantile(q)
            mask = (trades["r_squared"] > thresh) | trades["r_squared"].isna()
            filters.append((f"r_squared > p{int(q*100)} ({thresh:.4f})", mask))

    # Higher confidence
    for c in [0.74, 0.76, 0.78, 0.80]:
        mask = trades["pred"] >= c
        filters.append((f"conf >= {c:.2f}", mask))

    # Entry price filter
    for p in [0.55, 0.60, 0.65]:
        mask = trades["entry_price"] >= p
        filters.append((f"entry >= {p:.2f}", mask))

    # BTC vol_15m filters
    if "btc_vol_15m" in trades.columns:
        for q in [0.15, 0.25]:
            thresh = trades["btc_vol_15m"].dropna().quantile(q)
            mask = (trades["btc_vol_15m"] > thresh) | trades["btc_vol_15m"].isna()
            filters.append((f"btc_vol_15m > p{int(q*100)} ({thresh:.6f})", mask))

    baseline_n = len(trades)
    baseline_wr = trades["label"].mean()
    baseline_pnl = trades["pnl"].sum()
    baseline_ev = trades["pnl"].mean()
    baseline_bad_pnl = trades[trades["is_bad_day"]]["pnl"].sum()
    baseline_good_pnl = trades[~trades["is_bad_day"]]["pnl"].sum()

    print(f"\n  Baseline: {baseline_n} trades, WR={baseline_wr:.1%}, "
          f"PnL=${baseline_pnl:+.1f}, EV=${baseline_ev:+.3f}")
    print(f"  Baseline bad day PnL: ${baseline_bad_pnl:+.1f}, good day PnL: ${baseline_good_pnl:+.1f}")

    print(f"\n  {'Filter':<45s} {'Trades':>7s} {'WR':>6s} {'PnL':>8s} {'EV':>7s} "
          f"{'BadPnL':>8s} {'GoodPnL':>9s} {'BadTrd':>7s}")
    print("  " + "-" * 100)

    filter_results = []
    for name, mask in filters:
        subset = trades[mask]
        if len(subset) < 20:
            continue
        n = len(subset)
        wr = subset["label"].mean()
        pnl = subset["pnl"].sum()
        ev = subset["pnl"].mean()
        bad_pnl = subset[subset["is_bad_day"]]["pnl"].sum()
        good_pnl = subset[~subset["is_bad_day"]]["pnl"].sum()
        bad_n = subset["is_bad_day"].sum()

        filter_results.append((name, n, wr, pnl, ev, bad_pnl, good_pnl, bad_n))

    # Sort by EV
    filter_results.sort(key=lambda x: x[3], reverse=True)
    for name, n, wr, pnl, ev, bad_pnl, good_pnl, bad_n in filter_results:
        marker = " ***" if pnl > baseline_pnl * 1.1 else ""
        print(f"  {name:<45s} {n:>7d} {wr:>5.1%} ${pnl:>+6.1f} ${ev:>+5.3f} "
              f"${bad_pnl:>+6.1f} ${good_pnl:>+7.1f} {bad_n:>7d}{marker}")

    # ══════════════════════════════════════════════════════════════════════
    # 7. PER-BAD-DAY DEEP DIVE
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 90}")
    print("7. PER-BAD-DAY DEEP DIVE")
    print(f"{'=' * 90}")

    for bad_date in sorted(BAD_DATES):
        day_trades = trades[trades["date"] == bad_date]
        if len(day_trades) == 0:
            print(f"\n  {bad_date}: NO TRADES")
            continue

        n = len(day_trades)
        wins = day_trades["label"].sum()
        wr = day_trades["label"].mean()
        pnl = day_trades["pnl"].sum()

        print(f"\n  {bad_date}: {n} trades, {wins:.0f} wins, WR={wr:.1%}, PnL=${pnl:+.1f}")

        # Show each trade
        print(f"    {'Time':>8s} {'Conf':>6s} {'Entry':>6s} {'Win':>4s} {'PnL':>7s} "
              f"{'Vol5m':>8s} {'Spread':>7s} {'Ret300s':>9s} {'Drift':>7s} {'Choppy':>7s}")
        print("    " + "-" * 80)
        for _, t in day_trades.iterrows():
            ts_str = datetime.fromtimestamp(t["open_ts"], tz=timezone.utc).strftime("%H:%M")
            print(f"    {ts_str:>8s} {t['pred']:>5.3f} {t['entry_price']:>5.3f} "
                  f"{'Y' if t['label']==1 else 'N':>4s} ${t['pnl']:>+5.2f} "
                  f"{t['btc_vol_5m']:>7.5f} {t.get('w120_spread', 0):>6.4f} "
                  f"{t.get('btc_ret_300s', 0):>+8.5f} "
                  f"{t.get('abs_drift', 0):>6.4f} {t.get('choppiness', 0):>6.4f}")

    # ══════════════════════════════════════════════════════════════════════
    # 8. SUMMARY / CONCLUSIONS
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 90}")
    print("8. SUMMARY")
    print(f"{'=' * 90}")

    # Find the best filter by total PnL
    if filter_results:
        best = max(filter_results, key=lambda x: x[3])
        print(f"\n  Best filter by total PnL: {best[0]}")
        print(f"    {best[1]} trades, WR={best[2]:.1%}, PnL=${best[3]:+.1f}, EV=${best[4]:+.3f}")
        print(f"    vs baseline: PnL=${baseline_pnl:+.1f}, EV=${baseline_ev:+.3f}")
        pnl_improvement = best[3] - baseline_pnl
        print(f"    Improvement: ${pnl_improvement:+.1f}")

    # Find best filter by EV
    if filter_results:
        best_ev = max(filter_results, key=lambda x: x[4])
        print(f"\n  Best filter by EV/trade: {best_ev[0]}")
        print(f"    {best_ev[1]} trades, WR={best_ev[2]:.1%}, PnL=${best_ev[3]:+.1f}, EV=${best_ev[4]:+.3f}")

    # Highlight key bad vs good differences
    bad_vol = bad_trades["btc_vol_5m"].mean()
    good_vol = good_trades["btc_vol_5m"].mean()
    bad_spread = bad_trades["w120_spread"].mean()
    good_spread = good_trades["w120_spread"].mean()
    bad_entry = bad_trades["entry_price"].mean()
    good_entry = good_trades["entry_price"].mean()
    bad_conf = bad_trades["pred"].mean()
    good_conf = good_trades["pred"].mean()

    print(f"\n  Key differences (bad vs good):")
    print(f"    BTC vol 5m:  {bad_vol:.6f} vs {good_vol:.6f} ({(bad_vol/good_vol - 1)*100:+.1f}%)")
    print(f"    Spread:      {bad_spread:.5f} vs {good_spread:.5f} ({(bad_spread/good_spread - 1)*100:+.1f}%)")
    print(f"    Entry price: {bad_entry:.4f} vs {good_entry:.4f} ({(bad_entry/good_entry - 1)*100:+.1f}%)")
    print(f"    Confidence:  {bad_conf:.4f} vs {good_conf:.4f} ({(bad_conf/good_conf - 1)*100:+.1f}%)")
    print(f"    WR:          {bad_trades['label'].mean():.1%} vs {good_trades['label'].mean():.1%}")


if __name__ == "__main__":
    main()
