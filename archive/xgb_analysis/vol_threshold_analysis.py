#!/usr/bin/env python3
"""
Recheck volatility threshold with v3.2 model and current dataset.

Tests different percentile cutoffs and evaluates both WR and PnL impact.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from xgboost_flexible import (
    build_dataset,
    get_feature_columns,
    compute_vol_threshold,
    DROP_FEATURES,
)

warnings.filterwarnings("ignore", category=FutureWarning)

# v3.2 sweep-optimized params
PARAMS_V32 = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "max_depth": 2,
    "learning_rate": 0.014790,
    "subsample": 0.838960,
    "colsample_bytree": 0.951185,
    "min_child_weight": 42,
    "lambda": 0.630567,
    "alpha": 0.010124,
    "gamma": 0.457778,
    "seed": 42,
    "verbosity": 0,
}


def compute_pnl_stats(y_true, entry_prices, mask, bet_dollars=5.0):
    """Compute PnL stats for trades matching mask."""
    trades = []
    for i in np.where(mask)[0]:
        entry = entry_prices[i]
        if entry <= 0 or entry >= 1.0:
            continue
        n_tokens = bet_dollars / entry
        if y_true[i] == 1:
            pnl = n_tokens * (1.0 - entry)
        else:
            pnl = -n_tokens * entry
        trades.append({"entry": entry, "won": bool(y_true[i] == 1), "pnl": pnl, "cost": bet_dollars})

    if not trades:
        return {"n": 0}

    n = len(trades)
    wins = sum(1 for t in trades if t["won"])
    total_pnl = sum(t["pnl"] for t in trades)
    total_cost = sum(t["cost"] for t in trades)
    return {
        "n": n,
        "wins": wins,
        "wr": wins / n,
        "pnl": total_pnl,
        "cost": total_cost,
        "roi": total_pnl / total_cost if total_cost > 0 else 0,
        "pnl_per_trade": total_pnl / n,
    }


def main():
    print("Volatility Threshold Analysis (v3.2 model)")
    print("=" * 70)

    dataset = build_dataset(
        data_dir=Path("data/telonex_book_snapshots"),
        resolution_cache_path=Path("data/resolution_cache.json"),
        btc_path=Path("data/btc_quotes/btcusdt_quotes.parquet"),
        multi_window=True,
    )

    feature_cols = get_feature_columns(dataset)

    # Split: 70/15/15
    slug_order = dataset.groupby("slug")["open_ts"].first().sort_values()
    n = len(slug_order)
    te, ve = int(n * 0.7), int(n * 0.85)

    train_slugs = set(slug_order.index[:te])
    val_slugs = set(slug_order.index[te:ve])
    test_slugs = set(slug_order.index[ve:])

    train_df = dataset[dataset["slug"].isin(train_slugs)]
    val_df = dataset[dataset["slug"].isin(val_slugs)]
    test_df = dataset[dataset["slug"].isin(test_slugs)]

    def to_dm(df):
        X = df[feature_cols].values.astype(np.float32)
        X[~np.isfinite(X)] = 0.0
        return xgb.DMatrix(X, label=df["label"].values, feature_names=feature_cols)

    dtrain = to_dm(train_df)
    dval = to_dm(val_df)
    dtest = to_dm(test_df)

    y_test = test_df["label"].values

    print(f"Dataset: {len(dataset)} samples, {dataset['slug'].nunique()} markets")
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    print(f"Features: {len(feature_cols)}")

    # Train v3.2 model
    model = xgb.train(
        PARAMS_V32, dtrain,
        num_boost_round=2000,
        evals=[(dval, "val")],
        early_stopping_rounds=50,
        verbose_eval=False,
    )
    y_pred = model.predict(dtest)
    print(f"Model trained: {model.best_iteration} iterations")

    # Entry price column
    mid_cols = sorted([c for c in dataset.columns if c.endswith("_mid") and c.startswith("w")])
    entry_col = mid_cols[-1]
    entry_prices = test_df[entry_col].values
    print(f"Entry price column: {entry_col}")

    # Training vol distribution
    train_vol = train_df["btc_vol_5m"].dropna()
    train_vol = train_vol[train_vol > 0]
    test_vol = test_df["btc_vol_5m"].values

    print(f"\nTraining btc_vol_5m distribution:")
    for p in [10, 20, 25, 33, 40, 50, 67, 75, 90]:
        print(f"  p{p}: {train_vol.quantile(p/100):.6f}")

    current_threshold = compute_vol_threshold(train_df, percentile=33.0)
    print(f"\nCurrent production threshold (p33): {current_threshold:.6f}")
    print(f"Saved production value: 0.000740")

    # --- WR + PnL analysis at different cutoffs ---
    w120 = test_df["observe_time_s"].values == 120

    for conf_thresh in [0.70, 0.72]:
        conf_mask = y_pred >= conf_thresh
        print(f"\n{'='*70}")
        print(f"VOL FILTER IMPACT — conf >= {conf_thresh}, 120s window")
        print(f"{'='*70}")
        print(f"  {'Filter':<20s}  {'N':>5s}  {'WR':>6s}  {'PnL':>10s}  {'ROI':>8s}  {'$/trade':>8s}  {'Skipped':>8s}  {'Skip WR':>8s}")
        print("  " + "-" * 82)

        base_mask = w120 & conf_mask
        stats = compute_pnl_stats(y_test, entry_prices, base_mask)
        if stats["n"] > 0:
            print(f"  {'No filter':<20s}  {stats['n']:>5d}  {stats['wr']:>5.1%}  "
                  f"${stats['pnl']:>+9.2f}  {stats['roi']:>+7.2%}  ${stats['pnl_per_trade']:>+7.3f}  "
                  f"{'0':>8s}  {'':>8s}")

        # Cap at 0.90 entry price (our new cap)
        entry_cap = entry_prices <= 0.90
        cap_mask = w120 & conf_mask & entry_cap
        cap_stats = compute_pnl_stats(y_test, entry_prices, cap_mask)
        if cap_stats["n"] > 0:
            skip_cap = w120 & conf_mask & ~entry_cap
            skip_cap_stats = compute_pnl_stats(y_test, entry_prices, skip_cap)
            skip_wr_str = f"{skip_cap_stats['wr']:.0%}" if skip_cap_stats["n"] > 3 else "n/a"
            print(f"  {'+ entry<=0.90':<20s}  {cap_stats['n']:>5d}  {cap_stats['wr']:>5.1%}  "
                  f"${cap_stats['pnl']:>+9.2f}  {cap_stats['roi']:>+7.2%}  ${cap_stats['pnl_per_trade']:>+7.3f}  "
                  f"{skip_cap_stats['n']:>8d}  {skip_wr_str:>8s}")

        for pct in [10, 15, 20, 25, 30, 33, 40, 50]:
            cutoff = float(train_vol.quantile(pct / 100.0))
            vol_pass = test_vol > cutoff
            filtered_mask = w120 & conf_mask & vol_pass
            skipped_mask = w120 & conf_mask & ~vol_pass

            stats = compute_pnl_stats(y_test, entry_prices, filtered_mask)
            if stats["n"] < 5:
                continue

            skip_stats = compute_pnl_stats(y_test, entry_prices, skipped_mask)
            skip_wr_str = f"{skip_stats['wr']:.0%}" if skip_stats["n"] > 3 else "n/a"

            print(f"  {'p'+str(pct)+f' (>{cutoff:.5f})':<20s}  {stats['n']:>5d}  {stats['wr']:>5.1%}  "
                  f"${stats['pnl']:>+9.2f}  {stats['roi']:>+7.2%}  ${stats['pnl_per_trade']:>+7.3f}  "
                  f"{skip_stats['n']:>8d}  {skip_wr_str:>8s}")

        # Combined: vol filter + entry cap
        print(f"\n  Combined (vol filter + entry <= 0.90):")
        print(f"  {'Filter':<20s}  {'N':>5s}  {'WR':>6s}  {'PnL':>10s}  {'ROI':>8s}  {'$/trade':>8s}")
        print("  " + "-" * 60)

        for pct in [20, 25, 30, 33, 40]:
            cutoff = float(train_vol.quantile(pct / 100.0))
            vol_pass = test_vol > cutoff
            combined_mask = w120 & conf_mask & vol_pass & entry_cap
            stats = compute_pnl_stats(y_test, entry_prices, combined_mask)
            if stats["n"] < 5:
                continue
            print(f"  {'p'+str(pct)+f' + cap':<20s}  {stats['n']:>5d}  {stats['wr']:>5.1%}  "
                  f"${stats['pnl']:>+9.2f}  {stats['roi']:>+7.2%}  ${stats['pnl_per_trade']:>+7.3f}")

    # --- Walk-forward vol analysis ---
    print(f"\n\n{'='*70}")
    print("WALK-FORWARD VOL FILTER ANALYSIS")
    print(f"{'='*70}")

    dataset_120 = dataset[dataset["observe_time_s"] == 120].copy()
    slugs = slug_order.index.tolist()
    n_slugs = len(slugs)
    n_windows = 5
    initial_train_pct = 0.60
    test_chunk = (1.0 - initial_train_pct) / n_windows

    for pct_label, pct in [("p20", 20), ("p25", 25), ("p33", 33), ("p40", 40), ("No filter", None)]:
        all_trades = []

        for w in range(n_windows):
            train_end_pct = initial_train_pct + w * test_chunk
            test_end_pct = train_end_pct + test_chunk
            train_end_idx = int(n_slugs * train_end_pct)
            test_start_idx = train_end_idx
            test_end_idx = min(int(n_slugs * test_end_pct), n_slugs)

            w_train_slugs = set(slugs[:train_end_idx])
            w_test_slugs = set(slugs[test_start_idx:test_end_idx])

            train_all = dataset[dataset["slug"].isin(w_train_slugs)]
            val_start = int(len(w_train_slugs) * 0.9)
            train_slug_list = slug_order.index[:train_end_idx].tolist()
            val_slugs_set = set(train_slug_list[val_start:])
            w_train_only = train_all[~train_all["slug"].isin(val_slugs_set)]
            w_val_only = train_all[train_all["slug"].isin(val_slugs_set)]
            w_test_df = dataset_120[dataset_120["slug"].isin(w_test_slugs)]

            if len(w_test_df) < 10:
                continue

            X_train = w_train_only[feature_cols].values.astype(np.float32)
            X_val = w_val_only[feature_cols].values.astype(np.float32)
            X_test = w_test_df[feature_cols].values.astype(np.float32)
            for X in [X_train, X_val, X_test]:
                X[~np.isfinite(X)] = 0.0

            w_dtrain = xgb.DMatrix(X_train, label=w_train_only["label"].values, feature_names=feature_cols)
            w_dval = xgb.DMatrix(X_val, label=w_val_only["label"].values, feature_names=feature_cols)
            w_dtest = xgb.DMatrix(X_test, label=w_test_df["label"].values, feature_names=feature_cols)

            w_model = xgb.train(PARAMS_V32, w_dtrain, num_boost_round=2000,
                                evals=[(w_dval, "val")], early_stopping_rounds=50,
                                verbose_eval=False)

            w_pred = w_model.predict(w_dtest)
            w_entry = w_test_df[entry_col].values
            w_y = w_test_df["label"].values
            w_vol = w_test_df["btc_vol_5m"].values if "btc_vol_5m" in w_test_df.columns else None

            # Compute vol threshold from this window's training data
            if pct is not None:
                w_vol_thresh = compute_vol_threshold(w_train_only, percentile=float(pct))
            else:
                w_vol_thresh = None

            for i in range(len(w_pred)):
                if w_pred[i] < 0.72:
                    continue
                if w_vol is not None and w_vol_thresh is not None:
                    if w_vol[i] <= w_vol_thresh:
                        continue
                entry = w_entry[i]
                if entry <= 0 or entry >= 1.0:
                    continue
                if entry > 0.90:
                    continue

                n_tokens = 5.0 / entry
                if w_y[i] == 1:
                    pnl = n_tokens * (1.0 - entry)
                else:
                    pnl = -n_tokens * entry
                all_trades.append({"won": bool(w_y[i] == 1), "pnl": pnl, "cost": 5.0, "entry": entry})

        n_t = len(all_trades)
        if n_t == 0:
            print(f"  {pct_label:<12s}: no trades")
            continue
        wins = sum(1 for t in all_trades if t["won"])
        total_pnl = sum(t["pnl"] for t in all_trades)
        total_cost = sum(t["cost"] for t in all_trades)
        avg_entry = np.mean([t["entry"] for t in all_trades])
        print(f"  {pct_label:<12s}: {n_t:>4d} trades  WR={wins/n_t:>5.1%}  "
              f"PnL=${total_pnl:>+8.2f}  ROI={total_pnl/total_cost:>+6.2%}  "
              f"avg_entry={avg_entry:.3f}  $/trade=${total_pnl/n_t:>+.3f}")


if __name__ == "__main__":
    main()
