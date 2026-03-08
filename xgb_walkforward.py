#!/usr/bin/env python3
"""
Walk-forward backtest for XGBoost confidence model.

Simulates live trading by training on past data and testing on future data
in rolling windows. Computes realistic PnL accounting for entry prices.

Also tests the impact of a max entry price cap (e.g., 0.90).

Usage:
    python xgb_walkforward.py
    python xgb_walkforward.py --windows 8    # more walk-forward windows
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

from xgboost_flexible import (
    build_dataset,
    get_feature_columns,
    compute_vol_threshold,
    DROP_FEATURES,
)

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# v3.2 sweep-optimized params
# ---------------------------------------------------------------------------

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

# v3.1 params for comparison
PARAMS_V31 = {
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


def compute_pnl(y_true, y_pred, entry_prices, conf_threshold, vol_values=None,
                vol_threshold=None, max_entry=None):
    """Simulate trading PnL.

    For each market where pred >= threshold (and vol filter passes):
    - BUY at entry_price (leader mid at observation time)
    - If label == 1 (leader wins): profit = 1.0 - entry_price per token
    - If label == 0 (leader loses): loss = -entry_price per token
    - Position size = $5 / entry_price tokens (matching live bet sizing)
    """
    trades = []
    bet_dollars = 5.0

    for i in range(len(y_pred)):
        if y_pred[i] < conf_threshold:
            continue
        if vol_values is not None and vol_threshold is not None:
            if vol_values[i] <= vol_threshold:
                continue

        entry = entry_prices[i]
        if entry <= 0 or entry >= 1.0:
            continue
        if max_entry is not None and entry > max_entry:
            continue

        n_tokens = bet_dollars / entry
        if y_true[i] == 1:
            pnl = n_tokens * (1.0 - entry)  # win: tokens pay $1 each
        else:
            pnl = -n_tokens * entry  # loss: tokens worth $0

        trades.append({
            "pred": y_pred[i],
            "entry": entry,
            "won": bool(y_true[i] == 1),
            "pnl": pnl,
            "cost": bet_dollars,
        })

    return trades


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", type=int, default=5,
                        help="Number of walk-forward windows")
    args = parser.parse_args()

    print("XGBoost Walk-Forward Backtest")
    print("="*60)

    dataset = build_dataset(
        data_dir=Path("data/telonex_book_snapshots"),
        resolution_cache_path=Path("data/resolution_cache.json"),
        btc_path=Path("data/btc_quotes/btcusdt_quotes.parquet"),
        multi_window=True,
    )

    # Only use 120s window for walk-forward (matches live behavior)
    dataset_120 = dataset[dataset["observe_time_s"] == 120].copy()
    print(f"\nUsing 120s window: {len(dataset_120)} samples, {dataset_120['slug'].nunique()} markets")

    feature_cols = get_feature_columns(dataset)
    all_feature_cols = [c for c in dataset.columns if c not in {"label", "slug", "open_ts"}
                        and not c.startswith("_") and c not in DROP_FEATURES]

    # Get entry price column
    mid_cols = sorted([c for c in dataset.columns if c.endswith("_mid") and c.startswith("w")])
    entry_col = mid_cols[-1]  # last window mid = entry price proxy
    print(f"Entry price column: {entry_col}")

    # Sort by market open time
    slug_order = dataset_120.groupby("slug")["open_ts"].first().sort_values()
    slugs = slug_order.index.tolist()
    n_slugs = len(slugs)
    print(f"Total markets (120s): {n_slugs}")

    # Walk-forward: train on first N%, test on next chunk, slide forward
    n_windows = args.windows
    # Reserve first 60% for initial training, walk forward through remaining 40%
    initial_train_pct = 0.60
    test_chunk = (1.0 - initial_train_pct) / n_windows

    print(f"\nWalk-forward: {n_windows} windows")
    print(f"Initial training: {initial_train_pct:.0%} ({int(n_slugs * initial_train_pct)} markets)")
    print(f"Test chunk: {test_chunk:.1%} ({int(n_slugs * test_chunk)} markets) per window")

    # Run walk-forward for both v3.1 and v3.2
    for model_label, params, use_all_features in [
        ("v3.1 (old params, all features)", PARAMS_V31, True),
        ("v3.2 (sweep params, deduped)", PARAMS_V32, False),
    ]:
        print(f"\n{'='*70}")
        print(f"MODEL: {model_label}")
        print(f"{'='*70}")

        if use_all_features:
            fcols = [c for c in dataset.columns if c not in {"label", "slug", "open_ts"}
                     and not c.startswith("_")]
        else:
            fcols = feature_cols

        all_trades = []
        window_results = []

        for w in range(n_windows):
            train_end_pct = initial_train_pct + w * test_chunk
            test_end_pct = train_end_pct + test_chunk

            train_end_idx = int(n_slugs * train_end_pct)
            test_start_idx = train_end_idx
            test_end_idx = min(int(n_slugs * test_end_pct), n_slugs)

            # Use multi-window data for training (all obs times), 120s only for testing
            train_slugs = set(slugs[:train_end_idx])
            test_slugs = set(slugs[test_start_idx:test_end_idx])

            # Train on all observation windows
            train_all = dataset[dataset["slug"].isin(train_slugs)]
            # Val: last 10% of training markets
            val_start = int(len(train_slugs) * 0.9)
            train_slug_list = slug_order.index[:train_end_idx].tolist()
            val_slugs_set = set(train_slug_list[val_start:])
            train_only = train_all[~train_all["slug"].isin(val_slugs_set)]
            val_only = train_all[train_all["slug"].isin(val_slugs_set)]

            # Test on 120s only
            test_df = dataset_120[dataset_120["slug"].isin(test_slugs)]

            if len(test_df) < 10:
                continue

            X_train = train_only[fcols].values.astype(np.float32)
            y_train = train_only["label"].values
            X_val = val_only[fcols].values.astype(np.float32)
            y_val = val_only["label"].values
            X_test = test_df[fcols].values.astype(np.float32)
            y_test = test_df["label"].values

            for X in [X_train, X_val, X_test]:
                X[~np.isfinite(X)] = 0.0

            dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=fcols)
            dval = xgb.DMatrix(X_val, label=y_val, feature_names=fcols)
            dtest = xgb.DMatrix(X_test, label=y_test, feature_names=fcols)

            model = xgb.train(
                params, dtrain,
                num_boost_round=2000,
                evals=[(dval, "val")],
                early_stopping_rounds=50,
                verbose_eval=False,
            )

            y_pred = model.predict(dtest)
            entry_prices = test_df[entry_col].values
            vol_values = test_df["btc_vol_5m"].values if "btc_vol_5m" in test_df.columns else None
            vol_threshold = compute_vol_threshold(train_only, percentile=25.0)

            trades = compute_pnl(y_test, y_pred, entry_prices, 0.72,
                                vol_values, vol_threshold)

            all_trades.extend(trades)
            n_traded = len(trades)
            wins = sum(1 for t in trades if t["won"])
            pnl = sum(t["pnl"] for t in trades)
            wr = wins / n_traded if n_traded > 0 else 0

            window_results.append({
                "window": w + 1,
                "train_markets": len(train_slugs),
                "test_markets": len(test_slugs),
                "n_traded": n_traded,
                "wins": wins,
                "losses": n_traded - wins,
                "wr": wr,
                "pnl": pnl,
            })

            print(f"  Window {w+1}: train={len(train_slugs)} test={len(test_slugs)} "
                  f"trades={n_traded} WR={wr:.1%} PnL=${pnl:.2f}")

        # Summary
        total_trades = len(all_trades)
        total_wins = sum(1 for t in all_trades if t["won"])
        total_pnl = sum(t["pnl"] for t in all_trades)
        total_cost = sum(t["cost"] for t in all_trades)
        avg_entry = np.mean([t["entry"] for t in all_trades]) if all_trades else 0

        print(f"\n  TOTAL: {total_trades} trades, {total_wins}W/{total_trades - total_wins}L")
        print(f"  Win rate: {total_wins / total_trades:.1%}" if total_trades > 0 else "  No trades")
        print(f"  Total PnL: ${total_pnl:.2f}")
        print(f"  Total capital deployed: ${total_cost:.2f}")
        print(f"  ROI: {total_pnl / total_cost:.2%}" if total_cost > 0 else "")
        print(f"  Avg entry price: {avg_entry:.3f}")
        print(f"  Avg PnL/trade: ${total_pnl / total_trades:.3f}" if total_trades > 0 else "")

        # Entry price bucketing
        if all_trades:
            print(f"\n  PnL by entry price:")
            print(f"  {'Entry range':<16s}  {'N':>5s}  {'WR':>6s}  {'PnL':>10s}  {'PnL/trade':>10s}")
            print(f"  " + "-" * 55)

            entries = np.array([t["entry"] for t in all_trades])
            pnls = np.array([t["pnl"] for t in all_trades])
            wons = np.array([t["won"] for t in all_trades])

            for lo, hi in [(0.50, 0.55), (0.55, 0.60), (0.60, 0.65), (0.65, 0.70),
                           (0.70, 0.75), (0.75, 0.80), (0.80, 0.85), (0.85, 0.90),
                           (0.90, 0.95), (0.95, 1.0)]:
                mask = (entries >= lo) & (entries < hi)
                n = mask.sum()
                if n < 3:
                    continue
                wr = wons[mask].mean()
                bucket_pnl = pnls[mask].sum()
                avg_pnl = bucket_pnl / n
                print(f"  [{lo:.2f}, {hi:.2f})    {n:>5d}  {wr:>5.1%}  ${bucket_pnl:>+9.2f}  ${avg_pnl:>+9.3f}")

    # --- Max entry price cap analysis ---
    print(f"\n\n{'='*70}")
    print("MAX ENTRY PRICE CAP ANALYSIS (v3.2, walk-forward)")
    print(f"{'='*70}")

    # Rerun v3.2 walk-forward with different caps
    for max_entry_label, max_entry_val in [
        ("No cap", None), ("0.95", 0.95), ("0.90", 0.90),
        ("0.85", 0.85), ("0.80", 0.80),
    ]:
        cap_trades = []

        for w in range(n_windows):
            train_end_pct = initial_train_pct + w * test_chunk
            test_end_pct = train_end_pct + test_chunk
            train_end_idx = int(n_slugs * train_end_pct)
            test_start_idx = train_end_idx
            test_end_idx = min(int(n_slugs * test_end_pct), n_slugs)

            train_slugs = set(slugs[:train_end_idx])
            test_slugs_set = set(slugs[test_start_idx:test_end_idx])

            train_all = dataset[dataset["slug"].isin(train_slugs)]
            val_start = int(len(train_slugs) * 0.9)
            train_slug_list = slug_order.index[:train_end_idx].tolist()
            val_slugs_set = set(train_slug_list[val_start:])
            train_only = train_all[~train_all["slug"].isin(val_slugs_set)]
            val_only = train_all[train_all["slug"].isin(val_slugs_set)]
            test_df = dataset_120[dataset_120["slug"].isin(test_slugs_set)]

            if len(test_df) < 10:
                continue

            X_train = train_only[feature_cols].values.astype(np.float32)
            X_val = val_only[feature_cols].values.astype(np.float32)
            X_test = test_df[feature_cols].values.astype(np.float32)
            y_test = test_df["label"].values

            for X in [X_train, X_val, X_test]:
                X[~np.isfinite(X)] = 0.0

            dtrain = xgb.DMatrix(X_train, label=train_only["label"].values, feature_names=feature_cols)
            dval = xgb.DMatrix(X_val, label=val_only["label"].values, feature_names=feature_cols)
            dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_cols)

            model = xgb.train(PARAMS_V32, dtrain, num_boost_round=2000,
                              evals=[(dval, "val")], early_stopping_rounds=50,
                              verbose_eval=False)

            y_pred = model.predict(dtest)
            entry_prices = test_df[entry_col].values
            vol_values = test_df["btc_vol_5m"].values if "btc_vol_5m" in test_df.columns else None
            vol_threshold = compute_vol_threshold(train_only, percentile=25.0)

            trades = compute_pnl(y_test, y_pred, entry_prices, 0.72,
                                vol_values, vol_threshold, max_entry=max_entry_val)
            cap_trades.extend(trades)

        n = len(cap_trades)
        if n == 0:
            print(f"  Cap {max_entry_label:<8s}: no trades")
            continue
        wins = sum(1 for t in cap_trades if t["won"])
        pnl = sum(t["pnl"] for t in cap_trades)
        cost = sum(t["cost"] for t in cap_trades)
        avg_e = np.mean([t["entry"] for t in cap_trades])
        print(f"  Cap {max_entry_label:<8s}: {n:>4d} trades  WR={wins/n:>5.1%}  "
              f"PnL=${pnl:>+8.2f}  ROI={pnl/cost:>+6.2%}  avg_entry={avg_e:.3f}")


if __name__ == "__main__":
    main()
