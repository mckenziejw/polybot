#!/usr/bin/env python3
"""
Compare walk-forward results across different time-weight half-lives.
Tests: no weighting, 3.5d, 7d, 14d half-life.
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
    compute_time_weights,
    DROP_FEATURES,
)

warnings.filterwarnings("ignore", category=FutureWarning)

PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "max_depth": 2,
    "learning_rate": 0.005016,
    "subsample": 0.930265,
    "colsample_bytree": 0.985988,
    "min_child_weight": 41,
    "lambda": 0.224145,
    "alpha": 0.827467,
    "gamma": 1.705416,
    "seed": 42,
    "verbosity": 0,
}

CONF_THRESHOLD = 0.72
ENTRY_CAP = 0.90
N_WINDOWS = 6
HALF_LIVES = [None, 3.5, 7.0, 14.0]  # None = no weighting


def compute_pnl(y_true, y_pred, entry_prices, threshold):
    trades = []
    for i in range(len(y_true)):
        conf = y_pred[i]
        if conf < threshold:
            continue
        entry = entry_prices[i]
        if not np.isfinite(entry) or entry <= 0 or entry > ENTRY_CAP:
            continue
        won = bool(y_true[i] == 1)
        pnl = (1.0 - entry) if won else -entry
        trades.append({"won": won, "pnl": pnl, "conf": conf, "entry": entry})
    return trades


def run_walkforward(dataset, half_life):
    label = f"hl={half_life}d" if half_life else "no_weight"
    print(f"\n{'='*60}")
    print(f"  Running walk-forward: {label}")
    print(f"{'='*60}")

    fcols = get_feature_columns(dataset)

    # Use w120 observation time for testing
    dataset_120 = dataset[dataset["observe_time_s"] == 120].copy()
    entry_col = "w120_mid"

    slug_order = dataset_120.groupby("slug")["open_ts"].first().sort_values()
    slugs = slug_order.index.tolist()
    n_slugs = len(slugs)

    # Walk-forward: 60% initial train, slide through remaining 40%
    initial_train_pct = 0.60
    test_chunk = (1.0 - initial_train_pct) / N_WINDOWS

    all_trades = []
    window_results = []

    for w in range(N_WINDOWS):
        train_end_pct = initial_train_pct + w * test_chunk
        test_end_pct = train_end_pct + test_chunk

        train_end_idx = int(n_slugs * train_end_pct)
        test_start_idx = train_end_idx
        test_end_idx = min(int(n_slugs * test_end_pct), n_slugs)

        train_slugs = set(slugs[:train_end_idx])
        test_slugs = set(slugs[test_start_idx:test_end_idx])

        # Train on all observation windows
        train_all = dataset[dataset["slug"].isin(train_slugs)]
        val_start = int(len(train_slugs) * 0.9)
        train_slug_list = slugs[:train_end_idx]
        val_slugs_set = set(train_slug_list[val_start:])
        train_only = train_all[~train_all["slug"].isin(val_slugs_set)]
        val_only = train_all[train_all["slug"].isin(val_slugs_set)]

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

        if half_life is not None:
            train_weights = compute_time_weights(train_only, half_life_days=half_life)
            dtrain = xgb.DMatrix(X_train, label=y_train, weight=train_weights, feature_names=fcols)
        else:
            dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=fcols)

        dval = xgb.DMatrix(X_val, label=y_val, feature_names=fcols)
        dtest = xgb.DMatrix(X_test, label=y_test, feature_names=fcols)

        model = xgb.train(
            PARAMS, dtrain,
            num_boost_round=2000,
            evals=[(dval, "val")],
            early_stopping_rounds=50,
            verbose_eval=False,
        )

        y_pred = model.predict(dtest)
        entry_prices = test_df[entry_col].values

        trades = compute_pnl(y_test, y_pred, entry_prices, CONF_THRESHOLD)
        all_trades.extend(trades)

        n_traded = len(trades)
        wins = sum(1 for t in trades if t["won"])
        pnl = sum(t["pnl"] for t in trades)
        wr = wins / n_traded if n_traded > 0 else 0

        window_results.append({
            "window": w + 1,
            "n_traded": n_traded,
            "wins": wins,
            "wr": wr,
            "pnl": pnl,
        })
        print(f"  Window {w+1}: {n_traded} trades, WR={wr:.1%}, PnL=${pnl:.2f}")

    total_trades = len(all_trades)
    total_wins = sum(1 for t in all_trades if t["won"])
    total_pnl = sum(t["pnl"] for t in all_trades)
    total_wr = total_wins / total_trades if total_trades > 0 else 0

    # Last-window performance (most recent data — what matters most)
    last = window_results[-1] if window_results else {}

    return {
        "label": label,
        "total_trades": total_trades,
        "total_wr": total_wr,
        "total_pnl": total_pnl,
        "last_trades": last.get("n_traded", 0),
        "last_wr": last.get("wr", 0),
        "last_pnl": last.get("pnl", 0),
    }


if __name__ == "__main__":
    print("Loading dataset...")
    dataset = build_dataset(
        data_dir=Path("data/telonex_book_snapshots"),
        resolution_cache_path=Path("data/resolution_cache.json"),
        btc_path=Path("data/btc_quotes/btcusdt_quotes.parquet"),
        multi_window=True,
    )
    print(f"Dataset: {len(dataset)} samples, {dataset['slug'].nunique()} markets")

    results = []
    for hl in HALF_LIVES:
        r = run_walkforward(dataset, hl)
        results.append(r)

    print(f"\n{'='*70}")
    print(f"  COMPARISON SUMMARY (conf >= {CONF_THRESHOLD}, entry cap {ENTRY_CAP})")
    print(f"{'='*70}")
    print(f"  {'Weighting':<12} {'Trades':>7} {'WR':>7} {'PnL':>9}  |  {'Last WR':>7} {'Last #':>7} {'Last PnL':>9}")
    print(f"  {'-'*12} {'-'*7} {'-'*7} {'-'*9}  |  {'-'*7} {'-'*7} {'-'*9}")
    for r in results:
        print(f"  {r['label']:<12} {r['total_trades']:>7} {r['total_wr']:>6.1%} ${r['total_pnl']:>8.2f}  |  {r['last_wr']:>6.1%} {r['last_trades']:>7} ${r['last_pnl']:>8.2f}")
