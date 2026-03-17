"""Walk-forward training for Polymarket 5-minute BTC direction prediction.

Predicts whether BTC will be up or down over each fixed 5-minute window,
matching Polymarket Up/Down binary markets.

Labels are window-level (all candles in a window share the same label),
but features differ by position within the window.

Usage:
    python -m strategies.polymarket.train --data-dir data/cryptolake
    python -m strategies.polymarket.train --data-dir data/cryptolake --minute 0  # only train on window-open candles
"""

import argparse
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import xgboost as xgb
except ImportError:
    print("ERROR: xgboost not installed. Run: pip install xgboost")
    raise SystemExit(1)

from sklearn.metrics import accuracy_score, roc_auc_score

from .features import load_candles_range, assign_windows, compute_features, get_feature_columns

warnings.filterwarnings("ignore", category=FutureWarning)


# ── XGBoost config ────────────────────────────────────────────────────────

DEFAULT_XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "max_depth": 4,
    "learning_rate": 0.01,
    "n_estimators": 2000,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "min_child_weight": 50,
    "verbosity": 0,
    "tree_method": "hist",
}

N_SEEDS = 5
SEEDS = [42, 123, 456, 789, 1024]
EARLY_STOPPING = 50


# ── Training functions ────────────────────────────────────────────────────

def train_single_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
    n_jobs: int = -1,
) -> xgb.XGBClassifier:
    """Train one XGBoost model with early stopping."""
    params = {**DEFAULT_XGB_PARAMS, "seed": seed, "n_jobs": n_jobs}
    model = xgb.XGBClassifier(early_stopping_rounds=EARLY_STOPPING, **params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model


def train_ensemble(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_jobs_per_seed: int | None = None,
) -> list[xgb.XGBClassifier]:
    """Train seed ensemble sequentially."""
    total_cores = os.cpu_count() or 1
    cores_per_seed = n_jobs_per_seed or max(1, total_cores // N_SEEDS)
    return [
        train_single_model(X_train, y_train, X_val, y_val, seed, n_jobs=cores_per_seed)
        for seed in SEEDS
    ]


def ensemble_predict(
    models: list[xgb.XGBClassifier], X: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return (mean_probability, std_probability)."""
    preds = np.array([m.predict_proba(X)[:, 1] for m in models])
    return preds.mean(axis=0), preds.std(axis=0)


# ── Walk-forward ──────────────────────────────────────────────────────────

def walk_forward(args) -> dict:
    """Run walk-forward validation for Polymarket prediction."""
    data_dir = Path(args.data_dir)
    results_dir = Path(args.results_dir)
    model_dir = Path(args.model_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    symbol = args.symbol
    train_days = args.train_days
    test_days = args.test_days
    purge_windows = args.purge_windows  # gap in 5-min windows between train/test
    minute_filter = args.minute  # if set, only use candles at this minute position

    # Find available date range
    candles_dir = data_dir / "candles" / f"exchange=BINANCE_FUTURES" / f"symbol={symbol}"
    parquet_files = sorted(candles_dir.glob("*.parquet"))
    if not parquet_files:
        print(f"No candle data in {candles_dir}")
        return {}

    dates = [f.stem for f in parquet_files]
    first_date, last_date = dates[0], dates[-1]
    print(f"Data range: {first_date} to {last_date} ({len(dates)} days)")
    print(f"Symbol: {symbol}")
    if minute_filter is not None:
        print(f"Minute filter: only using minute {minute_filter} candles")

    # Generate walk-forward windows
    all_dates = pd.date_range(first_date, last_date, freq="D")
    windows = []
    i = train_days
    while i + test_days <= len(all_dates):
        tr_start = all_dates[i - train_days].strftime("%Y-%m-%d")
        tr_end = all_dates[i].strftime("%Y-%m-%d")
        te_start = all_dates[i].strftime("%Y-%m-%d")
        te_end_idx = min(i + test_days, len(all_dates))
        if te_end_idx < len(all_dates):
            te_end = all_dates[te_end_idx].strftime("%Y-%m-%d")
        else:
            te_end = (all_dates[-1] + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        windows.append((tr_start, tr_end, te_start, te_end))
        i += test_days

    print(f"Walk-forward: {len(windows)} windows ({train_days}d train, {test_days}d test)")
    if not windows:
        print("Not enough data")
        return {}

    all_results = []
    window_metrics = []

    for w_idx, (tr_start, tr_end, te_start, te_end) in enumerate(windows):
        print(f"\n{'='*60}")
        print(f"Window {w_idx + 1}/{len(windows)}: train {tr_start}->{tr_end}, test {te_start}->{te_end}")
        print(f"{'='*60}")

        # ── Load and build features ───────────────────────────────────
        # Load extra days before train start for feature warm-up (4h = 240 bars)
        warmup_start = (pd.Timestamp(tr_start) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        print("  Loading training data...")
        try:
            candles_train = load_candles_range(warmup_start, tr_end, data_dir, symbol=symbol)
        except ValueError as e:
            print(f"  Skipping: {e}")
            continue

        candles_train = assign_windows(candles_train)
        features_train = compute_features(candles_train)

        # Drop warm-up period (features need lookback)
        train_start_ms = pd.Timestamp(tr_start).timestamp() * 1000
        features_train = features_train[features_train["timestamp_ms"] >= train_start_ms].reset_index(drop=True)

        # Apply purge: drop last N 5-minute windows from training
        if purge_windows > 0:
            purge_ms = purge_windows * 5 * 60 * 1000
            max_ts = features_train["timestamp_ms"].max()
            features_train = features_train[features_train["timestamp_ms"] <= max_ts - purge_ms]
            print(f"  Purged last {purge_windows} windows ({purge_windows * 5}min) from training")

        # Apply minute filter
        if minute_filter is not None:
            # minute_in_window is normalized to 0-1, so minute 0 = 0.0, minute 4 = 1.0
            target_val = minute_filter / 4.0
            features_train = features_train[
                np.isclose(features_train["minute_in_window"], target_val)
            ].reset_index(drop=True)

        # Get feature columns and labels
        feat_cols = get_feature_columns(features_train)
        valid = features_train["label"].notna()
        X_all = features_train.loc[valid, feat_cols].values.astype(np.float32)
        y_all = features_train.loc[valid, "label"].values.astype(np.float32)
        ts_all = features_train.loc[valid, "timestamp_ms"].values

        if len(y_all) < 500:
            print(f"  Too few samples ({len(y_all)}), skipping")
            continue

        # Val split: last 10%
        val_start = int(len(X_all) * 0.9)
        X_tr, y_tr = X_all[:val_start], y_all[:val_start]
        X_val, y_val = X_all[val_start:], y_all[val_start:]

        up_pct = y_all.mean() * 100
        print(f"  Train: {len(X_tr):,}, Val: {len(X_val):,} ({up_pct:.1f}% up)")

        X_tr[~np.isfinite(X_tr)] = 0.0
        X_val[~np.isfinite(X_val)] = 0.0

        # ── Train ─────────────────────────────────────────────────────
        print(f"  Training {N_SEEDS}-seed ensemble...")
        models = train_ensemble(X_tr, y_tr, X_val, y_val, n_jobs_per_seed=args.n_jobs)

        val_mean, val_std = ensemble_predict(models, X_val)
        val_acc = accuracy_score(y_val, (val_mean > 0.5).astype(int))
        val_auc = roc_auc_score(y_val, val_mean) if len(np.unique(y_val)) > 1 else 0.5
        print(f"  Val acc: {val_acc:.4f}, AUC: {val_auc:.4f}")

        # ── Test ──────────────────────────────────────────────────────
        print("  Loading test data...")
        try:
            candles_test = load_candles_range(te_start, te_end, data_dir, symbol=symbol)
        except ValueError as e:
            print(f"  Skipping test: {e}")
            continue

        candles_test = assign_windows(candles_test)
        features_test = compute_features(candles_test)

        if minute_filter is not None:
            target_val = minute_filter / 4.0
            features_test = features_test[
                np.isclose(features_test["minute_in_window"], target_val)
            ].reset_index(drop=True)

        valid_test = features_test["label"].notna()
        X_test = features_test.loc[valid_test, feat_cols].values.astype(np.float32)
        y_test = features_test.loc[valid_test, "label"].values.astype(np.float32)
        ts_test = features_test.loc[valid_test, "timestamp_ms"].values
        wid_test = features_test.loc[valid_test, "window_id"].values
        X_test[~np.isfinite(X_test)] = 0.0

        if len(y_test) == 0:
            print("  No valid test samples, skipping")
            continue

        test_mean, test_std = ensemble_predict(models, X_test)
        test_acc = accuracy_score(y_test, (test_mean > 0.5).astype(int))
        test_auc = roc_auc_score(y_test, test_mean) if len(np.unique(y_test)) > 1 else 0.5

        print(f"  Test acc: {test_acc:.4f}, AUC: {test_auc:.4f}, n={len(y_test):,}")

        # Feature importance
        importance = np.zeros(len(feat_cols))
        for m in models:
            importance += m.feature_importances_
        importance /= len(models)
        top_idx = np.argsort(importance)[::-1][:10]
        print("  Top 10 features:")
        for rank, idx in enumerate(top_idx, 1):
            print(f"    {rank:2d}. {feat_cols[idx]:30s} {importance[idx]:.4f}")

        # Store results
        window_result = pd.DataFrame({
            "timestamp_ms": ts_test,
            "window_id": wid_test,
            "y_true": y_test,
            "y_pred": test_mean,
            "y_std": test_std,
            "window": w_idx,
        })
        all_results.append(window_result)

        window_metrics.append({
            "window": w_idx,
            "train_range": f"{tr_start} -> {tr_end}",
            "test_range": f"{te_start} -> {te_end}",
            "train_n": len(X_tr),
            "test_n": len(X_test),
            "val_acc": float(val_acc),
            "val_auc": float(val_auc),
            "test_acc": float(test_acc),
            "test_auc": float(test_auc),
        })

        # Save models
        for s_idx, m in enumerate(models):
            m.save_model(str(model_dir / f"window_{w_idx}_seed_{s_idx}.json"))

    if not all_results:
        print("\nNo windows completed.")
        return {"windows": window_metrics}

    # ── Aggregate ─────────────────────────────────────────────────────
    results_df = pd.concat(all_results, ignore_index=True)

    print(f"\n{'='*60}")
    print("AGGREGATE RESULTS")
    print(f"{'='*60}")

    overall_acc = accuracy_score(results_df["y_true"], (results_df["y_pred"] > 0.5).astype(int))
    overall_auc = roc_auc_score(results_df["y_true"], results_df["y_pred"])
    print(f"Overall accuracy: {overall_acc:.4f}")
    print(f"Overall AUC: {overall_auc:.4f}")
    print(f"Total test samples: {len(results_df):,}")

    # Unique windows
    n_windows_tested = results_df["window_id"].nunique()
    print(f"Unique 5-min windows tested: {n_windows_tested:,}")

    # Accuracy by minute position (if not filtered)
    if minute_filter is None and "minute_in_window" not in results_df.columns:
        # Re-derive minute from timestamp
        results_df["_minute"] = ((results_df["timestamp_ms"] - results_df["window_id"]) // 60_000).astype(int)
        print("\nAccuracy by minute in window:")
        for m in range(5):
            mask = results_df["_minute"] == m
            if mask.sum() > 0:
                sub = results_df[mask]
                acc = accuracy_score(sub["y_true"], (sub["y_pred"] > 0.5).astype(int))
                print(f"  Minute {m}: {acc:.4f} (n={mask.sum():,})")

    # Confidence threshold sweep
    print("\nConfidence threshold sweep:")
    print(f"  {'Threshold':>10} {'Acc':>8} {'N':>10} {'Windows':>10}")
    for thresh in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]:
        mask = (results_df["y_pred"] > thresh) | (results_df["y_pred"] < (1 - thresh))
        if mask.sum() == 0:
            continue
        sub = results_df[mask]
        pred = (sub["y_pred"] > 0.5).astype(int)
        acc = accuracy_score(sub["y_true"], pred)
        n_win = sub["window_id"].nunique()
        print(f"  {thresh:>10.2f} {acc:>8.4f} {len(sub):>10,} {n_win:>10,}")

    # Window-level accuracy (one prediction per window, using the first candle)
    first_per_window = results_df.groupby("window_id").first().reset_index()
    window_acc = accuracy_score(
        first_per_window["y_true"],
        (first_per_window["y_pred"] > 0.5).astype(int)
    )
    print(f"\nWindow-level accuracy (1 pred per window): {window_acc:.4f} ({len(first_per_window):,} windows)")

    # Per walk-forward window summary
    print("\nPer-window summary:")
    for wm in window_metrics:
        print(f"  W{wm['window']}: acc={wm['test_acc']:.4f} auc={wm['test_auc']:.4f} n={wm['test_n']:,} ({wm['test_range']})")

    # Save
    results_df.to_parquet(results_dir / "predictions.parquet", index=False)
    with open(results_dir / "window_metrics.json", "w") as f:
        json.dump(window_metrics, f, indent=2)

    print(f"\nResults saved to {results_dir}")

    return {
        "windows": window_metrics,
        "overall_acc": float(overall_acc),
        "overall_auc": float(overall_auc),
        "window_acc": float(window_acc),
        "total_samples": len(results_df),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Walk-forward training for Polymarket 5-min BTC prediction"
    )
    parser.add_argument("--data-dir", default="data/cryptolake",
                        help="CryptoLake data directory")
    parser.add_argument("--symbol", default="BTC-USDT-PERP",
                        help="Binance Futures symbol")
    parser.add_argument("--results-dir", default="data/polymarket_5m_results",
                        help="Output directory for predictions")
    parser.add_argument("--model-dir", default="data/polymarket_5m_models",
                        help="Output directory for models")
    parser.add_argument("--train-days", type=int, default=60,
                        help="Training window in days")
    parser.add_argument("--test-days", type=int, default=7,
                        help="Test window in days")
    parser.add_argument("--purge-windows", type=int, default=2,
                        help="Gap between train/test in 5-min windows (default 2 = 10min)")
    parser.add_argument("--minute", type=int, default=None, choices=[0, 1, 2, 3, 4],
                        help="Only use candles at this minute position (0=window open, 4=last)")
    parser.add_argument("--n-jobs", type=int, default=None,
                        help="XGBoost threads per seed")
    args = parser.parse_args()

    walk_forward(args)


if __name__ == "__main__":
    main()
