#!/usr/bin/env python3
"""
Optuna hyperparameter sweep for XGBoost confidence model.

Reuses the data pipeline from xgboost_flexible.py (cached features).
Optimizes on validation set AUC-ROC, reports final results on held-out test set.

Usage:
    python xgb_sweep.py                    # 100 trials (default)
    python xgb_sweep.py --n-trials 200     # more trials
    python xgb_sweep.py --save-best        # save best model + metadata to models/
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import xgboost as xgb
except ImportError:
    print("ERROR: xgboost not installed. Run: pip install xgboost")
    sys.exit(1)

try:
    import optuna
except ImportError:
    print("ERROR: optuna not installed. Run: pip install optuna")
    sys.exit(1)

from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

from xgboost_flexible import (
    build_dataset,
    get_feature_columns,
    compute_vol_threshold,
    FEATURE_CACHE_PATH,
)

warnings.filterwarnings("ignore", category=FutureWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------------------------
# Data split (identical to xgboost_flexible.py)
# ---------------------------------------------------------------------------

def split_dataset(dataset: pd.DataFrame, train_frac=0.7, val_frac=0.15):
    feature_cols = get_feature_columns(dataset)

    slug_order = dataset.groupby("slug")["open_ts"].first().sort_values()
    n_markets = len(slug_order)
    train_end = int(n_markets * train_frac)
    val_end = int(n_markets * (train_frac + val_frac))

    train_slugs = set(slug_order.index[:train_end])
    val_slugs = set(slug_order.index[train_end:val_end])
    test_slugs = set(slug_order.index[val_end:])

    train_df = dataset[dataset["slug"].isin(train_slugs)]
    val_df = dataset[dataset["slug"].isin(val_slugs)]
    test_df = dataset[dataset["slug"].isin(test_slugs)]

    def to_dmatrix(df):
        X = df[feature_cols].values.astype(np.float32)
        X[~np.isfinite(X)] = 0.0
        y = df["label"].values
        return xgb.DMatrix(X, label=y, feature_names=feature_cols), y

    dtrain, y_train = to_dmatrix(train_df)
    dval, y_val = to_dmatrix(val_df)
    dtest, y_test = to_dmatrix(test_df)

    return dtrain, y_train, dval, y_val, dtest, y_test, train_df, val_df, test_df, feature_cols


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------

def create_objective(dtrain, y_train, dval, y_val):
    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "verbosity": 0,
            "seed": 42,

            # Tree structure
            "max_depth": trial.suggest_int("max_depth", 2, 7),
            "min_child_weight": trial.suggest_int("min_child_weight", 5, 50),

            # Learning
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 3000),

            # Sampling
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),

            # Regularization
            "lambda": trial.suggest_float("lambda", 0.1, 10.0, log=True),
            "alpha": trial.suggest_float("alpha", 0.01, 5.0, log=True),

            # Gamma (min split loss)
            "gamma": trial.suggest_float("gamma", 0.0, 2.0),
        }

        n_rounds = params.pop("n_estimators")

        model = xgb.train(
            params, dtrain,
            num_boost_round=n_rounds,
            evals=[(dval, "val")],
            early_stopping_rounds=50,
            verbose_eval=False,
        )

        y_pred = model.predict(dval)

        # Primary objective: AUC-ROC on validation set
        auc = roc_auc_score(y_val, y_pred)

        # Store useful attrs for later analysis
        trial.set_user_attr("best_iteration", model.best_iteration)
        trial.set_user_attr("val_logloss", log_loss(y_val, y_pred))

        # Also check WR at conf >= 0.70 (our operational metric)
        mask_70 = y_pred >= 0.70
        if mask_70.sum() > 10:
            trial.set_user_attr("val_wr_70", float(y_val[mask_70].mean()))
            trial.set_user_attr("val_n_70", int(mask_70.sum()))
        else:
            trial.set_user_attr("val_wr_70", float("nan"))
            trial.set_user_attr("val_n_70", 0)

        return auc

    return objective


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_params(params, dtrain, y_train, dval, y_val, dtest, y_test, test_df, feature_cols):
    """Train with given params and evaluate on test set."""
    n_rounds = params.pop("n_estimators", 2000)

    model = xgb.train(
        params, dtrain,
        num_boost_round=n_rounds,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=50,
        verbose_eval=200,
    )

    y_pred = model.predict(dtest)
    auc = roc_auc_score(y_test, y_pred)
    ll = log_loss(y_test, y_pred)
    acc = accuracy_score(y_test, (y_pred >= 0.5).astype(int))

    print(f"\n{'='*70}")
    print("BEST PARAMS — TEST SET RESULTS")
    print(f"{'='*70}")
    print(f"  Best iteration: {model.best_iteration}")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  AUC-ROC:   {auc:.4f}")
    print(f"  Log Loss:  {ll:.4f}")

    # Threshold sweep
    print(f"\n{'Threshold':>10s}  {'Trades':>7s}  {'Win%':>6s}  {'EV/trade':>10s}  {'Coverage':>10s}")
    print("-" * 55)

    for threshold in [0.55, 0.60, 0.65, 0.70, 0.72, 0.75, 0.80]:
        mask = y_pred >= threshold
        n_trades = mask.sum()
        if n_trades < 10:
            continue
        win_rate = y_test[mask].mean()
        mids = test_df.iloc[np.where(mask)[0]]
        mid_col = [c for c in test_df.columns if c.endswith("_mid") and c.startswith("w")]
        if mid_col:
            avg_mid = mids[sorted(mid_col)[-1]].mean()
        else:
            avg_mid = 0.60
        ev = win_rate - avg_mid
        coverage = n_trades / len(y_test)
        print(f"  {threshold:>8.2f}  {n_trades:>7d}  {win_rate:>5.1%}  {ev:>+9.4f}    {coverage:>8.1%}")

    # Feature importance
    print(f"\n{'='*70}")
    print("TOP 20 FEATURE IMPORTANCE (gain)")
    print(f"{'='*70}")
    importance = model.get_score(importance_type="gain")
    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:20]
    max_gain = sorted_imp[0][1] if sorted_imp else 1.0
    for feat, gain in sorted_imp:
        bar = "#" * int(gain / max_gain * 40)
        print(f"  {feat:<32s} {gain:>10.1f}  {bar}")

    return model, y_pred


# ---------------------------------------------------------------------------
# Comparison with current params
# ---------------------------------------------------------------------------

CURRENT_PARAMS = {
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

def compare_with_current(dtrain, y_train, dval, y_val, dtest, y_test, test_df, feature_cols, best_params):
    print(f"\n{'='*70}")
    print("COMPARISON: CURRENT vs BEST SWEEP PARAMS")
    print(f"{'='*70}")

    results = {}
    for label, params in [("Current (v3)", {**CURRENT_PARAMS, "n_estimators": 2000}),
                           ("Sweep best", {**best_params})]:
        p = {**params}
        n_rounds = p.pop("n_estimators", 2000)

        model = xgb.train(
            p, dtrain,
            num_boost_round=n_rounds,
            evals=[(dval, "val")],
            early_stopping_rounds=50,
            verbose_eval=False,
        )

        y_pred = model.predict(dtest)
        auc = roc_auc_score(y_test, y_pred)

        row = {"auc": auc, "best_iter": model.best_iteration}

        for t in [0.65, 0.70, 0.72, 0.75]:
            m = y_pred >= t
            if m.sum() > 5:
                row[f"wr_{t}"] = y_test[m].mean()
                row[f"n_{t}"] = m.sum()
            else:
                row[f"wr_{t}"] = float("nan")
                row[f"n_{t}"] = 0

        results[label] = row

    # Print comparison table
    print(f"\n{'Metric':<20s}", end="")
    for label in results:
        print(f"  {label:>16s}", end="")
    print()
    print("-" * 56)

    print(f"{'AUC-ROC':<20s}", end="")
    for r in results.values():
        print(f"  {r['auc']:>16.4f}", end="")
    print()

    print(f"{'Best iteration':<20s}", end="")
    for r in results.values():
        print(f"  {r['best_iter']:>16d}", end="")
    print()

    for t in [0.65, 0.70, 0.72, 0.75]:
        print(f"{'WR @ ' + str(t):<20s}", end="")
        for r in results.values():
            wr = r.get(f"wr_{t}", float("nan"))
            n = r.get(f"n_{t}", 0)
            print(f"  {wr:>6.1%} (n={n:>4d})", end="")
        print()


# ---------------------------------------------------------------------------
# Save model
# ---------------------------------------------------------------------------

def save_best_model(model, params, feature_cols, train_df, vol_threshold):
    out_dir = Path("models")
    out_dir.mkdir(exist_ok=True)

    model_path = out_dir / "xgb_sweep_best.json"
    model.save_model(str(model_path))

    metadata = {
        "version": "v3.1-sweep",
        "params": params,
        "feature_columns": feature_cols,
        "n_features": len(feature_cols),
        "btc_mid_mean": float(train_df["btc_mid_zscore"].mean()) if "btc_mid_zscore" in train_df.columns else 0.0,
        "btc_mid_std": 1.0,
        "vol_threshold": vol_threshold,
        "confidence_threshold": 0.72,
    }

    meta_path = out_dir / "xgb_sweep_best_meta.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved model to {model_path}")
    print(f"Saved metadata to {meta_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="data/telonex_book_snapshots")
    parser.add_argument("--resolution-cache", default="data/resolution_cache.json")
    parser.add_argument("--btc-path", default="data/btc_quotes/btcusdt_quotes.parquet")
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--save-best", action="store_true")
    args = parser.parse_args()

    print("XGBoost Hyperparameter Sweep (Optuna)")
    print("=" * 50)
    print(f"  Trials: {args.n_trials}")
    print()

    # Load cached features (fast path)
    dataset = build_dataset(
        data_dir=Path(args.data_dir),
        resolution_cache_path=Path(args.resolution_cache),
        btc_path=Path(args.btc_path),
        multi_window=True,
    )

    print(f"\nDataset: {dataset.shape[0]} samples, {dataset['slug'].nunique()} markets")
    print(f"Base rate: {dataset['label'].mean():.3f}")

    # Split
    dtrain, y_train, dval, y_val, dtest, y_test, train_df, val_df, test_df, feature_cols = \
        split_dataset(dataset)

    print(f"Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")
    print(f"Features: {len(feature_cols)}")

    # Run sweep
    print(f"\nStarting {args.n_trials}-trial sweep (optimizing val AUC-ROC)...")
    study = optuna.create_study(direction="maximize", study_name="xgb-sweep")
    study.optimize(create_objective(dtrain, y_train, dval, y_val), n_trials=args.n_trials)

    # Results
    best = study.best_trial
    print(f"\n{'='*70}")
    print("SWEEP RESULTS")
    print(f"{'='*70}")
    print(f"  Best val AUC: {best.value:.4f}")
    print(f"  Best val WR@0.70: {best.user_attrs.get('val_wr_70', 'n/a')}")
    print(f"  Best iteration: {best.user_attrs.get('best_iteration', 'n/a')}")

    print(f"\n  Best params:")
    for k, v in sorted(best.params.items()):
        print(f"    {k}: {v}")

    # Reconstruct full params dict
    best_params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "verbosity": 0,
        "seed": 42,
        **best.params,
    }
    # n_estimators was part of the trial but needs to be passed separately
    best_params["n_estimators"] = best.params.get("n_estimators", 2000)

    # Top 10 trials
    print(f"\n  Top 10 trials:")
    print(f"  {'#':>4s}  {'AUC':>6s}  {'WR@70':>7s}  {'N@70':>6s}  {'depth':>5s}  {'lr':>8s}  {'sub':>5s}  {'col':>5s}  {'mcw':>5s}  {'λ':>6s}  {'α':>6s}  {'γ':>5s}")
    print("  " + "-" * 90)
    top_trials = sorted(study.trials, key=lambda t: t.value, reverse=True)[:10]
    for t in top_trials:
        p = t.params
        wr70 = t.user_attrs.get("val_wr_70", float("nan"))
        n70 = t.user_attrs.get("val_n_70", 0)
        print(f"  {t.number:>4d}  {t.value:>.4f}  {wr70:>6.1%}  {n70:>6d}  "
              f"{p['max_depth']:>5d}  {p['learning_rate']:>8.4f}  {p['subsample']:>.3f}  "
              f"{p['colsample_bytree']:>.3f}  {p['min_child_weight']:>5d}  "
              f"{p['lambda']:>6.3f}  {p['alpha']:>6.3f}  {p['gamma']:>5.2f}")

    # Evaluate best on test set
    eval_params = {k: v for k, v in best_params.items()}
    model, y_pred = evaluate_params(
        eval_params, dtrain, y_train, dval, y_val, dtest, y_test, test_df, feature_cols
    )

    # Compare with current production params
    compare_with_current(dtrain, y_train, dval, y_val, dtest, y_test, test_df, feature_cols, best_params)

    # Vol threshold from training data
    vol_threshold = compute_vol_threshold(train_df)
    if vol_threshold:
        print(f"\nVol threshold (p33 train): {vol_threshold:.6f}")

    # Save if requested
    if args.save_best:
        save_best_model(model, best_params, feature_cols, train_df, vol_threshold)

    # Print ready-to-use params block
    print(f"\n{'='*70}")
    print("COPY-PASTE PARAMS FOR xgboost_flexible.py")
    print(f"{'='*70}")
    print("params = {")
    print('    "objective": "binary:logistic",')
    print('    "eval_metric": "logloss",')
    for k in ["max_depth", "learning_rate", "subsample", "colsample_bytree",
              "min_child_weight", "lambda", "alpha", "gamma"]:
        v = best.params[k]
        if isinstance(v, int):
            print(f'    "{k}": {v},')
        else:
            print(f'    "{k}": {v:.6f},')
    print('    "seed": 42,')
    print('    "verbosity": 0,')
    print("}")
    print(f"# num_boost_round = {best.params.get('n_estimators', 2000)} (with early_stopping_rounds=50)")


if __name__ == "__main__":
    main()
