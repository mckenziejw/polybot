#!/usr/bin/env python3
"""
Optuna sweep on deduped feature set (drop abs_drift, other_mid, mid_sum).
Keeps max_drift — decorrelates from drift_from_50 during mean reversions.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import optuna

from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

from xgboost_flexible import build_dataset, get_feature_columns, compute_vol_threshold
from xgb_sweep import CURRENT_PARAMS

warnings.filterwarnings("ignore", category=FutureWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

DROP_FEATURES = ["abs_drift", "other_mid", "mid_sum"]

# ---------------------------------------------------------------------------

def split_dedup(dataset: pd.DataFrame, train_frac=0.7, val_frac=0.15):
    all_cols = get_feature_columns(dataset)
    feature_cols = [c for c in all_cols if c not in DROP_FEATURES]

    slug_order = dataset.groupby("slug")["open_ts"].first().sort_values()
    n = len(slug_order)
    te, ve = int(n * train_frac), int(n * (train_frac + val_frac))

    train_slugs = set(slug_order.index[:te])
    val_slugs = set(slug_order.index[te:ve])
    test_slugs = set(slug_order.index[ve:])

    def to_dm(df):
        X = df[feature_cols].values.astype(np.float32)
        X[~np.isfinite(X)] = 0.0
        y = df["label"].values
        return xgb.DMatrix(X, label=y, feature_names=feature_cols), y

    train_df = dataset[dataset["slug"].isin(train_slugs)]
    val_df = dataset[dataset["slug"].isin(val_slugs)]
    test_df = dataset[dataset["slug"].isin(test_slugs)]

    return (*to_dm(train_df), *to_dm(val_df), *to_dm(test_df),
            train_df, val_df, test_df, feature_cols)


def create_objective(dtrain, y_train, dval, y_val):
    def objective(trial):
        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "verbosity": 0, "seed": 42,
            "max_depth": trial.suggest_int("max_depth", 2, 7),
            "min_child_weight": trial.suggest_int("min_child_weight", 5, 50),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 3000),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "lambda": trial.suggest_float("lambda", 0.1, 10.0, log=True),
            "alpha": trial.suggest_float("alpha", 0.01, 5.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 2.0),
        }
        nr = params.pop("n_estimators")
        model = xgb.train(params, dtrain, num_boost_round=nr,
                          evals=[(dval, "val")], early_stopping_rounds=50,
                          verbose_eval=False)
        yp = model.predict(dval)
        auc = roc_auc_score(y_val, yp)

        trial.set_user_attr("best_iteration", model.best_iteration)
        m70 = yp >= 0.70
        trial.set_user_attr("val_wr_70", float(y_val[m70].mean()) if m70.sum() > 10 else float("nan"))
        trial.set_user_attr("val_n_70", int(m70.sum()))
        return auc
    return objective


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=100)
    args = parser.parse_args()

    print("XGBoost Sweep — Deduped Features")
    print(f"Dropping: {DROP_FEATURES}")
    print("="*60)

    dataset = build_dataset(
        data_dir=Path("data/telonex_book_snapshots"),
        resolution_cache_path=Path("data/resolution_cache.json"),
        btc_path=Path("data/btc_quotes/btcusdt_quotes.parquet"),
        multi_window=True,
    )

    dtrain, y_train, dval, y_val, dtest, y_test, train_df, val_df, test_df, feature_cols = \
        split_dedup(dataset)

    print(f"Dataset: {len(dataset)} samples, {dataset['slug'].nunique()} markets")
    print(f"Features: {len(feature_cols)} (was {len(get_feature_columns(dataset))})")
    print(f"Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")

    # --- Sweep ---
    print(f"\nRunning {args.n_trials}-trial sweep...")
    study = optuna.create_study(direction="maximize", study_name="xgb-dedup-sweep")
    study.optimize(create_objective(dtrain, y_train, dval, y_val), n_trials=args.n_trials)

    best = study.best_trial
    print(f"\n{'='*70}")
    print("SWEEP RESULTS (deduped features)")
    print(f"{'='*70}")
    print(f"  Best val AUC: {best.value:.4f}")
    print(f"  Val WR@0.70: {best.user_attrs.get('val_wr_70', 'n/a')}")

    print(f"\n  Best params:")
    for k, v in sorted(best.params.items()):
        print(f"    {k}: {v}")

    # Top 10 trials
    print(f"\n  Top 10 trials:")
    print(f"  {'#':>4s}  {'AUC':>6s}  {'WR@70':>7s}  {'N@70':>6s}  {'depth':>5s}  {'lr':>8s}  {'sub':>5s}  {'col':>5s}  {'mcw':>5s}  {'λ':>6s}  {'α':>6s}  {'γ':>5s}")
    print("  " + "-" * 90)
    for t in sorted(study.trials, key=lambda t: t.value, reverse=True)[:10]:
        p = t.params
        wr70 = t.user_attrs.get("val_wr_70", float("nan"))
        n70 = t.user_attrs.get("val_n_70", 0)
        print(f"  {t.number:>4d}  {t.value:>.4f}  {wr70:>6.1%}  {n70:>6d}  "
              f"{p['max_depth']:>5d}  {p['learning_rate']:>8.4f}  {p['subsample']:>.3f}  "
              f"{p['colsample_bytree']:>.3f}  {p['min_child_weight']:>5d}  "
              f"{p['lambda']:>6.3f}  {p['alpha']:>6.3f}  {p['gamma']:>5.2f}")

    # --- Evaluate best on test ---
    best_params = {
        "objective": "binary:logistic", "eval_metric": "logloss",
        "verbosity": 0, "seed": 42,
        **{k: v for k, v in best.params.items() if k != "n_estimators"},
    }
    nr = best.params.get("n_estimators", 2000)

    model_best = xgb.train(best_params, dtrain, num_boost_round=nr,
                           evals=[(dtrain, "train"), (dval, "val")],
                           early_stopping_rounds=50, verbose_eval=200)
    y_best = model_best.predict(dtest)

    # --- Also train current v3 on deduped features for fair comparison ---
    model_curr = xgb.train(CURRENT_PARAMS, dtrain, num_boost_round=2000,
                           evals=[(dval, "val")], early_stopping_rounds=50,
                           verbose_eval=False)
    y_curr = model_curr.predict(dtest)

    # --- And train current v3 on FULL features (the actual production baseline) ---
    all_cols = get_feature_columns(dataset)
    slug_order = dataset.groupby("slug")["open_ts"].first().sort_values()
    n_m = len(slug_order)
    te, ve = int(n_m * 0.7), int(n_m * 0.85)
    test_slugs_full = set(slug_order.index[ve:])
    train_full = dataset[dataset["slug"].isin(set(slug_order.index[:te]))]
    val_full = dataset[dataset["slug"].isin(set(slug_order.index[te:ve]))]
    test_full = dataset[dataset["slug"].isin(test_slugs_full)]

    def full_dm(df):
        X = df[all_cols].values.astype(np.float32)
        X[~np.isfinite(X)] = 0.0
        return xgb.DMatrix(X, label=df["label"].values, feature_names=all_cols)

    model_prod = xgb.train(CURRENT_PARAMS, full_dm(train_full), num_boost_round=2000,
                           evals=[(full_dm(val_full), "val")], early_stopping_rounds=50,
                           verbose_eval=False)
    y_prod = model_prod.predict(full_dm(test_full))
    y_test_full = test_full["label"].values

    w120 = test_df["observe_time_s"].values == 120
    w120_full = test_full["observe_time_s"].values == 120

    mid_col = sorted([c for c in test_df.columns if c.endswith("_mid") and c.startswith("w")])[-1]

    print(f"\n{'='*70}")
    print("THREE-WAY COMPARISON (all on test set, 120s window)")
    print(f"{'='*70}")
    print(f"  A = Current v3 params, full features (actual production)")
    print(f"  B = Current v3 params, deduped features")
    print(f"  C = Sweep-best params, deduped features")

    print(f"\n{'Metric':<20s}  {'A (prod)':>16s}  {'B (v3 dedup)':>16s}  {'C (sweep dedup)':>16s}")
    print("-" * 74)

    auc_a = roc_auc_score(y_test_full, y_prod)
    auc_b = roc_auc_score(y_test, y_curr)
    auc_c = roc_auc_score(y_test, y_best)
    print(f"{'AUC-ROC':<20s}  {auc_a:>16.4f}  {auc_b:>16.4f}  {auc_c:>16.4f}")
    print(f"{'Best iteration':<20s}  {model_prod.best_iteration:>16d}  {model_curr.best_iteration:>16d}  {model_best.best_iteration:>16d}")

    for thresh in [0.65, 0.70, 0.72, 0.75, 0.80]:
        parts = []
        for yp, w, yt in [(y_prod, w120_full, y_test_full), (y_curr, w120, y_test), (y_best, w120, y_test)]:
            m = w & (yp >= thresh)
            n = m.sum()
            wr = yt[m].mean() if n > 5 else float("nan")
            parts.append(f"{wr:>6.1%} (n={n:>4d})")
        print(f"{'WR@' + str(thresh) + ' (120s)':<20s}  {'  '.join(parts)}")

    # EV analysis at 0.72
    print(f"\n{'='*70}")
    print("EV BY ENTRY PRICE (conf >= 0.72, 120s window)")
    print(f"{'='*70}")

    for label, yp, w, yt, tdf in [
        ("A (prod)", y_prod, w120_full, y_test_full, test_full),
        ("B (v3 dedup)", y_curr, w120, y_test, test_df),
        ("C (sweep dedup)", y_best, w120, y_test, test_df),
    ]:
        m = w & (yp >= 0.72)
        if m.sum() < 10:
            continue
        mids = tdf.loc[m, mid_col].values
        wins = yt[m]
        avg_entry = mids.mean()
        wr = wins.mean()
        ev = wr - avg_entry

        print(f"\n  --- {label} (n={m.sum()}) ---")
        print(f"  Avg entry: {avg_entry:.3f}, WR: {wr:.1%}, EV: {ev:+.4f}")
        print(f"  {'Entry range':<16s}  {'N':>5s}  {'WR':>6s}  {'AvgMid':>8s}  {'EV':>10s}")
        print(f"  " + "-" * 50)
        for lo, hi in [(0.50, 0.55), (0.55, 0.60), (0.60, 0.65), (0.65, 0.70),
                       (0.70, 0.75), (0.75, 0.80), (0.80, 0.90), (0.90, 1.0)]:
            b = (mids >= lo) & (mids < hi)
            nb = b.sum()
            if nb < 3:
                continue
            bwr = wins[b].mean()
            bm = mids[b].mean()
            print(f"  [{lo:.2f}, {hi:.2f})    {nb:>5d}  {bwr:>5.1%}  {bm:>8.3f}  {bwr - bm:>+9.4f}")

    # Feature importance
    print(f"\n{'='*70}")
    print("TOP 20 FEATURE IMPORTANCE — SWEEP DEDUP MODEL (gain)")
    print(f"{'='*70}")
    imp = model_best.get_score(importance_type="gain")
    sorted_imp = sorted(imp.items(), key=lambda x: x[1], reverse=True)[:20]
    mx = sorted_imp[0][1] if sorted_imp else 1.0
    for feat, gain in sorted_imp:
        bar = "#" * int(gain / mx * 40)
        print(f"  {feat:<32s} {gain:>10.1f}  {bar}")

    # Vol threshold
    vol_t = compute_vol_threshold(train_df)
    if vol_t:
        print(f"\nVol threshold (p33 train): {vol_t:.6f}")

    # Copy-paste params
    print(f"\n{'='*70}")
    print("COPY-PASTE PARAMS")
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
    print(f"# num_boost_round = {best.params.get('n_estimators', 2000)}")
    print(f"# Drop features: {DROP_FEATURES}")


if __name__ == "__main__":
    main()
