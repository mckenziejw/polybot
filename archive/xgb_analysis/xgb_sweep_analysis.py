#!/usr/bin/env python3
"""
Analyze sweep results: entry price distribution and feature correlation impact.

1. Entry price analysis: what prices does the model actually trade at?
2. Correlation analysis: does drift feature redundancy hurt?
3. Test: retrain with deduplicated drift features, compare.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
from sklearn.metrics import roc_auc_score

from xgboost_flexible import build_dataset, get_feature_columns, OBSERVATION_WINDOWS
from xgb_sweep import split_dataset, CURRENT_PARAMS

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

print("Loading cached features...")
dataset = build_dataset(
    data_dir=Path("data/telonex_book_snapshots"),
    resolution_cache_path=Path("data/resolution_cache.json"),
    btc_path=Path("data/btc_quotes/btcusdt_quotes.parquet"),
    multi_window=True,
)

dtrain, y_train, dval, y_val, dtest, y_test, train_df, val_df, test_df, feature_cols = \
    split_dataset(dataset)

# ---------------------------------------------------------------------------
# 1. Entry price analysis
# ---------------------------------------------------------------------------

SWEEP_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "max_depth": 2,
    "learning_rate": 0.008005,
    "subsample": 0.925995,
    "colsample_bytree": 0.848806,
    "min_child_weight": 16,
    "lambda": 7.466019,
    "alpha": 0.259967,
    "gamma": 0.588038,
    "seed": 42,
    "verbosity": 0,
}

print("\n" + "="*70)
print("ENTRY PRICE ANALYSIS")
print("="*70)

# Get the last window mid column (this is the entry price proxy)
mid_cols = sorted([c for c in test_df.columns if c.endswith("_mid") and c.startswith("w")])
last_mid_col = mid_cols[-1] if mid_cols else None
print(f"Using {last_mid_col} as entry price proxy")

for label, params in [("Current v3", CURRENT_PARAMS), ("Sweep best", SWEEP_PARAMS)]:
    model = xgb.train(
        params, dtrain,
        num_boost_round=2000,
        evals=[(dval, "val")],
        early_stopping_rounds=50,
        verbose_eval=False,
    )
    y_pred = model.predict(dtest)

    print(f"\n--- {label} ---")

    # Only look at 120s window (matching live behavior)
    w120 = test_df["observe_time_s"].values == 120

    for thresh in [0.70, 0.72, 0.75]:
        mask = w120 & (y_pred >= thresh)
        if mask.sum() < 10:
            continue

        mids = test_df.loc[mask, last_mid_col].values
        wins = y_test[mask]

        # Entry price buckets
        print(f"\n  Threshold >= {thresh} ({mask.sum()} trades):")
        print(f"  {'Entry range':<16s}  {'N':>5s}  {'WR':>6s}  {'AvgMid':>8s}  {'EV/trade':>10s}")
        print(f"  " + "-" * 50)

        for lo, hi in [(0.50, 0.55), (0.55, 0.60), (0.60, 0.65), (0.65, 0.70),
                       (0.70, 0.75), (0.75, 0.80), (0.80, 0.90), (0.90, 1.0)]:
            bucket = (mids >= lo) & (mids < hi)
            n = bucket.sum()
            if n < 3:
                continue
            wr = wins[bucket].mean()
            avg_mid = mids[bucket].mean()
            # EV = WR * (1 - entry) - (1 - WR) * entry = WR - entry
            ev = wr - avg_mid
            print(f"  [{lo:.2f}, {hi:.2f})    {n:>5d}  {wr:>5.1%}  {avg_mid:>8.3f}  {ev:>+9.4f}")

        # Overall
        avg_entry = mids.mean()
        wr_all = wins.mean()
        ev_all = wr_all - avg_entry
        print(f"  {'Overall':<16s}  {mask.sum():>5d}  {wr_all:>5.1%}  {avg_entry:>8.3f}  {ev_all:>+9.4f}")

# ---------------------------------------------------------------------------
# 2. Feature correlation analysis
# ---------------------------------------------------------------------------

print(f"\n\n{'='*70}")
print("FEATURE CORRELATION ANALYSIS")
print("="*70)

# Compute correlation matrix for top features
drift_features = ["drift_from_50", "abs_drift", "other_mid", "max_drift", "mid_sum"]
available = [f for f in drift_features if f in test_df.columns]

if available:
    corr = test_df[available].corr()
    print("\nCorrelation matrix of drift-related features:")
    print(corr.round(3).to_string())

# Also check correlation of drift features with other top features
other_top = ["choppiness", "r_squared", "btc_ret_since_open", "btc_vol_15m",
             "w10_ask_depth_3", "w20_imbalance", "w20_mid_delta"]
other_avail = [f for f in other_top if f in test_df.columns]

if available and other_avail:
    print("\nCorrelation of drift features with other top features:")
    cross_corr = test_df[available + other_avail].corr().loc[available, other_avail]
    print(cross_corr.round(3).to_string())

# ---------------------------------------------------------------------------
# 3. Deduplicated model: drop redundant drift features
# ---------------------------------------------------------------------------

print(f"\n\n{'='*70}")
print("DEDUP EXPERIMENT: KEEP ONLY drift_from_50 (DROP abs_drift, other_mid)")
print("="*70)

# Strategy: keep drift_from_50, drop abs_drift and other_mid (the two most correlated)
drop_features = ["abs_drift", "other_mid"]
dedup_cols = [c for c in feature_cols if c not in drop_features]
print(f"Original features: {len(feature_cols)}, After dedup: {len(dedup_cols)}")

def make_dmatrix(df, cols):
    X = df[cols].values.astype(np.float32)
    X[~np.isfinite(X)] = 0.0
    return xgb.DMatrix(X, label=df["label"].values, feature_names=cols)

dtrain_d = make_dmatrix(train_df, dedup_cols)
dval_d = make_dmatrix(val_df, dedup_cols)
dtest_d = make_dmatrix(test_df, dedup_cols)

# Train with sweep params on deduped features
model_dedup = xgb.train(
    SWEEP_PARAMS, dtrain_d,
    num_boost_round=2000,
    evals=[(dval_d, "val")],
    early_stopping_rounds=50,
    verbose_eval=False,
)

# Train with sweep params on full features (baseline)
model_full = xgb.train(
    SWEEP_PARAMS, dtrain,
    num_boost_round=2000,
    evals=[(dval, "val")],
    early_stopping_rounds=50,
    verbose_eval=False,
)

y_full = model_full.predict(dtest)
y_dedup = model_dedup.predict(dtest_d)

print(f"\n{'Metric':<20s}  {'Full features':>16s}  {'Deduped':>16s}")
print("-" * 56)
print(f"{'AUC-ROC':<20s}  {roc_auc_score(y_test, y_full):>16.4f}  {roc_auc_score(y_test, y_dedup):>16.4f}")
print(f"{'Best iteration':<20s}  {model_full.best_iteration:>16d}  {model_dedup.best_iteration:>16d}")

w120_test = test_df["observe_time_s"].values == 120

for thresh in [0.65, 0.70, 0.72, 0.75]:
    for y_pred, name in [(y_full, "Full"), (y_dedup, "Dedup")]:
        m = w120_test & (y_pred >= thresh)
        n = m.sum()
        wr = y_test[m].mean() if n > 5 else float("nan")
        if name == "Full":
            print(f"{'WR@' + str(thresh) + ' (120s)':<20s}  {wr:>6.1%} (n={n:>4d})", end="")
        else:
            print(f"  {wr:>6.1%} (n={n:>4d})")

# Feature importance comparison
print(f"\nTop 15 features (dedup model):")
imp_dedup = model_dedup.get_score(importance_type="gain")
sorted_imp = sorted(imp_dedup.items(), key=lambda x: x[1], reverse=True)[:15]
max_gain = sorted_imp[0][1] if sorted_imp else 1.0
for feat, gain in sorted_imp:
    bar = "#" * int(gain / max_gain * 40)
    print(f"  {feat:<32s} {gain:>10.1f}  {bar}")

# ---------------------------------------------------------------------------
# 4. More aggressive dedup: also drop mid_sum, max_drift
# ---------------------------------------------------------------------------

print(f"\n\n{'='*70}")
print("AGGRESSIVE DEDUP: DROP abs_drift, other_mid, mid_sum, max_drift")
print("="*70)

drop_aggressive = ["abs_drift", "other_mid", "mid_sum", "max_drift"]
agg_cols = [c for c in feature_cols if c not in drop_aggressive]
print(f"Features: {len(feature_cols)} → {len(agg_cols)}")

dtrain_a = make_dmatrix(train_df, agg_cols)
dval_a = make_dmatrix(val_df, agg_cols)
dtest_a = make_dmatrix(test_df, agg_cols)

model_agg = xgb.train(
    SWEEP_PARAMS, dtrain_a,
    num_boost_round=2000,
    evals=[(dval_a, "val")],
    early_stopping_rounds=50,
    verbose_eval=False,
)

y_agg = model_agg.predict(dtest_a)

print(f"\n{'Metric':<20s}  {'Full':>16s}  {'Drop 2':>16s}  {'Drop 4':>16s}")
print("-" * 72)
print(f"{'AUC-ROC':<20s}  {roc_auc_score(y_test, y_full):>16.4f}  {roc_auc_score(y_test, y_dedup):>16.4f}  {roc_auc_score(y_test, y_agg):>16.4f}")

for thresh in [0.70, 0.72, 0.75]:
    parts = []
    for y_pred in [y_full, y_dedup, y_agg]:
        m = w120_test & (y_pred >= thresh)
        n = m.sum()
        wr = y_test[m].mean() if n > 5 else float("nan")
        parts.append(f"{wr:>6.1%} (n={n:>4d})")
    print(f"{'WR@' + str(thresh) + ' (120s)':<20s}  {'  '.join(parts)}")

# Entry price comparison at 0.72
print(f"\nEntry price distribution at conf >= 0.72 (120s window):")
for y_pred, name in [(y_full, "Full"), (y_dedup, "Drop 2"), (y_agg, "Drop 4")]:
    m = w120_test & (y_pred >= 0.72)
    if m.sum() < 5:
        continue
    mids = test_df.loc[m, last_mid_col].values
    print(f"  {name:<8s}: n={m.sum():>4d}  avg_entry={mids.mean():.3f}  "
          f"p25={np.percentile(mids, 25):.3f}  p50={np.percentile(mids, 50):.3f}  "
          f"p75={np.percentile(mids, 75):.3f}")

# Feature importance for aggressive dedup
print(f"\nTop 15 features (aggressive dedup):")
imp_agg = model_agg.get_score(importance_type="gain")
sorted_imp = sorted(imp_agg.items(), key=lambda x: x[1], reverse=True)[:15]
max_gain = sorted_imp[0][1] if sorted_imp else 1.0
for feat, gain in sorted_imp:
    bar = "#" * int(gain / max_gain * 40)
    print(f"  {feat:<32s} {gain:>10.1f}  {bar}")
