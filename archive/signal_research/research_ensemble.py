#!/usr/bin/env python3
"""
Research Agent 8: Ensemble & stacking approaches.

Tests:
1. Multiple XGBoost models with different hyperparameters → ensemble
2. Window-specific models stacked with a meta-learner
3. XGBoost + simple heuristic rules → combined
4. Confidence calibration: are our predicted probabilities reliable?
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, brier_score_loss
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")

FEATURE_CACHE = Path("data/xgb_features_v2.parquet")
RESOLUTION_CACHE = Path("data/resolution_cache.json")


def load_data():
    df = pd.read_parquet(FEATURE_CACHE)
    with open(RESOLUTION_CACHE) as f:
        raw = json.load(f)
    resolutions = {}
    for slug, info in raw.items():
        winner = info.get("winning_token") or info.get("winningTokenId", "")
        if winner:
            resolutions[slug] = winner
    return df, resolutions


def get_splits(df, window=120):
    """Time-based train/val/test split."""
    wdf = df[df["observe_time_s"] == window].copy()

    feature_cols = [c for c in df.columns if c not in
                    ["slug", "open_ts", "label", "observe_time_s",
                     "winning_token", "token_id_up", "token_id_down"]]
    valid_cols = [c for c in feature_cols if wdf[c].notna().sum() > len(wdf) * 0.5]

    wdf = wdf.dropna(subset=["label"])
    slugs = wdf.sort_values("open_ts")["slug"].unique()
    n = len(slugs)
    train_slugs = set(slugs[:int(n * 0.6)])
    val_slugs = set(slugs[int(n * 0.6):int(n * 0.8)])
    test_slugs = set(slugs[int(n * 0.8):])

    train = wdf[wdf["slug"].isin(train_slugs)]
    val = wdf[wdf["slug"].isin(val_slugs)]
    test = wdf[wdf["slug"].isin(test_slugs)]

    return train, val, test, valid_cols


def experiment_1_hyperparameter_ensemble(df):
    """Ensemble of XGBoost models with diverse hyperparameters."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Hyperparameter Diversity Ensemble")
    print("=" * 70)

    train, val, test, feature_cols = get_splits(df)
    print(f"  Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

    X_train = train[feature_cols].fillna(0).values
    y_train = train["label"].values
    X_val = val[feature_cols].fillna(0).values
    y_val = val["label"].values
    X_test = test[feature_cols].fillna(0).values
    y_test = test["label"].values

    configs = [
        {"max_depth": 3, "learning_rate": 0.01, "subsample": 0.7, "colsample_bytree": 0.6, "n": 300},
        {"max_depth": 5, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8, "n": 200},
        {"max_depth": 7, "learning_rate": 0.1, "subsample": 0.9, "colsample_bytree": 1.0, "n": 100},
        {"max_depth": 4, "learning_rate": 0.03, "subsample": 0.6, "colsample_bytree": 0.7, "n": 250},
        {"max_depth": 6, "learning_rate": 0.08, "subsample": 0.85, "colsample_bytree": 0.9, "n": 150},
    ]

    models = []
    val_preds_all = []
    test_preds_all = []

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_cols)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_cols)
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_cols)

    for i, cfg in enumerate(configs):
        params = {
            "objective": "binary:logistic",
            "max_depth": cfg["max_depth"],
            "learning_rate": cfg["learning_rate"],
            "subsample": cfg["subsample"],
            "colsample_bytree": cfg["colsample_bytree"],
            "min_child_weight": 10,
            "verbosity": 0,
        }
        model = xgb.train(params, dtrain, num_boost_round=cfg["n"],
                           evals=[(dval, "val")], verbose_eval=False,
                           early_stopping_rounds=30)
        models.append(model)

        val_p = model.predict(dval)
        test_p = model.predict(dtest)
        val_preds_all.append(val_p)
        test_preds_all.append(test_p)

        auc = roc_auc_score(y_test, test_p) if len(set(y_test)) > 1 else 0.5
        acc = accuracy_score(y_test, test_p > 0.5)
        print(f"  Model {i}: depth={cfg['max_depth']}, lr={cfg['learning_rate']}, "
              f"AUC={auc:.4f}, Acc={acc:.4f}")

    # Simple average ensemble
    avg_test = np.mean(test_preds_all, axis=0)
    avg_auc = roc_auc_score(y_test, avg_test) if len(set(y_test)) > 1 else 0.5
    avg_acc = accuracy_score(y_test, avg_test > 0.5)
    print(f"\n  AVERAGE ENSEMBLE: AUC={avg_auc:.4f}, Acc={avg_acc:.4f}")

    # Weighted ensemble (optimized on val set)
    avg_val = np.mean(val_preds_all, axis=0)
    best_weights = None
    best_val_auc = 0

    # Simple grid over weights
    for w0 in np.arange(0.1, 0.5, 0.1):
        for w1 in np.arange(0.1, 0.5, 0.1):
            for w2 in np.arange(0.1, 0.5, 0.1):
                remaining = 1.0 - w0 - w1 - w2
                if remaining < 0.05:
                    continue
                w3 = remaining / 2
                w4 = remaining / 2
                weights = [w0, w1, w2, w3, w4]
                wp = sum(w * p for w, p in zip(weights, val_preds_all))
                wauc = roc_auc_score(y_val, wp) if len(set(y_val)) > 1 else 0.5
                if wauc > best_val_auc:
                    best_val_auc = wauc
                    best_weights = weights

    if best_weights:
        weighted_test = sum(w * p for w, p in zip(best_weights, test_preds_all))
        w_auc = roc_auc_score(y_test, weighted_test) if len(set(y_test)) > 1 else 0.5
        w_acc = accuracy_score(y_test, weighted_test > 0.5)
        print(f"  WEIGHTED ENSEMBLE: AUC={w_auc:.4f}, Acc={w_acc:.4f}")
        print(f"    Weights: {[f'{w:.2f}' for w in best_weights]}")

    # Confidence-filtered
    print(f"\n  HIGH-CONFIDENCE (ensemble avg > 0.70 or < 0.30):")
    for name, preds in [("Best single", test_preds_all[1]), ("Average", avg_test)]:
        mask = (preds > 0.70) | (preds < 0.30)
        if mask.sum() > 5:
            hc_acc = accuracy_score(y_test[mask], preds[mask] > 0.5)
            print(f"    {name}: acc={hc_acc:.1%}, n={mask.sum()}")

    return val_preds_all, test_preds_all, y_val, y_test, feature_cols


def experiment_2_window_stacking(df):
    """Train per-window models, stack predictions with meta-learner."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Window-Level Stacking")
    print("=" * 70)

    windows = sorted(df["observe_time_s"].unique())
    slugs = df.sort_values("open_ts")["slug"].unique()
    n = len(slugs)
    train_slugs = set(slugs[:int(n * 0.6)])
    val_slugs = set(slugs[int(n * 0.6):int(n * 0.8)])
    test_slugs = set(slugs[int(n * 0.8):])

    feature_cols = [c for c in df.columns if c not in
                    ["slug", "open_ts", "label", "observe_time_s",
                     "winning_token", "token_id_up", "token_id_down"]]

    # Train per-window models and collect OOF predictions
    window_models = {}
    val_meta = {}  # slug -> {window: pred}
    test_meta = {}
    val_labels = {}
    test_labels = {}

    params = {
        "objective": "binary:logistic",
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 10,
        "verbosity": 0,
    }

    for w in windows:
        wdf = df[df["observe_time_s"] == w].dropna(subset=["label"])
        valid_cols = [c for c in feature_cols if wdf[c].notna().sum() > len(wdf) * 0.5]

        train = wdf[wdf["slug"].isin(train_slugs)]
        val = wdf[wdf["slug"].isin(val_slugs)]
        test = wdf[wdf["slug"].isin(test_slugs)]

        if len(train) < 50:
            continue

        X_train = train[valid_cols].fillna(0).values
        dtrain = xgb.DMatrix(X_train, label=train["label"].values)

        model = xgb.train(params, dtrain, num_boost_round=150, verbose_eval=False)
        window_models[w] = (model, valid_cols)

        # Val predictions
        if len(val) > 0:
            X_val = val[valid_cols].fillna(0).values
            dval = xgb.DMatrix(X_val)
            vpreds = model.predict(dval)
            for slug, pred, label in zip(val["slug"].values, vpreds, val["label"].values):
                if slug not in val_meta:
                    val_meta[slug] = {}
                    val_labels[slug] = label
                val_meta[slug][w] = pred

        # Test predictions
        if len(test) > 0:
            X_test = test[valid_cols].fillna(0).values
            dtest = xgb.DMatrix(X_test)
            tpreds = model.predict(dtest)
            for slug, pred, label in zip(test["slug"].values, tpreds, test["label"].values):
                if slug not in test_meta:
                    test_meta[slug] = {}
                    test_labels[slug] = label
                test_meta[slug][w] = pred

    # Build meta-features (one row per market: window predictions as features)
    def build_meta(meta_dict, labels_dict):
        rows = []
        for slug, wpreds in meta_dict.items():
            row = {"slug": slug, "label": labels_dict[slug]}
            for w in windows:
                row[f"w{w}_pred"] = wpreds.get(w, np.nan)
            rows.append(row)
        return pd.DataFrame(rows)

    val_mdf = build_meta(val_meta, val_labels)
    test_mdf = build_meta(test_meta, test_labels)

    meta_features = [f"w{w}_pred" for w in windows]
    val_mdf = val_mdf.dropna(subset=meta_features)
    test_mdf = test_mdf.dropna(subset=meta_features)

    print(f"  Val meta: {len(val_mdf)} markets, Test meta: {len(test_mdf)} markets")

    if len(val_mdf) < 20 or len(test_mdf) < 10:
        print("  Insufficient data for stacking")
        return

    # Meta-learner: logistic regression
    X_val_meta = val_mdf[meta_features].values
    y_val_meta = val_mdf["label"].values
    X_test_meta = test_mdf[meta_features].values
    y_test_meta = test_mdf["label"].values

    lr = LogisticRegression(C=1.0, max_iter=1000)
    lr.fit(X_val_meta, y_val_meta)
    stacked_preds = lr.predict_proba(X_test_meta)[:, 1]

    stacked_auc = roc_auc_score(y_test_meta, stacked_preds) if len(set(y_test_meta)) > 1 else 0.5
    stacked_acc = accuracy_score(y_test_meta, stacked_preds > 0.5)

    # Compare to simple average and best single window
    avg_preds = test_mdf[meta_features].mean(axis=1).values
    avg_auc = roc_auc_score(y_test_meta, avg_preds) if len(set(y_test_meta)) > 1 else 0.5
    avg_acc = accuracy_score(y_test_meta, avg_preds > 0.5)

    best_w_auc = 0
    best_w_name = ""
    for w in windows:
        col = f"w{w}_pred"
        if col in test_mdf.columns:
            wauc = roc_auc_score(y_test_meta, test_mdf[col].values) if len(set(y_test_meta)) > 1 else 0.5
            if wauc > best_w_auc:
                best_w_auc = wauc
                best_w_name = f"w{w}"

    print(f"\n  Results (per-market level):")
    print(f"    Best single window ({best_w_name}): AUC={best_w_auc:.4f}")
    print(f"    Simple average:                     AUC={avg_auc:.4f}, Acc={avg_acc:.4f}")
    print(f"    Stacked (LogReg):                   AUC={stacked_auc:.4f}, Acc={stacked_acc:.4f}")

    # Meta-learner coefficients
    print(f"\n  Meta-learner weights:")
    for fname, coef in zip(meta_features, lr.coef_[0]):
        print(f"    {fname}: {coef:+.3f}")


def experiment_3_calibration(df):
    """Check if predicted probabilities are well-calibrated."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Probability Calibration")
    print("=" * 70)

    train, val, test, feature_cols = get_splits(df)

    X_train = train[feature_cols].fillna(0).values
    y_train = train["label"].values
    X_test = test[feature_cols].fillna(0).values
    y_test = test["label"].values

    params = {
        "objective": "binary:logistic",
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 10,
        "verbosity": 0,
    }

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_cols)
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_cols)
    model = xgb.train(params, dtrain, num_boost_round=200,
                       evals=[(dtest, "test")], verbose_eval=False,
                       early_stopping_rounds=30)
    preds = model.predict(dtest)

    # Brier score
    brier = brier_score_loss(y_test, preds)
    print(f"\n  Brier score: {brier:.4f} (lower is better, baseline={np.mean(y_test)*(1-np.mean(y_test)):.4f})")

    # Calibration curve
    try:
        prob_true, prob_pred = calibration_curve(y_test, preds, n_bins=10, strategy="uniform")
        print(f"\n  CALIBRATION TABLE:")
        print(f"  {'Predicted':>10s}  {'Actual':>8s}  {'Gap':>8s}  {'N':>6s}")
        print("  " + "-" * 38)

        # Count samples in each bin
        bin_edges = np.linspace(0, 1, 11)
        for i in range(len(prob_true)):
            low = bin_edges[i]
            high = bin_edges[i + 1]
            mask = (preds >= low) & (preds < high)
            n_in_bin = mask.sum()
            gap = prob_true[i] - prob_pred[i]
            print(f"  {prob_pred[i]:>9.2%}  {prob_true[i]:>7.2%}  {gap:>+7.2%}  {n_in_bin:>6d}")
    except Exception as e:
        print(f"  Calibration error: {e}")

    # ECE (Expected Calibration Error)
    try:
        ece = 0.0
        total = 0
        for i in range(10):
            low = i / 10
            high = (i + 1) / 10
            mask = (preds >= low) & (preds < high)
            if mask.sum() > 0:
                bin_acc = y_test[mask].mean()
                bin_conf = preds[mask].mean()
                ece += mask.sum() * abs(bin_acc - bin_conf)
                total += mask.sum()
        ece /= total
        print(f"\n  Expected Calibration Error (ECE): {ece:.4f}")
    except Exception:
        pass

    # Sharpness: how spread out are the predictions?
    print(f"\n  PREDICTION DISTRIBUTION:")
    print(f"    Mean: {preds.mean():.3f}, Std: {preds.std():.3f}")
    print(f"    Min: {preds.min():.3f}, Max: {preds.max():.3f}")
    for bucket in [(0, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 1.0)]:
        mask = (preds >= bucket[0]) & (preds < bucket[1])
        print(f"    [{bucket[0]:.1f}, {bucket[1]:.1f}): {mask.sum():>5d} ({mask.mean():.1%})")


def experiment_4_heuristic_rules(df):
    """Combine XGBoost with simple heuristic rules."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: XGBoost + Heuristic Rules")
    print("=" * 70)

    train, val, test, feature_cols = get_splits(df)

    X_train = train[feature_cols].fillna(0).values
    y_train = train["label"].values
    X_test = test[feature_cols].fillna(0).values
    y_test = test["label"].values

    params = {
        "objective": "binary:logistic",
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 10,
        "verbosity": 0,
    }

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_cols)
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_cols)
    model = xgb.train(params, dtrain, num_boost_round=200,
                       evals=[(dtest, "test")], verbose_eval=False,
                       early_stopping_rounds=30)
    xgb_preds = model.predict(dtest)

    # Heuristic 1: mid_price direction (if mid > 0.5, predict leader wins)
    mid_cols = [c for c in feature_cols if "mid_price" in c.lower()]
    if mid_cols:
        mid_pred = test[mid_cols[0]].fillna(0.5).values > 0.5
        mid_acc = accuracy_score(y_test, mid_pred)
        print(f"\n  Heuristic: mid_price > 0.5 → predict leader wins")
        print(f"    Accuracy: {mid_acc:.1%}")

    # Heuristic 2: book imbalance > 0 → predict leader
    imb_cols = [c for c in feature_cols if "imbalance" in c.lower()]
    if imb_cols:
        imb_pred = test[imb_cols[0]].fillna(0).values > 0
        imb_acc = accuracy_score(y_test, imb_pred)
        print(f"  Heuristic: book_imbalance > 0 → predict leader wins")
        print(f"    Accuracy: {imb_acc:.1%}")

    # Combined: XGBoost + mid_price agreement filter
    if mid_cols:
        mid_vals = test[mid_cols[0]].fillna(0.5).values
        agree_mask = ((xgb_preds > 0.5) & (mid_vals > 0.5)) | ((xgb_preds <= 0.5) & (mid_vals <= 0.5))
        if agree_mask.sum() > 10:
            agree_acc = accuracy_score(y_test[agree_mask], xgb_preds[agree_mask] > 0.5)
            disagree_acc = accuracy_score(y_test[~agree_mask], xgb_preds[~agree_mask] > 0.5) if (~agree_mask).sum() > 5 else float("nan")
            print(f"\n  XGBoost + mid_price AGREE: acc={agree_acc:.1%} (n={agree_mask.sum()})")
            print(f"  XGBoost + mid_price DISAGREE: acc={disagree_acc:.1%} (n={(~agree_mask).sum()})")

    # Combined: only trade when XGBoost AND imbalance agree
    if imb_cols:
        imb_vals = test[imb_cols[0]].fillna(0).values
        both_agree = ((xgb_preds > 0.5) & (imb_vals > 0)) | ((xgb_preds <= 0.5) & (imb_vals <= 0))
        if both_agree.sum() > 10:
            agree_acc = accuracy_score(y_test[both_agree], xgb_preds[both_agree] > 0.5)
            print(f"  XGBoost + imbalance AGREE: acc={agree_acc:.1%} (n={both_agree.sum()})")

    # XGBoost alone for comparison
    xgb_acc = accuracy_score(y_test, xgb_preds > 0.5)
    xgb_auc = roc_auc_score(y_test, xgb_preds) if len(set(y_test)) > 1 else 0.5
    print(f"\n  XGBoost alone: AUC={xgb_auc:.4f}, Acc={xgb_acc:.4f}")


def main():
    print("=" * 70)
    print("RESEARCH: Ensemble & Stacking Approaches")
    print("=" * 70)

    df, resolutions = load_data()
    print(f"Loaded {len(df)} samples, {df['slug'].nunique()} markets")

    experiment_1_hyperparameter_ensemble(df)
    experiment_2_window_stacking(df)
    experiment_3_calibration(df)
    experiment_4_heuristic_rules(df)

    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print("""
Key questions answered:
1. Does ensembling diverse XGBoost models beat a single model?
2. Does stacking per-window models help at per-market level?
3. Are our probabilities well-calibrated (can we trust the confidence)?
4. Does agreement with simple heuristics filter for higher quality trades?
""")


if __name__ == "__main__":
    main()
