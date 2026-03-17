#!/usr/bin/env python3
"""
Research Agent 6: Volatility regime detection and regime-conditional XGBoost.

Key questions:
1. Can we detect volatility regimes from early orderbook data?
2. Does XGBoost performance vary by BTC volatility regime?
3. Can we build separate models per regime for better accuracy?
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss

warnings.filterwarnings("ignore")

FEATURE_CACHE = Path("data/xgb_features_v2.parquet")
RESOLUTION_CACHE = Path("data/resolution_cache.json")
BTC_QUOTES = Path("data/btc_quotes/btcusdt_quotes.parquet")
MARKET_DURATION_MS = 300_000


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


def load_btc():
    btc = pd.read_parquet(BTC_QUOTES)
    btc = btc.sort_values("timestamp_ms").reset_index(drop=True)
    return btc


def compute_btc_vol_features(df, btc):
    """Add BTC volatility regime features to the feature dataframe."""
    btc_ts = btc["timestamp_ms"].values
    btc_mid = btc["mid_price"].values

    # Pre-compute rolling vol on BTC
    btc_ret_1s = np.diff(np.log(btc_mid))
    btc_ret_1s = np.insert(btc_ret_1s, 0, 0)

    results = []
    for _, row in df.iterrows():
        open_ms = row["open_ts"] * 1000
        observe_ms = open_ms + row["observe_time_s"] * 1000

        # BTC vol in different lookback windows
        feats = {}
        for lb_s in [30, 60, 300, 900, 3600]:
            lb_ms = lb_s * 1000
            start_idx = np.searchsorted(btc_ts, observe_ms - lb_ms)
            end_idx = np.searchsorted(btc_ts, observe_ms)
            if end_idx - start_idx > 5:
                rets = btc_ret_1s[start_idx:end_idx]
                feats[f"btc_vol_{lb_s}s"] = np.std(rets) * np.sqrt(len(rets))
                feats[f"btc_vol_{lb_s}s_raw"] = np.std(rets)
                feats[f"btc_range_{lb_s}s"] = (btc_mid[start_idx:end_idx].max() -
                                                 btc_mid[start_idx:end_idx].min()) / btc_mid[end_idx - 1]
            else:
                feats[f"btc_vol_{lb_s}s"] = np.nan
                feats[f"btc_vol_{lb_s}s_raw"] = np.nan
                feats[f"btc_range_{lb_s}s"] = np.nan

        # Vol regime: high/medium/low based on 5min vol tertiles
        feats["btc_vol_5m"] = feats.get("btc_vol_300s", np.nan)
        results.append(feats)

    vol_df = pd.DataFrame(results, index=df.index)
    return pd.concat([df, vol_df], axis=1)


def classify_vol_regime(df, vol_col="btc_vol_5m"):
    """Classify into tercile vol regimes."""
    valid = df[vol_col].dropna()
    t33 = valid.quantile(0.33)
    t67 = valid.quantile(0.67)
    df = df.copy()
    df["vol_regime"] = "medium"
    df.loc[df[vol_col] < t33, "vol_regime"] = "low"
    df.loc[df[vol_col] > t67, "vol_regime"] = "high"
    return df, t33, t67


def run_experiment(df, feature_cols, window=120):
    """Run XGBoost experiments: pooled vs regime-specific."""
    wdf = df[df["observe_time_s"] == window].copy()
    wdf = wdf.dropna(subset=["label"] + feature_cols, how="any")

    if len(wdf) < 100:
        print(f"  Not enough data for window {window}s: {len(wdf)} samples")
        return

    # Time-based split
    slugs = wdf.sort_values("open_ts")["slug"].unique()
    split_idx = int(len(slugs) * 0.7)
    train_slugs = set(slugs[:split_idx])
    test_slugs = set(slugs[split_idx:])

    train = wdf[wdf["slug"].isin(train_slugs)]
    test = wdf[wdf["slug"].isin(test_slugs)]

    X_train, y_train = train[feature_cols].values, train["label"].values
    X_test, y_test = test[feature_cols].values, test["label"].values

    print(f"\n  Window {window}s: {len(train)} train, {len(test)} test")
    print(f"  Train base rate: {y_train.mean():.3f}, Test base rate: {y_test.mean():.3f}")

    # 1. Pooled model (all regimes)
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
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
    auc = roc_auc_score(y_test, preds)
    acc = accuracy_score(y_test, preds > 0.5)
    ll = log_loss(y_test, preds)

    print(f"\n  POOLED MODEL (all regimes):")
    print(f"    AUC: {auc:.4f}  Acc: {acc:.4f}  LogLoss: {ll:.4f}")

    # 2. Per-regime analysis
    print(f"\n  PER-REGIME BREAKDOWN:")
    for regime in ["low", "medium", "high"]:
        regime_test = test[test["vol_regime"] == regime]
        if len(regime_test) < 20:
            continue
        regime_dtest = xgb.DMatrix(regime_test[feature_cols].values,
                                    label=regime_test["label"].values,
                                    feature_names=feature_cols)
        regime_preds = model.predict(regime_dtest)
        regime_y = regime_test["label"].values
        r_auc = roc_auc_score(regime_y, regime_preds) if len(set(regime_y)) > 1 else float("nan")
        r_acc = accuracy_score(regime_y, regime_preds > 0.5)
        r_base = regime_y.mean()

        # High-confidence accuracy
        hi_mask = (regime_preds > 0.7) | (regime_preds < 0.3)
        hi_acc = accuracy_score(regime_y[hi_mask], regime_preds[hi_mask] > 0.5) if hi_mask.sum() > 5 else float("nan")
        hi_n = hi_mask.sum()

        print(f"    {regime:>6s} vol: n={len(regime_test):>4d}, base={r_base:.3f}, "
              f"AUC={r_auc:.3f}, Acc={r_acc:.3f}, HiConf={hi_acc:.3f} (n={hi_n})")

    # 3. Per-regime SEPARATE models
    print(f"\n  REGIME-SPECIFIC MODELS:")
    for regime in ["low", "medium", "high"]:
        regime_train = train[train["vol_regime"] == regime]
        regime_test = test[test["vol_regime"] == regime]
        if len(regime_train) < 50 or len(regime_test) < 20:
            print(f"    {regime:>6s} vol: insufficient data (train={len(regime_train)}, test={len(regime_test)})")
            continue

        r_dtrain = xgb.DMatrix(regime_train[feature_cols].values,
                                label=regime_train["label"].values,
                                feature_names=feature_cols)
        r_dtest = xgb.DMatrix(regime_test[feature_cols].values,
                               label=regime_test["label"].values,
                               feature_names=feature_cols)

        r_model = xgb.train(params, r_dtrain, num_boost_round=200,
                             evals=[(r_dtest, "test")], verbose_eval=False,
                             early_stopping_rounds=30)
        r_preds = r_model.predict(r_dtest)
        r_y = regime_test["label"].values
        r_auc = roc_auc_score(r_y, r_preds) if len(set(r_y)) > 1 else float("nan")
        r_acc = accuracy_score(r_y, r_preds > 0.5)

        # Compare to pooled
        pooled_preds = model.predict(r_dtest)
        p_auc = roc_auc_score(r_y, pooled_preds) if len(set(r_y)) > 1 else float("nan")
        p_acc = accuracy_score(r_y, pooled_preds > 0.5)

        print(f"    {regime:>6s} vol: Regime AUC={r_auc:.3f} vs Pooled AUC={p_auc:.3f} "
              f"| Regime Acc={r_acc:.3f} vs Pooled Acc={p_acc:.3f}")

    # 4. Vol features importance
    print(f"\n  TOP 15 FEATURES (pooled model):")
    importance = model.get_score(importance_type="gain")
    sorted_imp = sorted(importance.items(), key=lambda x: -x[1])[:15]
    for fname, gain in sorted_imp:
        is_vol = "vol" in fname or "range" in fname
        marker = " *** NEW" if is_vol else ""
        print(f"    {fname:>35s}: {gain:>10.1f}{marker}")


def main():
    print("=" * 70)
    print("RESEARCH: Volatility Regime Detection & Regime-Conditional XGBoost")
    print("=" * 70)

    df, resolutions = load_data()
    btc = load_btc()

    print(f"\nLoaded {len(df)} samples, {df['slug'].nunique()} markets")
    print(f"BTC quotes: {len(btc)} rows")

    # Add BTC volatility features
    print("\nComputing BTC volatility features...")
    df = compute_btc_vol_features(df, btc)

    # Classify vol regime
    df, t33, t67 = classify_vol_regime(df)
    print(f"\nVol regime thresholds: low < {t33:.6f} < medium < {t67:.6f} < high")
    print(f"Regime distribution:\n{df['vol_regime'].value_counts()}")

    # Check if vol regime correlates with outcomes
    print(f"\nOutcome by vol regime (all windows):")
    for regime in ["low", "medium", "high"]:
        rdf = df[df["vol_regime"] == regime]
        wr = rdf["label"].mean()
        n = len(rdf)
        print(f"  {regime:>6s}: WR={wr:.3f}, n={n}")

    # Get feature columns (original + new vol features)
    orig_cols = [c for c in df.columns if c not in
                 ["slug", "open_ts", "label", "observe_time_s", "vol_regime",
                  "winning_token", "token_id_up", "token_id_down"]]
    vol_cols = [c for c in orig_cols if "btc_vol" in c or "btc_range" in c]
    print(f"\nTotal features: {len(orig_cols)} ({len(vol_cols)} new vol features)")

    # Run experiments
    for window in [60, 90, 120]:
        run_experiment(df, orig_cols, window=window)

    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print("""
Key findings:
- If high-vol regime shows higher AUC: market is more predictable when volatile
  → concentrate trading during volatile periods
- If low-vol regime shows higher AUC: model works best in calm markets
  → avoid trading during volatility spikes
- If regime-specific models outperform pooled: worth training separate models
  → adds complexity but may improve edge
- If vol features rank high in importance: BTC volatility context is informative
  → include in production model
""")


if __name__ == "__main__":
    main()
