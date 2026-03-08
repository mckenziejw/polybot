#!/usr/bin/env python3
"""
Explore alternative target variables for XGBoost on Polymarket BTC Up/Down markets.

Instead of the binary "does the leader win?" target, we test:

  1. FINAL MID PRICE (regression) — predict the leader's final mid price
     - More granular than binary; a market where leader ends at 0.95 is very
       different from one ending at 0.52
     - Directly useful: if predicted final mid > current mid + fees, buy

  2. MAX ADVERSE DRIFT (regression) — predict the max price drop the leader
     experiences between observation time and market close
     - Key for MM: tells you "how bad can it get if I'm quoting?"
     - Also useful for stop-loss sizing on directional trades

  3. REGIME CLASSIFICATION (multi-class) — classify market into:
     {trending_leader, trending_other, choppy, coin_flip}
     - Directly maps to strategy: trend→directional, choppy→MM, coin_flip→sit out
     - Defined by price path shape, not just outcome

  4. SPREAD BLOWOUT (binary) — will the spread exceed 2x current spread
     at any point in the remaining market?
     - MM risk management: predicts adverse liquidity conditions

  5. TIME-TO-RESOLUTION (regression) — how many seconds until mid crosses
     0.85 or 0.15 (or never)?
     - Fast-resolving markets: directional early. Slow: MM opportunity window.

Each target is extracted from the raw 1s-resampled orderbook data. We reuse
the existing feature set from xgb_features_v2.parquet and just swap the label.

Usage:
    python explore_targets.py
    python explore_targets.py --max-markets 500   # quick test
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
    print("ERROR: xgboost not installed")
    sys.exit(1)

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

from xgboost_flexible import (
    FEATURE_CACHE_PATH,
    MARKET_DURATION_MS,
    OBSERVATION_WINDOWS,
    get_feature_columns,
    load_resolutions,
    resample_to_1s,
)

warnings.filterwarnings("ignore", category=FutureWarning)

DATA_DIR = Path("data/telonex_book_snapshots")
RESOLUTION_CACHE = Path("data/resolution_cache.json")


# ---------------------------------------------------------------------------
# Target extraction from raw market data
# ---------------------------------------------------------------------------

def extract_targets_for_market(
    up_1s: pd.DataFrame,
    down_1s: pd.DataFrame,
    open_ms: int,
    observe_s: float,
    winning_token: str,
) -> dict | None:
    """
    Extract all alternative targets for a single market at a given observation time.

    Returns dict of target values, or None if data is insufficient.
    """
    close_ms = open_ms + MARKET_DURATION_MS
    market_duration_s = MARKET_DURATION_MS / 1000.0

    # Determine leader at observation time
    up_pre = up_1s[up_1s["sec"] <= observe_s]
    down_pre = down_1s[down_1s["sec"] <= observe_s]

    if up_pre.empty or down_pre.empty:
        return None

    up_mid_obs = up_pre.iloc[-1]["mid_price"]
    down_mid_obs = down_pre.iloc[-1]["mid_price"]

    if up_mid_obs >= 0.50:
        leader_df, other_df = up_1s, down_1s
        leader_asset_id = up_pre.iloc[-1]["asset_id"]
    else:
        leader_df, other_df = down_1s, up_1s
        leader_asset_id = down_pre.iloc[-1]["asset_id"]

    leader_wins = 1 if leader_asset_id == winning_token else 0

    # Get leader and other price series AFTER observation time
    leader_future = leader_df[leader_df["sec"] > observe_s]
    other_future = other_df[other_df["sec"] > observe_s]

    # Also get leader price at observation
    leader_pre = leader_df[leader_df["sec"] <= observe_s]
    other_pre = other_df[other_df["sec"] <= observe_s]

    if leader_pre.empty or leader_future.empty:
        return None

    leader_mid_obs = leader_pre.iloc[-1]["mid_price"]
    other_mid_obs = other_pre.iloc[-1]["mid_price"] if not other_pre.empty else 1.0 - leader_mid_obs

    leader_future_mids = leader_future["mid_price"].values
    other_future_mids = other_future["mid_price"].values if not other_future.empty else np.array([])

    # Get spread series for future
    leader_future_spreads = leader_future["spread"].values if "spread" in leader_future.columns else np.array([])
    leader_obs_spread = leader_pre.iloc[-1].get("spread", 0.0)

    targets = {}
    targets["original_label"] = leader_wins

    # -----------------------------------------------------------------------
    # TARGET 1: Final mid price of the leader token
    # -----------------------------------------------------------------------
    # Use the last available mid price (at or near market close)
    leader_all = leader_df["mid_price"].values
    if len(leader_all) > 0:
        targets["final_mid"] = leader_all[-1]
        # Also compute "final mid relative to observation mid" — the actual useful thing
        targets["mid_change"] = leader_all[-1] - leader_mid_obs
    else:
        return None

    # -----------------------------------------------------------------------
    # TARGET 2: Max adverse drift (worst-case price drop for leader after obs)
    # -----------------------------------------------------------------------
    if len(leader_future_mids) > 0:
        min_future = leader_future_mids.min()
        targets["max_adverse_drift"] = leader_mid_obs - min_future  # positive = bad
        targets["max_favorable_drift"] = leader_future_mids.max() - leader_mid_obs  # positive = good

        # Normalized by distance from 0.50 (how much of the "lead" is lost)
        lead = leader_mid_obs - 0.50
        if lead > 0.01:
            targets["adverse_drift_pct"] = targets["max_adverse_drift"] / lead
        else:
            targets["adverse_drift_pct"] = 0.0

        # Binary: does the leader ever lose the lead? (mid drops below 0.50)
        targets["leader_loses_lead"] = int(min_future < 0.50)
    else:
        targets["max_adverse_drift"] = 0.0
        targets["max_favorable_drift"] = 0.0
        targets["adverse_drift_pct"] = 0.0
        targets["leader_loses_lead"] = 0

    # -----------------------------------------------------------------------
    # TARGET 3: Regime classification
    # -----------------------------------------------------------------------
    if len(leader_future_mids) >= 5:
        final = leader_future_mids[-1]
        drift = final - leader_mid_obs
        future_range = leader_future_mids.max() - leader_future_mids.min()

        # Count direction changes
        changes = np.diff(leader_future_mids)
        if len(changes) > 1:
            sign_changes = np.sum(np.diff(np.sign(changes)) != 0)
            choppiness = sign_changes / len(changes) if len(changes) > 0 else 0
        else:
            choppiness = 0

        # Classify regime
        abs_drift = abs(drift)
        if abs_drift > 0.10 and drift > 0:
            regime = 0  # trending_leader (leader extends lead)
        elif abs_drift > 0.10 and drift < 0:
            regime = 1  # trending_other (leader loses lead)
        elif abs_drift <= 0.05 and choppiness > 0.4:
            regime = 2  # choppy (lots of back-and-forth, small net move)
        else:
            regime = 3  # coin_flip (moderate moves, unclear direction)

        targets["regime"] = regime
        targets["choppiness_future"] = choppiness
    else:
        targets["regime"] = 3  # default to coin_flip
        targets["choppiness_future"] = 0.0

    # -----------------------------------------------------------------------
    # TARGET 4: Spread blowout
    # -----------------------------------------------------------------------
    if len(leader_future_spreads) > 0 and leader_obs_spread > 0:
        max_future_spread = leader_future_spreads.max()
        targets["spread_blowout"] = int(max_future_spread > 2.0 * leader_obs_spread)
        targets["max_spread_ratio"] = max_future_spread / leader_obs_spread
    else:
        targets["spread_blowout"] = 0
        targets["max_spread_ratio"] = 1.0

    # -----------------------------------------------------------------------
    # TARGET 5: Time to resolution
    # -----------------------------------------------------------------------
    # How many seconds until the leader mid crosses 0.85 (strong resolution)
    if len(leader_future_mids) > 0:
        leader_future_secs = leader_future["sec"].values
        resolved_mask = (leader_future_mids >= 0.85) | (leader_future_mids <= 0.15)

        if resolved_mask.any():
            first_resolve_idx = np.argmax(resolved_mask)
            targets["time_to_resolution_s"] = leader_future_secs[first_resolve_idx] - observe_s
            targets["resolves_early"] = int(targets["time_to_resolution_s"] < (market_duration_s - observe_s) / 2)
        else:
            targets["time_to_resolution_s"] = market_duration_s - observe_s  # never resolves
            targets["resolves_early"] = 0

        # Also: does it ever reach extreme (0.90+)?
        targets["reaches_extreme"] = int(leader_future_mids.max() >= 0.90 or leader_future_mids.min() <= 0.10)
    else:
        targets["time_to_resolution_s"] = market_duration_s - observe_s
        targets["resolves_early"] = 0
        targets["reaches_extreme"] = 0

    return targets


def build_target_dataset(
    feature_cache: pd.DataFrame,
    max_markets: int | None = None,
) -> pd.DataFrame:
    """
    Augment the cached feature dataset with alternative target columns.

    Iterates through raw parquet files to extract future price paths.
    """
    resolutions = load_resolutions(RESOLUTION_CACHE)

    # Get unique slugs from feature cache
    slugs = feature_cache["slug"].unique()
    if max_markets:
        slugs = slugs[:max_markets]

    print(f"Extracting alternative targets for {len(slugs)} markets...")

    all_targets = []
    skipped = 0

    for i, slug in enumerate(slugs):
        pf = DATA_DIR / f"{slug}.parquet"
        if not pf.exists() or slug not in resolutions:
            skipped += 1
            continue

        try:
            open_ts = int(slug.split("-")[-1])
        except ValueError:
            skipped += 1
            continue

        open_ms = open_ts * 1000
        close_ms = open_ms + MARKET_DURATION_MS

        try:
            df = pd.read_parquet(pf)
        except Exception:
            skipped += 1
            continue

        if "token_label" not in df.columns or "mid_price" not in df.columns:
            skipped += 1
            continue

        up_raw = df[df["token_label"] == "Up"].sort_values("exchange_timestamp")
        down_raw = df[df["token_label"] == "Down"].sort_values("exchange_timestamp")

        up_1s = resample_to_1s(up_raw, open_ms, close_ms)
        down_1s = resample_to_1s(down_raw, open_ms, close_ms)

        if up_1s.empty or down_1s.empty:
            skipped += 1
            continue

        winning_token = resolutions[slug]

        for obs_s in OBSERVATION_WINDOWS:
            targets = extract_targets_for_market(
                up_1s, down_1s, open_ms, obs_s, winning_token
            )
            if targets is None:
                continue

            targets["slug"] = slug
            targets["observe_time_s"] = obs_s
            all_targets.append(targets)

        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(slugs)} markets processed...")

    print(f"Extracted targets for {len(all_targets)} samples ({skipped} markets skipped)")
    targets_df = pd.DataFrame(all_targets)
    return targets_df


# ---------------------------------------------------------------------------
# Model training and evaluation helpers
# ---------------------------------------------------------------------------

def time_split(dataset, train_frac=0.70, val_frac=0.15):
    """Split by market time (same as xgboost_flexible.py)."""
    slug_order = dataset.groupby("slug")["open_ts"].first().sort_values()
    n = len(slug_order)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    train_slugs = set(slug_order.index[:train_end])
    val_slugs = set(slug_order.index[train_end:val_end])
    test_slugs = set(slug_order.index[val_end:])

    return (
        dataset[dataset["slug"].isin(train_slugs)],
        dataset[dataset["slug"].isin(val_slugs)],
        dataset[dataset["slug"].isin(test_slugs)],
    )


def train_xgb_classifier(X_train, y_train, X_val, y_val, feature_cols, num_classes=2):
    """Train XGBoost classifier, return model and best iteration."""
    for X in [X_train, X_val]:
        X[~np.isfinite(X)] = 0.0

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_cols)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_cols)

    if num_classes == 2:
        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "max_depth": 6, "learning_rate": 0.03,
            "subsample": 0.8, "colsample_bytree": 0.6,
            "min_child_weight": 15, "lambda": 2.0, "alpha": 0.5,
            "seed": 42, "verbosity": 0,
        }
    else:
        params = {
            "objective": "multi:softprob",
            "num_class": num_classes,
            "eval_metric": "mlogloss",
            "max_depth": 6, "learning_rate": 0.03,
            "subsample": 0.8, "colsample_bytree": 0.6,
            "min_child_weight": 15, "lambda": 2.0, "alpha": 0.5,
            "seed": 42, "verbosity": 0,
        }

    model = xgb.train(
        params, dtrain,
        num_boost_round=1500,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=50,
        verbose_eval=0,
    )
    return model


def train_xgb_regressor(X_train, y_train, X_val, y_val, feature_cols):
    """Train XGBoost regressor, return model."""
    for X in [X_train, X_val]:
        X[~np.isfinite(X)] = 0.0

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_cols)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_cols)

    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "max_depth": 6, "learning_rate": 0.03,
        "subsample": 0.8, "colsample_bytree": 0.6,
        "min_child_weight": 15, "lambda": 2.0, "alpha": 0.5,
        "seed": 42, "verbosity": 0,
    }

    model = xgb.train(
        params, dtrain,
        num_boost_round=1500,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=50,
        verbose_eval=0,
    )
    return model


def print_top_features(model, n=15):
    """Print top N features by gain."""
    importance = model.get_score(importance_type="gain")
    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:n]
    if not sorted_imp:
        return
    max_gain = sorted_imp[0][1]
    for feat, gain in sorted_imp:
        bar = "#" * int(gain / max_gain * 30)
        print(f"    {feat:<32s} {gain:>10.1f}  {bar}")


# ---------------------------------------------------------------------------
# Experiment runners
# ---------------------------------------------------------------------------

def experiment_final_mid(merged, feature_cols):
    """
    TARGET 1: Predict the leader's final mid price (regression).

    Why this is better than binary:
    - "Leader wins" doesn't distinguish between a dominant 0.95 close and a squeaky 0.52 close
    - Continuous target captures the DEGREE of confidence, not just direction
    - Directly actionable: buy if predicted_final > current_mid + fees
    """
    print(f"\n{'='*70}")
    print("EXPERIMENT 1: FINAL MID PRICE (regression)")
    print(f"{'='*70}")

    target_col = "final_mid"
    valid = merged.dropna(subset=[target_col])
    print(f"  Samples: {len(valid)}")
    print(f"  Target stats: mean={valid[target_col].mean():.3f}, std={valid[target_col].std():.3f}")
    print(f"  Target range: [{valid[target_col].min():.3f}, {valid[target_col].max():.3f}]")

    train_df, val_df, test_df = time_split(valid)

    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df[target_col].values
    X_val = val_df[feature_cols].values.astype(np.float32)
    y_val = val_df[target_col].values
    X_test = test_df[feature_cols].values.astype(np.float32)
    y_test = test_df[target_col].values

    model = train_xgb_regressor(X_train, y_train, X_val, y_val, feature_cols)

    # Evaluate
    X_test[~np.isfinite(X_test)] = 0.0
    dtest = xgb.DMatrix(X_test, feature_names=feature_cols)
    y_pred = model.predict(dtest)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\n  TEST RESULTS:")
    print(f"    RMSE:  {rmse:.4f}")
    print(f"    MAE:   {mae:.4f}")
    print(f"    R2:    {r2:.4f}")

    # --- Key question: can we derive a profitable trading rule? ---
    # If predicted final mid > current mid + fees, buy the leader
    # Current mid is stored in the feature set as the latest window mid
    latest_mid_cols = [c for c in feature_cols if c.startswith("w") and c.endswith("_mid")]
    if latest_mid_cols:
        # Get the latest window's mid for each test sample
        latest_w = sorted([int(c.split("_")[0][1:]) for c in latest_mid_cols])[-1]
        current_mid_col = f"w{latest_w}_mid"
        current_mids = test_df[current_mid_col].values

        # Fee model: ~2% at mid=0.50, falling to ~0.5% at mid=0.70
        fees = np.array([0.25 * p * (p * (1 - p)) ** 2 for p in current_mids])

        expected_profit = y_pred - current_mids - fees
        actual_label = test_df["original_label"].values

        print(f"\n  TRADING RULE: buy leader when predicted_final > current_mid + fees")
        for threshold in [0.00, 0.02, 0.05, 0.08, 0.10, 0.15]:
            mask = expected_profit >= threshold
            n_trades = mask.sum()
            if n_trades < 10:
                continue
            win_rate = actual_label[mask].mean()
            avg_mid = current_mids[mask].mean()
            # More accurate EV: win_rate * (1 - avg_mid) - (1 - win_rate) * avg_mid
            ev_per_dollar = win_rate - avg_mid
            print(f"    Threshold {threshold:+.2f}: {n_trades:>5d} trades, "
                  f"WR={win_rate:.1%}, avg_mid={avg_mid:.3f}, EV/$ = {ev_per_dollar:+.4f}")

    # Feature importance
    print(f"\n  TOP 15 FEATURES:")
    print_top_features(model)

    # --- By observation time ---
    if "observe_time_s" in test_df.columns:
        print(f"\n  RMSE BY OBSERVATION TIME:")
        for t in sorted(test_df["observe_time_s"].unique()):
            mask = test_df["observe_time_s"].values == t
            if mask.sum() < 20:
                continue
            t_rmse = np.sqrt(mean_squared_error(y_test[mask], y_pred[mask]))
            t_r2 = r2_score(y_test[mask], y_pred[mask])
            print(f"    {t:>4.0f}s: RMSE={t_rmse:.4f}, R2={t_r2:.4f} (n={mask.sum()})")

    return model, y_pred, y_test


def experiment_max_adverse_drift(merged, feature_cols):
    """
    TARGET 2: Predict max adverse drift (regression + binary).

    Why this matters:
    - For MM: if you know the mid will drop 0.15, don't quote tight
    - For directional: helps set mental stop-loss thresholds
    - For position sizing: higher expected adverse drift → smaller position
    """
    print(f"\n{'='*70}")
    print("EXPERIMENT 2: MAX ADVERSE DRIFT")
    print(f"{'='*70}")

    target_col = "max_adverse_drift"
    valid = merged.dropna(subset=[target_col])
    print(f"  Samples: {len(valid)}")
    print(f"  Drift stats: mean={valid[target_col].mean():.4f}, std={valid[target_col].std():.4f}")
    print(f"  Drift percentiles: P25={valid[target_col].quantile(0.25):.4f}, "
          f"P50={valid[target_col].quantile(0.50):.4f}, "
          f"P75={valid[target_col].quantile(0.75):.4f}, "
          f"P95={valid[target_col].quantile(0.95):.4f}")

    train_df, val_df, test_df = time_split(valid)

    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df[target_col].values
    X_val = val_df[feature_cols].values.astype(np.float32)
    y_val = val_df[target_col].values
    X_test = test_df[feature_cols].values.astype(np.float32)
    y_test = test_df[target_col].values

    # --- Regression ---
    print(f"\n  [2a] Regression: predict magnitude of max adverse drift")
    model_reg = train_xgb_regressor(X_train, y_train, X_val, y_val, feature_cols)

    X_test_clean = X_test.copy()
    X_test_clean[~np.isfinite(X_test_clean)] = 0.0
    dtest = xgb.DMatrix(X_test_clean, feature_names=feature_cols)
    y_pred_reg = model_reg.predict(dtest)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred_reg))
    mae = mean_absolute_error(y_test, y_pred_reg)
    r2 = r2_score(y_test, y_pred_reg)
    print(f"    RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")

    # --- Binary: does the leader ever lose the lead? ---
    print(f"\n  [2b] Binary: does the leader lose the lead (mid < 0.50)?")
    binary_col = "leader_loses_lead"
    y_train_b = train_df[binary_col].values
    y_val_b = val_df[binary_col].values
    y_test_b = test_df[binary_col].values

    base_rate = y_test_b.mean()
    print(f"    Base rate (loses lead): {base_rate:.1%}")

    model_bin = train_xgb_classifier(
        X_train.copy(), y_train_b, X_val.copy(), y_val_b, feature_cols
    )

    y_pred_b = model_bin.predict(dtest)
    y_pred_class = (y_pred_b >= 0.50).astype(int)
    acc = accuracy_score(y_test_b, y_pred_class)
    try:
        auc = roc_auc_score(y_test_b, y_pred_b)
    except ValueError:
        auc = 0.5

    print(f"    Accuracy: {acc:.1%} (vs base {max(base_rate, 1-base_rate):.1%})")
    print(f"    AUC-ROC:  {auc:.4f}")

    # Threshold sweep for "safe to MM" filter
    print(f"\n    THRESHOLD SWEEP (low prob = safe to MM):")
    print(f"    {'Threshold':>10s}  {'N_safe':>7s}  {'Actual_lose%':>12s}  {'N_risky':>8s}  {'Actual_lose%':>12s}")
    for thr in [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
        safe = y_pred_b < thr
        risky = y_pred_b >= thr
        n_safe = safe.sum()
        n_risky = risky.sum()
        if n_safe < 5 or n_risky < 5:
            continue
        safe_lose = y_test_b[safe].mean()
        risky_lose = y_test_b[risky].mean()
        print(f"    {thr:>10.2f}  {n_safe:>7d}  {safe_lose:>11.1%}  {n_risky:>8d}  {risky_lose:>11.1%}")

    print(f"\n  TOP 15 FEATURES (regression):")
    print_top_features(model_reg)

    # --- By observation time ---
    if "observe_time_s" in test_df.columns:
        print(f"\n  AUC BY OBSERVATION TIME (leader_loses_lead):")
        for t in sorted(test_df["observe_time_s"].unique()):
            mask = test_df["observe_time_s"].values == t
            if mask.sum() < 50:
                continue
            t_labels = y_test_b[mask]
            t_preds = y_pred_b[mask]
            if len(np.unique(t_labels)) < 2:
                continue
            t_auc = roc_auc_score(t_labels, t_preds)
            t_base = t_labels.mean()
            print(f"    {t:>4.0f}s: AUC={t_auc:.4f}, base_rate={t_base:.1%} (n={mask.sum()})")

    return model_reg, model_bin


def experiment_regime(merged, feature_cols):
    """
    TARGET 3: Regime classification (4-class).

    Classes:
      0 = trending_leader   (leader extends lead by >0.10)
      1 = trending_other    (leader loses lead by >0.10)
      2 = choppy            (small net move, high direction changes)
      3 = coin_flip         (moderate, unclear)

    Strategy mapping:
      trending_leader → buy leader early
      trending_other  → buy other / fade leader
      choppy          → market make (ideal!)
      coin_flip       → sit out or very tight MM
    """
    print(f"\n{'='*70}")
    print("EXPERIMENT 3: REGIME CLASSIFICATION (4-class)")
    print(f"{'='*70}")

    target_col = "regime"
    valid = merged.dropna(subset=[target_col])
    valid = valid[valid[target_col].isin([0, 1, 2, 3])]
    print(f"  Samples: {len(valid)}")

    regime_names = {0: "trending_leader", 1: "trending_other", 2: "choppy", 3: "coin_flip"}
    dist = valid[target_col].value_counts().sort_index()
    for cls, count in dist.items():
        print(f"    {regime_names[int(cls)]:<20s}: {count:>6d} ({count/len(valid):.1%})")

    train_df, val_df, test_df = time_split(valid)

    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df[target_col].values.astype(int)
    X_val = val_df[feature_cols].values.astype(np.float32)
    y_val = val_df[target_col].values.astype(int)
    X_test = test_df[feature_cols].values.astype(np.float32)
    y_test = test_df[target_col].values.astype(int)

    model = train_xgb_classifier(X_train, y_train, X_val, y_val, feature_cols, num_classes=4)

    X_test[~np.isfinite(X_test)] = 0.0
    dtest = xgb.DMatrix(X_test, feature_names=feature_cols)
    y_pred_proba = model.predict(dtest)  # shape: (n, 4)
    y_pred = y_pred_proba.argmax(axis=1)

    acc = accuracy_score(y_test, y_pred)
    print(f"\n  Overall accuracy: {acc:.1%}")
    print(f"\n  Classification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=[regime_names[i] for i in range(4)],
        zero_division=0,
    ))

    # --- The key question: does identifying "choppy" markets help MM? ---
    # If the model says "choppy", does the original leader-wins label look more uncertain?
    print(f"  REGIME vs ORIGINAL LABEL (leader_wins):")
    original_labels = test_df["original_label"].values
    for cls in range(4):
        mask = y_pred == cls
        if mask.sum() < 10:
            continue
        wr = original_labels[mask].mean()
        n = mask.sum()
        print(f"    Predicted {regime_names[cls]:<20s}: {n:>5d} markets, leader_wins={wr:.1%}")

    # Confidence-weighted: only count high-confidence regime predictions
    print(f"\n  HIGH-CONFIDENCE REGIME PREDICTIONS (max_prob > 0.50):")
    max_probs = y_pred_proba.max(axis=1)
    for cls in range(4):
        mask = (y_pred == cls) & (max_probs > 0.50)
        if mask.sum() < 10:
            continue
        wr = original_labels[mask].mean()
        n = mask.sum()
        actual_regime = y_test[mask]
        regime_acc = (actual_regime == cls).mean()
        print(f"    {regime_names[cls]:<20s}: {n:>5d} markets, leader_wins={wr:.1%}, regime_acc={regime_acc:.1%}")

    print(f"\n  TOP 15 FEATURES:")
    print_top_features(model)

    return model


def experiment_spread_blowout(merged, feature_cols):
    """
    TARGET 4: Predict spread blowout (binary).

    A spread blowout means the spread more than doubles from its current level.
    This signals liquidity deterioration — dangerous for market makers.
    """
    print(f"\n{'='*70}")
    print("EXPERIMENT 4: SPREAD BLOWOUT (binary)")
    print(f"{'='*70}")

    target_col = "spread_blowout"
    valid = merged.dropna(subset=[target_col])
    print(f"  Samples: {len(valid)}")
    base_rate = valid[target_col].mean()
    print(f"  Base rate (blowout): {base_rate:.1%}")

    train_df, val_df, test_df = time_split(valid)

    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df[target_col].values.astype(int)
    X_val = val_df[feature_cols].values.astype(np.float32)
    y_val = val_df[target_col].values.astype(int)
    X_test = test_df[feature_cols].values.astype(np.float32)
    y_test = test_df[target_col].values.astype(int)

    model = train_xgb_classifier(X_train, y_train, X_val, y_val, feature_cols)

    X_test[~np.isfinite(X_test)] = 0.0
    dtest = xgb.DMatrix(X_test, feature_names=feature_cols)
    y_pred_proba = model.predict(dtest)
    y_pred = (y_pred_proba >= 0.50).astype(int)

    acc = accuracy_score(y_test, y_pred)
    try:
        auc = roc_auc_score(y_test, y_pred_proba)
    except ValueError:
        auc = 0.5

    print(f"\n  TEST RESULTS:")
    print(f"    Accuracy: {acc:.1%} (vs base {max(base_rate, 1-base_rate):.1%})")
    print(f"    AUC-ROC:  {auc:.4f}")

    # --- Use as MM filter ---
    print(f"\n  SPREAD BLOWOUT AS MM FILTER (low = safe):")
    print(f"  {'Threshold':>10s}  {'N_safe':>7s}  {'Blowout%':>10s}  {'N_risky':>8s}  {'Blowout%':>10s}")
    for thr in [0.10, 0.20, 0.30, 0.40, 0.50]:
        safe = y_pred_proba < thr
        risky = y_pred_proba >= thr
        if safe.sum() < 5 or risky.sum() < 5:
            continue
        print(f"  {thr:>10.2f}  {safe.sum():>7d}  {y_test[safe].mean():>9.1%}  "
              f"{risky.sum():>8d}  {y_test[risky].mean():>9.1%}")

    print(f"\n  TOP 15 FEATURES:")
    print_top_features(model)

    return model


def experiment_time_to_resolution(merged, feature_cols):
    """
    TARGET 5: Time to resolution (regression + binary).

    Predicts how quickly the market reaches a decisive price (0.85/0.15).
    Fast resolution → directional signal is valuable, act early.
    Slow resolution → more time for MM to capture spread.
    """
    print(f"\n{'='*70}")
    print("EXPERIMENT 5: TIME TO RESOLUTION")
    print(f"{'='*70}")

    # --- Binary: does it resolve early? ---
    print(f"\n  [5a] Binary: does market resolve in first half of remaining time?")
    target_col = "resolves_early"
    valid = merged.dropna(subset=[target_col])
    base_rate = valid[target_col].mean()
    print(f"    Samples: {len(valid)}, base rate: {base_rate:.1%}")

    train_df, val_df, test_df = time_split(valid)

    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df[target_col].values.astype(int)
    X_val = val_df[feature_cols].values.astype(np.float32)
    y_val = val_df[target_col].values.astype(int)
    X_test = test_df[feature_cols].values.astype(np.float32)
    y_test = test_df[target_col].values.astype(int)

    model = train_xgb_classifier(X_train, y_train, X_val, y_val, feature_cols)

    X_test_clean = X_test.copy()
    X_test_clean[~np.isfinite(X_test_clean)] = 0.0
    dtest = xgb.DMatrix(X_test_clean, feature_names=feature_cols)
    y_pred_proba = model.predict(dtest)
    y_pred = (y_pred_proba >= 0.50).astype(int)

    acc = accuracy_score(y_test, y_pred)
    try:
        auc = roc_auc_score(y_test, y_pred_proba)
    except ValueError:
        auc = 0.5

    print(f"    Accuracy: {acc:.1%}")
    print(f"    AUC-ROC:  {auc:.4f}")

    # Cross-tab with original label
    print(f"\n    EARLY RESOLUTION vs LEADER WINS:")
    original_labels = test_df["original_label"].values
    for pred_resolve in [0, 1]:
        mask = y_pred == pred_resolve
        if mask.sum() < 10:
            continue
        wr = original_labels[mask].mean()
        actual_resolve = y_test[mask].mean()
        label = "resolves_early" if pred_resolve == 1 else "resolves_late"
        print(f"      Predicted {label:<16s}: {mask.sum():>5d}, leader_wins={wr:.1%}, actual_resolve={actual_resolve:.1%}")

    # --- Regression: predict time in seconds ---
    print(f"\n  [5b] Regression: time to resolution (seconds)")
    target_col_r = "time_to_resolution_s"
    y_train_r = train_df[target_col_r].values
    y_val_r = val_df[target_col_r].values
    y_test_r = test_df[target_col_r].values

    print(f"    Target stats: mean={y_test_r.mean():.1f}s, median={np.median(y_test_r):.1f}s")

    model_reg = train_xgb_regressor(
        X_train.copy(), y_train_r, X_val.copy(), y_val_r, feature_cols
    )
    y_pred_r = model_reg.predict(dtest)

    rmse = np.sqrt(mean_squared_error(y_test_r, y_pred_r))
    mae = mean_absolute_error(y_test_r, y_pred_r)
    r2 = r2_score(y_test_r, y_pred_r)
    print(f"    RMSE: {rmse:.1f}s, MAE: {mae:.1f}s, R2: {r2:.4f}")

    print(f"\n  TOP 15 FEATURES (resolves_early):")
    print_top_features(model)

    return model, model_reg


def experiment_combined_action(merged, feature_cols):
    """
    BONUS EXPERIMENT: Optimal action prediction (3-class).

    Instead of predicting market state, directly predict the optimal action:
      0 = sit_out     (no clear edge or adverse conditions)
      1 = mm          (uncertain market, good spread, stable)
      2 = directional (strong signal, leader likely to win decisively)

    The label is computed FROM other targets:
      - directional: leader wins AND final_mid > 0.80 AND no big adverse drift
      - mm:          choppy regime AND no spread blowout AND leader doesn't lose lead
      - sit_out:     everything else
    """
    print(f"\n{'='*70}")
    print("EXPERIMENT 6: OPTIMAL ACTION (3-class)")
    print(f"{'='*70}")

    # Construct the action label
    valid = merged.dropna(subset=["original_label", "final_mid", "max_adverse_drift",
                                   "regime", "spread_blowout", "leader_loses_lead"])

    def compute_action(row):
        # Directional: high confidence win with decisive close
        if row["original_label"] == 1 and row["final_mid"] >= 0.80 and row["max_adverse_drift"] < 0.10:
            return 2  # directional
        # MM: choppy/coin_flip, no spread blowout, leader doesn't lose lead
        if row["regime"] in (2, 3) and row["spread_blowout"] == 0 and row["leader_loses_lead"] == 0:
            return 1  # mm
        return 0  # sit_out

    valid = valid.copy()
    valid["action"] = valid.apply(compute_action, axis=1)

    action_names = {0: "sit_out", 1: "mm", 2: "directional"}
    dist = valid["action"].value_counts().sort_index()
    for cls, count in dist.items():
        print(f"    {action_names[int(cls)]:<15s}: {count:>6d} ({count/len(valid):.1%})")

    train_df, val_df, test_df = time_split(valid)

    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df["action"].values.astype(int)
    X_val = val_df[feature_cols].values.astype(np.float32)
    y_val = val_df["action"].values.astype(int)
    X_test = test_df[feature_cols].values.astype(np.float32)
    y_test = test_df["action"].values.astype(int)

    model = train_xgb_classifier(X_train, y_train, X_val, y_val, feature_cols, num_classes=3)

    X_test[~np.isfinite(X_test)] = 0.0
    dtest = xgb.DMatrix(X_test, feature_names=feature_cols)
    y_pred_proba = model.predict(dtest)  # shape: (n, 3)
    y_pred = y_pred_proba.argmax(axis=1)

    acc = accuracy_score(y_test, y_pred)
    print(f"\n  Overall accuracy: {acc:.1%}")
    print(f"\n  Classification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=[action_names[i] for i in range(3)],
        zero_division=0,
    ))

    # --- PnL simulation ---
    # For each predicted action, compute what WOULD have happened
    print(f"  SIMULATED PnL BY PREDICTED ACTION:")
    original_labels = test_df["original_label"].values
    final_mids = test_df["final_mid"].values

    for act in range(3):
        mask = y_pred == act
        if mask.sum() < 10:
            continue

        n = mask.sum()
        wr = original_labels[mask].mean()
        avg_final = final_mids[mask].mean()

        # Approximate PnL for each action
        if act == 2:  # directional
            # Get entry prices from latest window mid
            latest_mid_cols = sorted([c for c in feature_cols if c.startswith("w") and c.endswith("_mid")])
            if latest_mid_cols:
                latest_w = sorted([int(c.split("_")[0][1:]) for c in latest_mid_cols])[-1]
                entry_mids = test_df.iloc[np.where(mask)[0]][f"w{latest_w}_mid"].values
            else:
                entry_mids = np.full(n, 0.60)
            # PnL: win=$1-entry, lose=-entry
            pnls = np.where(original_labels[mask] == 1, 1.0 - entry_mids, -entry_mids)
            total_pnl = pnls.sum()
            mean_pnl = pnls.mean()
            print(f"    directional: {n:>5d} trades, WR={wr:.1%}, total_PnL=${total_pnl:+.2f}, mean=${mean_pnl:+.4f}")
        elif act == 1:  # mm
            # Approximate: MM makes ~spread/2 per market, loses on adverse selection
            spread_capture = 0.02  # ~$0.20 per 10 tokens
            adverse_rate = 1.0 - original_labels[mask].mean()  # fraction where leader loses
            # Simple model: make spread on winners, lose inventory on losers
            avg_pnl = spread_capture * 10 * 0.5 - adverse_rate * 0.05 * 10
            print(f"    mm:          {n:>5d} trades, WR={wr:.1%} (leader), est_PnL/trade=${avg_pnl:+.4f}")
        else:  # sit_out
            print(f"    sit_out:     {n:>5d} trades (skipped), leader_WR={wr:.1%}")

    print(f"\n  TOP 15 FEATURES:")
    print_top_features(model)

    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-markets", type=int, default=None)
    parser.add_argument("--skip-targets", action="store_true",
                        help="Skip target extraction, use cached targets")
    args = parser.parse_args()

    print("=" * 70)
    print("ALTERNATIVE TARGET VARIABLE EXPLORATION")
    print("=" * 70)

    # Load cached features
    if not FEATURE_CACHE_PATH.exists():
        print(f"ERROR: Feature cache not found at {FEATURE_CACHE_PATH}")
        print("Run xgboost_flexible.py first to build the feature cache.")
        sys.exit(1)

    print(f"\nLoading feature cache from {FEATURE_CACHE_PATH}...")
    features_df = pd.read_parquet(FEATURE_CACHE_PATH)
    print(f"  {len(features_df)} samples, {features_df['slug'].nunique()} markets")

    feature_cols = get_feature_columns(features_df)
    print(f"  {len(feature_cols)} features")

    # Build or load target dataset
    target_cache = Path("data/alternative_targets.parquet")

    if args.skip_targets and target_cache.exists():
        print(f"\nLoading cached targets from {target_cache}...")
        targets_df = pd.read_parquet(target_cache)
    else:
        targets_df = build_target_dataset(features_df, max_markets=args.max_markets)

        if args.max_markets is None:
            target_cache.parent.mkdir(parents=True, exist_ok=True)
            targets_df.to_parquet(target_cache, index=False)
            print(f"Cached targets to {target_cache}")

    # Merge features with targets
    print(f"\nMerging features with targets...")
    merged = features_df.merge(
        targets_df,
        on=["slug", "observe_time_s"],
        how="inner",
    )
    print(f"  Merged: {len(merged)} samples")
    print(f"  Target columns: {[c for c in targets_df.columns if c not in ('slug', 'observe_time_s')]}")

    # --- Run experiments ---
    experiment_final_mid(merged, feature_cols)
    experiment_max_adverse_drift(merged, feature_cols)
    experiment_regime(merged, feature_cols)
    experiment_spread_blowout(merged, feature_cols)
    experiment_time_to_resolution(merged, feature_cols)
    experiment_combined_action(merged, feature_cols)

    # --- Summary ---
    print(f"\n{'='*70}")
    print("SUMMARY AND RECOMMENDATIONS")
    print(f"{'='*70}")
    print("""
  The experiments above test 6 alternative targets. Key things to look for:

  1. FINAL MID (regression): If R2 is meaningfully > 0 and the trading rule
     shows positive EV after fees, this is strictly better than binary
     classification because it captures degree of confidence.

  2. MAX ADVERSE DRIFT: If the "leader loses lead" classifier has AUC > 0.65,
     it's a useful MM filter. Markets where the model says "safe" (low prob
     of losing lead) are ideal for market making.

  3. REGIME: If regime classification accuracy beats the base rate by >10pp
     AND predicted "choppy" markets actually have ~50% leader win rate,
     this directly identifies MM-friendly markets.

  4. SPREAD BLOWOUT: Even moderate AUC (>0.60) is useful here — you just
     need to avoid the worst 20% of markets for MM.

  5. TIME TO RESOLUTION: If early-resolving markets can be identified,
     you know when to deploy directional vs MM strategies.

  6. OPTIMAL ACTION: The holy grail — directly predicting what to do.
     But also the hardest (multi-class, constructed labels).

  NEXT STEPS:
  - Combine the best-performing targets into a multi-output model
  - Use the regime/drift predictions as additional FEATURES in the
    original binary model (stacking)
  - Test ensemble: regime model picks strategy, confidence model
    picks position size
""")


if __name__ == "__main__":
    main()
