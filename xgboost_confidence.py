#!/usr/bin/env python3
"""
XGBoost confidence filter for Polymarket BTC Up/Down markets.

Reads Telonex book snapshot parquet files, computes orderbook features at a
configurable observation time (default: 60s after market open), and trains
an XGBoost classifier to predict whether a directional bet will succeed.

This is a META-MODEL: it doesn't generate trading signals, it scores them.
Any upstream signal (mid-drift, momentum, etc.) can be filtered through
this model's confidence score.

Usage:
    python xgboost_confidence.py
    python xgboost_confidence.py --signal-time 30 --min-markets 500
    python xgboost_confidence.py --data-dir data/telonex_book_snapshots

Requires: xgboost, scikit-learn, pandas, numpy, pyarrow
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
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        log_loss,
        roc_auc_score,
    )
except ImportError:
    print("ERROR: scikit-learn not installed. Run: pip install scikit-learn")
    sys.exit(1)

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MARKET_DURATION_MS = 300_000  # 5 minutes

# Columns available in Telonex book snapshot parquet files
BOOK_COLUMNS = [
    "exchange_timestamp",
    "token_label",
    "asset_id",
    "mid_price",
    "spread",
    "book_imbalance",
    "bid_price_1", "bid_size_1",
    "bid_price_2", "bid_size_2",
    "bid_price_3", "bid_size_3",
    "bid_price_4", "bid_size_4",
    "bid_price_5", "bid_size_5",
    "ask_price_1", "ask_size_1",
    "ask_price_2", "ask_size_2",
    "ask_price_3", "ask_size_3",
    "ask_price_4", "ask_size_4",
    "ask_price_5", "ask_size_5",
]


# ---------------------------------------------------------------------------
# Resolution loading
# ---------------------------------------------------------------------------

def load_resolutions(cache_path: Path) -> dict[str, str]:
    """
    Load resolution_cache.json and return {slug: winning_token_id}.

    The cache format has token_outcomes mapping token_id -> 0.0 or 1.0,
    plus a winning_token field.
    """
    with open(cache_path) as f:
        raw = json.load(f)

    resolutions = {}
    for slug, info in raw.items():
        winner = info.get("winning_token") or info.get("winningTokenId", "")
        if winner:
            resolutions[slug] = winner

    return resolutions


# ---------------------------------------------------------------------------
# Feature extraction for a single market
# ---------------------------------------------------------------------------

def extract_features(
    df: pd.DataFrame,
    open_ms: int,
    signal_time_s: float,
) -> dict | None:
    """
    Extract orderbook features for BOTH tokens at signal time.

    Returns a dict of features, or None if data is insufficient.

    The approach:
    - Filter to the market window [open_ms, open_ms + MARKET_DURATION_MS)
    - For each token (Up/Down), get the book state at signal_time_s
    - Compute features from the snapshot and from the price history up to that point
    """
    close_ms = open_ms + MARKET_DURATION_MS

    # Filter to market window
    market_df = df[
        (df["exchange_timestamp"] >= open_ms) &
        (df["exchange_timestamp"] < close_ms)
    ].copy()

    if market_df.empty:
        return None

    market_df["elapsed_s"] = (market_df["exchange_timestamp"] - open_ms) / 1000.0

    # Split by token
    up_df = market_df[market_df["token_label"] == "Up"].sort_values("exchange_timestamp")
    down_df = market_df[market_df["token_label"] == "Down"].sort_values("exchange_timestamp")

    if up_df.empty or down_df.empty:
        return None

    # Get snapshots up to signal time
    up_pre = up_df[up_df["elapsed_s"] <= signal_time_s]
    down_pre = down_df[down_df["elapsed_s"] <= signal_time_s]

    if up_pre.empty or down_pre.empty:
        return None

    # Latest snapshot at signal time
    up_snap = up_pre.iloc[-1]
    down_snap = down_pre.iloc[-1]

    up_mid = up_snap["mid_price"]
    down_mid = down_snap["mid_price"]

    # Determine which token is "leading" (above 0.50)
    # We always frame the prediction from the perspective of the token above 0.50
    if up_mid >= 0.50:
        leader_snap = up_snap
        leader_pre = up_pre
        leader_label = "Up"
        leader_mid = up_mid
        other_mid = down_mid
    else:
        leader_snap = down_snap
        leader_pre = down_pre
        leader_label = "Down"
        leader_mid = down_mid
        other_mid = up_mid

    # ----- Snapshot features (at signal time) -----

    features = {}

    # Basic book state
    features["mid_price"] = leader_mid
    features["drift_from_50"] = leader_mid - 0.50
    features["abs_drift"] = abs(leader_mid - 0.50)
    features["spread"] = leader_snap["spread"]
    features["book_imbalance"] = (
        leader_snap["book_imbalance"] if "book_imbalance" in leader_snap.index else 0.0
    )

    # Multi-level depth (leader token)
    bid_sizes = []
    ask_sizes = []
    for lvl in range(1, 6):
        bs = leader_snap.get(f"bid_size_{lvl}", 0.0)
        As = leader_snap.get(f"ask_size_{lvl}", 0.0)
        bid_sizes.append(bs if pd.notna(bs) else 0.0)
        ask_sizes.append(As if pd.notna(As) else 0.0)

    features["total_bid_depth_3"] = sum(bid_sizes[:3])
    features["total_ask_depth_3"] = sum(ask_sizes[:3])
    features["total_bid_depth_5"] = sum(bid_sizes)
    features["total_ask_depth_5"] = sum(ask_sizes)

    total_depth_3 = features["total_bid_depth_3"] + features["total_ask_depth_3"]
    features["depth_imbalance_3"] = (
        (features["total_bid_depth_3"] - features["total_ask_depth_3"]) / total_depth_3
        if total_depth_3 > 0 else 0.0
    )

    total_depth_5 = features["total_bid_depth_5"] + features["total_ask_depth_5"]
    features["depth_imbalance_5"] = (
        (features["total_bid_depth_5"] - features["total_ask_depth_5"]) / total_depth_5
        if total_depth_5 > 0 else 0.0
    )

    # Bid/ask slope (price decay across levels)
    bp1 = leader_snap.get("bid_price_1", np.nan)
    bp3 = leader_snap.get("bid_price_3", np.nan)
    ap1 = leader_snap.get("ask_price_1", np.nan)
    ap3 = leader_snap.get("ask_price_3", np.nan)

    features["bid_slope"] = (bp1 - bp3) / 2.0 if pd.notna(bp1) and pd.notna(bp3) else 0.0
    features["ask_slope"] = (ap3 - ap1) / 2.0 if pd.notna(ap1) and pd.notna(ap3) else 0.0

    # ----- Price dynamics (time series up to signal time) -----

    mids = leader_pre["mid_price"].values
    timestamps = leader_pre["exchange_timestamp"].values

    n_ticks = len(mids)
    features["n_ticks_pre_signal"] = n_ticks

    if n_ticks >= 3:
        # Drift speed: linear slope of mid_price
        t_seconds = (timestamps - timestamps[0]) / 1000.0
        if t_seconds[-1] > 0:
            # Simple linear regression slope
            t_mean = t_seconds.mean()
            m_mean = mids.mean()
            numerator = np.sum((t_seconds - t_mean) * (mids - m_mean))
            denominator = np.sum((t_seconds - t_mean) ** 2)
            features["drift_speed"] = numerator / denominator if denominator > 0 else 0.0

            # R-squared of linear trend
            predicted = m_mean + features["drift_speed"] * (t_seconds - t_mean)
            ss_res = np.sum((mids - predicted) ** 2)
            ss_tot = np.sum((mids - m_mean) ** 2)
            features["trend_r_squared"] = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        else:
            features["drift_speed"] = 0.0
            features["trend_r_squared"] = 0.0

        # Realized volatility
        mid_changes = np.diff(mids)
        features["realized_vol"] = np.std(mid_changes) if len(mid_changes) > 1 else 0.0

        # Price range
        features["high_low_range"] = mids.max() - mids.min()

        # Choppiness: range / drift. High = choppy, low = trending
        abs_drift = abs(mids[-1] - mids[0])
        features["choppiness"] = (
            features["high_low_range"] / abs_drift
            if abs_drift > 0.001 else 10.0  # cap at 10 for near-zero drift
        )

        # Direction changes: how many times did mid cross its starting value?
        start_mid = mids[0]
        crossings = np.sum(np.diff(np.sign(mids - start_mid)) != 0)
        features["direction_changes"] = int(crossings)

        # Drift consistency: fraction of ticks where price moved in leader direction
        if len(mid_changes) > 0:
            if mids[-1] > mids[0]:
                features["drift_consistency"] = np.mean(mid_changes > 0)
            else:
                features["drift_consistency"] = np.mean(mid_changes < 0)
        else:
            features["drift_consistency"] = 0.5

        # Acceleration: compare drift speed in first vs second half
        half = len(mids) // 2
        if half > 1:
            first_half_drift = (mids[half] - mids[0])
            second_half_drift = (mids[-1] - mids[half])
            t_first = (timestamps[half] - timestamps[0]) / 1000.0
            t_second = (timestamps[-1] - timestamps[half]) / 1000.0
            speed_first = first_half_drift / t_first if t_first > 0 else 0.0
            speed_second = second_half_drift / t_second if t_second > 0 else 0.0
            features["drift_acceleration"] = speed_second - speed_first
        else:
            features["drift_acceleration"] = 0.0

        # Max drift seen so far
        features["max_drift_seen"] = np.max(np.abs(mids - 0.50))

    else:
        # Not enough data points for dynamics
        features["drift_speed"] = 0.0
        features["trend_r_squared"] = 0.0
        features["realized_vol"] = 0.0
        features["high_low_range"] = 0.0
        features["choppiness"] = 10.0
        features["direction_changes"] = 0
        features["drift_consistency"] = 0.5
        features["drift_acceleration"] = 0.0
        features["max_drift_seen"] = abs(leader_mid - 0.50)

    # ----- Cross-token features -----
    features["up_down_mid_sum"] = up_mid + down_mid  # should be ~1.0
    features["up_down_spread_diff"] = up_snap["spread"] - down_snap["spread"]

    # ----- Update frequency (data quality proxy) -----
    if n_ticks >= 2:
        duration_s = (timestamps[-1] - timestamps[0]) / 1000.0
        features["update_frequency"] = n_ticks / duration_s if duration_s > 0 else 0.0
    else:
        features["update_frequency"] = 0.0

    # ----- Store metadata (not used as features) -----
    features["_leader_label"] = leader_label
    features["_leader_asset_id"] = leader_snap["asset_id"]

    return features


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_dataset(
    data_dir: Path,
    resolution_cache_path: Path,
    signal_time_s: float = 60.0,
    min_markets: int = 100,
    max_markets: int | None = None,
) -> pd.DataFrame:
    """
    Build the feature matrix from Telonex parquet files.

    Returns a DataFrame where each row is one market, with features and labels.
    """
    resolutions = load_resolutions(resolution_cache_path)
    print(f"Loaded {len(resolutions)} resolutions")

    parquet_files = sorted(data_dir.glob("btc-updown-5m-*.parquet"))
    print(f"Found {len(parquet_files)} parquet files in {data_dir}")

    if max_markets:
        parquet_files = parquet_files[:max_markets]

    rows = []
    skipped = {"no_resolution": 0, "parse_error": 0, "no_features": 0, "no_winner_match": 0}

    for i, pf in enumerate(parquet_files):
        slug = pf.stem

        if slug not in resolutions:
            skipped["no_resolution"] += 1
            continue

        try:
            open_ts = int(slug.split("-")[-1])
        except ValueError:
            skipped["parse_error"] += 1
            continue

        open_ms = open_ts * 1000

        try:
            # Read available columns — some files may lack higher-level book data
            df = pd.read_parquet(pf)
            available = [c for c in BOOK_COLUMNS if c in df.columns]
            df = df[available]
        except Exception:
            skipped["parse_error"] += 1
            continue

        # Must have minimum required columns
        required = {"exchange_timestamp", "token_label", "asset_id", "mid_price", "spread"}
        if not required.issubset(set(df.columns)):
            skipped["parse_error"] += 1
            continue

        features = extract_features(df, open_ms, signal_time_s)
        if features is None:
            skipped["no_features"] += 1
            continue

        # Determine label: did the leading token (above 0.50) win?
        winning_token = resolutions[slug]
        leader_asset_id = features.pop("_leader_asset_id")
        leader_label = features.pop("_leader_label")

        # Check if the winning token matches the leader
        if leader_asset_id == winning_token:
            label = 1  # leading token won
        else:
            # Verify the winning token is the OTHER token
            # (it should be, in a two-token market)
            label = 0

        features["label"] = label
        features["slug"] = slug
        features["open_ts"] = open_ts
        rows.append(features)

        if (i + 1) % 500 == 0:
            print(f"  Processed {i+1}/{len(parquet_files)}, extracted {len(rows)} markets...")

    print(f"\nExtracted features for {len(rows)} markets")
    print(f"Skipped: {skipped}")

    if len(rows) < min_markets:
        print(f"ERROR: Only {len(rows)} markets (need {min_markets}). Check data paths.")
        sys.exit(1)

    dataset = pd.DataFrame(rows)

    # Sort by time for proper train/test split
    dataset = dataset.sort_values("open_ts").reset_index(drop=True)

    return dataset


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return columns to use as features (exclude metadata and label)."""
    exclude = {"label", "slug", "open_ts"}
    return [c for c in df.columns if c not in exclude]


def train_and_evaluate(
    dataset: pd.DataFrame,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
):
    """
    Train XGBoost with time-based split and evaluate.

    Split: first 70% train, next 15% validation (early stopping), final 15% test.
    No shuffling -- this is critical to avoid future leakage.
    """
    feature_cols = get_feature_columns(dataset)
    print(f"\nFeatures ({len(feature_cols)}): {feature_cols}")

    n = len(dataset)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    train_df = dataset.iloc[:train_end]
    val_df = dataset.iloc[train_end:val_end]
    test_df = dataset.iloc[val_end:]

    print(f"\nDataset split (time-ordered, no shuffling):")
    print(f"  Train: {len(train_df)} markets ({train_df['open_ts'].min()} - {train_df['open_ts'].max()})")
    print(f"  Val:   {len(val_df)} markets ({val_df['open_ts'].min()} - {val_df['open_ts'].max()})")
    print(f"  Test:  {len(test_df)} markets ({test_df['open_ts'].min()} - {test_df['open_ts'].max()})")

    X_train = train_df[feature_cols].values
    y_train = train_df["label"].values
    X_val = val_df[feature_cols].values
    y_val = val_df["label"].values
    X_test = test_df[feature_cols].values
    y_test = test_df["label"].values

    print(f"\n  Train base rate: {y_train.mean():.3f}")
    print(f"  Val base rate:   {y_val.mean():.3f}")
    print(f"  Test base rate:  {y_test.mean():.3f}")

    # ----- Train XGBoost -----

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_cols)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_cols)
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_cols)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 10,
        "lambda": 1.0,       # L2 regularization
        "alpha": 0.1,        # L1 regularization
        "seed": 42,
        "verbosity": 0,
    }

    print(f"\nTraining XGBoost...")
    print(f"  Params: max_depth={params['max_depth']}, lr={params['learning_rate']}, "
          f"subsample={params['subsample']}, colsample={params['colsample_bytree']}")

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=50,
        verbose_eval=100,
    )

    best_iteration = model.best_iteration
    print(f"\n  Best iteration: {best_iteration}")

    # ----- Evaluate on test set -----

    print(f"\n{'='*70}")
    print("TEST SET RESULTS")
    print(f"{'='*70}")

    y_pred_proba = model.predict(dtest)
    y_pred = (y_pred_proba >= 0.50).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    ll = log_loss(y_test, y_pred_proba)

    print(f"\n  Accuracy:  {accuracy:.4f}")
    print(f"  AUC-ROC:   {auc:.4f}")
    print(f"  Log Loss:  {ll:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Loses", "Wins"]))

    # ----- Feature importance -----

    print(f"\n{'='*70}")
    print("FEATURE IMPORTANCE (gain)")
    print(f"{'='*70}")

    importance = model.get_score(importance_type="gain")
    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)

    for feat, gain in sorted_imp:
        bar = "#" * int(gain / max(importance.values()) * 40)
        print(f"  {feat:<28s} {gain:>10.1f}  {bar}")

    # ----- Threshold sweep (the key output) -----

    print(f"\n{'='*70}")
    print("THRESHOLD SWEEP — Confidence Filter Performance")
    print(f"{'='*70}")
    print(f"{'Threshold':>10s}  {'Trades':>7s}  {'Win%':>6s}  {'EV/trade':>10s}  "
          f"{'Precision':>10s}  {'Coverage':>10s}")
    print("-" * 70)

    # Assume taker entry at ask. For simplicity, assume avg entry = 0.55
    # (mid + half spread). The real pipeline should use actual ask prices.
    # Here we approximate EV as: win_rate - avg_entry_price
    # Since we're buying the token at ~mid, EV/dollar = win_rate - mid

    for threshold in [0.45, 0.50, 0.52, 0.55, 0.58, 0.60, 0.65, 0.70, 0.75, 0.80]:
        mask = y_pred_proba >= threshold
        n_trades = mask.sum()
        if n_trades < 10:
            continue

        win_rate = y_test[mask].mean()
        # EV per trade: simplified — we buy at ~mid when signal fires.
        # If avg mid of filtered trades is M, EV = win_rate * (1 - M) - (1 - win_rate) * M
        #                                       = win_rate - M
        avg_mid = test_df.iloc[np.where(mask)[0]]["mid_price"].mean()
        ev_per_trade = win_rate - avg_mid

        coverage = n_trades / len(y_test)
        # Precision = win_rate for the positive class (trades taken)
        precision = win_rate

        print(f"  {threshold:>8.2f}  {n_trades:>7d}  {win_rate:>5.1%}  {ev_per_trade:>+9.4f}  "
              f"  {precision:>8.1%}    {coverage:>8.1%}")

    # ----- Calibration check -----

    print(f"\n{'='*70}")
    print("CALIBRATION CHECK — Predicted vs Actual Win Rate")
    print(f"{'='*70}")
    print(f"{'Pred Bucket':>12s}  {'N':>6s}  {'Predicted':>10s}  {'Actual':>10s}  {'Gap':>8s}")
    print("-" * 55)

    for lo, hi in [(0.0, 0.3), (0.3, 0.4), (0.4, 0.45), (0.45, 0.5),
                   (0.5, 0.55), (0.55, 0.6), (0.6, 0.7), (0.7, 1.0)]:
        mask = (y_pred_proba >= lo) & (y_pred_proba < hi)
        n_bucket = mask.sum()
        if n_bucket < 5:
            continue
        pred_mean = y_pred_proba[mask].mean()
        actual_mean = y_test[mask].mean()
        gap = actual_mean - pred_mean
        print(f"  {lo:.2f}-{hi:.2f}  {n_bucket:>6d}  {pred_mean:>9.3f}  {actual_mean:>9.3f}  {gap:>+7.3f}")

    return model, test_df, y_pred_proba


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        default="data/telonex_book_snapshots",
        help="Directory containing Telonex parquet files",
    )
    parser.add_argument(
        "--resolution-cache",
        default="data/resolution_cache.json",
        help="Path to resolution_cache.json",
    )
    parser.add_argument(
        "--signal-time",
        type=float,
        default=60.0,
        help="Seconds after market open to observe (default: 60)",
    )
    parser.add_argument(
        "--min-markets",
        type=int,
        default=100,
        help="Minimum markets required to proceed (default: 100)",
    )
    parser.add_argument(
        "--max-markets",
        type=int,
        default=None,
        help="Cap on number of markets to process (for testing)",
    )
    args = parser.parse_args()

    print(f"XGBoost Confidence Filter Pipeline")
    print(f"{'='*50}")
    print(f"  Data dir:        {args.data_dir}")
    print(f"  Resolution cache: {args.resolution_cache}")
    print(f"  Signal time:     {args.signal_time}s after open")
    print()

    dataset = build_dataset(
        data_dir=Path(args.data_dir),
        resolution_cache_path=Path(args.resolution_cache),
        signal_time_s=args.signal_time,
        min_markets=args.min_markets,
        max_markets=args.max_markets,
    )

    print(f"\nDataset shape: {dataset.shape}")
    print(f"Label distribution: {dataset['label'].value_counts().to_dict()}")
    print(f"Base rate (leader wins): {dataset['label'].mean():.3f}")

    model, test_df, predictions = train_and_evaluate(dataset)

    print(f"\nDone. Model trained on {len(dataset)} markets.")
    print(f"Use threshold sweep above to pick an operating point for live trading.")


if __name__ == "__main__":
    main()
