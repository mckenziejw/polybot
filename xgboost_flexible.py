#!/usr/bin/env python3
"""
XGBoost confidence filter v3 — flexible multi-window + BTC + micro features.

Key improvements over v1 (xgboost_confidence.py):
  1. Multi-window features: extracts orderbook state at 10s, 20s, 30s, 45s, 60s, 90s, 120s
  2. Trajectory features: how drift/spread/depth evolve over time (not just a snapshot)
  3. BTC price context: returns and vol from Binance BTC/USDT aligned to market open
  4. Adaptive observation: model receives time_elapsed as a feature, can be queried at any time
  5. The model is a general confidence scorer, not tied to any specific signal

v2.1 fixes:
  - Resample all orderbook data to 1s bars (matches live data, 10-50x faster)
  - Drop btc_mid (absolute price), replace with btc_mid_zscore (normalized)
  - Drop n_ticks/update_freq (distribution shift risk), no replacements needed
  - Feature caching: saves extracted features to parquet, subsequent runs load in <1s
  - BTC lookups use searchsorted for O(log n) instead of O(n) scans

v3 additions (from overnight research):
  - BTC volatility regime features: btc_vol_5m, btc_vol_30s_raw, btc_range_900s (Agent 6)
  - Microstructure dynamics: cross_mid_sum_trend, ask_depth_2c, spread_min,
    imb_std_10s, imb_max_10s, spread_quantile_75 (Agent 5)
  - Hyperparameter tuning: max_depth=3, lr=0.01 (Agent 8: shallower = less overfit)

Usage:
    python xgboost_flexible.py
    python xgboost_flexible.py --observe-at 30     # score at 30s instead of 60s
    python xgboost_flexible.py --multi-window       # train with all windows (default)
    python xgboost_flexible.py --max-markets 500    # quick test
    python xgboost_flexible.py --rebuild-cache      # force rebuild feature cache

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

# Windows at which we extract features (seconds after market open)
OBSERVATION_WINDOWS = [10, 20, 30, 45, 60, 90, 120]

# BTC feature lookback windows relative to market open (seconds before)
BTC_LOOKBACKS = [60, 300, 900, 3600]  # 1m, 5m, 15m, 1h

FEATURE_CACHE_PATH = Path("data/xgb_features_v3.parquet")

# v3.2: features dropped due to near-perfect correlation with drift_from_50
# abs_drift = |drift_from_50|, other_mid = 1 - leader_mid, mid_sum = leader_mid + other_mid
# max_drift KEPT — decorrelates during mean reversions (important signal)
DROP_FEATURES = {"abs_drift", "other_mid", "mid_sum"}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_resolutions(cache_path: Path) -> dict[str, str]:
    with open(cache_path) as f:
        raw = json.load(f)
    resolutions = {}
    for slug, info in raw.items():
        winner = info.get("winning_token") or info.get("winningTokenId", "")
        if winner:
            resolutions[slug] = winner
    return resolutions


def load_btc_quotes(btc_path: Path) -> pd.DataFrame | None:
    if not btc_path.exists():
        return None
    df = pd.read_parquet(btc_path)
    df = df.sort_values("timestamp_ms").reset_index(drop=True)
    return df


def resample_to_1s(token_df: pd.DataFrame, open_ms: int, close_ms: int) -> pd.DataFrame:
    """Resample raw orderbook ticks to 1-second bars (last value per second).

    This matches the resolution of our BTC training data and normalizes
    the data density across Telonex historical and live WebSocket feeds.
    """
    market_df = token_df[
        (token_df["exchange_timestamp"] >= open_ms) &
        (token_df["exchange_timestamp"] < close_ms)
    ].copy()

    if market_df.empty:
        return market_df

    # Compute second offset from market open
    market_df["sec"] = ((market_df["exchange_timestamp"] - open_ms) / 1000).astype(int)

    # Take last snapshot per second (same as forward-fill behavior)
    resampled = market_df.groupby("sec").last().reset_index()

    # Restore exchange_timestamp as the second boundary for consistency
    resampled["exchange_timestamp"] = open_ms + resampled["sec"] * 1000

    return resampled


# ---------------------------------------------------------------------------
# BTC feature extraction (optimized with searchsorted)
# ---------------------------------------------------------------------------

def extract_btc_features(
    btc_ts: np.ndarray,
    btc_mids: np.ndarray,
    btc_spreads: np.ndarray,
    btc_log_returns: np.ndarray,
    btc_rvols: dict[str, np.ndarray],
    market_open_ms: int,
    observe_ms: int,
    btc_mid_mean: float,
    btc_mid_std: float,
) -> dict:
    """Extract BTC context features using pre-sorted arrays and searchsorted."""
    features = {}

    obs_ts = market_open_ms + observe_ms
    obs_idx = np.searchsorted(btc_ts, obs_ts, side="right") - 1

    if obs_idx < 0:
        return _empty_btc_features()

    btc_mid = btc_mids[obs_idx]

    # Normalized BTC price (z-score) instead of raw absolute price
    features["btc_mid_zscore"] = (btc_mid - btc_mid_mean) / btc_mid_std if btc_mid_std > 0 else 0.0
    features["btc_spread"] = btc_spreads[obs_idx]
    features["btc_log_return"] = btc_log_returns[obs_idx]

    for window in ["30s", "60s", "300s"]:
        key = f"rvol_{window}"
        if key in btc_rvols:
            features[f"btc_rvol_{window}"] = btc_rvols[key][obs_idx]
        else:
            features[f"btc_rvol_{window}"] = 0.0

    # BTC returns over lookback windows relative to market open
    for lb in BTC_LOOKBACKS:
        lb_ms = market_open_ms - lb * 1000
        lb_idx = np.searchsorted(btc_ts, lb_ms, side="right") - 1
        if lb_idx >= 0:
            lb_mid = btc_mids[lb_idx]
            if lb_mid > 0:
                features[f"btc_ret_{lb}s"] = np.log(btc_mid / lb_mid)
            else:
                features[f"btc_ret_{lb}s"] = 0.0
        else:
            features[f"btc_ret_{lb}s"] = 0.0

    # BTC return during the market window (open to observation)
    open_idx = np.searchsorted(btc_ts, market_open_ms, side="right") - 1
    if open_idx >= 0:
        open_mid = btc_mids[open_idx]
        if open_mid > 0:
            features["btc_ret_since_open"] = np.log(btc_mid / open_mid)
        else:
            features["btc_ret_since_open"] = 0.0
    else:
        features["btc_ret_since_open"] = 0.0

    # BTC volatility during market window
    window_start_idx = np.searchsorted(btc_ts, market_open_ms, side="left")
    window_end_idx = np.searchsorted(btc_ts, obs_ts, side="right")

    if window_end_idx - window_start_idx >= 3:
        window_returns = btc_log_returns[window_start_idx:window_end_idx]
        valid = window_returns[np.isfinite(window_returns)]
        features["btc_window_vol"] = np.std(valid) if len(valid) > 1 else 0.0
        window_mids = btc_mids[window_start_idx:window_end_idx]
        features["btc_window_range"] = (
            window_mids.max() - window_mids.min()
        ) / btc_mid if btc_mid > 0 else 0.0
    else:
        features["btc_window_vol"] = 0.0
        features["btc_window_range"] = 0.0

    # --- BTC volatility regime features (v3) ---
    # Annualized vol over 5min and raw 30s vol are top-ranking vol features
    for lb_s, label in [(30, "30s"), (300, "5m"), (900, "15m")]:
        lb_ms = lb_s * 1000
        start_idx = np.searchsorted(btc_ts, obs_ts - lb_ms, side="left")
        end_idx = np.searchsorted(btc_ts, obs_ts, side="right")
        if end_idx - start_idx > 5:
            rets = btc_log_returns[start_idx:end_idx]
            rets = rets[np.isfinite(rets)]
            if len(rets) > 1:
                features[f"btc_vol_{label}"] = float(np.std(rets) * np.sqrt(len(rets)))
                features[f"btc_vol_{label}_raw"] = float(np.std(rets))
            else:
                features[f"btc_vol_{label}"] = 0.0
                features[f"btc_vol_{label}_raw"] = 0.0
            lb_mids = btc_mids[start_idx:end_idx]
            features[f"btc_range_{label}"] = float(
                (lb_mids.max() - lb_mids.min()) / btc_mid
            ) if btc_mid > 0 and len(lb_mids) > 0 else 0.0
        else:
            features[f"btc_vol_{label}"] = 0.0
            features[f"btc_vol_{label}_raw"] = 0.0
            features[f"btc_range_{label}"] = 0.0

    return features


def _empty_btc_features() -> dict:
    features = {
        "btc_mid_zscore": 0.0, "btc_spread": 0.0, "btc_log_return": 0.0,
        "btc_rvol_30s": 0.0, "btc_rvol_60s": 0.0, "btc_rvol_300s": 0.0,
        "btc_ret_since_open": 0.0, "btc_window_vol": 0.0, "btc_window_range": 0.0,
    }
    for lb in BTC_LOOKBACKS:
        features[f"btc_ret_{lb}s"] = 0.0
    # v3 vol regime features
    for label in ["30s", "5m", "15m"]:
        features[f"btc_vol_{label}"] = 0.0
        features[f"btc_vol_{label}_raw"] = 0.0
        features[f"btc_range_{label}"] = 0.0
    return features


# ---------------------------------------------------------------------------
# Orderbook feature extraction
# ---------------------------------------------------------------------------

def extract_book_snapshot_features(snap: pd.Series, prefix: str = "") -> dict:
    """Extract features from a single book snapshot."""
    f = {}
    p = prefix

    f[f"{p}mid"] = snap["mid_price"]
    f[f"{p}spread"] = snap.get("spread", 0.0)
    f[f"{p}imbalance"] = snap.get("book_imbalance", 0.0)

    bid_sizes = [snap.get(f"bid_size_{i}", 0.0) or 0.0 for i in range(1, 6)]
    ask_sizes = [snap.get(f"ask_size_{i}", 0.0) or 0.0 for i in range(1, 6)]

    raw_bid_depth_3 = sum(bid_sizes[:3])
    raw_ask_depth_3 = sum(ask_sizes[:3])

    total = raw_bid_depth_3 + raw_ask_depth_3
    # Normalize depth to fraction of total (regime-invariant)
    f[f"{p}bid_depth_3"] = raw_bid_depth_3 / total if total > 0 else 0.0
    f[f"{p}ask_depth_3"] = raw_ask_depth_3 / total if total > 0 else 0.0

    f[f"{p}depth_imb_3"] = (
        (raw_bid_depth_3 - raw_ask_depth_3) / total if total > 0 else 0.0
    )

    return f


def extract_trajectory_features(
    mids: np.ndarray,
    timestamps: np.ndarray,
) -> dict:
    """Extract features from the price trajectory up to observation time."""
    f = {}
    n = len(mids)

    if n < 3:
        return {
            "drift_speed": 0.0, "drift_accel": 0.0, "r_squared": 0.0,
            "realized_vol": 0.0, "range": 0.0, "choppiness": 10.0,
            "direction_changes": 0, "consistency": 0.5, "max_drift": 0.0,
            "drift_from_50": 0.0, "abs_drift": 0.0,
        }

    t_s = (timestamps - timestamps[0]) / 1000.0
    duration = t_s[-1]

    # Current drift
    f["drift_from_50"] = mids[-1] - 0.50
    f["abs_drift"] = abs(f["drift_from_50"])

    # Linear regression slope (drift speed)
    if duration > 0:
        t_mean, m_mean = t_s.mean(), mids.mean()
        num = np.sum((t_s - t_mean) * (mids - m_mean))
        den = np.sum((t_s - t_mean) ** 2)
        f["drift_speed"] = num / den if den > 0 else 0.0

        predicted = m_mean + f["drift_speed"] * (t_s - t_mean)
        ss_res = np.sum((mids - predicted) ** 2)
        ss_tot = np.sum((mids - m_mean) ** 2)
        f["r_squared"] = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    else:
        f["drift_speed"] = 0.0
        f["r_squared"] = 0.0

    # Acceleration (first half vs second half speed)
    half = n // 2
    if half > 1 and duration > 0:
        t_mid = t_s[half]
        speed1 = (mids[half] - mids[0]) / t_mid if t_mid > 0 else 0.0
        speed2 = (mids[-1] - mids[half]) / (duration - t_mid) if (duration - t_mid) > 0 else 0.0
        f["drift_accel"] = speed2 - speed1
    else:
        f["drift_accel"] = 0.0

    # Volatility and range
    changes = np.diff(mids)
    f["realized_vol"] = np.std(changes) if len(changes) > 1 else 0.0
    f["range"] = mids.max() - mids.min()

    # Choppiness
    abs_drift = abs(mids[-1] - mids[0])
    f["choppiness"] = f["range"] / abs_drift if abs_drift > 0.001 else 10.0

    # Direction changes
    f["direction_changes"] = int(np.sum(np.diff(np.sign(mids - mids[0])) != 0))

    # Consistency
    if len(changes) > 0:
        if mids[-1] > mids[0]:
            f["consistency"] = float(np.mean(changes > 0))
        else:
            f["consistency"] = float(np.mean(changes < 0))
    else:
        f["consistency"] = 0.5

    # Max drift from 0.50 seen at any point
    f["max_drift"] = float(np.max(np.abs(mids - 0.50)))

    return f


def extract_multi_window_features(
    token_df: pd.DataFrame,
    open_ms: int,
    windows: list[int],
    observe_s: float,
) -> dict:
    """
    Extract features at multiple time windows, plus trajectory features.

    token_df is already resampled to 1s bars with a 'sec' column.
    """
    features = {}

    # Filter to windows we can actually observe
    active_windows = [w for w in windows if w <= observe_s]
    if not active_windows:
        active_windows = [observe_s]

    prev_mid = None
    prev_spread = None

    for w in active_windows:
        pre = token_df[token_df["sec"] <= w]
        if pre.empty:
            continue

        snap = pre.iloc[-1]
        snap_f = extract_book_snapshot_features(snap, prefix=f"w{w}_")
        features.update(snap_f)

        # Delta from previous window
        if prev_mid is not None:
            features[f"w{w}_mid_delta"] = snap_f[f"w{w}_mid"] - prev_mid
            features[f"w{w}_spread_delta"] = snap_f[f"w{w}_spread"] - prev_spread

        prev_mid = snap_f[f"w{w}_mid"]
        prev_spread = snap_f[f"w{w}_spread"]

    # Trajectory features over the full observation window
    pre_all = token_df[token_df["sec"] <= observe_s]
    if len(pre_all) >= 3:
        traj = extract_trajectory_features(
            pre_all["mid_price"].values,
            pre_all["exchange_timestamp"].values,
        )
        features.update(traj)
    else:
        features.update({
            "drift_from_50": 0.0, "abs_drift": 0.0, "drift_speed": 0.0,
            "drift_accel": 0.0, "r_squared": 0.0, "realized_vol": 0.0,
            "range": 0.0, "choppiness": 10.0, "direction_changes": 0,
            "consistency": 0.5, "max_drift": 0.0,
        })

    # How many windows are we basing this on?
    features["n_windows_observed"] = len([w for w in active_windows if f"w{w}_mid" in features])

    return features


# ---------------------------------------------------------------------------
# Microstructure dynamics features (v3)
# ---------------------------------------------------------------------------

def _extract_micro_features(
    leader_df: pd.DataFrame,
    other_df: pd.DataFrame,
    observe_s: float,
) -> dict:
    """Extract the most useful microstructure dynamics features.

    Selected from Agent 5 research (71 features tested, 6 kept based on
    importance ranking and marginal AUC contribution):
      - cross_mid_sum_trend: Up+Down mid-sum trend (pricing efficiency drift)
      - ask_depth_2c: cumulative ask depth within 2 cents of mid
      - spread_min: minimum spread seen during observation
      - imb_std_10s: book imbalance volatility over recent 10s
      - imb_max_10s: max book imbalance in recent 10s
      - spread_quantile_75: 75th percentile of spread during observation
    """
    f = {}
    leader_pre = leader_df[leader_df["sec"] <= observe_s]
    other_pre = other_df[other_df["sec"] <= observe_s]
    n_leader = len(leader_pre)
    n_other = len(other_pre)

    # --- cross_mid_sum_trend: is Up+Down mid-sum drifting from 1.0? ---
    if n_leader >= 5 and n_other >= 5:
        # Align on common seconds
        leader_by_sec = leader_pre.set_index("sec")["mid_price"]
        other_by_sec = other_pre.set_index("sec")["mid_price"]
        common = sorted(set(leader_by_sec.index) & set(other_by_sec.index))
        if len(common) >= 5:
            mid_sums = leader_by_sec.reindex(common).values + other_by_sec.reindex(common).values
            # Linear trend of mid_sum over time
            x = np.arange(len(mid_sums), dtype=np.float64)
            x_mean = x.mean()
            ms_mean = mid_sums.mean()
            denom = np.sum((x - x_mean) ** 2)
            f["cross_mid_sum_trend"] = float(
                np.sum((x - x_mean) * (mid_sums - ms_mean)) / denom
            ) if denom > 0 else 0.0
        else:
            f["cross_mid_sum_trend"] = 0.0
    else:
        f["cross_mid_sum_trend"] = 0.0

    # --- ask_depth_2c: cumulative ask depth within 2 cents of mid ---
    # Normalized by total top-3 depth for regime invariance
    if n_leader > 0:
        snap = leader_pre.iloc[-1]
        mid = snap["mid_price"]
        ask_depth_2c = 0.0
        for i in range(1, 11):
            ap = snap.get(f"ask_price_{i}", None)
            az = snap.get(f"ask_size_{i}", 0.0) or 0.0
            if ap is not None and not (isinstance(ap, float) and np.isnan(ap)):
                if float(ap) <= mid + 0.02:
                    ask_depth_2c += float(az)
        # Normalize by total depth (bid+ask top-3) at this snapshot
        bid_d = sum(float(snap.get(f"bid_size_{i}", 0.0) or 0.0) for i in range(1, 4))
        ask_d = sum(float(snap.get(f"ask_size_{i}", 0.0) or 0.0) for i in range(1, 4))
        total_d = bid_d + ask_d
        f["ask_depth_2c"] = ask_depth_2c / total_d if total_d > 0 else 0.0
    else:
        f["ask_depth_2c"] = 0.0

    # --- spread_min, spread_quantile_75: spread regime features ---
    if n_leader >= 3:
        spreads = leader_pre["spread"].dropna().values
        if len(spreads) >= 3:
            f["spread_min"] = float(np.min(spreads))
            f["spread_quantile_75"] = float(np.percentile(spreads, 75))
        else:
            f["spread_min"] = 0.0
            f["spread_quantile_75"] = 0.0
    else:
        f["spread_min"] = 0.0
        f["spread_quantile_75"] = 0.0

    # --- imb_std_10s, imb_max_10s: book imbalance dynamics ---
    if n_leader >= 3 and "book_imbalance" in leader_pre.columns:
        imb = leader_pre["book_imbalance"].dropna().values
        recent = imb[-min(10, len(imb)):]
        f["imb_std_10s"] = float(np.std(recent)) if len(recent) >= 2 else 0.0
        f["imb_max_10s"] = float(np.max(recent)) if len(recent) >= 1 else 0.0
    else:
        f["imb_std_10s"] = 0.0
        f["imb_max_10s"] = 0.0

    return f


# ---------------------------------------------------------------------------
# Full market feature extraction
# ---------------------------------------------------------------------------

def extract_market_features(
    up_1s: pd.DataFrame,
    down_1s: pd.DataFrame,
    open_ms: int,
    observe_s: float,
    btc_arrays: dict | None = None,
    windows: list[int] | None = None,
) -> dict | None:
    """Extract all features for a single market at the given observation time.

    up_1s and down_1s are already resampled to 1s bars with a 'sec' column.
    """
    if windows is None:
        windows = OBSERVATION_WINDOWS

    if up_1s.empty or down_1s.empty:
        return None

    # Determine leader at observation time
    up_pre = up_1s[up_1s["sec"] <= observe_s]
    down_pre = down_1s[down_1s["sec"] <= observe_s]

    if up_pre.empty or down_pre.empty:
        return None

    up_mid = up_pre.iloc[-1]["mid_price"]
    down_mid = down_pre.iloc[-1]["mid_price"]

    if up_mid >= 0.50:
        leader_df, other_df = up_1s, down_1s
        leader_label = "Up"
        leader_asset_id = up_pre.iloc[-1]["asset_id"]
    else:
        leader_df, other_df = down_1s, up_1s
        leader_label = "Down"
        leader_asset_id = down_pre.iloc[-1]["asset_id"]

    # --- Leader token features (multi-window + trajectory) ---
    features = extract_multi_window_features(leader_df, open_ms, windows, observe_s)

    # --- Other token summary (just latest snapshot) ---
    other_pre = other_df[other_df["sec"] <= observe_s]
    if not other_pre.empty:
        other_snap = other_pre.iloc[-1]
        features["other_mid"] = other_snap["mid_price"]
        features["other_spread"] = other_snap.get("spread", 0.0)
        latest_w = max(w for w in windows if w <= observe_s)
        features["mid_sum"] = features.get(f"w{latest_w}_mid", up_mid) + features["other_mid"]
        features["spread_diff"] = features.get(f"w{latest_w}_spread", 0.0) - features["other_spread"]
    else:
        features["other_mid"] = 0.0
        features["other_spread"] = 0.0
        features["mid_sum"] = 0.0
        features["spread_diff"] = 0.0

    # --- Microstructure dynamics features (v3) ---
    features.update(_extract_micro_features(leader_df, other_df, observe_s))

    # --- Observation time as a feature ---
    features["observe_time_s"] = observe_s
    features["observe_time_pct"] = observe_s / (MARKET_DURATION_MS / 1000.0)

    # --- BTC context features ---
    if btc_arrays is not None:
        observe_ms = int(observe_s * 1000)
        btc_features = extract_btc_features(
            btc_arrays["ts"], btc_arrays["mids"], btc_arrays["spreads"],
            btc_arrays["log_returns"], btc_arrays["rvols"],
            open_ms, observe_ms,
            btc_arrays["mid_mean"], btc_arrays["mid_std"],
        )
        features.update(btc_features)

    # --- Metadata ---
    features["_leader_label"] = leader_label
    features["_leader_asset_id"] = leader_asset_id

    return features


# ---------------------------------------------------------------------------
# Dataset building
# ---------------------------------------------------------------------------

def build_dataset(
    data_dir: Path,
    resolution_cache_path: Path,
    observe_s: float = 60.0,
    multi_window: bool = True,
    btc_path: Path | None = None,
    max_markets: int | None = None,
    cache_path: Path | None = FEATURE_CACHE_PATH,
    rebuild_cache: bool = False,
) -> pd.DataFrame:
    """Build feature matrix from Telonex data, optionally with BTC features."""

    # Check cache first
    if cache_path and cache_path.exists() and not rebuild_cache and max_markets is None:
        print(f"Loading cached features from {cache_path}")
        dataset = pd.read_parquet(cache_path)
        print(f"  {len(dataset)} samples, {dataset['slug'].nunique()} markets")
        return dataset

    resolutions = load_resolutions(resolution_cache_path)
    print(f"Loaded {len(resolutions)} resolutions")

    # Pre-process BTC data into numpy arrays for fast searchsorted lookups
    btc_arrays = None
    btc_min_ms = btc_max_ms = 0
    if btc_path and btc_path.exists():
        btc_df = load_btc_quotes(btc_path)
        btc_min_ms = int(btc_df["timestamp_ms"].min())
        btc_max_ms = int(btc_df["timestamp_ms"].max())
        print(f"Loaded BTC quotes: {len(btc_df)} rows, "
              f"{pd.to_datetime(btc_min_ms, unit='ms')} to {pd.to_datetime(btc_max_ms, unit='ms')}")

        # Pre-extract numpy arrays for O(log n) lookups
        btc_arrays = {
            "ts": btc_df["timestamp_ms"].values.astype(np.int64),
            "mids": btc_df["mid_price"].values.astype(np.float64),
            "spreads": btc_df["spread"].values.astype(np.float64),
            "log_returns": btc_df["log_return"].values.astype(np.float64),
            "rvols": {},
            "mid_mean": btc_df["mid_price"].mean(),
            "mid_std": btc_df["mid_price"].std(),
        }
        for col in ["rvol_30s", "rvol_60s", "rvol_300s"]:
            if col in btc_df.columns:
                btc_arrays["rvols"][col] = btc_df[col].values.astype(np.float64)

        del btc_df  # free memory

    parquet_files = sorted(data_dir.glob("*-updown-5m-*.parquet"))
    print(f"Found {len(parquet_files)} parquet files")

    if max_markets:
        parquet_files = parquet_files[:max_markets]

    if multi_window:
        windows_to_train = OBSERVATION_WINDOWS
        print(f"Multi-window mode: training at {windows_to_train}")
    else:
        windows_to_train = [observe_s]
        print(f"Single-window mode: observing at {observe_s}s")

    rows = []
    skipped = {"no_resolution": 0, "parse_error": 0, "no_features": 0}

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
        close_ms = open_ms + MARKET_DURATION_MS

        # Skip if no BTC overlap
        if btc_arrays is not None:
            if open_ms < btc_min_ms or open_ms > btc_max_ms:
                continue

        try:
            df = pd.read_parquet(pf)
        except Exception:
            skipped["parse_error"] += 1
            continue

        required = {"exchange_timestamp", "token_label", "asset_id", "mid_price", "spread"}
        if not required.issubset(set(df.columns)):
            skipped["parse_error"] += 1
            continue

        # Resample both tokens to 1s bars
        up_raw = df[df["token_label"] == "Up"].sort_values("exchange_timestamp")
        down_raw = df[df["token_label"] == "Down"].sort_values("exchange_timestamp")

        up_1s = resample_to_1s(up_raw, open_ms, close_ms)
        down_1s = resample_to_1s(down_raw, open_ms, close_ms)

        if up_1s.empty or down_1s.empty:
            skipped["no_features"] += 1
            continue

        winning_token = resolutions[slug]

        for obs_s in windows_to_train:
            features = extract_market_features(
                up_1s, down_1s, open_ms, obs_s,
                btc_arrays=btc_arrays,
                windows=OBSERVATION_WINDOWS,
            )
            if features is None:
                skipped["no_features"] += 1
                continue

            leader_asset_id = features.pop("_leader_asset_id")
            features.pop("_leader_label")

            features["label"] = 1 if leader_asset_id == winning_token else 0
            features["slug"] = slug
            features["open_ts"] = open_ts
            rows.append(features)

        if (i + 1) % 500 == 0:
            print(f"  Processed {i+1}/{len(parquet_files)}, {len(rows)} samples...")

    print(f"\nExtracted {len(rows)} samples from {len(set(r['slug'] for r in rows))} markets")
    print(f"Skipped: {skipped}")

    dataset = pd.DataFrame(rows)
    dataset = dataset.sort_values("open_ts").reset_index(drop=True)

    # Save cache
    if cache_path and max_markets is None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        dataset.to_parquet(cache_path, index=False)
        print(f"Cached features to {cache_path} ({cache_path.stat().st_size / 1e6:.1f} MB)")

    return dataset


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def get_feature_columns(df: pd.DataFrame) -> list[str]:
    exclude = {"label", "slug", "open_ts"}
    return [c for c in df.columns if c not in exclude and not c.startswith("_") and c not in DROP_FEATURES]


def train_and_evaluate(dataset: pd.DataFrame, train_frac: float = 0.7, val_frac: float = 0.15):
    feature_cols = get_feature_columns(dataset)
    n_features = len(feature_cols)
    print(f"\nFeatures ({n_features}):")
    # Group by category
    btc_feats = [f for f in feature_cols if f.startswith("btc_")]
    window_feats = [f for f in feature_cols if f.startswith("w")]
    traj_feats = [f for f in feature_cols if f not in btc_feats and not f.startswith("w") and f not in {"other_mid", "other_spread", "mid_sum", "spread_diff", "observe_time_s", "observe_time_pct", "n_windows_observed"}]
    other_feats = [f for f in feature_cols if f not in btc_feats and f not in window_feats and f not in traj_feats]

    print(f"  Window snapshots ({len(window_feats)}): {window_feats[:5]}...")
    print(f"  Trajectory ({len(traj_feats)}): {traj_feats}")
    print(f"  BTC context ({len(btc_feats)}): {btc_feats}")
    print(f"  Other ({len(other_feats)}): {other_feats}")

    # Time-based split by unique markets
    unique_slugs = dataset["slug"].unique()
    n_markets = len(unique_slugs)
    slug_order = dataset.groupby("slug")["open_ts"].first().sort_values()
    train_end = int(n_markets * train_frac)
    val_end = int(n_markets * (train_frac + val_frac))

    train_slugs = set(slug_order.index[:train_end])
    val_slugs = set(slug_order.index[train_end:val_end])
    test_slugs = set(slug_order.index[val_end:])

    train_mask = dataset["slug"].isin(train_slugs)
    val_mask = dataset["slug"].isin(val_slugs)
    test_mask = dataset["slug"].isin(test_slugs)

    train_df = dataset[train_mask]
    val_df = dataset[val_mask]
    test_df = dataset[test_mask]

    print(f"\nSplit (by market, time-ordered):")
    print(f"  Train: {len(train_df)} samples ({len(train_slugs)} markets)")
    print(f"  Val:   {len(val_df)} samples ({len(val_slugs)} markets)")
    print(f"  Test:  {len(test_df)} samples ({len(test_slugs)} markets)")
    print(f"  Train base rate: {train_df['label'].mean():.3f}")
    print(f"  Val base rate:   {val_df['label'].mean():.3f}")
    print(f"  Test base rate:  {test_df['label'].mean():.3f}")

    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df["label"].values
    X_val = val_df[feature_cols].values.astype(np.float32)
    y_val = val_df["label"].values
    X_test = test_df[feature_cols].values.astype(np.float32)
    y_test = test_df["label"].values

    # Replace NaN/inf
    for X in [X_train, X_val, X_test]:
        X[~np.isfinite(X)] = 0.0

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_cols)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_cols)
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_cols)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": 2,           # v3.2: sweep-optimized (500 trials, deduped features)
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

    print(f"\nTraining XGBoost ({n_features} features)...")
    model = xgb.train(
        params, dtrain,
        num_boost_round=2000,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=50,
        verbose_eval=200,
    )
    print(f"  Best iteration: {model.best_iteration}")

    # --- Evaluate on test set ---
    print(f"\n{'='*70}")
    print("TEST SET RESULTS")
    print(f"{'='*70}")

    y_pred_proba = model.predict(dtest)
    y_pred = (y_pred_proba >= 0.50).astype(int)

    print(f"\n  Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"  AUC-ROC:   {roc_auc_score(y_test, y_pred_proba):.4f}")
    print(f"  Log Loss:  {log_loss(y_test, y_pred_proba):.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Loses", "Wins"]))

    # --- Results by observation time ---
    if "observe_time_s" in test_df.columns and test_df["observe_time_s"].nunique() > 1:
        print(f"\n{'='*70}")
        print("ACCURACY BY OBSERVATION TIME")
        print(f"{'='*70}")
        print(f"{'Time':>6s}  {'N':>6s}  {'Base':>6s}  {'WR':>6s}  {'AUC':>6s}  {'WR@0.65':>8s}  {'N@0.65':>7s}  {'WR@0.70':>8s}  {'N@0.70':>7s}")
        print("-" * 80)

        for t in sorted(test_df["observe_time_s"].unique()):
            t_mask = (test_df["observe_time_s"].values == t)
            if t_mask.sum() < 20:
                continue
            t_proba = y_pred_proba[t_mask]
            t_true = y_test[t_mask]
            t_pred = (t_proba >= 0.50).astype(int)
            t_acc = accuracy_score(t_true, t_pred)
            t_base = t_true.mean()
            try:
                t_auc = roc_auc_score(t_true, t_proba)
            except ValueError:
                t_auc = 0.5

            m65 = t_proba >= 0.65
            wr65 = t_true[m65].mean() if m65.sum() > 5 else float("nan")
            m70 = t_proba >= 0.70
            wr70 = t_true[m70].mean() if m70.sum() > 5 else float("nan")

            print(f"  {t:>4.0f}s  {t_mask.sum():>6d}  {t_base:>5.1%}  {t_acc:>5.1%}  {t_auc:>5.3f}"
                  f"  {wr65:>7.1%}  {m65.sum():>7d}  {wr70:>7.1%}  {m70.sum():>7d}")

    # --- Feature importance (top 25) ---
    print(f"\n{'='*70}")
    print("TOP 25 FEATURE IMPORTANCE (gain)")
    print(f"{'='*70}")

    importance = model.get_score(importance_type="gain")
    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:25]
    max_gain = sorted_imp[0][1] if sorted_imp else 1.0

    for feat, gain in sorted_imp:
        bar = "#" * int(gain / max_gain * 40)
        print(f"  {feat:<32s} {gain:>10.1f}  {bar}")

    # --- Threshold sweep ---
    print(f"\n{'='*70}")
    print("THRESHOLD SWEEP (test set, all observation times)")
    print(f"{'='*70}")
    print(f"{'Threshold':>10s}  {'Trades':>7s}  {'Win%':>6s}  {'EV/trade':>10s}  {'Coverage':>10s}")
    print("-" * 55)

    for threshold in [0.50, 0.55, 0.58, 0.60, 0.65, 0.70, 0.75, 0.80]:
        mask = y_pred_proba >= threshold
        n_trades = mask.sum()
        if n_trades < 10:
            continue
        win_rate = y_test[mask].mean()
        mids = test_df.iloc[np.where(mask)[0]]
        mid_col = [c for c in test_df.columns if c.endswith("_mid") and c.startswith("w")]
        if mid_col:
            last_mid_col = sorted(mid_col)[-1]
            avg_mid = mids[last_mid_col].mean()
        else:
            avg_mid = 0.60
        ev = win_rate - avg_mid
        coverage = n_trades / len(y_test)
        print(f"  {threshold:>8.2f}  {n_trades:>7d}  {win_rate:>5.1%}  {ev:>+9.4f}    {coverage:>8.1%}")

    # --- Calibration ---
    print(f"\n{'='*70}")
    print("CALIBRATION")
    print(f"{'='*70}")
    print(f"{'Bucket':>12s}  {'N':>6s}  {'Predicted':>10s}  {'Actual':>10s}  {'Gap':>8s}")
    print("-" * 55)

    for lo, hi in [(0.0, 0.35), (0.35, 0.45), (0.45, 0.55), (0.55, 0.65), (0.65, 0.75), (0.75, 1.0)]:
        mask = (y_pred_proba >= lo) & (y_pred_proba < hi)
        n = mask.sum()
        if n < 5:
            continue
        print(f"  {lo:.2f}-{hi:.2f}  {n:>6d}  {y_pred_proba[mask].mean():>9.3f}"
              f"  {y_test[mask].mean():>9.3f}  {y_test[mask].mean() - y_pred_proba[mask].mean():>+7.3f}")

    # --- Volatility regime filter ---
    vol_threshold = _vol_filter_analysis(train_df, test_df, y_pred_proba, y_test, feature_cols)

    return model, test_df, y_pred_proba, feature_cols, vol_threshold


def compute_vol_threshold(train_df: pd.DataFrame, percentile: float = 33.0) -> float | None:
    """Compute the btc_vol_5m threshold from training data.

    Returns the value below which markets should be skipped (low-vol regime
    where model has minimal edge). Returns None if btc_vol_5m is not available.

    This threshold is meant to be saved and used by the live trading system.
    """
    if "btc_vol_5m" not in train_df.columns:
        return None
    valid = train_df["btc_vol_5m"].dropna()
    valid = valid[valid > 0]
    if len(valid) < 100:
        return None
    return float(valid.quantile(percentile / 100.0))


def _vol_filter_analysis(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    y_pred_proba: np.ndarray,
    y_test: np.ndarray,
    feature_cols: list[str],
) -> float | None:
    """Analyze the impact of filtering out low-volatility markets."""

    print(f"\n{'='*70}")
    print("VOLATILITY REGIME FILTER")
    print(f"{'='*70}")

    if "btc_vol_5m" not in test_df.columns:
        print("  btc_vol_5m not available — skipping vol filter analysis")
        return None

    # Compute threshold from training data (avoid data leakage)
    vol_threshold = compute_vol_threshold(train_df, percentile=33.0)
    if vol_threshold is None:
        print("  Insufficient vol data in training set")
        return None

    train_vol = train_df["btc_vol_5m"].dropna()
    train_vol = train_vol[train_vol > 0]
    print(f"  Training btc_vol_5m: median={train_vol.median():.6f}, "
          f"p33={vol_threshold:.6f}, p67={train_vol.quantile(0.67):.6f}")

    test_vol = test_df["btc_vol_5m"].values

    # Show results at different percentile cutoffs
    print(f"\n  Impact of filtering low-vol markets (test set, 120s window, conf >= 0.70):")
    print(f"  {'Filter':>14s}  {'Trades':>7s}  {'WR':>6s}  {'Base':>6s}  {'Lift':>6s}  {'Skipped':>8s}")
    print("  " + "-" * 55)

    # Baseline (no filter, 120s, >= 0.70)
    w120 = test_df["observe_time_s"].values == 120
    conf70 = y_pred_proba >= 0.70
    base_mask = w120 & conf70
    if base_mask.sum() > 0:
        base_wr = y_test[base_mask].mean()
        base_n = base_mask.sum()
        base_rate = y_test[w120].mean()
        print(f"  {'No filter':>14s}  {base_n:>7d}  {base_wr:>5.1%}  {base_rate:>5.1%}  "
              f"{base_wr - base_rate:>+5.1%}  {'0':>8s}")

    for pct in [20, 25, 33, 40, 50]:
        cutoff = float(train_vol.quantile(pct / 100.0))
        vol_pass = test_vol > cutoff
        filtered_mask = w120 & conf70 & vol_pass
        skipped_mask = w120 & conf70 & ~vol_pass

        if filtered_mask.sum() < 10:
            continue

        f_wr = y_test[filtered_mask].mean()
        f_n = filtered_mask.sum()
        f_base = y_test[w120 & vol_pass].mean() if (w120 & vol_pass).sum() > 10 else float("nan")
        skip_n = skipped_mask.sum()
        skip_wr = y_test[skipped_mask].mean() if skip_n > 5 else float("nan")

        print(f"  {'p' + str(pct) + f' ({cutoff:.5f})':>14s}  {f_n:>7d}  {f_wr:>5.1%}  {f_base:>5.1%}  "
              f"{f_wr - f_base:>+5.1%}  {skip_n:>4d} ({skip_wr:.0%})")

    # Also show the all-windows version
    print(f"\n  Impact across ALL observation times (conf >= 0.70):")
    print(f"  {'Filter':>14s}  {'Trades':>7s}  {'WR':>6s}  {'Skipped':>8s}")
    print("  " + "-" * 42)

    all_conf70 = y_pred_proba >= 0.70
    if all_conf70.sum() > 0:
        print(f"  {'No filter':>14s}  {all_conf70.sum():>7d}  {y_test[all_conf70].mean():>5.1%}  {'0':>8s}")

    for pct in [25, 33, 40]:
        cutoff = float(train_vol.quantile(pct / 100.0))
        vol_pass = test_vol > cutoff
        filtered_mask = all_conf70 & vol_pass
        skip_n = (all_conf70 & ~vol_pass).sum()

        if filtered_mask.sum() < 10:
            continue
        f_wr = y_test[filtered_mask].mean()
        print(f"  {'p' + str(pct):>14s}  {filtered_mask.sum():>7d}  {f_wr:>5.1%}  {skip_n:>8d}")

    print(f"\n  Recommended vol filter: btc_vol_5m > {vol_threshold:.6f} (p33 of training set)")
    print(f"  Live usage: compute 5-min BTC log-return std * sqrt(n) from Binance feed;")
    print(f"  skip market if value is below threshold.")

    return vol_threshold


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="data/telonex_book_snapshots")
    parser.add_argument("--resolution-cache", default="data/resolution_cache.json")
    parser.add_argument("--btc-path", default="data/btc_quotes/btcusdt_quotes.parquet")
    parser.add_argument("--observe-at", type=float, default=60.0,
                        help="Observation time for single-window mode (default: 60s)")
    parser.add_argument("--multi-window", action="store_true", default=True,
                        help="Train on multiple observation windows (default: True)")
    parser.add_argument("--single-window", action="store_true",
                        help="Train on single observation time only")
    parser.add_argument("--max-markets", type=int, default=None)
    parser.add_argument("--no-btc", action="store_true",
                        help="Skip BTC features (use all markets, not just BTC overlap)")
    parser.add_argument("--rebuild-cache", action="store_true",
                        help="Force rebuild feature cache (ignore existing)")
    parser.add_argument("--cache-path", default=None,
                        help="Feature cache path (default: data/xgb_features_v3.parquet)")
    args = parser.parse_args()

    if args.single_window:
        args.multi_window = False

    btc_path = None if args.no_btc else Path(args.btc_path)
    cache_path = Path(args.cache_path) if args.cache_path else FEATURE_CACHE_PATH

    print("XGBoost Flexible Confidence Filter v3")
    print("=" * 50)
    print(f"  Data:        {args.data_dir}")
    print(f"  BTC quotes:  {btc_path or 'disabled'}")
    print(f"  Mode:        {'multi-window' if args.multi_window else f'single @ {args.observe_at}s'}")
    print(f"  Resampling:  1s bars")
    print()

    dataset = build_dataset(
        data_dir=Path(args.data_dir),
        resolution_cache_path=Path(args.resolution_cache),
        observe_s=args.observe_at,
        multi_window=args.multi_window,
        btc_path=btc_path,
        max_markets=args.max_markets,
        cache_path=cache_path,
        rebuild_cache=args.rebuild_cache,
    )

    print(f"\nDataset: {dataset.shape[0]} samples, {dataset.shape[1]} columns")
    print(f"Unique markets: {dataset['slug'].nunique()}")
    print(f"Label distribution: {dataset['label'].value_counts().to_dict()}")
    print(f"Base rate: {dataset['label'].mean():.3f}")

    if "observe_time_s" in dataset.columns:
        print(f"Observation times: {sorted(dataset['observe_time_s'].unique())}")

    model, test_df, predictions, feature_cols, vol_threshold = train_and_evaluate(dataset)

    print(f"\nDone. Model uses {len(feature_cols)} features.")
    if vol_threshold is not None:
        print(f"Vol filter threshold: btc_vol_5m > {vol_threshold:.6f}")


if __name__ == "__main__":
    main()
