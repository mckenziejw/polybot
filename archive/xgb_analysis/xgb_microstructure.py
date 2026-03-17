#!/usr/bin/env python3
"""
XGBoost microstructure feature experiment.

Extends xgboost_flexible.py with orderbook DYNAMICS features:
  1. Rate of change: mid, spread, depth velocity/acceleration over rolling windows
  2. Order flow imbalance dynamics: imbalance trend, oscillation, sudden shifts
  3. Spread regime: avg spread, spread vol, time at min tick
  4. Depth profile shape: L1 vs L3 vs L5 concentration, depth profile changes
  5. Price impact estimates: depth within X cents of mid
  6. Bid-ask asymmetry dynamics: which side is getting depleted faster
  7. Cross-token features: Up/Down correlation, divergence, relative spread

Usage:
    python xgb_microstructure.py                    # full run (uses cache)
    python xgb_microstructure.py --rebuild-cache    # rebuild features
    python xgb_microstructure.py --max-markets 200  # quick test
    python xgb_microstructure.py --ablation         # run feature group ablation study

Compares baseline (v2 features only) vs enhanced (v2 + microstructure) on same data.
"""

import argparse
import json
import sys
import time
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
OBSERVATION_WINDOWS = [10, 20, 30, 45, 60, 90, 120]
BTC_LOOKBACKS = [60, 300, 900, 3600]
MICRO_CACHE_PATH = Path("data/xgb_features_micro.parquet")
N_LEVELS = 10  # full depth available in data

# ---------------------------------------------------------------------------
# Data helpers (reused from xgboost_flexible)
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
    market_df = token_df[
        (token_df["exchange_timestamp"] >= open_ms) &
        (token_df["exchange_timestamp"] < close_ms)
    ].copy()
    if market_df.empty:
        return market_df
    market_df["sec"] = ((market_df["exchange_timestamp"] - open_ms) / 1000).astype(int)
    resampled = market_df.groupby("sec").last().reset_index()
    resampled["exchange_timestamp"] = open_ms + resampled["sec"] * 1000
    return resampled


# ---------------------------------------------------------------------------
# BTC features (copied from xgboost_flexible for self-contained script)
# ---------------------------------------------------------------------------

def extract_btc_features(btc_arrays, market_open_ms, observe_ms):
    if btc_arrays is None:
        return _empty_btc_features()
    btc_ts = btc_arrays["ts"]
    btc_mids = btc_arrays["mids"]
    btc_spreads = btc_arrays["spreads"]
    btc_log_returns = btc_arrays["log_returns"]
    btc_rvols = btc_arrays["rvols"]
    btc_mid_mean = btc_arrays["mid_mean"]
    btc_mid_std = btc_arrays["mid_std"]

    features = {}
    obs_ts = market_open_ms + observe_ms
    obs_idx = np.searchsorted(btc_ts, obs_ts, side="right") - 1
    if obs_idx < 0:
        return _empty_btc_features()

    btc_mid = btc_mids[obs_idx]
    features["btc_mid_zscore"] = (btc_mid - btc_mid_mean) / btc_mid_std if btc_mid_std > 0 else 0.0
    features["btc_spread"] = btc_spreads[obs_idx]
    features["btc_log_return"] = btc_log_returns[obs_idx]

    for window in ["30s", "60s", "300s"]:
        key = f"rvol_{window}"
        features[f"btc_rvol_{window}"] = btc_rvols.get(key, np.zeros(1))[min(obs_idx, len(btc_rvols.get(key, np.zeros(1)))-1)] if key in btc_rvols else 0.0

    for lb in BTC_LOOKBACKS:
        lb_ms = market_open_ms - lb * 1000
        lb_idx = np.searchsorted(btc_ts, lb_ms, side="right") - 1
        if lb_idx >= 0 and btc_mids[lb_idx] > 0:
            features[f"btc_ret_{lb}s"] = np.log(btc_mid / btc_mids[lb_idx])
        else:
            features[f"btc_ret_{lb}s"] = 0.0

    open_idx = np.searchsorted(btc_ts, market_open_ms, side="right") - 1
    if open_idx >= 0 and btc_mids[open_idx] > 0:
        features["btc_ret_since_open"] = np.log(btc_mid / btc_mids[open_idx])
    else:
        features["btc_ret_since_open"] = 0.0

    window_start_idx = np.searchsorted(btc_ts, market_open_ms, side="left")
    window_end_idx = np.searchsorted(btc_ts, obs_ts, side="right")
    if window_end_idx - window_start_idx >= 3:
        window_returns = btc_log_returns[window_start_idx:window_end_idx]
        valid = window_returns[np.isfinite(window_returns)]
        features["btc_window_vol"] = np.std(valid) if len(valid) > 1 else 0.0
        window_mids = btc_mids[window_start_idx:window_end_idx]
        features["btc_window_range"] = (window_mids.max() - window_mids.min()) / btc_mid if btc_mid > 0 else 0.0
    else:
        features["btc_window_vol"] = 0.0
        features["btc_window_range"] = 0.0

    return features


def _empty_btc_features():
    features = {
        "btc_mid_zscore": 0.0, "btc_spread": 0.0, "btc_log_return": 0.0,
        "btc_rvol_30s": 0.0, "btc_rvol_60s": 0.0, "btc_rvol_300s": 0.0,
        "btc_ret_since_open": 0.0, "btc_window_vol": 0.0, "btc_window_range": 0.0,
    }
    for lb in BTC_LOOKBACKS:
        features[f"btc_ret_{lb}s"] = 0.0
    return features


# ---------------------------------------------------------------------------
# BASELINE features (from xgboost_flexible — reproduced here)
# ---------------------------------------------------------------------------

def extract_baseline_snapshot(snap: pd.Series, prefix: str = "") -> dict:
    f = {}
    p = prefix
    f[f"{p}mid"] = snap["mid_price"]
    f[f"{p}spread"] = snap.get("spread", 0.0)
    f[f"{p}imbalance"] = snap.get("book_imbalance", 0.0)
    bid_sizes = [snap.get(f"bid_size_{i}", 0.0) or 0.0 for i in range(1, 6)]
    ask_sizes = [snap.get(f"ask_size_{i}", 0.0) or 0.0 for i in range(1, 6)]
    f[f"{p}bid_depth_3"] = sum(bid_sizes[:3])
    f[f"{p}ask_depth_3"] = sum(ask_sizes[:3])
    total = f[f"{p}bid_depth_3"] + f[f"{p}ask_depth_3"]
    f[f"{p}depth_imb_3"] = (f[f"{p}bid_depth_3"] - f[f"{p}ask_depth_3"]) / total if total > 0 else 0.0
    return f


def extract_baseline_trajectory(mids, timestamps):
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
    f["drift_from_50"] = mids[-1] - 0.50
    f["abs_drift"] = abs(f["drift_from_50"])
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
    half = n // 2
    if half > 1 and duration > 0:
        t_mid = t_s[half]
        speed1 = (mids[half] - mids[0]) / t_mid if t_mid > 0 else 0.0
        speed2 = (mids[-1] - mids[half]) / (duration - t_mid) if (duration - t_mid) > 0 else 0.0
        f["drift_accel"] = speed2 - speed1
    else:
        f["drift_accel"] = 0.0
    changes = np.diff(mids)
    f["realized_vol"] = np.std(changes) if len(changes) > 1 else 0.0
    f["range"] = mids.max() - mids.min()
    abs_drift = abs(mids[-1] - mids[0])
    f["choppiness"] = f["range"] / abs_drift if abs_drift > 0.001 else 10.0
    f["direction_changes"] = int(np.sum(np.diff(np.sign(mids - mids[0])) != 0))
    if len(changes) > 0:
        if mids[-1] > mids[0]:
            f["consistency"] = float(np.mean(changes > 0))
        else:
            f["consistency"] = float(np.mean(changes < 0))
    else:
        f["consistency"] = 0.5
    f["max_drift"] = float(np.max(np.abs(mids - 0.50)))
    return f


def extract_baseline_multi_window(token_df, open_ms, windows, observe_s):
    features = {}
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
        snap_f = extract_baseline_snapshot(snap, prefix=f"w{w}_")
        features.update(snap_f)
        if prev_mid is not None:
            features[f"w{w}_mid_delta"] = snap_f[f"w{w}_mid"] - prev_mid
            features[f"w{w}_spread_delta"] = snap_f[f"w{w}_spread"] - prev_spread
        prev_mid = snap_f[f"w{w}_mid"]
        prev_spread = snap_f[f"w{w}_spread"]
    pre_all = token_df[token_df["sec"] <= observe_s]
    if len(pre_all) >= 3:
        traj = extract_baseline_trajectory(pre_all["mid_price"].values, pre_all["exchange_timestamp"].values)
        features.update(traj)
    else:
        features.update({
            "drift_from_50": 0.0, "abs_drift": 0.0, "drift_speed": 0.0,
            "drift_accel": 0.0, "r_squared": 0.0, "realized_vol": 0.0,
            "range": 0.0, "choppiness": 10.0, "direction_changes": 0,
            "consistency": 0.5, "max_drift": 0.0,
        })
    features["n_windows_observed"] = len([w for w in active_windows if f"w{w}_mid" in features])
    return features


# =========================================================================
# NEW: MICROSTRUCTURE FEATURES
# =========================================================================

def _safe_div(num, den, default=0.0):
    return num / den if abs(den) > 1e-12 else default


def extract_micro_rate_of_change(token_1s: pd.DataFrame, observe_s: float) -> dict:
    """Rate-of-change features for mid, spread, depth over rolling windows.

    Captures VELOCITY and ACCELERATION of orderbook state changes.
    """
    f = {}
    pre = token_1s[token_1s["sec"] <= observe_s].copy()
    n = len(pre)

    if n < 5:
        return {
            "mid_roc_5s": 0.0, "mid_roc_10s": 0.0, "mid_roc_20s": 0.0,
            "spread_roc_5s": 0.0, "spread_roc_10s": 0.0,
            "depth_roc_5s": 0.0, "depth_roc_10s": 0.0,
            "mid_accel_5s": 0.0, "spread_accel_5s": 0.0,
            "mid_roc_std_10s": 0.0, "mid_roc_skew_10s": 0.0,
        }

    mids = pre["mid_price"].values
    spreads = pre["spread"].values

    # Total depth at top 3 levels
    bid_d3 = sum(pre.get(f"bid_size_{i}", pd.Series(0.0)).fillna(0).values for i in range(1, 4))
    ask_d3 = sum(pre.get(f"ask_size_{i}", pd.Series(0.0)).fillna(0).values for i in range(1, 4))
    total_depth = bid_d3 + ask_d3

    # Rate of change at different lookbacks
    for lb, label in [(5, "5s"), (10, "10s"), (20, "20s")]:
        if n > lb:
            f[f"mid_roc_{label}"] = mids[-1] - mids[-1-lb]
        else:
            f[f"mid_roc_{label}"] = mids[-1] - mids[0]

    for lb, label in [(5, "5s"), (10, "10s")]:
        if n > lb:
            f[f"spread_roc_{label}"] = spreads[-1] - spreads[-1-lb]
            f[f"depth_roc_{label}"] = _safe_div(
                total_depth[-1] - total_depth[-1-lb],
                total_depth[-1-lb] + 1.0
            )
        else:
            f[f"spread_roc_{label}"] = 0.0
            f[f"depth_roc_{label}"] = 0.0

    # Acceleration: second derivative of mid price over last 5s
    if n >= 11:
        roc_now = mids[-1] - mids[-1-5]
        roc_prev = mids[-1-5] - mids[-1-10]
        f["mid_accel_5s"] = roc_now - roc_prev
        spread_roc_now = spreads[-1] - spreads[-1-5]
        spread_roc_prev = spreads[-1-5] - spreads[-1-10]
        f["spread_accel_5s"] = spread_roc_now - spread_roc_prev
    else:
        f["mid_accel_5s"] = 0.0
        f["spread_accel_5s"] = 0.0

    # Distribution of 1s mid returns over last 10s
    if n >= 11:
        recent_changes = np.diff(mids[-11:])
        f["mid_roc_std_10s"] = np.std(recent_changes)
        mean_c = np.mean(recent_changes)
        std_c = np.std(recent_changes)
        f["mid_roc_skew_10s"] = _safe_div(
            np.mean((recent_changes - mean_c) ** 3), std_c ** 3
        ) if std_c > 1e-12 else 0.0
    else:
        f["mid_roc_std_10s"] = 0.0
        f["mid_roc_skew_10s"] = 0.0

    return f


def extract_micro_imbalance_dynamics(token_1s: pd.DataFrame, observe_s: float) -> dict:
    """How order flow imbalance EVOLVES over time.

    Not just current imbalance — trend, volatility, regime shifts.
    """
    f = {}
    pre = token_1s[token_1s["sec"] <= observe_s].copy()
    n = len(pre)

    if n < 5:
        return {
            "imb_mean_10s": 0.0, "imb_std_10s": 0.0, "imb_trend_10s": 0.0,
            "imb_mean_30s": 0.0, "imb_std_30s": 0.0, "imb_trend_30s": 0.0,
            "imb_regime_shift": 0.0, "imb_momentum": 0.0,
            "imb_max_10s": 0.0, "imb_min_10s": 0.0,
        }

    imb = pre["book_imbalance"].fillna(0.0).values

    # Imbalance statistics over windows
    for ws, label in [(10, "10s"), (30, "30s")]:
        window = imb[-min(ws, n):]
        f[f"imb_mean_{label}"] = np.mean(window)
        f[f"imb_std_{label}"] = np.std(window)
        # Trend: linear regression slope of imbalance
        if len(window) >= 3:
            t = np.arange(len(window), dtype=float)
            t_mean = t.mean()
            num = np.sum((t - t_mean) * (window - window.mean()))
            den = np.sum((t - t_mean) ** 2)
            f[f"imb_trend_{label}"] = _safe_div(num, den)
        else:
            f[f"imb_trend_{label}"] = 0.0

    # Regime shift: compare first half vs second half of observation
    half = n // 2
    if half >= 3:
        first_half_mean = np.mean(imb[:half])
        second_half_mean = np.mean(imb[half:])
        f["imb_regime_shift"] = second_half_mean - first_half_mean
    else:
        f["imb_regime_shift"] = 0.0

    # Momentum: is imbalance accelerating in one direction?
    if n >= 10:
        recent_5 = np.mean(imb[-5:])
        prev_5 = np.mean(imb[-10:-5])
        f["imb_momentum"] = recent_5 - prev_5
    else:
        f["imb_momentum"] = 0.0

    # Extremes in recent window
    recent = imb[-min(10, n):]
    f["imb_max_10s"] = np.max(recent)
    f["imb_min_10s"] = np.min(recent)

    return f


def extract_micro_spread_regime(token_1s: pd.DataFrame, observe_s: float) -> dict:
    """Spread regime features.

    Is the spread tight or wide? Stable or volatile? How much time at minimum tick?
    """
    f = {}
    pre = token_1s[token_1s["sec"] <= observe_s].copy()
    n = len(pre)

    if n < 3:
        return {
            "spread_mean": 0.0, "spread_std": 0.0, "spread_cv": 0.0,
            "spread_min": 0.0, "spread_max": 0.0,
            "spread_at_min_tick_pct": 0.0, "spread_quantile_75": 0.0,
            "spread_trend": 0.0, "spread_mean_ratio": 0.0,
        }

    spreads = pre["spread"].fillna(0.0).values

    f["spread_mean"] = np.mean(spreads)
    f["spread_std"] = np.std(spreads)
    f["spread_cv"] = _safe_div(f["spread_std"], f["spread_mean"])
    f["spread_min"] = np.min(spreads)
    f["spread_max"] = np.max(spreads)
    f["spread_quantile_75"] = np.percentile(spreads, 75)

    # Fraction of time at minimum tick (0.01 for Polymarket)
    min_tick = 0.01
    f["spread_at_min_tick_pct"] = np.mean(spreads <= min_tick + 1e-9)

    # Spread trend
    t = np.arange(n, dtype=float)
    t_mean = t.mean()
    num = np.sum((t - t_mean) * (spreads - spreads.mean()))
    den = np.sum((t - t_mean) ** 2)
    f["spread_trend"] = _safe_div(num, den)

    # Current spread relative to average (regime indicator)
    f["spread_mean_ratio"] = _safe_div(spreads[-1], f["spread_mean"]) if f["spread_mean"] > 0 else 1.0

    return f


def extract_micro_depth_profile(token_1s: pd.DataFrame, observe_s: float) -> dict:
    """Depth profile shape and dynamics.

    How is liquidity distributed across levels? Concentrated at top or spread out?
    """
    f = {}
    pre = token_1s[token_1s["sec"] <= observe_s]

    if pre.empty:
        return {
            "bid_concentration_l1": 0.0, "ask_concentration_l1": 0.0,
            "bid_concentration_l3": 0.0, "ask_concentration_l3": 0.0,
            "depth_total_10": 0.0, "bid_depth_10": 0.0, "ask_depth_10": 0.0,
            "depth_imb_10": 0.0,
            "bid_depth_change_pct": 0.0, "ask_depth_change_pct": 0.0,
            "depth_profile_asym": 0.0,
            "bid_wall_ratio": 0.0, "ask_wall_ratio": 0.0,
        }

    snap = pre.iloc[-1]

    # Compute depth at each level
    bid_sizes = [float(snap.get(f"bid_size_{i}", 0.0) or 0.0) for i in range(1, 11)]
    ask_sizes = [float(snap.get(f"ask_size_{i}", 0.0) or 0.0) for i in range(1, 11)]

    bid_total = sum(bid_sizes)
    ask_total = sum(ask_sizes)
    total = bid_total + ask_total

    # Concentration: what fraction of depth is at level 1?
    f["bid_concentration_l1"] = _safe_div(bid_sizes[0], bid_total)
    f["ask_concentration_l1"] = _safe_div(ask_sizes[0], ask_total)

    # Concentration at top 3: how much of total depth is in top 3 levels?
    f["bid_concentration_l3"] = _safe_div(sum(bid_sizes[:3]), bid_total)
    f["ask_concentration_l3"] = _safe_div(sum(ask_sizes[:3]), ask_total)

    # Full depth (all 10 levels)
    f["depth_total_10"] = total
    f["bid_depth_10"] = bid_total
    f["ask_depth_10"] = ask_total
    f["depth_imb_10"] = _safe_div(bid_total - ask_total, total)

    # Asymmetry: bid concentration vs ask concentration
    f["depth_profile_asym"] = f["bid_concentration_l1"] - f["ask_concentration_l1"]

    # Wall detection: max single level size relative to average
    if bid_total > 0:
        bid_avg = bid_total / 10.0
        f["bid_wall_ratio"] = max(bid_sizes) / bid_avg if bid_avg > 0 else 0.0
    else:
        f["bid_wall_ratio"] = 0.0

    if ask_total > 0:
        ask_avg = ask_total / 10.0
        f["ask_wall_ratio"] = max(ask_sizes) / ask_avg if ask_avg > 0 else 0.0
    else:
        f["ask_wall_ratio"] = 0.0

    # Depth change: compare current depth to earlier depth
    n = len(pre)
    if n >= 10:
        earlier = pre.iloc[-10]
        earlier_bid = sum(float(earlier.get(f"bid_size_{i}", 0.0) or 0.0) for i in range(1, 11))
        earlier_ask = sum(float(earlier.get(f"ask_size_{i}", 0.0) or 0.0) for i in range(1, 11))
        f["bid_depth_change_pct"] = _safe_div(bid_total - earlier_bid, earlier_bid + 1.0)
        f["ask_depth_change_pct"] = _safe_div(ask_total - earlier_ask, earlier_ask + 1.0)
    else:
        f["bid_depth_change_pct"] = 0.0
        f["ask_depth_change_pct"] = 0.0

    return f


def extract_micro_price_impact(token_1s: pd.DataFrame, observe_s: float) -> dict:
    """Price impact estimates: depth within X cents of mid.

    How much size would need to trade to move price by 1c, 2c, 3c?
    """
    f = {}
    pre = token_1s[token_1s["sec"] <= observe_s]

    if pre.empty:
        return {
            "bid_depth_1c": 0.0, "ask_depth_1c": 0.0,
            "bid_depth_2c": 0.0, "ask_depth_2c": 0.0,
            "bid_depth_3c": 0.0, "ask_depth_3c": 0.0,
            "impact_asym_1c": 0.0, "impact_asym_2c": 0.0,
        }

    snap = pre.iloc[-1]
    mid = snap["mid_price"]

    bid_prices = [float(snap.get(f"bid_price_{i}", 0.0) or 0.0) for i in range(1, 11)]
    bid_sizes = [float(snap.get(f"bid_size_{i}", 0.0) or 0.0) for i in range(1, 11)]
    ask_prices = [float(snap.get(f"ask_price_{i}", 0.0) or 0.0) for i in range(1, 11)]
    ask_sizes = [float(snap.get(f"ask_size_{i}", 0.0) or 0.0) for i in range(1, 11)]

    for threshold_cents, label in [(1, "1c"), (2, "2c"), (3, "3c")]:
        threshold = threshold_cents * 0.01
        bid_depth = sum(s for p, s in zip(bid_prices, bid_sizes) if p > 0 and mid - p <= threshold)
        ask_depth = sum(s for p, s in zip(ask_prices, ask_sizes) if p > 0 and p - mid <= threshold)
        f[f"bid_depth_{label}"] = bid_depth
        f[f"ask_depth_{label}"] = ask_depth

    # Asymmetry in depth close to mid
    for label in ["1c", "2c"]:
        total = f[f"bid_depth_{label}"] + f[f"ask_depth_{label}"]
        f[f"impact_asym_{label}"] = _safe_div(
            f[f"bid_depth_{label}"] - f[f"ask_depth_{label}"], total
        )

    return f


def extract_micro_asymmetry_dynamics(token_1s: pd.DataFrame, observe_s: float) -> dict:
    """Bid-ask asymmetry dynamics over time.

    Is one side consistently being depleted? Are refills happening faster on one side?
    """
    f = {}
    pre = token_1s[token_1s["sec"] <= observe_s].copy()
    n = len(pre)

    if n < 5:
        return {
            "bid_size1_mean": 0.0, "ask_size1_mean": 0.0,
            "bid_refill_rate": 0.0, "ask_refill_rate": 0.0,
            "side_pressure": 0.0, "depth_ratio_trend": 0.0,
            "bid_size1_cv": 0.0, "ask_size1_cv": 0.0,
        }

    bid1 = pre["bid_size_1"].fillna(0.0).values
    ask1 = pre["ask_size_1"].fillna(0.0).values

    f["bid_size1_mean"] = np.mean(bid1)
    f["ask_size1_mean"] = np.mean(ask1)

    # Coefficient of variation of top-of-book size (stability)
    f["bid_size1_cv"] = _safe_div(np.std(bid1), np.mean(bid1)) if np.mean(bid1) > 0 else 0.0
    f["ask_size1_cv"] = _safe_div(np.std(ask1), np.mean(ask1)) if np.mean(ask1) > 0 else 0.0

    # Refill rate: how quickly does size return after depletion?
    # Proxy: count sign changes in the difference of size (size dropped then recovered)
    bid1_diff = np.diff(bid1)
    ask1_diff = np.diff(ask1)

    # Refill = large drop followed by recovery within next bar
    bid_drops = bid1_diff < -np.percentile(np.abs(bid1_diff), 75) if len(bid1_diff) > 3 else np.array([False])
    ask_drops = ask1_diff < -np.percentile(np.abs(ask1_diff), 75) if len(ask1_diff) > 3 else np.array([False])

    f["bid_refill_rate"] = float(np.mean(bid_drops[:-1] & (bid1_diff[1:] > 0))) if len(bid_drops) > 1 else 0.0
    f["ask_refill_rate"] = float(np.mean(ask_drops[:-1] & (ask1_diff[1:] > 0))) if len(ask_drops) > 1 else 0.0

    # Side pressure: which side has more activity (more size changes)?
    bid_activity = np.std(bid1_diff) if len(bid1_diff) > 1 else 0.0
    ask_activity = np.std(ask1_diff) if len(ask1_diff) > 1 else 0.0
    total_activity = bid_activity + ask_activity
    f["side_pressure"] = _safe_div(bid_activity - ask_activity, total_activity)

    # Depth ratio trend: is bid/ask ratio trending?
    depth_ratio = np.where(ask1 > 0, bid1 / ask1, 1.0)
    if len(depth_ratio) >= 5:
        t = np.arange(len(depth_ratio), dtype=float)
        t_mean = t.mean()
        num = np.sum((t - t_mean) * (depth_ratio - depth_ratio.mean()))
        den = np.sum((t - t_mean) ** 2)
        f["depth_ratio_trend"] = _safe_div(num, den)
    else:
        f["depth_ratio_trend"] = 0.0

    return f


def extract_micro_cross_token(
    up_1s: pd.DataFrame, down_1s: pd.DataFrame,
    observe_s: float, leader_is_up: bool,
) -> dict:
    """Cross-token microstructure: how Up and Down books relate.

    Do they move in sync? Does one lead the other? Divergence signals?
    """
    f = {}
    up_pre = up_1s[up_1s["sec"] <= observe_s]
    down_pre = down_1s[down_1s["sec"] <= observe_s]

    if len(up_pre) < 5 or len(down_pre) < 5:
        return {
            "cross_mid_corr": 0.0, "cross_spread_corr": 0.0,
            "cross_mid_sum_dev": 0.0, "cross_spread_ratio": 0.0,
            "cross_depth_ratio": 0.0, "cross_imb_corr": 0.0,
            "cross_mid_sum_trend": 0.0, "cross_mid_sum_vol": 0.0,
            "other_mid_roc_5s": 0.0, "other_imb_mean_10s": 0.0,
        }

    # Align on sec (inner join on common seconds)
    up_indexed = up_pre.set_index("sec")
    down_indexed = down_pre.set_index("sec")
    common_secs = up_indexed.index.intersection(down_indexed.index)

    if len(common_secs) < 5:
        return {
            "cross_mid_corr": 0.0, "cross_spread_corr": 0.0,
            "cross_mid_sum_dev": 0.0, "cross_spread_ratio": 0.0,
            "cross_depth_ratio": 0.0, "cross_imb_corr": 0.0,
            "cross_mid_sum_trend": 0.0, "cross_mid_sum_vol": 0.0,
            "other_mid_roc_5s": 0.0, "other_imb_mean_10s": 0.0,
        }

    up_aligned = up_indexed.loc[common_secs]
    down_aligned = down_indexed.loc[common_secs]

    up_mids = up_aligned["mid_price"].values
    down_mids = down_aligned["mid_price"].values

    # Mid-price sum (should be ~1.0 for binary market): deviation indicates pricing inefficiency
    mid_sum = up_mids + down_mids
    f["cross_mid_sum_dev"] = np.mean(np.abs(mid_sum - 1.0))
    f["cross_mid_sum_vol"] = np.std(mid_sum)

    # Trend of mid sum (is the market becoming more/less efficient?)
    t = np.arange(len(mid_sum), dtype=float)
    t_mean = t.mean()
    num = np.sum((t - t_mean) * (mid_sum - mid_sum.mean()))
    den = np.sum((t - t_mean) ** 2)
    f["cross_mid_sum_trend"] = _safe_div(num, den)

    # Correlation between Up and Down mid changes
    up_changes = np.diff(up_mids)
    down_changes = np.diff(down_mids)
    if len(up_changes) >= 3:
        corr_matrix = np.corrcoef(up_changes, down_changes)
        f["cross_mid_corr"] = corr_matrix[0, 1] if np.isfinite(corr_matrix[0, 1]) else 0.0
    else:
        f["cross_mid_corr"] = 0.0

    # Spread correlation
    up_spreads = up_aligned["spread"].fillna(0).values
    down_spreads = down_aligned["spread"].fillna(0).values
    if np.std(up_spreads) > 0 and np.std(down_spreads) > 0:
        corr_matrix = np.corrcoef(up_spreads, down_spreads)
        f["cross_spread_corr"] = corr_matrix[0, 1] if np.isfinite(corr_matrix[0, 1]) else 0.0
    else:
        f["cross_spread_corr"] = 0.0

    # Cross-token spread ratio
    f["cross_spread_ratio"] = _safe_div(up_spreads[-1], down_spreads[-1] + 0.001)

    # Cross-token depth ratio (leader vs other)
    leader_aligned = up_aligned if leader_is_up else down_aligned
    other_aligned = down_aligned if leader_is_up else up_aligned

    leader_bid3 = sum(leader_aligned[f"bid_size_{i}"].fillna(0).values[-1] for i in range(1, 4))
    leader_ask3 = sum(leader_aligned[f"ask_size_{i}"].fillna(0).values[-1] for i in range(1, 4))
    other_bid3 = sum(other_aligned[f"bid_size_{i}"].fillna(0).values[-1] for i in range(1, 4))
    other_ask3 = sum(other_aligned[f"ask_size_{i}"].fillna(0).values[-1] for i in range(1, 4))

    leader_depth = leader_bid3 + leader_ask3
    other_depth = other_bid3 + other_ask3
    f["cross_depth_ratio"] = _safe_div(leader_depth, other_depth + 1.0)

    # Imbalance correlation
    up_imb = up_aligned["book_imbalance"].fillna(0).values
    down_imb = down_aligned["book_imbalance"].fillna(0).values
    if np.std(up_imb) > 0 and np.std(down_imb) > 0:
        corr_matrix = np.corrcoef(up_imb, down_imb)
        f["cross_imb_corr"] = corr_matrix[0, 1] if np.isfinite(corr_matrix[0, 1]) else 0.0
    else:
        f["cross_imb_corr"] = 0.0

    # Other token features (beyond just mid and spread)
    other_pre = other_aligned
    other_mids = other_pre["mid_price"].values
    n_other = len(other_mids)
    if n_other > 5:
        f["other_mid_roc_5s"] = other_mids[-1] - other_mids[-min(6, n_other)]
    else:
        f["other_mid_roc_5s"] = 0.0

    other_imb = other_pre["book_imbalance"].fillna(0).values
    f["other_imb_mean_10s"] = np.mean(other_imb[-min(10, len(other_imb)):])

    return f


# =========================================================================
# Combined feature extraction
# =========================================================================

def extract_all_features(
    up_1s: pd.DataFrame, down_1s: pd.DataFrame,
    open_ms: int, observe_s: float,
    btc_arrays: dict | None = None,
    windows: list[int] | None = None,
) -> dict | None:
    """Extract BOTH baseline and microstructure features for one market."""
    if windows is None:
        windows = OBSERVATION_WINDOWS

    if up_1s.empty or down_1s.empty:
        return None

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
        leader_is_up = True
    else:
        leader_df, other_df = down_1s, up_1s
        leader_label = "Down"
        leader_asset_id = down_pre.iloc[-1]["asset_id"]
        leader_is_up = False

    # --- BASELINE features ---
    features = extract_baseline_multi_window(leader_df, open_ms, windows, observe_s)

    # Other token summary
    other_pre_final = other_df[other_df["sec"] <= observe_s]
    if not other_pre_final.empty:
        other_snap = other_pre_final.iloc[-1]
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

    features["observe_time_s"] = observe_s
    features["observe_time_pct"] = observe_s / (MARKET_DURATION_MS / 1000.0)

    # --- MICROSTRUCTURE features ---
    features.update(extract_micro_rate_of_change(leader_df, observe_s))
    features.update(extract_micro_imbalance_dynamics(leader_df, observe_s))
    features.update(extract_micro_spread_regime(leader_df, observe_s))
    features.update(extract_micro_depth_profile(leader_df, observe_s))
    features.update(extract_micro_price_impact(leader_df, observe_s))
    features.update(extract_micro_asymmetry_dynamics(leader_df, observe_s))
    features.update(extract_micro_cross_token(up_1s, down_1s, observe_s, leader_is_up))

    # --- BTC features ---
    if btc_arrays is not None:
        observe_ms = int(observe_s * 1000)
        btc_features = extract_btc_features(btc_arrays, open_ms, observe_ms)
        features.update(btc_features)

    # --- Metadata ---
    features["_leader_label"] = leader_label
    features["_leader_asset_id"] = leader_asset_id

    return features


# =========================================================================
# Dataset building
# =========================================================================

def build_dataset(
    data_dir: Path,
    resolution_cache_path: Path,
    observe_s: float = 60.0,
    btc_path: Path | None = None,
    max_markets: int | None = None,
    cache_path: Path | None = MICRO_CACHE_PATH,
    rebuild_cache: bool = False,
) -> pd.DataFrame:
    """Build feature matrix with both baseline and microstructure features."""

    if cache_path and cache_path.exists() and not rebuild_cache and max_markets is None:
        print(f"Loading cached features from {cache_path}")
        dataset = pd.read_parquet(cache_path)
        print(f"  {len(dataset)} samples, {dataset['slug'].nunique()} markets")
        return dataset

    resolutions = load_resolutions(resolution_cache_path)
    print(f"Loaded {len(resolutions)} resolutions")

    btc_arrays = None
    btc_min_ms = btc_max_ms = 0
    if btc_path and btc_path.exists():
        btc_df = load_btc_quotes(btc_path)
        btc_min_ms = int(btc_df["timestamp_ms"].min())
        btc_max_ms = int(btc_df["timestamp_ms"].max())
        print(f"Loaded BTC quotes: {len(btc_df)} rows")
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
        del btc_df

    parquet_files = sorted(data_dir.glob("btc-updown-5m-*.parquet"))
    print(f"Found {len(parquet_files)} parquet files")

    if max_markets:
        parquet_files = parquet_files[:max_markets]

    windows_to_train = OBSERVATION_WINDOWS

    rows = []
    skipped = {"no_resolution": 0, "parse_error": 0, "no_features": 0}
    t0 = time.time()

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

        up_raw = df[df["token_label"] == "Up"].sort_values("exchange_timestamp")
        down_raw = df[df["token_label"] == "Down"].sort_values("exchange_timestamp")

        up_1s = resample_to_1s(up_raw, open_ms, close_ms)
        down_1s = resample_to_1s(down_raw, open_ms, close_ms)

        if up_1s.empty or down_1s.empty:
            skipped["no_features"] += 1
            continue

        winning_token = resolutions[slug]

        for obs_s in windows_to_train:
            features = extract_all_features(
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

        if (i + 1) % 200 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(parquet_files) - i - 1) / rate
            print(f"  Processed {i+1}/{len(parquet_files)}, {len(rows)} samples... "
                  f"({rate:.1f} markets/s, ETA {eta:.0f}s)")

    elapsed = time.time() - t0
    print(f"\nExtracted {len(rows)} samples from {len(set(r['slug'] for r in rows))} markets "
          f"in {elapsed:.1f}s")
    print(f"Skipped: {skipped}")

    dataset = pd.DataFrame(rows)
    dataset = dataset.sort_values("open_ts").reset_index(drop=True)

    if cache_path and max_markets is None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        dataset.to_parquet(cache_path, index=False)
        print(f"Cached features to {cache_path} ({cache_path.stat().st_size / 1e6:.1f} MB)")

    return dataset


# =========================================================================
# Training and evaluation
# =========================================================================

# Feature group definitions for ablation
MICRO_FEATURE_GROUPS = {
    "rate_of_change": [
        "mid_roc_5s", "mid_roc_10s", "mid_roc_20s",
        "spread_roc_5s", "spread_roc_10s",
        "depth_roc_5s", "depth_roc_10s",
        "mid_accel_5s", "spread_accel_5s",
        "mid_roc_std_10s", "mid_roc_skew_10s",
    ],
    "imbalance_dynamics": [
        "imb_mean_10s", "imb_std_10s", "imb_trend_10s",
        "imb_mean_30s", "imb_std_30s", "imb_trend_30s",
        "imb_regime_shift", "imb_momentum",
        "imb_max_10s", "imb_min_10s",
    ],
    "spread_regime": [
        "spread_mean", "spread_std", "spread_cv",
        "spread_min", "spread_max",
        "spread_at_min_tick_pct", "spread_quantile_75",
        "spread_trend", "spread_mean_ratio",
    ],
    "depth_profile": [
        "bid_concentration_l1", "ask_concentration_l1",
        "bid_concentration_l3", "ask_concentration_l3",
        "depth_total_10", "bid_depth_10", "ask_depth_10",
        "depth_imb_10",
        "bid_depth_change_pct", "ask_depth_change_pct",
        "depth_profile_asym",
        "bid_wall_ratio", "ask_wall_ratio",
    ],
    "price_impact": [
        "bid_depth_1c", "ask_depth_1c",
        "bid_depth_2c", "ask_depth_2c",
        "bid_depth_3c", "ask_depth_3c",
        "impact_asym_1c", "impact_asym_2c",
    ],
    "asymmetry_dynamics": [
        "bid_size1_mean", "ask_size1_mean",
        "bid_refill_rate", "ask_refill_rate",
        "side_pressure", "depth_ratio_trend",
        "bid_size1_cv", "ask_size1_cv",
    ],
    "cross_token": [
        "cross_mid_corr", "cross_spread_corr",
        "cross_mid_sum_dev", "cross_spread_ratio",
        "cross_depth_ratio", "cross_imb_corr",
        "cross_mid_sum_trend", "cross_mid_sum_vol",
        "other_mid_roc_5s", "other_imb_mean_10s",
    ],
}

ALL_MICRO_FEATURES = []
for group_feats in MICRO_FEATURE_GROUPS.values():
    ALL_MICRO_FEATURES.extend(group_feats)


def get_feature_columns(df: pd.DataFrame, exclude_micro: bool = False) -> list[str]:
    exclude = {"label", "slug", "open_ts"}
    cols = [c for c in df.columns if c not in exclude and not c.startswith("_")]
    if exclude_micro:
        cols = [c for c in cols if c not in ALL_MICRO_FEATURES]
    return cols


def train_model(X_train, y_train, X_val, y_val, feature_cols, verbose=False):
    """Train XGBoost with same hyperparams as baseline."""
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_cols)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_cols)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": 6,
        "learning_rate": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.6,
        "min_child_weight": 15,
        "lambda": 2.0,
        "alpha": 0.5,
        "seed": 42,
        "verbosity": 0,
    }

    model = xgb.train(
        params, dtrain,
        num_boost_round=2000,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=50,
        verbose_eval=200 if verbose else 0,
    )

    return model


def evaluate_model(model, X, y, feature_cols, label=""):
    dmat = xgb.DMatrix(X, feature_names=feature_cols)
    y_proba = model.predict(dmat)
    y_pred = (y_proba >= 0.50).astype(int)

    acc = accuracy_score(y, y_pred)
    try:
        auc = roc_auc_score(y, y_proba)
    except ValueError:
        auc = 0.5
    ll = log_loss(y, y_proba)

    return {"accuracy": acc, "auc": auc, "logloss": ll, "proba": y_proba, "label": label}


def split_data(dataset, feature_cols, train_frac=0.7, val_frac=0.15):
    """Time-based split by market."""
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

    def to_arrays(df):
        X = df[feature_cols].values.astype(np.float32)
        X[~np.isfinite(X)] = 0.0
        y = df["label"].values
        return X, y

    X_train, y_train = to_arrays(train_df)
    X_val, y_val = to_arrays(val_df)
    X_test, y_test = to_arrays(test_df)

    return (X_train, y_train, X_val, y_val, X_test, y_test,
            train_df, val_df, test_df)


def run_comparison(dataset: pd.DataFrame, verbose: bool = True):
    """Train baseline vs enhanced model and compare."""

    print("\n" + "=" * 70)
    print("EXPERIMENT: Baseline vs Microstructure Features")
    print("=" * 70)

    # --- Baseline ---
    baseline_cols = get_feature_columns(dataset, exclude_micro=True)
    enhanced_cols = get_feature_columns(dataset, exclude_micro=False)

    n_baseline = len(baseline_cols)
    n_enhanced = len(enhanced_cols)
    n_micro = n_enhanced - n_baseline

    print(f"\nBaseline features: {n_baseline}")
    print(f"Microstructure features: {n_micro}")
    print(f"Enhanced features: {n_enhanced}")

    # Split data
    (X_train_e, y_train, X_val_e, y_val, X_test_e, y_test,
     train_df, val_df, test_df) = split_data(dataset, enhanced_cols)

    # For baseline, select only baseline columns
    baseline_idx = [enhanced_cols.index(c) for c in baseline_cols]
    X_train_b = X_train_e[:, baseline_idx]
    X_val_b = X_val_e[:, baseline_idx]
    X_test_b = X_test_e[:, baseline_idx]

    print(f"\nTrain: {len(y_train)} samples, Val: {len(y_val)}, Test: {len(y_test)}")
    print(f"Train base rate: {y_train.mean():.3f}, Test base rate: {y_test.mean():.3f}")

    # Train baseline
    print(f"\n--- Training BASELINE ({n_baseline} features) ---")
    model_b = train_model(X_train_b, y_train, X_val_b, y_val, baseline_cols, verbose=verbose)
    results_b = evaluate_model(model_b, X_test_b, y_test, baseline_cols, "Baseline")

    # Train enhanced
    print(f"\n--- Training ENHANCED ({n_enhanced} features) ---")
    model_e = train_model(X_train_e, y_train, X_val_e, y_val, enhanced_cols, verbose=verbose)
    results_e = evaluate_model(model_e, X_test_e, y_test, enhanced_cols, "Enhanced")

    # --- Comparison ---
    print(f"\n{'=' * 70}")
    print("HEAD-TO-HEAD COMPARISON (test set)")
    print(f"{'=' * 70}")
    print(f"{'Metric':<15s}  {'Baseline':>10s}  {'Enhanced':>10s}  {'Delta':>10s}")
    print("-" * 50)

    for metric in ["accuracy", "auc", "logloss"]:
        b_val = results_b[metric]
        e_val = results_e[metric]
        delta = e_val - b_val
        better = "+" if (delta > 0 and metric != "logloss") or (delta < 0 and metric == "logloss") else ""
        print(f"  {metric:<13s}  {b_val:>10.4f}  {e_val:>10.4f}  {delta:>+9.4f} {better}")

    # --- By observation time ---
    if "observe_time_s" in test_df.columns and test_df["observe_time_s"].nunique() > 1:
        print(f"\n{'=' * 70}")
        print("AUC BY OBSERVATION TIME")
        print(f"{'=' * 70}")
        print(f"{'Time':>6s}  {'N':>6s}  {'Base AUC':>10s}  {'Enh AUC':>10s}  {'Delta':>8s}")
        print("-" * 50)

        for t in sorted(test_df["observe_time_s"].unique()):
            t_mask = (test_df["observe_time_s"].values == t)
            if t_mask.sum() < 30:
                continue
            t_true = y_test[t_mask]
            if len(np.unique(t_true)) < 2:
                continue
            try:
                b_auc = roc_auc_score(t_true, results_b["proba"][t_mask])
                e_auc = roc_auc_score(t_true, results_e["proba"][t_mask])
                delta = e_auc - b_auc
                print(f"  {t:>4.0f}s  {t_mask.sum():>6d}  {b_auc:>10.4f}  {e_auc:>10.4f}  {delta:>+7.4f}")
            except ValueError:
                pass

    # --- Threshold sweep comparison ---
    print(f"\n{'=' * 70}")
    print("THRESHOLD SWEEP: Win Rate Comparison")
    print(f"{'=' * 70}")
    print(f"{'Thresh':>8s}  {'B_N':>6s}  {'B_WR':>6s}  {'E_N':>6s}  {'E_WR':>6s}  {'WR_delta':>9s}")
    print("-" * 55)

    for threshold in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]:
        b_mask = results_b["proba"] >= threshold
        e_mask = results_e["proba"] >= threshold
        b_n = b_mask.sum()
        e_n = e_mask.sum()
        if b_n < 10 or e_n < 10:
            continue
        b_wr = y_test[b_mask].mean()
        e_wr = y_test[e_mask].mean()
        print(f"  {threshold:>6.2f}  {b_n:>6d}  {b_wr:>5.1%}  {e_n:>6d}  {e_wr:>5.1%}  {e_wr - b_wr:>+8.1%}")

    # --- Feature importance for enhanced model ---
    print(f"\n{'=' * 70}")
    print("TOP 30 FEATURE IMPORTANCE (gain) — Enhanced Model")
    print(f"{'=' * 70}")

    importance = model_e.get_score(importance_type="gain")
    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:30]
    max_gain = sorted_imp[0][1] if sorted_imp else 1.0

    for rank, (feat, gain) in enumerate(sorted_imp, 1):
        is_micro = feat in ALL_MICRO_FEATURES
        tag = " [MICRO]" if is_micro else ""
        bar = "#" * int(gain / max_gain * 35)
        print(f"  {rank:>2d}. {feat:<35s} {gain:>10.1f}  {bar}{tag}")

    # Count micro features in top N
    for top_n in [10, 20, 30]:
        top_feats = [f for f, _ in sorted_imp[:top_n]]
        micro_count = sum(1 for f in top_feats if f in ALL_MICRO_FEATURES)
        print(f"\n  Micro features in top {top_n}: {micro_count}/{top_n}")

    return model_b, model_e, results_b, results_e


def run_ablation(dataset: pd.DataFrame):
    """Run ablation study: add each feature group one at a time."""

    print(f"\n{'=' * 70}")
    print("ABLATION STUDY: Contribution of Each Feature Group")
    print(f"{'=' * 70}")

    baseline_cols = get_feature_columns(dataset, exclude_micro=True)

    # First train baseline
    (X_train, y_train, X_val, y_val, X_test, y_test,
     train_df, val_df, test_df) = split_data(dataset, baseline_cols)

    model_b = train_model(X_train, y_train, X_val, y_val, baseline_cols)
    results_b = evaluate_model(model_b, X_test, y_test, baseline_cols, "Baseline")

    print(f"\n  Baseline: AUC={results_b['auc']:.4f}, Acc={results_b['accuracy']:.4f}, "
          f"LL={results_b['logloss']:.4f} ({len(baseline_cols)} features)")

    print(f"\n{'Group':<25s}  {'Features':>9s}  {'AUC':>8s}  {'dAUC':>8s}  {'Acc':>8s}  {'dAcc':>8s}")
    print("-" * 75)

    group_results = {}

    for group_name, group_feats in MICRO_FEATURE_GROUPS.items():
        # Only include features that actually exist in the dataset
        available = [f for f in group_feats if f in dataset.columns]
        if not available:
            continue

        cols = baseline_cols + available
        (X_train_g, y_train, X_val_g, y_val, X_test_g, y_test,
         _, _, _) = split_data(dataset, cols)

        model_g = train_model(X_train_g, y_train, X_val_g, y_val, cols)
        results_g = evaluate_model(model_g, X_test_g, y_test, cols, group_name)

        d_auc = results_g["auc"] - results_b["auc"]
        d_acc = results_g["accuracy"] - results_b["accuracy"]

        print(f"  + {group_name:<22s}  {len(available):>9d}  {results_g['auc']:>8.4f}  "
              f"{d_auc:>+7.4f}  {results_g['accuracy']:>8.4f}  {d_acc:>+7.4f}")

        group_results[group_name] = {
            "auc": results_g["auc"],
            "d_auc": d_auc,
            "accuracy": results_g["accuracy"],
            "d_acc": d_acc,
            "n_features": len(available),
        }

    # --- Combined: all micro features ---
    all_cols = get_feature_columns(dataset, exclude_micro=False)
    (X_train_a, y_train, X_val_a, y_val, X_test_a, y_test,
     _, _, _) = split_data(dataset, all_cols)

    model_a = train_model(X_train_a, y_train, X_val_a, y_val, all_cols)
    results_a = evaluate_model(model_a, X_test_a, y_test, all_cols, "All micro")

    d_auc = results_a["auc"] - results_b["auc"]
    d_acc = results_a["accuracy"] - results_b["accuracy"]

    print("-" * 75)
    print(f"  + ALL MICRO             {len(all_cols) - len(baseline_cols):>9d}  "
          f"{results_a['auc']:>8.4f}  {d_auc:>+7.4f}  {results_a['accuracy']:>8.4f}  {d_acc:>+7.4f}")

    # Rank groups by AUC improvement
    print(f"\n  Groups ranked by AUC improvement:")
    ranked = sorted(group_results.items(), key=lambda x: x[1]["d_auc"], reverse=True)
    for rank, (name, res) in enumerate(ranked, 1):
        print(f"    {rank}. {name}: {res['d_auc']:>+.4f} AUC ({res['n_features']} features)")

    return group_results


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="data/telonex_book_snapshots")
    parser.add_argument("--resolution-cache", default="data/resolution_cache.json")
    parser.add_argument("--btc-path", default="data/btc_quotes/btcusdt_quotes.parquet")
    parser.add_argument("--max-markets", type=int, default=None)
    parser.add_argument("--no-btc", action="store_true")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--ablation", action="store_true",
                        help="Run feature group ablation study")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    btc_path = None if args.no_btc else Path(args.btc_path)

    print("XGBoost Microstructure Feature Experiment")
    print("=" * 50)
    print(f"  Data:        {args.data_dir}")
    print(f"  BTC quotes:  {btc_path or 'disabled'}")
    print(f"  Ablation:    {args.ablation}")
    print()

    dataset = build_dataset(
        data_dir=Path(args.data_dir),
        resolution_cache_path=Path(args.resolution_cache),
        btc_path=btc_path,
        max_markets=args.max_markets,
        rebuild_cache=args.rebuild_cache,
    )

    print(f"\nDataset: {dataset.shape[0]} samples, {dataset.shape[1]} columns")
    print(f"Unique markets: {dataset['slug'].nunique()}")
    print(f"Base rate: {dataset['label'].mean():.3f}")

    # Run comparison
    model_b, model_e, results_b, results_e = run_comparison(
        dataset, verbose=not args.quiet
    )

    # Run ablation if requested
    if args.ablation:
        run_ablation(dataset)

    print("\nDone.")


if __name__ == "__main__":
    main()
