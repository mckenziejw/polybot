"""Feature engineering for BTC directional classifier (v2).

200ms LOB snapshots → ~45 features. Core LOB + rolling stats + regime
features (Hurst, vol-of-vol, autocorrelation) + trade flow + temporal.

Features are grouped:
  A. Core LOB (15) — snapshot-level orderbook features
  B. Rolling window (15) — multi-horizon statistics
  C. Regime (7) — Hurst exponent, vol-of-vol, autocorrelation, spread regime
  D. Trade flow (10) — buy/sell volume, VPIN
  E. Temporal (4) — cyclical time encodings
"""

import numpy as np
import pandas as pd
from .config import DirectionalConfig


def resample_to_1min(
    ob_df: pd.DataFrame,
    trade_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Resample 200ms orderbook snapshots to 1-minute bars.

    Orderbook: last snapshot per minute (point-in-time state).
    Trades: sum volumes, max trade size, sum trade counts per minute.

    Parameters
    ----------
    ob_df : DataFrame
        200ms orderbook snapshots with timestamp_ms column.
    trade_df : DataFrame or None
        200ms-aggregated trade data with timestamp_ms column.

    Returns
    -------
    (ob_1m, trades_1m) — resampled DataFrames aligned by minute.
    """
    # Assign minute bucket (floor to nearest minute)
    ob_df = ob_df.copy()
    ob_df["_minute"] = ob_df["timestamp_ms"] // 60_000 * 60_000

    # Orderbook: take last snapshot per minute
    ob_1m = ob_df.groupby("_minute").last().reset_index()
    ob_1m["timestamp_ms"] = ob_1m["_minute"]
    ob_1m = ob_1m.drop(columns=["_minute"])

    # Trades: aggregate per minute
    trades_1m = None
    if trade_df is not None and not trade_df.empty:
        tr = trade_df.copy()
        tr["_minute"] = tr["timestamp_ms"] // 60_000 * 60_000

        agg = {}
        sum_cols = ["buy_volume", "sell_volume", "n_trades", "n_buy_trades", "n_sell_trades"]
        for col in sum_cols:
            if col in tr.columns:
                agg[col] = "sum"
        if "vwap" in tr.columns:
            agg["vwap"] = "last"
        if "max_trade_size" in tr.columns:
            agg["max_trade_size"] = "max"

        if agg:
            trades_1m = tr.groupby("_minute").agg(agg).reset_index()
            trades_1m["timestamp_ms"] = trades_1m["_minute"]
            trades_1m = trades_1m.drop(columns=["_minute"])

    return ob_1m, trades_1m


def compute_core_lob(df: pd.DataFrame, cfg: DirectionalConfig) -> pd.DataFrame:
    """Core LOB features from raw orderbook snapshot.

    Expects columns: bid_price_1..N, bid_size_1..N, ask_price_1..N, ask_size_1..N.
    """
    n = cfg.n_levels
    out = pd.DataFrame(index=df.index)

    bid_p = df[[f"bid_price_{i}" for i in range(1, n + 1)]].values
    bid_s = df[[f"bid_size_{i}" for i in range(1, n + 1)]].values
    ask_p = df[[f"ask_price_{i}" for i in range(1, n + 1)]].values
    ask_s = df[[f"ask_size_{i}" for i in range(1, n + 1)]].values

    # Mid, spread
    out["mid_price"] = (bid_p[:, 0] + ask_p[:, 0]) / 2.0
    out["spread"] = ask_p[:, 0] - bid_p[:, 0]
    out["spread_rel"] = out["spread"] / out["mid_price"]

    # Imbalance at different depths
    for depth, label in [(1, "1"), (5, "5"), (10, "10"), (20, "20")]:
        d = min(depth, n)
        b = bid_s[:, :d].sum(axis=1)
        a = ask_s[:, :d].sum(axis=1)
        total = b + a
        total = np.where(total == 0, 1.0, total)  # avoid div by zero
        out[f"imbalance_{label}"] = (b - a) / total

    # Depth slope: regression of cumulative volume vs price distance
    for side, prices, sizes, sign in [
        ("bid", bid_p, bid_s, -1),
        ("ask", ask_p, ask_s, 1),
    ]:
        cum_vol = np.cumsum(sizes, axis=1)
        dist = sign * (prices - prices[:, 0:1])  # distance from best
        # Simple linear regression slope per row
        # x = dist, y = cum_vol (across levels)
        x_mean = dist.mean(axis=1, keepdims=True)
        y_mean = cum_vol.mean(axis=1, keepdims=True)
        dx = dist - x_mean
        dy = cum_vol - y_mean
        denom = (dx * dx).sum(axis=1)
        denom = np.where(denom == 0, 1.0, denom)
        slope = (dx * dy).sum(axis=1) / denom
        out[f"depth_slope_{side}"] = slope

    # Microprice — stored as delta from mid (stationary)
    microprice_raw = (
        ask_p[:, 0] * bid_s[:, 0] + bid_p[:, 0] * ask_s[:, 0]
    ) / (bid_s[:, 0] + ask_s[:, 0] + 1e-12)
    out["microprice_delta"] = microprice_raw - out["mid_price"]

    # Total depth — normalized to ratio (stationary)
    total_bid = bid_s.sum(axis=1)
    total_ask = ask_s.sum(axis=1)
    total_depth = total_bid + total_ask
    total_depth_safe = np.where(total_depth == 0, 1.0, total_depth)
    out["depth_ratio"] = total_bid / total_depth_safe

    # Weighted mid — as delta from mid (stationary)
    w3_bid = (bid_p[:, :3] * bid_s[:, :3]).sum(axis=1)
    w3_ask = (ask_p[:, :3] * ask_s[:, :3]).sum(axis=1)
    w3_total = bid_s[:, :3].sum(axis=1) + ask_s[:, :3].sum(axis=1)
    w3_total = np.where(w3_total == 0, 1.0, w3_total)
    weighted_mid_raw = (w3_bid + w3_ask) / w3_total
    out["weighted_mid_delta"] = weighted_mid_raw - out["mid_price"]

    return out


def compute_regime_features(
    df: pd.DataFrame, cfg: DirectionalConfig
) -> pd.DataFrame:
    """Regime features: Hurst exponent, vol-of-vol, autocorrelation, spread regime.

    7 features total, all backward-looking (no future leakage).
    """
    from numpy.lib.stride_tricks import sliding_window_view

    out = pd.DataFrame(index=df.index)
    mid = df["mid_price"].values.copy()
    n = len(mid)

    # Forward-fill zero/negative prices to avoid log(0) spikes that
    # corrupt Hurst, vol-of-vol, and autocorrelation in surrounding windows
    mid[mid <= 0] = np.nan
    mid = pd.Series(mid).ffill().bfill().values
    log_mid = np.log(mid)
    log_ret = np.diff(log_mid, prepend=log_mid[0])

    # ── Hurst exponent at multiple scales (R/S method) ──────────────
    CHUNK_SIZE = 500_000
    for win in cfg.hurst_windows:
        label = {150: "30s", 300: "60s", 1500: "5min"}.get(win, f"{win}")
        hurst = np.full(n, np.nan)

        if n >= win:
            windows_view = sliding_window_view(log_ret, win)  # (n-win+1, win)
            n_windows = len(windows_view)

            for start in range(0, n_windows, CHUNK_SIZE):
                end = min(start + CHUNK_SIZE, n_windows)
                chunk = windows_view[start:end]
                cumdev = np.cumsum(chunk - chunk.mean(axis=1, keepdims=True), axis=1)
                R = cumdev.max(axis=1) - cumdev.min(axis=1)
                S = chunk.std(axis=1, ddof=1)
                S = np.where(S < 1e-12, 1e-12, S)
                rs = R / S
                rs = np.where(rs < 1e-12, 1e-12, rs)
                hurst[win - 1 + start:win - 1 + end] = (
                    np.log(rs) / np.log(win)
                )

        out[f"hurst_{label}"] = np.clip(
            np.nan_to_num(hurst, nan=0.5), 0.0, 1.0
        )

    # ── Vol-of-vol: rolling std of short-term volatility over medium window
    # 200ms: 5s vol over 60s. 1min: 5m vol over 1h.
    if cfg.snapshot_interval_ms >= 60_000:
        vol_short_win, vol_long_win = 5, 60
    else:
        vol_short_win, vol_long_win = 25, 300
    vol_short = pd.Series(log_ret).rolling(vol_short_win, min_periods=1).std()
    out["vol_of_vol"] = vol_short.rolling(vol_long_win, min_periods=1).std().fillna(0.0).values

    # ── Spread regime: rolling percentile of spread_rel over long window
    # 200ms: 5 min (1500). 1min: 1h (60).
    spread_regime_win = 60 if cfg.snapshot_interval_ms >= 60_000 else 1500
    spread_rel = df["spread_rel"]
    roll_rank = spread_rel.rolling(spread_regime_win, min_periods=1).rank(pct=True)
    out["spread_regime"] = roll_rank.fillna(0.5).values

    # ── Return autocorrelation (lag-1) over medium window
    # 200ms: 30s (150). 1min: 30m (30).
    autocorr_win = 30 if cfg.snapshot_interval_ms >= 60_000 else 150
    ret_series = pd.Series(log_ret)
    ret_shifted = ret_series.shift(1)
    roll_cov = ret_series.rolling(autocorr_win, min_periods=10).cov(ret_shifted)
    roll_var = ret_series.rolling(autocorr_win, min_periods=10).var()
    autocorr = (roll_cov / roll_var.replace(0, np.nan)).fillna(0.0).clip(-1, 1)
    out["return_autocorr"] = autocorr.values

    # ── Flow persistence: lag-1 autocorrelation of imbalance_5 over medium window
    imb = df["imbalance_5"]
    imb_shifted = imb.shift(1)
    imb_cov = imb.rolling(autocorr_win, min_periods=10).cov(imb_shifted)
    imb_var = imb.rolling(autocorr_win, min_periods=10).var()
    flow_persist = (imb_cov / imb_var.replace(0, np.nan)).fillna(0.0).clip(-1, 1)
    out["flow_persistence"] = flow_persist.values

    return out


def _rolling_windows(cfg: DirectionalConfig) -> dict[str, int]:
    """Return rolling window sizes (in bars) appropriate for the bar interval."""
    if cfg.snapshot_interval_ms >= 60_000:
        # 1-min bars: 5m, 15m, 30m, 1h
        return {"5m": 5, "15m": 15, "30m": 30, "1h": 60}
    else:
        # 200ms bars: 1s, 5s, 30s, 60s
        return {"1s": 5, "5s": 25, "30s": 150, "60s": 300}


def compute_rolling_features(
    df: pd.DataFrame, cfg: DirectionalConfig
) -> pd.DataFrame:
    """Rolling window features over multiple lookback horizons."""
    out = pd.DataFrame(index=df.index)
    mid = df["mid_price"]

    log_ret = np.log(mid / mid.shift(1)).fillna(0.0)

    windows = _rolling_windows(cfg)
    labels = list(windows.keys())

    for label, win in windows.items():
        # Return over window
        shifted = mid.shift(win)
        with np.errstate(divide="ignore", invalid="ignore"):
            out[f"return_{label}"] = np.where(
                shifted > 0, np.log(mid / shifted), 0.0
            )

        # Volatility over window
        out[f"volatility_{label}"] = log_ret.rolling(win, min_periods=1).std()

        # Imbalance statistics
        imb = df["imbalance_5"]
        out[f"imbalance_mean_{label}"] = imb.rolling(win, min_periods=1).mean()

        # Relative spread mean (stationary)
        out[f"spread_rel_mean_{label}"] = df["spread_rel"].rolling(
            win, min_periods=1
        ).mean()

    # Volatility regime ratio: shortest / second-longest
    vol_short = out[f"volatility_{labels[0]}"].replace(0, np.nan)
    vol_long = out[f"volatility_{labels[-2]}"].replace(0, np.nan)
    out["vol_ratio"] = (vol_short / vol_long).fillna(1.0).clip(-10, 10)

    # Imbalance stability (std over second-shortest window)
    imb_std_win = windows[labels[1]]
    out[f"imbalance_std_{labels[1]}"] = (
        df["imbalance_5"].rolling(imb_std_win, min_periods=1).std()
    )

    # Jump detector: max absolute return over two windows
    abs_ret = log_ret.abs()
    for label in [labels[1], labels[2]]:
        win = windows[label]
        out[f"max_return_{label}"] = abs_ret.rolling(win, min_periods=1).max()

    return out


def compute_trade_flow_features(
    trade_df: pd.DataFrame, cfg: DirectionalConfig
) -> pd.DataFrame:
    """Trade flow features from 200ms-aggregated trade data.

    Expects columns: buy_volume, sell_volume, n_trades, n_buy_trades,
    n_sell_trades, vwap, max_trade_size.
    """
    out = pd.DataFrame(index=trade_df.index)
    bv = trade_df["buy_volume"].fillna(0)
    sv = trade_df["sell_volume"].fillna(0)
    total = bv + sv

    # Buy/sell ratio at multiple windows
    for label, win in [("1s", 5), ("5s", 25), ("30s", 150)]:
        bv_sum = bv.rolling(win, min_periods=1).sum()
        sv_sum = sv.rolling(win, min_periods=1).sum()
        total_sum = bv_sum + sv_sum
        total_sum = total_sum.replace(0, 1.0)
        out[f"buy_sell_ratio_{label}"] = bv_sum / total_sum

    # Volume imbalance
    for label, win in [("5s", 25), ("30s", 150)]:
        bv_sum = bv.rolling(win, min_periods=1).sum()
        sv_sum = sv.rolling(win, min_periods=1).sum()
        total_sum = bv_sum + sv_sum
        total_sum = total_sum.replace(0, 1.0)
        out[f"volume_imbalance_{label}"] = (bv_sum - sv_sum) / total_sum

    # Trade intensity
    nt = trade_df["n_trades"].fillna(0)
    for label, win in [("1s", 5), ("5s", 25)]:
        out[f"trade_intensity_{label}"] = nt.rolling(win, min_periods=1).sum()

    # Large trade flag: max trade in last 5s > 2x rolling median of max_trade_size
    max_sz = trade_df["max_trade_size"].fillna(0)
    median_max_sz = max_sz.rolling(125, min_periods=1).median()  # 25s median
    median_max_sz = median_max_sz.replace(0, 1.0)
    out["large_trade_flag"] = (
        max_sz.rolling(25, min_periods=1).max() > 2.0 * median_max_sz
    ).astype(np.float32)

    # VPIN approximation: |buy - sell| / total over 30s buckets
    abs_imb = (bv - sv).abs()
    abs_imb_sum = abs_imb.rolling(150, min_periods=1).sum()
    total_sum = total.rolling(150, min_periods=1).sum().replace(0, 1.0)
    out["vpin_30s"] = abs_imb_sum / total_sum

    return out


def build_features(
    orderbook_df: pd.DataFrame,
    trade_df: pd.DataFrame | None,
    cfg: DirectionalConfig | None = None,
) -> pd.DataFrame:
    """Build full feature matrix from orderbook + trade DataFrames.

    Parameters
    ----------
    orderbook_df : DataFrame
        200ms orderbook snapshots with bid/ask price/size columns
        and timestamp_ms column.
    trade_df : DataFrame or None
        200ms-aggregated trade data, aligned by index with orderbook_df.
        If None, trade flow features are skipped.
    cfg : DirectionalConfig
        Configuration. Uses defaults if None.

    Returns
    -------
    DataFrame with ~45 feature columns + timestamp_ms.
    """
    if cfg is None:
        cfg = DirectionalConfig()

    print("  Computing core LOB features...")
    core = compute_core_lob(orderbook_df, cfg)

    parts = [core]

    print("  Computing rolling window features...")
    rolling = compute_rolling_features(core, cfg)
    parts.append(rolling)

    print("  Computing regime features...")
    regime = compute_regime_features(core, cfg)
    parts.append(regime)

    features = pd.concat(parts, axis=1)

    if trade_df is not None and len(trade_df) > 0:
        print("  Computing trade flow features...")
        ob_ts = orderbook_df["timestamp_ms"].values
        has_trade_ts = "timestamp_ms" in trade_df.columns
        same_length = has_trade_ts and len(trade_df) == len(orderbook_df)

        if same_length:
            # Same length and assumed aligned — use directly
            trade_feats = compute_trade_flow_features(trade_df, cfg)
            trade_feats.index = features.index
            features = pd.concat([features, trade_feats], axis=1)
        elif has_trade_ts:
            # Different lengths — merge_asof for nearest timestamp alignment
            trade_feats = compute_trade_flow_features(trade_df, cfg)
            trade_feats["_merge_ts"] = trade_df["timestamp_ms"].values
            features["_merge_ts"] = ob_ts
            # Save original order to restore after sort+merge
            features["_orig_order"] = np.arange(len(features))
            features = features.sort_values("_merge_ts")
            trade_feats = trade_feats.sort_values("_merge_ts")
            trade_cols = [c for c in trade_feats.columns if c != "_merge_ts"]
            features = pd.merge_asof(
                features, trade_feats, on="_merge_ts",
                direction="nearest",
                tolerance=cfg.snapshot_interval_ms,  # 200ms tolerance
            )
            features[trade_cols] = features[trade_cols].fillna(0.0)
            # Restore original row order
            features = features.sort_values("_orig_order").reset_index(drop=True)
            features = features.drop(columns=["_merge_ts", "_orig_order"])
        else:
            print("    Warning: trade_df has no timestamp_ms, skipping trade features")

    # Preserve timestamp
    features["timestamp_ms"] = orderbook_df["timestamp_ms"].values

    # Clean infinities/NaN
    numeric_cols = features.select_dtypes(include=[np.number]).columns
    features[numeric_cols] = features[numeric_cols].replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0.0)

    n_feat = len(features.columns) - 1  # exclude timestamp_ms
    print(f"  Built {n_feat} features for {len(features):,} snapshots")
    return features


def get_feature_columns(features_df: pd.DataFrame) -> list[str]:
    """Return feature column names (excludes non-stationary / meta columns)."""
    exclude = {"timestamp_ms", "mid_price", "spread",
               "label_tb", "holding_period", "barrier_type"}
    return [c for c in features_df.columns if c not in exclude]
