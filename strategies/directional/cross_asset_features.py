"""Cross-asset feature engineering for BTC directional classifier.

Computes features from ETH, SOL, XRP price data relative to BTC.
All features are backward-looking (no future leakage).

Input: BTC 1-minute features DataFrame + alt asset 1-minute klines.
Output: DataFrame with timestamp_ms + cross-asset features.
"""

import numpy as np
import pandas as pd
from pathlib import Path


# ── Data Loading ─────────────────────────────────────────────────────


def load_alt_klines_day(
    date_str: str,
    asset: str,
    data_dir: Path = Path("data"),
) -> pd.DataFrame | None:
    """Load one day of 1-minute resampled alt klines.

    Looks for pre-split daily parquets in data/{asset}_quotes_1m/.
    """
    path = data_dir / f"{asset}_quotes_1m" / f"{date_str}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return None


def load_alt_klines_range(
    start: str,
    end: str,
    asset: str,
    data_dir: Path = Path("data"),
) -> pd.DataFrame | None:
    """Load a date range of alt klines, concatenated."""
    dates = pd.date_range(start, end, freq="D", inclusive="left")
    frames = []
    for d in dates:
        df = load_alt_klines_day(d.strftime("%Y-%m-%d"), asset, data_dir)
        if df is not None:
            frames.append(df)
    if not frames:
        return None
    result = pd.concat(frames, ignore_index=True)
    return result.sort_values("timestamp_ms").reset_index(drop=True)


# ── Feature Computation ──────────────────────────────────────────────


def compute_cross_asset_features(
    btc_timestamps_ms: np.ndarray,
    btc_close: np.ndarray,
    alt_dfs: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Compute cross-asset features from BTC + alt 1-minute data.

    Parameters
    ----------
    btc_timestamps_ms : array of int64
        BTC 1-minute bar timestamps (the merge backbone).
    btc_close : array of float64
        BTC close prices aligned with timestamps.
    alt_dfs : dict mapping asset name (e.g. 'eth') to DataFrame
        Each DataFrame has columns: timestamp_ms, bid_price, ask_price.
        Already resampled to 1-minute.

    Returns
    -------
    DataFrame with timestamp_ms + cross-asset features.
    """
    # Build BTC backbone
    btc = pd.DataFrame({
        "timestamp_ms": btc_timestamps_ms,
        "btc_close": btc_close,
    }).sort_values("timestamp_ms").reset_index(drop=True)

    btc_log_ret_1m = np.log(btc["btc_close"] / btc["btc_close"].shift(1)).fillna(0)

    # Merge each alt onto BTC grid
    alt_rets = {}  # asset -> 1m log return Series aligned to BTC
    alt_volumes = {}  # not available from klines, skip volume share

    for asset, adf in alt_dfs.items():
        if adf is None or adf.empty:
            continue

        # Compute mid price
        alt = adf[["timestamp_ms"]].copy()
        alt["mid"] = (adf["bid_price"] + adf["ask_price"]) / 2

        # merge_asof onto BTC grid (backward — no future leakage)
        merged = pd.merge_asof(
            btc[["timestamp_ms"]],
            alt.sort_values("timestamp_ms"),
            on="timestamp_ms",
            direction="backward",
            tolerance=60_000,
        )

        # Forward-fill small gaps (up to 5 minutes)
        merged["mid"] = merged["mid"].ffill(limit=5)

        # 1m log return for this alt
        alt_log_ret = np.log(merged["mid"] / merged["mid"].shift(1)).fillna(0)
        alt_rets[asset] = alt_log_ret

    if not alt_rets:
        return pd.DataFrame({"timestamp_ms": btc_timestamps_ms})

    out = pd.DataFrame({"timestamp_ms": btc_timestamps_ms})
    alt_names = sorted(alt_rets.keys())

    # ── Per-alt features (ETH only — SOL/XRP contribute to aggregates) ──

    if "eth" in alt_rets:
        eth_ret = alt_rets["eth"]

        # Relative strength: BTC minus ETH return over 15m
        btc_ret_15m = btc_log_ret_1m.rolling(15, min_periods=1).sum()
        eth_ret_15m = eth_ret.rolling(15, min_periods=1).sum()
        out["btc_eth_return_diff_15m"] = (btc_ret_15m - eth_ret_15m).fillna(0).values

        # Lead-lag: lag-1 causal cross-correlation over 60-bar window
        # Does ETH at t-1 predict BTC at t? Use backward-looking data only.
        eth_lagged = eth_ret.shift(1)
        roll_cov = btc_log_ret_1m.rolling(60, min_periods=20).cov(eth_lagged)
        roll_var_eth = eth_lagged.rolling(60, min_periods=20).var()
        roll_var_btc = btc_log_ret_1m.rolling(60, min_periods=20).var()
        denom = np.sqrt(roll_var_eth * roll_var_btc).replace(0, np.nan)
        out["eth_lead_btc"] = (roll_cov / denom).fillna(0).clip(-1, 1).values

    # ── Aggregate cross-asset features ───────────────────────────────

    # Stack all alt 5m returns
    alt_5m_rets = {}
    alt_1m_rets_arr = []
    for asset in alt_names:
        ret = alt_rets[asset]
        ret_5m = ret.rolling(5, min_periods=1).sum()
        alt_5m_rets[asset] = ret_5m
        alt_1m_rets_arr.append(ret.values)

    # Altcoin momentum mean (mean of alt 5m returns)
    alt_5m_arr = np.column_stack([alt_5m_rets[a].values for a in alt_names])
    out["altcoin_momentum_mean"] = np.nanmean(alt_5m_arr, axis=1)

    # Cross-asset volatility dispersion
    # Std of 5m returns across BTC + all alts
    btc_5m_ret = btc_log_ret_1m.rolling(5, min_periods=1).sum().values
    all_5m = np.column_stack([btc_5m_ret] + [alt_5m_rets[a].values for a in alt_names])
    out["cross_asset_vol_dispersion"] = np.nanstd(all_5m, axis=1)

    # Clean
    for col in out.columns:
        if col == "timestamp_ms":
            continue
        out[col] = pd.Series(out[col]).replace([np.inf, -np.inf], np.nan).fillna(0).values

    n_feat = len(out.columns) - 1
    print(f"  Built {n_feat} cross-asset features for {len(out):,} bars")
    return out


# ── Cross-Asset Derivatives ──────────────────────────────────────────


def compute_cross_derivatives_features(
    btc_funding: pd.Series,
    btc_oi_delta: pd.Series,
    eth_funding: pd.Series | None,
    eth_oi_delta: pd.Series | None,
    timestamps_ms: np.ndarray,
) -> pd.DataFrame:
    """Compute cross-asset derivatives features.

    All inputs must be aligned to the same timestamp grid.

    Parameters
    ----------
    btc_funding : Series — BTC funding rate (already in features)
    btc_oi_delta : Series — BTC delta_oi_15m (already in features)
    eth_funding : Series — ETH funding rate (from CryptoLake)
    eth_oi_delta : Series — ETH delta_oi_15m
    timestamps_ms : array — timestamp backbone

    Returns
    -------
    DataFrame with timestamp_ms + cross-derivatives features.
    """
    out = pd.DataFrame({"timestamp_ms": timestamps_ms})

    if eth_funding is not None:
        # BTC-ETH funding rate spread
        out["eth_btc_funding_diff"] = (btc_funding - eth_funding).fillna(0).values

    if eth_oi_delta is not None and btc_oi_delta is not None:
        # Continuous OI divergence: difference in OI z-scores
        btc_oi_z = _zscore_rolling(btc_oi_delta, 60)
        eth_oi_z = _zscore_rolling(eth_oi_delta, 60)
        out["eth_btc_oi_divergence"] = (btc_oi_z - eth_oi_z).fillna(0).values

    n_feat = len(out.columns) - 1
    if n_feat > 0:
        print(f"  Built {n_feat} cross-derivatives features")
    return out


def _zscore_rolling(series: pd.Series, window: int) -> pd.Series:
    """Rolling z-score: (x - rolling_mean) / rolling_std."""
    mu = series.rolling(window, min_periods=1).mean()
    sigma = series.rolling(window, min_periods=1).std().replace(0, np.nan)
    return ((series - mu) / sigma).fillna(0)


# ── Merge Helper ─────────────────────────────────────────────────────


def merge_cross_asset_features(
    base_features: pd.DataFrame,
    cross_features: pd.DataFrame,
) -> pd.DataFrame:
    """Merge cross-asset features into base feature DataFrame.

    Uses merge_asof with backward direction and 1-minute tolerance.
    """
    if cross_features is None or len(cross_features.columns) <= 1:
        return base_features

    base = base_features.sort_values("timestamp_ms").copy()
    cross = cross_features.sort_values("timestamp_ms")

    cross_cols = [c for c in cross.columns if c != "timestamp_ms"]

    merged = pd.merge_asof(
        base,
        cross,
        on="timestamp_ms",
        direction="backward",
        tolerance=60_000,
    )

    for col in cross_cols:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0)

    return merged
