"""Derivatives feature engineering for BTC directional classifier.

Computes features from Crypto Lake funding rate, open interest, and
liquidation data. All features are backward-looking (no future leakage).

Input: raw Crypto Lake parquets (nanosecond timestamps).
Output: 1-minute aligned DataFrame ready to merge with orderbook features.
"""

import numpy as np
import pandas as pd
from pathlib import Path


def _ns_to_ms(ts: pd.Series) -> pd.Series:
    """Convert nanosecond timestamps to milliseconds."""
    return (ts // 1_000_000).astype("int64")


def _resample_last(df: pd.DataFrame, value_cols: list[str]) -> pd.DataFrame:
    """Resample irregular data to 1-min bars using last value per minute."""
    df = df.copy()
    df["_minute_ms"] = df["timestamp_ms"] // 60_000 * 60_000
    agg = {col: "last" for col in value_cols}
    result = df.groupby("_minute_ms").agg(agg).reset_index()
    result.rename(columns={"_minute_ms": "timestamp_ms"}, inplace=True)
    return result


# ── Funding Rate Features ─────────────────────────────────────────────


def compute_funding_features(funding_df: pd.DataFrame) -> pd.DataFrame:
    """Compute funding rate features from Crypto Lake funding data.

    Input columns: rate, mark_price, index_price, next_funding_time, timestamp
    All timestamps in nanoseconds.

    Returns DataFrame with timestamp_ms + funding features.
    """
    df = funding_df.copy()
    df["timestamp_ms"] = _ns_to_ms(df["timestamp"])

    # Resample to 1-min: last observation per minute
    resampled = _resample_last(df, ["rate", "mark_price", "index_price", "next_funding_time"])

    out = pd.DataFrame()
    out["timestamp_ms"] = resampled["timestamp_ms"]

    # Funding rate level
    out["funding_rate"] = resampled["rate"].values

    # Funding rate moving averages
    rate = resampled["rate"]
    out["funding_rate_ma_15m"] = rate.rolling(15, min_periods=1).mean().values
    out["funding_rate_ma_1h"] = rate.rolling(60, min_periods=1).mean().values

    # Rate change over windows
    out["funding_rate_change_15m"] = (rate - rate.shift(15)).fillna(0).values
    out["funding_rate_change_1h"] = (rate - rate.shift(60)).fillna(0).values

    # Minutes until next funding snapshot
    # next_funding_time is in seconds, current time in ms
    next_fund_s = resampled["next_funding_time"].values
    current_s = resampled["timestamp_ms"].values / 1000.0
    minutes_to = (next_fund_s - current_s) / 60.0
    # Clip to reasonable range (0 to 480 min = 8 hours)
    out["minutes_to_funding"] = np.clip(minutes_to, 0, 480)

    # Mark-index basis in bps
    mark = resampled["mark_price"].values
    index = resampled["index_price"].values
    safe_index = np.where(index > 0, index, 1.0)
    out["mark_index_basis_bps"] = (mark - index) / safe_index * 10_000

    # Basis moving average and deviation
    basis = out["mark_index_basis_bps"]
    out["basis_ma_15m"] = basis.rolling(15, min_periods=1).mean().values
    out["basis_std_1h"] = basis.rolling(60, min_periods=1).std().fillna(0).values

    # ── Higher-order funding features ─────────────────────────────────
    # Funding rate acceleration (change of change)
    rate_change_1m = rate.diff().fillna(0)
    out["funding_rate_acceleration"] = rate_change_1m.diff().fillna(0).values

    # Funding rate momentum (5m MA vs 1h MA)
    ma_5m = rate.rolling(5, min_periods=1).mean()
    ma_1h = rate.rolling(60, min_periods=1).mean()
    out["funding_rate_momentum"] = (ma_5m - ma_1h).fillna(0).values

    # Funding rate volatility (rolling std of rate changes)
    out["funding_rate_vol_1h"] = rate_change_1m.rolling(60, min_periods=1).std().fillna(0).values

    # Basis acceleration
    basis_change = basis.diff().fillna(0)
    out["basis_acceleration"] = basis_change.diff().fillna(0).values

    # Basis momentum (short vs long MA)
    basis_ma_5m = basis.rolling(5, min_periods=1).mean()
    basis_ma_1h = basis.rolling(60, min_periods=1).mean()
    out["basis_momentum"] = (basis_ma_5m - basis_ma_1h).fillna(0).values

    return out


# ── Open Interest Features ────────────────────────────────────────────


def compute_oi_features(oi_df: pd.DataFrame) -> pd.DataFrame:
    """Compute open interest features from Crypto Lake OI data.

    Input columns: open_interest, timestamp
    All timestamps in nanoseconds.

    Returns DataFrame with timestamp_ms + OI features.
    """
    df = oi_df.copy()
    df["timestamp_ms"] = _ns_to_ms(df["timestamp"])

    resampled = _resample_last(df, ["open_interest"])

    out = pd.DataFrame()
    out["timestamp_ms"] = resampled["timestamp_ms"]

    oi = resampled["open_interest"]

    # OI normalized by rolling mean (stationary)
    oi_ma_1h = oi.rolling(60, min_periods=1).mean()
    out["oi_relative"] = (oi / oi_ma_1h.replace(0, np.nan)).fillna(1.0).values

    # Delta OI at multiple windows (normalized by OI level for stationarity)
    oi_safe = oi.replace(0, np.nan)
    for label, win in [("1m", 1), ("5m", 5), ("15m", 15), ("1h", 60)]:
        delta = ((oi - oi.shift(win)) / oi_safe).fillna(0)
        out[f"delta_oi_{label}"] = delta.values

    # OI acceleration (change of 5m pct change)
    delta_5m_pct = (oi - oi.shift(5)) / oi_safe
    delta_5m_pct_prev = delta_5m_pct.shift(5)
    out["oi_acceleration"] = (delta_5m_pct - delta_5m_pct_prev).fillna(0).values

    # OI volatility (rolling std of 1-min pct changes)
    oi_pct_change_1m = (oi.diff() / oi_safe).fillna(0)
    out["oi_volatility_1h"] = oi_pct_change_1m.rolling(60, min_periods=1).std().fillna(0).values

    # ── Higher-order OI features ──────────────────────────────────────
    # OI momentum (5m pct change MA vs 1h pct change MA)
    oi_pct_ma_5m = oi_pct_change_1m.rolling(5, min_periods=1).mean()
    oi_pct_ma_1h = oi_pct_change_1m.rolling(60, min_periods=1).mean()
    out["oi_momentum"] = (oi_pct_ma_5m - oi_pct_ma_1h).fillna(0).values

    # OI-price divergence: OI rising while price falling (or vice versa)
    # Uses mark_price from funding if available, else skip
    # Computed in build_derivatives_features after merge

    return out


# ── Liquidation Features ──────────────────────────────────────────────


def compute_liquidation_features(liq_df: pd.DataFrame) -> pd.DataFrame:
    """Compute liquidation features from Crypto Lake liquidation data.

    Input columns: side, quantity, price, timestamp
    All timestamps in nanoseconds.

    Returns DataFrame with timestamp_ms + liquidation features.
    Sparse data: many minutes will have zero liquidations.
    """
    df = liq_df.copy()
    df["timestamp_ms"] = _ns_to_ms(df["timestamp"])
    df["_minute_ms"] = df["timestamp_ms"] // 60_000 * 60_000

    # Compute USD notional
    df["notional"] = df["quantity"] * df["price"]

    # Aggregate per minute
    buy_mask = df["side"] == "buy"
    sell_mask = df["side"] == "sell"

    # Group by minute
    minutes = df.groupby("_minute_ms")
    buy_groups = df[buy_mask].groupby("_minute_ms")
    sell_groups = df[sell_mask].groupby("_minute_ms")

    agg = pd.DataFrame()
    agg["timestamp_ms"] = sorted(df["_minute_ms"].unique())
    agg = agg.set_index("timestamp_ms")

    agg["liq_buy_notional"] = buy_groups["notional"].sum()
    agg["liq_sell_notional"] = sell_groups["notional"].sum()
    agg["liq_count"] = minutes["notional"].count()
    agg["liq_max_size"] = minutes["notional"].max()

    agg = agg.fillna(0).reset_index()

    # Reindex to complete minute grid so rolling windows span wall-clock time
    ts_min = agg["timestamp_ms"].min()
    ts_max = agg["timestamp_ms"].max()
    full_minutes = pd.DataFrame({
        "timestamp_ms": np.arange(ts_min, ts_max + 60_000, 60_000)
    })
    agg = full_minutes.merge(agg, on="timestamp_ms", how="left").fillna(0)

    out = pd.DataFrame()
    out["timestamp_ms"] = agg["timestamp_ms"]

    # Per-minute volumes
    buy_n = agg["liq_buy_notional"]
    sell_n = agg["liq_sell_notional"]
    total_n = buy_n + sell_n

    out["liq_buy_notional"] = buy_n.values
    out["liq_sell_notional"] = sell_n.values
    out["liq_total_notional"] = total_n.values
    out["liq_count"] = agg["liq_count"].values

    # Imbalance: (buy - sell) / (buy + sell), 0 when no liquidations
    safe_total = total_n.replace(0, np.nan)
    out["liq_imbalance"] = ((buy_n - sell_n) / safe_total).fillna(0).values

    # Rolling intensity
    for label, win in [("5m", 5), ("15m", 15), ("1h", 60)]:
        out[f"liq_intensity_{label}"] = total_n.rolling(win, min_periods=1).sum().values

    # Rolling max single liquidation (cascade indicator)
    max_sz = agg["liq_max_size"]
    out["liq_max_size_5m"] = max_sz.rolling(5, min_periods=1).max().values
    out["liq_max_size_15m"] = max_sz.rolling(15, min_periods=1).max().values

    # Liquidation rate acceleration
    intensity_5m = total_n.rolling(5, min_periods=1).sum()
    intensity_5m_prev = intensity_5m.shift(5)
    out["liq_acceleration"] = (intensity_5m - intensity_5m_prev).fillna(0).values

    return out


# ── Combined Loading & Merging ────────────────────────────────────────


def load_derivatives_day(
    date_str: str,
    data_dir: Path,
    exchange: str = "BINANCE_FUTURES",
    symbol: str = "BTC-USDT-PERP",
) -> dict[str, pd.DataFrame | None]:
    """Load one day of derivatives data from Crypto Lake parquets.

    Returns dict with keys 'funding', 'open_interest', 'liquidations'.
    Values are None if file doesn't exist.
    """
    result = {}
    for dtype in ["funding", "open_interest", "liquidations"]:
        path = data_dir / dtype / f"exchange={exchange}" / f"symbol={symbol}" / f"{date_str}.parquet"
        if path.exists():
            result[dtype] = pd.read_parquet(path)
        else:
            result[dtype] = None
    return result


def load_derivatives_range(
    start: str,
    end: str,
    data_dir: Path,
    exchange: str = "BINANCE_FUTURES",
    symbol: str = "BTC-USDT-PERP",
) -> dict[str, pd.DataFrame | None]:
    """Load a date range of derivatives data, concatenated.

    Parameters
    ----------
    start, end : str  'YYYY-MM-DD' (end exclusive)
    """
    dates = pd.date_range(start, end, freq="D", inclusive="left")
    frames = {"funding": [], "open_interest": [], "liquidations": []}

    for d in dates:
        ds = d.strftime("%Y-%m-%d")
        day_data = load_derivatives_day(ds, data_dir, exchange, symbol)
        for dtype, df in day_data.items():
            if df is not None:
                frames[dtype].append(df)

    result = {}
    for dtype, frame_list in frames.items():
        if frame_list:
            combined = pd.concat(frame_list, ignore_index=True)
            combined = combined.sort_values("timestamp").reset_index(drop=True)
            result[dtype] = combined
        else:
            result[dtype] = None

    return result


def build_derivatives_features(
    derivatives_data: dict[str, pd.DataFrame | None],
) -> pd.DataFrame | None:
    """Build all derivatives features and merge on timestamp_ms.

    Parameters
    ----------
    derivatives_data : dict from load_derivatives_range()

    Returns
    -------
    DataFrame with timestamp_ms + all derivatives features, or None if no data.
    """
    parts = []

    if derivatives_data.get("funding") is not None:
        funding_feats = compute_funding_features(derivatives_data["funding"])
        parts.append(funding_feats)

    if derivatives_data.get("open_interest") is not None:
        oi_feats = compute_oi_features(derivatives_data["open_interest"])
        parts.append(oi_feats)

    if derivatives_data.get("liquidations") is not None:
        liq_feats = compute_liquidation_features(derivatives_data["liquidations"])
        parts.append(liq_feats)

    if not parts:
        return None

    # Use highest-resolution source as merge backbone
    parts.sort(key=lambda p: len(p), reverse=True)
    merged = parts[0]
    for part in parts[1:]:
        merged = pd.merge_asof(
            merged.sort_values("timestamp_ms"),
            part.sort_values("timestamp_ms"),
            on="timestamp_ms",
            direction="backward",
            tolerance=60_000,  # 1 minute tolerance
        )

    # Fill missing values (liquidation minutes with no events → 0)
    for col in merged.columns:
        if col == "timestamp_ms":
            continue
        if "liq_" in col:
            merged[col] = merged[col].fillna(0)
        else:
            merged[col] = merged[col].ffill().fillna(0)

    # ── Cross-feature: OI-price divergence ────────────────────────────
    # OI rising while price falling = bearish divergence (and vice versa)
    if "delta_oi_15m" in merged.columns and "mark_index_basis_bps" in merged.columns:
        basis = merged["mark_index_basis_bps"]
        basis_change_15m = basis - basis.shift(15)
        oi_change_15m = merged["delta_oi_15m"]
        # Divergence: OI direction vs price direction (product < 0 = divergent)
        merged["oi_price_divergence"] = (
            np.sign(oi_change_15m) * np.sign(basis_change_15m) * -1
        ).fillna(0).values
        # Magnitude-weighted divergence
        merged["oi_price_divergence_mag"] = (
            oi_change_15m * basis_change_15m * -1
        ).fillna(0).values

    # Liquidation-funding interaction: high liquidations + extreme funding = cascade risk
    if "liq_intensity_15m" in merged.columns and "funding_rate" in merged.columns:
        liq_int = merged["liq_intensity_15m"]
        fr = merged["funding_rate"]
        # Normalize liq intensity by rolling mean
        liq_ma = liq_int.rolling(60, min_periods=1).mean().replace(0, np.nan)
        liq_rel = (liq_int / liq_ma).fillna(0)
        merged["liq_funding_interaction"] = (liq_rel * fr.abs()).fillna(0).values

    return merged


def merge_with_orderbook_features(
    ob_features: pd.DataFrame,
    deriv_features: pd.DataFrame | None,
) -> pd.DataFrame:
    """Merge derivatives features into orderbook feature DataFrame.

    Uses merge_asof to align by timestamp_ms with 1-minute tolerance.
    Missing derivatives data is filled with 0.
    """
    if deriv_features is None or deriv_features.empty:
        return ob_features

    ob = ob_features.sort_values("timestamp_ms").copy()
    deriv = deriv_features.sort_values("timestamp_ms")

    deriv_cols = [c for c in deriv.columns if c != "timestamp_ms"]

    merged = pd.merge_asof(
        ob,
        deriv,
        on="timestamp_ms",
        direction="backward",
        tolerance=60_000,
    )

    # Fill any NaN derivatives columns with 0
    for col in deriv_cols:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0)

    return merged
