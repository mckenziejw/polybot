"""OHLCV feature engineering for BTC directional classifier.

Computes features from 1-minute candles as an alternative to orderbook
features. Designed for 15-60 minute prediction horizons where orderbook
microstructure has little predictive power.

All features are backward-looking (no future leakage).
"""

import numpy as np
import pandas as pd
from pathlib import Path

from .config import DirectionalConfig


def _ns_to_ms(ts: pd.Series) -> pd.Series:
    """Convert nanosecond timestamps to milliseconds."""
    return (ts // 1_000_000).astype("int64")


def load_candles_day(
    date_str: str,
    data_dir: Path,
    exchange: str = "BINANCE_FUTURES",
    symbol: str = "BTC-USDT-PERP",
) -> pd.DataFrame | None:
    """Load one day of candle data from Crypto Lake parquet."""
    path = data_dir / "candles" / f"exchange={exchange}" / f"symbol={symbol}" / f"{date_str}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return None


def load_candles_range(
    start: str,
    end: str,
    data_dir: Path,
    exchange: str = "BINANCE_FUTURES",
    symbol: str = "BTC-USDT-PERP",
) -> pd.DataFrame:
    """Load a date range of candle data, concatenated.

    Parameters
    ----------
    start, end : str  'YYYY-MM-DD' (end exclusive)
    """
    dates = pd.date_range(start, end, freq="D", inclusive="left")
    frames = []

    for d in dates:
        ds = d.strftime("%Y-%m-%d")
        df = load_candles_day(ds, data_dir, exchange, symbol)
        if df is not None:
            frames.append(df)

    if not frames:
        raise ValueError(f"No candle data found for {start} to {end}")

    result = pd.concat(frames, ignore_index=True)
    result = result.sort_values("timestamp").reset_index(drop=True)
    return result


def compute_ohlcv_features(
    candles_df: pd.DataFrame,
    cfg: DirectionalConfig,
) -> pd.DataFrame:
    """Compute features from 1-minute OHLCV candles.

    Parameters
    ----------
    candles_df : DataFrame
        Crypto Lake candles with columns: timestamp (ns), open, high, low,
        close, volume, trades.
    cfg : DirectionalConfig

    Returns
    -------
    DataFrame with timestamp_ms, mid_price, and all computed features.
    """
    df = candles_df.copy()
    df["timestamp_ms"] = _ns_to_ms(df["timestamp"])

    out = pd.DataFrame()
    out["timestamp_ms"] = df["timestamp_ms"].values
    out["mid_price"] = df["close"].values  # use close as "mid" for labeling

    o = df["open"]
    h = df["high"]
    l = df["low"]
    c = df["close"]
    v = df["volume"]
    n_trades = df["trades"]

    # ── Price features ─────────────────────────────────────────────────

    # Log returns at multiple horizons
    log_c = np.log(c)
    log_ret_1m = log_c.diff().fillna(0)

    for label, win in [("5m", 5), ("15m", 15), ("30m", 30), ("1h", 60)]:
        shifted = c.shift(win)
        with np.errstate(divide="ignore", invalid="ignore"):
            out[f"return_{label}"] = np.where(
                shifted > 0, np.log(c / shifted), 0.0
            )

    # Volatility at multiple horizons
    for label, win in [("5m", 5), ("15m", 15), ("30m", 30), ("1h", 60)]:
        out[f"volatility_{label}"] = log_ret_1m.rolling(win, min_periods=1).std().values

    # Volatility ratio (short/long)
    vol_5m = out["volatility_5m"].replace(0, np.nan)
    vol_30m = out["volatility_30m"].replace(0, np.nan)
    out["vol_ratio"] = (vol_5m / vol_30m).fillna(1.0).clip(-10, 10).values

    # ── Candle structure features ──────────────────────────────────────

    # Body ratio: (close - open) / (high - low)
    bar_range = (h - l).replace(0, np.nan)
    out["body_ratio"] = ((c - o) / bar_range).fillna(0).clip(-1, 1).values

    # Upper/lower shadow ratio
    out["upper_shadow"] = ((h - np.maximum(o, c)) / bar_range).fillna(0).values
    out["lower_shadow"] = ((np.minimum(o, c) - l) / bar_range).fillna(0).values

    # Garman-Klass volatility (uses OHLC, more efficient than close-close)
    log_hl = np.log(h / l)
    log_co = np.log(c / o)
    gk_var = 0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2
    for label, win in [("15m", 15), ("1h", 60)]:
        out[f"gk_volatility_{label}"] = gk_var.rolling(win, min_periods=1).mean().fillna(0).values

    # ── Volume features ────────────────────────────────────────────────

    # Volume relative to rolling mean (stationary)
    for label, win in [("15m", 15), ("1h", 60), ("4h", 240)]:
        v_ma = v.rolling(win, min_periods=1).mean().replace(0, np.nan)
        out[f"volume_relative_{label}"] = (v / v_ma).fillna(1.0).clip(0, 10).values

    # Volume momentum
    v_5m = v.rolling(5, min_periods=1).sum()
    v_30m = v.rolling(30, min_periods=1).sum().replace(0, np.nan)
    out["volume_momentum"] = (v_5m / v_30m).fillna(1.0).clip(0, 10).values

    # Trade count relative
    nt_ma = n_trades.rolling(60, min_periods=1).mean().replace(0, np.nan)
    out["trades_relative"] = (n_trades / nt_ma).fillna(1.0).clip(0, 10).values

    # ── Trend/momentum features ────────────────────────────────────────

    # Return autocorrelation (lag-1) over 30 bars
    ret_shifted = log_ret_1m.shift(1)
    roll_cov = log_ret_1m.rolling(30, min_periods=10).cov(ret_shifted)
    roll_var = log_ret_1m.rolling(30, min_periods=10).var()
    out["return_autocorr"] = (roll_cov / roll_var.replace(0, np.nan)).fillna(0).clip(-1, 1).values

    # Price relative to moving averages
    for label, win in [("15m", 15), ("1h", 60), ("4h", 240)]:
        ma = c.rolling(win, min_periods=1).mean()
        out[f"price_vs_ma_{label}"] = ((c - ma) / ma.replace(0, np.nan)).fillna(0).values

    # RSI (14-bar)
    delta = c.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=14, adjust=False).mean()
    avg_loss = loss.ewm(span=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out["rsi_14"] = (1 - 1 / (1 + rs)).fillna(0.5).values

    # ── Jump / tail features ───────────────────────────────────────────

    abs_ret = log_ret_1m.abs()
    for label, win in [("5m", 5), ("15m", 15)]:
        out[f"max_return_{label}"] = abs_ret.rolling(win, min_periods=1).max().values

    # ── Higher-order distribution features ────────────────────────────

    # Skewness of 1m returns over 2h (120 bars — need ~100+ for stable estimate)
    out["vol_skew_2h"] = log_ret_1m.rolling(120, min_periods=30).skew().fillna(0).clip(-3, 3).values

    # Tail frequency: fraction of 1m returns exceeding 2σ over 4h
    # More robust than kurtosis at limited sample sizes
    # Use 1h vol as threshold (not same window as counting) to avoid circularity
    vol_1h_for_tail = log_ret_1m.rolling(60, min_periods=10).std().replace(0, np.nan)
    exceeds_2sigma = (abs_ret > 2 * vol_1h_for_tail).astype(float).fillna(0)
    out["tail_frequency_4h"] = exceeds_2sigma.rolling(240, min_periods=30).mean().fillna(0).values

    # Clean
    numeric_cols = out.select_dtypes(include=[np.number]).columns
    out[numeric_cols] = out[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    n_feat = len(out.columns) - 2  # exclude timestamp_ms and mid_price
    print(f"  Built {n_feat} OHLCV features for {len(out):,} bars")
    return out


def get_ohlcv_feature_columns(features_df: pd.DataFrame) -> list[str]:
    """Return feature column names (excludes non-feature columns)."""
    exclude = {"timestamp_ms", "mid_price",
               "label_tb", "holding_period", "barrier_type"}
    return [c for c in features_df.columns if c not in exclude]
