"""Feature engineering for Polymarket 5-minute BTC direction prediction.

Builds features from 1-minute Binance candles aligned to fixed 5-minute windows
(matching Polymarket market windows: 00:00-00:05, 00:05-00:10, etc.).

Key difference from directional classifier: each candle's label is whether the
WINDOW it belongs to goes up or down (close@end vs open@start), not a rolling
forward return. Candles within the same window share the same label but have
different features (intra-window position, price progress, etc.).
"""

import numpy as np
import pandas as pd
from pathlib import Path


def _ns_to_ms(ts: pd.Series) -> pd.Series:
    """Convert nanosecond timestamps to milliseconds."""
    return (ts // 1_000_000).astype("int64")


def load_candles_range(
    start: str,
    end: str,
    data_dir: Path,
    exchange: str = "BINANCE_FUTURES",
    symbol: str = "BTC-USDT-PERP",
) -> pd.DataFrame:
    """Load a date range of 1m candle data from CryptoLake parquets."""
    candles_dir = data_dir / "candles" / f"exchange={exchange}" / f"symbol={symbol}"
    dates = pd.date_range(start, end, freq="D", inclusive="left")
    frames = []
    for d in dates:
        path = candles_dir / f"{d.strftime('%Y-%m-%d')}.parquet"
        if path.exists():
            frames.append(pd.read_parquet(path))
    if not frames:
        raise ValueError(f"No candle data for {symbol} in {start} to {end}")
    df = pd.concat(frames, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    df["timestamp_ms"] = _ns_to_ms(df["timestamp"])
    return df


def assign_windows(df: pd.DataFrame, window_minutes: int = 5) -> pd.DataFrame:
    """Assign each candle to its fixed 5-minute window and compute labels.

    Adds columns:
    - window_id: floor of timestamp to window boundary (ms)
    - minute_in_window: 0-4 position within window
    - window_open: open price of the window's first candle
    - window_close: close price of the window's last candle
    - label: 1 if window_close > window_open, 0 otherwise, NaN if flat
    - price_since_open: (current_close - window_open) / window_open
    """
    window_ms = window_minutes * 60 * 1000
    df = df.copy()
    df["window_id"] = (df["timestamp_ms"] // window_ms) * window_ms

    # Minute position within the window (0, 1, 2, 3, 4)
    df["minute_in_window"] = ((df["timestamp_ms"] - df["window_id"]) // 60_000).astype(int)

    # Window open = open of the first candle (minute 0)
    # Window close = close of the last candle (minute 4)
    window_open = df.groupby("window_id")["open"].first()
    window_close = df.groupby("window_id")["close"].last()
    window_count = df.groupby("window_id").size()

    df["window_open"] = df["window_id"].map(window_open)
    df["window_close"] = df["window_id"].map(window_close)
    df["window_candle_count"] = df["window_id"].map(window_count)

    # Label: 1 if up, 0 if down, NaN if exactly flat
    df["label"] = np.where(
        df["window_close"] > df["window_open"], 1.0,
        np.where(df["window_close"] < df["window_open"], 0.0, np.nan)
    )

    # Drop incomplete windows (< window_minutes candles)
    df.loc[df["window_candle_count"] < window_minutes, "label"] = np.nan

    # Intra-window price progress: how much has price moved since window open
    df["price_since_open"] = (df["close"] - df["window_open"]) / df["window_open"]

    return df


def compute_features(candles: pd.DataFrame) -> pd.DataFrame:
    """Compute all features for Polymarket prediction.

    Combines:
    1. Standard OHLCV features (backward-looking, no future leakage)
    2. Intra-window features (position, price progress)

    Parameters
    ----------
    candles : DataFrame with columns from load_candles_range + assign_windows

    Returns
    -------
    DataFrame with timestamp_ms, window_id, label, and all feature columns.
    """
    c = candles["close"].astype(float)
    o = candles["open"].astype(float)
    h = candles["high"].astype(float)
    l = candles["low"].astype(float)
    v = candles["volume"].astype(float)
    n_trades = candles["trades"].astype(float)

    out = pd.DataFrame()
    out["timestamp_ms"] = candles["timestamp_ms"].values
    out["window_id"] = candles["window_id"].values
    out["label"] = candles["label"].values

    # ── Intra-window features (unique to Polymarket) ──────────────────

    # Position within the 5-minute window (0.0 to 0.8)
    out["minute_in_window"] = candles["minute_in_window"].values / 4.0  # normalize to 0-1

    # Price change since window open (key signal: already up = likely stays up)
    out["price_since_open"] = candles["price_since_open"].values

    # Intra-window momentum: are we accelerating or decelerating within the window?
    # Use 1m return relative to overall window direction
    log_c = np.log(c)
    log_ret_1m = log_c.diff().fillna(0)
    out["current_1m_return"] = log_ret_1m.values

    # Intra-window volatility so far (rolling within window)
    # Higher vol within window = more uncertainty about outcome
    out["intra_window_vol"] = candles.groupby("window_id")["close"].transform(
        lambda x: x.pct_change().expanding().std()
    ).fillna(0).values

    # ── Standard backward-looking features ────────────────────────────

    # Log returns at multiple horizons
    for label, win in [("5m", 5), ("15m", 15), ("30m", 30), ("1h", 60)]:
        shifted = c.shift(win)
        with np.errstate(divide="ignore", invalid="ignore"):
            out[f"return_{label}"] = np.where(shifted > 0, np.log(c / shifted), 0.0)

    # Volatility at multiple horizons
    for label, win in [("5m", 5), ("15m", 15), ("30m", 30), ("1h", 60)]:
        out[f"volatility_{label}"] = log_ret_1m.rolling(win, min_periods=1).std().values

    # Volatility ratio (short/long)
    vol_5m = out["volatility_5m"].replace(0, np.nan)
    vol_30m = out["volatility_30m"].replace(0, np.nan)
    out["vol_ratio"] = (vol_5m / vol_30m).fillna(1.0).clip(-10, 10).values

    # Candle structure
    bar_range = (h - l).replace(0, np.nan)
    out["body_ratio"] = ((c - o) / bar_range).fillna(0).clip(-1, 1).values
    out["upper_shadow"] = ((h - np.maximum(o, c)) / bar_range).fillna(0).values
    out["lower_shadow"] = ((np.minimum(o, c) - l) / bar_range).fillna(0).values

    # Garman-Klass volatility
    log_hl = np.log(h / l)
    log_co = np.log(c / o)
    gk_var = 0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2
    for label, win in [("15m", 15), ("1h", 60)]:
        out[f"gk_volatility_{label}"] = gk_var.rolling(win, min_periods=1).mean().fillna(0).values

    # Volume features
    for label, win in [("15m", 15), ("1h", 60), ("4h", 240)]:
        v_ma = v.rolling(win, min_periods=1).mean().replace(0, np.nan)
        out[f"volume_relative_{label}"] = (v / v_ma).fillna(1.0).clip(0, 10).values

    v_5m = v.rolling(5, min_periods=1).sum()
    v_30m = v.rolling(30, min_periods=1).sum().replace(0, np.nan)
    out["volume_momentum"] = (v_5m / v_30m).fillna(1.0).clip(0, 10).values

    nt_ma = n_trades.rolling(60, min_periods=1).mean().replace(0, np.nan)
    out["trades_relative"] = (n_trades / nt_ma).fillna(1.0).clip(0, 10).values

    # Trend/momentum
    ret_shifted = log_ret_1m.shift(1)
    roll_cov = log_ret_1m.rolling(30, min_periods=10).cov(ret_shifted)
    roll_var = log_ret_1m.rolling(30, min_periods=10).var()
    out["return_autocorr"] = (roll_cov / roll_var.replace(0, np.nan)).fillna(0).clip(-1, 1).values

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

    # Jump/tail features
    abs_ret = log_ret_1m.abs()
    for label, win in [("5m", 5), ("15m", 15)]:
        out[f"max_return_{label}"] = abs_ret.rolling(win, min_periods=1).max().values

    # Higher-order
    out["vol_skew_2h"] = log_ret_1m.rolling(120, min_periods=30).skew().fillna(0).clip(-3, 3).values

    vol_1h_for_tail = log_ret_1m.rolling(60, min_periods=10).std().replace(0, np.nan)
    exceeds_2sigma = (abs_ret > 2 * vol_1h_for_tail).astype(float).fillna(0)
    out["tail_frequency_4h"] = exceeds_2sigma.rolling(240, min_periods=30).mean().fillna(0).values

    # ── Cross-window features ─────────────────────────────────────────

    # Previous window outcome (did the last 5-min window go up or down?)
    # This captures short-term momentum/mean-reversion at the window level
    window_labels = candles.groupby("window_id")["label"].first()
    prev_window_label = window_labels.shift(1)
    prev_window_id = pd.Series(candles["window_id"].unique()).shift(1)
    prev_map = dict(zip(candles["window_id"].unique(), prev_window_label.values))
    out["prev_window_outcome"] = candles["window_id"].map(prev_map).fillna(0.5).values

    # Streak: how many consecutive windows went the same direction?
    outcomes = window_labels.dropna().values
    streaks = np.zeros(len(outcomes))
    for i in range(1, len(outcomes)):
        if outcomes[i-1] == outcomes[i-2] if i >= 2 else False:
            streaks[i] = streaks[i-1] + 1 if outcomes[i-1] == outcomes[max(0, i-2)] else 1
        else:
            streaks[i] = 1
    streak_map = dict(zip(window_labels.dropna().index, streaks))
    out["window_streak"] = candles["window_id"].map(streak_map).fillna(0).values

    # Clean infinities
    numeric_cols = out.select_dtypes(include=[np.number]).columns
    out[numeric_cols] = out[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    n_feat = len(get_feature_columns(out))
    print(f"  Built {n_feat} features for {len(out):,} bars ({out['window_id'].nunique():,} windows)")
    return out


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return feature column names (excludes metadata/label columns)."""
    exclude = {"timestamp_ms", "window_id", "label"}
    return [c for c in df.columns if c not in exclude]
