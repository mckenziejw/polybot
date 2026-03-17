"""Multi-horizon labeling for BTC directional classifier.

Labels each 200ms snapshot with future return and directional class
at multiple horizons. Uses volatility-adjusted thresholds for balanced
class distribution across market regimes.
"""

import numpy as np
import pandas as pd

try:
    from numba import njit
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False

from .config import DirectionalConfig


def _barrier_scan_numba(log_mid, tp_thresh, sl_thresh, flat_thresh, max_h):
    """Numba-jitted single-pass barrier scan. ~100x faster than vectorized."""
    if not _HAS_NUMBA:
        raise RuntimeError("numba required for _barrier_scan_numba")
    return _barrier_scan_jit(log_mid, tp_thresh, sl_thresh, flat_thresh, max_h)


def _barrier_scan_numpy(log_mid, tp_thresh, sl_thresh, flat_thresh, max_h):
    """Fallback vectorized barrier scan (no numba)."""
    n = len(log_mid)
    labels = np.full(n, np.nan)
    holding = np.full(n, np.nan)
    barrier = np.zeros(n, dtype=np.int8)  # 0=none, 1=tp, 2=sl, 3=vert, 4=flat

    settled = np.zeros(n, dtype=bool)
    settled[n - max_h:] = True

    for k in range(1, max_h + 1):
        if settled.all():
            break

        future_ret = np.empty(n)
        future_ret[:] = np.nan
        future_ret[:n - k] = log_mid[k:] - log_mid[:n - k]

        tp_hit = (~settled) & (future_ret >= tp_thresh)
        labels[tp_hit] = 1.0
        holding[tp_hit] = k
        barrier[tp_hit] = 1
        settled[tp_hit] = True

        sl_hit = (~settled) & (future_ret <= -sl_thresh)
        labels[sl_hit] = 0.0
        holding[sl_hit] = k
        barrier[sl_hit] = 2
        settled[sl_hit] = True

    vert_mask = (~settled) & (np.arange(n) < n - max_h)
    if vert_mask.any():
        idx = np.where(vert_mask)[0]
        vert_ret = log_mid[idx + max_h] - log_mid[idx]
        ft = flat_thresh[idx]
        for j, i in enumerate(idx):
            if vert_ret[j] > ft[j]:
                labels[i] = 1.0
                holding[i] = max_h
                barrier[i] = 3
            elif vert_ret[j] < -ft[j]:
                labels[i] = 0.0
                holding[i] = max_h
                barrier[i] = 3
            else:
                holding[i] = max_h
                barrier[i] = 4

    return labels, holding, barrier


# Deferred JIT compilation — defined at module level so numba compiles once
if _HAS_NUMBA:
    @njit(cache=True)
    def _barrier_scan_jit(log_mid, tp_thresh, sl_thresh, flat_thresh, max_h):
        n = len(log_mid)
        labels = np.full(n, np.nan)
        holding = np.full(n, np.nan)
        barrier = np.zeros(n, dtype=np.int8)  # 0=none, 1=tp, 2=sl, 3=vert, 4=flat

        for i in range(n - max_h):
            tp = tp_thresh[i]
            sl = sl_thresh[i]
            base = log_mid[i]
            hit = False

            for k in range(1, max_h + 1):
                ret = log_mid[i + k] - base
                if ret >= tp:
                    labels[i] = 1.0
                    holding[i] = k
                    barrier[i] = 1  # tp
                    hit = True
                    break
                elif ret <= -sl:
                    labels[i] = 0.0
                    holding[i] = k
                    barrier[i] = 2  # sl
                    hit = True
                    break

            if not hit:
                # Vertical barrier
                ret = log_mid[i + max_h] - base
                if ret > flat_thresh[i]:
                    labels[i] = 1.0
                    holding[i] = max_h
                    barrier[i] = 3  # vert
                elif ret < -flat_thresh[i]:
                    labels[i] = 0.0
                    holding[i] = max_h
                    barrier[i] = 3  # vert
                else:
                    holding[i] = max_h
                    barrier[i] = 4  # flat

        return labels, holding, barrier


def compute_labels(
    mid_prices: np.ndarray,
    cfg: DirectionalConfig | None = None,
) -> pd.DataFrame:
    """Compute multi-horizon labels from mid-price series.

    Parameters
    ----------
    mid_prices : 1D array
        Mid-price at each 200ms snapshot.
    cfg : DirectionalConfig
        Configuration (horizons, thresholds).

    Returns
    -------
    DataFrame with columns per horizon:
        return_{h}       — raw log return
        label_{h}        — ternary: -1 (down), 0 (flat), +1 (up)
        label_binary_{h} — binary: 0 (down) / 1 (up), NaN for flat
    """
    if cfg is None:
        cfg = DirectionalConfig()

    n = len(mid_prices)
    out = pd.DataFrame(index=range(n))
    sigma = cfg.label_threshold_sigma
    vol_win = cfg.label_vol_window  # 5 min rolling window for vol

    log_mid = np.log(mid_prices)

    for h in cfg.horizons:
        h_label = str(h)

        # Future return at horizon h
        future = np.roll(log_mid, -h)
        future[-h:] = np.nan
        ret = future - log_mid
        out[f"return_{h_label}"] = ret

        # Rolling volatility from REALIZED (backward-looking) h-step returns
        # to avoid look-ahead bias (must not use future returns for thresholds)
        realized_ret = log_mid - np.roll(log_mid, h)
        realized_ret[:h] = 0.0  # no valid lookback for first h points
        realized_series = pd.Series(realized_ret)
        min_per = min(100, vol_win // 2)
        rolling_vol = realized_series.rolling(vol_win, min_periods=min_per).std()
        rolling_vol = rolling_vol.fillna(
            realized_series.expanding(min_periods=max(10, min_per // 2)).std()
        )
        rolling_vol = rolling_vol.fillna(1e-8)  # fallback
        rolling_vol = rolling_vol.clip(lower=1e-8)

        # Ternary label: up/flat/down
        threshold = sigma * rolling_vol.values
        label = np.where(ret > threshold, 1, np.where(ret < -threshold, -1, 0))
        # NaN where return is NaN (end of series)
        label = np.where(np.isnan(ret), np.nan, label)
        out[f"label_{h_label}"] = label

        # Binary label: exclude flat
        binary = np.where(label == 1, 1.0, np.where(label == -1, 0.0, np.nan))
        out[f"label_binary_{h_label}"] = binary

    return out


def compute_triple_barrier_labels(
    mid_prices: np.ndarray,
    cfg: DirectionalConfig | None = None,
) -> pd.DataFrame:
    """Compute triple-barrier labels from mid-price series.

    For each snapshot, scans forward up to max_barrier_horizon steps:
      - TP hit first → label = 1.0
      - SL hit first → label = 0.0
      - Vertical barrier: label by return direction (NaN if within flat zone)

    Barrier thresholds are backward-looking realized vol × sigma multipliers.

    Parameters
    ----------
    mid_prices : 1D array
        Mid-price at each 200ms snapshot.
    cfg : DirectionalConfig

    Returns
    -------
    DataFrame with columns: label_tb (0/1/NaN), holding_period (snapshots),
    barrier_type (tp/sl/vert/flat).
    """
    if cfg is None:
        cfg = DirectionalConfig()

    n = len(mid_prices)
    max_h = cfg.max_barrier_horizon
    vol_win = cfg.barrier_vol_window
    tp_sigma = cfg.tp_sigma
    sl_sigma = cfg.sl_sigma
    flat_sigma = cfg.label_threshold_sigma
    flat_vol_win = cfg.label_vol_window

    log_mid = np.log(mid_prices)

    # Backward-looking realized vol of max_h-step returns for barrier thresholds
    realized_ret = log_mid - np.roll(log_mid, max_h)
    realized_ret[:max_h] = 0.0
    realized_series = pd.Series(realized_ret)
    min_per = min(100, vol_win // 2)
    rolling_vol = realized_series.rolling(vol_win, min_periods=min_per).std()
    rolling_vol = rolling_vol.fillna(
        realized_series.expanding(min_periods=max(10, min_per // 2)).std()
    )
    rolling_vol = rolling_vol.fillna(1e-8).clip(lower=1e-8).values

    tp_thresh = tp_sigma * rolling_vol
    sl_thresh = sl_sigma * rolling_vol

    # Flat-zone threshold for vertical barrier (uses shorter-horizon vol)
    flat_ret = log_mid - np.roll(log_mid, cfg.primary_horizon)
    flat_ret[:cfg.primary_horizon] = 0.0
    flat_series = pd.Series(flat_ret)
    flat_min_per = min(100, flat_vol_win // 2)
    flat_vol = flat_series.rolling(flat_vol_win, min_periods=flat_min_per).std()
    flat_vol = flat_vol.fillna(
        flat_series.expanding(min_periods=max(10, flat_min_per // 2)).std()
    )
    flat_vol = flat_vol.fillna(1e-8).clip(lower=1e-8).values
    flat_thresh = flat_sigma * flat_vol

    # Run barrier scan — numba if available (~100x faster), else numpy fallback
    if _HAS_NUMBA:
        labels, holding, barrier = _barrier_scan_numba(
            log_mid, tp_thresh, sl_thresh, flat_thresh, max_h
        )
    else:
        labels, holding, barrier = _barrier_scan_numpy(
            log_mid, tp_thresh, sl_thresh, flat_thresh, max_h
        )

    # Convert int8 barrier codes to strings
    barrier_map = {0: "", 1: "tp", 2: "sl", 3: "vert", 4: "flat"}
    barrier_type = np.array([barrier_map[b] for b in barrier], dtype=object)

    return pd.DataFrame({
        "label_tb": labels,
        "holding_period": holding,
        "barrier_type": barrier_type,
    })


def label_stats(labels_df: pd.DataFrame, cfg: DirectionalConfig | None = None):
    """Print label distribution statistics."""
    if cfg is None:
        cfg = DirectionalConfig()

    # Triple barrier stats
    if "label_tb" in labels_df.columns:
        tb = labels_df["label_tb"]
        valid = tb.dropna()
        n = len(valid)
        if n > 0:
            up = (valid == 1.0).sum()
            down = (valid == 0.0).sum()
            flat_n = len(tb) - n
            print(
                f"  Triple barrier: up={up/n:.1%} down={down/n:.1%} "
                f"(n={n:,}, flat/NaN={flat_n:,})"
            )
        if "holding_period" in labels_df.columns:
            hp = labels_df["holding_period"].dropna()
            if len(hp) > 0:
                print(
                    f"  Holding period: mean={hp.mean():.0f} "
                    f"median={hp.median():.0f} snapshots "
                    f"({hp.mean() * cfg.snapshot_interval_ms / 1000:.1f}s / "
                    f"{hp.median() * cfg.snapshot_interval_ms / 1000:.1f}s)"
                )
        if "barrier_type" in labels_df.columns:
            bt = labels_df["barrier_type"]
            for btype in ["tp", "sl", "vert", "flat"]:
                count = (bt == btype).sum()
                if count > 0:
                    print(f"    {btype}: {count:,} ({count/len(bt):.1%})")
        return

    # Legacy fixed-horizon stats
    for h in cfg.horizons:
        h_label = str(h)
        seconds = h * cfg.snapshot_interval_ms / 1000.0

        ternary = labels_df[f"label_{h_label}"]
        valid = ternary.dropna()
        n = len(valid)
        if n == 0:
            print(f"  Horizon {h} ({seconds:.1f}s): no valid labels")
            continue

        up = (valid == 1).sum()
        flat = (valid == 0).sum()
        down = (valid == -1).sum()

        binary = labels_df[f"label_binary_{h_label}"].dropna()
        n_binary = len(binary)

        print(
            f"  Horizon {h} ({seconds:.1f}s): "
            f"up={up/n:.1%} flat={flat/n:.1%} down={down/n:.1%} "
            f"(n={n:,}, binary_n={n_binary:,})"
        )
