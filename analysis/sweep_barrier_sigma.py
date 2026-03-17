"""Sweep tp/sl sigma on a single walk-forward window.

Computes features once, then re-labels at each sigma value and trains+evaluates.
Reports accuracy, AUC, move sizes, and fee viability for each.

Usage:
    python -m analysis.sweep_barrier_sigma
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd

from strategies.directional.config import DirectionalConfig
from strategies.directional.features import build_features, get_feature_columns
from strategies.directional.labels import compute_triple_barrier_labels, label_stats
from strategies.directional.train import train_ensemble, ensemble_predict

from sklearn.metrics import accuracy_score, roc_auc_score


def load_days(directory: Path, dates: list[str]) -> pd.DataFrame:
    frames = []
    for d in dates:
        p = directory / f"{d}.parquet"
        if p.exists():
            frames.append(pd.read_parquet(p))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values("timestamp_ms").reset_index(drop=True)


def main():
    cfg = DirectionalConfig()
    ob_dir = Path("data/bybit_orderbook_raw")
    tr_dir = Path("data/bybit_trades_200ms")

    # Use window ~midway through dataset for representative results
    # Window 15: train 2025-08-27 → 2025-11-25, test 2025-11-25 → 2025-12-02
    all_dates = sorted(f.stem for f in ob_dir.glob("*.parquet"))
    train_start_idx = all_dates.index("2025-08-27")
    train_end_idx = all_dates.index("2025-11-25")
    test_end_idx = all_dates.index("2025-12-02")

    train_dates = all_dates[train_start_idx:train_end_idx]
    test_dates = all_dates[train_end_idx:test_end_idx]

    print(f"Train: {train_dates[0]} → {train_dates[-1]} ({len(train_dates)} days)")
    print(f"Test:  {test_dates[0]} → {test_dates[-1]} ({len(test_dates)} days)")

    # ── Load and compute features (once) ──────────────────────────────
    print("\nLoading training orderbook...")
    ob_train = load_days(ob_dir, train_dates)
    tr_train = load_days(tr_dir, train_dates)
    tr_train = tr_train if not tr_train.empty else None
    print(f"  {len(ob_train):,} snapshots")

    print("Computing training features...")
    t0 = time.time()
    feat_train = build_features(ob_train, tr_train, cfg)
    print(f"  Done in {time.time() - t0:.0f}s")

    # Purge last 60s
    max_ts = feat_train["timestamp_ms"].max()
    feat_train = feat_train[feat_train["timestamp_ms"] <= max_ts - cfg.purge_seconds * 1000]

    # Subsample
    idx = np.arange(0, len(feat_train), cfg.train_subsample)
    feat_train_sub = feat_train.iloc[idx].reset_index(drop=True)
    print(f"  Subsampled to {len(feat_train_sub):,}")

    del ob_train, tr_train

    print("\nLoading test orderbook...")
    ob_test = load_days(ob_dir, test_dates)
    tr_test = load_days(tr_dir, test_dates)
    tr_test = tr_test if not tr_test.empty else None
    print(f"  {len(ob_test):,} snapshots")

    print("Computing test features...")
    feat_test = build_features(ob_test, tr_test, cfg)
    del ob_test, tr_test

    feat_cols = get_feature_columns(feat_train_sub)

    # ── Sweep sigma values ────────────────────────────────────────────
    sigmas = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    fee_bps = 8.5  # round-trip cost

    print(f"\n{'='*80}")
    print(f"BARRIER SIGMA SWEEP")
    print(f"{'='*80}")
    print(f"{'σ':>5} {'Thresh':>8} {'Labels':>10} {'Flat%':>6} "
          f"{'ValAcc':>7} {'ValAUC':>7} {'TstAcc':>7} {'TstAUC':>7} "
          f"{'AvgMove':>8} {'>Fee%':>6} {'HP(s)':>7}")

    for sigma in sigmas:
        cfg_s = DirectionalConfig()
        cfg_s.tp_sigma = sigma
        cfg_s.sl_sigma = sigma

        # Compute labels for train
        labels_train = compute_triple_barrier_labels(
            feat_train_sub["mid_price"].values, cfg_s
        )

        y_col = "label_tb"
        valid = labels_train[y_col].notna()
        n_valid = valid.sum()
        n_total = len(labels_train)
        flat_pct = 1 - n_valid / n_total

        if n_valid < 1000:
            print(f"{sigma:>5.1f} {'—':>8} {n_valid:>10,} {flat_pct:>6.1%}  too few labels")
            continue

        X_all = feat_train_sub.loc[valid, feat_cols].values.astype(np.float32)
        y_all = labels_train.loc[valid, y_col].values.astype(np.float32)
        X_all[~np.isfinite(X_all)] = 0.0

        val_start = int(len(X_all) * 0.9)
        X_tr, y_tr = X_all[:val_start], y_all[:val_start]
        X_val, y_val = X_all[val_start:], y_all[val_start:]

        # Train ensemble (sequential — numba JIT in parent makes fork unsafe)
        models = train_ensemble(X_tr, y_tr, X_val, y_val, cfg_s, parallel=False)

        val_mean, _ = ensemble_predict(models, X_val)
        val_acc = accuracy_score(y_val, (val_mean > 0.5).astype(int))
        val_auc = roc_auc_score(y_val, val_mean) if len(np.unique(y_val)) > 1 else 0.5

        # Test labels
        labels_test = compute_triple_barrier_labels(
            feat_test["mid_price"].values, cfg_s
        )

        valid_test = labels_test[y_col].notna()
        X_test = feat_test.loc[valid_test, feat_cols].values.astype(np.float32)
        y_test = labels_test.loc[valid_test, y_col].values.astype(np.float32)
        X_test[~np.isfinite(X_test)] = 0.0

        test_mean, _ = ensemble_predict(models, X_test)
        test_acc = accuracy_score(y_test, (test_mean > 0.5).astype(int))
        test_auc = roc_auc_score(y_test, test_mean) if len(np.unique(y_test)) > 1 else 0.5

        # Compute actual move sizes at barrier exit
        mid_test = feat_test["mid_price"].values
        log_mid = np.log(mid_test)
        hp = labels_test["holding_period"].values
        n_test = len(mid_test)
        exit_idx = np.clip((np.arange(n_test) + hp).astype(int), 0, n_test - 1)
        actual_ret_bps = np.where(
            valid_test,
            np.abs(log_mid[exit_idx.astype(int)] - log_mid) * 10000,
            np.nan,
        )
        valid_ret = actual_ret_bps[valid_test.values]
        avg_move_bps = np.nanmean(valid_ret)
        pct_above_fee = np.nanmean(valid_ret > fee_bps) * 100

        # Mean holding period
        hp_valid = hp[valid_test.values]
        mean_hp_s = np.nanmean(hp_valid) * cfg.snapshot_interval_ms / 1000

        # Threshold in bps (approximate from vol)
        thresh_bps = sigma * 5.0  # ~5 bps per sigma from our earlier measurement

        print(f"{sigma:>5.1f} {thresh_bps:>7.1f}bp {n_valid:>10,} {flat_pct:>6.1%} "
              f"{val_acc:>7.4f} {val_auc:>7.4f} {test_acc:>7.4f} {test_auc:>7.4f} "
              f"{avg_move_bps:>7.1f}bp {pct_above_fee:>5.1f}% {mean_hp_s:>7.1f}")

    print(f"\nFee cost: {fee_bps} bps round-trip (taker + maker + slippage)")


if __name__ == "__main__":
    main()
