"""Walk-forward training for BTC directional classifier.

Trains seed-ensemble XGBoost models on 200ms LOB features with
walk-forward validation. Produces per-window predictions, feature
importance, and accuracy reports.

Usage:
    python -m strategies.directional.train [--data-dir data/bybit_orderbook]
"""

import argparse
import json
import os
import warnings
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import xgboost as xgb
except ImportError:
    print("ERROR: xgboost not installed. Run: pip install xgboost")
    raise SystemExit(1)

from sklearn.metrics import accuracy_score, roc_auc_score

from .config import DirectionalConfig
from .derivatives_features import (
    build_derivatives_features,
    load_derivatives_range,
    merge_with_orderbook_features,
)
from .features import build_features, get_feature_columns, resample_to_1min
from .labels import compute_triple_barrier_labels, label_stats
from .ohlcv_features import get_ohlcv_feature_columns

# Default to pre-computed features; fall back to raw orderbook if not available
_USE_PRECOMPUTED = True

warnings.filterwarnings("ignore", category=FutureWarning)


def load_day(
    date_str: str, cfg: DirectionalConfig
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Load orderbook + trade parquet for a single day."""
    ob_path = cfg.orderbook_dir / f"{date_str}.parquet"
    tr_path = cfg.trades_dir / f"{date_str}.parquet"

    ob = pd.read_parquet(ob_path) if ob_path.exists() else None
    tr = pd.read_parquet(tr_path) if tr_path.exists() else None
    return ob, tr


def load_precomputed_day(
    date_str: str, features_dir: Path
) -> pd.DataFrame | None:
    """Load pre-computed feature parquet for a single day."""
    path = features_dir / f"{date_str}.parquet"
    return pd.read_parquet(path) if path.exists() else None


def load_date_range(
    start: str, end: str, cfg: DirectionalConfig
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Load and concatenate daily files for a date range.

    Parameters
    ----------
    start, end : str  'YYYY-MM-DD' (end exclusive)
    """
    dates = pd.date_range(start, end, freq="D", inclusive="left")
    ob_frames, tr_frames = [], []

    for d in dates:
        ds = d.strftime("%Y-%m-%d")
        ob, tr = load_day(ds, cfg)
        if ob is not None:
            ob_frames.append(ob)
            if tr is not None:
                tr_frames.append(tr)

    if not ob_frames:
        raise ValueError(f"No orderbook data found for {start} to {end}")

    ob_all = pd.concat(ob_frames, ignore_index=True)
    tr_all = pd.concat(tr_frames, ignore_index=True) if tr_frames else None

    print(f"  Loaded {len(ob_all):,} snapshots ({len(ob_frames)} days)")
    return ob_all, tr_all


def load_precomputed_range(
    start: str, end: str, features_dir: Path
) -> pd.DataFrame:
    """Load and concatenate pre-computed feature parquets for a date range."""
    dates = pd.date_range(start, end, freq="D", inclusive="left")
    frames = []

    for d in dates:
        ds = d.strftime("%Y-%m-%d")
        df = load_precomputed_day(ds, features_dir)
        if df is not None:
            frames.append(df)

    if not frames:
        raise ValueError(f"No feature data found for {start} to {end}")

    result = pd.concat(frames, ignore_index=True)
    print(f"  Loaded {len(result):,} snapshots ({len(frames)} days) [precomputed]")
    return result


def prepare_dataset(
    ob_df: pd.DataFrame,
    tr_df: pd.DataFrame | None,
    cfg: DirectionalConfig,
    subsample: int = 1,
    resample_1m: bool = False,
    derivatives_data: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Build features + triple-barrier labels, optionally subsample.

    Returns (features_df, labels_df, feature_columns).
    """
    if resample_1m:
        print(f"  Resampling {len(ob_df):,} snapshots to 1-min bars...")
        ob_df, tr_df = resample_to_1min(ob_df, tr_df)
        print(f"  → {len(ob_df):,} bars")

    features = build_features(ob_df, tr_df, cfg)

    # Merge derivatives features if available
    if derivatives_data is not None:
        print("  Computing derivatives features...")
        deriv_feats = build_derivatives_features(derivatives_data)
        if deriv_feats is not None:
            n_before = len(features.columns)
            features = merge_with_orderbook_features(features, deriv_feats)
            n_after = len(features.columns)
            print(f"  Merged {n_after - n_before} derivatives features")

    labels = compute_triple_barrier_labels(features["mid_price"].values, cfg)

    if subsample > 1:
        idx = np.arange(0, len(features), subsample)
        features = features.iloc[idx].reset_index(drop=True)
        labels = labels.iloc[idx].reset_index(drop=True)
        print(f"  Subsampled to {len(features):,} snapshots (every {subsample})")

    feat_cols = get_feature_columns(features)
    return features, labels, feat_cols


def prepare_precomputed(
    features_df: pd.DataFrame,
    cfg: DirectionalConfig,
    subsample: int = 1,
    force_relabel: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Extract or compute triple-barrier labels from pre-computed features.

    If label_tb column exists and force_relabel is False, uses precomputed
    labels. Otherwise computes labels from mid_price using current cfg
    barrier parameters (tp_sigma, sl_sigma).

    Returns (features_df, labels_df, feature_columns).
    """
    label_cols = ["label_tb", "holding_period", "barrier_type"]
    has_precomputed_labels = all(c in features_df.columns for c in label_cols)

    if has_precomputed_labels and not force_relabel:
        print("  Using precomputed labels from feature parquets")
        labels = features_df[label_cols].copy()
        features_df = features_df.drop(columns=label_cols)
    else:
        # Drop stale label columns if present
        drop_cols = [c for c in label_cols if c in features_df.columns]
        if drop_cols:
            features_df = features_df.drop(columns=drop_cols)
        tp_sl_desc = f"tp={cfg.tp_sigma:.1f}σ sl={cfg.sl_sigma:.1f}σ"
        print(f"  Computing triple-barrier labels ({tp_sl_desc})...")
        labels = compute_triple_barrier_labels(features_df["mid_price"].values, cfg)

    if subsample > 1:
        idx = np.arange(0, len(features_df), subsample)
        features_df = features_df.iloc[idx].reset_index(drop=True)
        labels = labels.iloc[idx].reset_index(drop=True)
        print(f"  Subsampled to {len(features_df):,} snapshots (every {subsample})")

    # Auto-detect feature type: OHLCV features have rsi_14, orderbook has imbalance_1
    if "rsi_14" in features_df.columns and "imbalance_1" not in features_df.columns:
        feat_cols = get_ohlcv_feature_columns(features_df)
    else:
        feat_cols = get_feature_columns(features_df)
    return features_df, labels, feat_cols


def train_single_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg: DirectionalConfig,
    seed: int,
    n_jobs: int = -1,
) -> xgb.XGBClassifier:
    """Train one XGBoost model with early stopping."""
    params = cfg.xgb_params(seed)
    params["n_jobs"] = n_jobs
    model = xgb.XGBClassifier(
        early_stopping_rounds=cfg.early_stopping,
        **params,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    return model


def _train_seed_worker(args):
    """Top-level worker for ProcessPoolExecutor (must be picklable)."""
    X_train, y_train, X_val, y_val, cfg, seed, n_jobs = args
    return train_single_model(X_train, y_train, X_val, y_val, cfg, seed, n_jobs=n_jobs)


def train_ensemble(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg: DirectionalConfig,
    parallel: bool = True,
    n_jobs_override: int | None = None,
) -> list[xgb.XGBClassifier]:
    """Train seed ensemble of n_seeds models.

    When parallel=True, uses ProcessPoolExecutor so each seed gets its own
    OpenMP thread pool. Set parallel=False when numba JIT has already been
    used in the parent process (fork + numba = deadlock).

    n_jobs_override: if set, each seed uses exactly this many threads.
    Use when running multiple training processes in parallel to avoid
    thread oversubscription.
    """
    n_seeds = cfg.n_seeds
    seeds = cfg.seeds[:n_seeds]

    if n_jobs_override is not None:
        cores_per_seed = n_jobs_override
    else:
        total_cores = os.cpu_count() or 1
        cores_per_seed = max(1, total_cores // n_seeds)

    if not parallel or n_seeds <= 1:
        return [
            train_single_model(X_train, y_train, X_val, y_val, cfg, seed, n_jobs=cores_per_seed)
            for seed in seeds
        ]

    worker_args = [
        (X_train, y_train, X_val, y_val, cfg, seed, cores_per_seed)
        for seed in seeds
    ]

    with ProcessPoolExecutor(max_workers=n_seeds) as pool:
        models = list(pool.map(_train_seed_worker, worker_args))
    return models


def ensemble_predict(
    models: list[xgb.XGBClassifier], X: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Ensemble prediction: mean probability + uncertainty (std).

    Returns (mean_proba, std_proba) both shape (n_samples,).
    """
    preds = np.array([m.predict_proba(X)[:, 1] for m in models])
    return preds.mean(axis=0), preds.std(axis=0)


def walk_forward(cfg: DirectionalConfig, args, n_jobs_per_seed: int | None = None,
                 force_relabel: bool = False) -> dict:
    """Run full walk-forward validation.

    Returns dict with per-window results and aggregate metrics.
    """
    cfg.orderbook_dir = Path(args.data_dir)
    cfg.trades_dir = Path(args.trades_dir)
    cfg.model_dir = Path(args.model_dir)
    cfg.model_dir.mkdir(parents=True, exist_ok=True)

    derivatives_dir = Path(args.derivatives_dir) if args.derivatives_dir else None
    if derivatives_dir and derivatives_dir.exists():
        print(f"Using derivatives data from {derivatives_dir}")
    else:
        derivatives_dir = None

    features_dir = Path(args.features_dir)
    use_precomputed = _USE_PRECOMPUTED and features_dir.exists() and any(features_dir.glob("*.parquet"))
    if use_precomputed:
        print(f"Using pre-computed features from {features_dir}")

    y_col = "label_tb"

    # Find available date range — check features dir first, then orderbook
    scan_dir = features_dir if use_precomputed else cfg.orderbook_dir
    parquet_files = sorted(scan_dir.glob("*.parquet"))
    if not parquet_files:
        print(f"No parquet files in {scan_dir}")
        return {}

    dates = [f.stem for f in parquet_files]
    first_date = dates[0]
    last_date = dates[-1]
    print(f"Data range: {first_date} to {last_date} ({len(dates)} days)")

    # Generate walk-forward windows
    # Note: purge is applied at timestamp level after loading, not by dropping days.
    # We load train through day before test, then drop last purge_seconds of training.
    all_dates = pd.date_range(first_date, last_date, freq="D")
    windows = []
    train_days = cfg.train_days
    test_days = cfg.test_days

    i = train_days
    while i + test_days <= len(all_dates):
        train_start = all_dates[i - train_days].strftime("%Y-%m-%d")
        # Load up to the day before test starts (no day-level purge gap)
        train_end = all_dates[i].strftime("%Y-%m-%d")  # exclusive end
        test_start = all_dates[i].strftime("%Y-%m-%d")
        test_end_idx = min(i + test_days, len(all_dates))
        # Use timedelta to get exclusive end date when at array boundary
        if test_end_idx < len(all_dates):
            test_end = all_dates[test_end_idx].strftime("%Y-%m-%d")
        else:
            test_end = (all_dates[-1] + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        windows.append((train_start, train_end, test_start, test_end))
        i += test_days

    print(f"Walk-forward: {len(windows)} windows")
    if not windows:
        print("Not enough data for even one window")
        return {}

    all_results = []
    window_metrics = []

    for w_idx, (tr_start, tr_end, te_start, te_end) in enumerate(windows):
        print(f"\n{'='*60}")
        print(f"Window {w_idx + 1}/{len(windows)}: "
              f"train {tr_start}→{tr_end}, test {te_start}→{te_end}")
        print(f"{'='*60}")

        # Load training data
        print("Loading training data...")
        if use_precomputed:
            try:
                feats_raw = load_precomputed_range(tr_start, tr_end, features_dir)
            except ValueError as e:
                print(f"  Skipping window: {e}")
                continue

            # Apply purge on precomputed features
            if cfg.purge_seconds > 0 and len(feats_raw) > 0:
                max_ts = feats_raw["timestamp_ms"].max()
                purge_cutoff = max_ts - cfg.purge_seconds * 1000
                feats_raw = feats_raw[feats_raw["timestamp_ms"] <= purge_cutoff]
                print(f"  Purged last {cfg.purge_seconds}s of training data")

            if len(feats_raw) == 0:
                print("  Training data empty after purge, skipping window")
                continue

            feats_train, labels_train, feat_cols = prepare_precomputed(
                feats_raw, cfg, subsample=cfg.train_subsample,
                force_relabel=force_relabel,
            )
            del feats_raw
        else:
            try:
                ob_train, tr_train = load_date_range(tr_start, tr_end, cfg)
            except ValueError as e:
                print(f"  Skipping window: {e}")
                continue

            # Apply purge: drop last purge_seconds of training data
            if cfg.purge_seconds > 0 and len(ob_train) > 0:
                max_ts = ob_train["timestamp_ms"].max()
                purge_cutoff = max_ts - cfg.purge_seconds * 1000
                ob_train = ob_train[ob_train["timestamp_ms"] <= purge_cutoff]
                if tr_train is not None:
                    tr_train = tr_train[tr_train["timestamp_ms"] <= purge_cutoff]
                print(f"  Purged last {cfg.purge_seconds}s of training data")

            if len(ob_train) == 0:
                print("  Training data empty after purge, skipping window")
                continue

            # Load derivatives data for training window
            train_deriv = None
            if derivatives_dir:
                print("  Loading training derivatives...")
                train_deriv = load_derivatives_range(
                    tr_start, tr_end, derivatives_dir
                )

            resample = cfg.snapshot_interval_ms >= 60_000
            feats_train, labels_train, feat_cols = prepare_dataset(
                ob_train, tr_train, cfg, subsample=cfg.train_subsample,
                resample_1m=resample, derivatives_data=train_deriv,
            )
            del ob_train, tr_train, train_deriv

        # Label stats
        label_stats(labels_train, cfg)

        # Binary label from triple barrier
        valid_mask = labels_train[y_col].notna()
        X_all = feats_train.loc[valid_mask, feat_cols].values.astype(np.float32)
        y_all = labels_train.loc[valid_mask, y_col].values.astype(np.float32)
        ts_all = feats_train.loc[valid_mask, "timestamp_ms"].values

        if len(y_all) < 1000:
            print(f"  Too few valid labels ({len(y_all)}), skipping")
            continue

        # Val split: last 10% of training
        val_start = int(len(X_all) * 0.9)
        X_tr, y_tr = X_all[:val_start], y_all[:val_start]
        X_val, y_val = X_all[val_start:], y_all[val_start:]

        print(f"  Train: {len(X_tr):,}, Val: {len(X_val):,}")

        # Clean non-finite values
        X_tr[~np.isfinite(X_tr)] = 0.0
        X_val[~np.isfinite(X_val)] = 0.0

        # Train ensemble
        print(f"  Training {cfg.n_seeds}-seed ensemble...")
        # Use sequential training to avoid fork+numba deadlock and
        # ProcessPoolExecutor serialization overhead. n_jobs controls
        # XGBoost's internal thread parallelism per model.
        models = train_ensemble(
            X_tr, y_tr, X_val, y_val, cfg,
            parallel=False, n_jobs_override=n_jobs_per_seed,
        )

        # Validate on val set
        val_mean, val_std = ensemble_predict(models, X_val)
        val_acc = accuracy_score(y_val, (val_mean > 0.5).astype(int))
        val_auc = roc_auc_score(y_val, val_mean) if len(np.unique(y_val)) > 1 else 0.5
        print(f"  Val accuracy: {val_acc:.4f}, AUC: {val_auc:.4f}")

        # Load test data
        print("Loading test data...")
        if use_precomputed:
            try:
                feats_test_raw = load_precomputed_range(te_start, te_end, features_dir)
            except ValueError as e:
                print(f"  Skipping test: {e}")
                continue

            feats_test, labels_test, feat_cols_test = prepare_precomputed(
                feats_test_raw, cfg, subsample=1,  # full resolution for test
                force_relabel=force_relabel,
            )
            del feats_test_raw
        else:
            try:
                ob_test, tr_test = load_date_range(te_start, te_end, cfg)
            except ValueError as e:
                print(f"  Skipping test: {e}")
                continue

            # Load derivatives data for test window
            test_deriv = None
            if derivatives_dir:
                print("  Loading test derivatives...")
                test_deriv = load_derivatives_range(
                    te_start, te_end, derivatives_dir
                )

            resample = cfg.snapshot_interval_ms >= 60_000
            feats_test, labels_test, feat_cols_test = prepare_dataset(
                ob_test, tr_test, cfg, subsample=1,  # full resolution for test
                resample_1m=resample, derivatives_data=test_deriv,
            )
            del ob_test, tr_test, test_deriv

        # Ensure test has same feature columns as train (handle missing trade data)
        missing_cols = set(feat_cols) - set(feat_cols_test)
        for mc in missing_cols:
            feats_test[mc] = 0.0
        extra_cols = set(feat_cols_test) - set(feat_cols)
        if extra_cols:
            print(f"  Warning: test has {len(extra_cols)} extra feature columns (dropped)")

        valid_test = labels_test[y_col].notna()
        X_test = feats_test.loc[valid_test, feat_cols].values.astype(np.float32)
        y_test = labels_test.loc[valid_test, y_col].values.astype(np.float32)
        ts_test = feats_test.loc[valid_test, "timestamp_ms"].values
        X_test[~np.isfinite(X_test)] = 0.0

        # Ensemble predict on test
        test_mean, test_std = ensemble_predict(models, X_test)
        test_acc = accuracy_score(y_test, (test_mean > 0.5).astype(int))
        test_auc = roc_auc_score(y_test, test_mean) if len(np.unique(y_test)) > 1 else 0.5

        print(f"  Test accuracy: {test_acc:.4f}, AUC: {test_auc:.4f}")
        print(f"  Test samples: {len(y_test):,}")

        # Feature importance (average gain across ensemble)
        importance = np.zeros(len(feat_cols))
        for m in models:
            imp = m.feature_importances_
            importance += imp
        importance /= len(models)
        top_idx = np.argsort(importance)[::-1][:15]
        print("  Top 15 features (gain):")
        for rank, idx in enumerate(top_idx, 1):
            print(f"    {rank:2d}. {feat_cols[idx]:30s} {importance[idx]:.4f}")

        # Extract holding_period for test rows
        hp_test = labels_test.loc[valid_test, "holding_period"].values

        # Store results
        window_result = pd.DataFrame({
            "timestamp_ms": ts_test,
            "y_true": y_test,
            "y_pred": test_mean,
            "y_std": test_std,
            "holding_period": hp_test,
            "window": w_idx,
        })
        all_results.append(window_result)

        mean_hp = float(np.nanmean(hp_test))
        mean_hp_s = mean_hp * cfg.snapshot_interval_ms / 1000.0
        print(f"  Mean holding period: {mean_hp:.0f} snapshots ({mean_hp_s:.1f}s)")

        window_metrics.append({
            "window": w_idx,
            "train_range": f"{tr_start} → {tr_end}",
            "test_range": f"{te_start} → {te_end}",
            "train_n": len(X_tr),
            "test_n": len(X_test),
            "val_acc": float(val_acc),
            "val_auc": float(val_auc),
            "test_acc": float(test_acc),
            "test_auc": float(test_auc),
            "mean_holding_period": float(mean_hp),
        })

        # Save models
        for s_idx, m in enumerate(models):
            model_path = cfg.model_dir / f"window_{w_idx}_seed_{s_idx}.json"
            m.save_model(str(model_path))

    if not all_results:
        print("\nNo windows completed successfully.")
        return {"windows": window_metrics}

    # Aggregate results
    results_df = pd.concat(all_results, ignore_index=True)

    print(f"\n{'='*60}")
    print("AGGREGATE RESULTS")
    print(f"{'='*60}")

    overall_acc = accuracy_score(
        results_df["y_true"], (results_df["y_pred"] > 0.5).astype(int)
    )
    overall_auc = roc_auc_score(results_df["y_true"], results_df["y_pred"])
    print(f"Overall accuracy: {overall_acc:.4f}")
    print(f"Overall AUC: {overall_auc:.4f}")
    print(f"Total test samples: {len(results_df):,}")

    # Confidence threshold sweep
    print("\nConfidence threshold sweep:")
    print(f"{'Threshold':>10} {'Acc':>8} {'N':>10} {'Trades/day':>12}")
    # Count actual test days from results timestamps
    test_dates = pd.to_datetime(
        results_df["timestamp_ms"].astype(np.int64), unit="ms", utc=True
    ).dt.date.nunique()
    total_days = max(test_dates, 1)
    for thresh in cfg.confidence_sweep:
        mask = (results_df["y_pred"] > thresh) | (results_df["y_pred"] < (1 - thresh))
        if mask.sum() == 0:
            continue
        sub = results_df[mask]
        # For binary: predict up if > thresh, down if < (1-thresh)
        pred = (sub["y_pred"] > 0.5).astype(int)
        acc = accuracy_score(sub["y_true"], pred)
        n = len(sub)
        tpd = n / max(total_days, 1)
        print(f"{thresh:>10.2f} {acc:>8.4f} {n:>10,} {tpd:>12.1f}")

    # Per-window summary
    print("\nPer-window summary:")
    for wm in window_metrics:
        print(
            f"  W{wm['window']}: acc={wm['test_acc']:.4f} "
            f"auc={wm['test_auc']:.4f} n={wm['test_n']:,} "
            f"({wm['test_range']})"
        )

    # Save results
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    results_df.to_parquet(results_dir / "predictions.parquet", index=False)
    with open(results_dir / "window_metrics.json", "w") as f:
        json.dump(window_metrics, f, indent=2)

    print(f"\nResults saved to {results_dir}")

    return {
        "windows": window_metrics,
        "overall_acc": float(overall_acc),
        "overall_auc": float(overall_auc),
        "total_samples": len(results_df),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Walk-forward training for BTC directional classifier (v2)"
    )
    parser.add_argument(
        "--data-dir", default="data/bybit_orderbook_raw",
        help="Directory with daily orderbook parquet files",
    )
    parser.add_argument(
        "--trades-dir", default="data/bybit_trades_200ms",
        help="Directory with daily trade parquet files",
    )
    parser.add_argument(
        "--variant", choices=["ob", "ohlcv"], default=None,
        help="Feature variant shortcut: sets features/model/results dirs automatically",
    )
    parser.add_argument(
        "--features-dir", default=None,
        help="Directory with pre-computed feature parquets (used if available)",
    )
    parser.add_argument(
        "--model-dir", default=None,
        help="Directory to save trained models",
    )
    parser.add_argument(
        "--results-dir", default=None,
        help="Directory to save predictions and metrics",
    )
    parser.add_argument(
        "--horizon", type=int, default=None,
        help="Override primary horizon (in snapshots, default 25=5s)",
    )
    parser.add_argument(
        "--subsample", type=int, default=None,
        help="Override training subsample rate (default 5=every 1s)",
    )
    parser.add_argument(
        "--train-days", type=int, default=None,
        help="Override training window size in days",
    )
    parser.add_argument(
        "--test-days", type=int, default=None,
        help="Override test window size in days",
    )
    parser.add_argument(
        "--resolution", choices=["200ms", "1m"], default="200ms",
        help="Bar resolution: 200ms (original) or 1m (resampled)",
    )
    parser.add_argument(
        "--derivatives-dir", default=None,
        help="Directory with Crypto Lake derivatives data (funding/OI/liquidations)",
    )
    parser.add_argument(
        "--n-jobs", type=int, default=None,
        help="Threads per XGBoost seed (default: auto = total_cores / n_seeds). "
             "Set explicitly when running multiple training processes in parallel.",
    )
    parser.add_argument(
        "--tp-sigma", type=float, default=None,
        help="Override take-profit sigma for triple barrier labels",
    )
    parser.add_argument(
        "--sl-sigma", type=float, default=None,
        help="Override stop-loss sigma for triple barrier labels",
    )
    args = parser.parse_args()

    # Resolve variant-based directory defaults
    variant = args.variant or "ob"
    if args.features_dir is None:
        args.features_dir = f"data/directional_features_v3_{variant}"
    if args.model_dir is None:
        args.model_dir = f"data/directional_models_v3_{variant}"
    if args.results_dir is None:
        args.results_dir = f"data/directional_results_v3_{variant}"

    if args.resolution == "1m":
        cfg = DirectionalConfig.config_1m()
        print(f"Using 1-minute bar configuration (v3, variant={variant})")
    else:
        cfg = DirectionalConfig()
    if args.horizon is not None:
        cfg.primary_horizon = args.horizon
    if args.subsample is not None:
        cfg.train_subsample = args.subsample
    if args.train_days is not None:
        cfg.train_days = args.train_days
    if args.test_days is not None:
        cfg.test_days = args.test_days
    if args.tp_sigma is not None:
        cfg.tp_sigma = args.tp_sigma
    if args.sl_sigma is not None:
        cfg.sl_sigma = args.sl_sigma

    # Override results/model dirs with barrier info when sigma is non-default
    if args.tp_sigma is not None or args.sl_sigma is not None:
        tp = cfg.tp_sigma
        sl = cfg.sl_sigma
        barrier_tag = f"tp{tp:.1f}_sl{sl:.1f}"
        if args.model_dir is None:
            args.model_dir = f"data/directional_models_v3_{variant}_{barrier_tag}"
        if args.results_dir is None:
            args.results_dir = f"data/directional_results_v3_{variant}_{barrier_tag}"

    relabel = args.tp_sigma is not None or args.sl_sigma is not None
    walk_forward(cfg, args, n_jobs_per_seed=args.n_jobs, force_relabel=relabel)


if __name__ == "__main__":
    main()
