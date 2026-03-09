#!/usr/bin/env python3
"""
XGBoost v3 model server for live trading.

Trains the model on ALL available data at startup, then serves predictions
via HTTP. The TypeScript execution strategy sends accumulated orderbook
snapshots + BTC prices; this server extracts features using the exact same
code as training (zero train/serve skew) and returns predictions.

Endpoints:
    POST /predict   — receive snapshots, return prediction + vol assessment
    GET  /health    — liveness check

Usage:
    python model_server.py                  # train on all data, serve on :8000
    python model_server.py --port 8001      # custom port
    python model_server.py --model-path data/xgb_model.json  # load pre-trained
"""

import argparse
import json
import sys
import warnings
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

# Reuse exact feature extraction from the training pipeline
from xgboost_flexible import (
    FEATURE_CACHE_PATH,
    OBSERVATION_WINDOWS,
    BTC_LOOKBACKS,
    MARKET_DURATION_MS,
    extract_market_features,
    extract_btc_features,
    get_feature_columns,
    compute_time_weights,
    _empty_btc_features,
)

warnings.filterwarnings("ignore")

MODEL_PATH = Path("data/xgb_model.json")
META_PATH = Path("data/xgb_model_meta.json")
SPOT_QUOTES_PATH = Path("data/btc_quotes/btcusdt_quotes.parquet")


def train_production_model() -> tuple[xgb.Booster, list[str], float | None]:
    """Train the v3 XGBoost model on ALL available data (no holdout).

    For production, we use every market to maximize model quality.
    The vol threshold is computed from the full dataset.
    """
    df = pd.read_parquet(FEATURE_CACHE_PATH)
    feature_cols = get_feature_columns(df)

    print(f"Training on {len(df)} samples, {df['slug'].nunique()} markets, {len(feature_cols)} features")

    # Use 90% for training, last 10% for early stopping
    slug_order = df.groupby("slug")["open_ts"].first().sort_values()
    n_markets = len(slug_order)
    train_end = int(n_markets * 0.90)

    train_slugs = set(slug_order.index[:train_end])
    val_slugs = set(slug_order.index[train_end:])

    train_df = df[df["slug"].isin(train_slugs)]
    val_df = df[df["slug"].isin(val_slugs)]

    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df["label"].values
    X_val = val_df[feature_cols].values.astype(np.float32)
    y_val = val_df["label"].values

    for X in [X_train, X_val]:
        X[~np.isfinite(X)] = 0.0

    # Time-weighted training: recent data gets higher weight (7-day half-life)
    train_weights = compute_time_weights(train_df, half_life_days=7.0)
    print(f"  Time weights: min={train_weights.min():.3f}, max={train_weights.max():.3f}, "
          f"mean={train_weights.mean():.3f}")

    dtrain = xgb.DMatrix(X_train, label=y_train, weight=train_weights, feature_names=feature_cols)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_cols)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": 2,           # v3.3: sweep-optimized (300 trials, 101 features w/ time)
        "learning_rate": 0.005016,
        "subsample": 0.930265,
        "colsample_bytree": 0.985988,
        "min_child_weight": 41,
        "lambda": 0.224145,
        "alpha": 0.827467,
        "gamma": 1.705416,
        "seed": 42,
        "verbosity": 0,
    }

    model = xgb.train(
        params, dtrain,
        num_boost_round=2000,
        evals=[(dval, "val")],
        early_stopping_rounds=50,
        verbose_eval=False,
    )
    print(f"  Best iteration: {model.best_iteration}")

    # Vol threshold disabled in v3.2 — model handles low-vol regimes well
    vol_threshold = None

    # Spot price stats for z-score normalization — must come from training data
    # to avoid distribution shift in live inference
    btc_mid_mean = 0.0
    btc_mid_std = 1.0
    if SPOT_QUOTES_PATH.exists():
        quotes_df = pd.read_parquet(SPOT_QUOTES_PATH)
        btc_mid_mean = float(quotes_df["mid_price"].mean())
        btc_mid_std = float(quotes_df["mid_price"].std())
        print(f"  Spot stats: mean={btc_mid_mean:.2f}, std={btc_mid_std:.2f}")

    return model, feature_cols, vol_threshold, btc_mid_mean, btc_mid_std


def save_model(
    model: xgb.Booster,
    feature_cols: list[str],
    vol_threshold: float | None,
    btc_mid_mean: float = 0.0,
    btc_mid_std: float = 1.0,
):
    """Save model and metadata to disk."""
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(MODEL_PATH))

    meta = {
        "feature_cols": feature_cols,
        "vol_threshold": vol_threshold,
        "observation_windows": OBSERVATION_WINDOWS,
        "confidence_threshold": 0.72,
        "model_path": str(MODEL_PATH),
        "btc_mid_mean": btc_mid_mean,
        "btc_mid_std": btc_mid_std,
    }
    META_PATH.write_text(json.dumps(meta, indent=2))
    print(f"  Saved model to {MODEL_PATH} and metadata to {META_PATH}")


def load_model() -> tuple[xgb.Booster, list[str], float | None, float, float]:
    """Load model and metadata from disk."""
    model = xgb.Booster()
    model.load_model(str(MODEL_PATH))

    meta = json.loads(META_PATH.read_text())
    feature_cols = meta["feature_cols"]
    vol_threshold = meta.get("vol_threshold")
    btc_mid_mean = meta.get("btc_mid_mean", 0.0)
    btc_mid_std = meta.get("btc_mid_std", 1.0)

    print(f"  Loaded model from {MODEL_PATH} ({len(feature_cols)} features)")
    print(f"  Vol threshold: {vol_threshold}")
    print(f"  BTC stats: mean={btc_mid_mean:.2f}, std={btc_mid_std:.2f}")
    return model, feature_cols, vol_threshold, btc_mid_mean, btc_mid_std


def snapshots_to_dataframe(snapshots: list[dict], market_open_ms: int) -> pd.DataFrame:
    """Convert raw orderbook snapshots from the TS strategy into a DataFrame
    matching the format expected by extract_market_features().

    Each snapshot has: timestamp_ms, mid_price, spread, book_imbalance,
    bid_prices[], bid_sizes[], ask_prices[], ask_sizes[]
    """
    rows = []
    for snap in snapshots:
        row = {
            "exchange_timestamp": snap["timestamp_ms"],
            "sec": int((snap["timestamp_ms"] - market_open_ms) / 1000),
            "mid_price": snap["mid_price"],
            "spread": snap.get("spread", 0.0),
            "book_imbalance": snap.get("book_imbalance", 0.0),
            "asset_id": snap.get("asset_id", ""),
        }
        # Map bid/ask arrays to bid_price_1..N, bid_size_1..N columns
        for i, (p, s) in enumerate(
            zip(snap.get("bid_prices", []), snap.get("bid_sizes", [])), 1
        ):
            row[f"bid_price_{i}"] = p
            row[f"bid_size_{i}"] = s
        for i, (p, s) in enumerate(
            zip(snap.get("ask_prices", []), snap.get("ask_sizes", [])), 1
        ):
            row[f"ask_price_{i}"] = p
            row[f"ask_size_{i}"] = s

        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # Ensure sec column exists and is sorted
    df = df.sort_values("sec").reset_index(drop=True)
    # Deduplicate to last-per-second (same as resample_to_1s)
    df = df.groupby("sec").last().reset_index()
    return df


def build_btc_arrays(
    btc_prices: list[dict],
    btc_mid_mean: float,
    btc_mid_std: float,
) -> dict | None:
    """Convert BTC price array from TS strategy into the numpy arrays
    expected by extract_btc_features().

    IMPORTANT: Resamples to 1-second intervals to match training data.
    The live Binance bookTicker feed sends updates every ~10ms, but the
    training BTC data is exactly 1s. Without resampling, features like
    btc_vol_5m and btc_rvol_* would have wildly different distributions
    (different n in std * sqrt(n), different return magnitudes per step).

    btc_mid_mean/btc_mid_std: training-time statistics for z-score normalization.
    These MUST come from the training dataset, not live data.

    Each entry has: timestamp_ms, mid_price, spread (optional)
    """
    if not btc_prices or len(btc_prices) < 3:
        return None

    ts_raw = np.array([p["timestamp_ms"] for p in btc_prices], dtype=np.int64)
    mids_raw = np.array([p["mid_price"] for p in btc_prices], dtype=np.float64)
    spreads_raw = np.array([p.get("spread", 0.0) for p in btc_prices], dtype=np.float64)

    # Sort by timestamp
    order = np.argsort(ts_raw)
    ts_raw = ts_raw[order]
    mids_raw = mids_raw[order]
    spreads_raw = spreads_raw[order]

    # Resample to 1-second bars (last value per second), matching training data
    # Bucket by flooring to nearest 1000ms
    sec_buckets = ts_raw // 1000
    unique_secs, last_indices = np.unique(sec_buckets, return_index=False), None
    # For each unique second, take the last data point
    _, last_idx = np.unique(sec_buckets[::-1], return_index=True)
    last_idx = len(sec_buckets) - 1 - last_idx  # reverse back to forward indices
    last_idx = np.sort(last_idx)

    ts = ts_raw[last_idx]
    mids = mids_raw[last_idx]
    spreads = spreads_raw[last_idx]

    if len(ts) < 3:
        return None

    # Compute log returns (now at 1s intervals, matching training)
    log_returns = np.zeros_like(mids)
    log_returns[1:] = np.diff(np.log(np.maximum(mids, 1e-10)))

    # Compute rolling volatilities
    rvols = {}
    for window_s, key in [(30, "rvol_30s"), (60, "rvol_60s"), (300, "rvol_300s")]:
        window_ms = window_s * 1000
        rvol = np.zeros(len(ts))
        for i in range(1, len(ts)):
            start = np.searchsorted(ts, ts[i] - window_ms, side="left")
            if i - start > 2:
                rvol[i] = np.std(log_returns[start:i+1])
        rvols[key] = rvol

    return {
        "ts": ts,
        "mids": mids,
        "spreads": spreads,
        "log_returns": log_returns,
        "rvols": rvols,
        "mid_mean": btc_mid_mean,
        "mid_std": btc_mid_std,
    }


def predict(
    model: xgb.Booster,
    feature_cols: list[str],
    vol_threshold: float | None,
    btc_mid_mean: float,
    btc_mid_std: float,
    request_data: dict,
) -> dict:
    """Run prediction on a single market.

    request_data format:
    {
        "market_open_ms": int,
        "observe_s": float (default 120),
        "up_snapshots": [{ timestamp_ms, mid_price, spread, book_imbalance,
                           bid_prices, bid_sizes, ask_prices, ask_sizes, asset_id }],
        "down_snapshots": [...],
        "btc_prices": [{ timestamp_ms, mid_price, spread? }]
    }
    """
    market_open_ms = request_data["market_open_ms"]
    observe_s = request_data.get("observe_s", 120.0)

    up_df = snapshots_to_dataframe(request_data.get("up_snapshots", []), market_open_ms)
    down_df = snapshots_to_dataframe(request_data.get("down_snapshots", []), market_open_ms)

    if up_df.empty or down_df.empty:
        return {
            "error": "insufficient_data",
            "prediction": 0.5,
            "should_trade": False,
            "reason": "No orderbook snapshots available",
        }

    btc_arrays = build_btc_arrays(
        request_data.get("btc_prices", []), btc_mid_mean, btc_mid_std
    )

    features = extract_market_features(
        up_df, down_df, market_open_ms, observe_s,
        btc_arrays=btc_arrays,
        windows=OBSERVATION_WINDOWS,
    )

    if features is None:
        return {
            "error": "feature_extraction_failed",
            "prediction": 0.5,
            "should_trade": False,
            "reason": "Feature extraction returned None",
        }

    # Extract metadata before building feature vector
    leader_label = features.pop("_leader_label", "Up")
    leader_asset_id = features.pop("_leader_asset_id", "")

    # Build feature vector in the exact order the model expects
    fvec = np.zeros(len(feature_cols), dtype=np.float32)
    for i, col in enumerate(feature_cols):
        val = features.get(col, 0.0)
        if val is None or (isinstance(val, float) and not np.isfinite(val)):
            val = 0.0
        fvec[i] = float(val)

    dmat = xgb.DMatrix(fvec.reshape(1, -1), feature_names=feature_cols)
    pred = float(model.predict(dmat)[0])

    # Vol filter removed in v3.2 — analysis showed it filters out winners
    # with negligible WR/ROI improvement. The model handles low-vol regimes
    # well on its own with depth=2 trees.
    btc_vol_5m = features.get("btc_vol_5m", 0.0)
    vol_pass = True  # always pass — kept for metadata logging compatibility

    confidence_threshold = 0.72
    should_trade = pred >= confidence_threshold

    # Determine which token to buy
    # The model predicts P(leader wins). Leader is whichever token has mid >= 0.50.
    # If pred >= threshold, buy the leader token.
    buy_token = "up" if leader_label == "Up" else "down"

    # Entry price: leader mid at observation time
    latest_w = max(w for w in OBSERVATION_WINDOWS if w <= observe_s)
    entry_price = features.get(f"w{latest_w}_mid", 0.0)

    return {
        "prediction": pred,
        "should_trade": should_trade,
        "buy_token": buy_token,
        "leader_label": leader_label,
        "leader_asset_id": leader_asset_id,
        "entry_price": entry_price,
        "btc_vol_5m": btc_vol_5m,
        "vol_threshold": vol_threshold,
        "vol_pass": vol_pass,
        "confidence_threshold": confidence_threshold,
    }


class ModelHandler(BaseHTTPRequestHandler):
    """HTTP request handler for model predictions."""

    # Class-level references set at startup
    model: xgb.Booster
    feature_cols: list[str]
    vol_threshold: float | None
    btc_mid_mean: float
    btc_mid_std: float

    def do_GET(self):
        if self.path == "/health":
            self._respond(200, {"status": "ok", "features": len(self.feature_cols)})
        else:
            self._respond(404, {"error": "not_found"})

    def do_POST(self):
        if self.path == "/predict":
            try:
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length)
                request_data = json.loads(body)
                result = predict(
                    self.model, self.feature_cols, self.vol_threshold,
                    self.btc_mid_mean, self.btc_mid_std, request_data,
                )
                self._respond(200, result)
            except json.JSONDecodeError as e:
                self._respond(400, {"error": "invalid_json", "detail": str(e)})
            except Exception as e:
                print(f"[ModelServer] Prediction error: {e}")
                import traceback
                traceback.print_exc()
                self._respond(500, {"error": "prediction_failed", "detail": str(e)})
        else:
            self._respond(404, {"error": "not_found"})

    def _respond(self, status: int, data: dict):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        """Suppress default request logging (too noisy)."""
        pass


def main():
    parser = argparse.ArgumentParser(description="XGBoost v3 model server")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--model-path",
        default=None,
        help="Path to pre-trained model JSON. Defaults to data/xgb_model.json.",
    )
    parser.add_argument(
        "--meta-path",
        default=None,
        help="Path to model metadata JSON. Defaults to data/xgb_model_meta.json.",
    )
    parser.add_argument("--train", action="store_true",
                        help="Force retrain even if saved model exists")
    parser.add_argument(
        "--feature-cache",
        default=None,
        help="Path to feature cache parquet (for training). Defaults to data/xgb_features_v3.parquet.",
    )
    parser.add_argument(
        "--spot-quotes",
        default=None,
        help="Path to spot price quotes parquet (for z-score stats). Defaults to data/btc_quotes/btcusdt_quotes.parquet.",
    )
    args = parser.parse_args()

    # Override global paths if provided
    global MODEL_PATH, META_PATH, FEATURE_CACHE_PATH
    if args.model_path:
        MODEL_PATH = Path(args.model_path)
    if args.meta_path:
        META_PATH = Path(args.meta_path)
    if args.feature_cache:
        FEATURE_CACHE_PATH = Path(args.feature_cache)
    global SPOT_QUOTES_PATH
    if args.spot_quotes:
        SPOT_QUOTES_PATH = Path(args.spot_quotes)

    # Train or load model
    if MODEL_PATH.exists() and META_PATH.exists() and not args.train:
        print(f"Loading saved model from {MODEL_PATH}...")
        model, feature_cols, vol_threshold, btc_mid_mean, btc_mid_std = load_model()
    elif not args.train:
        print(f"Model not found at {MODEL_PATH}, training from scratch...")
        model, feature_cols, vol_threshold, btc_mid_mean, btc_mid_std = train_production_model()
        save_model(model, feature_cols, vol_threshold, btc_mid_mean, btc_mid_std)
    else:
        print("Training production model on all data...")
        model, feature_cols, vol_threshold, btc_mid_mean, btc_mid_std = train_production_model()
        save_model(model, feature_cols, vol_threshold, btc_mid_mean, btc_mid_std)

    # Set class-level references for the handler
    ModelHandler.model = model
    ModelHandler.feature_cols = feature_cols
    ModelHandler.vol_threshold = vol_threshold
    ModelHandler.btc_mid_mean = btc_mid_mean
    ModelHandler.btc_mid_std = btc_mid_std

    server = HTTPServer((args.host, args.port), ModelHandler)
    print(f"\nModel server running on http://{args.host}:{args.port}")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Vol filter: disabled (v3.2)")
    print(f"  Endpoints: POST /predict, GET /health")
    print(f"  Press Ctrl+C to stop\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.server_close()


if __name__ == "__main__":
    main()
