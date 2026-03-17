"""Preprocess features → daily parquets for walk-forward training (v3).

Two variants:
  A) Orderbook 1-min resampled + derivatives  → directional_features_v3_ob
  B) OHLCV candles + derivatives              → directional_features_v3_ohlcv

Each variant preprocesses once, producing daily parquets with all features +
triple-barrier labels. Training then loads precomputed parquets directly.

Usage:
    # Both variants (default):
    python -m pipelines.preprocess_features

    # Orderbook only:
    python -m pipelines.preprocess_features --variant ob

    # OHLCV only:
    python -m pipelines.preprocess_features --variant ohlcv

    # Custom dirs:
    python -m pipelines.preprocess_features --variant ohlcv --output-dir data/custom_features
"""

import argparse
import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd

from strategies.directional.config import DirectionalConfig
from strategies.directional.derivatives_features import (
    build_derivatives_features,
    load_derivatives_day,
)
from strategies.directional.features import build_features, resample_to_1min
from strategies.directional.labels import compute_triple_barrier_labels
from strategies.directional.ohlcv_features import (
    compute_ohlcv_features,
    load_candles_day,
)
from strategies.directional.cross_asset_features import (
    compute_cross_asset_features,
    compute_cross_derivatives_features,
    load_alt_klines_day,
    merge_cross_asset_features,
)

# Default directories
DEFAULT_OB_DIR = "data/bybit_orderbook_raw"
DEFAULT_TRADES_DIR = "data/bybit_trades_200ms"
DEFAULT_CANDLES_DIR = "data/cryptolake"
DEFAULT_DERIV_DIR = "data/cryptolake"
DEFAULT_OUT_OB = "data/directional_features_v3_ob"
DEFAULT_OUT_OHLCV = "data/directional_features_v3_ohlcv"

# Backward overlap for rolling window warm-up
OVERLAP_DAYS_BACK = 2
# Forward overlap for labels (need max_barrier_horizon bars of future data)
# At 1-min bars with 60-bar horizon, 1 day gives plenty of headroom
OVERLAP_DAYS_FWD = 1


def _load_alt_klines_days(
    dates: list[str],
    asset: str,
    data_dir: Path,
) -> pd.DataFrame | None:
    """Load alt klines for an explicit list of dates."""
    frames = []
    for d in dates:
        df = load_alt_klines_day(d, asset, data_dir)
        if df is not None:
            frames.append(df)
    if not frames:
        return None
    result = pd.concat(frames, ignore_index=True)
    return result.sort_values("timestamp_ms").reset_index(drop=True)


def load_parquets(directory: Path, dates: list[str]) -> pd.DataFrame:
    """Load daily parquets for given dates into one DataFrame."""
    frames = []
    for date_str in dates:
        path = directory / f"{date_str}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        if df.empty:
            continue
        df["_date"] = date_str
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    result = pd.concat(frames, ignore_index=True)
    result = result.sort_values("timestamp_ms").reset_index(drop=True)
    return result


def load_candles_multi_day(
    dates: list[str],
    candles_dir: Path,
    exchange: str = "BINANCE_FUTURES",
    symbol: str = "BTC-USDT-PERP",
) -> pd.DataFrame:
    """Load multiple days of candle data with _date column preserved."""
    frames = []
    for date_str in dates:
        df = load_candles_day(date_str, candles_dir, exchange, symbol)
        if df is not None and not df.empty:
            df["_date"] = date_str
            frames.append(df)

    if not frames:
        return pd.DataFrame()

    result = pd.concat(frames, ignore_index=True)
    result = result.sort_values("timestamp").reset_index(drop=True)
    return result


def load_derivatives_multi_day(
    dates: list[str],
    deriv_dir: Path,
    exchange: str = "BINANCE_FUTURES",
    symbol: str = "BTC-USDT-PERP",
) -> dict[str, pd.DataFrame | None]:
    """Load multiple days of derivatives data, concatenated by type."""
    frames = {"funding": [], "open_interest": [], "liquidations": []}

    for date_str in dates:
        day_data = load_derivatives_day(date_str, deriv_dir, exchange, symbol)
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


def preprocess_ob_chunk(
    target_dates: list[str],
    all_chunk_dates: list[str],
    cfg: DirectionalConfig,
    ob_dir: Path,
    trades_dir: Path,
    deriv_dir: Path | None,
    output_dir: Path,
    resume: bool,
) -> int:
    """Process one chunk of orderbook variant: resample → features → derivatives → labels."""
    target_set = set(target_dates)

    # Load orderbook
    ob = load_parquets(ob_dir, all_chunk_dates)
    if ob.empty:
        print("    No orderbook data loaded, skipping chunk.")
        return 0

    ob_dates = ob["_date"].values
    ob_no_date = ob.drop(columns=["_date"])
    del ob
    gc.collect()

    ob_gb = ob_no_date.memory_usage(deep=True).sum() / 1e9
    print(f"    Loaded {len(ob_no_date):,} snapshots ({ob_gb:.1f} GB)")

    # Load trades
    tr = load_parquets(trades_dir, all_chunk_dates)
    if not tr.empty:
        tr = tr.drop(columns=["_date"])
        print(f"    Loaded {len(tr):,} trade bins")
    else:
        tr = None
        print("    No trade data, proceeding without trade features.")

    # Resample to 1-minute
    print("    Resampling to 1-minute bars...")
    ob_1m, tr_1m = resample_to_1min(ob_no_date, tr)
    print(f"    → {len(ob_1m):,} bars")

    del ob_no_date, tr
    gc.collect()

    # Compute orderbook features
    print("    Computing features...")
    features = build_features(ob_1m, tr_1m, cfg)

    del ob_1m, tr_1m
    gc.collect()

    # Merge derivatives
    if deriv_dir is not None:
        print("    Loading derivatives...")
        deriv_data = load_derivatives_multi_day(all_chunk_dates, deriv_dir)
        deriv_feats = build_derivatives_features(deriv_data)
        if deriv_feats is not None:
            from strategies.directional.derivatives_features import merge_with_orderbook_features
            n_before = len(features.columns)
            features = merge_with_orderbook_features(features, deriv_feats)
            n_after = len(features.columns)
            print(f"    Merged {n_after - n_before} derivatives features")
        del deriv_data, deriv_feats
        gc.collect()

    # Derive _date from bar timestamps (after all merges to preserve alignment)
    features["_date"] = pd.to_datetime(
        features["timestamp_ms"], unit="ms", utc=True
    ).dt.strftime("%Y-%m-%d").values

    # Compute labels
    print("    Computing triple-barrier labels...")
    labels = compute_triple_barrier_labels(features["mid_price"].values, cfg)
    features["label_tb"] = labels["label_tb"].values
    features["holding_period"] = labels["holding_period"].values
    features["barrier_type"] = labels["barrier_type"].values
    del labels

    # Save target dates only
    saved = 0
    for date_str, group in features.groupby("_date", sort=True):
        if date_str not in target_set:
            continue
        if resume and (output_dir / f"{date_str}.parquet").exists():
            continue
        day_df = group.drop(columns=["_date"]).reset_index(drop=True)
        day_df.to_parquet(output_dir / f"{date_str}.parquet", index=False)
        saved += 1

    del features
    gc.collect()
    return saved


def preprocess_ohlcv_chunk(
    target_dates: list[str],
    all_chunk_dates: list[str],
    cfg: DirectionalConfig,
    candles_dir: Path,
    deriv_dir: Path | None,
    output_dir: Path,
    resume: bool,
    alt_data_dir: Path | None = None,
    symbol: str = "BTC-USDT-PERP",
) -> int:
    """Process one chunk of OHLCV variant: candles → features → derivatives → cross-asset → labels."""
    target_set = set(target_dates)

    # Load candles for the target symbol
    candles = load_candles_multi_day(all_chunk_dates, candles_dir, symbol=symbol)
    if candles.empty:
        print("    No candle data loaded, skipping chunk.")
        return 0

    candle_dates = candles["_date"].values
    candles_no_date = candles.drop(columns=["_date"])
    del candles
    gc.collect()

    print(f"    Loaded {len(candles_no_date):,} candle bars")

    # Compute OHLCV features
    print("    Computing OHLCV features...")
    features = compute_ohlcv_features(candles_no_date, cfg)

    # Save BTC timestamps and close for cross-asset features
    btc_timestamps = features["timestamp_ms"].values
    btc_close = features["mid_price"].values

    del candles_no_date
    gc.collect()

    # Merge derivatives
    deriv_feats = None
    if deriv_dir is not None:
        print(f"    Loading {symbol} derivatives...")
        deriv_data = load_derivatives_multi_day(all_chunk_dates, deriv_dir, symbol=symbol)
        deriv_feats = build_derivatives_features(deriv_data)
        if deriv_feats is not None:
            from strategies.directional.derivatives_features import merge_with_orderbook_features
            n_before = len(features.columns)
            features = merge_with_orderbook_features(features, deriv_feats)
            n_after = len(features.columns)
            print(f"    Merged {n_after - n_before} derivatives features")
        del deriv_data
        gc.collect()

    # Cross-asset features (only for BTC — alts would need BTC as reference, TODO)
    is_btc = symbol.startswith("BTC")
    if alt_data_dir is not None and is_btc:
        # Load alt klines for all chunk dates
        alt_dfs = {}
        for asset in ["eth", "sol", "xrp"]:
            adf = _load_alt_klines_days(all_chunk_dates, asset, alt_data_dir)
            if adf is not None and not adf.empty:
                alt_dfs[asset] = adf

        if alt_dfs:
            print(f"    Computing cross-asset features ({', '.join(alt_dfs.keys())})...")
            cross_feats = compute_cross_asset_features(
                btc_timestamps, btc_close, alt_dfs,
            )
            features = merge_cross_asset_features(features, cross_feats)
            del cross_feats
        else:
            print("    No alt kline data found, skipping cross-asset features.")

        del alt_dfs
        gc.collect()

        # Cross-asset derivatives (ETH funding/OI)
        if deriv_dir is not None and "funding_rate" in features.columns:
            eth_deriv_data = load_derivatives_multi_day(
                all_chunk_dates, deriv_dir,
                symbol="ETH-USDT-PERP",
            )
            eth_funding_feats = None
            eth_oi_feats = None

            from strategies.directional.derivatives_features import (
                compute_funding_features,
                compute_oi_features,
                _ns_to_ms,
            )

            if eth_deriv_data.get("funding") is not None:
                eth_fund = compute_funding_features(eth_deriv_data["funding"])
                # Merge ETH funding onto BTC grid
                eth_fund_merged = pd.merge_asof(
                    pd.DataFrame({"timestamp_ms": features["timestamp_ms"]}),
                    eth_fund[["timestamp_ms", "funding_rate"]].sort_values("timestamp_ms"),
                    on="timestamp_ms",
                    direction="backward",
                    tolerance=60_000,
                )
                eth_funding_feats = eth_fund_merged["funding_rate"].fillna(0)

            if eth_deriv_data.get("open_interest") is not None:
                eth_oi = compute_oi_features(eth_deriv_data["open_interest"])
                if "delta_oi_15m" in eth_oi.columns:
                    eth_oi_merged = pd.merge_asof(
                        pd.DataFrame({"timestamp_ms": features["timestamp_ms"]}),
                        eth_oi[["timestamp_ms", "delta_oi_15m"]].sort_values("timestamp_ms"),
                        on="timestamp_ms",
                        direction="backward",
                        tolerance=60_000,
                    )
                    eth_oi_feats = eth_oi_merged["delta_oi_15m"].fillna(0)

            cross_deriv = compute_cross_derivatives_features(
                btc_funding=features["funding_rate"],
                btc_oi_delta=features.get("delta_oi_15m"),
                eth_funding=eth_funding_feats,
                eth_oi_delta=eth_oi_feats,
                timestamps_ms=features["timestamp_ms"].values,
            )
            features = merge_cross_asset_features(features, cross_deriv)
            del eth_deriv_data, cross_deriv
            gc.collect()

    del deriv_feats
    gc.collect()

    # Derive _date from bar timestamps (after all merges to preserve alignment)
    features["_date"] = pd.to_datetime(
        features["timestamp_ms"], unit="ms", utc=True
    ).dt.strftime("%Y-%m-%d").values

    # Compute labels
    print("    Computing triple-barrier labels...")
    labels = compute_triple_barrier_labels(features["mid_price"].values, cfg)
    features["label_tb"] = labels["label_tb"].values
    features["holding_period"] = labels["holding_period"].values
    features["barrier_type"] = labels["barrier_type"].values
    del labels

    # Save target dates only
    saved = 0
    for date_str, group in features.groupby("_date", sort=True):
        if date_str not in target_set:
            continue
        if resume and (output_dir / f"{date_str}.parquet").exists():
            continue
        day_df = group.drop(columns=["_date"]).reset_index(drop=True)
        day_df.to_parquet(output_dir / f"{date_str}.parquet", index=False)
        saved += 1

    del features
    gc.collect()
    return saved


def run_preprocessing(
    variant: str,
    dates: list[str],
    cfg: DirectionalConfig,
    ob_dir: Path | None,
    trades_dir: Path | None,
    candles_dir: Path | None,
    deriv_dir: Path | None,
    output_dir: Path,
    resume: bool,
    chunk_days: int,
    alt_data_dir: Path | None = None,
    symbol: str = "BTC-USDT-PERP",
):
    """Run preprocessing for one variant using overlapping chunks."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter to dates with source data
    if variant == "ob":
        available = {f.stem for f in ob_dir.glob("*.parquet")}
    else:
        # OHLCV: check candle files under candles_dir/candles/exchange/symbol/
        candle_path = candles_dir / "candles" / "exchange=BINANCE_FUTURES" / f"symbol={symbol}"
        available = {f.stem for f in candle_path.glob("*.parquet")} if candle_path.exists() else set()

    variant_dates = [d for d in dates if d in available]
    if not variant_dates:
        print(f"  No source data found for variant '{variant}'")
        return

    if resume:
        missing = [d for d in variant_dates if not (output_dir / f"{d}.parquet").exists()]
        if not missing:
            print(f"  All {len(variant_dates)} days already exist, skipping.")
            return
        print(f"  {len(missing)}/{len(variant_dates)} days need processing.")

    t0 = time.time()
    total_saved = 0

    i = 0
    while i < len(variant_dates):
        chunk_end = min(i + chunk_days, len(variant_dates))
        target_dates = variant_dates[i:chunk_end]

        # Build overlap dates
        overlap_start = max(0, i - OVERLAP_DAYS_BACK)
        overlap_back = variant_dates[overlap_start:i]
        fwd_end = min(chunk_end + OVERLAP_DAYS_FWD, len(variant_dates))
        overlap_fwd = variant_dates[chunk_end:fwd_end]
        all_chunk_dates = overlap_back + target_dates + overlap_fwd

        # Skip if all target dates exist
        if resume:
            missing_in_chunk = [d for d in target_dates
                                if not (output_dir / f"{d}.parquet").exists()]
            if not missing_in_chunk:
                print(f"  Chunk {target_dates[0]}→{target_dates[-1]}: all exist, skipping.")
                i = chunk_end
                continue

        print(f"\n  Chunk: {target_dates[0]} → {target_dates[-1]} "
              f"({len(target_dates)} days + {len(overlap_back)}b/{len(overlap_fwd)}f overlap)")

        if variant == "ob":
            saved = preprocess_ob_chunk(
                target_dates, all_chunk_dates, cfg,
                ob_dir, trades_dir, deriv_dir, output_dir, resume,
            )
        else:
            saved = preprocess_ohlcv_chunk(
                target_dates, all_chunk_dates, cfg,
                candles_dir, deriv_dir, output_dir, resume,
                alt_data_dir=alt_data_dir,
                symbol=symbol,
            )

        total_saved += saved
        elapsed = time.time() - t0
        pct = chunk_end / len(variant_dates) * 100
        print(f"    Saved {saved} days ({pct:.0f}% complete, {elapsed:.0f}s elapsed)")

        i = chunk_end

    elapsed = time.time() - t0
    print(f"\n  {total_saved} days saved in {elapsed:.0f}s")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess features for BTC directional classifier (v3)"
    )
    parser.add_argument(
        "--variant", choices=["ob", "ohlcv", "both"], default="both",
        help="Feature variant: ob (orderbook+deriv), ohlcv (candles+deriv), both",
    )
    parser.add_argument("--start", default=None, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="End date YYYY-MM-DD exclusive")
    parser.add_argument("--ob-dir", default=DEFAULT_OB_DIR)
    parser.add_argument("--trades-dir", default=DEFAULT_TRADES_DIR)
    parser.add_argument("--candles-dir", default=DEFAULT_CANDLES_DIR)
    parser.add_argument("--derivatives-dir", default=DEFAULT_DERIV_DIR)
    parser.add_argument("--output-dir", default=None,
                        help="Override output dir (default: per-variant)")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", action="store_false", dest="resume")
    parser.add_argument("--alt-data-dir", default="data",
                        help="Base dir for alt kline daily parquets (contains {asset}_quotes_1m/)")
    parser.add_argument("--chunk-days", type=int, default=30)
    parser.add_argument("--symbol", default="BTC-USDT-PERP",
                        help="Binance Futures symbol (e.g. WIF-USDT-PERP)")
    args = parser.parse_args()

    symbol = args.symbol
    cfg = DirectionalConfig.config_1m()

    # Build date list from all available sources
    all_dates = set()
    ob_dir = Path(args.ob_dir)
    if ob_dir.exists() and symbol == "BTC-USDT-PERP":
        all_dates.update(f.stem for f in ob_dir.glob("*.parquet"))
    candle_path = Path(args.candles_dir) / "candles" / "exchange=BINANCE_FUTURES" / f"symbol={symbol}"
    if candle_path.exists():
        all_dates.update(f.stem for f in candle_path.glob("*.parquet"))

    if not all_dates:
        print("No data files found")
        return

    dates = sorted(all_dates)
    if args.start:
        dates = [d for d in dates if d >= args.start]
    if args.end:
        dates = [d for d in dates if d < args.end]

    print(f"Date range: {dates[0]} → {dates[-1]} ({len(dates)} days)")

    deriv_dir = Path(args.derivatives_dir)
    if not deriv_dir.exists():
        print(f"Warning: derivatives dir {deriv_dir} not found, proceeding without")
        deriv_dir = None

    alt_data_dir = Path(args.alt_data_dir)
    is_btc = symbol.startswith("BTC")
    if is_btc:
        # Check if any alt 1m data exists
        has_alt = any(
            (alt_data_dir / f"{a}_quotes_1m").exists()
            for a in ["eth", "sol", "xrp"]
        )
        if not has_alt:
            print("Warning: no alt kline data found, proceeding without cross-asset features")
            alt_data_dir = None
    else:
        # Non-BTC: skip cross-asset features for now
        alt_data_dir = None

    variants = []
    if args.variant in ("ob", "both"):
        variants.append("ob")
    if args.variant in ("ohlcv", "both"):
        variants.append("ohlcv")

    # For non-BTC symbols, derive a short name for output dirs
    sym_short = symbol.split("-")[0].lower()  # e.g. "WIF-USDT-PERP" → "wif"

    for variant in variants:
        if args.output_dir:
            out_dir = Path(args.output_dir)
        else:
            base = DEFAULT_OUT_OB if variant == "ob" else DEFAULT_OUT_OHLCV
            # Append symbol suffix for non-BTC
            out_dir = Path(base) if is_btc else Path(f"{base}_{sym_short}")

        print(f"\n{'='*60}")
        print(f"Preprocessing: {variant.upper()} variant ({symbol}) → {out_dir}")
        print(f"{'='*60}")

        run_preprocessing(
            variant=variant,
            dates=dates,
            cfg=cfg,
            ob_dir=ob_dir if variant == "ob" else None,
            trades_dir=Path(args.trades_dir) if variant == "ob" else None,
            candles_dir=Path(args.candles_dir) if variant == "ohlcv" else None,
            deriv_dir=deriv_dir,
            output_dir=out_dir,
            resume=args.resume,
            chunk_days=args.chunk_days,
            alt_data_dir=alt_data_dir if variant == "ohlcv" else None,
            symbol=symbol,
        )

    print("\nPreprocessing complete.")


if __name__ == "__main__":
    main()
