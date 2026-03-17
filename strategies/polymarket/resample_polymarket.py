"""Resample tick-level Polymarket orderbook to 1-minute bars per 5-min window.

Reads normalized Telonex parquets (one per market slug) and produces a single
DataFrame with one row per minute per market, containing Up token prices and
resolution outcomes.

Processes files one at a time to avoid OOM on large datasets.

Usage:
    python -m strategies.polymarket.resample_polymarket \
        --input-dirs data/telonex_btc_5m_mar7_17 data/sample_1k \
        --resolution-cache data/resolution_cache.json \
        --output data/polymarket_1m_bars.parquet
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def resample_one_slug(slug_df: pd.DataFrame, slug: str, resolutions: dict) -> pd.DataFrame | None:
    """Resample a single slug's tick data to 1-minute bars. Returns None if no outcome."""
    # Resolve outcome
    outcome = _parse_outcome(slug, resolutions)
    if np.isnan(outcome):
        return None

    if "token_label" not in slug_df.columns:
        return None

    up_df = slug_df[slug_df["token_label"] == "Up"]
    down_df = slug_df[slug_df["token_label"] == "Down"]

    if len(up_df) == 0:
        return None

    up_df = up_df.copy()
    down_df = down_df.copy()

    up_df["ts_ms"] = up_df["exchange_timestamp"].astype("int64")
    # Parse window start from slug: btc-updown-5m-{epoch_seconds}
    try:
        window_start_ms = int(slug.split("-")[-1]) * 1000
    except (ValueError, IndexError):
        return None

    up_df["minute_in_window"] = ((up_df["ts_ms"] - window_start_ms) // 60_000).clip(0, 4).astype(int)

    # Take first snapshot per minute (closest to minute boundary)
    up_first = (
        up_df.sort_values("ts_ms")
        .groupby("minute_in_window")
        .first()
        .reset_index()
    )

    row = pd.DataFrame({
        "slug": slug,
        "window_start_ms": window_start_ms,
        "minute_in_window": up_first["minute_in_window"],
        "up_mid": up_first["mid_price"].values,
        "up_bid": up_first["bid_price_1"].values,
        "up_ask": up_first["ask_price_1"].values,
        "up_spread": up_first["spread"].values,
    })

    # Down prices
    if len(down_df) > 0:
        down_df = down_df.copy()
        down_df["ts_ms"] = down_df["exchange_timestamp"].astype("int64")
        down_df["minute_in_window"] = ((down_df["ts_ms"] - window_start_ms) // 60_000).clip(0, 4).astype(int)
        down_first = (
            down_df.sort_values("ts_ms")
            .groupby("minute_in_window")
            .first()
            .reset_index()
        )
        down_cols = down_first[["minute_in_window", "mid_price", "bid_price_1", "ask_price_1"]].copy()
        down_cols.columns = ["minute_in_window", "down_mid", "down_bid", "down_ask"]
        row = row.merge(down_cols, on="minute_in_window", how="left")
    else:
        row["down_mid"] = np.nan
        row["down_bid"] = np.nan
        row["down_ask"] = np.nan

    row["outcome"] = outcome
    row["window_id"] = (window_start_ms // 300_000) * 300_000
    return row


def _parse_outcome(slug: str, resolutions: dict) -> float:
    """Parse outcome from resolution cache."""
    res = resolutions.get(slug)
    if not res:
        return np.nan
    if "winnerLabel" in res:
        return 1.0 if res["winnerLabel"] == "Up" else 0.0
    if "winningTokenId" in res and "clobTokenIds" in res:
        try:
            idx = res["clobTokenIds"].index(res["winningTokenId"])
            return 1.0 if res["outcomes"][idx] == "Up" else 0.0
        except (ValueError, KeyError):
            pass
    return np.nan


def process_individual_files(dir_path: Path, resolutions: dict) -> list[pd.DataFrame]:
    """Process a directory of per-slug parquet files (one file = one market)."""
    files = sorted(dir_path.glob("*.parquet"))
    results = []
    errors = 0
    for i, f in enumerate(files):
        slug = f.stem  # filename is slug.parquet
        try:
            df = pd.read_parquet(f)
            if len(df) == 0:
                continue
            bar = resample_one_slug(df, slug, resolutions)
            if bar is not None:
                results.append(bar)
        except Exception:
            errors += 1
            continue

        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(files)} files processed, {len(results)} with outcomes")

    print(f"  {dir_path.name}: {len(results)} markets resampled from {len(files)} files ({errors} errors)")
    return results


def process_combined_file(file_path: Path, resolutions: dict) -> list[pd.DataFrame]:
    """Process a single large parquet containing multiple slugs."""
    print(f"  Loading {file_path.name}...")
    df = pd.read_parquet(file_path)
    print(f"  Loaded {len(df):,} rows, processing by slug...")

    slugs = df["slug"].unique()
    results = []
    for i, slug in enumerate(slugs):
        slug_df = df[df["slug"] == slug]
        bar = resample_one_slug(slug_df, slug, resolutions)
        if bar is not None:
            results.append(bar)

        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(slugs)} slugs processed, {len(results)} with outcomes")

    print(f"  {file_path.name}: {len(results)} markets resampled from {len(slugs)} slugs")
    return results


def main():
    parser = argparse.ArgumentParser(description="Resample Polymarket orderbook to 1-minute bars")
    parser.add_argument("--input-dirs", nargs="+", required=True, help="Normalized Telonex data directories")
    parser.add_argument("--resolution-cache", default="data/resolution_cache.json")
    parser.add_argument("--output", default="data/polymarket_1m_bars.parquet")
    args = parser.parse_args()

    with open(args.resolution_cache) as f:
        resolutions = json.load(f)
    print(f"Loaded {len(resolutions)} resolutions")

    all_results = []
    for d in args.input_dirs:
        d = Path(d)
        if not d.exists():
            print(f"Warning: {d} does not exist, skipping")
            continue

        # Check if directory contains individual per-slug files or a single combined file
        parquets = list(d.glob("*.parquet"))
        if len(parquets) == 1 and not parquets[0].stem.startswith("btc-updown"):
            # Single combined file (like sample_1k)
            all_results.extend(process_combined_file(parquets[0], resolutions))
        else:
            # Individual per-slug files
            all_results.extend(process_individual_files(d, resolutions))

    if not all_results:
        print("No data produced!")
        return

    out = pd.concat(all_results, ignore_index=True)

    # Deduplicate by slug + minute_in_window (in case both dirs have same market)
    before = len(out)
    out = out.drop_duplicates(subset=["slug", "minute_in_window"], keep="first")
    if len(out) < before:
        print(f"  Deduped: {before} -> {len(out)} rows")

    out = out.sort_values(["window_start_ms", "minute_in_window"]).reset_index(drop=True)
    print(f"\nTotal: {len(out):,} minute-bars from {out['slug'].nunique():,} markets")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output, index=False)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
