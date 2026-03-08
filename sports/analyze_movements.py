"""
Analyze collected sports market price snapshots.

Computes volatility, spreads, liquidity, and identifies sudden price movements.

Usage:
    python sports/analyze_movements.py [--data-dir data/sports/snapshots] [--min-samples 10]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "sports" / "snapshots"


def load_all_snapshots(data_dir: Path) -> pd.DataFrame:
    """Load and concatenate all parquet snapshot files."""
    files = sorted(data_dir.glob("*.parquet"))
    if not files:
        print(f"No parquet files found in {data_dir}")
        sys.exit(1)

    dfs = []
    for f in files:
        df = pd.read_parquet(f)
        df["_source_file"] = f.name
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    if "timestamp" in combined.columns:
        combined["timestamp"] = pd.to_datetime(combined["timestamp"])
        combined = combined.sort_values("timestamp")

    print(f"Loaded {len(combined):,} snapshots from {len(files)} files")
    print(f"Date range: {combined['timestamp'].min()} to {combined['timestamp'].max()}")
    print()
    return combined


def market_summary(df: pd.DataFrame, min_samples: int = 10) -> pd.DataFrame:
    """Per-market summary statistics."""
    groups = df.groupby(["market_id", "outcome"])

    records = []
    for (market_id, outcome), g in groups:
        if len(g) < min_samples:
            continue

        mid = g["mid"].dropna()
        spread = g["spread"].dropna()

        if len(mid) < 2:
            continue

        # Price returns (absolute changes between snapshots)
        returns = mid.diff().dropna()

        records.append({
            "market_id": market_id,
            "outcome": outcome,
            "sport": g["sport"].iloc[0],
            "title": g["title"].iloc[0][:60],
            "n_snapshots": len(g),
            "mid_mean": round(mid.mean(), 4),
            "mid_std": round(mid.std(), 4),
            "mid_min": round(mid.min(), 4),
            "mid_max": round(mid.max(), 4),
            "mid_range": round(mid.max() - mid.min(), 4),
            "avg_spread": round(spread.mean(), 4) if len(spread) > 0 else None,
            "avg_bid_depth": round(g["bid_depth"].mean(), 2),
            "avg_ask_depth": round(g["ask_depth"].mean(), 2),
            "max_abs_return": round(returns.abs().max(), 4) if len(returns) > 0 else None,
            "volatility": round(returns.std(), 6) if len(returns) > 1 else None,
        })

    return pd.DataFrame(records)


def detect_large_moves(df: pd.DataFrame, threshold: float = 0.02) -> pd.DataFrame:
    """
    Find snapshots where mid moved more than `threshold` since previous snapshot
    for the same asset.
    """
    df = df.sort_values(["asset_id", "timestamp"])
    df["mid_prev"] = df.groupby("asset_id")["mid"].shift(1)
    df["mid_change"] = (df["mid"] - df["mid_prev"]).abs()

    moves = df[df["mid_change"] >= threshold].copy()
    moves = moves.sort_values("mid_change", ascending=False)

    return moves[["timestamp", "asset_id", "market_id", "sport", "title", "outcome",
                   "mid_prev", "mid", "mid_change", "spread", "bid_depth", "ask_depth"]]


def print_report(summary: pd.DataFrame, large_moves: pd.DataFrame):
    """Print analysis report to stdout."""
    print("=" * 80)
    print("POLYMARKET SPORTS — PRICE ANALYSIS REPORT")
    print("=" * 80)

    # ── Overview by sport ──
    print("\n--- Markets by Sport ---")
    by_sport = summary.groupby("sport").agg(
        n_markets=("market_id", "nunique"),
        avg_spread=("avg_spread", "mean"),
        avg_volatility=("volatility", "mean"),
        avg_bid_depth=("avg_bid_depth", "mean"),
    ).round(4)
    print(by_sport.to_string())

    # ── Most volatile markets ──
    print("\n--- Top 15 Most Volatile Markets ---")
    top_vol = summary.nlargest(15, "volatility")
    cols = ["sport", "title", "outcome", "mid_mean", "mid_range", "volatility",
            "avg_spread", "avg_bid_depth", "n_snapshots"]
    print(top_vol[cols].to_string(index=False))

    # ── Widest spreads ──
    print("\n--- Top 10 Widest Average Spreads ---")
    top_spread = summary.nlargest(10, "avg_spread")
    cols = ["sport", "title", "outcome", "avg_spread", "avg_bid_depth", "avg_ask_depth"]
    print(top_spread[cols].to_string(index=False))

    # ── Large moves ──
    print(f"\n--- Large Price Moves (>= 2 cent jump between snapshots) ---")
    if len(large_moves) == 0:
        print("  None detected.")
    else:
        print(f"  {len(large_moves)} large moves found")
        print()
        show = large_moves.head(20)
        cols = ["timestamp", "sport", "title", "outcome", "mid_prev", "mid",
                "mid_change", "spread"]
        print(show[cols].to_string(index=False))

    # ── Liquidity ──
    print("\n--- Liquidity Distribution ---")
    for col in ["bid_depth", "ask_depth"]:
        avg = col.replace("_", " ").title()
        vals = summary[f"avg_{col}"]
        print(f"  {avg}: median={vals.median():.1f}  mean={vals.mean():.1f}  "
              f"p10={vals.quantile(0.1):.1f}  p90={vals.quantile(0.9):.1f}")

    print("\n" + "=" * 80)
    print(f"Total markets analyzed: {summary['market_id'].nunique()}")
    print(f"Total outcomes tracked: {len(summary)}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Analyze Polymarket sports price data")
    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR),
                        help="Directory with snapshot parquet files")
    parser.add_argument("--min-samples", type=int, default=10,
                        help="Minimum snapshots per market to include (default: 10)")
    parser.add_argument("--move-threshold", type=float, default=0.02,
                        help="Threshold for 'large move' detection (default: 0.02 = 2 cents)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        print("Run monitor_prices.py first to collect data.")
        sys.exit(1)

    df = load_all_snapshots(data_dir)
    summary = market_summary(df, min_samples=args.min_samples)

    if summary.empty:
        print(f"No markets with >= {args.min_samples} snapshots. Collect more data.")
        sys.exit(0)

    large_moves = detect_large_moves(df, threshold=args.move_threshold)
    print_report(summary, large_moves)


if __name__ == "__main__":
    main()
