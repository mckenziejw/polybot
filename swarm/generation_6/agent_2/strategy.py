"""
Orderbook Shape Convexity as Volatility-Regime Filter for Polymarket BTC 5m Up/Down.

Hypothesis: Concave depth curves (large size at levels 1-2, thinning at 3-5) indicate
low-uncertainty regime. Fade deviations by buying token whose mid dropped below its
6-snapshot rolling mean. Convex curves signal uncertainty -> skip.
"""

import pyarrow.parquet as pq
import pandas as pd
import numpy as np
import json
from pathlib import Path

SAMPLE_BOOK = "data/sample_1k/book_snapshots_sample_1k.parquet"
SAMPLE_RESOLUTION = "data/sample_1k/resolution_cache_sample.json"
OUTPUT_DIR = Path("/home/ubuntu/wss_test/swarm/generation_6/agent_2")

FEE_FUNC = lambda p: 0.0222 * p * 0.25 * (p * (1 - p)) ** 2
MAX_ENTRY_PRICE = 0.85
MIN_ORDER_SIZE = 5
ROLLING_WINDOW = 6
SUBSAMPLE = 100  # aggressive subsample: ~400 snapshots per token per market

COLS = ["slug", "token_label", "exchange_timestamp", "mid_price",
        "bid_size_1", "bid_size_2", "bid_size_3", "bid_size_4", "bid_size_5",
        "ask_size_1", "ask_size_2", "ask_size_3", "ask_size_4", "ask_size_5"]


def load_resolution_cache():
    """Load and resolve all winner labels."""
    with open(SAMPLE_RESOLUTION) as f:
        rc = json.load(f)

    # Read only mapping columns (small)
    pf = pq.ParquetFile(SAMPLE_BOOK)
    # Read first row group for mapping (all slugs should appear)
    mapping_chunks = []
    for i in range(pf.metadata.num_row_groups):
        rg = pf.read_row_group(i, columns=["slug", "asset_id", "token_label"]).to_pandas()
        mapping_chunks.append(rg.drop_duplicates(subset=["slug", "asset_id"]))

    mapping_df = pd.concat(mapping_chunks).drop_duplicates(subset=["slug", "asset_id"])

    lookup = {}
    for _, row in mapping_df.iterrows():
        lookup[(row["slug"], row["asset_id"])] = row["token_label"]

    resolved = {}
    for slug, info in rc.items():
        if "winnerLabel" in info:
            resolved[slug] = info["winnerLabel"]
        else:
            wid = str(info.get("winningTokenId") or info.get("winning_token"))
            label = lookup.get((slug, wid))
            if label:
                resolved[slug] = label
    return resolved


def compute_convexity_vec(sizes):
    """Second derivative of cumulative depth. sizes: (N, 5) -> (N,)."""
    cum = np.cumsum(sizes, axis=1)
    d2 = cum[:, 2:] - 2 * cum[:, 1:-1] + cum[:, :-2]
    return np.mean(d2, axis=1)


def load_subsampled_data():
    """Load data row-group by row-group, subsampling within each."""
    pf = pq.ParquetFile(SAMPLE_BOOK)
    chunks = []
    for i in range(pf.metadata.num_row_groups):
        rg = pf.read_row_group(i, columns=COLS).to_pandas()
        # Subsample: every Nth row (roughly, not perfect per-slug but good enough)
        chunks.append(rg.iloc[::SUBSAMPLE])
        del rg

    df = pd.concat(chunks, ignore_index=True)
    del chunks
    # Now sort properly per slug+token and re-index
    df = df.sort_values(["slug", "token_label", "exchange_timestamp"]).reset_index(drop=True)
    return df


def process_market(slug_df, slug, resolution, convexity_threshold):
    """Process one market, return first valid trade or None."""
    winner = resolution.get(slug)
    if winner is None:
        return None

    up = slug_df[slug_df["token_label"] == "Up"]
    dn = slug_df[slug_df["token_label"] == "Down"]

    if len(up) < ROLLING_WINDOW + 1 or len(dn) < ROLLING_WINDOW + 1:
        return None

    # Convexity for Up
    bid_sizes_up = up[["bid_size_1", "bid_size_2", "bid_size_3", "bid_size_4", "bid_size_5"]].values
    ask_sizes_up = up[["ask_size_1", "ask_size_2", "ask_size_3", "ask_size_4", "ask_size_5"]].values
    up_bid_conv = compute_convexity_vec(bid_sizes_up)
    up_ask_conv = compute_convexity_vec(ask_sizes_up)

    # Convexity for Down
    bid_sizes_dn = dn[["bid_size_1", "bid_size_2", "bid_size_3", "bid_size_4", "bid_size_5"]].values
    ask_sizes_dn = dn[["ask_size_1", "ask_size_2", "ask_size_3", "ask_size_4", "ask_size_5"]].values
    dn_bid_conv = compute_convexity_vec(bid_sizes_dn)
    dn_ask_conv = compute_convexity_vec(ask_sizes_dn)

    # Rolling mean
    up_mid = up["mid_price"].values
    dn_mid = dn["mid_price"].values
    up_roll = pd.Series(up_mid).rolling(ROLLING_WINDOW, min_periods=ROLLING_WINDOW).mean().values
    dn_roll = pd.Series(dn_mid).rolling(ROLLING_WINDOW, min_periods=ROLLING_WINDOW).mean().values

    n = min(len(up), len(dn))

    for i in range(ROLLING_WINDOW, n):
        if np.isnan(up_roll[i]) or np.isnan(dn_roll[i]):
            continue

        all_concave = (
            up_bid_conv[i] < convexity_threshold
            and up_ask_conv[i] < convexity_threshold
            and dn_bid_conv[i] < convexity_threshold
            and dn_ask_conv[i] < convexity_threshold
        )
        if not all_concave:
            continue

        up_below = up_mid[i] < up_roll[i]
        dn_below = dn_mid[i] < dn_roll[i]

        if up_below and not dn_below and up_mid[i] <= MAX_ENTRY_PRICE:
            price = up_mid[i]
            fee = FEE_FUNC(price)
            payout = 1.0 if winner == "Up" else 0.0
            return {
                "slug": slug, "side": "Up", "entry_price": float(price),
                "fee": float(fee), "payout": float(payout),
                "pnl": float((payout - price - fee) * MIN_ORDER_SIZE),
                "won": winner == "Up",
            }
        elif dn_below and not up_below and dn_mid[i] <= MAX_ENTRY_PRICE:
            price = dn_mid[i]
            fee = FEE_FUNC(price)
            payout = 1.0 if winner == "Down" else 0.0
            return {
                "slug": slug, "side": "Down", "entry_price": float(price),
                "fee": float(fee), "payout": float(payout),
                "pnl": float((payout - price - fee) * MIN_ORDER_SIZE),
                "won": winner == "Down",
            }

    return None


def run_backtest(df, resolution, convexity_threshold):
    """Run backtest on pre-loaded data."""
    trades = []
    for slug, slug_df in df.groupby("slug"):
        trade = process_market(slug_df, slug, resolution, convexity_threshold)
        if trade is not None:
            trades.append(trade)
    return pd.DataFrame(trades) if trades else None


def compute_metrics(trades_df):
    n = len(trades_df)
    if n == 0:
        return {"total_trades": 0, "win_rate": 0.0, "pnl_total": 0.0,
                "pnl_per_trade": 0.0, "max_drawdown": 0.0, "sharpe": 0.0}
    wins = trades_df["won"].sum()
    total_pnl = trades_df["pnl"].sum()
    cum_pnl = trades_df["pnl"].cumsum()
    drawdown = float((cum_pnl - cum_pnl.cummax()).min())
    std = trades_df["pnl"].std()
    sharpe = float(trades_df["pnl"].mean() / std) if std > 0 else 0.0

    return {
        "total_trades": int(n),
        "win_rate": round(float(wins / n), 4),
        "pnl_total": round(float(total_pnl), 4),
        "pnl_per_trade": round(float(total_pnl / n), 4),
        "max_drawdown": round(drawdown, 4),
        "sharpe": round(sharpe, 4),
        "avg_entry_price": round(float(trades_df["entry_price"].mean()), 4),
    }


def main():
    print("=" * 70)
    print("ORDERBOOK SHAPE CONVEXITY STRATEGY")
    print("=" * 70)

    # Load data once
    print("\nLoading resolution cache...")
    resolution = load_resolution_cache()
    print(f"Resolved {len(resolution)} markets")

    print("\nLoading and subsampling data...")
    df = load_subsampled_data()
    print(f"Working dataset: {len(df):,} rows, {df['slug'].nunique()} markets")

    # Quick data quality check
    print(f"\nMid-price range: [{df['mid_price'].min():.3f}, {df['mid_price'].max():.3f}]")
    print(f"Avg bid_size_1: {df['bid_size_1'].mean():.1f}")

    # Check convexity distribution
    sample_sizes = df[["bid_size_1", "bid_size_2", "bid_size_3", "bid_size_4", "bid_size_5"]].values[:10000]
    sample_conv = compute_convexity_vec(sample_sizes)
    print(f"\nConvexity distribution (sample):")
    print(f"  Mean: {np.mean(sample_conv):.1f}, Median: {np.median(sample_conv):.1f}")
    print(f"  P10: {np.percentile(sample_conv, 10):.1f}, P90: {np.percentile(sample_conv, 90):.1f}")
    print(f"  Min: {np.min(sample_conv):.1f}, Max: {np.max(sample_conv):.1f}")

    # Parameter sweep
    thresholds = [-200, -100, -50, -20, -10, 0, 10, 50, 100, 500, 999999]
    sweep_results = []

    for thresh in thresholds:
        print(f"\n--- Threshold: {thresh} ---", end=" ")
        trades_df = run_backtest(df, resolution, thresh)
        if trades_df is not None and len(trades_df) > 0:
            metrics = compute_metrics(trades_df)
            metrics["threshold"] = thresh
            sweep_results.append(metrics)
            print(f"Trades: {metrics['total_trades']}, WR: {metrics['win_rate']:.2%}, "
                  f"PnL: {metrics['pnl_total']:.2f}, Sharpe: {metrics['sharpe']:.3f}")
        else:
            print("No trades")
            sweep_results.append({"threshold": thresh, "total_trades": 0})

    # Baseline = threshold=999999
    baseline = next((r for r in sweep_results if r["threshold"] == 999999), None)

    # Best filtered
    valid_filtered = [r for r in sweep_results
                      if r.get("total_trades", 0) >= 10 and r["threshold"] != 999999]
    best = max(valid_filtered, key=lambda r: r.get("pnl_per_trade", -999)) if valid_filtered else None

    print(f"\n\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    if baseline and baseline.get("total_trades", 0) > 0:
        print(f"Baseline (no filter): Trades={baseline['total_trades']}, "
              f"WR={baseline.get('win_rate',0):.2%}, PnL={baseline.get('pnl_total',0):.2f}, "
              f"PnL/trade={baseline.get('pnl_per_trade',0):.4f}")
    if best:
        print(f"Best filtered (thresh={best['threshold']}): Trades={best['total_trades']}, "
              f"WR={best['win_rate']:.2%}, PnL={best['pnl_total']:.2f}, "
              f"PnL/trade={best['pnl_per_trade']:.4f}")

    # Verdict
    final_metrics = best if best else (baseline if baseline and baseline.get("total_trades", 0) > 0 else {
        "total_trades": 0, "win_rate": 0.0, "pnl_total": 0.0,
        "pnl_per_trade": 0.0, "max_drawdown": 0.0, "sharpe": 0.0,
    })

    filter_adds_value = False
    if best and baseline and baseline.get("total_trades", 0) > 0:
        filter_adds_value = (
            best.get("pnl_per_trade", 0) > baseline.get("pnl_per_trade", -999) * 1.05
        )
        # Also check if best has higher win rate
        print(f"\nFilter improvement: PnL/trade {baseline.get('pnl_per_trade',0):.4f} -> "
              f"{best.get('pnl_per_trade',0):.4f}, "
              f"WR {baseline.get('win_rate',0):.2%} -> {best.get('win_rate',0):.2%}")

    if final_metrics.get("total_trades", 0) >= 10 and final_metrics.get("pnl_total", 0) > 0:
        if filter_adds_value:
            verdict = "promising"
        else:
            verdict = "marginal"
    else:
        verdict = "failed"

    # Write results
    results = {
        "strategy_name": "orderbook_shape_convexity",
        "description": (
            "Uses second derivative of cumulative bid/ask depth across 5 orderbook levels "
            "as a volatility-regime filter. Concave curves (negative convexity = thick near "
            "top, thin at depth) indicate low uncertainty / strong consensus. In concave regimes, "
            "buy the token whose mid-price just dropped below its 6-snapshot rolling mean "
            "(mean-reversion fade). Skip convex regimes (uncertain markets)."
        ),
        "hypothesis": (
            "Concave orderbook depth curves indicate market consensus and low uncertainty. "
            "Mean-reversion signals (price dipping below rolling mean) should be more reliable "
            "in these regimes compared to convex (uncertain) regimes."
        ),
        "metrics": {
            "total_trades": final_metrics.get("total_trades", 0),
            "win_rate": final_metrics.get("win_rate", 0.0),
            "pnl_total": final_metrics.get("pnl_total", 0.0),
            "pnl_per_trade": final_metrics.get("pnl_per_trade", 0.0),
            "max_drawdown": final_metrics.get("max_drawdown", 0.0),
            "sharpe": final_metrics.get("sharpe", 0.0),
        },
        "baseline_metrics": {
            "total_trades": baseline.get("total_trades", 0) if baseline else 0,
            "win_rate": baseline.get("win_rate", 0.0) if baseline else 0.0,
            "pnl_total": baseline.get("pnl_total", 0.0) if baseline else 0.0,
            "pnl_per_trade": baseline.get("pnl_per_trade", 0.0) if baseline else 0.0,
        },
        "parameters": {
            "convexity_threshold": best["threshold"] if best else "N/A",
            "rolling_window": ROLLING_WINDOW,
            "max_entry_price": MAX_ENTRY_PRICE,
            "min_order_size": MIN_ORDER_SIZE,
            "snapshot_subsample": SUBSAMPLE,
        },
        "sweep_results": [
            {k: v for k, v in r.items()} for r in sweep_results
        ],
        "convexity_filter_adds_value": filter_adds_value,
        "verdict": verdict,
        "next_steps": (
            "If promising: validate on full dataset (process per-file), "
            "tune asymmetric bid/ask thresholds, add spread filter, "
            "simulate execution delay. "
            "If failed: orderbook shape convexity does not reliably predict "
            "mean-reversion quality in 5-minute binary prediction markets. "
            "Consider alternative regime filters (e.g., spread-based, imbalance-based)."
        ),
    }

    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nVerdict: {verdict}")
    print(f"Filter adds value over baseline: {filter_adds_value}")
    print(f"Results saved to {OUTPUT_DIR / 'results.json'}")


if __name__ == "__main__":
    main()
