"""
Orderbook Shape Convexity Strategy for Polymarket 5m BTC Up/Down Markets

HYPOTHESIS: The curvature (second derivative) of cumulative bid/ask depth across
levels 1-5 in the first snapshot after market open reveals informed market maker
positioning. A convex bid curve (accelerating size deeper in book) vs concave ask
curve may signal directional intent. Trade the side with greater convexity.

Two signal variants tested:
1. Raw convexity: mean(second_diff(cumsize)) for bids vs asks
2. Deep level ratio: bid_size_L3-L5 / (bid_size_L3-L5 + ask_size_L3-L5)
   (simplified version — where is the deep liquidity concentrated?)
"""

import pandas as pd
import numpy as np
import json
import glob
import os

DATA_DIR = "/home/josh/Documents/projects/wss_test/data/telonex_book_snapshots"
RESOLUTION_CACHE = "/home/josh/Documents/projects/wss_test/data/resolution_cache.json"
OUTPUT_DIR = "/home/josh/Documents/projects/wss_test/swarm/generation_0/agent_0"

# Strategy parameters
ENTRY_PRICE_CAP = 0.85
POSITION_SIZE = 10  # tokens per trade
SNAPSHOT_WINDOW_SEC = 5  # use first N seconds of snapshots for signal
MIN_LEVELS_POPULATED = 4  # need at least 4 levels for meaningful convexity
DEEP_RATIO_THRESHOLD = 0.2  # only trade top/bottom quintile of deep_ratio


def compute_fee(price: float) -> float:
    """Polymarket taker fee for crypto markets."""
    return 0.0222 * price * 0.25 * (price * (1 - price)) ** 2


def load_resolution_cache() -> dict:
    """Load and normalize resolution cache to slug -> winner_label (Up/Down)."""
    with open(RESOLUTION_CACHE) as f:
        rc = json.load(f)

    results = {}
    for slug, v in rc.items():
        if "winnerLabel" in v:
            results[slug] = v["winnerLabel"]
        elif "winning_token" in v and "clobTokenIds" in v and "outcomes" in v:
            wt = v["winning_token"]
            for i, tid in enumerate(v["clobTokenIds"]):
                if tid == wt:
                    results[slug] = v["outcomes"][i]
                    break
    return results


def compute_convexity_vectorized(sizes: np.ndarray) -> np.ndarray:
    """
    Compute convexity (mean second derivative of cumulative size) for multiple rows.
    sizes: shape (N, 5) - sizes at levels 1-5
    Returns: shape (N,) - convexity values
    """
    cumsize = np.cumsum(sizes, axis=1)  # (N, 5)
    first_diff = np.diff(cumsize, axis=1)  # (N, 4)
    second_diff = np.diff(first_diff, axis=1)  # (N, 3)
    return np.mean(second_diff, axis=1)  # (N,)


def process_single_market(filepath: str, resolutions: dict) -> dict | None:
    """Extract signals from first snapshots of a single market."""
    df = pd.read_parquet(filepath)
    slug = df["slug"].iloc[0]

    if slug not in resolutions:
        return None

    winner = resolutions[slug]
    epoch_ms = int(slug.split("-")[-1]) * 1000
    epoch_sec = int(slug.split("-")[-1])

    # Get first SNAPSHOT_WINDOW_SEC seconds for Up and Down tokens
    up = df[df["token_label"] == "Up"]
    down = df[df["token_label"] == "Down"]
    up_early = up[(up["exchange_timestamp"] - epoch_ms) <= SNAPSHOT_WINDOW_SEC * 1000]
    dn_early = down[(down["exchange_timestamp"] - epoch_ms) <= SNAPSHOT_WINDOW_SEC * 1000]

    if len(up_early) < 3:
        return None

    bid_cols = [f"bid_size_{i}" for i in range(1, 6)]
    ask_cols = [f"ask_size_{i}" for i in range(1, 6)]

    bid_sizes = up_early[bid_cols].values  # (N, 5)
    ask_sizes = up_early[ask_cols].values  # (N, 5)

    # Filter to snapshots with enough populated levels
    bid_populated = np.sum(bid_sizes > 0, axis=1)
    ask_populated = np.sum(ask_sizes > 0, axis=1)
    valid_mask = (bid_populated >= MIN_LEVELS_POPULATED) & (
        ask_populated >= MIN_LEVELS_POPULATED
    )

    if valid_mask.sum() < 3:
        return None

    bid_valid = bid_sizes[valid_mask]
    ask_valid = ask_sizes[valid_mask]

    # Signal 1: Raw convexity difference
    bid_convexity = np.median(compute_convexity_vectorized(bid_valid))
    ask_convexity = np.median(compute_convexity_vectorized(ask_valid))
    convexity_diff = bid_convexity - ask_convexity

    # Signal 2: Deep level ratio (L3-L5 bid share)
    bid_deep = bid_valid[:, 2:].sum(axis=1)  # L3+L4+L5
    ask_deep = ask_valid[:, 2:].sum(axis=1)
    deep_total = bid_deep + ask_deep
    deep_valid = deep_total > 0
    if deep_valid.sum() < 3:
        return None
    deep_ratio = float(np.median(bid_deep[deep_valid] / deep_total[deep_valid]))

    # Entry prices
    up_ask = float(up_early["ask_price_1"].median())
    dn_ask = float(dn_early["ask_price_1"].median()) if len(dn_early) > 0 else 1.0 - up_ask

    return {
        "slug": slug,
        "epoch": epoch_sec,
        "winner": winner,
        "convexity_diff": float(convexity_diff),
        "bid_convexity": float(bid_convexity),
        "ask_convexity": float(ask_convexity),
        "deep_ratio": float(deep_ratio),
        "up_ask": float(up_ask),
        "dn_ask": float(dn_ask),
        "n_snapshots": int(valid_mask.sum()),
    }


def run_backtest():
    """Run full backtest across all available markets."""
    print("Loading resolution cache...")
    resolutions = load_resolution_cache()
    print(f"  {len(resolutions)} resolved markets")

    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.parquet")))
    print(f"  {len(files)} orderbook snapshot files")

    # Process all markets
    signals = []
    skipped = 0
    for i, f in enumerate(files):
        if i % 1000 == 0:
            print(f"  Processing {i}/{len(files)}...")
        r = process_single_market(f, resolutions)
        if r is not None:
            signals.append(r)
        else:
            skipped += 1

    print(f"\n  Signals extracted: {len(signals)}, skipped: {skipped}")

    if not signals:
        print("No signals generated!")
        return write_results(pd.DataFrame(), {})

    df = pd.DataFrame(signals).sort_values("epoch").reset_index(drop=True)

    # ===== SIGNAL 1: Raw Convexity =====
    print("\n" + "=" * 60)
    print("SIGNAL 1: RAW CONVEXITY (second derivative difference)")
    print("=" * 60)
    evaluate_signal(df, "convexity_diff", "Raw Convexity")

    # ===== SIGNAL 2: Deep Level Ratio =====
    print("\n" + "=" * 60)
    print("SIGNAL 2: DEEP LEVEL RATIO (L3-L5 bid share)")
    print("=" * 60)
    evaluate_signal(df, "deep_ratio", "Deep Ratio")

    # ===== PRIMARY STRATEGY: Deep ratio quintile filter =====
    print("\n" + "=" * 60)
    print("PRIMARY STRATEGY: Deep ratio quintile filter (top/bottom 20%)")
    print("=" * 60)

    q80 = df["deep_ratio"].quantile(0.8)
    q20 = df["deep_ratio"].quantile(0.2)

    # Build trades
    fee_vec = np.vectorize(compute_fee)

    # Top quintile: buy Up
    top = df[df["deep_ratio"] >= q80].copy()
    top["side"] = "Up"
    top["entry_price"] = top["up_ask"]

    # Bottom quintile: buy Down
    bot = df[df["deep_ratio"] <= q20].copy()
    bot["side"] = "Down"
    bot["entry_price"] = bot["dn_ask"]

    trades = pd.concat([top, bot]).sort_values("epoch").reset_index(drop=True)
    trades = trades[(trades["entry_price"] > 0) & (trades["entry_price"] <= ENTRY_PRICE_CAP)]

    trades["won"] = trades["side"] == trades["winner"]
    trades["fee"] = fee_vec(trades["entry_price"])
    trades["cost"] = trades["entry_price"] + trades["fee"]
    trades["pnl"] = np.where(
        trades["won"],
        (1.0 - trades["cost"]) * POSITION_SIZE,
        -trades["cost"] * POSITION_SIZE,
    )
    trades["cumulative_pnl"] = trades["pnl"].cumsum()

    metrics = compute_metrics(trades)
    print_report(trades, metrics)

    # Temporal stability check
    print("\n--- Temporal Stability (4 chunks) ---")
    chunk_size = len(trades) // 4
    for i in range(4):
        chunk = trades.iloc[i * chunk_size : (i + 1) * chunk_size]
        dates = pd.to_datetime(chunk["epoch"], unit="s")
        wr = chunk["won"].mean()
        pnl = chunk["pnl"].sum()
        print(
            f"  {dates.iloc[0].strftime('%m/%d')}-{dates.iloc[-1].strftime('%m/%d')}: "
            f"WR={wr:.1%}, PnL=${pnl:.2f}, N={len(chunk)}"
        )

    # Save trades
    trades.to_csv(os.path.join(OUTPUT_DIR, "trades.csv"), index=False)

    return write_results(trades, metrics)


def evaluate_signal(df: pd.DataFrame, col: str, name: str):
    """Evaluate a signal column's predictive power for Up/Down resolution."""
    is_up = (df["winner"] == "Up").astype(float)
    corr = df[col].corr(is_up)
    print(f"  Correlation with Up win: {corr:.4f}")

    # Median split
    median = df[col].median()
    above = df[df[col] > median]
    below = df[df[col] <= median]
    print(f"  Above median -> Up WR: {(above['winner'] == 'Up').mean():.1%} (N={len(above)})")
    print(f"  Below median -> Down WR: {(below['winner'] == 'Down').mean():.1%} (N={len(below)})")

    # Quintile analysis
    for pct_label, lo_q, hi_q in [
        ("Top 20%", 0.8, 1.0),
        ("Top 10%", 0.9, 1.0),
        ("Top 5%", 0.95, 1.0),
        ("Bot 20%", 0.0, 0.2),
        ("Bot 10%", 0.0, 0.1),
        ("Bot 5%", 0.0, 0.05),
    ]:
        lo = df[col].quantile(lo_q) if lo_q > 0 else df[col].min() - 1
        hi = df[col].quantile(hi_q) if hi_q < 1 else df[col].max() + 1
        sub = df[(df[col] >= lo) & (df[col] < hi)]
        if len(sub) > 0:
            up_wr = (sub["winner"] == "Up").mean()
            print(f"  {pct_label}: Up WR={up_wr:.1%} (N={len(sub)})")


def compute_metrics(trades_df: pd.DataFrame) -> dict:
    """Compute strategy performance metrics."""
    total_trades = len(trades_df)
    if total_trades == 0:
        return {"total_trades": 0, "win_rate": 0.0, "pnl_total": 0.0,
                "pnl_per_trade": 0.0, "max_drawdown": 0.0, "sharpe": 0.0}

    wins = trades_df["won"].sum()
    win_rate = wins / total_trades
    pnl_total = trades_df["pnl"].sum()
    pnl_per_trade = pnl_total / total_trades

    cum_pnl = trades_df["pnl"].cumsum()
    running_max = cum_pnl.cummax()
    max_drawdown = float((cum_pnl - running_max).min())

    # Sharpe using ~288 markets/day grouping
    daily_pnl = trades_df.groupby(trades_df.index // 60)["pnl"].sum()
    if len(daily_pnl) > 1 and daily_pnl.std() > 0:
        sharpe = float(daily_pnl.mean() / daily_pnl.std() * np.sqrt(365))
    else:
        sharpe = 0.0

    return {
        "total_trades": int(total_trades),
        "win_rate": round(float(win_rate), 4),
        "pnl_total": round(float(pnl_total), 2),
        "pnl_per_trade": round(float(pnl_per_trade), 4),
        "max_drawdown": round(max_drawdown, 2),
        "sharpe": round(sharpe, 2),
    }


def print_report(trades_df: pd.DataFrame, metrics: dict):
    """Print backtest report."""
    print(f"Total trades:    {metrics['total_trades']}")
    print(f"Win rate:        {metrics['win_rate']:.1%}")
    print(f"Total PnL:       ${metrics['pnl_total']:.2f}")
    print(f"PnL/trade:       ${metrics['pnl_per_trade']:.4f}")
    print(f"Max drawdown:    ${metrics['max_drawdown']:.2f}")
    print(f"Sharpe ratio:    {metrics['sharpe']:.2f}")

    print(f"\nAvg entry price: {trades_df['entry_price'].mean():.3f}")
    print(f"Avg fee:         {trades_df['fee'].mean():.6f}")

    for side in ["Up", "Down"]:
        sub = trades_df[trades_df["side"] == side]
        if len(sub) > 0:
            print(f"  {side}: WR={sub['won'].mean():.1%}, PnL=${sub['pnl'].sum():.2f} ({len(sub)} trades)")


def write_results(trades_df: pd.DataFrame, metrics: dict) -> dict:
    """Write standardized results JSON."""
    if not metrics:
        metrics = {"total_trades": 0, "win_rate": 0.0, "pnl_total": 0.0,
                   "pnl_per_trade": 0.0, "max_drawdown": 0.0, "sharpe": 0.0}

    pnl_per_trade = metrics.get("pnl_per_trade", 0.0)
    win_rate = metrics.get("win_rate", 0.0)

    if metrics["total_trades"] == 0:
        verdict = "failed"
    elif pnl_per_trade > 0.02 and win_rate > 0.52:
        # Downgrade: deep_ratio signal is temporally unstable (67% in one
        # chunk, 50% in another). Aggregate looks good but most profit comes
        # from a single ~2-day window. Not robust enough for "promising".
        verdict = "marginal"
    elif pnl_per_trade > 0.0:
        verdict = "marginal"
    else:
        verdict = "failed"

    output = {
        "strategy_name": "orderbook_convexity",
        "description": (
            "Two signals tested: (1) Raw convexity — second derivative of cumulative "
            "bid vs ask depth across levels 1-5 in first 5s after market open. "
            "(2) Deep level ratio — share of deep liquidity (L3-L5) on bid vs ask side. "
            "Primary strategy trades top/bottom 20% of deep_ratio."
        ),
        "hypothesis": (
            "A convex bid curve (accelerating size deeper in book) vs concave "
            "ask curve signals informed market makers layering defensively, "
            "predicting upward resolution. Deep level ratio is the simplified "
            "version: where is the deep liquidity concentrated?"
        ),
        "metrics": metrics,
        "parameters": {
            "entry_price_cap": ENTRY_PRICE_CAP,
            "position_size": POSITION_SIZE,
            "snapshot_window_sec": SNAPSHOT_WINDOW_SEC,
            "min_levels_populated": MIN_LEVELS_POPULATED,
            "deep_ratio_quintile_filter": DEEP_RATIO_THRESHOLD,
        },
        "verdict": verdict,
        "next_steps": (
            "Raw convexity (second derivative) has zero predictive power (50.3% WR, "
            "correlation 0.006). Deep level ratio shows marginal signal (55.1% WR in "
            "top+bottom quintile, +$264 total) but is temporally unstable — strong in "
            "Mar 1-3 (67% WR) but flat in other periods. Likely regime-dependent or "
            "an artifact of a few dominant MMs changing behavior. Not robust enough "
            "for live trading. Could investigate: (a) combining with mid_price level "
            "as filter, (b) interaction with time-of-day, (c) whether signal improves "
            "in high-spread markets where MMs have more room to layer."
        ),
    }

    with open(os.path.join(OUTPUT_DIR, "results.json"), "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nVerdict: {verdict}")
    print(f"Results written to {OUTPUT_DIR}/results.json")
    return output


if __name__ == "__main__":
    run_backtest()
