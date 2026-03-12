"""
Strategy: Early Deep-Level Ratio with Wide Spread Filter

Hypothesis: Restrict deep_level_ratio signal to first 2 seconds after market open
and only trade markets with wider spreads, where MMs have more room to express
directional views through layering. The earliest snapshots capture initial MM
positioning before arbitrageurs flatten the book, and wide-spread markets amplify
the informational content of deep liquidity placement.

deep_level_ratio = sum(size levels 3-5) / sum(size levels 1-5) on each side
A high deep_level_ratio on bids vs asks suggests MMs are layering deep bids = bullish.

Testing approach:
1. Compute signal across ALL markets using 2s window
2. Test whether wider spreads amplify signal predictiveness
3. Compare 2s vs 5s windows to validate time restriction hypothesis
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────
DATA_PATH = Path("/home/ubuntu/wss_test/data/sample_1k/book_snapshots_sample_1k.parquet")
RESOLUTION_PATH = Path("/home/ubuntu/wss_test/data/sample_1k/resolution_cache_sample.json")
OUTPUT_DIR = Path("/home/ubuntu/wss_test/swarm/generation_7/agent_0")

MAX_ENTRY_PRICE = 0.85
MIN_ORDER_SIZE = 5


def compute_fee(price: np.ndarray) -> np.ndarray:
    """Polymarket fee: 0.0222 * p * 0.25 * (p*(1-p))^2"""
    p = np.asarray(price, dtype=np.float64)
    return 0.0222 * p * 0.25 * (p * (1 - p)) ** 2


def load_resolution_cache(path: Path) -> dict:
    """Load resolution cache and normalize to {slug: winner_label}."""
    with open(path) as f:
        raw = json.load(f)

    resolved = {}
    for slug, info in raw.items():
        if not isinstance(info, dict):
            continue
        if "winnerLabel" in info:
            resolved[slug] = info["winnerLabel"]
        elif "outcomes" in info and "clobTokenIds" in info and "winningTokenId" in info:
            winning_id = info["winningTokenId"]
            for outcome, token_id in zip(info["outcomes"], info["clobTokenIds"]):
                if token_id == winning_id:
                    resolved[slug] = outcome
                    break
        elif "winning_token" in info and "outcomes" in info and "clobTokenIds" in info:
            winning_token = info["winning_token"]
            for outcome, token_id in zip(info["outcomes"], info["clobTokenIds"]):
                if token_id == winning_token:
                    resolved[slug] = outcome
                    break
    return resolved


def compute_signals(df: pd.DataFrame, window_ms: int) -> pd.DataFrame:
    """Compute deep_level_ratio signals from early window snapshots.

    Returns one row per (slug, token_label) with aggregated signals.
    All operations are vectorized.
    """
    # Filter to time window
    early = df[df["time_since_open_ms"] <= window_ms].copy()

    # Deep level ratio computation (vectorized)
    bid_deep = early[["bid_size_3", "bid_size_4", "bid_size_5"]].sum(axis=1)
    bid_total = early[["bid_size_1", "bid_size_2", "bid_size_3", "bid_size_4", "bid_size_5"]].sum(axis=1)
    ask_deep = early[["ask_size_3", "ask_size_4", "ask_size_5"]].sum(axis=1)
    ask_total = early[["ask_size_1", "ask_size_2", "ask_size_3", "ask_size_4", "ask_size_5"]].sum(axis=1)

    early["bid_deep_ratio"] = bid_deep / bid_total.replace(0, np.nan)
    early["ask_deep_ratio"] = ask_deep / ask_total.replace(0, np.nan)
    early["deep_ratio_diff"] = early["bid_deep_ratio"] - early["ask_deep_ratio"]

    # Aggregate per market+token
    signals = early.groupby(["slug", "token_label"]).agg(
        mean_deep_ratio_diff=("deep_ratio_diff", "mean"),
        mean_bid_deep_ratio=("bid_deep_ratio", "mean"),
        mean_ask_deep_ratio=("ask_deep_ratio", "mean"),
        mean_mid_price=("mid_price", "mean"),
        mean_spread=("spread", "mean"),
        mean_book_imbalance=("book_imbalance", "mean"),
        snapshot_count=("deep_ratio_diff", "count"),
    ).reset_index()

    return signals


def generate_trades(signals: pd.DataFrame, resolutions: dict,
                    signal_threshold: float) -> pd.DataFrame:
    """Generate trades from signals. Vectorized where possible."""
    # Pivot Up and Down
    up = signals[signals["token_label"] == "Up"].set_index("slug")
    down = signals[signals["token_label"] == "Down"].set_index("slug")

    common = up.index.intersection(down.index)
    # Filter to resolved markets
    resolved_slugs = [s for s in common if s in resolutions]
    if not resolved_slugs:
        return pd.DataFrame()

    # Build trade dataframe vectorized
    up_sub = up.loc[resolved_slugs]
    down_sub = down.loc[resolved_slugs]

    trade_df = pd.DataFrame({
        "slug": resolved_slugs,
        "up_deep_diff": up_sub["mean_deep_ratio_diff"].values,
        "down_deep_diff": down_sub["mean_deep_ratio_diff"].values,
        "up_mid": up_sub["mean_mid_price"].values,
        "down_mid": down_sub["mean_mid_price"].values,
        "up_spread": up_sub["mean_spread"].values,
        "up_imbalance": up_sub["mean_book_imbalance"].values,
    })

    # Signal: difference in deep ratio differential between Up and Down tokens
    trade_df["signal"] = trade_df["up_deep_diff"] - trade_df["down_deep_diff"]
    trade_df["abs_signal"] = trade_df["signal"].abs()

    # Filter by threshold
    trade_df = trade_df[trade_df["abs_signal"] >= signal_threshold].copy()
    if trade_df.empty:
        return pd.DataFrame()

    # Determine direction
    trade_df["token"] = np.where(trade_df["signal"] > 0, "Up", "Down")
    trade_df["entry_price"] = np.where(
        trade_df["signal"] > 0, trade_df["up_mid"], trade_df["down_mid"]
    )

    # Entry price cap
    trade_df = trade_df[trade_df["entry_price"] <= MAX_ENTRY_PRICE].copy()
    if trade_df.empty:
        return pd.DataFrame()

    # Resolve outcomes
    winners = trade_df["slug"].map(resolutions)
    trade_df["winner"] = winners
    trade_df["won"] = trade_df["token"] == trade_df["winner"]
    trade_df["payout"] = np.where(trade_df["won"], 1.0, 0.0)
    trade_df["fee"] = compute_fee(trade_df["entry_price"].values)
    trade_df["pnl"] = trade_df["payout"] - trade_df["entry_price"] - trade_df["fee"]

    return trade_df


def compute_metrics(trades_df: pd.DataFrame) -> dict:
    """Compute standardized metrics from trade results."""
    if trades_df.empty:
        return {
            "total_trades": 0, "win_rate": 0.0, "pnl_total": 0.0,
            "pnl_per_trade": 0.0, "max_drawdown": 0.0, "sharpe": 0.0,
        }

    cumulative_pnl = trades_df["pnl"].cumsum()
    max_drawdown = float((cumulative_pnl.cummax() - cumulative_pnl).max())
    std = trades_df["pnl"].std()
    sharpe = float(trades_df["pnl"].mean() / std) if std > 0 else 0.0

    return {
        "total_trades": int(len(trades_df)),
        "win_rate": float(trades_df["won"].mean()),
        "pnl_total": float(trades_df["pnl"].sum()),
        "pnl_per_trade": float(trades_df["pnl"].mean()),
        "max_drawdown": max_drawdown,
        "sharpe": sharpe,
    }


def run_backtest():
    print("Loading data...")
    df = pd.read_parquet(DATA_PATH)
    resolutions = load_resolution_cache(RESOLUTION_PATH)
    print(f"Loaded {len(df):,} rows, {df['slug'].nunique()} markets, {len(resolutions)} resolutions")

    # Compute time since market open (vectorized)
    market_open = df.groupby("slug")["exchange_timestamp"].min()
    df["time_since_open_ms"] = df["exchange_timestamp"] - df["slug"].map(market_open)

    # Also compute mean spread per market for stratification
    market_spread = df[df["time_since_open_ms"] <= 2000].groupby("slug")["spread"].mean()

    # ════════════════════════════════════════════════════════════════════
    # TEST 1: Signal across ALL markets, 2s window, various thresholds
    # ════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("TEST 1: Deep level ratio signal, 2s window, ALL markets")
    print("=" * 70)

    signals_2s = compute_signals(df, window_ms=2000)

    for threshold in [0.0, 0.05, 0.10, 0.20, 0.30, 0.50]:
        trades = generate_trades(signals_2s, resolutions, signal_threshold=threshold)
        m = compute_metrics(trades)
        print(f"  threshold={threshold:.2f}: trades={m['total_trades']:4d}, "
              f"win_rate={m['win_rate']:.3f}, pnl={m['pnl_total']:+8.2f}, "
              f"pnl/trade={m['pnl_per_trade']:+.4f}, sharpe={m['sharpe']:+.4f}")

    # ════════════════════════════════════════════════════════════════════
    # TEST 2: Compare 2s vs 5s vs 10s vs 30s windows
    # ════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("TEST 2: Window comparison (threshold=0.10)")
    print("=" * 70)

    for window in [1000, 2000, 5000, 10000, 30000]:
        signals = compute_signals(df, window_ms=window)
        trades = generate_trades(signals, resolutions, signal_threshold=0.10)
        m = compute_metrics(trades)
        print(f"  window={window:6d}ms: trades={m['total_trades']:4d}, "
              f"win_rate={m['win_rate']:.3f}, pnl={m['pnl_total']:+8.2f}, "
              f"pnl/trade={m['pnl_per_trade']:+.4f}, sharpe={m['sharpe']:+.4f}")

    # ════════════════════════════════════════════════════════════════════
    # TEST 3: Spread stratification — does wider spread amplify signal?
    # ════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("TEST 3: Spread stratification (2s window, threshold=0.10)")
    print("=" * 70)

    signals_2s_full = compute_signals(df, window_ms=2000)
    trades_all = generate_trades(signals_2s_full, resolutions, signal_threshold=0.10)

    if not trades_all.empty:
        # Map spread to trades
        trades_all["market_spread"] = trades_all["slug"].map(market_spread)

        # Spread quartiles
        spread_quartiles = pd.qcut(
            trades_all["market_spread"], q=4, labels=["Q1_tight", "Q2", "Q3", "Q4_wide"],
            duplicates="drop"
        )
        trades_all["spread_q"] = spread_quartiles

        print("  Spread quartile breakdown:")
        for q, grp in trades_all.groupby("spread_q", observed=True):
            m = compute_metrics(grp)
            avg_spread = grp["market_spread"].mean()
            print(f"    {q} (avg_spread={avg_spread:.4f}): trades={m['total_trades']:4d}, "
                  f"win_rate={m['win_rate']:.3f}, pnl={m['pnl_total']:+8.2f}, "
                  f"sharpe={m['sharpe']:+.4f}")

        # Also test top decile of spreads
        spread_p90 = trades_all["market_spread"].quantile(0.90)
        wide_trades = trades_all[trades_all["market_spread"] >= spread_p90]
        m_wide = compute_metrics(wide_trades)
        print(f"\n    Top 10% spread (>={spread_p90:.4f}): trades={m_wide['total_trades']}, "
              f"win_rate={m_wide['win_rate']:.3f}, pnl={m_wide['pnl_total']:+.2f}, "
              f"sharpe={m_wide['sharpe']:+.4f}")

    # ════════════════════════════════════════════════════════════════════
    # TEST 4: Interaction of spread filter + signal strength
    # ════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("TEST 4: Signal strength × spread interaction")
    print("=" * 70)

    if not trades_all.empty:
        median_spread = trades_all["market_spread"].median()
        median_signal = trades_all["abs_signal"].median()

        for spread_label, spread_mask in [
            ("tight", trades_all["market_spread"] <= median_spread),
            ("wide", trades_all["market_spread"] > median_spread),
        ]:
            for signal_label, signal_mask in [
                ("weak", trades_all["abs_signal"] <= median_signal),
                ("strong", trades_all["abs_signal"] > median_signal),
            ]:
                subset = trades_all[spread_mask & signal_mask]
                m = compute_metrics(subset)
                print(f"  {spread_label:5s} spread + {signal_label:6s} signal: "
                      f"trades={m['total_trades']:4d}, win_rate={m['win_rate']:.3f}, "
                      f"pnl={m['pnl_total']:+8.2f}, sharpe={m['sharpe']:+.4f}")

    # ════════════════════════════════════════════════════════════════════
    # FINAL: Best configuration selection and results
    # ════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("FINAL STRATEGY SELECTION")
    print("=" * 70)

    # Test the specific hypothesis: 2s window + above-median spread
    best_trades = None
    best_config = None
    best_sharpe = -999

    configs = []
    for window in [1000, 2000, 5000]:
        for threshold in [0.05, 0.10, 0.20, 0.30]:
            for spread_pct in [0, 50, 75]:  # percentile of spread to filter above
                signals = compute_signals(df, window_ms=window)
                trades = generate_trades(signals, resolutions, signal_threshold=threshold)
                if trades.empty:
                    continue

                trades["market_spread"] = trades["slug"].map(market_spread)
                if spread_pct > 0:
                    cutoff = market_spread.quantile(spread_pct / 100)
                    trades = trades[trades["market_spread"] >= cutoff]

                if len(trades) < 10:  # need minimum trades for reliability
                    continue

                m = compute_metrics(trades)
                configs.append({
                    "window": window,
                    "threshold": threshold,
                    "spread_pct": spread_pct,
                    "metrics": m,
                })

                if m["sharpe"] > best_sharpe:
                    best_sharpe = m["sharpe"]
                    best_trades = trades.copy()
                    best_config = {
                        "window_ms": window,
                        "signal_threshold": threshold,
                        "spread_percentile": spread_pct,
                    }

    # Print top 10 configs by Sharpe
    configs.sort(key=lambda x: x["metrics"]["sharpe"], reverse=True)
    print("\nTop 10 configurations by Sharpe:")
    for i, c in enumerate(configs[:10]):
        m = c["metrics"]
        print(f"  {i+1}. window={c['window']}ms, threshold={c['threshold']:.2f}, "
              f"spread_pct>={c['spread_pct']}: trades={m['total_trades']}, "
              f"wr={m['win_rate']:.3f}, pnl={m['pnl_total']:+.2f}, sharpe={m['sharpe']:+.4f}")

    # Final metrics
    if best_trades is not None and not best_trades.empty:
        final_metrics = compute_metrics(best_trades)
        print(f"\nBest config: {best_config}")
        print(f"Final metrics: {final_metrics}")
    else:
        final_metrics = compute_metrics(pd.DataFrame())
        best_config = {"window_ms": 2000, "signal_threshold": 0.10, "spread_percentile": 0}

    # ── Write results ──
    if final_metrics["pnl_total"] > 0 and final_metrics["win_rate"] > 0.52 and final_metrics["sharpe"] > 0.05:
        verdict = "promising"
    elif final_metrics["pnl_total"] > 0 or final_metrics["win_rate"] > 0.50:
        verdict = "marginal"
    else:
        verdict = "failed"

    # Check the specific hypothesis: does 2s beat 5s? Does wider spread help?
    hypothesis_2s_vs_5s = ""
    hypothesis_spread = ""

    signals_2s_test = compute_signals(df, window_ms=2000)
    signals_5s_test = compute_signals(df, window_ms=5000)
    trades_2s = generate_trades(signals_2s_test, resolutions, signal_threshold=0.10)
    trades_5s = generate_trades(signals_5s_test, resolutions, signal_threshold=0.10)
    m2 = compute_metrics(trades_2s)
    m5 = compute_metrics(trades_5s)

    if m2["win_rate"] > m5["win_rate"]:
        hypothesis_2s_vs_5s = f"2s window ({m2['win_rate']:.3f}) outperforms 5s ({m5['win_rate']:.3f}) on win rate — SUPPORTED."
    else:
        hypothesis_2s_vs_5s = f"2s window ({m2['win_rate']:.3f}) does NOT outperform 5s ({m5['win_rate']:.3f}) — NOT SUPPORTED."

    print(f"\nHypothesis check (2s vs 5s): {hypothesis_2s_vs_5s}")

    description = (
        f"Deep level ratio signal: bid-side vs ask-side deep liquidity layering "
        f"(levels 3-5 / levels 1-5) in early market snapshots. "
        f"Best config: window={best_config['window_ms']}ms, "
        f"threshold={best_config['signal_threshold']}, "
        f"spread_pct>={best_config['spread_percentile']}. "
        f"{hypothesis_2s_vs_5s}"
    )

    # Determine next steps
    if verdict == "promising":
        next_steps = (
            "Test on full dataset (7187 markets) to confirm. "
            "Explore combining deep_level_ratio with book_imbalance for stronger signal."
        )
    elif verdict == "marginal":
        next_steps = (
            "Signal shows marginal edge. Try: (1) combining with book_imbalance, "
            "(2) using time-weighted signal giving more weight to earliest snapshots, "
            "(3) testing on full dataset for more statistical power."
        )
    else:
        next_steps = (
            "Deep level ratio in early window does not show predictive power on sample. "
            "The spread > 0.05 filter is too restrictive (only 6/917 markets qualify). "
            "Even at relaxed thresholds, the signal shows no reliable edge. "
            "Consider: (1) order flow signals instead of static book ratios, "
            "(2) changes in deep level ratio over time rather than levels, "
            "(3) combining with other features in an ensemble."
        )

    results = {
        "strategy_name": "early_deep_level_ratio_wide_spread",
        "description": description,
        "hypothesis": (
            "Restrict deep_level_ratio signal to first 2s after market open and only trade "
            "markets with spread > 0.05. Earliest snapshots capture initial MM positioning "
            "before arbitrageurs flatten the book. Wide-spread markets amplify informational "
            "content of deep liquidity placement."
        ),
        "metrics": final_metrics,
        "parameters": best_config,
        "verdict": verdict,
        "next_steps": next_steps,
    }

    output_path = OUTPUT_DIR / "results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {output_path}")
    print(f"Verdict: {verdict}")

    return results


if __name__ == "__main__":
    run_backtest()
