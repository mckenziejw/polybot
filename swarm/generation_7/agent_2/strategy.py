"""
Strategy: Inverse Deep Level Ratio (Spoofing Detection)

Hypothesis: When deep liquidity (L3-L5) is heavily concentrated on the ASK side
(bottom quintile of deep_ratio), buy YES — hypothesizing that deep ask-side layering
represents informed sellers providing liquidity they expect to buy back cheaper
(a bluff/spoofing pattern).

Signal: deep_level_ratio = deep_bid_size / (deep_bid_size + deep_ask_size)
  where deep = sum of sizes at levels 3, 4, 5
Buy YES when deep_level_ratio is in the BOTTOM quintile (heavy ask-side deep liquidity)

Parameters:
  - snapshot_window: 10 seconds after market open for pattern stabilization
  - min_levels_populated: 5 on both sides
  - entry_price_cap: 0.85
  - quintile_threshold: bottom 20% of deep_level_ratio
"""

import pandas as pd
import numpy as np
import json
import os

# === CONFIG ===
SAMPLE_BOOK = "data/sample_1k/book_snapshots_sample_1k.parquet"
SAMPLE_RESOLUTION = "data/sample_1k/resolution_cache_sample.json"
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
SNAPSHOT_WINDOW_MS = 10_000  # 10 seconds
MIN_LEVELS = 5
ENTRY_PRICE_CAP = 0.85
MIN_ORDER_SIZE = 5
QUINTILE_THRESHOLD = 0.20  # bottom 20%

# === Fee calculation ===
def calc_fee(p):
    """Polymarket fee: 0.0222 * p * 0.25 * (p*(1-p))^2"""
    return 0.0222 * p * 0.25 * (p * (1.0 - p)) ** 2


def load_resolution_cache(path):
    """Load resolution cache and map slug -> winning token label."""
    with open(path) as f:
        cache = json.load(f)
    return cache


def resolve_winner_label(slug, info, df_slug_tokens):
    """Determine winner label (Up/Down) from resolution cache entry."""
    # If winnerLabel is directly available
    if "winnerLabel" in info:
        return info["winnerLabel"]

    # Otherwise, match winningTokenId to asset_id in data
    winning_id = info.get("winningTokenId", "")
    if not winning_id:
        return None

    # Look up in df_slug_tokens
    match = df_slug_tokens.get(winning_id)
    return match


def main():
    print("Loading data...")
    df = pd.read_parquet(SAMPLE_BOOK)
    resolution_cache = load_resolution_cache(SAMPLE_RESOLUTION)

    # Build asset_id -> token_label mapping per slug
    print("Building token ID mappings...")
    token_map = (
        df[["slug", "asset_id", "token_label"]]
        .drop_duplicates()
        .set_index(["slug", "asset_id"])["token_label"]
    )

    # Build slug -> {asset_id: label} dict
    slug_token_dicts = {}
    for (slug, asset_id), label in token_map.items():
        if slug not in slug_token_dicts:
            slug_token_dicts[slug] = {}
        slug_token_dicts[slug][asset_id] = label

    # Resolve all winners
    print("Resolving market outcomes...")
    slug_winner = {}
    for slug, info in resolution_cache.items():
        stokens = slug_token_dicts.get(slug, {})
        label = resolve_winner_label(slug, info, stokens)
        if label:
            slug_winner[slug] = label

    print(f"Resolved {len(slug_winner)} / {len(resolution_cache)} markets")

    # Filter to YES (Up) token only for signal computation
    print("Computing signals on Up tokens...")
    df_up = df[df["token_label"] == "Up"].copy()

    # Get market start time per slug
    market_start = df_up.groupby("slug")["exchange_timestamp"].min()
    df_up = df_up.join(market_start.rename("market_start"), on="slug")

    # Filter to snapshot window: first 10 seconds
    elapsed = df_up["exchange_timestamp"] - df_up["market_start"]
    df_up = df_up[elapsed <= SNAPSHOT_WINDOW_MS].copy()

    # Check min levels populated (all 5 levels must have non-zero size)
    bid_size_cols = [f"bid_size_{i}" for i in range(1, 6)]
    ask_size_cols = [f"ask_size_{i}" for i in range(1, 6)]

    bid_populated = (df_up[bid_size_cols] > 0).sum(axis=1)
    ask_populated = (df_up[ask_size_cols] > 0).sum(axis=1)
    df_up = df_up[(bid_populated >= MIN_LEVELS) & (ask_populated >= MIN_LEVELS)].copy()

    print(f"Snapshots after filtering: {len(df_up)}")

    if len(df_up) == 0:
        print("No data after filtering!")
        write_results(0, 0.0, 0.0, 0.0, 0.0, 0.0, "failed")
        return

    # Compute deep level sizes (L3-L5)
    deep_bid_cols = [f"bid_size_{i}" for i in range(3, 6)]
    deep_ask_cols = [f"ask_size_{i}" for i in range(3, 6)]

    df_up["deep_bid_size"] = df_up[deep_bid_cols].sum(axis=1)
    df_up["deep_ask_size"] = df_up[deep_ask_cols].sum(axis=1)

    total_deep = df_up["deep_bid_size"] + df_up["deep_ask_size"]
    df_up["deep_level_ratio"] = np.where(
        total_deep > 0,
        df_up["deep_bid_size"] / total_deep,
        0.5  # neutral if no deep liquidity
    )

    # Aggregate per market: take the LAST snapshot in the window (most stable)
    df_up_sorted = df_up.sort_values(["slug", "exchange_timestamp"])
    market_signals = df_up_sorted.groupby("slug").agg(
        deep_level_ratio=("deep_level_ratio", "last"),
        entry_price=("mid_price", "last"),
        ask_price_1=("ask_price_1", "last"),
        deep_bid_size=("deep_bid_size", "last"),
        deep_ask_size=("deep_ask_size", "last"),
        n_snapshots=("deep_level_ratio", "count"),
    ).reset_index()

    # Use ask_price_1 as entry (we'd be buying the ask)
    market_signals["entry_price"] = market_signals["ask_price_1"]

    # Apply entry price cap
    market_signals = market_signals[market_signals["entry_price"] <= ENTRY_PRICE_CAP].copy()
    market_signals = market_signals[market_signals["entry_price"] > 0].copy()

    print(f"Markets with valid signals: {len(market_signals)}")

    # Compute quintile threshold for deep_level_ratio
    q_threshold = market_signals["deep_level_ratio"].quantile(QUINTILE_THRESHOLD)
    print(f"Deep level ratio stats:")
    print(f"  Mean: {market_signals['deep_level_ratio'].mean():.4f}")
    print(f"  Median: {market_signals['deep_level_ratio'].median():.4f}")
    print(f"  Q20 threshold: {q_threshold:.4f}")
    print(f"  Min: {market_signals['deep_level_ratio'].min():.4f}")
    print(f"  Max: {market_signals['deep_level_ratio'].max():.4f}")

    # INVERSE signal: buy YES when deep_level_ratio is LOW (heavy ask-side deep liquidity)
    trade_signals = market_signals[market_signals["deep_level_ratio"] <= q_threshold].copy()

    print(f"Trade signals (bottom quintile): {len(trade_signals)}")

    # Map outcomes
    trade_signals["winner"] = trade_signals["slug"].map(slug_winner)
    trade_signals = trade_signals.dropna(subset=["winner"])

    print(f"Trades with resolved outcomes: {len(trade_signals)}")

    if len(trade_signals) == 0:
        print("No trades to evaluate!")
        write_results(0, 0.0, 0.0, 0.0, 0.0, 0.0, "failed")
        return

    # Calculate PnL
    # We buy YES (Up token) at entry_price
    # If Up wins: payout = 1.0, profit = 1.0 - entry_price - fee
    # If Down wins: payout = 0.0, profit = -entry_price - fee (lose the purchase price)
    trade_signals["won"] = (trade_signals["winner"] == "Up").astype(int)
    trade_signals["fee"] = trade_signals["entry_price"].apply(calc_fee)
    trade_signals["pnl"] = np.where(
        trade_signals["won"] == 1,
        1.0 - trade_signals["entry_price"] - trade_signals["fee"],
        0.0 - trade_signals["entry_price"] - trade_signals["fee"]
    )

    # Metrics
    total_trades = len(trade_signals)
    wins = trade_signals["won"].sum()
    win_rate = wins / total_trades if total_trades > 0 else 0.0
    pnl_total = trade_signals["pnl"].sum()
    pnl_per_trade = trade_signals["pnl"].mean()

    # Drawdown
    cumulative_pnl = trade_signals["pnl"].cumsum()
    running_max = cumulative_pnl.cummax()
    drawdown = (cumulative_pnl - running_max)
    max_drawdown = drawdown.min()

    # Sharpe (annualized assuming ~288 5-min markets per day)
    if trade_signals["pnl"].std() > 0:
        sharpe = (pnl_per_trade / trade_signals["pnl"].std()) * np.sqrt(288)
    else:
        sharpe = 0.0

    print(f"\n=== RESULTS ===")
    print(f"Total trades: {total_trades}")
    print(f"Wins: {wins}, Win rate: {win_rate:.4f}")
    print(f"PnL total: {pnl_total:.4f}")
    print(f"PnL per trade: {pnl_per_trade:.4f}")
    print(f"Max drawdown: {max_drawdown:.4f}")
    print(f"Sharpe: {sharpe:.4f}")

    # Comparison: what's the baseline win rate for all markets?
    all_winners = market_signals["slug"].map(slug_winner)
    baseline_up_rate = (all_winners == "Up").mean()
    print(f"\nBaseline Up win rate: {baseline_up_rate:.4f}")
    print(f"Strategy win rate: {win_rate:.4f}")
    print(f"Edge over baseline: {win_rate - baseline_up_rate:.4f}")

    # Also compute PnL for top quintile (original signal direction) for comparison
    q_top = market_signals["deep_level_ratio"].quantile(1.0 - QUINTILE_THRESHOLD)
    top_signals = market_signals[market_signals["deep_level_ratio"] >= q_top].copy()
    top_signals["winner"] = top_signals["slug"].map(slug_winner)
    top_signals = top_signals.dropna(subset=["winner"])
    top_signals["won"] = (top_signals["winner"] == "Up").astype(int)
    top_wr = top_signals["won"].mean() if len(top_signals) > 0 else 0.0
    print(f"\nComparison - Top quintile (original direction) win rate: {top_wr:.4f}")
    print(f"Top quintile trades: {len(top_signals)}")

    # Quintile analysis
    print("\n=== QUINTILE ANALYSIS ===")
    market_signals_full = market_signals.copy()
    market_signals_full["winner"] = market_signals_full["slug"].map(slug_winner)
    market_signals_full = market_signals_full.dropna(subset=["winner"])
    market_signals_full["won"] = (market_signals_full["winner"] == "Up").astype(int)
    market_signals_full["quintile"] = pd.qcut(
        market_signals_full["deep_level_ratio"], 5, labels=["Q1(low)", "Q2", "Q3", "Q4", "Q5(high)"]
    )
    quintile_stats = market_signals_full.groupby("quintile").agg(
        count=("won", "count"),
        win_rate=("won", "mean"),
        avg_deep_ratio=("deep_level_ratio", "mean"),
    )
    print(quintile_stats.to_string())

    # Verdict
    edge = win_rate - baseline_up_rate
    if pnl_per_trade > 0.01 and edge > 0.03:
        verdict = "promising"
    elif pnl_per_trade > 0 and edge > 0.01:
        verdict = "marginal"
    else:
        verdict = "failed"

    print(f"\nVerdict: {verdict}")

    write_results(
        total_trades, win_rate, pnl_total, pnl_per_trade,
        max_drawdown, sharpe, verdict,
        baseline_up_rate=baseline_up_rate, edge=edge, q_threshold=q_threshold
    )

    # Save detailed trade log
    trade_log_path = os.path.join(OUTPUT_DIR, "trade_log.csv")
    trade_signals[["slug", "deep_level_ratio", "entry_price", "winner", "won", "fee", "pnl"]].to_csv(
        trade_log_path, index=False
    )
    print(f"Trade log saved to {trade_log_path}")


def write_results(total_trades, win_rate, pnl_total, pnl_per_trade,
                  max_drawdown, sharpe, verdict, **extra):
    results = {
        "strategy_name": "inverse_deep_level_ratio",
        "description": (
            "Buy YES when deep liquidity (L3-L5) is heavily concentrated on the ASK side "
            "(bottom quintile of deep_level_ratio). Hypothesizes that deep ask-side layering "
            "represents informed spoofing/bluffing."
        ),
        "hypothesis": (
            "When deep ask-side liquidity dominates (low deep_level_ratio), it signals "
            "informed sellers layering asks they expect to buy back cheaper — a spoofing "
            "pattern. Buying YES in these conditions should capture the subsequent price "
            "reversal as the bluff is withdrawn."
        ),
        "metrics": {
            "total_trades": int(total_trades),
            "win_rate": round(float(win_rate), 6),
            "pnl_total": round(float(pnl_total), 6),
            "pnl_per_trade": round(float(pnl_per_trade), 6),
            "max_drawdown": round(float(max_drawdown), 6),
            "sharpe": round(float(sharpe), 6),
        },
        "parameters": {
            "snapshot_window_ms": SNAPSHOT_WINDOW_MS,
            "min_levels_populated": MIN_LEVELS,
            "entry_price_cap": ENTRY_PRICE_CAP,
            "quintile_threshold": QUINTILE_THRESHOLD,
            "signal": "deep_level_ratio = deep_bid / (deep_bid + deep_ask), levels 3-5",
            "direction": "buy YES when ratio in bottom quintile (inverse signal)",
        },
        "extra": {k: round(float(v), 6) if isinstance(v, float) else v for k, v in extra.items()},
        "verdict": verdict,
        "next_steps": (
            "If promising: validate on full dataset, test sensitivity to window size and "
            "quintile threshold. If failed: the spoofing hypothesis may not hold in this "
            "market structure, or the signal may be too noisy at 5-min resolution."
        ),
    }

    out_path = os.path.join(OUTPUT_DIR, "results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results written to {out_path}")


if __name__ == "__main__":
    main()
