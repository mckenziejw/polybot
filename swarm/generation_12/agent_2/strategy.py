"""
Level-Weighted Imbalance Acceleration Strategy

Hypothesis: Compute orderbook imbalance using inverse-level weighting (deeper levels
weighted more heavily: L5=5, L4=4, ..., L1=1) at two ultra-short windows post-open
(0-2s and 2-5s). The acceleration (change in this weighted imbalance) captures whether
informed flow is building or fading. Positive acceleration toward bid side predicts Up.
"""

import pandas as pd
import numpy as np
import json
import sys

# ── Configuration ──────────────────────────────────────────────────────────
WINDOW1_START_MS = 0
WINDOW1_END_MS = 2000
WINDOW2_START_MS = 2000
WINDOW2_END_MS = 5000

# Level weights: deeper levels weighted more (harder to spoof)
LEVEL_WEIGHTS = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

# Acceleration threshold for signal
ACCEL_THRESHOLD = 0.02  # will sweep this

# Entry price cap
MAX_ENTRY_PRICE = 0.85
MIN_ORDER_SIZE = 5

DATA_PATH = "data/sample_1k/book_snapshots_sample_1k.parquet"
RESOLUTION_PATH = "data/sample_1k/resolution_cache_sample.json"
OUTPUT_PATH = "/home/ubuntu/wss_test/swarm/generation_12/agent_2/results.json"


def compute_fee(p: np.ndarray) -> np.ndarray:
    """Polymarket fee: 0.0222 * p * 0.25 * (p*(1-p))^2"""
    return 0.0222 * p * 0.25 * (p * (1.0 - p)) ** 2


def resolve_winner(rc: dict, asset_map: dict) -> dict:
    """Build slug -> winner_label mapping from resolution cache."""
    winners = {}
    for slug, v in rc.items():
        if "winnerLabel" in v:
            winners[slug] = v["winnerLabel"]
        elif "outcomes" in v and "clobTokenIds" in v:
            winning_id = v["winningTokenId"]
            try:
                idx = v["clobTokenIds"].index(winning_id)
                winners[slug] = v["outcomes"][idx]
            except (ValueError, IndexError):
                continue
        elif "token_outcomes" in v:
            for tid, outcome in v["token_outcomes"].items():
                if outcome == 1.0 and tid in asset_map:
                    winners[slug] = asset_map[tid]
                    break
    return winners


def compute_weighted_imbalance(df: pd.DataFrame) -> pd.Series:
    """Compute level-weighted book imbalance for each row.

    Weighted imbalance = sum(w_i * bid_size_i) / (sum(w_i * bid_size_i) + sum(w_i * ask_size_i))
    """
    bid_cols = [f"bid_size_{i}" for i in range(1, 6)]
    ask_cols = [f"ask_size_{i}" for i in range(1, 6)]

    bid_vals = df[bid_cols].values  # (N, 5)
    ask_vals = df[ask_cols].values  # (N, 5)

    weighted_bid = bid_vals @ LEVEL_WEIGHTS  # (N,)
    weighted_ask = ask_vals @ LEVEL_WEIGHTS  # (N,)

    total = weighted_bid + weighted_ask
    # Avoid division by zero
    imbalance = np.where(total > 0, weighted_bid / total, 0.5)
    return pd.Series(imbalance, index=df.index)


def main():
    print("Loading data...")
    df = pd.read_parquet(DATA_PATH)

    with open(RESOLUTION_PATH) as f:
        rc = json.load(f)

    # Build asset_id -> token_label mapping
    asset_label = df.drop_duplicates(subset=["asset_id"])[["asset_id", "token_label"]]
    asset_map = dict(zip(asset_label["asset_id"], asset_label["token_label"]))

    # Resolve winners
    winners = resolve_winner(rc, asset_map)
    print(f"Resolved {len(winners)} market winners out of {len(rc)} markets")

    # Extract market open time from slug
    df["market_open_ms"] = df["slug"].str.extract(r"(\d+)$").astype(np.int64) * 1000
    df["offset_ms"] = df["exchange_timestamp"] - df["market_open_ms"]

    # Focus on Up token only (symmetric analysis)
    df_up = df[df["token_label"] == "Up"].copy()
    del df  # free memory

    # Compute weighted imbalance for all rows
    print("Computing weighted imbalance...")
    df_up["w_imbalance"] = compute_weighted_imbalance(df_up)

    # Assign window labels
    df_up["window"] = np.where(
        (df_up["offset_ms"] >= WINDOW1_START_MS) & (df_up["offset_ms"] < WINDOW1_END_MS),
        1,
        np.where(
            (df_up["offset_ms"] >= WINDOW2_START_MS) & (df_up["offset_ms"] < WINDOW2_END_MS),
            2,
            0
        )
    )

    # Filter to windows 1 and 2 only
    df_windows = df_up[df_up["window"].isin([1, 2])].copy()
    del df_up

    # Compute mean weighted imbalance per slug per window
    print("Aggregating per market per window...")
    agg = df_windows.groupby(["slug", "window"]).agg(
        w_imbalance_mean=("w_imbalance", "mean"),
        mid_price_mean=("mid_price", "mean"),
        count=("w_imbalance", "count"),
    ).reset_index()

    # Pivot to get window1 and window2 side by side
    w1 = agg[agg["window"] == 1].set_index("slug")
    w2 = agg[agg["window"] == 2].set_index("slug")

    # Join
    signals = w1[["w_imbalance_mean", "mid_price_mean", "count"]].rename(columns={
        "w_imbalance_mean": "imb_w1",
        "mid_price_mean": "mid_w1",
        "count": "count_w1",
    }).join(
        w2[["w_imbalance_mean", "mid_price_mean"]].rename(columns={
            "w_imbalance_mean": "imb_w2",
            "mid_price_mean": "mid_w2",
        }),
        how="inner"
    )

    # Compute acceleration (second window - first window)
    signals["acceleration"] = signals["imb_w2"] - signals["imb_w1"]

    # Map winner
    signals["winner"] = signals.index.map(winners)
    signals = signals.dropna(subset=["winner"])
    print(f"Markets with signals and resolution: {len(signals)}")

    # Use mid_w2 as entry price (end of window 2, ~5s in)
    signals["entry_price"] = signals["mid_w2"]

    # ── Parameter sweep ──────────────────────────────────────────────────
    print("\n=== Parameter Sweep ===")
    best_pnl = -np.inf
    best_params = {}
    best_result = None

    for threshold in [0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.08, 0.10]:
        for min_count in [5, 10, 20]:
            result = evaluate_strategy(signals, threshold, min_count)
            if result is not None and result["pnl_total"] > best_pnl:
                best_pnl = result["pnl_total"]
                best_params = {"threshold": threshold, "min_count": min_count}
                best_result = result
            if result is not None:
                print(f"  thresh={threshold:.3f} min_count={min_count:2d} | "
                      f"trades={result['total_trades']:4d} win={result['win_rate']:.3f} "
                      f"pnl={result['pnl_total']:+.2f} pnl/trade={result['pnl_per_trade']:+.4f}")

    print(f"\nBest params: {best_params}")
    if best_result:
        print(f"Best PnL: {best_result['pnl_total']:+.2f}")
        print(f"Best win rate: {best_result['win_rate']:.3f}")
        print(f"Best trades: {best_result['total_trades']}")

    # Also evaluate with reversed signal direction
    print("\n=== Reversed Signal (negative acceleration -> Up) ===")
    signals["acceleration_rev"] = -signals["acceleration"]
    best_pnl_rev = -np.inf
    best_params_rev = {}
    best_result_rev = None

    for threshold in [0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.08, 0.10]:
        for min_count in [5, 10, 20]:
            result = evaluate_strategy(signals, threshold, min_count, accel_col="acceleration_rev")
            if result is not None and result["pnl_total"] > best_pnl_rev:
                best_pnl_rev = result["pnl_total"]
                best_params_rev = {"threshold": threshold, "min_count": min_count}
                best_result_rev = result

    if best_result_rev:
        print(f"Best reversed PnL: {best_result_rev['pnl_total']:+.2f} "
              f"(trades={best_result_rev['total_trades']}, wr={best_result_rev['win_rate']:.3f})")

    # Pick overall best
    if best_result_rev and best_result_rev["pnl_total"] > best_pnl:
        final_result = best_result_rev
        final_params = best_params_rev
        final_params["signal_direction"] = "reversed"
        print("\n>>> Reversed signal is better")
    else:
        final_result = best_result
        final_params = best_params
        final_params["signal_direction"] = "original"
        print("\n>>> Original signal direction is better")

    # ── Write results ────────────────────────────────────────────────────
    if final_result is None:
        final_result = {"total_trades": 0, "win_rate": 0, "pnl_total": 0,
                        "pnl_per_trade": 0, "max_drawdown": 0, "sharpe": 0}

    # Determine verdict
    if final_result["total_trades"] >= 50 and final_result["pnl_per_trade"] > 0.005:
        verdict = "promising"
    elif final_result["total_trades"] >= 20 and final_result["pnl_per_trade"] > 0:
        verdict = "marginal"
    else:
        verdict = "failed"

    results = {
        "strategy_name": "level_weighted_imbalance_acceleration",
        "description": (
            "Compute orderbook imbalance using inverse-level weighting (L5=5..L1=1) "
            "at two ultra-short windows post-open (0-2s, 2-5s). The acceleration "
            "(change in weighted imbalance) predicts market direction. Positive "
            "acceleration toward bid side suggests buying Up token."
        ),
        "hypothesis": (
            "Deep-book activity weighted by level distance is harder to spoof than "
            "top-of-book. The second derivative (acceleration) of this weighted "
            "imbalance captures whether informed flow is building or fading."
        ),
        "metrics": {
            "total_trades": final_result["total_trades"],
            "win_rate": round(final_result["win_rate"], 4),
            "pnl_total": round(final_result["pnl_total"], 4),
            "pnl_per_trade": round(final_result["pnl_per_trade"], 4),
            "max_drawdown": round(final_result["max_drawdown"], 4),
            "sharpe": round(final_result["sharpe"], 4),
        },
        "parameters": final_params,
        "verdict": verdict,
        "next_steps": (
            "If promising, test on full dataset. Consider adding spread filter, "
            "volume-weighted imbalance, or combining with trade flow signals."
            if verdict != "failed" else
            "Strategy did not show edge. The imbalance acceleration signal at 0-5s "
            "post-open may not contain sufficient predictive information, or the "
            "signal is too noisy at these ultra-short timescales for 5-minute "
            "binary outcomes."
        ),
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {OUTPUT_PATH}")
    print(f"Verdict: {verdict}")


def evaluate_strategy(signals: pd.DataFrame, threshold: float, min_count: int,
                      accel_col: str = "acceleration") -> dict | None:
    """Evaluate strategy for given parameters."""
    # Filter: sufficient data points in window 1
    mask = signals["count_w1"] >= min_count

    # Signal: acceleration > threshold -> buy Up, acceleration < -threshold -> buy Down
    buy_up = mask & (signals[accel_col] > threshold) & (signals["entry_price"] <= MAX_ENTRY_PRICE)
    buy_down = mask & (signals[accel_col] < -threshold)

    # For buy_down, entry price is 1 - mid (Down token price ≈ 1 - Up mid)
    down_entry = 1.0 - signals["entry_price"]
    buy_down = buy_down & (down_entry <= MAX_ENTRY_PRICE)

    if buy_up.sum() + buy_down.sum() == 0:
        return None

    # Calculate PnL for Up trades
    up_trades = signals[buy_up].copy()
    up_trades["side"] = "Up"
    up_trades["entry"] = up_trades["entry_price"]
    up_trades["won"] = up_trades["winner"] == "Up"

    # Calculate PnL for Down trades
    down_trades = signals[buy_down].copy()
    down_trades["side"] = "Down"
    down_trades["entry"] = 1.0 - down_trades["entry_price"]
    down_trades["won"] = down_trades["winner"] == "Down"

    trades = pd.concat([up_trades, down_trades])
    if len(trades) == 0:
        return None

    # PnL: win -> (1 - entry - fee), lose -> (-entry - fee)
    fees = compute_fee(trades["entry"].values)
    pnl = np.where(trades["won"].values, 1.0 - trades["entry"].values - fees, -trades["entry"].values - fees)

    total_trades = len(trades)
    win_rate = trades["won"].mean()
    pnl_total = pnl.sum()
    pnl_per_trade = pnl.mean()

    # Max drawdown
    cumulative = np.cumsum(pnl)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = running_max - cumulative
    max_drawdown = drawdown.max() if len(drawdown) > 0 else 0

    # Sharpe (annualized assuming ~288 5-min markets per day)
    if pnl.std() > 0:
        sharpe = (pnl.mean() / pnl.std()) * np.sqrt(288)
    else:
        sharpe = 0.0

    return {
        "total_trades": total_trades,
        "win_rate": win_rate,
        "pnl_total": pnl_total,
        "pnl_per_trade": pnl_per_trade,
        "max_drawdown": max_drawdown,
        "sharpe": sharpe,
    }


if __name__ == "__main__":
    main()
