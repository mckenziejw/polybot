"""
Delayed Deep Level Ratio Strategy for Polymarket 5m BTC Up/Down Markets

HYPOTHESIS: Deep level ratio computed from a later snapshot window (15-30s after
market open instead of 0-5s). The first 5 seconds capture stale/pre-positioned
orders rather than informed reactions. By waiting 15-30s, MMs have adjusted their
deep levels based on opening mid-price, producing a stronger signal.

Deep level ratio = sum(bid_size_3..5) / (sum(bid_size_3..5) + sum(ask_size_3..5))
for the Up token. High ratio → MMs are loading deep bids on Up → bullish signal.
"""

import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE_DIR = Path("/home/josh/Documents/projects/wss_test")
BOOK_DIR = BASE_DIR / "data" / "telonex_book_snapshots"
RESOLUTION_CACHE = BASE_DIR / "data" / "resolution_cache.json"
OUTPUT_DIR = BASE_DIR / "swarm" / "generation_2" / "agent_1"

# Strategy parameters
DELAYED_WINDOW_START_MS = 15_000  # 15s after market open
DELAYED_WINDOW_END_MS = 30_000    # 30s after market open
EARLY_WINDOW_START_MS = 0         # For comparison baseline
EARLY_WINDOW_END_MS = 5_000
ENTRY_PRICE_CAP = 0.85
MIN_ORDER_SIZE = 5
POSITION_SIZE = 20  # tokens per trade


def compute_fee(price: float) -> float:
    """Polymarket taker fee for crypto markets."""
    return 0.0222 * price * 0.25 * (price * (1 - price)) ** 2


def load_resolutions() -> dict:
    """Load resolution cache: slug -> winnerLabel (Up/Down)."""
    with open(RESOLUTION_CACHE) as f:
        cache = json.load(f)
    return {slug: info["winnerLabel"] for slug, info in cache.items()
            if "winnerLabel" in info}


def process_single_market(filepath: Path, slug: str, window_start_ms: int,
                          window_end_ms: int) -> dict | None:
    """Extract deep level ratio and entry price from a single market file."""
    df = pd.read_parquet(filepath)
    epoch_s = int(slug.split("-")[-1])
    epoch_ms = epoch_s * 1000

    # Filter to Up token in the specified time window
    mask = (
        (df["token_label"] == "Up") &
        (df["exchange_timestamp"] >= epoch_ms + window_start_ms) &
        (df["exchange_timestamp"] <= epoch_ms + window_end_ms)
    )
    window = df.loc[mask]

    if len(window) < 5:
        return None

    # Compute deep level sizes (levels 3-5) - vectorized
    deep_bid = window[["bid_size_3", "bid_size_4", "bid_size_5"]].sum(axis=1)
    deep_ask = window[["ask_size_3", "ask_size_4", "ask_size_5"]].sum(axis=1)
    total = deep_bid + deep_ask

    # Average deep level ratio across the window
    valid = total > 0
    if valid.sum() < 3:
        return None

    ratios = np.where(valid, deep_bid / total, np.nan)
    avg_ratio = np.nanmean(ratios)

    # Entry price: median ask_price_1 in window (what we'd actually pay)
    entry_price = window["ask_price_1"].median()

    # Also grab mid_price for reference
    mid_price = window["mid_price"].median()

    return {
        "slug": slug,
        "deep_ratio": avg_ratio,
        "entry_price_up": entry_price,
        "entry_price_down": 1.0 - window["bid_price_1"].median(),  # complement
        "mid_price": mid_price,
        "n_snapshots": len(window),
    }


def run_backtest():
    """Run the delayed deep level ratio backtest."""
    t0 = time.time()
    resolutions = load_resolutions()
    print(f"Loaded {len(resolutions)} resolutions")

    # Get all book snapshot files
    files = sorted(BOOK_DIR.glob("*.parquet"))
    print(f"Found {len(files)} book snapshot files")

    # Process all markets for both windows
    delayed_results = []
    early_results = []

    for filepath in files:
        slug = filepath.stem
        if slug not in resolutions:
            continue

        delayed = process_single_market(filepath, slug,
                                        DELAYED_WINDOW_START_MS, DELAYED_WINDOW_END_MS)
        early = process_single_market(filepath, slug,
                                      EARLY_WINDOW_START_MS, EARLY_WINDOW_END_MS)

        if delayed is not None:
            delayed["winner"] = resolutions[slug]
            delayed_results.append(delayed)
        if early is not None:
            early["winner"] = resolutions[slug]
            early_results.append(early)

    delayed_df = pd.DataFrame(delayed_results)
    early_df = pd.DataFrame(early_results)
    print(f"Markets with delayed window data: {len(delayed_df)}")
    print(f"Markets with early window data: {len(early_df)}")
    elapsed = time.time() - t0
    print(f"Data loading: {elapsed:.1f}s")

    # === ANALYSIS: Delayed window (15-30s) ===
    print("\n" + "=" * 60)
    print("DELAYED WINDOW (15-30s) ANALYSIS")
    print("=" * 60)
    metrics_delayed = analyze_signal(delayed_df, "delayed")

    # === ANALYSIS: Early window (0-5s) for comparison ===
    print("\n" + "=" * 60)
    print("EARLY WINDOW (0-5s) BASELINE")
    print("=" * 60)
    metrics_early = analyze_signal(early_df, "early")

    # === Write results ===
    write_results(metrics_delayed, metrics_early, delayed_df, early_df)


def analyze_signal(df: pd.DataFrame, label: str) -> dict:
    """Analyze deep level ratio signal and simulate trades."""
    df = df.copy()
    df["up_won"] = (df["winner"] == "Up").astype(int)

    # Quintile analysis
    df["quintile"] = pd.qcut(df["deep_ratio"], 5, labels=False, duplicates="drop")
    print(f"\nDeep ratio distribution:")
    print(f"  Mean: {df['deep_ratio'].mean():.4f}")
    print(f"  Std:  {df['deep_ratio'].std():.4f}")
    print(f"  Min:  {df['deep_ratio'].min():.4f}")
    print(f"  Max:  {df['deep_ratio'].max():.4f}")

    print(f"\nQuintile breakdown (Up win rate):")
    quintile_stats = df.groupby("quintile").agg(
        count=("up_won", "count"),
        up_wr=("up_won", "mean"),
        avg_ratio=("deep_ratio", "mean"),
        avg_entry=("entry_price_up", "mean"),
    )
    for q, row in quintile_stats.iterrows():
        print(f"  Q{q}: n={row['count']:4.0f}, Up WR={row['up_wr']:.3f}, "
              f"avg_ratio={row['avg_ratio']:.4f}, avg_entry={row['avg_entry']:.3f}")

    # Trading strategy: Q4 buy Up, Q0 buy Down (extreme quintiles)
    # Q4 = highest deep bid ratio → bullish signal → buy Up
    # Q0 = lowest deep bid ratio → bearish signal → buy Down
    trades = []

    # Q4: Buy Up
    q4 = df[df["quintile"] == 4].copy()
    q4_valid = q4[q4["entry_price_up"] <= ENTRY_PRICE_CAP]
    if len(q4_valid) > 0:
        fee = q4_valid["entry_price_up"].apply(compute_fee)
        cost = (q4_valid["entry_price_up"] + fee) * POSITION_SIZE
        payout = np.where(q4_valid["up_won"] == 1, POSITION_SIZE, 0.0)
        pnl = payout - cost.values
        q4_trades = pd.DataFrame({
            "slug": q4_valid["slug"],
            "side": "buy_up",
            "entry_price": q4_valid["entry_price_up"],
            "fee": fee,
            "won": q4_valid["up_won"],
            "pnl": pnl,
        })
        trades.append(q4_trades)

    # Q0: Buy Down (entry = 1 - bid_price_1 for Up token)
    q0 = df[df["quintile"] == 0].copy()
    q0_valid = q0[q0["entry_price_down"] <= ENTRY_PRICE_CAP]
    if len(q0_valid) > 0:
        fee = q0_valid["entry_price_down"].apply(compute_fee)
        cost = (q0_valid["entry_price_down"] + fee) * POSITION_SIZE
        down_won = (q0_valid["up_won"] == 0).astype(int)
        payout = np.where(down_won == 1, POSITION_SIZE, 0.0)
        pnl = payout - cost.values
        q0_trades = pd.DataFrame({
            "slug": q0_valid["slug"],
            "side": "buy_down",
            "entry_price": q0_valid["entry_price_down"],
            "fee": fee,
            "won": down_won.values,
            "pnl": pnl,
        })
        trades.append(q0_trades)

    if not trades:
        print("\nNo trades generated!")
        return {"total_trades": 0, "win_rate": 0, "pnl_total": 0,
                "pnl_per_trade": 0, "max_drawdown": 0, "sharpe": 0}

    all_trades = pd.concat(trades, ignore_index=True)
    # Sort by slug (time order)
    all_trades = all_trades.sort_values("slug").reset_index(drop=True)

    total = len(all_trades)
    wins = all_trades["won"].sum()
    wr = wins / total
    pnl_total = all_trades["pnl"].sum()
    pnl_per = pnl_total / total

    # Drawdown
    cum_pnl = all_trades["pnl"].cumsum()
    running_max = cum_pnl.cummax()
    drawdown = (cum_pnl - running_max).min()

    # Sharpe (annualized, assuming ~288 markets/day for 5m markets)
    if all_trades["pnl"].std() > 0:
        sharpe = (all_trades["pnl"].mean() / all_trades["pnl"].std()) * np.sqrt(288)
    else:
        sharpe = 0.0

    print(f"\n--- {label.upper()} TRADING RESULTS ---")
    print(f"Total trades: {total}")
    print(f"  Q4 buy Up:   {len(all_trades[all_trades['side']=='buy_up'])}")
    print(f"  Q0 buy Down: {len(all_trades[all_trades['side']=='buy_down'])}")
    print(f"Win rate: {wr:.3f}")
    print(f"PnL total: ${pnl_total:.2f}")
    print(f"PnL/trade: ${pnl_per:.4f}")
    print(f"Max drawdown: ${drawdown:.2f}")
    print(f"Sharpe: {sharpe:.3f}")

    # Breakdown by side
    for side in ["buy_up", "buy_down"]:
        subset = all_trades[all_trades["side"] == side]
        if len(subset) > 0:
            print(f"\n  {side}: n={len(subset)}, WR={subset['won'].mean():.3f}, "
                  f"PnL=${subset['pnl'].sum():.2f}, avg_entry={subset['entry_price'].mean():.3f}")

    metrics = {
        "total_trades": int(total),
        "win_rate": round(float(wr), 4),
        "pnl_total": round(float(pnl_total), 2),
        "pnl_per_trade": round(float(pnl_per), 4),
        "max_drawdown": round(float(drawdown), 2),
        "sharpe": round(float(sharpe), 3),
    }
    return metrics


def write_results(metrics_delayed, metrics_early, delayed_df, early_df):
    """Write standardized results JSON."""
    # Determine verdict
    if metrics_delayed["pnl_per_trade"] > 0.05:
        verdict = "promising"
    elif metrics_delayed["pnl_per_trade"] > -0.02:
        verdict = "marginal"
    else:
        verdict = "failed"

    results = {
        "strategy_name": "delayed_deep_level_ratio",
        "description": (
            "Deep level ratio (bid_size_3..5 vs ask_size_3..5) computed from "
            "15-30s window after market open instead of first 5s. Hypothesis: "
            "MMs need time to react to opening price, so delayed snapshot "
            "captures informed positioning rather than stale pre-positioned orders."
        ),
        "hypothesis": (
            "Deep level ratio from 15-30s window produces stronger directional "
            "signal than 0-5s window because MMs have had time to observe the "
            "opening mid-price and adjust their deep levels accordingly."
        ),
        "metrics": metrics_delayed,
        "baseline_metrics": metrics_early,
        "parameters": {
            "delayed_window_ms": [DELAYED_WINDOW_START_MS, DELAYED_WINDOW_END_MS],
            "early_window_ms": [EARLY_WINDOW_START_MS, EARLY_WINDOW_END_MS],
            "entry_price_cap": ENTRY_PRICE_CAP,
            "position_size": POSITION_SIZE,
            "quintile_filter": "Q0 (buy Down) + Q4 (buy Up)",
            "deep_levels": "3-5",
        },
        "verdict": verdict,
        "next_steps": "",
    }

    # Set next_steps based on verdict
    if verdict == "promising":
        results["next_steps"] = (
            "Validate with walk-forward splits. Test sensitivity to window "
            "placement (10-25s, 20-35s). Check if signal decays over time."
        )
    elif verdict == "marginal":
        results["next_steps"] = (
            "Marginal edge — likely consumed by spread/fees in live trading. "
            "Could combine with other signals (trade flow, mid-price momentum) "
            "as a feature in a composite model."
        )
    else:
        results["next_steps"] = (
            "Signal does not exist or is too weak. The deep level ratio does "
            "not carry actionable information at either time window."
        )

    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {OUTPUT_DIR / 'results.json'}")
    print(f"Verdict: {verdict}")


if __name__ == "__main__":
    run_backtest()
