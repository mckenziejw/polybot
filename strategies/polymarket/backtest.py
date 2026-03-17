"""Backtest Polymarket 5-minute predictions against actual orderbook prices.

Joins model predictions with real Polymarket Up/Down token prices to compute
realistic PnL accounting for actual bid/ask spreads.

Two modes:
  - minute-0 sniper: trade only at window open
  - all-minutes: trade at every minute, pick the best signal

Usage:
    python -m strategies.polymarket.backtest \
        --predictions data/polymarket_5m_results_m0/predictions.parquet \
        --orderbook data/polymarket_1m_bars.parquet \
        --mode sniper

    python -m strategies.polymarket.backtest \
        --predictions data/polymarket_5m_results/predictions.parquet \
        --orderbook data/polymarket_1m_bars.parquet \
        --mode all-minutes
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def load_data(
    predictions_path: Path,
    orderbook_path: Path,
) -> pd.DataFrame:
    """Load and join predictions with orderbook prices."""
    preds = pd.read_parquet(predictions_path)
    ob = pd.read_parquet(orderbook_path)

    print(f"Predictions: {len(preds):,} rows, {preds['window_id'].nunique():,} windows")
    print(f"  Range: {pd.to_datetime(preds['timestamp_ms'].min(), unit='ms')} to "
          f"{pd.to_datetime(preds['timestamp_ms'].max(), unit='ms')}")
    print(f"Orderbook: {len(ob):,} rows, {ob['slug'].nunique():,} markets")
    print(f"  Range: {pd.to_datetime(ob['window_start_ms'].min(), unit='ms')} to "
          f"{pd.to_datetime(ob['window_start_ms'].max(), unit='ms')}")

    # Derive minute_in_window from predictions
    if "minute_in_window" not in preds.columns:
        # minute_in_window was normalized to 0-1 in features; or we can derive from timestamp
        preds["minute_in_window"] = ((preds["timestamp_ms"] - preds["window_id"]) // 60_000).astype(int)

    # Join on window_id + minute
    merged = preds.merge(
        ob[["window_id", "minute_in_window", "up_mid", "up_ask", "up_bid",
            "up_spread", "down_mid", "down_ask", "down_bid", "outcome", "slug"]],
        on=["window_id", "minute_in_window"],
        how="inner",
    )

    print(f"Matched: {len(merged):,} rows ({merged['window_id'].nunique():,} windows)")
    return merged


def compute_fee_vec(prices: pd.Series) -> pd.Series:
    """Polymarket crypto taker fee per token (vectorized).

    fee_rate = 0.0625 * p * (1-p), applied to notional (price * quantity).
    Per-token fee = fee_rate * price = 0.0625 * p^2 * (1-p).
    Max fee rate = 1.5625% at p = 0.50.
    """
    return 0.0625 * prices * prices * (1 - prices)


def run_backtest_sniper(
    merged: pd.DataFrame,
    bet_size: float = 10.0,
    confidence_thresholds: list[float] | None = None,
) -> pd.DataFrame:
    """Minute-0 sniper: buy at window open based on prediction.

    For each 5-min window:
    - If model says up (y_pred > 0.5): buy Up token at up_ask
    - If model says down (y_pred < 0.5): buy Down token at down_ask
    - PnL = (1.0 - entry_price) * tokens if correct, else -entry_price * tokens
    """
    if confidence_thresholds is None:
        confidence_thresholds = [0.50, 0.52, 0.55, 0.60, 0.65, 0.70]

    # Filter to minute 0 only
    m0 = merged[merged["minute_in_window"] == 0].copy()
    if len(m0) == 0:
        print("No minute-0 data found")
        return pd.DataFrame()

    print(f"\n{'='*60}")
    print("SNIPER BACKTEST (minute-0 entry)")
    print(f"{'='*60}")
    print(f"Windows: {len(m0):,}")
    print(f"Bet size: ${bet_size:.0f}")

    results = []
    for thresh in confidence_thresholds:
        # Filter by confidence
        confident = m0[
            (m0["y_pred"] > thresh) | (m0["y_pred"] < (1 - thresh))
        ].copy()

        if len(confident) == 0:
            continue

        # Decision: buy Up if pred > 0.5, buy Down if pred < 0.5
        confident["buy_up"] = confident["y_pred"] > 0.5

        # Entry prices (ask side — what we'd actually pay)
        confident["entry_price"] = np.where(
            confident["buy_up"],
            confident["up_ask"],
            confident["down_ask"],
        )

        # Tokens purchased
        confident["tokens"] = bet_size / confident["entry_price"]

        # Fee
        confident["fee"] = compute_fee_vec(confident["entry_price"]) * confident["tokens"]

        # PnL per trade
        # If we bought Up and Up won: pnl = (1.0 - entry) * tokens - fee
        # If we bought Up and Down won: pnl = -entry * tokens - fee
        # If we bought Down and Down won: pnl = (1.0 - entry) * tokens - fee
        # If we bought Down and Up won: pnl = -entry * tokens - fee
        correct = np.where(
            confident["buy_up"],
            confident["outcome"] == 1.0,
            confident["outcome"] == 0.0,
        )
        confident["correct"] = correct

        confident["pnl"] = np.where(
            correct,
            (1.0 - confident["entry_price"]) * confident["tokens"] - confident["fee"],
            -confident["entry_price"] * confident["tokens"] - confident["fee"],
        )

        # Summary
        n = len(confident)
        wr = confident["correct"].mean()
        total_pnl = confident["pnl"].sum()
        avg_pnl = confident["pnl"].mean()
        avg_entry = confident["entry_price"].mean()
        avg_fee = confident["fee"].mean()
        sharpe = confident["pnl"].mean() / confident["pnl"].std() * np.sqrt(288) if confident["pnl"].std() > 0 else 0

        results.append({
            "threshold": thresh,
            "trades": n,
            "win_rate": wr,
            "total_pnl": total_pnl,
            "avg_pnl": avg_pnl,
            "avg_entry": avg_entry,
            "avg_fee": avg_fee,
            "sharpe": sharpe,
        })

    results_df = pd.DataFrame(results)
    print(f"\n{'Conf':>6} {'Trades':>8} {'WR':>8} {'Total PnL':>12} {'$/trade':>10} "
          f"{'Avg Entry':>10} {'Avg Fee':>8} {'Sharpe':>8}")
    print("-" * 76)
    for _, r in results_df.iterrows():
        print(f"{r['threshold']:>6.2f} {r['trades']:>8,.0f} {r['win_rate']:>8.3f} "
              f"${r['total_pnl']:>11,.2f} ${r['avg_pnl']:>9.4f} "
              f"${r['avg_entry']:>9.3f} ${r['avg_fee']:>7.4f} {r['sharpe']:>8.2f}")

    return results_df


def run_backtest_all_minutes(
    merged: pd.DataFrame,
    bet_size: float = 10.0,
    confidence_thresholds: list[float] | None = None,
) -> pd.DataFrame:
    """All-minutes approach: trade at the best signal within each window.

    For each window, finds the minute with the strongest model conviction
    (highest |y_pred - 0.5|) and trades there.
    """
    if confidence_thresholds is None:
        confidence_thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]

    print(f"\n{'='*60}")
    print("ALL-MINUTES BACKTEST (best signal per window)")
    print(f"{'='*60}")

    results = []
    for thresh in confidence_thresholds:
        # Filter by confidence first
        confident = merged[
            (merged["y_pred"] > thresh) | (merged["y_pred"] < (1 - thresh))
        ].copy()

        if len(confident) == 0:
            continue

        # For each window, pick the minute with strongest conviction
        confident["conviction"] = (confident["y_pred"] - 0.5).abs()
        best_per_window = confident.loc[
            confident.groupby("window_id")["conviction"].idxmax()
        ].copy()

        # Decision and entry
        best_per_window["buy_up"] = best_per_window["y_pred"] > 0.5
        best_per_window["entry_price"] = np.where(
            best_per_window["buy_up"],
            best_per_window["up_ask"],
            best_per_window["down_ask"],
        )
        best_per_window["tokens"] = bet_size / best_per_window["entry_price"]
        best_per_window["fee"] = compute_fee_vec(best_per_window["entry_price"]) * best_per_window["tokens"]

        correct = np.where(
            best_per_window["buy_up"],
            best_per_window["outcome"] == 1.0,
            best_per_window["outcome"] == 0.0,
        )
        best_per_window["correct"] = correct
        best_per_window["pnl"] = np.where(
            correct,
            (1.0 - best_per_window["entry_price"]) * best_per_window["tokens"] - best_per_window["fee"],
            -best_per_window["entry_price"] * best_per_window["tokens"] - best_per_window["fee"],
        )

        n = len(best_per_window)
        wr = best_per_window["correct"].mean()
        total_pnl = best_per_window["pnl"].sum()
        avg_pnl = best_per_window["pnl"].mean()
        avg_entry = best_per_window["entry_price"].mean()
        avg_fee = best_per_window["fee"].mean()
        avg_minute = best_per_window["minute_in_window"].mean()
        sharpe = best_per_window["pnl"].mean() / best_per_window["pnl"].std() * np.sqrt(288) if best_per_window["pnl"].std() > 0 else 0

        results.append({
            "threshold": thresh,
            "trades": n,
            "win_rate": wr,
            "total_pnl": total_pnl,
            "avg_pnl": avg_pnl,
            "avg_entry": avg_entry,
            "avg_fee": avg_fee,
            "avg_minute": avg_minute,
            "sharpe": sharpe,
        })

    results_df = pd.DataFrame(results)
    print(f"\n{'Conf':>6} {'Trades':>8} {'WR':>8} {'Total PnL':>12} {'$/trade':>10} "
          f"{'Avg Entry':>10} {'Avg Min':>8} {'Sharpe':>8}")
    print("-" * 76)
    for _, r in results_df.iterrows():
        print(f"{r['threshold']:>6.2f} {r['trades']:>8,.0f} {r['win_rate']:>8.3f} "
              f"${r['total_pnl']:>11,.2f} ${r['avg_pnl']:>9.4f} "
              f"${r['avg_entry']:>9.3f} {r['avg_minute']:>8.1f} {r['sharpe']:>8.2f}")

    return results_df


def run_backtest_per_minute(
    merged: pd.DataFrame,
    bet_size: float = 10.0,
    confidence_threshold: float = 0.55,
) -> pd.DataFrame:
    """Breakdown by minute position: shows how accuracy and PnL change through the window."""
    print(f"\n{'='*60}")
    print(f"PER-MINUTE BREAKDOWN (conf >= {confidence_threshold})")
    print(f"{'='*60}")

    results = []
    for minute in range(5):
        sub = merged[merged["minute_in_window"] == minute].copy()
        confident = sub[
            (sub["y_pred"] > confidence_threshold) | (sub["y_pred"] < (1 - confidence_threshold))
        ]

        if len(confident) == 0:
            continue

        confident = confident.copy()
        confident["buy_up"] = confident["y_pred"] > 0.5
        confident["entry_price"] = np.where(
            confident["buy_up"],
            confident["up_ask"],
            confident["down_ask"],
        )
        confident["tokens"] = bet_size / confident["entry_price"]
        confident["fee"] = compute_fee_vec(confident["entry_price"]) * confident["tokens"]

        correct = np.where(
            confident["buy_up"],
            confident["outcome"] == 1.0,
            confident["outcome"] == 0.0,
        )
        confident["correct"] = correct
        confident["pnl"] = np.where(
            correct,
            (1.0 - confident["entry_price"]) * confident["tokens"] - confident["fee"],
            -confident["entry_price"] * confident["tokens"] - confident["fee"],
        )

        results.append({
            "minute": minute,
            "trades": len(confident),
            "win_rate": confident["correct"].mean(),
            "total_pnl": confident["pnl"].sum(),
            "avg_pnl": confident["pnl"].mean(),
            "avg_entry": confident["entry_price"].mean(),
            "avg_up_spread": confident["up_spread"].mean(),
        })

    results_df = pd.DataFrame(results)
    print(f"\n{'Min':>4} {'Trades':>8} {'WR':>8} {'Total PnL':>12} {'$/trade':>10} "
          f"{'Avg Entry':>10} {'Avg Spread':>10}")
    print("-" * 66)
    for _, r in results_df.iterrows():
        print(f"{r['minute']:>4.0f} {r['trades']:>8,.0f} {r['win_rate']:>8.3f} "
              f"${r['total_pnl']:>11,.2f} ${r['avg_pnl']:>9.4f} "
              f"${r['avg_entry']:>9.3f} ${r['avg_up_spread']:>9.4f}")

    return results_df


def main():
    parser = argparse.ArgumentParser(description="Backtest Polymarket 5-min predictions")
    parser.add_argument("--predictions", required=True, help="Predictions parquet")
    parser.add_argument("--orderbook", default="data/polymarket_1m_bars.parquet",
                        help="Resampled 1-min Polymarket orderbook")
    parser.add_argument("--bet-size", type=float, default=10.0, help="Bet size in dollars")
    parser.add_argument("--mode", choices=["sniper", "all-minutes", "both"], default="both")
    args = parser.parse_args()

    merged = load_data(Path(args.predictions), Path(args.orderbook))

    if args.mode in ("sniper", "both"):
        run_backtest_sniper(merged, bet_size=args.bet_size)

    if args.mode in ("all-minutes", "both"):
        run_backtest_all_minutes(merged, bet_size=args.bet_size)

    # Always show per-minute breakdown
    run_backtest_per_minute(merged, bet_size=args.bet_size)


if __name__ == "__main__":
    main()
