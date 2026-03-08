#!/usr/bin/env python3
"""
Risk of ruin analysis for the v3 XGBoost + vol filter strategy.

Uses the walk-forward out-of-sample predictions from analyze_vol_filter_full.py
to simulate realistic trade sequences and compute:
  - Risk of ruin at different bankrolls and position sizes
  - Drawdown distribution (max, median, percentiles)
  - Kelly criterion sizing
  - Time to double / probability of profit at various horizons
  - Sensitivity to win rate degradation (live vs backtest gap)
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

warnings.filterwarnings("ignore")

FEATURE_CACHE = Path("data/xgb_features_v3.parquet")
N_SIMS = 50_000


def polymarket_fee(price: float) -> float:
    return 100 * price * 0.25 * (price * (1 - price)) ** 2 / 10000


def build_trade_pnl_array(position_size: float = 5.0) -> np.ndarray:
    """Run walk-forward backtest and return the PnL array for
    conf>=0.70, vol p25 filtered trades at 120s window.

    Returns array of per-trade PnL values (out-of-sample).
    """
    df = pd.read_parquet(FEATURE_CACHE)
    feature_cols = [c for c in df.columns if c not in ("label", "slug", "open_ts") and not c.startswith("_")]

    slug_order = df.groupby("slug")["open_ts"].first().sort_values()
    ordered_slugs = slug_order.index.tolist()
    n_markets = len(ordered_slugs)

    min_train = 1000
    step = 200
    val_frac = 0.15

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": 3,
        "learning_rate": 0.01,
        "subsample": 0.7,
        "colsample_bytree": 0.6,
        "min_child_weight": 15,
        "lambda": 2.0,
        "alpha": 0.5,
        "seed": 42,
        "verbosity": 0,
    }

    all_preds = []
    train_end = min_train

    while train_end < n_markets:
        test_end = min(train_end + step, n_markets)
        val_start = int(train_end * (1 - val_frac))

        actual_train = set(ordered_slugs[:val_start])
        actual_val = set(ordered_slugs[val_start:train_end])
        test_slugs_fold = set(ordered_slugs[train_end:test_end])

        train_data = df[df["slug"].isin(actual_train)]
        val_data = df[df["slug"].isin(actual_val)]
        test_data = df[df["slug"].isin(test_slugs_fold)]

        X_train = train_data[feature_cols].values.astype(np.float32)
        y_train = train_data["label"].values
        X_val = val_data[feature_cols].values.astype(np.float32)
        y_val = val_data["label"].values
        X_test = test_data[feature_cols].values.astype(np.float32)
        y_test = test_data["label"].values

        for X in [X_train, X_val, X_test]:
            X[~np.isfinite(X)] = 0.0

        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_cols)
        dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_cols)
        dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_cols)

        model = xgb.train(params, dtrain, num_boost_round=2000,
                          evals=[(dval, "val")], early_stopping_rounds=50,
                          verbose_eval=False)

        preds = model.predict(dtest)

        for i, (_, row) in enumerate(test_data.iterrows()):
            mid_col = "w120_mid" if "w120_mid" in row.index else None
            entry_price = row[mid_col] if mid_col and pd.notna(row[mid_col]) else 0.70
            all_preds.append({
                "pred": preds[i],
                "label": row["label"],
                "btc_vol_5m": row.get("btc_vol_5m", np.nan),
                "entry_price": entry_price,
                "observe_time_s": row["observe_time_s"],
            })

        train_end = test_end

    pdf = pd.DataFrame(all_preds)

    # Vol threshold from first training window
    first_train = df[df["slug"].isin(set(ordered_slugs[:min_train]))]
    train_vol = first_train["btc_vol_5m"].dropna()
    train_vol = train_vol[train_vol > 0]
    vol_p25 = float(train_vol.quantile(0.25))

    # Filter: 120s, conf >= 0.70, vol p25
    w120 = pdf[pdf["observe_time_s"] == 120].copy()
    mask = (w120["pred"] >= 0.70) & (w120["btc_vol_5m"] > vol_p25)
    trades = w120[mask].copy().reset_index(drop=True)

    # Compute PnL per trade
    fees = trades["entry_price"].apply(lambda p: polymarket_fee(p) * position_size)
    pnl = np.where(
        trades["label"] == 1,
        (1.0 - trades["entry_price"]) * position_size - fees,
        -trades["entry_price"] * position_size - fees,
    )

    return pnl, trades


def monte_carlo_ror(
    trade_pnls: np.ndarray,
    bankroll: float,
    n_trades: int,
    n_sims: int = N_SIMS,
) -> dict:
    """Monte Carlo risk of ruin simulation.

    Samples n_trades from empirical PnL distribution with replacement.
    Returns ruin probability and drawdown statistics.
    """
    ruin = 0
    max_drawdowns = []
    final_bankrolls = []

    for _ in range(n_sims):
        sim_pnls = np.random.choice(trade_pnls, size=n_trades, replace=True)
        equity = np.cumsum(sim_pnls) + bankroll
        peak = np.maximum.accumulate(equity)
        drawdown = equity - peak

        if equity.min() <= 0:
            ruin += 1

        max_drawdowns.append(drawdown.min())
        final_bankrolls.append(equity[-1])

    max_drawdowns = np.array(max_drawdowns)
    final_bankrolls = np.array(final_bankrolls)

    return {
        "ror": ruin / n_sims,
        "max_dd_median": np.median(max_drawdowns),
        "max_dd_p5": np.percentile(max_drawdowns, 5),
        "max_dd_p1": np.percentile(max_drawdowns, 1),
        "final_median": np.median(final_bankrolls),
        "final_p5": np.percentile(final_bankrolls, 5),
        "final_p95": np.percentile(final_bankrolls, 95),
        "prob_profit": (final_bankrolls > bankroll).mean(),
        "prob_double": (final_bankrolls > bankroll * 2).mean(),
    }


def degraded_pnl(trade_pnls: np.ndarray, trades_df: pd.DataFrame,
                  wr_penalty: float, position_size: float) -> np.ndarray:
    """Create a degraded PnL array simulating lower live win rate.

    Randomly flips some winners to losers to reduce WR by wr_penalty percentage points.
    """
    pnl = trade_pnls.copy()
    labels = trades_df["label"].values.copy()
    entries = trades_df["entry_price"].values

    winners = np.where(labels == 1)[0]
    n_flip = int(len(labels) * wr_penalty)
    if n_flip > len(winners):
        n_flip = len(winners)

    flip_idx = np.random.choice(winners, size=n_flip, replace=False)
    for idx in flip_idx:
        ep = entries[idx]
        fee = polymarket_fee(ep) * position_size
        # Was a winner, now a loser
        pnl[idx] = -ep * position_size - fee

    return pnl


def main():
    position_size = 5.0

    print("=" * 80)
    print(f"RISK OF RUIN ANALYSIS — v3 XGBoost + Vol Filter (${position_size:.0f}/trade)")
    print("=" * 80)

    print("\nRunning walk-forward backtest to build OOS trade PnL array...")
    trade_pnls, trades_df = build_trade_pnl_array(position_size)

    n = len(trade_pnls)
    wr = (trades_df["label"] == 1).mean()
    avg_win = trade_pnls[trade_pnls > 0].mean() if (trade_pnls > 0).sum() > 0 else 0
    avg_loss = trade_pnls[trade_pnls < 0].mean() if (trade_pnls < 0).sum() > 0 else 0
    avg_entry = trades_df["entry_price"].mean()

    print(f"\n  OOS trades: {n}")
    print(f"  Win rate: {wr:.1%}")
    print(f"  Avg entry price: ${avg_entry:.3f}")
    print(f"  Avg win: ${avg_win:+.3f}")
    print(f"  Avg loss: ${avg_loss:+.3f}")
    print(f"  Avg PnL/trade: ${trade_pnls.mean():+.4f}")
    print(f"  Median PnL/trade: ${np.median(trade_pnls):+.4f}")
    print(f"  Std PnL/trade: ${trade_pnls.std():.4f}")
    print(f"  Total PnL: ${trade_pnls.sum():+.2f}")

    # --- Kelly Criterion ---
    print(f"\n{'='*80}")
    print("KELLY CRITERION")
    print(f"{'='*80}")

    p = wr
    # For binary payoff: win pays (1-entry)/entry, lose pays -1
    avg_odds = avg_win / abs(avg_loss) if abs(avg_loss) > 0 else 1.0
    kelly_frac = p - (1 - p) / avg_odds
    print(f"  Win probability: {p:.3f}")
    print(f"  Win/loss ratio: {avg_odds:.3f}")
    print(f"  Full Kelly fraction: {kelly_frac:.3f}")
    print(f"  Half Kelly fraction: {kelly_frac/2:.3f}")
    print(f"  Quarter Kelly fraction: {kelly_frac/4:.3f}")
    print(f"  At $5/trade, Kelly-optimal bankroll: ${position_size / kelly_frac:.0f}" if kelly_frac > 0 else "  Kelly says don't bet")
    print(f"  At $5/trade, half-Kelly bankroll: ${position_size / (kelly_frac/2):.0f}" if kelly_frac > 0 else "")

    # --- Risk of Ruin by bankroll ---
    print(f"\n{'='*80}")
    print("RISK OF RUIN BY STARTING BANKROLL")
    print(f"  ({N_SIMS:,} simulations, {n} trades per sim, ${position_size:.0f}/trade)")
    print(f"{'='*80}")

    print(f"\n  {'Bankroll':>10s}  {'RoR':>6s}  {'MedDD':>8s}  {'P5 DD':>8s}  {'P1 DD':>8s}  "
          f"{'MedFinal':>10s}  {'P(profit)':>10s}  {'P(2x)':>7s}")
    print("  " + "-" * 85)

    for bankroll in [50, 100, 200, 300, 500, 750, 1000]:
        r = monte_carlo_ror(trade_pnls, bankroll, n_trades=n)
        print(f"  ${bankroll:>8d}  {r['ror']:>5.2%}  ${r['max_dd_median']:>+6.0f}  ${r['max_dd_p5']:>+6.0f}  "
              f"${r['max_dd_p1']:>+6.0f}  ${r['final_median']:>8.0f}  {r['prob_profit']:>9.1%}  {r['prob_double']:>6.1%}")

    # --- Risk of Ruin by horizon (number of trades) ---
    print(f"\n{'='*80}")
    print("RISK OF RUIN BY TRADING HORIZON ($500 bankroll)")
    print(f"{'='*80}")

    print(f"\n  {'Trades':>8s}  {'~Days':>6s}  {'RoR':>6s}  {'MedDD':>8s}  {'MedFinal':>10s}  "
          f"{'P(profit)':>10s}  {'P(2x)':>7s}")
    print("  " + "-" * 70)

    trades_per_day = 70  # from walk-forward: ~70 trades/day with vol filter
    for n_trades in [100, 200, 500, 1000, 2000, 5000]:
        r = monte_carlo_ror(trade_pnls, 500, n_trades=n_trades)
        days = n_trades / trades_per_day
        print(f"  {n_trades:>8d}  {days:>5.1f}  {r['ror']:>5.2%}  ${r['max_dd_median']:>+6.0f}  "
              f"${r['final_median']:>8.0f}  {r['prob_profit']:>9.1%}  {r['prob_double']:>6.1%}")

    # --- Sensitivity to WR degradation ---
    print(f"\n{'='*80}")
    print("SENSITIVITY: WHAT IF LIVE WR IS LOWER THAN BACKTEST?")
    print(f"  ($500 bankroll, {n} trades, ${position_size:.0f}/trade)")
    print(f"{'='*80}")

    print(f"\n  {'WR Penalty':>10s}  {'Effective WR':>12s}  {'RoR':>6s}  {'Avg PnL':>9s}  "
          f"{'MedFinal':>10s}  {'P(profit)':>10s}")
    print("  " + "-" * 70)

    for penalty_pp in [0, 2, 5, 8, 10, 15, 20]:
        penalty = penalty_pp / 100.0
        if penalty == 0:
            degraded = trade_pnls
        else:
            degraded = degraded_pnl(trade_pnls, trades_df, penalty, position_size)

        effective_wr = wr - penalty
        r = monte_carlo_ror(degraded, 500, n_trades=n)
        print(f"  {penalty_pp:>8d}pp  {effective_wr:>11.1%}  {r['ror']:>5.2%}  "
              f"${degraded.mean():>+7.4f}  ${r['final_median']:>8.0f}  {r['prob_profit']:>9.1%}")

    # --- Drawdown Distribution ---
    print(f"\n{'='*80}")
    print("DRAWDOWN DISTRIBUTION ($500 bankroll, full horizon)")
    print(f"{'='*80}")

    dd_results = []
    for _ in range(N_SIMS):
        sim_pnls = np.random.choice(trade_pnls, size=n, replace=True)
        equity = np.cumsum(sim_pnls) + 500
        peak = np.maximum.accumulate(equity)
        dd = equity - peak
        dd_results.append(dd.min())

    dd_arr = np.array(dd_results)
    for pct in [50, 25, 10, 5, 2, 1]:
        val = np.percentile(dd_arr, pct)
        print(f"  {pct:>3d}th percentile max drawdown: ${val:+.2f}")

    print(f"\n  Mean max drawdown: ${dd_arr.mean():+.2f}")
    print(f"  Worst max drawdown: ${dd_arr.min():+.2f}")

    # --- Consecutive Loss Analysis ---
    print(f"\n{'='*80}")
    print("CONSECUTIVE LOSS STREAKS (empirical from {n} OOS trades)")
    print(f"{'='*80}")

    labels = (trade_pnls > 0).astype(int)
    max_loss_streak = 0
    current_streak = 0
    streaks = []

    for won in labels:
        if won == 0:
            current_streak += 1
        else:
            if current_streak > 0:
                streaks.append(current_streak)
            current_streak = 0
    if current_streak > 0:
        streaks.append(current_streak)

    streaks = np.array(streaks) if streaks else np.array([0])
    print(f"  Longest loss streak: {streaks.max()}")
    print(f"  Average loss streak: {streaks.mean():.1f}")
    print(f"  Max dollar loss in longest streak: ${streaks.max() * abs(avg_loss):.2f}")
    print(f"  As % of $500 bankroll: {streaks.max() * abs(avg_loss) / 500:.1%}")

    # --- Summary ---
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"  Strategy: v3 XGBoost, conf>=0.70, vol p25 filter, 120s window")
    print(f"  Position size: ${position_size:.0f}/trade")
    print(f"  OOS win rate: {wr:.1%}")
    print(f"  Avg PnL/trade: ${trade_pnls.mean():+.4f}")
    print(f"  Kelly fraction: {kelly_frac:.3f} (half-Kelly bankroll: ${position_size / (kelly_frac/2):.0f})" if kelly_frac > 0 else "")
    r500 = monte_carlo_ror(trade_pnls, 500, n_trades=n)
    print(f"  RoR at $500: {r500['ror']:.2%}")
    print(f"  Median max DD at $500: ${r500['max_dd_median']:+.0f}")
    print(f"  Breakeven WR (approx): {avg_entry:.1%} (entry price)")
    print(f"  Safety margin: {wr - avg_entry:.1%} ({(wr - avg_entry) / avg_entry:.0%} above breakeven)")


if __name__ == "__main__":
    main()
