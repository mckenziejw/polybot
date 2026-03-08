#!/usr/bin/env python3
"""
Position sizing analysis for v3.2 XGBoost strategy.

Uses walk-forward OOS predictions with:
  - v3.2 params (deduped features, sweep-optimized)
  - No vol filter
  - 0.90 entry cap
  - $1600 bankroll

Computes Kelly criterion, risk-of-ruin at various position sizes,
and optimal sizing recommendations.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from xgboost_flexible import build_dataset, get_feature_columns, DROP_FEATURES

warnings.filterwarnings("ignore")

N_SIMS = 50_000
BANKROLL = 1600.0

PARAMS_V32 = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "max_depth": 2,
    "learning_rate": 0.014790,
    "subsample": 0.838960,
    "colsample_bytree": 0.951185,
    "min_child_weight": 42,
    "lambda": 0.630567,
    "alpha": 0.010124,
    "gamma": 0.457778,
    "seed": 42,
    "verbosity": 0,
}


def polymarket_fee(price: float) -> float:
    """Taker fee for crypto markets."""
    return 100 * price * 0.25 * (price * (1 - price)) ** 2 / 10000


def build_trade_array(position_size: float = 5.0):
    """Walk-forward backtest → OOS trade PnL array.

    v3.2 params, no vol filter, 0.90 entry cap.
    """
    dataset = build_dataset(
        data_dir=Path("data/telonex_book_snapshots"),
        resolution_cache_path=Path("data/resolution_cache.json"),
        btc_path=Path("data/btc_quotes/btcusdt_quotes.parquet"),
        multi_window=True,
    )

    feature_cols = get_feature_columns(dataset)
    dataset_120 = dataset[dataset["observe_time_s"] == 120].copy()
    slug_order = dataset.groupby("slug")["open_ts"].first().sort_values()
    slugs = slug_order.index.tolist()
    n_slugs = len(slugs)

    mid_cols = sorted([c for c in dataset.columns if c.endswith("_mid") and c.startswith("w")])
    entry_col = mid_cols[-1]

    # Walk-forward with expanding window
    n_windows = 8
    initial_train_pct = 0.50
    test_chunk = (1.0 - initial_train_pct) / n_windows

    all_trades = []

    for w in range(n_windows):
        train_end_pct = initial_train_pct + w * test_chunk
        test_end_pct = train_end_pct + test_chunk
        train_end_idx = int(n_slugs * train_end_pct)
        test_start_idx = train_end_idx
        test_end_idx = min(int(n_slugs * test_end_pct), n_slugs)

        train_slugs = set(slugs[:train_end_idx])
        test_slugs_set = set(slugs[test_start_idx:test_end_idx])

        train_all = dataset[dataset["slug"].isin(train_slugs)]
        val_start = int(len(train_slugs) * 0.9)
        train_slug_list = slug_order.index[:train_end_idx].tolist()
        val_slugs_set = set(train_slug_list[val_start:])
        train_only = train_all[~train_all["slug"].isin(val_slugs_set)]
        val_only = train_all[train_all["slug"].isin(val_slugs_set)]
        test_df = dataset_120[dataset_120["slug"].isin(test_slugs_set)]

        if len(test_df) < 10:
            continue

        X_train = train_only[feature_cols].values.astype(np.float32)
        X_val = val_only[feature_cols].values.astype(np.float32)
        X_test = test_df[feature_cols].values.astype(np.float32)
        for X in [X_train, X_val, X_test]:
            X[~np.isfinite(X)] = 0.0

        dtrain = xgb.DMatrix(X_train, label=train_only["label"].values, feature_names=feature_cols)
        dval = xgb.DMatrix(X_val, label=val_only["label"].values, feature_names=feature_cols)
        dtest = xgb.DMatrix(X_test, label=test_df["label"].values, feature_names=feature_cols)

        model = xgb.train(PARAMS_V32, dtrain, num_boost_round=2000,
                          evals=[(dval, "val")], early_stopping_rounds=50,
                          verbose_eval=False)

        preds = model.predict(dtest)
        entries = test_df[entry_col].values
        labels = test_df["label"].values

        for i in range(len(preds)):
            if preds[i] < 0.72:
                continue
            entry = entries[i]
            if entry <= 0 or entry >= 1.0 or entry > 0.90:
                continue

            all_trades.append({
                "pred": preds[i],
                "label": int(labels[i]),
                "entry": entry,
            })

    trades_df = pd.DataFrame(all_trades)
    return trades_df


def compute_pnl_array(trades_df: pd.DataFrame, position_size: float) -> np.ndarray:
    """Compute PnL array for a given position size."""
    entries = trades_df["entry"].values
    labels = trades_df["label"].values

    pnl = np.zeros(len(trades_df))
    for i in range(len(trades_df)):
        entry = entries[i]
        n_tokens = position_size / entry
        fee = polymarket_fee(entry) * position_size
        if labels[i] == 1:
            pnl[i] = n_tokens * (1.0 - entry) - fee
        else:
            pnl[i] = -n_tokens * entry - fee
    return pnl


def monte_carlo(trade_pnls, bankroll, n_trades, n_sims=N_SIMS):
    """Monte Carlo risk-of-ruin simulation."""
    ruin = 0
    max_dds = []
    finals = []

    for _ in range(n_sims):
        sim = np.random.choice(trade_pnls, size=n_trades, replace=True)
        equity = np.cumsum(sim) + bankroll
        peak = np.maximum.accumulate(equity)
        dd = equity - peak

        if equity.min() <= 0:
            ruin += 1
        max_dds.append(dd.min())
        finals.append(equity[-1])

    max_dds = np.array(max_dds)
    finals = np.array(finals)
    return {
        "ror": ruin / n_sims,
        "max_dd_median": np.median(max_dds),
        "max_dd_p5": np.percentile(max_dds, 5),
        "max_dd_p1": np.percentile(max_dds, 1),
        "final_median": np.median(finals),
        "final_p5": np.percentile(finals, 5),
        "final_p95": np.percentile(finals, 95),
        "prob_profit": (finals > bankroll).mean(),
        "prob_double": (finals > bankroll * 2).mean(),
    }


def main():
    print("=" * 80)
    print(f"POSITION SIZING ANALYSIS — v3.2 XGBoost (${BANKROLL:.0f} bankroll)")
    print("  No vol filter, 0.90 entry cap, conf >= 0.72")
    print("=" * 80)

    print("\nBuilding walk-forward OOS trade array...")
    trades_df = build_trade_array()
    n = len(trades_df)
    wr = trades_df["label"].mean()
    avg_entry = trades_df["entry"].mean()
    print(f"  OOS trades: {n}")
    print(f"  Win rate: {wr:.1%}")
    print(f"  Avg entry: {avg_entry:.3f}")

    # --- Base stats at $5 ---
    pnl_5 = compute_pnl_array(trades_df, 5.0)
    avg_win = pnl_5[pnl_5 > 0].mean()
    avg_loss = pnl_5[pnl_5 < 0].mean()
    print(f"\n  At $5/trade:")
    print(f"    Avg win: ${avg_win:+.3f}")
    print(f"    Avg loss: ${avg_loss:+.3f}")
    print(f"    Avg PnL/trade: ${pnl_5.mean():+.4f}")
    print(f"    Std PnL/trade: ${pnl_5.std():.4f}")
    print(f"    Total PnL: ${pnl_5.sum():+.2f}")

    # --- Kelly Criterion ---
    print(f"\n{'='*80}")
    print("KELLY CRITERION")
    print(f"{'='*80}")

    p = wr
    avg_odds = avg_win / abs(avg_loss) if abs(avg_loss) > 0 else 1.0
    kelly_frac = p - (1 - p) / avg_odds

    print(f"  Win probability: {p:.3f}")
    print(f"  Win/loss ratio (at $5): {avg_odds:.3f}")
    print(f"  Full Kelly fraction: {kelly_frac:.4f}")
    print(f"  Half Kelly fraction: {kelly_frac/2:.4f}")
    print(f"  Quarter Kelly fraction: {kelly_frac/4:.4f}")

    if kelly_frac > 0:
        full_kelly_bet = BANKROLL * kelly_frac
        half_kelly_bet = BANKROLL * kelly_frac / 2
        quarter_kelly_bet = BANKROLL * kelly_frac / 4
        print(f"\n  On ${BANKROLL:.0f} bankroll:")
        print(f"    Full Kelly bet: ${full_kelly_bet:.2f}")
        print(f"    Half Kelly bet: ${half_kelly_bet:.2f}")
        print(f"    Quarter Kelly bet: ${quarter_kelly_bet:.2f}")
    else:
        print("  Kelly says don't bet!")
        return

    # --- Risk of Ruin at different position sizes ---
    print(f"\n{'='*80}")
    print(f"RISK OF RUIN BY POSITION SIZE (${BANKROLL:.0f} bankroll, {n} trades)")
    print(f"{'='*80}")

    print(f"\n  {'Bet $':>7s}  {'% Bank':>7s}  {'RoR':>6s}  {'MedDD':>8s}  {'P5 DD':>8s}  "
          f"{'P1 DD':>8s}  {'MedFinal':>10s}  {'P(profit)':>10s}  {'P(2x)':>7s}")
    print("  " + "-" * 90)

    for bet_size in [5, 10, 15, 20, 25, 30, 40, 50, 75, 100]:
        pnl = compute_pnl_array(trades_df, bet_size)
        r = monte_carlo(pnl, BANKROLL, n_trades=n)
        pct_bank = bet_size / BANKROLL * 100
        print(f"  ${bet_size:>5d}  {pct_bank:>6.1f}%  {r['ror']:>5.2%}  ${r['max_dd_median']:>+7.0f}  "
              f"${r['max_dd_p5']:>+7.0f}  ${r['max_dd_p1']:>+7.0f}  ${r['final_median']:>9.0f}  "
              f"{r['prob_profit']:>9.1%}  {r['prob_double']:>6.1%}")

    # --- Time to double ---
    print(f"\n{'='*80}")
    print("TIME TO PROFIT TARGETS (by position size)")
    print(f"{'='*80}")

    trades_per_day = 100  # ~100 eligible trades per day at conf>=0.72 no vol filter

    print(f"\n  Assuming ~{trades_per_day} trades/day")
    print(f"\n  {'Bet $':>7s}  {'$/day (med)':>11s}  {'Days to 2x':>11s}  {'Days to +$500':>14s}  "
          f"{'30d P(profit)':>14s}")
    print("  " + "-" * 70)

    for bet_size in [5, 10, 15, 20, 25, 30, 40, 50]:
        pnl = compute_pnl_array(trades_df, bet_size)
        avg_pnl_per_trade = pnl.mean()
        daily_ev = avg_pnl_per_trade * trades_per_day

        # Sim 30 days
        r30 = monte_carlo(pnl, BANKROLL, n_trades=trades_per_day * 30)
        # Days to double (median)
        if daily_ev > 0:
            days_2x = BANKROLL / daily_ev
            days_500 = 500 / daily_ev
        else:
            days_2x = float("inf")
            days_500 = float("inf")

        print(f"  ${bet_size:>5d}  ${daily_ev:>+9.2f}  {days_2x:>10.0f}  {days_500:>13.0f}  "
              f"{r30['prob_profit']:>13.1%}")

    # --- Sensitivity: what if live WR is worse? ---
    print(f"\n{'='*80}")
    print("SENSITIVITY: WR DEGRADATION (live vs backtest)")
    print(f"  (${BANKROLL:.0f} bankroll, $20/trade, {n} trades)")
    print(f"{'='*80}")

    print(f"\n  {'WR Hit':>8s}  {'Eff. WR':>8s}  {'RoR':>6s}  {'Avg $/tr':>9s}  "
          f"{'MedFinal':>10s}  {'P(profit)':>10s}")
    print("  " + "-" * 65)

    for penalty_pp in [0, 2, 4, 6, 8, 10, 12, 15]:
        penalty = penalty_pp / 100.0
        effective_wr = wr - penalty

        # Build degraded PnL by flipping some winners to losers
        pnl = compute_pnl_array(trades_df, 20.0)
        if penalty > 0:
            labels = trades_df["label"].values.copy()
            entries = trades_df["entry"].values
            winners = np.where(labels == 1)[0]
            n_flip = int(len(labels) * penalty)
            if n_flip > len(winners):
                n_flip = len(winners)
            np.random.seed(42)
            flip_idx = np.random.choice(winners, size=n_flip, replace=False)
            for idx in flip_idx:
                ep = entries[idx]
                fee = polymarket_fee(ep) * 20.0
                pnl[idx] = -(ep * 20.0) / ep * ep - fee  # loss = -bet
                pnl[idx] = -20.0 - fee

        r = monte_carlo(pnl, BANKROLL, n_trades=n)
        print(f"  {penalty_pp:>6d}pp  {effective_wr:>7.1%}  {r['ror']:>5.2%}  "
              f"${pnl.mean():>+7.4f}  ${r['final_median']:>8.0f}  {r['prob_profit']:>9.1%}")

    # --- Drawdown distribution at recommended size ---
    print(f"\n{'='*80}")
    print("DRAWDOWN DISTRIBUTION ($20/trade, $1600 bankroll)")
    print(f"{'='*80}")

    pnl_20 = compute_pnl_array(trades_df, 20.0)
    dd_results = []
    for _ in range(N_SIMS):
        sim = np.random.choice(pnl_20, size=n, replace=True)
        equity = np.cumsum(sim) + BANKROLL
        peak = np.maximum.accumulate(equity)
        dd_results.append((equity - peak).min())

    dd_arr = np.array(dd_results)
    for pct in [50, 25, 10, 5, 2, 1]:
        val = np.percentile(dd_arr, pct)
        print(f"  {pct:>3d}th percentile max drawdown: ${val:+.2f} ({val/BANKROLL:.1%} of bankroll)")

    print(f"\n  Mean max DD: ${dd_arr.mean():+.2f} ({dd_arr.mean()/BANKROLL:.1%})")
    print(f"  Worst max DD: ${dd_arr.min():+.2f} ({dd_arr.min()/BANKROLL:.1%})")

    # --- Consecutive loss analysis ---
    print(f"\n{'='*80}")
    print("CONSECUTIVE LOSS STREAKS (from OOS trades)")
    print(f"{'='*80}")

    labels = (pnl_5 > 0).astype(int)
    streaks = []
    current = 0
    for won in labels:
        if won == 0:
            current += 1
        else:
            if current > 0:
                streaks.append(current)
            current = 0
    if current > 0:
        streaks.append(current)
    streaks = np.array(streaks) if streaks else np.array([0])

    print(f"  Longest loss streak: {streaks.max()}")
    print(f"  Average loss streak: {streaks.mean():.1f}")
    for bet in [5, 10, 20, 30, 50]:
        max_streak_loss = streaks.max() * bet
        print(f"  Max streak loss at ${bet}/trade: ${max_streak_loss:.0f} "
              f"({max_streak_loss/BANKROLL:.1%} of bankroll)")

    # --- Summary ---
    print(f"\n{'='*80}")
    print("SUMMARY & RECOMMENDATION")
    print(f"{'='*80}")
    print(f"  Bankroll: ${BANKROLL:.0f}")
    print(f"  Strategy: v3.2 XGBoost, conf>=0.72, no vol filter, entry<=0.90")
    print(f"  OOS WR: {wr:.1%}, avg entry: {avg_entry:.3f}")
    print(f"  Kelly fraction: {kelly_frac:.4f}")
    print(f"  Full Kelly bet: ${BANKROLL * kelly_frac:.2f}")
    print(f"  Half Kelly bet: ${BANKROLL * kelly_frac / 2:.2f}")

    # Find max bet with RoR < 1%
    for bet_size in [5, 10, 15, 20, 25, 30, 40, 50, 75, 100]:
        pnl = compute_pnl_array(trades_df, bet_size)
        r = monte_carlo(pnl, BANKROLL, n_trades=n)
        if r["ror"] > 0.01:
            print(f"\n  Max bet with RoR < 1%: ${bet_size - 5}")
            break

    print(f"\n  Recommended: half-Kelly (${BANKROLL * kelly_frac / 2:.0f}/trade)")
    print(f"  Conservative: quarter-Kelly (${BANKROLL * kelly_frac / 4:.0f}/trade)")


if __name__ == "__main__":
    main()
