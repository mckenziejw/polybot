#!/usr/bin/env python3
"""
Risk of ruin analysis for v3.3 XGBoost strategy with limit-at-prediction.

Fully vectorized Monte Carlo (numpy batch operations).
Uses walk-forward OOS predictions with limit-at-prediction entry logic.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from xgboost_flexible import build_dataset, get_feature_columns, compute_vol_threshold

warnings.filterwarnings("ignore")

FEATURE_CACHE = Path("data/xgb_features_v3.parquet")
N_SIMS = 50_000

# v3.3 sweep-optimized params
PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "max_depth": 2,
    "learning_rate": 0.005016,
    "subsample": 0.930265,
    "colsample_bytree": 0.985988,
    "min_child_weight": 41,
    "lambda": 0.224145,
    "alpha": 0.827467,
    "gamma": 1.705416,
    "seed": 42,
    "verbosity": 0,
}


def polymarket_fee(price: float) -> float:
    return 100 * price * 0.25 * (price * (1 - price)) ** 2 / 10000


def build_trade_array() -> pd.DataFrame:
    """Walk-forward backtest returning per-trade data."""
    df = pd.read_parquet(FEATURE_CACHE)
    feature_cols = get_feature_columns(df)

    slug_order = df.groupby("slug")["open_ts"].first().sort_values()
    ordered_slugs = slug_order.index.tolist()
    n_markets = len(ordered_slugs)

    min_train = 1000
    step = 200
    val_frac = 0.15

    all_rows = []
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
        X_val = val_data[feature_cols].values.astype(np.float32)
        X_test = test_data[feature_cols].values.astype(np.float32)

        for X in [X_train, X_val, X_test]:
            X[~np.isfinite(X)] = 0.0

        dtrain = xgb.DMatrix(X_train, label=train_data["label"].values, feature_names=feature_cols)
        dval = xgb.DMatrix(X_val, label=val_data["label"].values, feature_names=feature_cols)
        dtest = xgb.DMatrix(X_test, label=test_data["label"].values, feature_names=feature_cols)

        model = xgb.train(PARAMS, dtrain, num_boost_round=2000,
                          evals=[(dval, "val")], early_stopping_rounds=50,
                          verbose_eval=False)

        preds = model.predict(dtest)

        for i, (idx, row) in enumerate(test_data.iterrows()):
            mid = row.get("w120_mid", 0.70)
            spread = row.get("w120_spread", 0.02)
            if pd.isna(mid):
                mid = 0.70
            if pd.isna(spread):
                spread = 0.02
            best_ask = mid + spread / 2
            all_rows.append({
                "pred": preds[i],
                "label": row["label"],
                "entry_mid": mid,
                "best_ask": best_ask,
                "observe_time_s": row["observe_time_s"],
            })

        train_end = test_end

    return pd.DataFrame(all_rows)


def compute_limit_entries(trades_df: pd.DataFrame, conf_threshold: float = 0.60,
                          bet_dollars: float = 20.0):
    """Compute per-trade PnL using limit-at-prediction entry logic."""
    w120 = trades_df[trades_df["observe_time_s"] == 120].copy()
    filtered = w120[w120["pred"] >= conf_threshold].copy()

    pnls, entries, labels_out, etypes = [], [], [], []

    for _, row in filtered.iterrows():
        pred, best_ask, label = row["pred"], row["best_ask"], row["label"]

        if pred >= best_ask:
            entry = best_ask
            etype = "instant"
        else:
            entry = np.floor(pred * 100) / 100
            etype = "limit"

        if entry < 0.45 or entry > 0.85 or entry <= 0 or entry >= 1.0:
            continue

        n_tokens = bet_dollars / entry
        fee = polymarket_fee(entry) * bet_dollars
        pnl = n_tokens * (1.0 - entry) - fee if label == 1 else -n_tokens * entry - fee

        pnls.append(pnl)
        entries.append(entry)
        labels_out.append(label)
        etypes.append(etype)

    return np.array(pnls), np.array(entries), np.array(labels_out), np.array(etypes)


def mc_fixed_size(trade_pnls: np.ndarray, bankroll: float,
                  n_trades: int, n_sims: int = N_SIMS) -> dict:
    """Vectorized Monte Carlo for fixed position sizing."""
    idx = np.random.randint(0, len(trade_pnls), size=(n_sims, n_trades))
    sim_pnls = trade_pnls[idx]
    equity = np.cumsum(sim_pnls, axis=1) + bankroll
    peaks = np.maximum.accumulate(equity, axis=1)
    drawdowns = equity - peaks

    min_equity = equity.min(axis=1)
    max_dd = drawdowns.min(axis=1)
    final = equity[:, -1]

    return {
        "ror": (min_equity <= 0).mean(),
        "max_dd_median": np.median(max_dd),
        "max_dd_p5": np.percentile(max_dd, 5),
        "max_dd_p1": np.percentile(max_dd, 1),
        "final_median": np.median(final),
        "final_p5": np.percentile(final, 5),
        "final_p25": np.percentile(final, 25),
        "final_p75": np.percentile(final, 75),
        "final_p95": np.percentile(final, 95),
        "prob_profit": (final > bankroll).mean(),
        "prob_double": (final > bankroll * 2).mean(),
    }


def main():
    bankroll = 1600.0
    bet_dollars = 20.0

    print("=" * 80)
    print("RISK OF RUIN — v3.3 XGBoost, Limit-at-Prediction, conf>=0.60")
    print("=" * 80)

    print("\nBuilding walk-forward OOS trade array...")
    trades_df = build_trade_array()
    print(f"  Raw OOS samples: {len(trades_df)}")

    pnls, entries, labels, etypes = compute_limit_entries(
        trades_df, conf_threshold=0.60, bet_dollars=bet_dollars
    )

    n = len(pnls)
    wr = labels.mean()
    n_instant = (etypes == "instant").sum()
    n_limit = (etypes == "limit").sum()

    print(f"\n  Tradeable: {n} ({n_instant} instant + {n_limit} limit)")
    print(f"  Win rate: {wr:.1%}")
    print(f"  Avg entry: {entries.mean():.3f} (instant: {entries[etypes == 'instant'].mean():.3f}, limit: {entries[etypes == 'limit'].mean():.3f})")
    print(f"  Avg PnL/trade: ${pnls.mean():+.3f}")
    print(f"  Std PnL/trade: ${pnls.std():.3f}")
    print(f"  Total PnL: ${pnls.sum():+.2f}")

    # --- Kelly info (just the numbers, no simulation) ---
    avg_win = pnls[pnls > 0].mean()
    avg_loss = abs(pnls[pnls < 0].mean())
    b = avg_win / avg_loss
    kelly = wr - (1 - wr) / b
    print(f"\n  Kelly fraction: {kelly:.4f} ({kelly*100:.2f}%)")
    print(f"  Half-Kelly: {kelly/2:.4f} → ${bankroll * kelly / 2:.2f}/trade at ${bankroll:.0f}")

    # ==========================================================================
    # Fixed-size RoR by bet size
    # ==========================================================================
    print(f"\n{'='*80}")
    print(f"RISK OF RUIN BY BET SIZE (${bankroll:.0f} bankroll, 2000 trades, {N_SIMS:,} sims)")
    print(f"{'='*80}")

    print(f"\n  {'Bet':>6s}  {'RoR':>6s}  {'MedDD':>8s}  {'P5 DD':>8s}  "
          f"{'P1 DD':>8s}  {'Med Final':>10s}  {'P(profit)':>10s}  {'P(2x)':>7s}")
    print("  " + "-" * 80)

    for bet_size in [10, 20, 40, 80, 100]:
        scaled = pnls * (bet_size / bet_dollars)
        r = mc_fixed_size(scaled, bankroll, n_trades=2000)
        print(f"  ${bet_size:>4d}  {r['ror']:>5.2%}  ${r['max_dd_median']:>+6.0f}  "
              f"${r['max_dd_p5']:>+6.0f}  ${r['max_dd_p1']:>+6.0f}  "
              f"${r['final_median']:>8.0f}  {r['prob_profit']:>9.1%}  {r['prob_double']:>6.1%}")

    # ==========================================================================
    # Fixed $20 by horizon
    # ==========================================================================
    print(f"\n{'='*80}")
    print(f"FIXED $20/TRADE — BY HORIZON (${bankroll:.0f} bankroll)")
    print(f"{'='*80}")

    horizons = [
        ("1 day", 132),
        ("1 week", 924),
        ("2 weeks", 1848),
        ("1 month", 3960),
        ("3 months", 10000),
    ]

    print(f"\n  {'Horizon':>10s}  {'Trades':>7s}  {'RoR':>6s}  {'MedDD':>8s}  "
          f"{'P5':>10s}  {'P25':>10s}  {'Median':>10s}  {'P75':>10s}  {'P95':>10s}")
    print("  " + "-" * 95)

    for label, n_trades in horizons:
        r = mc_fixed_size(pnls, bankroll, n_trades=n_trades)
        print(f"  {label:>10s}  {n_trades:>7d}  {r['ror']:>5.2%}  ${r['max_dd_median']:>+6.0f}  "
              f"${r['final_p5']:>8.0f}  ${r['final_p25']:>8.0f}  ${r['final_median']:>8.0f}  "
              f"${r['final_p75']:>8.0f}  ${r['final_p95']:>8.0f}")

    # ==========================================================================
    # WR degradation sensitivity
    # ==========================================================================
    print(f"\n{'='*80}")
    print("SENSITIVITY: WHAT IF LIVE WR IS LOWER? ($20/trade, 1-month)")
    print(f"{'='*80}")

    print(f"\n  {'WR Drop':>8s}  {'Eff WR':>7s}  {'RoR':>6s}  {'Avg PnL':>9s}  "
          f"{'Med Final':>10s}  {'P(profit)':>10s}")
    print("  " + "-" * 60)

    for penalty_pp in [0, 3, 5, 8, 10, 15]:
        degraded = pnls.copy()
        if penalty_pp > 0:
            winners = np.where(labels == 1)[0]
            n_flip = min(int(len(labels) * penalty_pp / 100), len(winners))
            flip_idx = np.random.choice(winners, size=n_flip, replace=False)
            for fi in flip_idx:
                ep = entries[fi]
                fee = polymarket_fee(ep) * bet_dollars
                n_tokens = bet_dollars / ep
                degraded[fi] = -n_tokens * ep - fee

        eff_wr = wr - penalty_pp / 100
        r = mc_fixed_size(degraded, bankroll, n_trades=3960)
        print(f"  {penalty_pp:>6d}pp  {eff_wr:>6.1%}  {r['ror']:>5.2%}  "
              f"${degraded.mean():>+7.3f}  ${r['final_median']:>8.0f}  {r['prob_profit']:>9.1%}")

    # ==========================================================================
    # Loss streaks
    # ==========================================================================
    print(f"\n{'='*80}")
    print(f"CONSECUTIVE LOSS STREAKS ({n} OOS trades)")
    print(f"{'='*80}")

    won = (pnls > 0).astype(int)
    streaks = []
    current = 0
    for w in won:
        if w == 0:
            current += 1
        else:
            if current > 0:
                streaks.append(current)
            current = 0
    if current > 0:
        streaks.append(current)
    streaks = np.array(streaks) if streaks else np.array([0])

    max_loss_dollars = streaks.max() * abs(pnls[pnls < 0].mean())
    print(f"  Longest streak: {streaks.max()}")
    print(f"  Average streak: {streaks.mean():.1f}")
    print(f"  Max streak dollar loss: ${max_loss_dollars:.2f} ({max_loss_dollars/bankroll:.1%} of bankroll)")


if __name__ == "__main__":
    main()
