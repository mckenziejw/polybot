#!/usr/bin/env python3
"""
Analyze vol filter impact on portfolio metrics: coverage, drawdown, Sharpe.

Simulates trading the v3 XGBoost model at different confidence thresholds
with and without the vol filter, computing:
  - Coverage (trades per day)
  - Cumulative PnL
  - Max drawdown
  - Sharpe ratio (daily)
  - Risk of ruin (Monte Carlo)
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

warnings.filterwarnings("ignore")

FEATURE_CACHE = Path("data/xgb_features_v3.parquet")
MARKET_DURATION_S = 300  # 5 minutes


def polymarket_fee(price: float) -> float:
    """Taker fee as fraction of notional."""
    return 100 * price * 0.25 * (price * (1 - price)) ** 2 / 10000


def simulate_trading(test_df, y_pred_proba, y_test, conf_threshold, vol_filter_col=None, vol_threshold=None, position_size=10.0):
    """Simulate sequential trading and compute portfolio metrics."""

    # Use 120s window only (our actual trading window)
    w120 = test_df["observe_time_s"].values == 120
    conf_mask = y_pred_proba >= conf_threshold

    if vol_filter_col and vol_threshold is not None:
        vol_pass = test_df[vol_filter_col].values > vol_threshold
    else:
        vol_pass = np.ones(len(test_df), dtype=bool)

    trade_mask = w120 & conf_mask & vol_pass

    # Get trade-level data
    trade_idx = np.where(trade_mask)[0]
    if len(trade_idx) < 5:
        return None

    trades = []
    for idx in trade_idx:
        row = test_df.iloc[idx]
        pred = y_pred_proba[idx]
        actual = y_test[idx]

        # Entry price = leader mid at 120s (approximate)
        mid_cols = [c for c in test_df.columns if c == "w120_mid"]
        entry_price = row[mid_cols[0]] if mid_cols else 0.70

        # Fee on entry
        fee = polymarket_fee(entry_price) * position_size

        if actual == 1:  # winner
            pnl = (1.0 - entry_price) * position_size - fee
        else:  # loser
            pnl = -entry_price * position_size - fee

        trades.append({
            "open_ts": row["open_ts"],
            "slug": row["slug"],
            "pred": pred,
            "actual": actual,
            "entry_price": entry_price,
            "pnl": pnl,
            "fee": fee,
        })

    tdf = pd.DataFrame(trades).sort_values("open_ts").reset_index(drop=True)

    # Cumulative PnL
    tdf["cum_pnl"] = tdf["pnl"].cumsum()

    # Max drawdown
    tdf["peak"] = tdf["cum_pnl"].cummax()
    tdf["drawdown"] = tdf["cum_pnl"] - tdf["peak"]
    max_dd = tdf["drawdown"].min()

    # Win rate
    wr = tdf["actual"].mean()

    # Daily aggregation for Sharpe
    # Markets are every 5 minutes, ~288/day. Group by approximate day.
    tdf["day"] = (tdf["open_ts"] - tdf["open_ts"].iloc[0]) // 86400
    daily_pnl = tdf.groupby("day")["pnl"].sum()
    daily_trades = tdf.groupby("day")["pnl"].count()

    sharpe = daily_pnl.mean() / daily_pnl.std() * np.sqrt(365) if daily_pnl.std() > 0 else 0.0

    # Coverage: trades per day
    n_days = tdf["day"].nunique()
    trades_per_day = len(tdf) / max(n_days, 1)

    # Total PnL
    total_pnl = tdf["pnl"].sum()
    total_fees = tdf["fee"].sum()
    avg_pnl_per_trade = tdf["pnl"].mean()

    # Risk of ruin (Monte Carlo: what's P(bankroll hits 0) starting with $500?)
    bankroll_start = 500.0
    n_sims = 10000
    ruin_count = 0
    trade_pnls = tdf["pnl"].values

    for _ in range(n_sims):
        # Bootstrap: sample trades with replacement, same number of trades
        sim_pnls = np.random.choice(trade_pnls, size=len(trade_pnls), replace=True)
        cum = np.cumsum(sim_pnls) + bankroll_start
        if cum.min() <= 0:
            ruin_count += 1

    risk_of_ruin = ruin_count / n_sims

    return {
        "n_trades": len(tdf),
        "n_days": n_days,
        "trades_per_day": trades_per_day,
        "win_rate": wr,
        "total_pnl": total_pnl,
        "total_fees": total_fees,
        "avg_pnl_per_trade": avg_pnl_per_trade,
        "max_drawdown": max_dd,
        "sharpe": sharpe,
        "risk_of_ruin": risk_of_ruin,
        "cum_pnl_series": tdf["cum_pnl"].values,
        "daily_pnl": daily_pnl,
    }


def main():
    print("=" * 75)
    print("VOL FILTER PORTFOLIO ANALYSIS")
    print("=" * 75)

    # Load data and train model (same as xgboost_flexible.py)
    df = pd.read_parquet(FEATURE_CACHE)
    print(f"Loaded {len(df)} samples, {df['slug'].nunique()} markets")

    feature_cols = [c for c in df.columns if c not in ("label", "slug", "open_ts") and not c.startswith("_")]

    # Time-based split
    slug_order = df.groupby("slug")["open_ts"].first().sort_values()
    n_markets = len(slug_order)
    train_slugs = set(slug_order.index[:int(n_markets * 0.7)])
    val_slugs = set(slug_order.index[int(n_markets * 0.7):int(n_markets * 0.85)])
    test_slugs = set(slug_order.index[int(n_markets * 0.85):])

    train_df = df[df["slug"].isin(train_slugs)]
    val_df = df[df["slug"].isin(val_slugs)]
    test_df = df[df["slug"].isin(test_slugs)]

    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df["label"].values
    X_val = val_df[feature_cols].values.astype(np.float32)
    y_val = val_df["label"].values
    X_test = test_df[feature_cols].values.astype(np.float32)
    y_test = test_df["label"].values

    for X in [X_train, X_val, X_test]:
        X[~np.isfinite(X)] = 0.0

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_cols)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_cols)
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_cols)

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

    print("Training model...")
    model = xgb.train(params, dtrain, num_boost_round=2000,
                       evals=[(dval, "val")], early_stopping_rounds=50, verbose_eval=False)
    y_pred = model.predict(dtest)
    print(f"Model trained. Best iteration: {model.best_iteration}")

    # Vol threshold from training data
    train_vol = train_df["btc_vol_5m"].dropna()
    train_vol = train_vol[train_vol > 0]
    vol_p25 = float(train_vol.quantile(0.25))
    vol_p33 = float(train_vol.quantile(0.33))
    print(f"Vol thresholds: p25={vol_p25:.6f}, p33={vol_p33:.6f}")

    # Test period info
    test_ts = test_df.groupby("slug")["open_ts"].first().sort_values()
    from datetime import datetime, timezone
    start_dt = datetime.fromtimestamp(test_ts.iloc[0], tz=timezone.utc)
    end_dt = datetime.fromtimestamp(test_ts.iloc[-1], tz=timezone.utc)
    n_days = (end_dt - start_dt).days
    print(f"Test period: {start_dt.date()} to {end_dt.date()} ({n_days} days, {len(test_slugs)} markets)")

    # Run simulations
    configs = [
        ("No filter, conf>=0.65", 0.65, None, None),
        ("No filter, conf>=0.70", 0.70, None, None),
        ("No filter, conf>=0.75", 0.75, None, None),
        ("No filter, conf>=0.80", 0.80, None, None),
        ("Vol p25, conf>=0.65", 0.65, "btc_vol_5m", vol_p25),
        ("Vol p25, conf>=0.70", 0.70, "btc_vol_5m", vol_p25),
        ("Vol p25, conf>=0.75", 0.75, "btc_vol_5m", vol_p25),
        ("Vol p25, conf>=0.80", 0.80, "btc_vol_5m", vol_p25),
        ("Vol p33, conf>=0.65", 0.65, "btc_vol_5m", vol_p33),
        ("Vol p33, conf>=0.70", 0.70, "btc_vol_5m", vol_p33),
        ("Vol p33, conf>=0.75", 0.75, "btc_vol_5m", vol_p33),
        ("Vol p33, conf>=0.80", 0.80, "btc_vol_5m", vol_p33),
    ]

    print(f"\n  Position size: $10 per trade, Starting bankroll: $500 (for risk of ruin)")
    print(f"\n  {'Config':<28s} {'Trades':>7s} {'Tr/Day':>7s} {'WR':>6s} "
          f"{'PnL':>9s} {'Fees':>7s} {'$/Trade':>8s} {'MaxDD':>8s} {'Sharpe':>7s} {'RoR':>6s}")
    print("  " + "-" * 105)

    for name, conf, vol_col, vol_thr in configs:
        r = simulate_trading(test_df, y_pred, y_test, conf, vol_col, vol_thr)
        if r is None:
            print(f"  {name:<28s}  insufficient trades")
            continue
        print(f"  {name:<28s} {r['n_trades']:>7d} {r['trades_per_day']:>6.1f}  {r['win_rate']:>5.1%} "
              f"${r['total_pnl']:>+7.1f} ${r['total_fees']:>5.1f} ${r['avg_pnl_per_trade']:>+6.3f} "
              f"${r['max_drawdown']:>+7.1f} {r['sharpe']:>6.2f}  {r['risk_of_ruin']:>5.1%}")

    # Detailed comparison: best no-filter vs best vol-filter
    print(f"\n{'='*75}")
    print("DETAILED COMPARISON: conf>=0.70")
    print(f"{'='*75}")

    for name, conf, vol_col, vol_thr in [
        ("No vol filter", 0.70, None, None),
        ("Vol p25 filter", 0.70, "btc_vol_5m", vol_p25),
    ]:
        r = simulate_trading(test_df, y_pred, y_test, conf, vol_col, vol_thr)
        if r is None:
            continue
        print(f"\n  {name}:")
        print(f"    Trades: {r['n_trades']}, over {r['n_days']} days ({r['trades_per_day']:.1f}/day)")
        print(f"    Win rate: {r['win_rate']:.1%}")
        print(f"    Total PnL: ${r['total_pnl']:+.2f} (fees: ${r['total_fees']:.2f})")
        print(f"    Avg PnL/trade: ${r['avg_pnl_per_trade']:+.4f}")
        print(f"    Max drawdown: ${r['max_drawdown']:+.2f}")
        print(f"    Sharpe (annualized): {r['sharpe']:.2f}")
        print(f"    Risk of ruin ($500 start): {r['risk_of_ruin']:.1%}")

        # Daily stats
        daily = r["daily_pnl"]
        print(f"    Daily PnL: mean=${daily.mean():+.2f}, std=${daily.std():.2f}, "
              f"min=${daily.min():+.2f}, max=${daily.max():+.2f}")
        print(f"    Winning days: {(daily > 0).sum()}/{len(daily)} ({(daily > 0).mean():.0%})")


if __name__ == "__main__":
    main()
