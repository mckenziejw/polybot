#!/usr/bin/env python3
"""
Walk-forward backtest of v3 XGBoost + vol filter across ALL historic data.

Uses expanding window: train on markets [0..T], predict on [T..T+step].
Slide forward, retrain, repeat. This gives out-of-sample predictions for
every market (except the initial training window).
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

warnings.filterwarnings("ignore")

FEATURE_CACHE = Path("data/xgb_features_v3.parquet")


def polymarket_fee(price: float) -> float:
    return 100 * price * 0.25 * (price * (1 - price)) ** 2 / 10000


def main():
    print("=" * 80)
    print("WALK-FORWARD BACKTEST: v3 XGBoost + Vol Filter (full history)")
    print("=" * 80)

    df = pd.read_parquet(FEATURE_CACHE)
    feature_cols = [c for c in df.columns if c not in ("label", "slug", "open_ts") and not c.startswith("_")]

    # Sort markets chronologically
    slug_order = df.groupby("slug")["open_ts"].first().sort_values()
    ordered_slugs = slug_order.index.tolist()
    n_markets = len(ordered_slugs)

    print(f"Total: {len(df)} samples, {n_markets} markets")
    print(f"Features: {len(feature_cols)}")

    from datetime import datetime, timezone
    start_dt = datetime.fromtimestamp(slug_order.iloc[0], tz=timezone.utc)
    end_dt = datetime.fromtimestamp(slug_order.iloc[-1], tz=timezone.utc)
    total_days = (end_dt - start_dt).days + 1
    print(f"Period: {start_dt.date()} to {end_dt.date()} ({total_days} days)")

    # Walk-forward params
    min_train = 1000       # minimum markets to train on
    step = 200             # predict next 200 markets, then retrain
    val_frac = 0.15        # last 15% of training window for early stopping

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

    # Collect all out-of-sample predictions
    all_preds = []  # (slug, open_ts, observe_time_s, pred, label, btc_vol_5m, entry_price)

    fold = 0
    train_end = min_train

    while train_end < n_markets:
        test_end = min(train_end + step, n_markets)
        train_slugs = set(ordered_slugs[:train_end])
        test_slugs_fold = set(ordered_slugs[train_end:test_end])

        # Split training into train/val for early stopping
        val_start = int(train_end * (1 - val_frac))
        actual_train = set(ordered_slugs[:val_start])
        actual_val = set(ordered_slugs[val_start:train_end])

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
                "slug": row["slug"],
                "open_ts": row["open_ts"],
                "observe_time_s": row["observe_time_s"],
                "pred": preds[i],
                "label": row["label"],
                "btc_vol_5m": row.get("btc_vol_5m", np.nan),
                "entry_price": entry_price,
            })

        fold += 1
        if fold % 5 == 0:
            test_dt = datetime.fromtimestamp(slug_order.iloc[train_end], tz=timezone.utc)
            print(f"  Fold {fold}: trained on {train_end} markets, "
                  f"testing {test_end - train_end}, date ~{test_dt.date()}, "
                  f"total OOS preds: {len(all_preds)}")

        train_end = test_end

    pdf = pd.DataFrame(all_preds)
    print(f"\nTotal OOS predictions: {len(pdf)}, covering {pdf['slug'].nunique()} markets")

    # Compute vol threshold from first training window
    first_train = df[df["slug"].isin(set(ordered_slugs[:min_train]))]
    train_vol = first_train["btc_vol_5m"].dropna()
    train_vol = train_vol[train_vol > 0]
    vol_p25 = float(train_vol.quantile(0.25))
    vol_p33 = float(train_vol.quantile(0.33))
    print(f"Vol thresholds (from first {min_train} markets): p25={vol_p25:.6f}, p33={vol_p33:.6f}")

    # Filter to 120s window for trading simulation
    w120 = pdf[pdf["observe_time_s"] == 120].copy().sort_values("open_ts").reset_index(drop=True)
    print(f"120s window predictions: {len(w120)} markets")

    # Simulate trading
    def simulate(mask_name, mask):
        trades = w120[mask].copy()
        if len(trades) < 5:
            return None

        fees = trades["entry_price"].apply(lambda p: polymarket_fee(p) * 10.0)
        pnl = np.where(
            trades["label"] == 1,
            (1.0 - trades["entry_price"]) * 10.0 - fees,
            -trades["entry_price"] * 10.0 - fees,
        )
        trades["pnl"] = pnl
        trades["cum_pnl"] = pnl.cumsum()
        trades["peak"] = trades["cum_pnl"].cummax()
        trades["drawdown"] = trades["cum_pnl"] - trades["peak"]

        # Daily aggregation
        trades["day"] = (trades["open_ts"] - trades["open_ts"].iloc[0]) // 86400
        daily = trades.groupby("day")["pnl"].sum()

        n_days = trades["day"].nunique()
        sharpe = daily.mean() / daily.std() * np.sqrt(365) if daily.std() > 0 else 0.0

        # Risk of ruin Monte Carlo ($500 start)
        ruin = 0
        tpnl = trades["pnl"].values
        for _ in range(10000):
            sim = np.cumsum(np.random.choice(tpnl, size=len(tpnl), replace=True)) + 500.0
            if sim.min() <= 0:
                ruin += 1

        return {
            "name": mask_name,
            "n_trades": len(trades),
            "n_days": n_days,
            "trades_per_day": len(trades) / max(n_days, 1),
            "wr": trades["label"].mean(),
            "total_pnl": trades["pnl"].sum(),
            "total_fees": fees.sum(),
            "avg_pnl": trades["pnl"].mean(),
            "max_dd": trades["drawdown"].min(),
            "sharpe": sharpe,
            "ror": ruin / 10000,
            "winning_days": (daily > 0).sum(),
            "total_days_traded": len(daily),
            "daily_mean": daily.mean(),
            "daily_std": daily.std(),
            "daily_min": daily.min(),
            "daily_max": daily.max(),
        }

    # Build masks
    configs = []
    for conf in [0.65, 0.70, 0.75, 0.80]:
        conf_mask = w120["pred"] >= conf
        configs.append((f"No filter, >={conf:.2f}", conf_mask))

        vol_mask = w120["btc_vol_5m"] > vol_p25
        configs.append((f"Vol p25, >={conf:.2f}", conf_mask & vol_mask))

    print(f"\n  $10/trade, $500 starting bankroll")
    print(f"\n  {'Config':<26s} {'Trades':>7s} {'Tr/Day':>7s} {'WR':>6s} "
          f"{'PnL':>9s} {'$/Trade':>8s} {'MaxDD':>8s} {'Sharpe':>7s} {'RoR':>6s} {'WinDays':>8s}")
    print("  " + "-" * 100)

    results = []
    for name, mask in configs:
        r = simulate(name, mask)
        if r is None:
            print(f"  {name:<26s}  insufficient trades")
            continue
        results.append(r)
        wd = f"{r['winning_days']}/{r['total_days_traded']}"
        print(f"  {name:<26s} {r['n_trades']:>7d} {r['trades_per_day']:>6.1f}  {r['wr']:>5.1%} "
              f"${r['total_pnl']:>+7.0f} ${r['avg_pnl']:>+6.3f} "
              f"${r['max_dd']:>+7.0f} {r['sharpe']:>6.2f}  {r['ror']:>5.1%} {wd:>8s}")

    # Detailed comparison
    print(f"\n{'='*80}")
    print("DETAILED: conf>=0.70")
    print(f"{'='*80}")

    for name, mask in [
        ("No filter", w120["pred"] >= 0.70),
        ("Vol p25", (w120["pred"] >= 0.70) & (w120["btc_vol_5m"] > vol_p25)),
    ]:
        r = simulate(name, mask)
        if r is None:
            continue
        print(f"\n  {r['name']}:")
        print(f"    Trades: {r['n_trades']}, over {r['n_days']} days ({r['trades_per_day']:.1f}/day)")
        print(f"    Win rate: {r['wr']:.1%}")
        print(f"    Total PnL: ${r['total_pnl']:+.2f} (fees: ${r['total_fees']:.2f})")
        print(f"    Avg PnL/trade: ${r['avg_pnl']:+.4f}")
        print(f"    Max drawdown: ${r['max_dd']:+.2f}")
        print(f"    Sharpe (annualized): {r['sharpe']:.2f}")
        print(f"    Risk of ruin ($500 start): {r['ror']:.1%}")
        print(f"    Daily PnL: mean=${r['daily_mean']:+.2f}, std=${r['daily_std']:.2f}, "
              f"min=${r['daily_min']:+.2f}, max=${r['daily_max']:+.2f}")
        print(f"    Winning days: {r['winning_days']}/{r['total_days_traded']} "
              f"({r['winning_days']/max(r['total_days_traded'],1):.0%})")

    # Time-series breakdown: show WR by day
    print(f"\n{'='*80}")
    print("DAILY BREAKDOWN (conf>=0.70, vol p25)")
    print(f"{'='*80}")

    best_mask = (w120["pred"] >= 0.70) & (w120["btc_vol_5m"] > vol_p25)
    trades = w120[best_mask].copy().sort_values("open_ts")
    if len(trades) > 0:
        from datetime import datetime, timezone
        trades["date"] = trades["open_ts"].apply(
            lambda ts: datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d"))
        fees = trades["entry_price"].apply(lambda p: polymarket_fee(p) * 10.0)
        trades["pnl"] = np.where(
            trades["label"] == 1,
            (1.0 - trades["entry_price"]) * 10.0 - fees,
            -trades["entry_price"] * 10.0 - fees,
        )

        daily = trades.groupby("date").agg(
            n_trades=("pnl", "count"),
            wins=("label", "sum"),
            wr=("label", "mean"),
            pnl=("pnl", "sum"),
            avg_entry=("entry_price", "mean"),
        )
        daily["cum_pnl"] = daily["pnl"].cumsum()

        print(f"  {'Date':<12s} {'Trades':>7s} {'Wins':>5s} {'WR':>6s} "
              f"{'PnL':>8s} {'CumPnL':>9s} {'AvgEntry':>9s}")
        print("  " + "-" * 62)
        for date, row in daily.iterrows():
            print(f"  {date:<12s} {row['n_trades']:>7.0f} {row['wins']:>5.0f} {row['wr']:>5.1%} "
                  f"${row['pnl']:>+6.1f} ${row['cum_pnl']:>+7.1f} {row['avg_entry']:>8.3f}")


if __name__ == "__main__":
    main()
