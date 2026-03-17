#!/usr/bin/env python3
"""
Critical analysis of xgboost_flexible.py results.
Checks: Sharpe ratio, risk of ruin, lift over base rate, fee impact,
spread impact, effective sample size, and train/live data gap risks.
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

# Re-use the pipeline
from xgboost_flexible import (
    OBSERVATION_WINDOWS,
    build_dataset,
    get_feature_columns,
)


def main():
    print("=" * 70)
    print("CRITICAL ANALYSIS — XGBoost Flexible v2")
    print("=" * 70)

    # --- Build dataset (same as main run) ---
    print("\nBuilding dataset...")
    dataset = build_dataset(
        data_dir=Path("data/telonex_book_snapshots"),
        resolution_cache_path=Path("data/resolution_cache.json"),
        observe_s=60.0,
        multi_window=True,
        btc_path=Path("data/btc_quotes/btcusdt_quotes.parquet"),
    )

    feature_cols = get_feature_columns(dataset)

    # Time-based split (same as training)
    unique_slugs = dataset["slug"].unique()
    n_markets = len(unique_slugs)
    slug_order = dataset.groupby("slug")["open_ts"].first().sort_values()
    train_end = int(n_markets * 0.70)
    val_end = int(n_markets * 0.85)

    train_slugs = set(slug_order.index[:train_end])
    val_slugs = set(slug_order.index[train_end:val_end])
    test_slugs = set(slug_order.index[val_end:])

    train_df = dataset[dataset["slug"].isin(train_slugs)]
    val_df = dataset[dataset["slug"].isin(val_slugs)]
    test_df = dataset[dataset["slug"].isin(test_slugs)]

    # Train model
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
        "max_depth": 6,
        "learning_rate": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.6,
        "min_child_weight": 15,
        "lambda": 2.0,
        "alpha": 0.5,
        "seed": 42,
        "verbosity": 0,
    }

    print("Training model...")
    model = xgb.train(
        params, dtrain,
        num_boost_round=2000,
        evals=[(dval, "val")],
        early_stopping_rounds=50,
        verbose_eval=0,
    )
    print(f"  Best iteration: {model.best_iteration}")

    y_pred_proba = model.predict(dtest)

    # =====================================================================
    # 1. EFFECTIVE SAMPLE SIZE
    # =====================================================================
    print(f"\n{'=' * 70}")
    print("1. EFFECTIVE SAMPLE SIZE")
    print(f"{'=' * 70}")
    n_test_samples = len(y_test)
    n_test_markets = len(test_slugs)
    n_windows = len(OBSERVATION_WINDOWS)
    print(f"  Test samples: {n_test_samples} ({n_windows} windows × ~{n_test_markets} markets)")
    print(f"  Test markets (independent outcomes): {n_test_markets}")
    print(f"  WARNING: 7 samples per market share the SAME label.")
    print(f"  True degrees of freedom = {n_test_markets}, not {n_test_samples}")
    print(f"  All statistical estimates below use per-MARKET aggregation.")

    # =====================================================================
    # 2. PER-MARKET ANALYSIS (proper independence)
    # =====================================================================
    print(f"\n{'=' * 70}")
    print("2. PER-MARKET PREDICTIONS (best observation window)")
    print(f"{'=' * 70}")

    test_df_copy = test_df.copy()
    test_df_copy["pred_proba"] = y_pred_proba

    # For each market, take the prediction at the BEST observation window
    # (highest confidence). This simulates "query at all times, trade at best"
    market_preds_best = test_df_copy.groupby("slug").agg(
        best_proba=("pred_proba", "max"),
        label=("label", "first"),
        open_ts=("open_ts", "first"),
    )

    # Also compute: for each market, what's the prediction at each window?
    # And what's the prediction at the LATEST window (120s)?
    market_preds_120 = test_df_copy[test_df_copy["observe_time_s"] == 120].set_index("slug")[["pred_proba", "label"]]

    for name, mp in [("Best window (max confidence)", market_preds_best),
                     ("120s window only", market_preds_120)]:
        print(f"\n  --- {name} ---")
        proba_col = "best_proba" if "best_proba" in mp.columns else "pred_proba"
        for threshold in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
            mask = mp[proba_col] >= threshold
            n = mask.sum()
            if n < 5:
                continue
            wr = mp.loc[mask, "label"].mean()
            print(f"    Threshold {threshold:.2f}: {n:>4d} markets, WR={wr:.1%}, base={mp['label'].mean():.1%}, lift={wr - mp['label'].mean():+.1%}")

    # =====================================================================
    # 3. SHARPE RATIO
    # =====================================================================
    print(f"\n{'=' * 70}")
    print("3. SHARPE RATIO (per-market, assuming fixed $1 bet)")
    print(f"{'=' * 70}")

    # Simulate PnL per market at various thresholds
    # Assume: buy leader token at mid price, pay spread + fees
    # Payout: $1 if win, $0 if lose. Cost: mid_price + half_spread + fee
    # Fee: C * p * 0.25 * (p*(1-p))^2 where C=100 (basis points? no, the fee formula)
    # Actually fee = c_f * p * (1-p) where c_f ≈ 0.02 (2%) max effective

    for threshold in [0.60, 0.65, 0.70, 0.75, 0.80]:
        # Get markets where best prediction exceeds threshold
        mask = market_preds_best["best_proba"] >= threshold
        n = mask.sum()
        if n < 10:
            continue

        filtered = market_preds_best[mask]

        # Get the actual mid price at observation time for these markets
        # (proxy: use the test_df mid values)
        filtered_slugs = filtered.index
        mid_prices = []
        for slug in filtered_slugs:
            slug_data = test_df_copy[test_df_copy["slug"] == slug]
            # Use the window that gave the best prediction
            best_idx = slug_data["pred_proba"].idxmax()
            best_row = slug_data.loc[best_idx]
            # Find the latest window mid column
            mid_cols = [c for c in slug_data.columns if c.endswith("_mid") and c.startswith("w")]
            if mid_cols:
                last_mid = sorted(mid_cols)[-1]
                mid_prices.append(best_row[last_mid])
            else:
                mid_prices.append(0.60)

        mid_prices = np.array(mid_prices)
        labels = filtered["label"].values

        # Costs
        half_spread = 0.015  # typical 3-cent spread / 2
        # Polymarket fee: C * p * 0.25 * (p*(1-p))^2, max ~1.56% at p=0.50
        # At typical entry (p=0.55-0.70), fee is lower
        fees = 100 * mid_prices * 0.25 * (mid_prices * (1 - mid_prices)) ** 2 / 10000
        cost = mid_prices + half_spread + fees

        # PnL per trade: win=$1-cost, lose=-cost
        pnl = np.where(labels == 1, 1.0 - cost, -cost)

        mean_pnl = pnl.mean()
        std_pnl = pnl.std()
        sharpe_per_trade = mean_pnl / std_pnl if std_pnl > 0 else 0

        # Annualize: ~288 5-min windows per day, trade `coverage` of them
        coverage = n / n_test_markets
        trades_per_day = 288 * coverage
        daily_pnl = mean_pnl * trades_per_day
        daily_std = std_pnl * np.sqrt(trades_per_day)
        daily_sharpe = daily_pnl / daily_std if daily_std > 0 else 0
        annual_sharpe = daily_sharpe * np.sqrt(252)

        print(f"\n  Threshold {threshold:.2f} ({n} markets, {coverage:.0%} coverage):")
        print(f"    Mean PnL/trade:    ${mean_pnl:+.4f}")
        print(f"    Std PnL/trade:     ${std_pnl:.4f}")
        print(f"    Sharpe (per trade): {sharpe_per_trade:.3f}")
        print(f"    Trades/day (est):  {trades_per_day:.0f}")
        print(f"    Daily PnL (est):   ${daily_pnl:+.2f}")
        print(f"    Annual Sharpe:     {annual_sharpe:.2f}")
        print(f"    Win rate:          {labels.mean():.1%}")
        print(f"    Avg entry (mid):   {mid_prices.mean():.3f}")
        print(f"    Avg fee:           {fees.mean():.4f}")

    # =====================================================================
    # 4. RISK OF RUIN
    # =====================================================================
    print(f"\n{'=' * 70}")
    print("4. RISK OF RUIN (Kelly-based)")
    print(f"{'=' * 70}")

    for threshold in [0.65, 0.70, 0.75, 0.80]:
        mask = market_preds_best["best_proba"] >= threshold
        n = mask.sum()
        if n < 10:
            continue

        filtered = market_preds_best[mask]
        wr = filtered["label"].mean()
        # Average odds: buy at ~mid, win $1
        avg_mid = 0.65  # approximate
        b = (1.0 - avg_mid) / avg_mid  # odds ratio (win/risk)
        p = wr
        q = 1 - p

        # Kelly fraction
        kelly = p - q / b if b > 0 else 0
        half_kelly = kelly / 2

        # Risk of ruin with fixed fraction f of bankroll
        # For fixed fraction betting: P(ruin) = (q/p)^(bankroll/unit) for even money
        # More general: if edge exists, risk of ruin with Kelly sizing → 0 as N→∞
        # With half Kelly, risk of ruin is extremely low
        # But with FIXED position size (not fractional), risk of ruin matters

        # Monte Carlo risk of ruin (10000 sims, 500 trades, $100 bankroll, $5 per trade)
        bankroll = 100
        bet_size = 5
        n_trades_sim = 500
        n_sims = 10000
        ruins = 0
        max_drawdowns = []
        final_values = []

        for _ in range(n_sims):
            balance = bankroll
            peak = bankroll
            max_dd = 0
            for _ in range(n_trades_sim):
                if balance < bet_size:
                    ruins += 1
                    break
                if np.random.random() < wr:
                    profit = bet_size * (1.0 - avg_mid) / avg_mid - bet_size * 0.02  # win minus fees
                    balance += profit
                else:
                    balance -= bet_size
                peak = max(peak, balance)
                dd = (peak - balance) / peak
                max_dd = max(max_dd, dd)
            max_drawdowns.append(max_dd)
            final_values.append(balance)

        ruin_pct = ruins / n_sims * 100
        avg_dd = np.mean(max_drawdowns) * 100
        median_final = np.median(final_values)
        p5_final = np.percentile(final_values, 5)

        print(f"\n  Threshold {threshold:.2f} (WR={wr:.1%}, n={n}, b={b:.2f}):")
        print(f"    Kelly fraction:     {kelly:.3f} (half Kelly: {half_kelly:.3f})")
        print(f"    Monte Carlo (500 trades, $100 bankroll, $5/trade):")
        print(f"      Risk of ruin:     {ruin_pct:.1f}%")
        print(f"      Avg max drawdown: {avg_dd:.1f}%")
        print(f"      Median final:     ${median_final:.0f}")
        print(f"      5th pctile final: ${p5_final:.0f}")

    # =====================================================================
    # 5. LIFT OVER BASE RATE (the real question)
    # =====================================================================
    print(f"\n{'=' * 70}")
    print("5. LIFT OVER BASE RATE BY OBSERVATION TIME")
    print(f"{'=' * 70}")
    print(f"  {'Time':>5s}  {'Base':>6s}  {'WR@0.70':>8s}  {'Lift':>7s}  {'N@0.70':>7s}  {'WR@0.80':>8s}  {'Lift80':>7s}  {'N@0.80':>7s}")
    print("-" * 75)

    for t in sorted(test_df_copy["observe_time_s"].unique()):
        t_data = test_df_copy[test_df_copy["observe_time_s"] == t]
        base = t_data["label"].mean()

        m70 = t_data["pred_proba"] >= 0.70
        wr70 = t_data.loc[m70, "label"].mean() if m70.sum() > 5 else float("nan")
        n70 = m70.sum()

        m80 = t_data["pred_proba"] >= 0.80
        wr80 = t_data.loc[m80, "label"].mean() if m80.sum() > 5 else float("nan")
        n80 = m80.sum()

        lift70 = wr70 - base if not np.isnan(wr70) else float("nan")
        lift80 = wr80 - base if not np.isnan(wr80) else float("nan")

        print(f"  {t:>4.0f}s  {base:>5.1%}  {wr70:>7.1%}  {lift70:>+6.1%}  {n70:>7d}  {wr80:>7.1%}  {lift80:>+6.1%}  {n80:>7d}")

    # =====================================================================
    # 6. DATA GAP ANALYSIS: Training vs Live
    # =====================================================================
    print(f"\n{'=' * 70}")
    print("6. TRAINING vs LIVE DATA COMPARISON")
    print(f"{'=' * 70}")

    # Check orderbook data density
    sample_slugs = list(test_slugs)[:20]
    ticks_per_market = []
    for slug in sample_slugs:
        pf = Path(f"data/telonex_book_snapshots/{slug}.parquet")
        if pf.exists():
            df = pd.read_parquet(pf)
            try:
                open_ts = int(slug.split("-")[-1])
                open_ms = open_ts * 1000
                close_ms = open_ms + 300_000
                market_df = df[(df["exchange_timestamp"] >= open_ms) & (df["exchange_timestamp"] < close_ms)]
                # Per token
                for label in ["Up", "Down"]:
                    token_df = market_df[market_df["token_label"] == label]
                    if not token_df.empty:
                        duration_s = (token_df["exchange_timestamp"].max() - token_df["exchange_timestamp"].min()) / 1000
                        ticks_per_market.append({
                            "slug": slug,
                            "token": label,
                            "n_ticks": len(token_df),
                            "duration_s": duration_s,
                            "freq_hz": len(token_df) / duration_s if duration_s > 0 else 0,
                        })
            except ValueError:
                pass

    if ticks_per_market:
        tick_df = pd.DataFrame(ticks_per_market)
        print(f"\n  Telonex orderbook snapshot frequency (test set sample):")
        print(f"    Median ticks/market/token: {tick_df['n_ticks'].median():.0f}")
        print(f"    Median frequency:          {tick_df['freq_hz'].median():.1f} Hz")
        print(f"    Min/Max ticks:             {tick_df['n_ticks'].min()}-{tick_df['n_ticks'].max()}")
        print(f"    Min/Max frequency:          {tick_df['freq_hz'].min():.1f}-{tick_df['freq_hz'].max():.1f} Hz")

    print(f"""
  TRAINING DATA (Telonex):
    - Orderbook: periodic snapshots (see frequency above)
    - Each snapshot: full L2 book (10 levels bid/ask, mid, spread, imbalance)
    - BTC: 1-second bars from btc_pipeline.py (mid, spread, log_return, rvol)
    - Coverage: Feb 6-24, 2026 (~3,728 markets with BTC overlap)

  LIVE DATA (execution component):
    - Orderbook: WebSocket real-time ticks (every price_change event)
    - Much HIGHER frequency than Telonex snapshots
    - BTC: tick-level from Binance WebSocket (higher resolution than 1s bars)
    - Key difference: live orderbook is NOISIER (every microstructure event)
      vs Telonex which is periodic snapshots (smoothed/sampled)

  POTENTIAL GAPS:
    1. Feature extraction uses 'last tick before window' — with denser live data,
       this snapshot will be less 'smooth' than Telonex. Spread/imbalance will
       be noisier due to transient order states.
    2. BTC features use 1s bars in training. Live BTC ticks are sub-100ms.
       Need to resample live BTC to 1s bars to match training distribution.
    3. 'update_freq' feature will be MUCH higher live (possibly 10-100x).
       This feature is in top 5 importance — could break predictions.
    4. Telonex data has token_label='Up'/'Down', live has 'Yes'/'No'.
       Must map correctly or leader detection fails.
    5. No trade/fill data in features — this is good, avoids a live data gap.
""")

    # =====================================================================
    # 7. FEATURE SENSITIVITY — which features change most live?
    # =====================================================================
    print(f"{'=' * 70}")
    print("7. FEATURES MOST AT RISK OF DISTRIBUTION SHIFT")
    print(f"{'=' * 70}")

    importance = model.get_score(importance_type="gain")
    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)

    risky_features = {
        "update_freq": "MUCH higher live (dense ticks vs periodic snapshots)",
        "n_ticks": "MUCH higher live (same reason as update_freq)",
        "btc_mid": "Absolute BTC price — drifts over time, model may overfit to range",
        "btc_rvol_30s": "Different granularity live (tick vs 1s bar)",
        "btc_rvol_60s": "Different granularity live (tick vs 1s bar)",
        "btc_rvol_300s": "Different granularity live (tick vs 1s bar)",
    }

    print(f"\n  {'Feature':<25s}  {'Importance':>10s}  {'Risk':>5s}  Description")
    print("-" * 90)
    for feat, gain in sorted_imp[:30]:
        risk = "HIGH" if feat in risky_features else "low"
        desc = risky_features.get(feat, "")
        marker = " <<<" if risk == "HIGH" else ""
        print(f"  {feat:<25s}  {gain:>10.1f}  {risk:>5s}  {desc}{marker}")

    # =====================================================================
    # 8. CONSECUTIVE LOSS STREAKS (for psychology / position sizing)
    # =====================================================================
    print(f"\n{'=' * 70}")
    print("8. WORST CASE LOSS STREAKS (per-market, threshold=0.70)")
    print(f"{'=' * 70}")

    mask = market_preds_best["best_proba"] >= 0.70
    if mask.sum() > 20:
        filtered = market_preds_best[mask].sort_values("open_ts")
        outcomes = filtered["label"].values

        # Find longest loss streak
        max_streak = 0
        current_streak = 0
        streaks = []
        for o in outcomes:
            if o == 0:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                if current_streak > 0:
                    streaks.append(current_streak)
                current_streak = 0
        if current_streak > 0:
            streaks.append(current_streak)

        print(f"  Total trades: {len(outcomes)}")
        print(f"  Wins: {outcomes.sum()}, Losses: {len(outcomes) - outcomes.sum()}")
        print(f"  Longest loss streak: {max_streak}")
        if streaks:
            print(f"  Loss streak distribution: mean={np.mean(streaks):.1f}, median={np.median(streaks):.0f}, max={max(streaks)}")
            print(f"  Streaks >= 3: {sum(1 for s in streaks if s >= 3)}")
            print(f"  Streaks >= 5: {sum(1 for s in streaks if s >= 5)}")

        # Rolling win rate (50-trade windows)
        if len(outcomes) >= 50:
            rolling_wr = pd.Series(outcomes).rolling(50).mean().dropna()
            print(f"\n  Rolling 50-trade WR:")
            print(f"    Min:    {rolling_wr.min():.1%}")
            print(f"    Max:    {rolling_wr.max():.1%}")
            print(f"    Mean:   {rolling_wr.mean():.1%}")
            print(f"    Worst drawdown window: {rolling_wr.min():.1%}")

    print(f"\n{'=' * 70}")
    print("DONE — Review above before going live.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
