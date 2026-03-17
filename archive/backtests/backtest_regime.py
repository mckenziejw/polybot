#!/usr/bin/env python3
"""
Backtest regime-switching strategy: MM in low-confidence markets,
directional in high-confidence markets.

Uses the trained XGBoost model to classify market regimes at each
observation window, then simulates:
  - Market making (post bids/asks near mid) in uncertain markets
  - Directional (buy leader) in confident markets

Computes PnL, Sharpe, drawdown for each mode and combined.
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

warnings.filterwarnings("ignore")

from xgboost_flexible import (
    MARKET_DURATION_MS,
    OBSERVATION_WINDOWS,
    build_dataset,
    get_feature_columns,
    resample_to_1s,
)

# --- Strategy parameters ---
MM_CONFIDENCE_LO = 0.0    # model confidence below this → MM mode
MM_CONFIDENCE_HI = 0.57   # model confidence above this → exit MM / go directional
DIRECTIONAL_THRESHOLD = 0.75  # go directional above this
MM_SPREAD_HALF = 0.02     # post bid at mid-0.02, ask at mid+0.02
MM_QUOTE_SIZE = 10.0      # tokens per side
DIRECTIONAL_SIZE = 10.0   # tokens for directional bet
TAKER_FEE_BPS = 0         # Polymarket maker fee is 0, taker varies
# Polymarket fee: C * p * 0.25 * (p*(1-p))^2 where C=100
# At p=0.50: ~1.56%. At p=0.60: ~0.86%. At p=0.70: ~0.44%

def polymarket_fee(price: float) -> float:
    """Compute Polymarket fee as fraction of notional."""
    return 100 * price * 0.25 * (price * (1 - price)) ** 2 / 10000


def simulate_market_making(
    leader_mids: np.ndarray,
    leader_timestamps: np.ndarray,
    other_mids: np.ndarray,
    other_timestamps: np.ndarray,
    open_ms: int,
    entry_time_s: float,
    exit_time_s: float,
    label: int,  # 1 = leader wins
) -> dict:
    """
    Simulate market making on BOTH tokens during [entry_time_s, exit_time_s].

    We post bids and asks on both the leader and other token.
    Simplified model: we assume our quotes get filled proportional to
    how much time the mid spends near our quote levels.

    More realistic model: track when mid crosses our quote levels.
    """
    close_ms = open_ms + MARKET_DURATION_MS

    # Filter to our active window
    leader_mask = (
        (leader_timestamps >= open_ms + entry_time_s * 1000) &
        (leader_timestamps < open_ms + exit_time_s * 1000)
    )
    other_mask = (
        (other_timestamps >= open_ms + entry_time_s * 1000) &
        (other_timestamps < open_ms + exit_time_s * 1000)
    )

    l_mids = leader_mids[leader_mask]
    o_mids = other_mids[other_mask]

    if len(l_mids) < 5 or len(o_mids) < 5:
        return {"pnl": 0, "n_fills": 0, "mode": "mm", "skipped": True}

    # Our quote levels on leader token
    l_mid_avg = l_mids.mean()
    l_bid = l_mid_avg - MM_SPREAD_HALF
    l_ask = l_mid_avg + MM_SPREAD_HALF

    # Count how many times mid crosses our levels (proxy for fills)
    # Bid fill: mid drops below our bid (someone sells to us)
    # Ask fill: mid rises above our ask (someone buys from us)
    l_bid_fills = np.sum(l_mids[1:] <= l_bid)
    l_ask_fills = np.sum(l_mids[1:] >= l_ask)
    l_round_trips = min(l_bid_fills, l_ask_fills)

    # Same for other token
    o_mid_avg = o_mids.mean()
    o_bid = o_mid_avg - MM_SPREAD_HALF
    o_ask = o_mid_avg + MM_SPREAD_HALF
    o_bid_fills = np.sum(o_mids[1:] <= o_bid)
    o_ask_fills = np.sum(o_mids[1:] >= o_ask)
    o_round_trips = min(o_bid_fills, o_ask_fills)

    # PnL from completed round trips (spread captured)
    spread_pnl = (l_round_trips + o_round_trips) * MM_SPREAD_HALF * 2 * MM_QUOTE_SIZE

    # Risk: inventory at market close
    # Net inventory = bid_fills - ask_fills on each side
    l_net = l_bid_fills - l_ask_fills  # positive = long leader
    o_net = o_bid_fills - o_ask_fills  # positive = long other

    # Cap inventory at reasonable levels
    l_net = np.clip(l_net, -5, 5)
    o_net = np.clip(o_net, -5, 5)

    # Inventory PnL at market close
    # Leader token: worth $1 if label=1, $0 if label=0
    # Other token: worth $1 if label=0, $0 if label=1
    l_inventory_value = l_net * MM_QUOTE_SIZE * (1.0 if label == 1 else 0.0)
    l_inventory_cost = l_net * MM_QUOTE_SIZE * l_mid_avg
    o_inventory_value = o_net * MM_QUOTE_SIZE * (1.0 if label == 0 else 0.0)
    o_inventory_cost = o_net * MM_QUOTE_SIZE * o_mid_avg

    inventory_pnl = (l_inventory_value - l_inventory_cost) + (o_inventory_value - o_inventory_cost)

    # Fees on fills
    total_fills = l_bid_fills + l_ask_fills + o_bid_fills + o_ask_fills
    avg_price = (l_mid_avg + o_mid_avg) / 2
    fee_per_fill = polymarket_fee(avg_price) * MM_QUOTE_SIZE
    total_fees = total_fills * fee_per_fill

    total_pnl = spread_pnl + inventory_pnl - total_fees

    return {
        "pnl": total_pnl,
        "spread_pnl": spread_pnl,
        "inventory_pnl": inventory_pnl,
        "fees": total_fees,
        "n_round_trips": l_round_trips + o_round_trips,
        "n_fills": total_fills,
        "l_net_inventory": l_net,
        "o_net_inventory": o_net,
        "mode": "mm",
        "skipped": False,
    }


def simulate_directional(
    entry_mid: float,
    label: int,
    size: float = DIRECTIONAL_SIZE,
) -> dict:
    """
    Simulate directional bet: buy leader token at entry_mid.
    Payout: $1 if win, $0 if lose.
    """
    # Cost includes half-spread (we're taking, not making)
    entry_cost = entry_mid + 0.01  # ~1 cent slippage/half-spread
    fee = polymarket_fee(entry_cost) * size
    cost = entry_cost * size + fee

    if label == 1:
        pnl = 1.0 * size - cost
    else:
        pnl = -cost

    return {
        "pnl": pnl,
        "entry_price": entry_cost,
        "fee": fee,
        "mode": "directional",
        "skipped": False,
    }


def main():
    print("=" * 70)
    print("REGIME-SWITCHING BACKTEST")
    print("=" * 70)
    print(f"  MM zone:          confidence < {MM_CONFIDENCE_HI}")
    print(f"  Directional zone: confidence >= {DIRECTIONAL_THRESHOLD}")
    print(f"  Dead zone:        {MM_CONFIDENCE_HI} - {DIRECTIONAL_THRESHOLD}")
    print(f"  MM spread:        ±{MM_SPREAD_HALF}")
    print(f"  Quote size:       {MM_QUOTE_SIZE} tokens")
    print()

    # Build dataset and train model (same as xgboost_flexible.py)
    print("Building dataset...")
    dataset = build_dataset(
        data_dir=Path("data/telonex_book_snapshots"),
        resolution_cache_path=Path("data/resolution_cache.json"),
        observe_s=60.0,
        multi_window=True,
        btc_path=Path("data/btc_quotes/btcusdt_quotes.parquet"),
    )

    feature_cols = get_feature_columns(dataset)

    # Split
    slug_order = dataset.groupby("slug")["open_ts"].first().sort_values()
    n_markets = len(slug_order)
    train_end = int(n_markets * 0.70)
    val_end = int(n_markets * 0.85)

    train_slugs = set(slug_order.index[:train_end])
    val_slugs = set(slug_order.index[train_end:val_end])
    test_slugs = set(slug_order.index[val_end:])

    train_df = dataset[dataset["slug"].isin(train_slugs)]
    val_df = dataset[dataset["slug"].isin(val_slugs)]
    test_df = dataset[dataset["slug"].isin(test_slugs)]

    # Train
    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df["label"].values
    X_val = val_df[feature_cols].values.astype(np.float32)
    y_val = val_df["label"].values

    for X in [X_train, X_val]:
        X[~np.isfinite(X)] = 0.0

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_cols)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_cols)

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

    # --- Run backtest on test set markets ---
    print(f"\nRunning backtest on {len(test_slugs)} test markets...")

    data_dir = Path("data/telonex_book_snapshots")
    results = []

    for i, slug in enumerate(sorted(test_slugs)):
        pf = data_dir / f"{slug}.parquet"
        if not pf.exists():
            continue

        try:
            open_ts = int(slug.split("-")[-1])
        except ValueError:
            continue

        open_ms = open_ts * 1000

        try:
            df = pd.read_parquet(pf)
        except Exception:
            continue

        if "token_label" not in df.columns:
            continue

        close_ms = open_ms + MARKET_DURATION_MS

        # Resample to 1s bars (matches training data)
        up_raw = df[df["token_label"] == "Up"].sort_values("exchange_timestamp")
        down_raw = df[df["token_label"] == "Down"].sort_values("exchange_timestamp")
        up_df = resample_to_1s(up_raw, open_ms, close_ms)
        down_df = resample_to_1s(down_raw, open_ms, close_ms)

        if len(up_df) < 10 or len(down_df) < 10:
            continue

        # Get model predictions for this market at each window
        slug_data = dataset[(dataset["slug"] == slug) & dataset["slug"].isin(test_slugs)]
        if slug_data.empty:
            continue

        X_slug = slug_data[feature_cols].values.astype(np.float32)
        X_slug[~np.isfinite(X_slug)] = 0.0
        dslug = xgb.DMatrix(X_slug, feature_names=feature_cols)
        predictions = model.predict(dslug)
        label = slug_data["label"].iloc[0]

        # Map predictions to observation times
        window_preds = {}
        for j, row in enumerate(slug_data.itertuples()):
            obs_t = row.observe_time_s
            window_preds[obs_t] = predictions[j]

        # Determine leader/other from the data
        # At 10s, check which token has mid >= 0.50
        up_10 = up_df[(up_df["exchange_timestamp"] - open_ms) / 1000 <= 10]
        down_10 = down_df[(down_df["exchange_timestamp"] - open_ms) / 1000 <= 10]

        if up_10.empty or down_10.empty:
            continue

        up_mid_10 = up_10.iloc[-1]["mid_price"]
        if up_mid_10 >= 0.50:
            leader_mids = up_df["mid_price"].values
            leader_ts = up_df["exchange_timestamp"].values
            other_mids = down_df["mid_price"].values
            other_ts = down_df["exchange_timestamp"].values
        else:
            leader_mids = down_df["mid_price"].values
            leader_ts = down_df["exchange_timestamp"].values
            other_mids = up_df["mid_price"].values
            other_ts = up_df["exchange_timestamp"].values

        # --- Decision logic ---
        # Check confidence at early windows (10s, 20s) for regime classification
        early_conf = max(
            window_preds.get(10, 0.5),
            window_preds.get(20, 0.5),
        )

        # Check confidence at later windows for directional
        late_conf = max(
            window_preds.get(90, 0.5),
            window_preds.get(120, 0.5),
        )

        # Also track the confidence trajectory
        all_confs = [window_preds.get(w, 0.5) for w in OBSERVATION_WINDOWS if w in window_preds]
        max_conf = max(all_confs) if all_confs else 0.5

        market_result = {
            "slug": slug,
            "label": label,
            "early_conf": early_conf,
            "late_conf": late_conf,
            "max_conf": max_conf,
            "mode": "none",
            "pnl": 0.0,
        }

        # MODE 1: Market making in uncertain markets
        if early_conf < MM_CONFIDENCE_HI:
            # MM from 10s to 120s (or until we'd switch modes)
            # Find when/if confidence crosses threshold
            exit_time = 120.0  # default: MM until 120s
            for w in OBSERVATION_WINDOWS:
                if w in window_preds and window_preds[w] >= MM_CONFIDENCE_HI and w > 20:
                    exit_time = float(w)
                    break

            mm_result = simulate_market_making(
                leader_mids, leader_ts,
                other_mids, other_ts,
                open_ms,
                entry_time_s=10.0,
                exit_time_s=exit_time,
                label=label,
            )

            if not mm_result["skipped"]:
                market_result["mode"] = "mm"
                market_result["pnl"] = mm_result["pnl"]
                market_result["spread_pnl"] = mm_result.get("spread_pnl", 0)
                market_result["inventory_pnl"] = mm_result.get("inventory_pnl", 0)
                market_result["fees"] = mm_result.get("fees", 0)
                market_result["n_round_trips"] = mm_result.get("n_round_trips", 0)
                market_result["mm_exit_time"] = exit_time

            # If confidence later rises, also take directional
            if late_conf >= DIRECTIONAL_THRESHOLD:
                # Get mid at the late window
                late_window = 120 if window_preds.get(120, 0) >= DIRECTIONAL_THRESHOLD else 90
                leader_at_late = leader_mids[
                    (leader_ts >= open_ms + late_window * 1000 - 2000) &
                    (leader_ts <= open_ms + late_window * 1000 + 2000)
                ]
                if len(leader_at_late) > 0:
                    dir_result = simulate_directional(leader_at_late[-1], label)
                    market_result["pnl"] += dir_result["pnl"]
                    market_result["mode"] = "mm+dir"
                    market_result["dir_pnl"] = dir_result["pnl"]

        # MODE 2: Pure directional in high-confidence markets
        elif max_conf >= DIRECTIONAL_THRESHOLD:
            # Find earliest window where confidence exceeds threshold
            entry_window = None
            for w in OBSERVATION_WINDOWS:
                if window_preds.get(w, 0) >= DIRECTIONAL_THRESHOLD:
                    entry_window = w
                    break

            if entry_window is not None:
                leader_at_entry = leader_mids[
                    (leader_ts >= open_ms + entry_window * 1000 - 2000) &
                    (leader_ts <= open_ms + entry_window * 1000 + 2000)
                ]
                if len(leader_at_entry) > 0:
                    dir_result = simulate_directional(leader_at_entry[-1], label)
                    market_result["mode"] = "directional"
                    market_result["pnl"] = dir_result["pnl"]
                    market_result["entry_price"] = dir_result["entry_price"]
                    market_result["entry_window"] = entry_window

        results.append(market_result)

        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(test_slugs)}...")

    results_df = pd.DataFrame(results)

    # --- Results ---
    print(f"\n{'=' * 70}")
    print("BACKTEST RESULTS")
    print(f"{'=' * 70}")

    total_markets = len(results_df)
    active_markets = len(results_df[results_df["mode"] != "none"])
    print(f"\n  Total test markets:  {total_markets}")
    print(f"  Active markets:      {active_markets} ({active_markets/total_markets:.0%})")

    for mode in ["mm", "directional", "mm+dir", "none"]:
        mode_df = results_df[results_df["mode"] == mode]
        if mode_df.empty:
            continue
        n = len(mode_df)
        total_pnl = mode_df["pnl"].sum()
        mean_pnl = mode_df["pnl"].mean()
        std_pnl = mode_df["pnl"].std()
        wr = (mode_df["pnl"] > 0).mean()
        sharpe = mean_pnl / std_pnl if std_pnl > 0 else 0

        print(f"\n  --- {mode.upper()} ({n} markets, {n/total_markets:.0%} coverage) ---")
        print(f"    Total PnL:       ${total_pnl:+.2f}")
        print(f"    Mean PnL/market: ${mean_pnl:+.4f}")
        print(f"    Std PnL/market:  ${std_pnl:.4f}")
        print(f"    Win rate:        {wr:.1%}")
        print(f"    Sharpe (trade):  {sharpe:.3f}")

        if mode in ("mm", "mm+dir") and "spread_pnl" in mode_df.columns:
            print(f"    Spread PnL:      ${mode_df['spread_pnl'].sum():+.2f}")
            print(f"    Inventory PnL:   ${mode_df['inventory_pnl'].sum():+.2f}")
            print(f"    Fees:            ${mode_df['fees'].sum():.2f}")
            if "n_round_trips" in mode_df.columns:
                print(f"    Avg round trips: {mode_df['n_round_trips'].mean():.1f}")

    # Combined results
    active = results_df[results_df["mode"] != "none"]
    if not active.empty:
        print(f"\n  --- COMBINED (all active, {len(active)} markets) ---")
        total_pnl = active["pnl"].sum()
        mean_pnl = active["pnl"].mean()
        std_pnl = active["pnl"].std()
        wr = (active["pnl"] > 0).mean()
        sharpe = mean_pnl / std_pnl if std_pnl > 0 else 0

        print(f"    Total PnL:       ${total_pnl:+.2f}")
        print(f"    Mean PnL/market: ${mean_pnl:+.4f}")
        print(f"    Win rate:        {wr:.1%}")
        print(f"    Sharpe (trade):  {sharpe:.3f}")

        # Annualized estimates
        coverage = len(active) / total_markets
        trades_per_day = 288 * coverage  # 288 five-min windows per day
        daily_pnl = mean_pnl * trades_per_day
        daily_std = std_pnl * np.sqrt(trades_per_day)
        annual_sharpe = (daily_pnl / daily_std) * np.sqrt(252) if daily_std > 0 else 0

        print(f"    Coverage:        {coverage:.0%}")
        print(f"    Est trades/day:  {trades_per_day:.0f}")
        print(f"    Est daily PnL:   ${daily_pnl:+.2f}")
        print(f"    Annual Sharpe:   {annual_sharpe:.2f}")

        # Cumulative PnL curve stats
        cum_pnl = active.sort_values("slug")["pnl"].cumsum()
        peak = cum_pnl.expanding().max()
        drawdown = cum_pnl - peak
        max_dd = drawdown.min()

        print(f"    Max drawdown:    ${max_dd:.2f}")
        print(f"    Final PnL:       ${cum_pnl.iloc[-1]:+.2f}")

    # --- Sweep MM threshold ---
    print(f"\n{'=' * 70}")
    print("MM THRESHOLD SWEEP")
    print(f"{'=' * 70}")
    print(f"  {'MM_HI':>8s}  {'DIR_THR':>8s}  {'N_MM':>6s}  {'N_DIR':>6s}  {'PnL':>10s}  {'WR':>6s}  {'Sharpe':>7s}")
    print("-" * 65)

    # Re-simulate with different thresholds
    for mm_hi in [0.52, 0.55, 0.57, 0.60, 0.65]:
        for dir_thr in [0.70, 0.75, 0.80]:
            n_mm = 0
            n_dir = 0
            pnls = []

            for _, row in results_df.iterrows():
                early = row["early_conf"]
                late = row["late_conf"]
                max_c = row["max_conf"]
                lab = row["label"]

                if early < mm_hi:
                    # MM mode — use pre-computed results if available
                    # For sweep, approximate: just use the directional component
                    # since we can't re-simulate MM with different params here
                    if late >= dir_thr:
                        n_dir += 1
                        # Approximate directional PnL
                        entry = 0.70  # rough proxy
                        pnl = (1.0 - entry - 0.01) * DIRECTIONAL_SIZE if lab == 1 else -(entry + 0.01) * DIRECTIONAL_SIZE
                        pnl -= polymarket_fee(entry) * DIRECTIONAL_SIZE
                        pnls.append(pnl)
                    else:
                        n_mm += 1
                        # Use actual MM result if we have it
                        if row["mode"] in ("mm", "mm+dir"):
                            pnls.append(row.get("spread_pnl", 0) + row.get("inventory_pnl", 0) - row.get("fees", 0))
                        else:
                            pnls.append(0)
                elif max_c >= dir_thr:
                    n_dir += 1
                    entry = 0.70
                    pnl = (1.0 - entry - 0.01) * DIRECTIONAL_SIZE if lab == 1 else -(entry + 0.01) * DIRECTIONAL_SIZE
                    pnl -= polymarket_fee(entry) * DIRECTIONAL_SIZE
                    pnls.append(pnl)

            if pnls:
                total = sum(pnls)
                wr = sum(1 for p in pnls if p > 0) / len(pnls)
                mean_p = np.mean(pnls)
                std_p = np.std(pnls)
                sh = mean_p / std_p if std_p > 0 else 0
                print(f"  {mm_hi:>8.2f}  {dir_thr:>8.2f}  {n_mm:>6d}  {n_dir:>6d}  ${total:>+8.2f}  {wr:>5.1%}  {sh:>+6.3f}")

    print(f"\n{'=' * 70}")
    print("DONE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
