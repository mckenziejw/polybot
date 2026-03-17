#!/usr/bin/env python3
"""
Research Agent 7: Entry/exit timing optimization.

Instead of "predict the winner at time T", asks:
1. What is the optimal ENTRY time? (earliest window with reliable signal)
2. What is the optimal EXIT time? (when should we close?)
3. Can we use sequential observations to improve confidence over time?
4. Price trajectory analysis: what happens AFTER our prediction?
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score

warnings.filterwarnings("ignore")

FEATURE_CACHE = Path("data/xgb_features_v2.parquet")
RESOLUTION_CACHE = Path("data/resolution_cache.json")
DATA_DIR = Path("data/telonex_book_snapshots")
MARKET_DURATION_MS = 300_000


def load_data():
    df = pd.read_parquet(FEATURE_CACHE)
    with open(RESOLUTION_CACHE) as f:
        raw = json.load(f)
    resolutions = {}
    for slug, info in raw.items():
        winner = info.get("winning_token") or info.get("winningTokenId", "")
        if winner:
            resolutions[slug] = winner
    return df, resolutions


def resample_to_1s(token_df, open_ms, close_ms):
    market_df = token_df[
        (token_df["exchange_timestamp"] >= open_ms) &
        (token_df["exchange_timestamp"] < close_ms)
    ].copy()
    if market_df.empty:
        return market_df
    market_df["sec"] = ((market_df["exchange_timestamp"] - open_ms) / 1000).astype(int)
    resampled = market_df.groupby("sec").last().reset_index()
    resampled["exchange_timestamp"] = open_ms + resampled["sec"] * 1000
    return resampled


def experiment_1_optimal_entry(df):
    """Find the earliest observation window with reliable signal."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Optimal Entry Timing")
    print("=" * 70)
    print("Question: What's the earliest window where our model is reliable?")

    windows = sorted(df["observe_time_s"].unique())
    feature_cols = [c for c in df.columns if c not in
                    ["slug", "open_ts", "label", "observe_time_s",
                     "winning_token", "token_id_up", "token_id_down"]]

    # Time-based split
    slugs = df.sort_values("open_ts")["slug"].unique()
    split_idx = int(len(slugs) * 0.7)
    train_slugs = set(slugs[:split_idx])
    test_slugs = set(slugs[split_idx:])

    params = {
        "objective": "binary:logistic",
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 10,
        "verbosity": 0,
    }

    results = []
    for w in windows:
        wdf = df[df["observe_time_s"] == w].dropna(subset=["label"] + feature_cols[:10])
        train = wdf[wdf["slug"].isin(train_slugs)]
        test = wdf[wdf["slug"].isin(test_slugs)]

        if len(train) < 50 or len(test) < 20:
            continue

        valid_cols = [c for c in feature_cols if train[c].notna().sum() > len(train) * 0.5]
        X_train = train[valid_cols].fillna(0).values
        y_train = train["label"].values
        X_test = test[valid_cols].fillna(0).values
        y_test = test["label"].values

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        model = xgb.train(params, dtrain, num_boost_round=150,
                           evals=[(dtest, "test")], verbose_eval=False,
                           early_stopping_rounds=30)

        preds = model.predict(dtest)
        auc = roc_auc_score(y_test, preds) if len(set(y_test)) > 1 else 0.5
        base = y_test.mean()

        # Confidence-filtered accuracy at different thresholds
        for thr in [0.60, 0.65, 0.70, 0.75, 0.80]:
            mask = (preds > thr) | (preds < (1 - thr))
            if mask.sum() > 5:
                acc = accuracy_score(y_test[mask], preds[mask] > 0.5)
                results.append({
                    "window": w, "threshold": thr, "n_trades": mask.sum(),
                    "accuracy": acc, "auc": auc, "base_rate": base,
                    "lift": acc - base
                })

    rdf = pd.DataFrame(results)
    if rdf.empty:
        print("  No results generated.")
        return

    # Display results
    print(f"\n  {'Window':>8s}  {'Thr':>5s}  {'Trades':>7s}  {'Acc':>6s}  {'Base':>6s}  {'Lift':>6s}  {'AUC':>6s}")
    print("  " + "-" * 55)
    for _, r in rdf.iterrows():
        marker = " <<<" if r["lift"] > 0.10 and r["n_trades"] > 20 else ""
        print(f"  {r['window']:>7.0f}s  {r['threshold']:>5.2f}  {r['n_trades']:>7.0f}  "
              f"{r['accuracy']:>5.1%}  {r['base_rate']:>5.1%}  {r['lift']:>+5.1%}  {r['auc']:>5.3f}{marker}")

    # Find earliest profitable window
    good = rdf[(rdf["lift"] > 0.05) & (rdf["n_trades"] > 15)]
    if not good.empty:
        earliest = good.loc[good["window"].idxmin()]
        print(f"\n  Earliest profitable window: {earliest['window']:.0f}s "
              f"(thr={earliest['threshold']:.2f}, lift={earliest['lift']:.1%}, n={earliest['n_trades']:.0f})")
    else:
        print("\n  No window achieves >5% lift with sufficient trades")


def experiment_2_sequential_confidence(df):
    """Can we use sequential windows to build confidence?"""
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Sequential Confidence Building")
    print("=" * 70)
    print("Question: Does observing multiple windows improve prediction?")

    feature_cols = [c for c in df.columns if c not in
                    ["slug", "open_ts", "label", "observe_time_s",
                     "winning_token", "token_id_up", "token_id_down"]]

    windows = sorted(df["observe_time_s"].unique())
    slugs = df.sort_values("open_ts")["slug"].unique()
    split_idx = int(len(slugs) * 0.7)
    test_slugs = set(slugs[split_idx:])

    # Build per-window models
    params = {
        "objective": "binary:logistic",
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 10,
        "verbosity": 0,
    }

    # Get predictions from each window for test markets
    market_preds = {}  # slug -> {window: pred}
    market_labels = {}

    for w in windows:
        wdf = df[df["observe_time_s"] == w].dropna(subset=["label"])
        train = wdf[~wdf["slug"].isin(test_slugs)]
        test = wdf[wdf["slug"].isin(test_slugs)]

        if len(train) < 50 or len(test) < 10:
            continue

        valid_cols = [c for c in feature_cols if train[c].notna().sum() > len(train) * 0.5]
        X_train = train[valid_cols].fillna(0).values
        X_test = test[valid_cols].fillna(0).values

        dtrain = xgb.DMatrix(X_train, label=train["label"].values)
        dtest = xgb.DMatrix(X_test)
        model = xgb.train(params, dtrain, num_boost_round=150, verbose_eval=False)
        preds = model.predict(dtest)

        for slug, pred, label in zip(test["slug"].values, preds, test["label"].values):
            if slug not in market_preds:
                market_preds[slug] = {}
                market_labels[slug] = label
            market_preds[slug][w] = pred

    # Sequential strategy: at each window, check if we have enough confidence
    print(f"\n  SEQUENTIAL DECISION STRATEGY:")
    print(f"  At each window, average all predictions so far. Trade when avg > threshold.")

    for thr in [0.65, 0.70, 0.75, 0.80]:
        decisions = []  # (slug, window_decided, avg_pred, label)

        for slug, wpreds in market_preds.items():
            label = market_labels[slug]
            cumulative = []
            decided = False

            for w in windows:
                if w in wpreds:
                    cumulative.append(wpreds[w])
                    avg = np.mean(cumulative)
                    if avg > thr or avg < (1 - thr):
                        decisions.append((slug, w, avg, label))
                        decided = True
                        break

            # If never decided, skip
            if not decided and cumulative:
                avg = np.mean(cumulative)
                # Only enter at the last window if confident enough
                if avg > thr or avg < (1 - thr):
                    decisions.append((slug, windows[-1], avg, label))

        if not decisions:
            print(f"  thr={thr:.2f}: no decisions made")
            continue

        dec_df = pd.DataFrame(decisions, columns=["slug", "window", "avg_pred", "label"])
        acc = accuracy_score(dec_df["label"], dec_df["avg_pred"] > 0.5)
        avg_window = dec_df["window"].mean()
        n = len(dec_df)

        # Early vs late decisions
        early = dec_df[dec_df["window"] <= 60]
        late = dec_df[dec_df["window"] > 60]
        early_acc = accuracy_score(early["label"], early["avg_pred"] > 0.5) if len(early) > 5 else float("nan")
        late_acc = accuracy_score(late["label"], late["avg_pred"] > 0.5) if len(late) > 5 else float("nan")

        print(f"  thr={thr:.2f}: n={n:>4d}, acc={acc:.1%}, avg_entry={avg_window:.0f}s "
              f"| early(<60s)={early_acc:.1%}(n={len(early)}), late(>60s)={late_acc:.1%}(n={len(late)})")


def experiment_3_price_trajectory(df):
    """Analyze what happens to prices AFTER our prediction."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Post-Prediction Price Trajectory")
    print("=" * 70)
    print("Question: After entering at time T, what is the price path?")

    # Load raw parquet files and trace price paths
    parquet_files = sorted(DATA_DIR.glob("btc-updown-5m-*.parquet"))
    print(f"  Found {len(parquet_files)} market files")

    # For each market, trace mid-price at 10s intervals after entry
    trajectories_win = []  # price paths for winning trades
    trajectories_lose = []
    n_processed = 0

    for pf in parquet_files[:2000]:
        slug = pf.stem
        try:
            open_ts = int(slug.split("-")[-1])
        except ValueError:
            continue

        open_ms = open_ts * 1000
        close_ms = open_ms + MARKET_DURATION_MS

        try:
            raw = pd.read_parquet(pf)
        except Exception:
            continue

        if "token_label" not in raw.columns or "mid_price" not in raw.columns:
            continue

        up_df = raw[raw["token_label"] == "Up"]
        resampled = resample_to_1s(up_df, open_ms, close_ms)
        if len(resampled) < 120:
            continue

        # Check if this market is in our feature cache
        market_rows = df[(df["slug"] == slug) & (df["observe_time_s"] == 60)]
        if market_rows.empty:
            continue

        label = market_rows["label"].iloc[0]
        mid_at_60s = resampled[resampled["sec"] == 60]["mid_price"].values
        if len(mid_at_60s) == 0:
            continue
        entry_mid = mid_at_60s[0]

        # Trace price at 10s intervals from t=60 to t=300
        path = []
        for t in range(60, 301, 10):
            row = resampled[resampled["sec"] == t]
            if len(row) > 0:
                path.append(row["mid_price"].values[0] - entry_mid)
            elif path:
                path.append(path[-1])  # forward fill
            else:
                path.append(0.0)

        if label == 1:
            trajectories_win.append(path)
        else:
            trajectories_lose.append(path)

        n_processed += 1

    print(f"  Processed {n_processed} markets ({len(trajectories_win)} wins, {len(trajectories_lose)} losses)")

    if not trajectories_win or not trajectories_lose:
        print("  Insufficient data for trajectory analysis")
        return

    win_arr = np.array(trajectories_win)
    lose_arr = np.array(trajectories_lose)
    times = list(range(60, 301, 10))

    print(f"\n  Mid-price change (from entry at t=60s) for WINNING markets:")
    print(f"  {'Time':>6s}  {'Mean':>8s}  {'Median':>8s}  {'Std':>8s}  {'Min':>8s}  {'Max':>8s}")
    for i, t in enumerate(times):
        vals = win_arr[:, i]
        print(f"  {t:>5d}s  {np.mean(vals):>+7.4f}  {np.median(vals):>+7.4f}  "
              f"{np.std(vals):>7.4f}  {np.min(vals):>+7.4f}  {np.max(vals):>+7.4f}")

    print(f"\n  Mid-price change (from entry at t=60s) for LOSING markets:")
    for i, t in enumerate(times):
        vals = lose_arr[:, i]
        print(f"  {t:>5d}s  {np.mean(vals):>+7.4f}  {np.median(vals):>+7.4f}  "
              f"{np.std(vals):>7.4f}  {np.min(vals):>+7.4f}  {np.max(vals):>+7.4f}")

    # Max drawdown analysis
    print(f"\n  MAX DRAWDOWN from entry (entry at t=60s, hold to close):")
    win_mdd = np.min(win_arr, axis=1)
    lose_mdd = np.min(lose_arr, axis=1)
    for name, mdd in [("Winning", win_mdd), ("Losing", lose_mdd)]:
        print(f"    {name}: mean={np.mean(mdd):+.4f}, median={np.median(mdd):+.4f}, "
              f"worst={np.min(mdd):+.4f}")

    # Best exit time
    print(f"\n  OPTIMAL EXIT (max avg PnL):")
    for name, arr in [("Winning", win_arr), ("Losing", lose_arr), ("All", np.vstack([win_arr, lose_arr]))]:
        mean_path = np.mean(arr, axis=0)
        best_idx = np.argmax(mean_path)
        print(f"    {name}: best exit at t={times[best_idx]}s, PnL={mean_path[best_idx]:+.4f}")


def main():
    print("=" * 70)
    print("RESEARCH: Entry/Exit Timing Optimization")
    print("=" * 70)

    df, resolutions = load_data()
    print(f"Loaded {len(df)} samples, {df['slug'].nunique()} markets")

    experiment_1_optimal_entry(df)
    experiment_2_sequential_confidence(df)
    experiment_3_price_trajectory(df)

    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print("""
Next steps based on findings:
1. If early windows (10-30s) have any signal: enables faster entry, more trading time
2. If sequential averaging helps: implement rolling confidence in live system
3. If max drawdown is small for winners: tighter stops are viable
4. If optimal exit is before market close: exit early, reduce variance
""")


if __name__ == "__main__":
    main()
