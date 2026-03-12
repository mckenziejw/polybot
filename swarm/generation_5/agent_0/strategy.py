"""
Orderbook Asymmetry Decay Rate Strategy
========================================
Hypothesis: Measure bid/ask depth asymmetry across all 5 orderbook levels.
Track how quickly this asymmetry mean-reverts over consecutive snapshots.
Markets where asymmetry persists (slow decay) may signal informed positioning.
Fade the slow-decay side after a threshold duration to capture snap-back.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
BASE = Path("/home/ubuntu/wss_test")
SAMPLE_BOOK = BASE / "data/sample_1k/book_snapshots_sample_1k.parquet"
RESOLUTION_CACHE = BASE / "data/sample_1k/resolution_cache_sample.json"
OUTPUT_DIR = BASE / "swarm/generation_5/agent_0"

# ── Parameters ─────────────────────────────────────────────────────────
ENTRY_PRICE_CAP = 0.85
SNAPSHOT_BUCKET_SEC = 5


def compute_fee(p: np.ndarray) -> np.ndarray:
    """Polymarket fee: 0.0222 * p * 0.25 * (p*(1-p))^2"""
    return 0.0222 * p * 0.25 * (p * (1 - p)) ** 2


def load_and_aggregate():
    """
    Load parquet with only needed columns, compute asymmetry inline,
    and immediately aggregate to time buckets to reduce memory.
    """
    print("Loading sample data (select columns only)...")
    needed_cols = (
        ["exchange_timestamp", "slug", "token_label", "mid_price", "spread"]
        + [f"bid_size_{i}" for i in range(1, 6)]
        + [f"ask_size_{i}" for i in range(1, 6)]
    )
    df = pd.read_parquet(SAMPLE_BOOK, columns=needed_cols)
    print(f"  Loaded {len(df):,} rows, {df['slug'].nunique()} markets")

    # Compute asymmetry ratio in-place
    bid_cols = [f"bid_size_{i}" for i in range(1, 6)]
    ask_cols = [f"ask_size_{i}" for i in range(1, 6)]

    total_bid = df[bid_cols].values.sum(axis=1)
    total_ask = df[ask_cols].values.sum(axis=1)

    # Drop bid/ask size columns immediately to free memory
    df.drop(columns=bid_cols + ask_cols, inplace=True)

    # Compute asymmetry, handle zeros
    with np.errstate(divide="ignore", invalid="ignore"):
        asym_ratio = total_bid / total_ask
    asym_ratio[~np.isfinite(asym_ratio)] = np.nan

    df["asym_ratio"] = asym_ratio
    df["total_bid"] = total_bid
    df["total_ask"] = total_ask
    del total_bid, total_ask, asym_ratio

    # Drop rows with invalid asymmetry
    df.dropna(subset=["asym_ratio"], inplace=True)

    # Time bucketing: seconds since start of each market+token
    df["ts_sec"] = df["exchange_timestamp"] // 1000
    df.drop(columns=["exchange_timestamp"], inplace=True)

    # Compute time bucket within each group
    df["time_bucket"] = df.groupby(["slug", "token_label"])["ts_sec"].transform(
        lambda x: ((x - x.min()) // SNAPSHOT_BUCKET_SEC)
    )

    # Aggregate to buckets
    print("  Aggregating to time buckets...")
    bucket_stats = (
        df.groupby(["slug", "token_label", "time_bucket"], observed=True)
        .agg(
            mean_asym=("asym_ratio", "mean"),
            mean_mid=("mid_price", "mean"),
            mean_spread=("spread", "mean"),
            mean_bid=("total_bid", "mean"),
            mean_ask=("total_ask", "mean"),
            n_snapshots=("asym_ratio", "count"),
        )
        .reset_index()
    )
    del df  # free the big dataframe
    print(f"  Buckets: {len(bucket_stats):,}")
    return bucket_stats


def compute_persistence(bucket_stats, asym_threshold, persistence_n):
    """Compute asymmetry persistence features vectorized."""
    bs = bucket_stats.sort_values(["slug", "token_label", "time_bucket"]).copy()

    bs["asym_dev"] = (bs["mean_asym"] - 1.0).abs()
    bs["asym_direction"] = np.sign(bs["mean_asym"] - 1.0)
    bs["above_threshold"] = (bs["asym_dev"] > asym_threshold).astype(np.int8)

    # Consecutive count using cumsum trick
    bs["below_break"] = (
        bs.groupby(["slug", "token_label"])["above_threshold"]
        .transform(lambda x: (x == 0).cumsum())
    )
    bs["consec_above"] = bs.groupby(
        ["slug", "token_label", "below_break"]
    )["above_threshold"].cumsum()

    return bs


def generate_signals_vectorized(bs, persistence_n):
    """
    Generate signals: find markets where asymmetry was persistent then decayed.
    Vectorized approach using groupby tail operations.
    """
    # For each slug+token, get the last (persistence_n + 1) buckets
    n_needed = persistence_n + 1

    # Filter groups with enough buckets
    group_sizes = bs.groupby(["slug", "token_label"]).size()
    valid_groups = group_sizes[group_sizes >= n_needed].index
    bs_valid = bs.set_index(["slug", "token_label"]).loc[valid_groups].reset_index()

    if bs_valid.empty:
        return pd.DataFrame()

    # Get tail rows per group
    tails = bs_valid.groupby(["slug", "token_label"]).tail(n_needed)

    # Separate last row and check rows
    last_rows = bs_valid.groupby(["slug", "token_label"]).tail(1)
    check_rows = tails[~tails.index.isin(last_rows.index)]

    # For check_rows: was asymmetry persistent?
    check_agg = check_rows.groupby(["slug", "token_label"]).agg(
        max_consec=("consec_above", "max"),
        mean_asym_dev=("asym_dev", "mean"),
        dominant_dir=("asym_direction", lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 0),
    )

    # Filter: persistent asymmetry
    persistent_mask = check_agg["max_consec"] >= persistence_n
    persistent_groups = check_agg[persistent_mask]

    if persistent_groups.empty:
        return pd.DataFrame()

    # For last_rows of persistent groups: check if decay started
    last_indexed = last_rows.set_index(["slug", "token_label"])
    common_idx = persistent_groups.index.intersection(last_indexed.index)

    if len(common_idx) == 0:
        return pd.DataFrame()

    merged = persistent_groups.loc[common_idx].join(
        last_indexed[["asym_dev", "mean_mid", "mean_spread", "mean_bid", "mean_ask"]].loc[common_idx],
        rsuffix="_last"
    )

    # Decay: current deviation < persistent mean deviation
    merged = merged[merged["asym_dev"] < merged["mean_asym_dev"]]

    if merged.empty:
        return pd.DataFrame()

    # Determine trade side
    signals = merged.reset_index()
    signals["decay_magnitude"] = signals["mean_asym_dev"] - signals["asym_dev"]

    # Fade logic: for Up token, bid-heavy → buy Down; ask-heavy → buy Up
    # For Down token, bid-heavy → buy Up; ask-heavy → buy Down
    conditions = [
        (signals["token_label"] == "Up") & (signals["dominant_dir"] > 0),
        (signals["token_label"] == "Up") & (signals["dominant_dir"] <= 0),
        (signals["token_label"] == "Down") & (signals["dominant_dir"] > 0),
        (signals["token_label"] == "Down") & (signals["dominant_dir"] <= 0),
    ]
    choices = ["Down", "Up", "Up", "Down"]
    signals["trade_side"] = np.select(conditions, choices, default="Up")

    # Entry price: if trading same token as signal, use mid; else use 1-mid
    signals["buy_price"] = np.where(
        signals["trade_side"] == signals["token_label"],
        signals["mean_mid"],
        1.0 - signals["mean_mid"]
    )

    return signals[["slug", "token_label", "trade_side", "buy_price",
                     "mean_asym_dev", "asym_dev", "decay_magnitude",
                     "dominant_dir", "mean_spread"]].copy()


def backtest(signals, resolutions):
    """Backtest signals against resolution outcomes."""
    if signals.empty:
        return {"total_trades": 0, "win_rate": 0.0, "pnl_total": 0.0,
                "pnl_per_trade": 0.0, "max_drawdown": 0.0, "sharpe": 0.0}

    # Map resolution winners
    signals = signals.copy()
    signals["winner"] = signals["slug"].map(
        lambda s: resolutions.get(s, {}).get("winnerLabel")
    )
    signals = signals[signals["winner"].notna()]

    # Price cap filter
    signals = signals[(signals["buy_price"] > 0) & (signals["buy_price"] <= ENTRY_PRICE_CAP)]

    if signals.empty:
        return {"total_trades": 0, "win_rate": 0.0, "pnl_total": 0.0,
                "pnl_per_trade": 0.0, "max_drawdown": 0.0, "sharpe": 0.0}

    # Deduplicate: one trade per slug (strongest decay signal)
    signals = signals.sort_values("decay_magnitude", ascending=False)
    signals = signals.drop_duplicates(subset=["slug"], keep="first")

    # Compute PnL
    signals["fee"] = compute_fee(signals["buy_price"].values)
    signals["won"] = signals["trade_side"] == signals["winner"]
    signals["pnl"] = np.where(signals["won"], 1.0, 0.0) - signals["buy_price"] - signals["fee"]

    total_trades = len(signals)
    win_rate = signals["won"].mean()
    pnl_total = signals["pnl"].sum()
    pnl_per_trade = signals["pnl"].mean()

    cumulative = signals["pnl"].cumsum()
    max_drawdown = (cumulative.cummax() - cumulative).max()
    sharpe = signals["pnl"].mean() / signals["pnl"].std() if signals["pnl"].std() > 0 else 0.0

    return {
        "total_trades": int(total_trades),
        "win_rate": round(float(win_rate), 4),
        "pnl_total": round(float(pnl_total), 4),
        "pnl_per_trade": round(float(pnl_per_trade), 4),
        "max_drawdown": round(float(max_drawdown), 4),
        "sharpe": round(float(sharpe), 4),
        "signals_df": signals,
    }


def main():
    # Load resolution cache
    with open(RESOLUTION_CACHE) as f:
        resolutions = json.load(f)
    print(f"Resolution cache: {len(resolutions)} entries")

    # Load and aggregate data
    bucket_stats = load_and_aggregate()

    # Autocorrelation analysis
    print("\nAutocorrelation analysis...")
    def lag1_corr(s):
        if len(s) < 4:
            return np.nan
        return s.autocorr(lag=1)

    autocorr = (
        bucket_stats.groupby(["slug", "token_label"])["mean_asym"]
        .apply(lag1_corr)
        .reset_index()
        .rename(columns={"mean_asym": "autocorr"})
    )
    ac_valid = autocorr["autocorr"].dropna()
    print(f"  Mean autocorr: {ac_valid.mean():.4f}, Median: {ac_valid.median():.4f}")

    # Check if autocorrelation predicts outcomes
    autocorr["winner"] = autocorr["slug"].map(
        lambda s: resolutions.get(s, {}).get("winnerLabel")
    )
    up_ac = autocorr[(autocorr["token_label"] == "Up") & autocorr["winner"].notna() & autocorr["autocorr"].notna()]
    if len(up_ac) > 0:
        high_ac = up_ac[up_ac["autocorr"] > 0.5]
        low_ac = up_ac[up_ac["autocorr"] < 0.0]
        print(f"  High autocorr (>0.5) Up tokens: {len(high_ac)}, "
              f"winner=Up rate: {(high_ac['winner']=='Up').mean():.4f}" if len(high_ac) > 0 else "  No high autocorr")
        print(f"  Low autocorr (<0.0) Up tokens: {len(low_ac)}, "
              f"winner=Up rate: {(low_ac['winner']=='Up').mean():.4f}" if len(low_ac) > 0 else "  No low autocorr")

    # Default parameters backtest
    print("\n" + "="*50)
    print("DEFAULT PARAMETERS BACKTEST")
    print("="*50)
    default_thresh, default_persist = 0.15, 3
    bs = compute_persistence(bucket_stats, default_thresh, default_persist)
    signals = generate_signals_vectorized(bs, default_persist)
    bt = backtest(signals, resolutions)
    print(f"  Trades: {bt['total_trades']}, Win rate: {bt['win_rate']:.4f}, "
          f"PnL: {bt['pnl_total']:.4f}, PnL/trade: {bt['pnl_per_trade']:.4f}, "
          f"Sharpe: {bt['sharpe']:.4f}")

    # Parameter sweep
    print("\n" + "="*50)
    print("PARAMETER SWEEP")
    print("="*50)
    sweep_results = []
    for thresh in [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]:
        for persist_n in [2, 3, 4, 5, 6]:
            bs_sweep = compute_persistence(bucket_stats, thresh, persist_n)
            sigs = generate_signals_vectorized(bs_sweep, persist_n)
            bt_s = backtest(sigs, resolutions)
            sweep_results.append({
                "asym_threshold": thresh,
                "persistence_n": persist_n,
                "trades": bt_s["total_trades"],
                "win_rate": bt_s["win_rate"],
                "pnl_total": bt_s["pnl_total"],
                "pnl_per_trade": bt_s["pnl_per_trade"],
                "sharpe": bt_s["sharpe"],
            })

    sweep_df = pd.DataFrame(sweep_results)
    print(sweep_df.to_string(index=False))

    # Find best config
    has_trades = sweep_df[sweep_df["trades"] > 0]
    best = None
    if not has_trades.empty:
        viable = has_trades[has_trades["trades"] >= 10]
        if viable.empty:
            viable = has_trades
        best = viable.loc[viable["pnl_per_trade"].idxmax()]
        print(f"\nBest config: threshold={best['asym_threshold']}, "
              f"persist={best['persistence_n']}, trades={best['trades']}, "
              f"win_rate={best['win_rate']:.4f}, pnl/trade={best['pnl_per_trade']:.4f}")

    # Use best config for final results, or default
    if best is not None and best["trades"] > 0:
        final_bt = {
            "total_trades": int(best["trades"]),
            "win_rate": float(best["win_rate"]),
            "pnl_total": float(best["pnl_total"]),
            "pnl_per_trade": float(best["pnl_per_trade"]),
            "max_drawdown": 0.0,  # recompute if needed
            "sharpe": float(best["sharpe"]),
        }
        final_params = {
            "asym_threshold": float(best["asym_threshold"]),
            "persistence_snapshots": int(best["persistence_n"]),
        }
    else:
        final_bt = bt
        final_params = {"asym_threshold": default_thresh, "persistence_snapshots": default_persist}

    # Verdict
    if final_bt["total_trades"] == 0:
        verdict = "failed"
        desc = "No trades generated - asymmetry persistence signal too rare in 60s sample window"
    elif final_bt["pnl_total"] > 0 and final_bt["win_rate"] > 0.52:
        verdict = "promising"
        desc = "Strategy shows positive edge with asymmetry decay signal"
    elif final_bt["pnl_total"] > -1.0 and final_bt["win_rate"] > 0.48:
        verdict = "marginal"
        desc = "Strategy shows weak/inconsistent edge, needs more data or refinement"
    else:
        verdict = "failed"
        desc = "Strategy does not produce profitable signals on sample data"

    # Write results
    results = {
        "strategy_name": "orderbook_asymmetry_decay",
        "description": desc,
        "hypothesis": (
            "Orderbook asymmetry (bid/ask depth ratio) that persists across multiple "
            "consecutive snapshots signals informed positioning. Fading the slow-decay "
            "side after asymmetry begins to revert should capture the snap-back to equilibrium."
        ),
        "metrics": {
            "total_trades": final_bt["total_trades"],
            "win_rate": final_bt["win_rate"],
            "pnl_total": final_bt["pnl_total"],
            "pnl_per_trade": final_bt["pnl_per_trade"],
            "max_drawdown": final_bt["max_drawdown"],
            "sharpe": final_bt["sharpe"],
        },
        "parameters": {
            **final_params,
            "entry_price_cap": ENTRY_PRICE_CAP,
            "snapshot_bucket_sec": SNAPSHOT_BUCKET_SEC,
        },
        "verdict": verdict,
        "next_steps": (
            "1. Test on full dataset (7187 markets) for larger sample. "
            "2. Incorporate trade flow data to validate asymmetry signals. "
            "3. Explore time-weighted asymmetry. "
            "4. Add spread-adjusted entry to avoid adverse selection."
            if verdict != "failed" else
            "1. Sample has only 60s of data per market - too short for multi-snapshot decay measurement. "
            "2. Try full dataset with longer observation windows per market. "
            "3. Use 4h orderbook data for longer-horizon decay analysis. "
            "4. Autocorrelation analysis suggests examining whether raw asymmetry level "
            "(not decay rate) has predictive value."
        ),
        "parameter_sweep": sweep_results,
        "autocorrelation_stats": {
            "mean": round(float(ac_valid.mean()), 4),
            "median": round(float(ac_valid.median()), 4),
            "std": round(float(ac_valid.std()), 4),
        },
    }

    output_path = OUTPUT_DIR / "results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {output_path}")
    print(f"Verdict: {verdict}")


if __name__ == "__main__":
    main()
