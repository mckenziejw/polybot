"""
Strategy: early_deep_level_ratio_wide_spread with mid-price + UTC hour filters

Hypothesis: Apply the early_deep_level_ratio_wide_spread signal (2s window, threshold 0.3)
but restrict to markets with mid-price between 0.35-0.65 and opening time in the first
6 hours UTC. The mid-price filter isolates uncertain markets where orderbook information
content is highest, while the time filter targets hours when market-maker activity is densest.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

SAMPLE_BOOK = Path("/home/ubuntu/wss_test/data/sample_1k/book_snapshots_sample_1k.parquet")
RESOLUTION_CACHE = Path("/home/ubuntu/wss_test/data/sample_1k/resolution_cache_sample.json")
OUTPUT_DIR = Path("/home/ubuntu/wss_test/swarm/generation_8/agent_2")

# Parameters
WINDOW_MS = 2000
SIGNAL_THRESHOLD = 0.3
MID_PRICE_LOW = 0.35
MID_PRICE_HIGH = 0.65
UTC_HOUR_MAX = 6
ENTRY_PRICE_CAP = 0.85
MIN_ORDER_TOKENS = 5

NEEDED_COLS = [
    "exchange_timestamp", "slug", "token_label", "mid_price", "spread",
    "bid_size_1", "bid_size_2", "bid_size_3", "bid_size_4", "bid_size_5",
    "ask_size_1", "ask_size_2", "ask_size_3", "ask_size_4", "ask_size_5",
]


def compute_fee(price):
    return 0.0222 * price * 0.25 * (price * (1 - price)) ** 2


def load_resolutions():
    with open(RESOLUTION_CACHE) as f:
        raw = json.load(f)
    resolutions = {}
    for slug, info in raw.items():
        if "winnerLabel" in info:
            resolutions[slug] = info["winnerLabel"]
        elif "winning_token" in info and "clobTokenIds" in info:
            wt = info["winning_token"]
            tokens = info["clobTokenIds"]
            outcomes = info.get("outcomes", ["Up", "Down"])
            if wt in tokens:
                resolutions[slug] = outcomes[tokens.index(wt)]
    return resolutions


def compute_signal(df):
    """Compute deep level ratio signal for a filtered dataframe. Fully vectorized."""
    bid_deep = df[["bid_size_3", "bid_size_4", "bid_size_5"]].values.sum(axis=1)
    bid_all = df[["bid_size_1", "bid_size_2", "bid_size_3", "bid_size_4", "bid_size_5"]].values.sum(axis=1)
    ask_deep = df[["ask_size_3", "ask_size_4", "ask_size_5"]].values.sum(axis=1)
    ask_all = df[["ask_size_1", "ask_size_2", "ask_size_3", "ask_size_4", "ask_size_5"]].values.sum(axis=1)

    bid_ratio = np.where(bid_all > 0, bid_deep / bid_all, 0.0)
    ask_ratio = np.where(ask_all > 0, ask_deep / ask_all, 0.0)
    df = df.copy()
    df["deep_ratio_diff"] = bid_ratio - ask_ratio
    return df


def run_backtest_core(df, resolutions, label="Main"):
    """Core backtest logic operating on pre-filtered dataframe."""
    if df.empty:
        print(f"  [{label}] No data after filters")
        return None

    # Compute per-market open timestamp, keep first WINDOW_MS
    market_open = df.groupby("slug")["exchange_timestamp"].transform("min")
    df = df[df["exchange_timestamp"] - market_open <= WINDOW_MS]

    df = compute_signal(df)

    # Aggregate per slug+token
    agg = df.groupby(["slug", "token_label"]).agg(
        deep_ratio_diff=("deep_ratio_diff", "mean"),
        mid_price=("mid_price", "median"),
        spread=("spread", "median"),
    ).reset_index()

    up = agg[agg["token_label"] == "Up"].set_index("slug")
    down = agg[agg["token_label"] == "Down"].set_index("slug")
    common = up.index.intersection(down.index)
    if len(common) == 0:
        print(f"  [{label}] No common slugs")
        return None

    up, down = up.loc[common], down.loc[common]
    signal = (up["deep_ratio_diff"] - down["deep_ratio_diff"]).values

    trades = pd.DataFrame({
        "slug": common,
        "signal": signal,
        "up_mid": up["mid_price"].values,
        "down_mid": down["mid_price"].values,
        "up_spread": up["spread"].values,
        "down_spread": down["spread"].values,
    })

    # Trade direction
    trades["direction"] = np.where(trades["signal"] > SIGNAL_THRESHOLD, "Up",
                          np.where(trades["signal"] < -SIGNAL_THRESHOLD, "Down", "None"))
    active = trades[trades["direction"] != "None"].copy()

    if active.empty:
        print(f"  [{label}] No trades pass threshold")
        return None

    # Entry price (ask side approximation)
    active["entry_price"] = np.where(
        active["direction"] == "Up",
        active["up_mid"] + active["up_spread"] / 2,
        active["down_mid"] + active["down_spread"] / 2,
    )
    active = active[active["entry_price"] <= ENTRY_PRICE_CAP].copy()

    # Resolve
    active["winner"] = active["slug"].map(resolutions)
    active = active.dropna(subset=["winner"])

    if active.empty:
        print(f"  [{label}] No resolved trades")
        return None

    active["won"] = (active["direction"] == active["winner"]).astype(int)
    active["fee"] = compute_fee(active["entry_price"].values)
    active["pnl"] = np.where(
        active["won"] == 1,
        1.0 - active["entry_price"] - active["fee"],
        -active["entry_price"] - active["fee"],
    ) * MIN_ORDER_TOKENS

    total = len(active)
    wins = int(active["won"].sum())
    wr = wins / total
    pnl = active["pnl"].sum()
    ppt = pnl / total
    cum_pnl = active["pnl"].cumsum()
    max_dd = (cum_pnl.cummax() - cum_pnl).max()
    sharpe = active["pnl"].mean() / active["pnl"].std() if active["pnl"].std() > 0 else 0.0

    print(f"  [{label}] {total} trades, WR={wr:.3f} ({wins}/{total}), "
          f"PnL=${pnl:.2f}, PnL/trade=${ppt:.4f}, MaxDD=${max_dd:.2f}, Sharpe={sharpe:.4f}")

    # Direction breakdown
    for d in ["Up", "Down"]:
        sub = active[active["direction"] == d]
        if len(sub) > 0:
            print(f"    {d}: {len(sub)} trades, WR={sub['won'].mean():.3f}, PnL=${sub['pnl'].sum():.2f}")

    return {
        "total_trades": total, "wins": wins, "win_rate": wr,
        "pnl_total": pnl, "pnl_per_trade": ppt,
        "max_drawdown": max_dd, "sharpe": sharpe,
        "trades_df": active,
    }


def run_backtest():
    print("Loading data (selected columns only)...")
    df = pd.read_parquet(SAMPLE_BOOK, columns=NEEDED_COLS)
    resolutions = load_resolutions()
    n_markets = df["slug"].nunique()
    print(f"Loaded {len(df):,} rows, {n_markets} markets, {len(resolutions)} resolutions")

    # Extract epoch from slug for UTC hour filter
    slugs = df["slug"].drop_duplicates()
    slug_epoch = slugs.str.extract(r"(\d+)$", expand=False).astype(np.int64)
    slug_hour = (slug_epoch % 86400) // 3600
    slug_df = pd.DataFrame({"slug": slugs.values, "utc_hour": slug_hour.values})

    # UTC hour filter
    hour_valid = slug_df[slug_df["utc_hour"] < UTC_HOUR_MAX]["slug"]
    print(f"Markets in UTC 0-{UTC_HOUR_MAX-1}: {len(hour_valid)} / {len(slug_df)}")

    # Mid-price filter: quick check on Up tokens
    # Get first few rows per slug to estimate mid_price (avoid full groupby)
    up_data = df[df["token_label"] == "Up"]
    up_mid_median = up_data.groupby("slug")["mid_price"].median()
    mid_valid = up_mid_median[(up_mid_median >= MID_PRICE_LOW) & (up_mid_median <= MID_PRICE_HIGH)].index
    print(f"Markets with mid-price [{MID_PRICE_LOW}, {MID_PRICE_HIGH}]: {len(mid_valid)} / {n_markets}")

    # Combined filters
    both_valid = set(hour_valid) & set(mid_valid)
    mid_only_valid = set(mid_valid)
    print(f"Markets passing both filters: {len(both_valid)}")

    # Run main backtest (both filters)
    print(f"\n{'='*60}")
    print("MAIN: Mid-price + UTC hour filters")
    print(f"{'='*60}")
    df_main = df[df["slug"].isin(both_valid)]
    result_main = run_backtest_core(df_main, resolutions, "MidPrice+UTC")

    # Comparison: mid-price filter only (no UTC hour)
    print(f"\n{'='*60}")
    print("COMPARISON: Mid-price filter only (no UTC hour)")
    print(f"{'='*60}")
    df_mid = df[df["slug"].isin(mid_only_valid)]
    result_mid = run_backtest_core(df_mid, resolutions, "MidPrice only")

    # Comparison: no filters (baseline)
    print(f"\n{'='*60}")
    print("BASELINE: No extra filters (original signal)")
    print(f"{'='*60}")
    result_base = run_backtest_core(df, resolutions, "Baseline")

    # Also test with wide spread filter (top 50% spread)
    print(f"\n{'='*60}")
    print("VARIANT: Mid-price + wide spread (top 50%)")
    print(f"{'='*60}")
    up_spread_median = up_data.groupby("slug")["spread"].median()
    spread_threshold = up_spread_median.median()
    wide_spread_slugs = up_spread_median[up_spread_median >= spread_threshold].index
    wide_mid_slugs = set(mid_valid) & set(wide_spread_slugs)
    print(f"  Wide spread + mid-price markets: {len(wide_mid_slugs)}")
    df_wide = df[df["slug"].isin(wide_mid_slugs)]
    result_wide = run_backtest_core(df_wide, resolutions, "MidPrice+WideSpread")

    # Pick best result for output
    best = result_main
    best_label = "main (mid-price + UTC hour)"

    # Write results
    if best is None:
        verdict = "failed"
        metrics = {"total_trades": 0, "win_rate": 0.0, "pnl_total": 0.0,
                   "pnl_per_trade": 0.0, "max_drawdown": 0.0, "sharpe": 0.0}
        next_steps = "Primary filters too restrictive. Mid-price + wide spread variant may work better."
    else:
        t = best
        metrics = {
            "total_trades": int(t["total_trades"]),
            "win_rate": round(float(t["win_rate"]), 4),
            "pnl_total": round(float(t["pnl_total"]), 4),
            "pnl_per_trade": round(float(t["pnl_per_trade"]), 4),
            "max_drawdown": round(float(t["max_drawdown"]), 4),
            "sharpe": round(float(t["sharpe"]), 4),
        }
        if t["total_trades"] >= 20 and t["win_rate"] > 0.55 and t["pnl_total"] > 0 and t["sharpe"] > 0.3:
            verdict = "promising"
        elif t["total_trades"] >= 10 and t["win_rate"] > 0.52 and t["pnl_total"] > 0:
            verdict = "marginal"
        else:
            verdict = "failed"

        if verdict == "promising":
            next_steps = ("Validate on full 7,187-market dataset. Test spread-weighted position sizing. "
                          "Explore combining with book_imbalance for signal confirmation.")
        elif verdict == "marginal":
            next_steps = ("Relax UTC hours to 0-12. Add wide-spread filter. "
                          "Test lower threshold (0.2). Run on full dataset.")
        else:
            next_steps = ("UTC hour filter may be too restrictive for sample. "
                          "Try mid-price + wide-spread combo instead. Run on full dataset for larger N.")

    # Include comparison metrics in results
    comparisons = {}
    for label, r in [("mid_price_only", result_mid), ("baseline", result_base),
                     ("mid_price_wide_spread", result_wide)]:
        if r is not None:
            comparisons[label] = {
                "total_trades": int(r["total_trades"]),
                "win_rate": round(float(r["win_rate"]), 4),
                "pnl_total": round(float(r["pnl_total"]), 4),
                "sharpe": round(float(r["sharpe"]), 4),
            }

    results = {
        "strategy_name": "early_deep_level_ratio_wide_spread_midprice_utc",
        "description": "Deep level ratio signal (2s window, threshold 0.3) filtered by mid-price 0.35-0.65 and UTC hours 0-6",
        "hypothesis": ("Mid-price filter isolates uncertain markets where orderbook information "
                       "content is highest. UTC hour filter targets peak market-maker activity."),
        "metrics": metrics,
        "parameters": {
            "window_ms": WINDOW_MS,
            "signal_threshold": SIGNAL_THRESHOLD,
            "mid_price_range": [MID_PRICE_LOW, MID_PRICE_HIGH],
            "utc_hour_max": UTC_HOUR_MAX,
            "entry_price_cap": ENTRY_PRICE_CAP,
            "min_order_tokens": MIN_ORDER_TOKENS,
        },
        "comparisons": comparisons,
        "verdict": verdict,
        "next_steps": next_steps,
    }

    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nVerdict: {verdict}")
    print(f"Results written to {OUTPUT_DIR / 'results.json'}")


if __name__ == "__main__":
    run_backtest()
