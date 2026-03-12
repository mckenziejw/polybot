"""
Cross-Token Implied Probability Divergence Strategy
====================================================
Memory-efficient implementation: load only needed columns, process per-market.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

DATA_PATH = Path("/home/ubuntu/wss_test/data/sample_1k/book_snapshots_sample_1k.parquet")
RESOLUTION_PATH = Path("/home/ubuntu/wss_test/data/sample_1k/resolution_cache_sample.json")
OUTPUT_DIR = Path("/home/ubuntu/wss_test/swarm/generation_6/agent_1")

# Strategy parameters
SUM_THRESHOLD = 0.92
MAX_ENTRY_PRICE = 0.85
MIN_ORDER_SIZE = 5
DEPTH_RATIO_MIN = 1.2

NEEDED_COLS = [
    "exchange_timestamp", "slug", "asset_id", "token_label",
    "bid_price_1", "bid_size_1", "bid_size_2", "bid_size_3", "bid_size_4", "bid_size_5",
    "ask_price_1",
]


def calc_fee(p):
    """Vectorized fee calculation."""
    return 0.0222 * p * 0.25 * (p * (1 - p)) ** 2


def load_resolution_cache(path):
    with open(path) as f:
        raw = json.load(f)
    results = {}
    unresolved = {}
    for slug, info in raw.items():
        if "winnerLabel" in info:
            results[slug] = info["winnerLabel"]
        elif "outcomes" in info and "clobTokenIds" in info and "winningTokenId" in info:
            idx = info["clobTokenIds"].index(info["winningTokenId"])
            results[slug] = info["outcomes"][idx]
        elif "token_outcomes" in info:
            for token_id, val in info["token_outcomes"].items():
                if val == 1.0:
                    unresolved[slug] = token_id
                    break
    return results, unresolved


def run_backtest():
    print("Loading data (selected columns only)...")
    df = pd.read_parquet(DATA_PATH, columns=NEEDED_COLS)
    print(f"Loaded {len(df):,} rows, {df['slug'].nunique()} markets")

    resolution_cache, unresolved = load_resolution_cache(RESOLUTION_PATH)

    # Resolve token_id-based entries using asset_id mapping from data
    if unresolved:
        token_map = (
            df[df["slug"].isin(unresolved.keys())]
            [["slug", "asset_id", "token_label"]]
            .drop_duplicates()
        )
        for slug, winning_id in unresolved.items():
            match = token_map[
                (token_map["slug"] == slug) & (token_map["asset_id"] == winning_id)
            ]
            if len(match) > 0:
                resolution_cache[slug] = match.iloc[0]["token_label"]

    print(f"Resolved {len(resolution_cache)} markets")

    # Filter to resolved markets only
    resolved_slugs = set(resolution_cache.keys())
    df = df[df["slug"].isin(resolved_slugs)].copy()
    print(f"Filtered: {len(df):,} rows, {df['slug'].nunique()} markets")

    # ── Compute depth ratio (vectorized) ─────────────────────────────────
    deep_bid = (df["bid_size_2"].fillna(0) + df["bid_size_3"].fillna(0) +
                df["bid_size_4"].fillna(0) + df["bid_size_5"].fillna(0))
    top_bid = df["bid_size_1"].fillna(0).clip(lower=1e-9)
    df["depth_ratio"] = deep_bid / top_bid

    # ── Split Up/Down and merge on (slug, exchange_timestamp) ────────────
    # Keep only essential columns to minimize memory during merge
    print("Pairing Up/Down snapshots...")
    up_cols = ["slug", "exchange_timestamp", "bid_price_1", "ask_price_1", "depth_ratio"]
    dn_cols = up_cols.copy()

    up = df.loc[df["token_label"] == "Up", up_cols]
    down = df.loc[df["token_label"] == "Down", dn_cols]

    # Free main df
    del df

    pairs = up.merge(down, on=["slug", "exchange_timestamp"], suffixes=("_up", "_dn"))
    del up, down
    print(f"Paired snapshots: {len(pairs):,}")

    # ── Compute signals (vectorized) ─────────────────────────────────────
    bid_sum = pairs["bid_price_1_up"] + pairs["bid_price_1_dn"]

    # Signal mask: bid_sum below threshold
    sig = bid_sum < SUM_THRESHOLD

    dr_up = pairs["depth_ratio_up"]
    dr_dn = pairs["depth_ratio_dn"]
    ask_up = pairs["ask_price_1_up"]
    ask_dn = pairs["ask_price_1_dn"]

    # Buy Up: depth ratio advantage + price cap
    buy_up = sig & (dr_up > dr_dn * DEPTH_RATIO_MIN) & (ask_up <= MAX_ENTRY_PRICE) & (ask_up > 0)
    # Buy Down: depth ratio advantage + price cap (only if not already buying Up)
    buy_dn = sig & (dr_dn > dr_up * DEPTH_RATIO_MIN) & ~buy_up & (ask_dn <= MAX_ENTRY_PRICE) & (ask_dn > 0)

    print(f"Buy Up signals: {buy_up.sum():,}, Buy Down signals: {buy_dn.sum():,}")

    # ── Build trade records ──────────────────────────────────────────────
    trade_up = pd.DataFrame({
        "slug": pairs.loc[buy_up, "slug"].values,
        "ts": pairs.loc[buy_up, "exchange_timestamp"].values,
        "side": "Up",
        "entry_price": ask_up[buy_up].values,
        "bid_sum": bid_sum[buy_up].values,
        "depth_ratio": dr_up[buy_up].values,
    })
    trade_dn = pd.DataFrame({
        "slug": pairs.loc[buy_dn, "slug"].values,
        "ts": pairs.loc[buy_dn, "exchange_timestamp"].values,
        "side": "Down",
        "entry_price": ask_dn[buy_dn].values,
        "bid_sum": bid_sum[buy_dn].values,
        "depth_ratio": dr_dn[buy_dn].values,
    })
    del pairs, bid_sum, sig

    trades = pd.concat([trade_up, trade_dn], ignore_index=True)
    del trade_up, trade_dn

    if len(trades) == 0:
        print("No trades generated.")
        return write_failed_results("No trade signals generated")

    # First signal per market
    trades = trades.sort_values("ts")
    trades = trades.groupby("slug").first().reset_index()
    print(f"Trades (first per market): {len(trades)}")

    # ── Evaluate P&L ─────────────────────────────────────────────────────
    trades["winner"] = trades["slug"].map(resolution_cache)
    trades["won"] = trades["side"] == trades["winner"]
    trades["fee"] = calc_fee(trades["entry_price"])
    trades["payout"] = np.where(trades["won"], 1.0, 0.0)
    trades["pnl_per_token"] = trades["payout"] - trades["entry_price"] - trades["fee"]
    trades["pnl"] = trades["pnl_per_token"] * MIN_ORDER_SIZE

    # ── Metrics ──────────────────────────────────────────────────────────
    total_trades = len(trades)
    wins = int(trades["won"].sum())
    win_rate = wins / total_trades
    pnl_total = float(trades["pnl"].sum())
    pnl_per_trade = float(trades["pnl"].mean())
    cumulative = trades["pnl"].cumsum()
    max_drawdown = float((cumulative - cumulative.cummax()).min())
    std = trades["pnl"].std()
    sharpe = float((trades["pnl"].mean() / std) * np.sqrt(288)) if std > 0 else 0.0

    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)
    print(f"Total trades:     {total_trades}")
    print(f"Wins:             {wins} ({win_rate:.1%})")
    print(f"Total P&L:        ${pnl_total:.2f}")
    print(f"P&L per trade:    ${pnl_per_trade:.4f}")
    print(f"Max drawdown:     ${max_drawdown:.2f}")
    print(f"Sharpe ratio:     {sharpe:.2f}")
    print(f"Avg entry price:  {trades['entry_price'].mean():.4f}")
    print(f"Avg bid sum:      {trades['bid_sum'].mean():.4f}")
    print("=" * 60)

    # ── Parameter sensitivity (reload pairs efficiently) ─────────────────
    print("\nParameter sensitivity (reloading minimal data)...")
    # Reload just for sensitivity - minimal columns
    df2 = pd.read_parquet(DATA_PATH, columns=NEEDED_COLS)
    df2 = df2[df2["slug"].isin(resolved_slugs)]
    deep_bid2 = (df2["bid_size_2"].fillna(0) + df2["bid_size_3"].fillna(0) +
                 df2["bid_size_4"].fillna(0) + df2["bid_size_5"].fillna(0))
    df2["depth_ratio"] = deep_bid2 / df2["bid_size_1"].fillna(0).clip(lower=1e-9)

    cols = ["slug", "exchange_timestamp", "bid_price_1", "ask_price_1", "depth_ratio"]
    up2 = df2.loc[df2["token_label"] == "Up", cols]
    dn2 = df2.loc[df2["token_label"] == "Down", cols]
    del df2
    p2 = up2.merge(dn2, on=["slug", "exchange_timestamp"], suffixes=("_up", "_dn"))
    del up2, dn2

    bs2 = p2["bid_price_1_up"] + p2["bid_price_1_dn"]

    print(f"  {'thresh':>7s} {'dr_min':>7s} {'trades':>7s} {'WR':>7s} {'PnL':>10s}")
    for thresh in [0.85, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98]:
        for dr_min in [1.0, 1.2, 1.5, 2.0]:
            s = bs2 < thresh
            bu = s & (p2["depth_ratio_up"] > p2["depth_ratio_dn"] * dr_min) & \
                 (p2["ask_price_1_up"] <= MAX_ENTRY_PRICE) & (p2["ask_price_1_up"] > 0)
            bd = s & (p2["depth_ratio_dn"] > p2["depth_ratio_up"] * dr_min) & ~bu & \
                 (p2["ask_price_1_dn"] <= MAX_ENTRY_PRICE) & (p2["ask_price_1_dn"] > 0)

            t_u = pd.DataFrame({
                "slug": p2.loc[bu, "slug"].values,
                "ts": p2.loc[bu, "exchange_timestamp"].values,
                "side": "Up",
                "entry": p2.loc[bu, "ask_price_1_up"].values,
            })
            t_d = pd.DataFrame({
                "slug": p2.loc[bd, "slug"].values,
                "ts": p2.loc[bd, "exchange_timestamp"].values,
                "side": "Down",
                "entry": p2.loc[bd, "ask_price_1_dn"].values,
            })
            t = pd.concat([t_u, t_d], ignore_index=True)
            if len(t) == 0:
                continue
            t = t.sort_values("ts").groupby("slug").first().reset_index()
            t["won"] = t["side"] == t["slug"].map(resolution_cache)
            t["fee"] = calc_fee(t["entry"])
            t["pnl"] = (np.where(t["won"], 1.0, 0.0) - t["entry"] - t["fee"]) * MIN_ORDER_SIZE
            wr = t["won"].mean()
            pnl = t["pnl"].sum()
            n = len(t)
            print(f"  {thresh:7.2f} {dr_min:7.1f} {n:7d} {wr:6.1%} ${pnl:+10.2f}")

    del p2, bs2

    # ── Also test: no depth ratio filter (just bid_sum signal) ───────────
    print("\n  Baseline: buy random side when bid_sum < threshold (no depth filter):")
    df3 = pd.read_parquet(DATA_PATH, columns=["exchange_timestamp", "slug", "token_label", "bid_price_1", "ask_price_1"])
    df3 = df3[df3["slug"].isin(resolved_slugs)]
    up3 = df3.loc[df3["token_label"] == "Up", ["slug", "exchange_timestamp", "bid_price_1", "ask_price_1"]]
    dn3 = df3.loc[df3["token_label"] == "Down", ["slug", "exchange_timestamp", "bid_price_1", "ask_price_1"]]
    del df3
    p3 = up3.merge(dn3, on=["slug", "exchange_timestamp"], suffixes=("_up", "_dn"))
    del up3, dn3
    bs3 = p3["bid_price_1_up"] + p3["bid_price_1_dn"]

    for thresh in [0.85, 0.90, 0.92, 0.94, 0.96, 0.98]:
        s = bs3 < thresh
        # Always buy Up (to see if bid_sum alone has signal)
        mask = s & (p3["ask_price_1_up"] <= MAX_ENTRY_PRICE) & (p3["ask_price_1_up"] > 0)
        t = pd.DataFrame({
            "slug": p3.loc[mask, "slug"].values,
            "ts": p3.loc[mask, "exchange_timestamp"].values,
            "entry": p3.loc[mask, "ask_price_1_up"].values,
        })
        if len(t) == 0:
            continue
        t = t.sort_values("ts").groupby("slug").first().reset_index()
        t["won"] = t["slug"].map(resolution_cache) == "Up"
        t["fee"] = calc_fee(t["entry"])
        t["pnl"] = (np.where(t["won"], 1.0, 0.0) - t["entry"] - t["fee"]) * MIN_ORDER_SIZE
        wr = t["won"].mean()
        pnl = t["pnl"].sum()
        n = len(t)
        print(f"  always_up thresh={thresh:.2f}: trades={n:4d} WR={wr:.1%} PnL=${pnl:+.2f}")

    # ── Write results ────────────────────────────────────────────────────
    verdict = "promising" if pnl_total > 0 and win_rate > 0.55 else (
        "marginal" if pnl_total > 0 or win_rate > 0.50 else "failed"
    )

    sample_trades = []
    for _, row in trades.head(10).iterrows():
        sample_trades.append({
            "slug": str(row["slug"]),
            "side": str(row["side"]),
            "entry_price": float(row["entry_price"]),
            "bid_sum": float(row["bid_sum"]),
            "depth_ratio": float(row["depth_ratio"]),
            "won": bool(row["won"]),
            "pnl_per_token": float(row["pnl_per_token"]),
        })

    results = {
        "strategy_name": "cross_token_implied_probability_divergence",
        "description": (
            "Compare sum of best-bid on Up and Down tokens. When sum drops "
            "below threshold, buy the token with higher Level 2-5 depth ratio."
        ),
        "hypothesis": (
            "When both sides are cheap (bid_sum < threshold), the token with "
            "deeper relative book support is more likely to win, as informed "
            "participants stack bids at deeper levels before mid-price adjusts."
        ),
        "metrics": {
            "total_trades": total_trades,
            "win_rate": round(win_rate, 4),
            "pnl_total": round(pnl_total, 2),
            "pnl_per_trade": round(pnl_per_trade, 4),
            "max_drawdown": round(max_drawdown, 2),
            "sharpe": round(sharpe, 2)
        },
        "parameters": {
            "sum_threshold": SUM_THRESHOLD,
            "max_entry_price": MAX_ENTRY_PRICE,
            "min_order_size": MIN_ORDER_SIZE,
            "depth_ratio_min": DEPTH_RATIO_MIN,
            "fee_formula": "0.0222 * p * 0.25 * (p*(1-p))^2"
        },
        "sample_trades": sample_trades,
        "verdict": verdict,
        "next_steps": ""
    }

    if verdict == "promising":
        results["next_steps"] = (
            "Test on full dataset. Analyze time-of-day effects. "
            "Optimize entry timing within the 60s observation window."
        )
    elif verdict == "marginal":
        results["next_steps"] = (
            "Investigate tighter mispricing thresholds. Combine with book imbalance. "
            "Test on full dataset for larger sample."
        )
    else:
        results["next_steps"] = (
            "Depth ratio alone may not predict winners in 5m BTC markets. "
            "Consider: (1) depth-weighted price, (2) spread regime conditioning, "
            "(3) adding trade flow momentum, (4) the bid_sum signal itself may "
            "be noise since markets are efficient at the top of book."
        )

    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {OUTPUT_DIR / 'results.json'}")

    trades.to_csv(OUTPUT_DIR / "trades_detail.csv", index=False)
    print(f"Trade details written to {OUTPUT_DIR / 'trades_detail.csv'}")

    return results


def write_failed_results(reason):
    results = {
        "strategy_name": "cross_token_implied_probability_divergence",
        "description": "Cross-token implied probability divergence with depth ratio signal",
        "hypothesis": "Depth ratio advantage predicts winner when both tokens are cheap",
        "metrics": {"total_trades": 0, "win_rate": 0.0, "pnl_total": 0.0,
                    "pnl_per_trade": 0.0, "max_drawdown": 0.0, "sharpe": 0.0},
        "parameters": {"sum_threshold": SUM_THRESHOLD, "max_entry_price": MAX_ENTRY_PRICE,
                       "depth_ratio_min": DEPTH_RATIO_MIN},
        "verdict": "failed",
        "next_steps": f"Strategy failed: {reason}."
    }
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


if __name__ == "__main__":
    run_backtest()
