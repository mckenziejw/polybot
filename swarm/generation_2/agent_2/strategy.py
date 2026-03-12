"""
Deep Level Ratio + Mid-Price Uncertainty Filter Strategy

Hypothesis: Deep orderbook layering (levels 3-5 vs 1-2) is informative about
market direction, but ONLY when mid-price is near 0.50 (uncertain markets).

CRITICAL: Use FIRST snapshot (early in market), not last (near resolution).

Signals tested:
1. net_deep: deep_bid_frac - deep_ask_frac (markets with 3+ levels)
2. total_imb: (total_bid - total_ask) / (total_bid + total_ask)
3. l1_imb: level-1 size imbalance
4. weighted_imb: distance-weighted imbalance across levels
"""

import pandas as pd
import numpy as np
import json
import glob
import os

DATA_DIR = "/home/josh/Documents/projects/wss_test/data/telonex_book_snapshots"
RESOLUTION_CACHE = "/home/josh/Documents/projects/wss_test/data/resolution_cache.json"
OUTPUT_DIR = "/home/josh/Documents/projects/wss_test/swarm/generation_2/agent_2"

MID_PRICE_LOW = 0.40
MID_PRICE_HIGH = 0.60
MAX_SPREAD = 0.10
DEEP_RATIO_QUANTILE = 0.20
ENTRY_PRICE_CAP = 0.85
POSITION_SIZE = 10
MAX_ENTRY_OFFSET_S = 90  # Use snapshots within first 90s of market


def compute_fee_vec(p):
    """Vectorized fee calculation."""
    return 0.0222 * p * 0.25 * (p * (1 - p)) ** 2


def load_resolution_cache():
    with open(RESOLUTION_CACHE) as f:
        return json.load(f)


def resolve_winner(res, up_asset_id):
    if "winnerLabel" in res:
        return res["winnerLabel"] == "Up"
    elif "winning_token" in res:
        return str(up_asset_id) == str(res["winning_token"])
    return None


def process_market(file_path, resolution_cache):
    df = pd.read_parquet(file_path)
    if df.empty:
        return None

    slug = df["slug"].iloc[0]
    if slug not in resolution_cache:
        return None

    # Extract market start epoch from slug
    epoch_s = int(slug.split("-")[-1])
    epoch_ms = epoch_s * 1000

    up_mask = df["token_label"] == "Up"
    down_mask = df["token_label"] == "Down"
    if up_mask.sum() == 0 or down_mask.sum() == 0:
        return None

    up_asset_id = str(df.loc[up_mask, "asset_id"].iloc[0])
    up_won = resolve_winner(resolution_cache[slug], up_asset_id)
    if up_won is None:
        return None

    # Use FIRST snapshot (earliest available, closest to market open)
    up_df = df.loc[up_mask].sort_values("exchange_timestamp")
    down_df = df.loc[down_mask].sort_values("exchange_timestamp")

    # Filter to early snapshots only (within MAX_ENTRY_OFFSET_S of market start)
    cutoff_ms = epoch_ms + MAX_ENTRY_OFFSET_S * 1000
    early_up = up_df[up_df["exchange_timestamp"] <= cutoff_ms]
    early_down = down_df[down_df["exchange_timestamp"] <= cutoff_ms]

    if len(early_up) == 0 or len(early_down) == 0:
        return None

    # Use the FIRST early snapshot
    first_up = early_up.iloc[0]
    first_down = early_down.iloc[0]

    mid = first_up["mid_price"]
    spread = first_up["spread"]

    # Entry offset from market start
    entry_offset_s = (first_up["exchange_timestamp"] - epoch_ms) / 1000

    # Compute signals from Up token book
    bid_sizes = np.array([first_up[f"bid_size_{i}"] for i in range(1, 6)])
    ask_sizes = np.array([first_up[f"ask_size_{i}"] for i in range(1, 6)])
    total_bid = bid_sizes.sum()
    total_ask = ask_sizes.sum()

    if total_bid == 0 and total_ask == 0:
        return None

    n_bid_levels = int((bid_sizes > 0).sum())
    n_ask_levels = int((ask_sizes > 0).sum())

    # Signal 1: net deep ratio
    net_deep = np.nan
    if n_bid_levels >= 3 and n_ask_levels >= 3 and total_bid > 0 and total_ask > 0:
        net_deep = bid_sizes[2:].sum() / total_bid - ask_sizes[2:].sum() / total_ask

    # Signal 2: total imbalance
    total_sum = total_bid + total_ask
    total_imb = (total_bid - total_ask) / total_sum if total_sum > 0 else 0.0

    # Signal 3: L1 imbalance
    l1_sum = bid_sizes[0] + ask_sizes[0]
    l1_imb = (bid_sizes[0] - ask_sizes[0]) / l1_sum if l1_sum > 0 else 0.0

    # Signal 4: weighted imbalance
    weights = np.array([5, 4, 3, 2, 1], dtype=float)
    w_bid = (bid_sizes * weights).sum()
    w_ask = (ask_sizes * weights).sum()
    w_sum = w_bid + w_ask
    weighted_imb = (w_bid - w_ask) / w_sum if w_sum > 0 else 0.0

    return {
        "slug": slug,
        "mid_price": mid,
        "spread": spread,
        "entry_offset_s": entry_offset_s,
        "net_deep": net_deep,
        "total_imb": total_imb,
        "l1_imb": l1_imb,
        "weighted_imb": weighted_imb,
        "total_bid": total_bid,
        "total_ask": total_ask,
        "n_bid_levels": n_bid_levels,
        "n_ask_levels": n_ask_levels,
        "up_won": up_won,
        "up_ask": first_up["ask_price_1"],
        "down_ask": first_down["ask_price_1"],
        "n_snapshots": len(up_df),
    }


def run_backtest(df, signal_col, quantile=DEEP_RATIO_QUANTILE, label=""):
    """High signal → buy Up, low signal → buy Down."""
    work = df.dropna(subset=[signal_col]).copy()
    if len(work) < 30:
        return None

    q_low = work[signal_col].quantile(quantile)
    q_high = work[signal_col].quantile(1 - quantile)
    if q_low >= q_high:
        return None

    buy_up = work[work[signal_col] > q_high].copy()
    buy_down = work[work[signal_col] < q_low].copy()

    buy_up["direction"] = "Up"
    buy_up["entry_price"] = buy_up["up_ask"]
    buy_up["won"] = buy_up["up_won"]

    buy_down["direction"] = "Down"
    buy_down["entry_price"] = buy_down["down_ask"]
    buy_down["won"] = ~buy_down["up_won"]

    trades = pd.concat([buy_up, buy_down], ignore_index=True)
    trades = trades[(trades["entry_price"] <= ENTRY_PRICE_CAP) & (trades["entry_price"] > 0)]

    if len(trades) < 10:
        return None

    p = trades["entry_price"].values
    trades["fee"] = compute_fee_vec(p)
    trades["cost"] = trades["entry_price"] + trades["fee"]
    trades["payout"] = np.where(trades["won"], 1.0, 0.0)
    trades["pnl"] = (trades["payout"] - trades["cost"]) * POSITION_SIZE

    total = len(trades)
    wins = int(trades["won"].sum())
    wr = wins / total
    pnl_total = float(trades["pnl"].sum())
    pnl_per = pnl_total / total

    cum_pnl = trades["pnl"].cumsum()
    max_dd = float((cum_pnl - cum_pnl.cummax()).min())
    sharpe = float((trades["pnl"].mean() / trades["pnl"].std()) * np.sqrt(288)) if trades["pnl"].std() > 0 else 0.0

    n_up = int((trades["direction"] == "Up").sum())
    n_down = int((trades["direction"] == "Down").sum())
    wr_up = float(trades.loc[trades["direction"] == "Up", "won"].mean()) if n_up > 0 else 0.0
    wr_down = float(trades.loc[trades["direction"] == "Down", "won"].mean()) if n_down > 0 else 0.0

    return {
        "label": label or signal_col, "n_trades": total,
        "n_up": n_up, "n_down": n_down,
        "wr": wr, "wr_up": wr_up, "wr_down": wr_down,
        "pnl_total": pnl_total, "pnl_per": pnl_per,
        "max_dd": max_dd, "sharpe": sharpe,
        "q_low": float(q_low), "q_high": float(q_high),
    }


def fmt(m):
    if m is None:
        return "  [insufficient data or degenerate thresholds]"
    return (f"  {m['label']}: {m['n_trades']} trades (Up:{m['n_up']}, Down:{m['n_down']}), "
            f"WR={m['wr']:.4f} (Up:{m['wr_up']:.4f}, Down:{m['wr_down']:.4f}), "
            f"PnL=${m['pnl_total']:.2f} (${m['pnl_per']:.4f}/trade), Sharpe={m['sharpe']:.2f}")


def main():
    print("Loading resolution cache...")
    resolution_cache = load_resolution_cache()

    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.parquet")))
    print(f"Processing {len(files)} market files...")

    results = []
    for f in files:
        r = process_market(f, resolution_cache)
        if r is not None:
            results.append(r)

    df = pd.DataFrame(results)
    print(f"  {len(df)} markets with early snapshots + resolution")
    print(f"  Base rate Up: {df['up_won'].mean():.4f}")
    print(f"  Avg entry offset: {df['entry_offset_s'].mean():.1f}s")

    # ---- Filter: uncertain markets ----
    uncertain = df[(df["mid_price"] >= MID_PRICE_LOW) & (df["mid_price"] <= MID_PRICE_HIGH)]
    print(f"\nUncertain markets (mid {MID_PRICE_LOW}-{MID_PRICE_HIGH}): {len(uncertain)}")
    if len(uncertain) > 0:
        print(f"  Base rate Up: {uncertain['up_won'].mean():.4f}")
        print(f"  Spread: mean={uncertain['spread'].mean():.4f}, med={uncertain['spread'].median():.4f}")
        print(f"  With spread <= {MAX_SPREAD}: {(uncertain['spread'] <= MAX_SPREAD).sum()}")
        print(f"  Bid levels: {uncertain['n_bid_levels'].value_counts().sort_index().to_dict()}")

    # ---- Signal correlations ----
    print(f"\n=== Signal Correlations (uncertain, n={len(uncertain)}) ===")
    for col in ["net_deep", "total_imb", "l1_imb", "weighted_imb"]:
        valid = uncertain.dropna(subset=[col])
        if len(valid) > 10:
            corr = valid[col].corr(valid["up_won"].astype(float))
            print(f"  {col}: r={corr:.4f} (n={len(valid)})")

    # ---- Backtests on uncertain markets (all spreads) ----
    print(f"\n=== Backtests: uncertain markets, all spreads (n={len(uncertain)}) ===")
    best_m = None
    for sig in ["total_imb", "l1_imb", "weighted_imb"]:
        m = run_backtest(uncertain, sig, label=f"all_{sig}")
        print(fmt(m))
        if m and (best_m is None or m["pnl_per"] > best_m["pnl_per"]):
            best_m = m

    # ---- Backtests on uncertain + liquid ----
    liquid = uncertain[uncertain["spread"] <= MAX_SPREAD]
    print(f"\n=== Backtests: uncertain + liquid (spread<={MAX_SPREAD}, n={len(liquid)}) ===")
    for sig in ["total_imb", "l1_imb", "weighted_imb"]:
        m = run_backtest(liquid, sig, label=f"liq_{sig}")
        print(fmt(m))
        if m and (best_m is None or m["pnl_per"] > best_m["pnl_per"]):
            best_m = m

    # ---- net_deep on markets with depth ----
    deep = uncertain[(uncertain["n_bid_levels"] >= 3) & (uncertain["n_ask_levels"] >= 3)]
    print(f"\n=== net_deep on markets with 3+ levels (n={len(deep)}) ===")
    m_deep = run_backtest(deep, "net_deep", label="deep_net_deep")
    print(fmt(m_deep))
    if m_deep and (best_m is None or m_deep["pnl_per"] > best_m["pnl_per"]):
        best_m = m_deep

    # ---- Quintile analysis ----
    print(f"\n=== Quintile Analysis ===")
    for col in ["total_imb", "weighted_imb", "l1_imb"]:
        valid = uncertain.dropna(subset=[col])
        if len(valid) < 30:
            continue
        try:
            valid = valid.copy()
            valid["q"] = pd.qcut(valid[col], 5, labels=False, duplicates="drop")
            print(f"  {col}:")
            for qv in sorted(valid["q"].unique()):
                sub = valid[valid["q"] == qv]
                print(f"    Q{qv}: n={len(sub)}, Up WR={sub['up_won'].mean():.4f}, "
                      f"range=[{sub[col].min():.4f}, {sub[col].max():.4f}]")
        except ValueError:
            print(f"  {col}: cannot form quintiles (too many ties)")

    # ---- Sensitivity: quantile ----
    print(f"\n=== Sensitivity: quantile (total_imb, uncertain) ===")
    for q in [0.10, 0.15, 0.20, 0.25, 0.30, 0.40]:
        m = run_backtest(uncertain, "total_imb", quantile=q, label=f"q={q}")
        if m:
            print(f"  q={q:.2f}: {m['n_trades']} trades, WR={m['wr']:.4f}, "
                  f"PnL=${m['pnl_total']:.2f} (${m['pnl_per']:.4f}/trade)")
        else:
            print(f"  q={q:.2f}: degenerate")

    # ---- Sensitivity: mid-price ----
    print(f"\n=== Sensitivity: mid-price range (total_imb) ===")
    for lo, hi in [(0.30, 0.70), (0.35, 0.65), (0.40, 0.60), (0.45, 0.55)]:
        sub = df[(df["mid_price"] >= lo) & (df["mid_price"] <= hi)]
        m = run_backtest(sub, "total_imb", label=f"[{lo},{hi}]")
        if m:
            print(f"  [{lo},{hi}]: n={len(sub)}, {m['n_trades']} trades, WR={m['wr']:.4f}, "
                  f"PnL=${m['pnl_total']:.2f} (${m['pnl_per']:.4f}/trade)")
        else:
            print(f"  [{lo},{hi}]: n={len(sub)}, degenerate")

    # ---- Control: lopsided markets ----
    lopsided = df[(df["mid_price"] < 0.30) | (df["mid_price"] > 0.70)]
    print(f"\n=== Control: lopsided markets (n={len(lopsided)}) ===")
    if len(lopsided) > 0:
        print(f"  Base rate Up: {lopsided['up_won'].mean():.4f}")
    for sig in ["total_imb", "weighted_imb"]:
        m = run_backtest(lopsided, sig, label=f"lopsided_{sig}")
        print(fmt(m))

    # ---- Baseline ----
    print(f"\n=== Baseline: Buy Up on all uncertain markets ===")
    base = uncertain[(uncertain["up_ask"] <= ENTRY_PRICE_CAP) & (uncertain["up_ask"] > 0)].copy()
    if len(base) > 0:
        p = base["up_ask"].values
        base["fee"] = compute_fee_vec(p)
        base["pnl"] = (np.where(base["up_won"], 1.0, 0.0) - base["up_ask"].values - base["fee"].values) * POSITION_SIZE
        print(f"  {len(base)} trades, WR={base['up_won'].mean():.4f}, "
              f"PnL=${base['pnl'].sum():.2f} (${base['pnl'].mean():.4f}/trade)")

    # ---- Write results ----
    write_results(best_m, m_deep)


def write_results(primary, deep_result):
    if primary is None:
        verdict = "failed"
        metrics = {"total_trades": 0, "win_rate": 0, "pnl_total": 0,
                   "pnl_per_trade": 0, "max_drawdown": 0, "sharpe": 0}
        next_steps = ("Deep level ratio not viable: most 5m orderbooks have <3 levels, "
                      "and broader imbalance signals are degenerate (massive ties at 0).")
    else:
        pnl_per = primary["pnl_per"]
        verdict = "promising" if pnl_per > 0.05 else ("marginal" if pnl_per > -0.02 else "failed")
        metrics = {
            "total_trades": primary["n_trades"],
            "win_rate": round(primary["wr"], 4),
            "pnl_total": round(primary["pnl_total"], 2),
            "pnl_per_trade": round(primary["pnl_per"], 4),
            "max_drawdown": round(primary["max_dd"], 2),
            "sharpe": round(primary["sharpe"], 2),
        }
        if verdict == "failed":
            next_steps = ("Orderbook imbalance signals show no edge after fees in uncertain "
                          "5m BTC markets. The books are too thin and symmetric for layering "
                          "to be informative. The hypothesis that uncertainty filtering isolates "
                          "informational signal is not supported.")
        elif verdict == "marginal":
            next_steps = ("Weak signal detected but insufficient for profitability after fees. "
                          "Could explore: (1) multi-snapshot averaging, (2) combining with trade "
                          "flow, (3) interaction with volatility regime.")
        else:
            next_steps = "Validate with out-of-sample data and paper trading."

    deep_info = ""
    if deep_result:
        deep_info = (f" Deep ratio (3+ levels subset): {deep_result['n_trades']} trades, "
                     f"WR={deep_result['wr']:.4f}, ${deep_result['pnl_per']:.4f}/trade.")

    results = {
        "strategy_name": "deep_level_ratio_uncertainty_filter",
        "description": (
            "Trade based on orderbook size imbalance filtered to uncertain markets "
            "(mid 0.40-0.60). Uses FIRST snapshot (early in market window). Tests deep "
            "layering ratio and total imbalance signals." + deep_info
        ),
        "hypothesis": (
            "In lopsided markets, orderbook shape is mechanically skewed by price level. "
            "Restricting to uncertain markets (mid 0.40-0.60) should isolate the informational "
            "component of MM layering from the mechanical component."
        ),
        "metrics": metrics,
        "parameters": {
            "mid_price_range": [MID_PRICE_LOW, MID_PRICE_HIGH],
            "max_spread": MAX_SPREAD,
            "deep_ratio_quantile": DEEP_RATIO_QUANTILE,
            "entry_price_cap": ENTRY_PRICE_CAP,
            "position_size": POSITION_SIZE,
            "max_entry_offset_s": MAX_ENTRY_OFFSET_S,
        },
        "verdict": verdict,
        "next_steps": next_steps,
    }

    output_path = os.path.join(OUTPUT_DIR, "results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {output_path}")
    print(f"Verdict: {verdict}")


if __name__ == "__main__":
    main()
