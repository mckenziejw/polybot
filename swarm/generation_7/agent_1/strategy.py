"""
Strategy: Deep Level Ratio + Time-of-Day + Mid-Price Filter

Hypothesis: Combine deep_level_ratio (L3-L5 bid vs ask concentration) with a
time-of-day filter (first 2h UTC), mid-price band 0.25-0.75, and 15% quintile
threshold to capture extreme orderbook imbalances.
"""

import json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path

BASE = Path("/home/ubuntu/wss_test")
PARQUET = BASE / "data/sample_1k/book_snapshots_sample_1k.parquet"
RESOLUTION = BASE / "data/sample_1k/resolution_cache_sample.json"
OUTPUT_DIR = BASE / "swarm/generation_7/agent_1"

def polymarket_fee(p: np.ndarray) -> np.ndarray:
    return 0.0222 * p * 0.25 * (p * (1 - p)) ** 2

# ── Load resolution ──
with open(RESOLUTION) as f:
    resolution = json.load(f)
slug_winner = {s: info.get("winnerLabel", "") for s, info in resolution.items()}

# ── Process parquet in row-group chunks to avoid OOM ──
print("Processing parquet in chunks...")
LOAD_COLS = ["slug", "token_label", "mid_price",
             "bid_size_3", "bid_size_4", "bid_size_5",
             "ask_size_3", "ask_size_4", "ask_size_5"]

pf = pq.ParquetFile(PARQUET)
chunk_aggs = []

for i in range(pf.metadata.num_row_groups):
    table = pf.read_row_group(i, columns=LOAD_COLS)
    chunk = table.to_pandas()

    # Compute deep_level_ratio per row
    deep_bid = chunk["bid_size_3"] + chunk["bid_size_4"] + chunk["bid_size_5"]
    deep_ask = chunk["ask_size_3"] + chunk["ask_size_4"] + chunk["ask_size_5"]
    deep_total = deep_bid + deep_ask
    chunk["dlr"] = np.where(deep_total > 0, deep_bid / deep_total, 0.5)

    # Aggregate per slug+token_label within this chunk
    agg = chunk.groupby(["slug", "token_label"]).agg(
        dlr_sum=("dlr", "sum"),
        dlr_count=("dlr", "count"),
        mid_sum=("mid_price", "sum"),
        mid_count=("mid_price", "count"),
    ).reset_index()
    chunk_aggs.append(agg)

    if (i + 1) % 10 == 0:
        print(f"  Processed {i+1}/{pf.metadata.num_row_groups} row groups")

print(f"  Processed all {pf.metadata.num_row_groups} row groups")

# ── Combine chunk aggregates ──
all_aggs = pd.concat(chunk_aggs, ignore_index=True)
final = all_aggs.groupby(["slug", "token_label"]).agg(
    dlr_sum=("dlr_sum", "sum"),
    dlr_count=("dlr_count", "sum"),
    mid_sum=("mid_sum", "sum"),
    mid_count=("mid_count", "sum"),
).reset_index()

# Compute means (using sum/count since we can't easily do median across chunks)
final["dlr_mean"] = final["dlr_sum"] / final["dlr_count"]
final["mid_mean"] = final["mid_sum"] / final["mid_count"]

print(f"Aggregated to {len(final)} slug/token rows, {final['slug'].nunique()} markets")

# ── Extract hour from slug ──
final["epoch"] = final["slug"].str.split("-").str[-1].astype(np.int64)
final["hour_utc"] = (final["epoch"] % 86400) // 3600

# ── Pivot to get Up and Down side by side ──
up = final[final["token_label"] == "Up"][["slug", "dlr_mean", "mid_mean", "hour_utc", "epoch"]].copy()
down = final[final["token_label"] == "Down"][["slug", "dlr_mean", "mid_mean"]].copy()

merged = up.merge(down, on="slug", suffixes=("_up", "_down"))
merged["winner"] = merged["slug"].map(slug_winner)
merged = merged.dropna(subset=["winner"])
merged = merged[merged["winner"] != ""]

print(f"Markets with resolution: {len(merged)}")

# ── Hour distribution ──
print("\nHour distribution:")
print(merged.groupby("hour_utc").size().sort_index().to_string())

# ── Strategy runner ──
def run_strategy(data, name, use_time_filter=True, quantile_pct=0.15):
    print(f"\n{'='*60}")
    print(f"Strategy: {name}")
    print(f"{'='*60}")

    filtered = data[(data["mid_mean_up"] >= 0.25) & (data["mid_mean_up"] <= 0.75)].copy()

    if use_time_filter:
        filtered = filtered[filtered["hour_utc"] < 2].copy()

    if len(filtered) < 10:
        print(f"  Only {len(filtered)} markets — insufficient")
        return None

    print(f"  Markets after filters: {len(filtered)}")

    q_low = filtered["dlr_mean_up"].quantile(quantile_pct)
    q_high = filtered["dlr_mean_up"].quantile(1 - quantile_pct)
    print(f"  DLR quantiles ({quantile_pct:.0%}): low={q_low:.4f}, high={q_high:.4f}")

    trades = []

    # HIGH dlr → buy Up
    high = filtered[filtered["dlr_mean_up"] >= q_high].copy()
    high["side"] = "Up"
    high["entry_price"] = high["mid_mean_up"]
    trades.append(high)

    # LOW dlr → buy Down
    low = filtered[filtered["dlr_mean_up"] <= q_low].copy()
    low["side"] = "Down"
    low["entry_price"] = low["mid_mean_down"]
    trades.append(low)

    all_trades = pd.concat(trades, ignore_index=True)
    all_trades = all_trades[all_trades["entry_price"] <= 0.85].copy()

    if len(all_trades) == 0:
        print("  No trades after price cap")
        return None

    all_trades["fee"] = polymarket_fee(all_trades["entry_price"].values)
    all_trades["cost"] = all_trades["entry_price"] + all_trades["fee"]
    all_trades["won"] = (all_trades["winner"] == all_trades["side"]).astype(int)
    all_trades["pnl"] = all_trades["won"] * 1.0 - all_trades["cost"]

    n = len(all_trades)
    wr = all_trades["won"].mean()
    total_pnl = all_trades["pnl"].sum()
    pnl_pt = total_pnl / n
    cum = all_trades["pnl"].cumsum()
    max_dd = (cum.cummax() - cum).max()
    std = all_trades["pnl"].std()
    sharpe = pnl_pt / std if std > 0 else 0.0
    sharpe_ann = sharpe * np.sqrt(288 * 365) if std > 0 else 0.0

    print(f"  Trades: {n}, WR: {wr:.1%}, PnL: {total_pnl:.4f}, "
          f"PnL/t: {pnl_pt:.4f}, MaxDD: {max_dd:.4f}, Sharpe(ann): {sharpe_ann:.2f}")

    for side in ["Up", "Down"]:
        s = all_trades[all_trades["side"] == side]
        if len(s) > 0:
            print(f"    {side}: {len(s)} trades, WR={s['won'].mean():.1%}, PnL={s['pnl'].sum():.4f}")

    # Baseline
    base_won = (filtered["winner"] == "Up").mean()
    base_fee = polymarket_fee(filtered["mid_mean_up"].values).mean()
    base_pnl = base_won - filtered["mid_mean_up"].mean() - base_fee
    print(f"  Baseline (buy Up): WR={base_won:.1%}, PnL/t≈{base_pnl:.4f}")

    return {"n_trades": n, "win_rate": wr, "total_pnl": total_pnl,
            "pnl_per_trade": pnl_pt, "max_drawdown": max_dd,
            "sharpe": sharpe, "sharpe_annual": sharpe_ann}

# ── Run variants ──
r1 = run_strategy(merged, "Full (2h UTC + mid 0.25-0.75 + Q15%)", True, 0.15)
r2 = run_strategy(merged, "No time filter (mid 0.25-0.75 + Q15%)", False, 0.15)
r3 = run_strategy(merged, "No time filter (mid 0.25-0.75 + Q20%)", False, 0.20)

# Also try broader time windows
r4 = run_strategy(merged[merged["hour_utc"] < 4], "First 4h UTC + Q15%", False, 0.15)
r5 = run_strategy(merged[merged["hour_utc"] < 6], "First 6h UTC + Q15%", False, 0.15)

# ── Pick best ──
variants = {"full_2h_q15": r1, "no_time_q15": r2, "no_time_q20": r3,
            "first_4h_q15": r4, "first_6h_q15": r5}
best_name, best = "none", None
for k, v in variants.items():
    if v is not None and (best is None or v["pnl_per_trade"] > best["pnl_per_trade"]):
        best_name, best = k, v

if best is None:
    verdict = "failed"
    metrics = {"total_trades": 0, "win_rate": 0.0, "pnl_total": 0.0,
               "pnl_per_trade": 0.0, "max_drawdown": 0.0, "sharpe": 0.0}
    desc = "Insufficient data or no profitable variant found."
else:
    if best["pnl_per_trade"] > 0.01 and best["n_trades"] >= 20:
        verdict = "promising"
    elif best["pnl_per_trade"] > 0 and best["n_trades"] >= 10:
        verdict = "marginal"
    else:
        verdict = "failed"
    metrics = {
        "total_trades": best["n_trades"],
        "win_rate": round(best["win_rate"], 4),
        "pnl_total": round(best["total_pnl"], 4),
        "pnl_per_trade": round(best["pnl_per_trade"], 4),
        "max_drawdown": round(best["max_drawdown"], 4),
        "sharpe": round(best["sharpe_annual"], 4),
    }
    desc = f"Deep level ratio (L3-L5) with filters. Best variant: {best_name}."

results = {
    "strategy_name": "deep_level_ratio_time_filtered",
    "description": desc,
    "hypothesis": (
        "Combine deep_level_ratio (L3-L5 bid vs ask concentration) with time-of-day "
        "filter (first 2h UTC), mid-price band 0.25-0.75, and 15% quintile threshold "
        "to capture extreme orderbook imbalances in BTC 5m binary markets."
    ),
    "metrics": metrics,
    "parameters": {
        "deep_levels": "L3-L5",
        "mid_price_range": [0.25, 0.75],
        "time_filter_hours": 2,
        "quantile_threshold": 0.15,
        "entry_price_cap": 0.85,
    },
    "verdict": verdict,
    "next_steps": (
        "If promising on sample: validate on full 7,187-market dataset. "
        "If failed: deep level ratio in early book snapshots (first 60s) likely "
        "lacks predictive power — consider snapshots closer to expiry or combining "
        "with trade flow signals."
    ),
    "all_variants": {k: v for k, v in variants.items()},
}

with open(OUTPUT_DIR / "results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nVERDICT: {verdict} (best variant: {best_name})")
print(f"Results written to {OUTPUT_DIR / 'results.json'}")
