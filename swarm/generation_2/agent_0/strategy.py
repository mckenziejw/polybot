"""
Deep Level Ratio Strategy — High-Spread Filter

Hypothesis: In wide-spread markets (spread >= 4c), deep orderbook levels (3-5)
reveal MM directional layering. Top/bottom 20% quintiles of deep_ratio
(deep_bid_size / deep_ask_size) for the Up token predict resolution.

- Top quintile deep_ratio → MMs layering deep bids on Up → buy Up
- Bottom quintile deep_ratio → MMs layering deep asks on Up → buy Down
"""

import os
import json
import glob
import numpy as np
import pandas as pd
from pathlib import Path

BASE = "/home/josh/Documents/projects/wss_test"
BOOK_DIR = os.path.join(BASE, "data/telonex_book_snapshots")
RESOLUTION_CACHE = os.path.join(BASE, "data/resolution_cache.json")
OUTPUT_DIR = os.path.join(BASE, "swarm/generation_2/agent_0")

# Parameters
SPREAD_MIN = 0.04          # only trade wide-spread markets
QUINTILE_PCT = 0.20        # top/bottom 20%
ENTRY_PRICE_CAP = 0.85     # never buy above this
MIN_ORDER_SIZE = 5          # minimum tokens


def polymarket_fee(p: np.ndarray) -> np.ndarray:
    """Vectorized fee calculation."""
    return 0.0222 * p * 0.25 * (p * (1 - p)) ** 2


def load_resolution_cache():
    """Load resolution cache and build slug -> winning_token_id map."""
    with open(RESOLUTION_CACHE) as f:
        rc = json.load(f)
    return rc


def process_snapshots():
    """Process all book snapshot files, compute deep_ratio per market."""
    files = sorted(glob.glob(os.path.join(BOOK_DIR, "*.parquet")))
    print(f"Found {len(files)} snapshot files")

    rc = load_resolution_cache()

    all_signals = []

    for i, fpath in enumerate(files):
        slug = Path(fpath).stem

        # Skip if no resolution data
        if slug not in rc:
            continue

        df = pd.read_parquet(fpath)

        if df.empty:
            continue

        # Get only "Up" token rows (we'll compute deep_ratio on Up token)
        up_rows = df[df["token_label"] == "Up"].copy()
        down_rows = df[df["token_label"] == "Down"].copy()

        if up_rows.empty or down_rows.empty:
            continue

        # Use the LAST snapshot (closest to market close) for signal
        up_last = up_rows.iloc[-1]
        down_last = down_rows.iloc[-1]

        spread_up = up_last["spread"]
        spread_down = down_last["spread"]

        # Use the Up token's spread as the market spread indicator
        spread = spread_up

        # Filter: only wide-spread markets
        if spread < SPREAD_MIN:
            continue

        # Compute deep level sizes (levels 3-5) for Up token
        deep_bid = up_last["bid_size_3"] + up_last["bid_size_4"] + up_last["bid_size_5"]
        deep_ask = up_last["ask_size_3"] + up_last["ask_size_4"] + up_last["ask_size_5"]

        # Avoid division by zero
        if deep_bid + deep_ask == 0:
            continue

        deep_ratio = deep_bid / (deep_bid + deep_ask)  # normalized 0-1

        # Entry prices
        up_ask = up_last["ask_price_1"]
        down_ask = down_last["ask_price_1"]

        # Determine winner — resolution cache has 3 formats
        res = rc[slug]
        up_asset = str(up_last["asset_id"])
        down_asset = str(down_last["asset_id"])

        if "winnerLabel" in res:
            # Format 2: direct label
            winner = res["winnerLabel"]
        elif "outcomes" in res and "clobTokenIds" in res:
            # Format 3: outcomes list + clobTokenIds list, winningTokenId
            wt = res["winningTokenId"]
            if wt == up_asset:
                winner = "Up"
            elif wt == down_asset:
                winner = "Down"
            else:
                # Try matching by position in clobTokenIds
                try:
                    idx = res["clobTokenIds"].index(wt)
                    winner = res["outcomes"][idx]
                except (ValueError, IndexError):
                    continue
        elif "token_outcomes" in res:
            # Format 1: token_outcomes dict
            token_outcomes = res["token_outcomes"]
            if up_asset in token_outcomes:
                winner = "Up" if token_outcomes[up_asset] == 1.0 else "Down"
            elif down_asset in token_outcomes:
                winner = "Down" if token_outcomes[down_asset] == 1.0 else "Up"
            else:
                continue
        else:
            continue

        all_signals.append({
            "slug": slug,
            "spread": spread,
            "deep_ratio": deep_ratio,
            "deep_bid": deep_bid,
            "deep_ask": deep_ask,
            "up_ask": up_ask,
            "down_ask": down_ask,
            "winner": winner,
            "mid_price_up": up_last["mid_price"],
        })

        if (i + 1) % 1000 == 0:
            print(f"  Processed {i+1}/{len(files)} files, {len(all_signals)} signals so far")

    print(f"Total signals (spread >= {SPREAD_MIN}): {len(all_signals)}")
    return pd.DataFrame(all_signals)


def backtest(signals: pd.DataFrame):
    """Run quintile-based backtest on deep_ratio signals."""
    if signals.empty:
        print("No signals to backtest")
        return empty_results()

    n = len(signals)
    print(f"\n=== Backtest: {n} high-spread markets ===")
    print(f"Spread stats: mean={signals['spread'].mean():.4f}, "
          f"median={signals['spread'].median():.4f}")
    print(f"Deep ratio stats: mean={signals['deep_ratio'].mean():.4f}, "
          f"std={signals['deep_ratio'].std():.4f}")

    # Quintile thresholds
    q_low = signals["deep_ratio"].quantile(QUINTILE_PCT)
    q_high = signals["deep_ratio"].quantile(1 - QUINTILE_PCT)
    print(f"Quintile thresholds: low={q_low:.4f}, high={q_high:.4f}")

    # Top quintile: high deep_ratio → buy Up
    top = signals[signals["deep_ratio"] >= q_high].copy()
    # Bottom quintile: low deep_ratio → buy Down
    bottom = signals[signals["deep_ratio"] <= q_low].copy()

    print(f"Top quintile (buy Up): {len(top)} markets")
    print(f"Bottom quintile (buy Down): {len(bottom)} markets")

    # --- Top quintile: buy Up ---
    top["side"] = "Up"
    top["entry_price"] = top["up_ask"]
    top["win"] = (top["winner"] == "Up").astype(int)

    # --- Bottom quintile: buy Down ---
    bottom["side"] = "Down"
    bottom["entry_price"] = bottom["down_ask"]
    bottom["win"] = (bottom["winner"] == "Down").astype(int)

    trades = pd.concat([top, bottom], ignore_index=True)

    # Apply entry price cap
    trades = trades[trades["entry_price"] <= ENTRY_PRICE_CAP].copy()
    trades = trades[trades["entry_price"] > 0].copy()
    print(f"After entry price cap ({ENTRY_PRICE_CAP}): {len(trades)} trades")

    if trades.empty:
        print("No trades after filtering")
        return empty_results()

    # Calculate PnL per trade (buying 1 share)
    # Fee on entry
    trades["fee"] = polymarket_fee(trades["entry_price"].values)
    # PnL: win pays 1.00, lose pays 0.00
    trades["pnl"] = trades["win"] * (1.0 - trades["entry_price"]) - \
                    (1 - trades["win"]) * trades["entry_price"] - \
                    trades["fee"]

    # Position size: 5 tokens minimum
    trades["pnl_sized"] = trades["pnl"] * MIN_ORDER_SIZE

    total_trades = len(trades)
    wins = trades["win"].sum()
    win_rate = wins / total_trades
    pnl_total = trades["pnl_sized"].sum()
    pnl_per_trade = trades["pnl_sized"].mean()

    # Max drawdown
    cumulative = trades["pnl_sized"].cumsum()
    running_max = cumulative.cummax()
    drawdown = cumulative - running_max
    max_drawdown = drawdown.min()

    # Sharpe (annualized, assuming ~288 5m markets per day)
    daily_pnl = trades["pnl_sized"].values
    sharpe = (daily_pnl.mean() / daily_pnl.std() * np.sqrt(288)) if daily_pnl.std() > 0 else 0.0

    print(f"\n=== RESULTS ===")
    print(f"Total trades:   {total_trades}")
    print(f"Wins:           {wins} ({win_rate:.1%})")
    print(f"PnL total:      ${pnl_total:.2f}")
    print(f"PnL/trade:      ${pnl_per_trade:.4f}")
    print(f"Max drawdown:   ${max_drawdown:.2f}")
    print(f"Sharpe:         {sharpe:.2f}")

    # Breakdown by side
    for side in ["Up", "Down"]:
        s = trades[trades["side"] == side]
        if len(s) > 0:
            print(f"\n  {side}: {len(s)} trades, WR={s['win'].mean():.1%}, "
                  f"PnL=${s['pnl_sized'].sum():.2f}, avg=${s['pnl_sized'].mean():.4f}")

    # Entry price distribution
    print(f"\nEntry price stats:")
    print(f"  Mean: {trades['entry_price'].mean():.4f}")
    print(f"  Median: {trades['entry_price'].median():.4f}")

    # Compare to baseline (all high-spread markets, no signal)
    print(f"\n=== BASELINE (all high-spread, buy Up at ask) ===")
    baseline = signals.copy()
    baseline = baseline[baseline["up_ask"] <= ENTRY_PRICE_CAP].copy()
    baseline = baseline[baseline["up_ask"] > 0].copy()
    if len(baseline) > 0:
        baseline["fee"] = polymarket_fee(baseline["up_ask"].values)
        baseline["win"] = (baseline["winner"] == "Up").astype(int)
        baseline["pnl"] = baseline["win"] * (1.0 - baseline["up_ask"]) - \
                          (1 - baseline["win"]) * baseline["up_ask"] - \
                          baseline["fee"]
        print(f"  N={len(baseline)}, WR={baseline['win'].mean():.1%}, "
              f"PnL/trade=${baseline['pnl'].mean():.4f}")

    # Verdict
    if pnl_per_trade > 0.02 and total_trades >= 50:
        verdict = "promising"
    elif pnl_per_trade > 0 and total_trades >= 30:
        verdict = "marginal"
    else:
        verdict = "failed"

    return {
        "strategy_name": "deep_level_ratio_high_spread",
        "description": "Deep orderbook level ratio (levels 3-5 bid/ask) filtered to high-spread markets (>=4c). Top/bottom 20% quintiles predict resolution direction.",
        "hypothesis": "In wide-spread markets, MMs have more room to express directional views via deep level layering. Deep bid-heavy → buy Up, deep ask-heavy → buy Down.",
        "metrics": {
            "total_trades": int(total_trades),
            "win_rate": round(float(win_rate), 4),
            "pnl_total": round(float(pnl_total), 2),
            "pnl_per_trade": round(float(pnl_per_trade), 4),
            "max_drawdown": round(float(max_drawdown), 2),
            "sharpe": round(float(sharpe), 2),
        },
        "parameters": {
            "spread_min": SPREAD_MIN,
            "quintile_pct": QUINTILE_PCT,
            "entry_price_cap": ENTRY_PRICE_CAP,
            "min_order_size": MIN_ORDER_SIZE,
        },
        "verdict": verdict,
        "next_steps": "",
    }


def empty_results():
    return {
        "strategy_name": "deep_level_ratio_high_spread",
        "description": "Deep orderbook level ratio filtered to high-spread markets.",
        "hypothesis": "In wide-spread markets, deep level layering reveals MM directional views.",
        "metrics": {
            "total_trades": 0,
            "win_rate": 0.0,
            "pnl_total": 0.0,
            "pnl_per_trade": 0.0,
            "max_drawdown": 0.0,
            "sharpe": 0.0,
        },
        "parameters": {
            "spread_min": SPREAD_MIN,
            "quintile_pct": QUINTILE_PCT,
            "entry_price_cap": ENTRY_PRICE_CAP,
            "min_order_size": MIN_ORDER_SIZE,
        },
        "verdict": "failed",
        "next_steps": "Insufficient data or no signal found.",
    }


if __name__ == "__main__":
    signals = process_snapshots()

    if not signals.empty:
        # Save signals for analysis
        signals.to_csv(os.path.join(OUTPUT_DIR, "signals_debug.csv"), index=False)

    results = backtest(signals)

    # Fill in next_steps based on verdict
    if results["verdict"] == "failed":
        results["next_steps"] = (
            "Deep level ratio does not predict direction even in high-spread markets. "
            "The signal may be too noisy or MMs don't differentiate layering by spread regime. "
            "Could try: time-weighted deep ratio, level-size momentum, or cross-token ratio."
        )
    elif results["verdict"] == "marginal":
        results["next_steps"] = (
            "Slight edge detected but likely not robust. "
            "Try tighter quintile filters (10%), time-decay weighting, or combining with book_imbalance."
        )
    else:
        results["next_steps"] = (
            "Signal appears promising. Validate with walk-forward, check for time-clustering, "
            "and estimate realistic fill rates at these spread levels."
        )

    with open(os.path.join(OUTPUT_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nVerdict: {results['verdict']}")
    print(f"Results saved to {os.path.join(OUTPUT_DIR, 'results.json')}")
