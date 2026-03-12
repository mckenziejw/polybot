"""
Strategy: Deep Level Ratio — Down-Side Only (v3, no look-ahead bias)

HYPOTHESIS: The deep_level_ratio signal (ask-side deep/shallow ratio) shows
strong Down-side predictiveness at the market level. However, the signal
needs substantial observation time to stabilize. We use an expanding mean
(cumulative mean up to each snapshot) so we only use information available
at entry time.

We enter a Down position when:
  - Expanding mean deep ratio exceeds threshold
  - At least N seconds of observation (signal stabilization)
  - Wide spread filter (>= 75th percentile)
  - Mid-price filter (0.35–0.65)
  - Entry price cap: never buy above 0.85
"""

import pandas as pd
import numpy as np
import json

# ── CONFIG ──────────────────────────────────────────────────────────────────
DEEP_RATIO_THRESHOLD = 3.0  # lower threshold captures more trades; 5.0+ needs look-ahead
SPREAD_PERCENTILE = 75
MID_PRICE_LO = 0.35
MID_PRICE_HI = 0.65
MAX_ENTRY_PRICE = 0.85
MIN_ORDER_SIZE = 5
MIN_OBSERVATION_MS = 5000  # 5s — short enough to be actionable, long enough for signal

DATA_PATH = "data/sample_1k/book_snapshots_sample_1k.parquet"
RESOLUTION_PATH = "data/sample_1k/resolution_cache_sample.json"
OUTPUT_PATH = "swarm/generation_11/agent_0/results.json"

NEEDED_COLS = [
    "exchange_timestamp", "slug", "token_label", "mid_price", "spread",
    "ask_price_1", "ask_size_1", "ask_size_2", "ask_size_3", "ask_size_4", "ask_size_5",
]


def polymarket_fee(p: np.ndarray) -> np.ndarray:
    return 0.0222 * p * 0.25 * (p * (1 - p)) ** 2


def run_backtest():
    print("Loading data (selected columns only)...")
    df = pd.read_parquet(DATA_PATH, columns=NEEDED_COLS)
    print(f"  Loaded {len(df):,} rows, {df['slug'].nunique()} markets")

    with open(RESOLUTION_PATH) as f:
        resolution_cache = json.load(f)

    slug_winner = {}
    for slug, v in resolution_cache.items():
        if "winnerLabel" in v:
            slug_winner[slug] = v["winnerLabel"]
        elif "winningTokenId" in v and "outcomes" in v and "clobTokenIds" in v:
            try:
                idx = v["clobTokenIds"].index(v["winningTokenId"])
                slug_winner[slug] = v["outcomes"][idx]
            except (ValueError, IndexError):
                continue
    print(f"  Resolved {len(slug_winner)} market winners")

    # Filter to Down token rows with resolved markets
    mask = (df["token_label"] == "Down") & (df["slug"].isin(slug_winner))
    df = df[mask]
    print(f"  Down-token rows (resolved markets): {len(df):,}")

    # Compute spread threshold
    spread_threshold = df["spread"].quantile(SPREAD_PERCENTILE / 100.0)
    print(f"  Spread {SPREAD_PERCENTILE}th pctl threshold: {spread_threshold:.6f}")

    # Sort by slug + timestamp for correct cumulative computation
    df = df.sort_values(["slug", "exchange_timestamp"]).reset_index(drop=True)

    # Compute per-snapshot deep ratio
    deep = df["ask_size_3"].values + df["ask_size_4"].values + df["ask_size_5"].values
    shallow = df["ask_size_1"].values + df["ask_size_2"].values
    with np.errstate(divide="ignore", invalid="ignore"):
        dr = np.where(shallow > 0, deep / shallow, np.nan)

    # Compute EXPANDING (cumulative) mean per market — no look-ahead bias
    print("Computing expanding mean deep ratio per market...")
    dr_series = pd.Series(dr, index=df.index)
    cum_sum = dr_series.groupby(df["slug"]).cumsum()
    cum_count = dr_series.groupby(df["slug"]).cumcount() + 1
    expanding_mean = cum_sum / cum_count

    # Time elapsed since first snapshot per market
    min_ts = df.groupby("slug")["exchange_timestamp"].transform("min")
    time_elapsed = df["exchange_timestamp"].values - min_ts.values

    # Apply filters
    filt = (
        (expanding_mean.values >= DEEP_RATIO_THRESHOLD)
        & (df["spread"].values >= spread_threshold)
        & (df["mid_price"].values >= MID_PRICE_LO)
        & (df["mid_price"].values <= MID_PRICE_HI)
        & (df["ask_price_1"].values <= MAX_ENTRY_PRICE)
        & np.isfinite(expanding_mean.values)
        & (time_elapsed >= MIN_OBSERVATION_MS)
    )

    signals = df[filt]
    print(f"  Signal rows after all filters: {len(signals):,}")

    if len(signals) == 0:
        print("No signals generated!")
        _write_empty_results(spread_threshold)
        return

    # Take first signal per market
    idx_first = signals.groupby("slug")["exchange_timestamp"].idxmin()
    trades = signals.loc[idx_first].copy()
    print(f"  Unique trade entries: {len(trades)}")

    # P&L computation
    trades["entry_price"] = trades["ask_price_1"]
    trades["winner"] = trades["slug"].map(slug_winner)
    trades["payout"] = np.where(trades["winner"] == "Down", 1.0, 0.0)
    trades["fee"] = polymarket_fee(trades["entry_price"].values)
    trades["pnl_per_token"] = trades["payout"] - trades["entry_price"] - trades["fee"]
    trades["pnl"] = trades["pnl_per_token"] * MIN_ORDER_SIZE

    # Metrics
    total_trades = len(trades)
    wins = int((trades["pnl"] > 0).sum())
    win_rate = wins / total_trades if total_trades > 0 else 0.0
    pnl_total = float(trades["pnl"].sum())
    pnl_per_trade = float(trades["pnl"].mean()) if total_trades > 0 else 0.0
    cum_pnl = trades["pnl"].cumsum()
    max_drawdown = float((cum_pnl - cum_pnl.cummax()).min()) if total_trades > 0 else 0.0
    std = trades["pnl"].std()
    sharpe = float((trades["pnl"].mean() / std) * np.sqrt(288)) if std > 0 else 0.0

    print("\n" + "=" * 60)
    print("BACKTEST RESULTS: Deep Level Ratio — Down Only (v3, no lookahead)")
    print("=" * 60)
    print(f"  Total trades:    {total_trades}")
    print(f"  Win rate:        {win_rate:.4f} ({wins}/{total_trades})")
    print(f"  Total P&L:       {pnl_total:.4f}")
    print(f"  P&L per trade:   {pnl_per_trade:.4f}")
    print(f"  Max drawdown:    {max_drawdown:.4f}")
    print(f"  Sharpe ratio:    {sharpe:.4f}")
    print(f"  Avg entry price: {trades['entry_price'].mean():.4f}")
    print(f"  Down win rate:   {(trades['winner'] == 'Down').mean():.4f}")

    # Sensitivity analysis
    print("\n  --- Threshold x Observation Window sensitivity ---")
    print(f"  {'Thresh':>6} {'ObsWin':>6} | {'Trades':>6} {'WR':>7} {'PnL':>10} {'PnL/T':>8} {'Sharpe':>8}")
    print("  " + "-" * 60)
    for thresh in [3.0, 5.0, 7.0, 10.0]:
        for obs_ms in [5000, 15000, 30000, 45000]:
            _filt = (
                (expanding_mean.values >= thresh)
                & (df["spread"].values >= spread_threshold)
                & (df["mid_price"].values >= MID_PRICE_LO)
                & (df["mid_price"].values <= MID_PRICE_HI)
                & (df["ask_price_1"].values <= MAX_ENTRY_PRICE)
                & np.isfinite(expanding_mean.values)
                & (time_elapsed >= obs_ms)
            )
            _sigs = df[_filt]
            if len(_sigs) == 0:
                print(f"  {thresh:>6.1f} {obs_ms//1000:>5}s |      0       -          -        -        -")
                continue
            _idx = _sigs.groupby("slug")["exchange_timestamp"].idxmin()
            _tr = _sigs.loc[_idx]
            _entry = _tr["ask_price_1"].values
            _winner = _tr["slug"].map(slug_winner).values
            _payout = np.where(_winner == "Down", 1.0, 0.0)
            _fee = polymarket_fee(_entry)
            _pnl = (_payout - _entry - _fee) * MIN_ORDER_SIZE
            _n = len(_tr)
            _wins = int((_pnl > 0).sum())
            _wr = _wins / _n if _n > 0 else 0
            _std = _pnl.std()
            _sharpe = (_pnl.mean() / _std) * np.sqrt(288) if _std > 0 else 0
            print(f"  {thresh:>6.1f} {obs_ms//1000:>5}s | {_n:>6d} {_wr:>7.4f} {_pnl.sum():>10.2f} {_pnl.mean():>8.4f} {_sharpe:>8.2f}")

    # Verdict
    if win_rate > 0.55 and pnl_total > 0 and total_trades >= 30:
        verdict = "promising"
    elif win_rate > 0.50 and pnl_total > 0:
        verdict = "marginal"
    else:
        verdict = "failed"

    print(f"\n  Verdict: {verdict}")

    # Write results
    results = {
        "strategy_name": "deep_level_ratio_down_only",
        "description": (
            "Use expanding mean deep_level_ratio (ask-side deep/shallow, cumulative "
            "up to each snapshot — no look-ahead bias) to select Down-side trades. "
            "Enter when expanding mean crosses threshold after minimum observation period. "
            "Combined with spread >= 75th pctl and mid-price 0.35-0.65 filters."
        ),
        "hypothesis": (
            "Deep ask-side layering (ratio of deep to shallow ask sizes) predicts "
            "Down outcomes. Markets where MMs consistently layer deep asks resolve "
            "Down more often. Using expanding mean avoids look-ahead bias — signal "
            "is computed only from data available at entry time."
        ),
        "metrics": {
            "total_trades": total_trades,
            "win_rate": round(win_rate, 4),
            "pnl_total": round(pnl_total, 4),
            "pnl_per_trade": round(pnl_per_trade, 4),
            "max_drawdown": round(max_drawdown, 4),
            "sharpe": round(sharpe, 4),
        },
        "parameters": {
            "deep_ratio_threshold": DEEP_RATIO_THRESHOLD,
            "min_observation_ms": MIN_OBSERVATION_MS,
            "spread_percentile": SPREAD_PERCENTILE,
            "mid_price_range": [MID_PRICE_LO, MID_PRICE_HI],
            "max_entry_price": MAX_ENTRY_PRICE,
            "min_order_size": MIN_ORDER_SIZE,
            "signal_side": "Down only",
            "spread_threshold_value": round(float(spread_threshold), 6),
        },
        "verdict": verdict,
        "next_steps": _next_steps(verdict, total_trades, win_rate, pnl_total),
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {OUTPUT_PATH}")


def _next_steps(verdict, total_trades, win_rate, pnl_total):
    if verdict == "promising":
        return (
            "Validate on full dataset (7187 markets). Test robustness across "
            "time-of-day and volatility regimes. Key risk: signal needs 30s+ "
            "observation, leaving only 4-4.5min to expiry. Check if this timing "
            "provides enough liquidity to fill orders."
        )
    elif verdict == "marginal":
        return (
            "Edge is thin. Try: 1) Different observation windows, "
            "2) Add book imbalance as secondary filter, "
            "3) Test on full dataset for significance."
        )
    else:
        return (
            "Strategy failed. The expanding mean may not stabilize quickly enough "
            "with only 60s of data. Consider: 1) Full dataset with longer observation, "
            "2) Alternative signal construction."
        )


def _write_empty_results(spread_threshold=0.0):
    results = {
        "strategy_name": "deep_level_ratio_down_only",
        "description": "No trades generated.",
        "hypothesis": "See strategy.py.",
        "metrics": {
            "total_trades": 0, "win_rate": 0.0, "pnl_total": 0.0,
            "pnl_per_trade": 0.0, "max_drawdown": 0.0, "sharpe": 0.0,
        },
        "parameters": {"spread_threshold": round(float(spread_threshold), 6)},
        "verdict": "failed",
        "next_steps": "Relax filters or test alternative signal construction.",
    }
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Empty results written to {OUTPUT_PATH}")


if __name__ == "__main__":
    run_backtest()
