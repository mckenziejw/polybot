"""
Strategy: Spread Mean-Reversion with Asymmetric Liquidity Replenishment

Hypothesis: When the bid-ask spread on the Up token widens beyond its rolling
20-snapshot median (temporary liquidity withdrawal), the side where depth recovers
first predicts resolution. If ask-side depth on Up refills faster than bid-side,
makers are positioning for Up resolution → buy Up. Vice versa → buy Down.
"""

import json
import gc
import numpy as np
import pandas as pd
from pathlib import Path

ROLLING_WINDOW = 20
DEPTH_LEVELS = 3
RECOVERY_LOOKBACK = 5
MAX_ENTRY_PRICE = 0.85
BASE_ORDER_SIZE = 50

DATA_PATH = Path("/home/ubuntu/wss_test/data/sample_1k/book_snapshots_sample_1k.parquet")
RESOLUTION_PATH = Path("/home/ubuntu/wss_test/data/sample_1k/resolution_cache_sample.json")
OUTPUT_DIR = Path("/home/ubuntu/wss_test/swarm/generation_6/agent_0")


def polymarket_fee(p):
    p = np.asarray(p, dtype=np.float64)
    return 0.0222 * p * 0.25 * (p * (1 - p)) ** 2


def load_resolution():
    with open(RESOLUTION_PATH) as f:
        return json.load(f)


def resolve_winner(resolution_raw, token_map):
    resolution = {}
    for slug, info in resolution_raw.items():
        if "winnerLabel" in info:
            resolution[slug] = info["winnerLabel"]
        elif "outcomes" in info and "clobTokenIds" in info and "winningTokenId" in info:
            try:
                idx = info["clobTokenIds"].index(info["winningTokenId"])
                resolution[slug] = info["outcomes"][idx]
            except (ValueError, IndexError):
                pass
        elif "winning_token" in info:
            wt = info["winning_token"]
            if wt in token_map:
                resolution[slug] = token_map[wt]
        elif "winningTokenId" in info:
            wt = info["winningTokenId"]
            if wt in token_map:
                resolution[slug] = token_map[wt]
    return resolution


def run_backtest(spread_threshold=1.0, verbose=True):
    """Run backtest. Loads only Up token data to save memory."""
    if verbose:
        print("Loading data...")

    # Load only needed columns, filter to Up token immediately
    cols = ['exchange_timestamp', 'slug', 'token_label', 'asset_id',
            'mid_price', 'spread', 'bid_price_1',
            'bid_size_1', 'bid_size_2', 'bid_size_3',
            'ask_price_1', 'ask_size_1', 'ask_size_2', 'ask_size_3']

    df = pd.read_parquet(DATA_PATH, columns=cols)
    if verbose:
        print(f"  Loaded {len(df):,} rows")

    # Build token map before filtering
    token_map_df = df[['asset_id', 'token_label']].drop_duplicates()
    token_map = dict(zip(token_map_df['asset_id'], token_map_df['token_label']))

    # Filter to Up token only — halves memory
    df = df[df['token_label'] == 'Up'].copy()
    df.drop(columns=['token_label', 'asset_id'], inplace=True)
    if verbose:
        print(f"  Up token rows: {len(df):,}, {df['slug'].nunique()} markets")

    # Downcast floats to float32
    float_cols = df.select_dtypes('float64').columns
    df[float_cols] = df[float_cols].astype(np.float32)
    gc.collect()

    # Resolution
    resolution_raw = load_resolution()
    resolution = resolve_winner(resolution_raw, token_map)
    resolved = {s for s, v in resolution.items() if v is not None}
    if verbose:
        print(f"  Resolved markets: {len(resolved)}")

    # Filter to resolved markets only
    df = df[df['slug'].isin(resolved)].copy()
    if verbose:
        print(f"  Rows after resolution filter: {len(df):,}")

    # Sort
    df.sort_values(['slug', 'exchange_timestamp'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # ── Vectorized signal computation ──
    if verbose:
        print("Computing signals...")

    # Rolling median spread per market
    df['spread_median'] = df.groupby('slug')['spread'].transform(
        lambda x: x.rolling(ROLLING_WINDOW, min_periods=5).median()
    )

    # Wide spread
    df['wide'] = df['spread'] > df['spread_median'] * (1 + spread_threshold)

    # Depth
    df['bid_depth'] = df['bid_size_1'] + df['bid_size_2'] + df['bid_size_3']
    df['ask_depth'] = df['ask_size_1'] + df['ask_size_2'] + df['ask_size_3']

    # Depth change over RECOVERY_LOOKBACK
    df['bid_d'] = df.groupby('slug')['bid_depth'].diff(RECOVERY_LOOKBACK)
    df['ask_d'] = df.groupby('slug')['ask_depth'].diff(RECOVERY_LOOKBACK)

    # Recovery signal: ask recovery - bid recovery
    df['signal'] = df['ask_d'] - df['bid_d']

    # Filter candidates
    candidates = df[df['wide'] & df['signal'].notna()].copy()
    if verbose:
        print(f"  Wide-spread snapshots: {df['wide'].sum():,}")
        print(f"  Candidates with signal: {len(candidates):,}")

    if len(candidates) == 0:
        if verbose:
            print("  No candidates!")
        return pd.DataFrame(), _empty_metrics()

    # Pick best signal per market
    candidates['abs_signal'] = candidates['signal'].abs()
    idx_best = candidates.groupby('slug')['abs_signal'].idxmax()
    trades = candidates.loc[idx_best].copy()

    # Direction
    trades['buy_token'] = np.where(trades['signal'] > 0, 'Up', 'Down')

    # Entry price
    trades['entry_price'] = np.where(
        trades['buy_token'] == 'Up',
        trades['ask_price_1'],
        1.0 - trades['bid_price_1']
    ).astype(np.float64)

    # Filter entry price
    trades = trades[(trades['entry_price'] <= MAX_ENTRY_PRICE) &
                    (trades['entry_price'] > 0.01)].copy()

    # Resolution
    trades['winner'] = trades['slug'].map(resolution)
    trades = trades.dropna(subset=['winner'])

    # P&L
    trades['won'] = trades['buy_token'] == trades['winner']
    trades['fee'] = polymarket_fee(trades['entry_price'].values)
    trades['pnl'] = np.where(
        trades['won'],
        (1.0 - trades['entry_price'] - trades['fee']) * BASE_ORDER_SIZE,
        (-trades['entry_price'] - trades['fee']) * BASE_ORDER_SIZE
    )

    metrics = _compute_metrics(trades)
    if verbose:
        _print_metrics(trades, metrics)

    del df, candidates
    gc.collect()
    return trades, metrics


def _empty_metrics():
    return {"total_trades": 0, "win_rate": 0.0, "pnl_total": 0.0,
            "pnl_per_trade": 0.0, "max_drawdown": 0.0, "sharpe": 0.0}


def _compute_metrics(trades):
    if len(trades) == 0:
        return _empty_metrics()
    n = len(trades)
    wins = int(trades['won'].sum())
    pnl_total = float(trades['pnl'].sum())
    cumul = trades['pnl'].cumsum()
    max_dd = float((cumul - cumul.cummax()).min())
    std = trades['pnl'].std()
    sharpe = float(trades['pnl'].mean() / std * np.sqrt(288)) if std > 0 else 0.0
    return {
        "total_trades": n, "win_rate": round(wins / n, 4),
        "pnl_total": round(pnl_total, 2), "pnl_per_trade": round(pnl_total / n, 4),
        "max_drawdown": round(max_dd, 2), "sharpe": round(sharpe, 4),
    }


def _print_metrics(trades, m):
    print(f"\n── Results ──")
    print(f"  Trades: {m['total_trades']}")
    print(f"  Win rate: {m['win_rate']:.1%}")
    print(f"  PnL total: ${m['pnl_total']:.2f}")
    print(f"  PnL/trade: ${m['pnl_per_trade']:.4f}")
    print(f"  Max DD: ${m['max_drawdown']:.2f}")
    print(f"  Sharpe: {m['sharpe']:.4f}")
    if len(trades) > 0:
        for label in ['Up', 'Down']:
            t = trades[trades['buy_token'] == label]
            if len(t):
                print(f"  {label}: {len(t)} trades, WR={t['won'].mean():.1%}")
        print(f"  Avg entry: {trades['entry_price'].mean():.3f}")


def main():
    # Default parameters
    trades, metrics = run_backtest(spread_threshold=1.0, verbose=True)

    # Parameter sweep
    print("\n── Parameter Sweep ──")
    sweep = []
    for st in [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0]:
        _, m = run_backtest(spread_threshold=st, verbose=False)
        m['spread_threshold'] = st
        sweep.append(m)
        print(f"  thresh={st:.2f}: n={m['total_trades']:3d}, "
              f"WR={m['win_rate']:.1%}, PnL=${m['pnl_total']:8.2f}, "
              f"Sharpe={m['sharpe']:.3f}")

    sweep_df = pd.DataFrame(sweep)
    best = sweep_df.sort_values('pnl_total', ascending=False).iloc[0]

    # Verdict
    if metrics['total_trades'] >= 20 and metrics['win_rate'] > 0.55 and metrics['pnl_total'] > 0:
        verdict = "promising"
    elif metrics['total_trades'] >= 10 and metrics['pnl_total'] > 0:
        verdict = "marginal"
    else:
        verdict = "failed"

    results = {
        "strategy_name": "spread_mean_reversion_liquidity_recovery",
        "description": (
            "Detects when Up token bid-ask spread widens beyond rolling 20-snapshot median, "
            "then measures asymmetric depth recovery across top 3 book levels. "
            "Buy the token whose side market makers are replenishing faster."
        ),
        "hypothesis": (
            "When spread widens (liquidity withdrawal), the side where depth recovers first "
            "indicates which token market makers are re-quoting toward fair value, "
            "predicting resolution direction."
        ),
        "metrics": metrics,
        "parameters": {
            "rolling_window": ROLLING_WINDOW,
            "spread_threshold": 1.0,
            "depth_levels": DEPTH_LEVELS,
            "recovery_lookback": RECOVERY_LOOKBACK,
            "max_entry_price": MAX_ENTRY_PRICE,
            "base_order_size": BASE_ORDER_SIZE,
        },
        "sweep_summary": sweep,
        "sweep_best": {
            "spread_threshold": float(best['spread_threshold']),
            "total_trades": int(best['total_trades']),
            "win_rate": float(best['win_rate']),
            "pnl_total": float(best['pnl_total']),
        },
        "verdict": verdict,
        "next_steps": (
            "If promising: validate on full dataset with time-based OOS split. "
            "If failed: 60s observation window (sample data) may be too short for "
            "spread mean-reversion dynamics. Also, early-market snapshots may reflect "
            "opening noise rather than informative liquidity signals."
        ),
    }

    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Verdict: {verdict}")
    print(f"  Best sweep: thresh={best['spread_threshold']}, "
          f"PnL=${best['pnl_total']:.2f}, WR={best['win_rate']:.1%}")
    print(f"  Written to {OUTPUT_DIR / 'results.json'}")


if __name__ == "__main__":
    main()
