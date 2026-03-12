"""
Strategy: Tight Window + Wide Spread + Extreme Imbalance
========================================================
Hypothesis: Tighten window to 1000ms, keep wide-spread filter (>= 75th pctile),
mid-price filter (0.35-0.65), and raise signal threshold to 0.4 for book_imbalance.
Only the most extreme early MM positioning in uncertain, wide-spread markets.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

# === Parameters ===
WINDOW_MS = 1000          # First 1000ms of market
SPREAD_PCTILE = 75        # >= 75th percentile of spread (wide-spread filter)
MID_PRICE_LOW = 0.35      # Mid-price filter
MID_PRICE_HIGH = 0.65
IMBALANCE_THRESHOLD = 0.4 # Absolute book imbalance >= 0.4
MAX_ENTRY_PRICE = 0.85    # Never buy above this
MIN_ORDER_SIZE = 5        # Minimum tokens
ORDER_SIZE = 10           # Fixed order size in tokens


def compute_fee(p: np.ndarray) -> np.ndarray:
    """Polymarket fee: 0.0222 * p * 0.25 * (p*(1-p))^2"""
    return 0.0222 * p * 0.25 * (p * (1.0 - p)) ** 2


def resolve_markets(rc: dict, token_map: pd.DataFrame) -> dict:
    """Resolve all markets to winner label (Up/Down)."""
    resolved = {}
    for slug, info in rc.items():
        if 'winnerLabel' in info:
            resolved[slug] = info['winnerLabel']
        elif 'winning_token' in info:
            wt = info['winning_token']
            match = token_map[(token_map['slug'] == slug) & (token_map['asset_id'] == wt)]
            if len(match) > 0:
                resolved[slug] = match['token_label'].iloc[0]
        elif 'winningTokenId' in info and 'clobTokenIds' in info and 'outcomes' in info:
            wt = info['winningTokenId']
            for tid, outcome in zip(info['clobTokenIds'], info['outcomes']):
                if tid == wt:
                    resolved[slug] = outcome
                    break
        elif 'winningTokenId' in info:
            wt = info['winningTokenId']
            match = token_map[(token_map['slug'] == slug) & (token_map['asset_id'] == wt)]
            if len(match) > 0:
                resolved[slug] = match['token_label'].iloc[0]
    return resolved


def run_backtest():
    """Run the strategy backtest on sample data."""
    # Load data
    print("Loading data...")
    df = pd.read_parquet(
        'data/sample_1k/book_snapshots_sample_1k.parquet'
    )
    with open('data/sample_1k/resolution_cache_sample.json') as f:
        rc = json.load(f)

    # Build token map for resolution
    token_map = df[['slug', 'asset_id', 'token_label']].drop_duplicates(
        subset=['slug', 'asset_id']
    )
    resolved = resolve_markets(rc, token_map)
    print(f"Resolved {len(resolved)} / {len(rc)} markets")

    # Compute market open time per slug
    print("Computing market open times...")
    slug_open = df.groupby('slug')['exchange_timestamp'].min()
    df = df.merge(slug_open.rename('market_open_ts'), on='slug')
    df['time_since_open'] = df['exchange_timestamp'] - df['market_open_ts']

    # Filter to first WINDOW_MS
    print(f"Filtering to first {WINDOW_MS}ms window...")
    early = df[df['time_since_open'] <= WINDOW_MS].copy()
    print(f"  Rows in window: {len(early):,}, Markets: {early['slug'].nunique()}")

    # Compute spread percentile threshold
    spread_threshold = early['spread'].quantile(SPREAD_PCTILE / 100.0)
    print(f"  Spread 75th percentile: {spread_threshold:.4f}")

    # Filter: wide spread
    early = early[early['spread'] >= spread_threshold]
    print(f"  After wide-spread filter: {len(early):,} rows, {early['slug'].nunique()} markets")

    # Filter: mid-price in [0.35, 0.65]
    early = early[
        (early['mid_price'] >= MID_PRICE_LOW) &
        (early['mid_price'] <= MID_PRICE_HIGH)
    ]
    print(f"  After mid-price filter: {len(early):,} rows, {early['slug'].nunique()} markets")

    # Aggregate: compute mean imbalance per slug+token_label in the window
    # We want to look at Up token imbalance to decide direction
    print("Computing signals per market...")
    up_data = early[early['token_label'] == 'Up']

    if len(up_data) == 0:
        print("No data after filters!")
        return _empty_results()

    # Mean imbalance and best ask for Up token per slug
    signals = up_data.groupby('slug').agg(
        mean_imbalance=('book_imbalance', 'mean'),
        median_imbalance=('book_imbalance', 'median'),
        mean_mid=('mid_price', 'mean'),
        best_ask=('ask_price_1', 'mean'),  # average best ask in window
        best_bid=('bid_price_1', 'mean'),
        n_snapshots=('book_imbalance', 'count'),
        mean_spread=('spread', 'mean'),
    ).reset_index()

    # Also get Down token best ask
    down_data = early[early['token_label'] == 'Down']
    down_signals = down_data.groupby('slug').agg(
        down_best_ask=('ask_price_1', 'mean'),
        down_best_bid=('bid_price_1', 'mean'),
    ).reset_index()
    signals = signals.merge(down_signals, on='slug', how='left')

    print(f"  Markets with signals: {len(signals)}")
    print(f"  Imbalance distribution:")
    print(f"    mean: {signals['mean_imbalance'].mean():.4f}")
    print(f"    std:  {signals['mean_imbalance'].std():.4f}")
    print(f"    abs >= {IMBALANCE_THRESHOLD}: {(signals['mean_imbalance'].abs() >= IMBALANCE_THRESHOLD).sum()}")

    # Generate trades: extreme imbalance signals
    trades = []
    for _, row in signals.iterrows():
        slug = row['slug']
        imbalance = row['mean_imbalance']

        if slug not in resolved:
            continue

        winner = resolved[slug]

        if imbalance >= IMBALANCE_THRESHOLD:
            # Strong bid-side imbalance on Up token -> buy Up
            entry_price = row['best_ask']
            if entry_price > MAX_ENTRY_PRICE or np.isnan(entry_price):
                continue
            token_bought = 'Up'
            payout = 1.0 if winner == 'Up' else 0.0

        elif imbalance <= -IMBALANCE_THRESHOLD:
            # Strong ask-side imbalance on Up token -> buy Down
            entry_price = row['down_best_ask']
            if entry_price > MAX_ENTRY_PRICE or np.isnan(entry_price):
                continue
            token_bought = 'Down'
            payout = 1.0 if winner == 'Down' else 0.0

        else:
            continue

        fee = compute_fee(np.array([entry_price]))[0]
        cost = entry_price + fee
        pnl = (payout - cost) * ORDER_SIZE

        trades.append({
            'slug': slug,
            'token_bought': token_bought,
            'entry_price': entry_price,
            'fee': fee,
            'cost': cost,
            'payout': payout,
            'pnl': pnl,
            'winner': winner,
            'won': token_bought == winner,
            'imbalance': imbalance,
            'spread': row['mean_spread'],
            'mid_price': row['mean_mid'],
        })

    if not trades:
        print("No trades generated!")
        return _empty_results()

    trades_df = pd.DataFrame(trades)
    print(f"\n=== RESULTS ===")
    print(f"Total trades: {len(trades_df)}")

    # Metrics
    n_trades = len(trades_df)
    n_wins = trades_df['won'].sum()
    win_rate = n_wins / n_trades
    total_pnl = trades_df['pnl'].sum()
    pnl_per_trade = total_pnl / n_trades

    # Sharpe (annualized - assuming ~288 5m markets per day)
    daily_trades = n_trades  # this is a sample, compute Sharpe from trade-level
    trade_returns = trades_df['pnl'] / (trades_df['cost'] * ORDER_SIZE)
    sharpe = (trade_returns.mean() / trade_returns.std() * np.sqrt(252 * 288)
              if trade_returns.std() > 0 else 0.0)

    # Max drawdown
    cumulative_pnl = trades_df['pnl'].cumsum()
    running_max = cumulative_pnl.cummax()
    drawdown = cumulative_pnl - running_max
    max_drawdown = drawdown.min()

    print(f"Wins: {n_wins} / {n_trades} ({win_rate:.1%})")
    print(f"Total PnL: ${total_pnl:.2f}")
    print(f"PnL per trade: ${pnl_per_trade:.2f}")
    print(f"Max drawdown: ${max_drawdown:.2f}")
    print(f"Sharpe ratio: {sharpe:.2f}")
    print(f"Avg entry price: {trades_df['entry_price'].mean():.4f}")
    print(f"Avg spread: {trades_df['spread'].mean():.4f}")
    print(f"Avg |imbalance|: {trades_df['imbalance'].abs().mean():.4f}")

    # Breakdown by direction
    print(f"\n--- By direction ---")
    for token in ['Up', 'Down']:
        sub = trades_df[trades_df['token_bought'] == token]
        if len(sub) > 0:
            print(f"  {token}: {len(sub)} trades, "
                  f"win rate {sub['won'].mean():.1%}, "
                  f"PnL ${sub['pnl'].sum():.2f}")

    # Determine verdict
    if sharpe >= 1.0 and win_rate >= 0.55 and pnl_per_trade > 0:
        verdict = "promising"
    elif sharpe >= 0.5 or (win_rate >= 0.52 and pnl_per_trade > 0):
        verdict = "marginal"
    else:
        verdict = "failed"

    print(f"\nVerdict: {verdict}")

    results = {
        "strategy_name": "tight_window_wide_spread_extreme_imbalance",
        "description": (
            "1000ms window + 75th-pctile spread filter + mid-price 0.35-0.65 + "
            "book imbalance threshold 0.4. Buy Up when imbalance >= 0.4 (strong bid), "
            "buy Down when imbalance <= -0.4 (strong ask on Up token)."
        ),
        "hypothesis": (
            "Tighten window from 2000ms to 1000ms while keeping wide-spread filter "
            "(spread >= 75th pctile) and mid-price filter (0.35-0.65), raise signal "
            "threshold from 0.3 to 0.4 to capture only extreme imbalances. The 1s window "
            "showed Sharpe 1.02 on broader sample; combining with filters that produced "
            "highest per-trade PnL ($0.79) and win rate (66.7%) should yield fewer but "
            "higher-quality trades."
        ),
        "metrics": {
            "total_trades": int(n_trades),
            "win_rate": round(float(win_rate), 4),
            "pnl_total": round(float(total_pnl), 2),
            "pnl_per_trade": round(float(pnl_per_trade), 4),
            "max_drawdown": round(float(max_drawdown), 2),
            "sharpe": round(float(sharpe), 2),
        },
        "parameters": {
            "window_ms": WINDOW_MS,
            "spread_percentile": SPREAD_PCTILE,
            "mid_price_range": [MID_PRICE_LOW, MID_PRICE_HIGH],
            "imbalance_threshold": IMBALANCE_THRESHOLD,
            "max_entry_price": MAX_ENTRY_PRICE,
            "order_size": ORDER_SIZE,
        },
        "verdict": verdict,
        "next_steps": "",
    }

    # Set next_steps based on verdict
    if verdict == "promising":
        results["next_steps"] = (
            "Run on full dataset (7187 files). Test robustness across time periods. "
            "Optimize order size. Test sensitivity to threshold (0.35-0.45)."
        )
    elif verdict == "marginal":
        results["next_steps"] = (
            "Try adjusting imbalance threshold (0.35 or 0.45). "
            "Consider adding volume-weighted imbalance. Test on full dataset."
        )
    else:
        results["next_steps"] = (
            "The combination of tight window + wide spread + high threshold is too "
            "restrictive or the signal doesn't survive fees. Try relaxing one filter."
        )

    return results, trades_df


def _empty_results():
    results = {
        "strategy_name": "tight_window_wide_spread_extreme_imbalance",
        "description": "1000ms window + wide spread + extreme imbalance filter",
        "hypothesis": "See strategy.py header",
        "metrics": {
            "total_trades": 0,
            "win_rate": 0.0,
            "pnl_total": 0.0,
            "pnl_per_trade": 0.0,
            "max_drawdown": 0.0,
            "sharpe": 0.0,
        },
        "parameters": {},
        "verdict": "failed",
        "next_steps": "No trades generated. Filters too restrictive.",
    }
    return results, pd.DataFrame()


if __name__ == '__main__':
    results, trades_df = run_backtest()

    out_dir = Path(__file__).parent
    with open(out_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {out_dir / 'results.json'}")

    if len(trades_df) > 0:
        trades_df.to_csv(out_dir / 'trades.csv', index=False)
        print(f"Trade log written to {out_dir / 'trades.csv'}")
