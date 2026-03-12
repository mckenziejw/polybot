"""
Strategy: Ultra-Early Orderbook Imbalance (500ms window, threshold 0.25)

Hypothesis: The earliest orderbook snapshots (first 500ms) contain the most
informative MM positioning before reactive flow arrives. Narrowing from 2000ms
to 500ms isolates the initial liquidity placement signal, while lowering the
signal threshold from 0.3 to 0.25 maintains trade count.

Filters: UTC 0-6 hours, mid-price 0.35-0.65, entry price cap 0.85.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import pyarrow.parquet as pq

# ── Parameters ──────────────────────────────────────────────────────────
ENTRY_WINDOW_MS = 500
IMBALANCE_THRESHOLD = 0.25
UTC_HOUR_MIN = 0
UTC_HOUR_MAX = 6
MID_PRICE_MIN = 0.35
MID_PRICE_MAX = 0.65
ENTRY_PRICE_CAP = 0.85
ORDER_SIZE_TOKENS = 5

NEEDED_COLS = [
    'exchange_timestamp', 'slug', 'asset_id', 'token_label',
    'mid_price', 'book_imbalance', 'bid_price_1', 'ask_price_1',
]


def calc_fee(p: np.ndarray) -> np.ndarray:
    return 0.0222 * p * 0.25 * (p * (1 - p)) ** 2


def load_resolutions(path: str) -> dict:
    with open(path) as f:
        rc = json.load(f)
    results = {}
    for slug, info in rc.items():
        if 'winnerLabel' in info:
            results[slug] = info['winnerLabel']
        elif 'winningTokenId' in info and 'clobTokenIds' in info and 'outcomes' in info:
            wid = info['winningTokenId']
            try:
                idx = info['clobTokenIds'].index(wid)
                results[slug] = info['outcomes'][idx]
            except (ValueError, IndexError):
                continue
        elif 'token_outcomes' in info:
            for tid, outcome_val in info['token_outcomes'].items():
                if outcome_val == 1.0:
                    results[slug] = tid
                    break
    return results


def run_strategy():
    print("Loading data (selected columns only)...")
    df = pq.read_table(
        'data/sample_1k/book_snapshots_sample_1k.parquet',
        columns=NEEDED_COLS,
    ).to_pandas()
    print(f"Loaded {len(df):,} rows")

    resolutions = load_resolutions('data/sample_1k/resolution_cache_sample.json')

    # Extract market epoch from slug
    df['market_epoch_ms'] = df['slug'].str.rsplit('-', n=1).str[1].astype(np.int64) * 1000
    df['offset_ms'] = df['exchange_timestamp'] - df['market_epoch_ms']

    # Filter to first 500ms window
    mask = (df['offset_ms'] >= 0) & (df['offset_ms'] <= ENTRY_WINDOW_MS)
    df = df[mask]
    print(f"Rows in 500ms window: {len(df):,}")

    # UTC hour filter
    df['utc_hour'] = (df['market_epoch_ms'] // 1000 % 86400) // 3600
    df = df[(df['utc_hour'] >= UTC_HOUR_MIN) & (df['utc_hour'] <= UTC_HOUR_MAX)]
    print(f"Rows after UTC 0-6 filter: {len(df):,}")

    # Split Up and Down - keep Down asset_ids for resolution
    down_assets = (
        df.loc[df['token_label'] == 'Down', ['slug', 'asset_id']]
        .drop_duplicates('slug')
        .set_index('slug')['asset_id']
        .to_dict()
    )

    # Focus on Up tokens for signal
    df_up = df[df['token_label'] == 'Up']
    df_up = df_up[(df_up['mid_price'] >= MID_PRICE_MIN) & (df_up['mid_price'] <= MID_PRICE_MAX)]
    print(f"Up-token rows after mid-price filter: {len(df_up):,}")

    # Free memory
    del df

    # Aggregate per market
    market_signals = df_up.groupby('slug').agg(
        mean_imbalance=('book_imbalance', 'mean'),
        mean_mid=('mid_price', 'mean'),
        mean_bid1=('bid_price_1', 'mean'),
        mean_ask1=('ask_price_1', 'mean'),
        snapshot_count=('book_imbalance', 'count'),
        first_asset_id=('asset_id', 'first'),
    ).reset_index()
    del df_up

    print(f"Markets with signals: {len(market_signals)}")

    # Generate trades
    buy_up = market_signals[market_signals['mean_imbalance'] > IMBALANCE_THRESHOLD].copy()
    buy_up['side'] = 'Up'
    buy_up['entry_price'] = buy_up['mean_ask1']

    buy_down = market_signals[market_signals['mean_imbalance'] < -IMBALANCE_THRESHOLD].copy()
    buy_down['side'] = 'Down'
    buy_down['entry_price'] = 1.0 - buy_down['mean_bid1']

    all_trades = pd.concat([buy_up, buy_down], ignore_index=True)
    all_trades = all_trades[all_trades['entry_price'] <= ENTRY_PRICE_CAP]
    print(f"Total trades after price cap: {len(all_trades)}")

    if len(all_trades) == 0:
        return write_empty_results("No trades generated")

    # Resolve outcomes (vectorized)
    all_trades['winner'] = all_trades['slug'].map(resolutions)

    # Label-based resolution
    mask_label = all_trades['winner'].isin(['Up', 'Down'])
    all_trades.loc[mask_label, 'won'] = (
        all_trades.loc[mask_label, 'winner'] == all_trades.loc[mask_label, 'side']
    ).astype(float)

    # Token-ID resolution for Up
    mask_tok_up = (~mask_label) & (all_trades['side'] == 'Up') & all_trades['winner'].notna()
    if mask_tok_up.any():
        all_trades.loc[mask_tok_up, 'won'] = (
            all_trades.loc[mask_tok_up, 'winner'] == all_trades.loc[mask_tok_up, 'first_asset_id']
        ).astype(float)

    # Token-ID resolution for Down
    mask_tok_down = (~mask_label) & (all_trades['side'] == 'Down') & all_trades['winner'].notna()
    if mask_tok_down.any():
        all_trades['down_asset_id'] = all_trades['slug'].map(down_assets)
        all_trades.loc[mask_tok_down, 'won'] = (
            all_trades.loc[mask_tok_down, 'winner'] == all_trades.loc[mask_tok_down, 'down_asset_id']
        ).astype(float)

    all_trades = all_trades.dropna(subset=['won'])
    print(f"Resolved trades: {len(all_trades)}")

    if len(all_trades) == 0:
        return write_empty_results("No resolved trades")

    # PnL
    p = all_trades['entry_price'].values
    fee = calc_fee(p)
    cost = (p + fee) * ORDER_SIZE_TOKENS
    payout = all_trades['won'].values * ORDER_SIZE_TOKENS
    all_trades['pnl'] = payout - cost

    total_trades = len(all_trades)
    wins = int(all_trades['won'].sum())
    win_rate = wins / total_trades
    pnl_total = float(all_trades['pnl'].sum())
    pnl_per_trade = pnl_total / total_trades

    cumulative = all_trades['pnl'].cumsum()
    max_drawdown = float((cumulative - cumulative.cummax()).min())

    pnl_std = all_trades['pnl'].std()
    sharpe = float((all_trades['pnl'].mean() / pnl_std) * np.sqrt(288)) if pnl_std > 0 else 0.0

    print(f"\n{'='*50}")
    print(f"RESULTS: Ultra-Early Imbalance (500ms / thresh 0.25)")
    print(f"{'='*50}")
    print(f"Total trades:    {total_trades}")
    print(f"Wins:            {wins}")
    print(f"Win rate:        {win_rate:.4f}")
    print(f"PnL total:       {pnl_total:.4f}")
    print(f"PnL per trade:   {pnl_per_trade:.4f}")
    print(f"Max drawdown:    {max_drawdown:.4f}")
    print(f"Sharpe:          {sharpe:.4f}")

    for side in ['Up', 'Down']:
        s = all_trades[all_trades['side'] == side]
        if len(s) > 0:
            print(f"  {side}: {len(s)} trades, wr={s['won'].mean():.4f}, pnl={s['pnl'].sum():.4f}")

    all_trades['abs_imbal'] = all_trades['mean_imbalance'].abs()
    for lo, hi in [(0.25, 0.35), (0.35, 0.50), (0.50, 1.0)]:
        s = all_trades[(all_trades['abs_imbal'] >= lo) & (all_trades['abs_imbal'] < hi)]
        if len(s) > 0:
            print(f"  Imbal [{lo:.2f},{hi:.2f}): {len(s)} trades, wr={s['won'].mean():.4f}, pnl={s['pnl'].sum():.4f}")

    # Verdict
    if pnl_per_trade > 0.02 and win_rate > 0.52 and total_trades >= 20:
        verdict = "promising"
    elif pnl_per_trade > 0 and win_rate > 0.50:
        verdict = "marginal"
    else:
        verdict = "failed"

    next_steps = ""
    if verdict == "promising":
        next_steps = "Validate on full dataset. Test robustness across different UTC windows. Consider position sizing by imbalance magnitude."
    elif verdict == "marginal":
        next_steps = "Test on full dataset to see if edge holds with more data. Consider tighter imbalance thresholds or combining with spread signal."
    else:
        next_steps = "500ms window may be too narrow for reliable signal. Consider 1000ms or 2000ms variants, or combining imbalance with other features."

    results = {
        "strategy_name": "ultra_early_imbalance_500ms_thresh025",
        "description": "Buy based on mean book imbalance in first 500ms of market open. "
                       "Positive imbalance -> buy Up, negative -> buy Down. "
                       "UTC 0-6 hours, mid-price 0.35-0.65 filter.",
        "hypothesis": "The earliest 500ms orderbook snapshots contain the most informative "
                      "MM positioning before reactive flow arrives. Lowering threshold to 0.25 "
                      "maintains trade count despite tighter window.",
        "metrics": {
            "total_trades": total_trades,
            "win_rate": round(win_rate, 4),
            "pnl_total": round(pnl_total, 4),
            "pnl_per_trade": round(pnl_per_trade, 4),
            "max_drawdown": round(max_drawdown, 4),
            "sharpe": round(sharpe, 4),
        },
        "parameters": {
            "entry_window_ms": ENTRY_WINDOW_MS,
            "imbalance_threshold": IMBALANCE_THRESHOLD,
            "utc_hour_range": [UTC_HOUR_MIN, UTC_HOUR_MAX],
            "mid_price_range": [MID_PRICE_MIN, MID_PRICE_MAX],
            "entry_price_cap": ENTRY_PRICE_CAP,
            "order_size_tokens": ORDER_SIZE_TOKENS,
        },
        "verdict": verdict,
        "next_steps": next_steps,
    }

    out_path = Path(__file__).parent / 'results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nVerdict: {verdict}")
    print(f"Results written to {out_path}")


def write_empty_results(reason: str):
    results = {
        "strategy_name": "ultra_early_imbalance_500ms_thresh025",
        "description": "Buy based on mean book imbalance in first 500ms of market open.",
        "hypothesis": "The earliest 500ms snapshots contain most informative MM positioning.",
        "metrics": {
            "total_trades": 0, "win_rate": 0.0, "pnl_total": 0.0,
            "pnl_per_trade": 0.0, "max_drawdown": 0.0, "sharpe": 0.0,
        },
        "parameters": {
            "entry_window_ms": ENTRY_WINDOW_MS,
            "imbalance_threshold": IMBALANCE_THRESHOLD,
        },
        "verdict": "failed",
        "next_steps": reason,
    }
    out_path = Path(__file__).parent / 'results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results written to {out_path}")


if __name__ == '__main__':
    run_strategy()
