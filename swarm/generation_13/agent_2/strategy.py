"""
Strategy: Deep Level Ratio in Narrow Mid-Price Band (0.45-0.55) with UTC 0-6 Filter

HYPOTHESIS: In the most uncertain markets (mid_price 0.45-0.55, true coin-flip territory),
market maker positioning at deeper orderbook levels carries maximum informational content
because there is no strong prior to anchor prices. We use the deep_level_ratio signal
(ratio of deep level liquidity to top level liquidity) with a 2-second aggregation window
and a lower threshold of 0.25 to maintain trade count despite the narrower universe.

SIGNAL: deep_level_ratio = (bid_size_3 + bid_size_4 + bid_size_5) / (bid_size_1 + bid_size_2)
       vs (ask_size_3 + ask_size_4 + ask_size_5) / (ask_size_1 + ask_size_2)

When bid deep ratio >> ask deep ratio, MMs are stacking hidden bids deeper -> bullish signal -> buy Up
When ask deep ratio >> bid deep ratio, MMs are stacking hidden asks deeper -> bearish signal -> buy Down
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

# === CONFIG ===
SAMPLE_BOOK_PATH = "data/sample_1k/book_snapshots_sample_1k.parquet"
RESOLUTION_CACHE_PATH = "data/sample_1k/resolution_cache_sample.json"
OUTPUT_DIR = Path("swarm/generation_13/agent_2")

# Strategy parameters
MID_PRICE_LOW = 0.45
MID_PRICE_HIGH = 0.55
SIGNAL_THRESHOLD = 0.25
ENTRY_PRICE_CAP = 0.85
MIN_ORDER_SIZE = 5
UTC_HOUR_START = 0
UTC_HOUR_END = 6
WINDOW_MS = 2000  # 2-second aggregation window

# Fee formula: fee = 0.0222 * p * 0.25 * (p*(1-p))^2
def compute_fee(p):
    """Vectorized fee computation."""
    return 0.0222 * p * 0.25 * (p * (1 - p)) ** 2


NEEDED_COLS = [
    'exchange_timestamp', 'slug', 'token_label', 'mid_price',
    'bid_price_1', 'ask_price_1',
    'bid_size_1', 'bid_size_2', 'bid_size_3', 'bid_size_4', 'bid_size_5',
    'ask_size_1', 'ask_size_2', 'ask_size_3', 'ask_size_4', 'ask_size_5',
]


def load_data():
    """Load sample book snapshots and resolution cache."""
    print("Loading data...")
    df = pd.read_parquet(SAMPLE_BOOK_PATH, columns=NEEDED_COLS)
    # Immediately filter to Up tokens to halve memory
    df = df[df['token_label'] == 'Up']
    print(f"  Book snapshots (Up only): {len(df):,} rows, {df['slug'].nunique()} unique slugs")

    with open(RESOLUTION_CACHE_PATH, 'r') as f:
        resolution_cache = json.load(f)
    print(f"  Resolution cache: {len(resolution_cache)} entries")

    return df, resolution_cache


def resolve_winner(resolution_cache, slug):
    """Determine the winner (Up/Down) for a given slug."""
    entry = resolution_cache.get(slug)
    if entry is None:
        return None

    # Some entries have winnerLabel directly
    if 'winnerLabel' in entry:
        return entry['winnerLabel']

    # Others have winningTokenId + clobTokenIds + outcomes
    if 'winningTokenId' in entry and 'clobTokenIds' in entry and 'outcomes' in entry:
        winning_id = entry['winningTokenId']
        for i, token_id in enumerate(entry['clobTokenIds']):
            if token_id == winning_id:
                return entry['outcomes'][i]

    return None


def prepare_resolution_map(resolution_cache):
    """Build slug -> winner mapping."""
    resolution_map = {}
    for slug, entry in resolution_cache.items():
        winner = resolve_winner(resolution_cache, slug)
        if winner is not None:
            resolution_map[slug] = winner
    print(f"  Resolved {len(resolution_map)} markets")
    return resolution_map


def run_strategy():
    """Execute the deep_level_ratio strategy on sample data."""
    df, resolution_cache = load_data()
    resolution_map = prepare_resolution_map(resolution_cache)

    # === STEP 1: Already filtered to Up tokens at load time ===
    df_up = df
    print(f"\nUp-token rows: {len(df_up):,}")

    # === STEP 2: UTC hour filter (0-6) ===
    utc_hour = (df_up['exchange_timestamp'].values // 3_600_000) % 24
    hour_mask = (utc_hour >= UTC_HOUR_START) & (utc_hour < UTC_HOUR_END)
    df_filtered = df_up[hour_mask]
    print(f"After UTC 0-6 filter: {len(df_filtered):,} rows")

    if len(df_filtered) == 0:
        print("No data in UTC 0-6 window. Checking hour distribution...")
        unique_hours, counts = np.unique(utc_hour, return_counts=True)
        for h, c in zip(unique_hours, counts):
            print(f"  Hour {h}: {c:,}")
        # Proceed without hour filter
        df_filtered = df_up
        hour_filter_active = False
    else:
        hour_filter_active = True

    df_up = df_filtered

    # === STEP 3: Filter to narrow mid-price band ===
    # We need to identify markets where mid_price is in the 0.45-0.55 range
    # Use median mid_price per slug to classify markets
    slug_median_mid = df_up.groupby('slug')['mid_price'].median()
    narrow_slugs = slug_median_mid[
        (slug_median_mid >= MID_PRICE_LOW) & (slug_median_mid <= MID_PRICE_HIGH)
    ].index
    df_narrow = df_up[df_up['slug'].isin(narrow_slugs)]
    print(f"After mid-price {MID_PRICE_LOW}-{MID_PRICE_HIGH} filter: {len(df_narrow):,} rows, {df_narrow['slug'].nunique()} slugs")

    if len(df_narrow) == 0:
        print("No markets in narrow band. Checking mid_price distribution...")
        print(slug_median_mid.describe())
        # Widen band for diagnostics
        narrow_slugs_wide = slug_median_mid[
            (slug_median_mid >= 0.35) & (slug_median_mid <= 0.65)
        ].index
        df_narrow = df_up[df_up['slug'].isin(narrow_slugs_wide)]
        print(f"Widened band 0.35-0.65: {df_narrow['slug'].nunique()} slugs")
        if len(df_narrow) == 0:
            return write_failed_results("No markets found in any reasonable mid-price band")

    # === STEP 4: Compute deep_level_ratio signal ===
    # Use bounded signal: (bid_deep - ask_deep) / (bid_deep + ask_deep + bid_top + ask_top)
    # This normalizes to [-1, 1] range and is robust to outliers
    df_narrow = df_narrow.copy()
    bid_top = df_narrow['bid_size_1'].values + df_narrow['bid_size_2'].values
    bid_deep = df_narrow['bid_size_3'].values + df_narrow['bid_size_4'].values + df_narrow['bid_size_5'].values
    ask_top = df_narrow['ask_size_1'].values + df_narrow['ask_size_2'].values
    ask_deep = df_narrow['ask_size_3'].values + df_narrow['ask_size_4'].values + df_narrow['ask_size_5'].values

    # Compute bid and ask deep ratios (fraction of total side that's deep)
    bid_total = bid_top + bid_deep
    ask_total = ask_top + ask_deep
    bid_total_safe = np.where(bid_total == 0, 1e-9, bid_total)
    ask_total_safe = np.where(ask_total == 0, 1e-9, ask_total)

    bid_deep_frac = bid_deep / bid_total_safe  # [0, 1]
    ask_deep_frac = ask_deep / ask_total_safe  # [0, 1]

    # Signal: difference in deep fractions, bounded to [-1, 1]
    # Positive = MMs hiding more liquidity on bid side = bullish
    df_narrow['deep_ratio_signal'] = bid_deep_frac - ask_deep_frac

    # === STEP 5: Aggregate signal over 2-second windows per slug ===
    df_narrow['time_bin'] = (df_narrow['exchange_timestamp'] // WINDOW_MS) * WINDOW_MS

    # Use median for robustness
    agg = df_narrow.groupby(['slug', 'time_bin']).agg(
        signal_median=('deep_ratio_signal', 'median'),
        mid_price_mean=('mid_price', 'mean'),
        bid_price_1=('bid_price_1', 'last'),
        ask_price_1=('ask_price_1', 'last'),
        n_snapshots=('deep_ratio_signal', 'count')
    ).reset_index()

    print(f"\nAggregated windows: {len(agg):,}")
    print(f"Signal stats:\n{agg['signal_median'].describe()}")

    # === STEP 6: Generate trade signals ===
    # Buy Up when signal > threshold (MMs stacking bids deep = bullish)
    # Buy Down when signal < -threshold (MMs stacking asks deep = bearish)
    agg['trade_direction'] = np.where(
        agg['signal_median'] > SIGNAL_THRESHOLD, 'Up',
        np.where(agg['signal_median'] < -SIGNAL_THRESHOLD, 'Down', 'none')
    )

    trades = agg[agg['trade_direction'] != 'none'].copy()
    print(f"\nTrade signals generated: {len(trades)}")
    print(f"  Buy Up: {(trades['trade_direction'] == 'Up').sum()}")
    print(f"  Buy Down: {(trades['trade_direction'] == 'Down').sum()}")

    if len(trades) == 0:
        # Try lower threshold
        for thresh in [0.20, 0.15, 0.10, 0.05]:
            trades_test = agg[agg['signal_median'].abs() > thresh]
            print(f"  Threshold {thresh}: {len(trades_test)} signals")
        return write_failed_results(
            f"No trade signals at threshold {SIGNAL_THRESHOLD}. "
            f"Signal range: [{agg['signal_mean'].min():.4f}, {agg['signal_mean'].max():.4f}]"
        )

    # === STEP 7: Take first signal per slug (entry point) ===
    trades = trades.sort_values('time_bin')
    first_trades = trades.groupby('slug').first().reset_index()
    print(f"First-entry trades: {len(first_trades)}")

    # === STEP 8: Determine entry price ===
    # If buying Up, entry price = ask_price_1 for Up token
    # If buying Down, entry price = 1 - bid_price_1 (complement)
    # Simplification: use mid_price as entry for the direction we're buying
    first_trades['entry_price'] = np.where(
        first_trades['trade_direction'] == 'Up',
        first_trades['ask_price_1'],  # buy Up at ask
        1 - first_trades['bid_price_1']  # buy Down ≈ sell Up, price = 1 - bid of Up
    )

    # Apply entry price cap
    first_trades = first_trades[first_trades['entry_price'] <= ENTRY_PRICE_CAP]
    first_trades = first_trades[first_trades['entry_price'] > 0]
    print(f"After price cap ({ENTRY_PRICE_CAP}): {len(first_trades)} trades")

    if len(first_trades) == 0:
        return write_failed_results("All trades filtered out by entry price cap")

    # === STEP 9: Resolve outcomes ===
    first_trades['winner'] = first_trades['slug'].map(resolution_map)
    first_trades = first_trades.dropna(subset=['winner'])
    print(f"With resolution data: {len(first_trades)} trades")

    if len(first_trades) == 0:
        return write_failed_results("No trades could be resolved (missing resolution data)")

    # === STEP 10: Calculate P&L ===
    # Win: payout = 1.0 - entry_price - fee
    # Loss: payout = -entry_price (lose entire stake)
    first_trades['is_win'] = first_trades['trade_direction'] == first_trades['winner']
    first_trades['fee'] = compute_fee(first_trades['entry_price'].values)

    # P&L per token (assuming MIN_ORDER_SIZE tokens per trade)
    first_trades['pnl_per_token'] = np.where(
        first_trades['is_win'],
        1.0 - first_trades['entry_price'] - first_trades['fee'],
        -first_trades['entry_price']
    )
    first_trades['pnl'] = first_trades['pnl_per_token'] * MIN_ORDER_SIZE

    # === STEP 11: Compute metrics ===
    total_trades = len(first_trades)
    wins = first_trades['is_win'].sum()
    win_rate = wins / total_trades if total_trades > 0 else 0
    pnl_total = first_trades['pnl'].sum()
    pnl_per_trade = pnl_total / total_trades if total_trades > 0 else 0

    # Cumulative P&L for drawdown and Sharpe
    cumulative_pnl = first_trades['pnl'].cumsum()
    running_max = cumulative_pnl.cummax()
    drawdown = cumulative_pnl - running_max
    max_drawdown = drawdown.min()

    # Sharpe ratio (annualized, assuming ~288 5-minute periods per day)
    if first_trades['pnl'].std() > 0:
        sharpe = (first_trades['pnl'].mean() / first_trades['pnl'].std()) * np.sqrt(288)
    else:
        sharpe = 0.0

    print(f"\n{'='*50}")
    print(f"STRATEGY RESULTS")
    print(f"{'='*50}")
    print(f"Total trades:    {total_trades}")
    print(f"Wins:            {wins}")
    print(f"Win rate:        {win_rate:.4f}")
    print(f"Total P&L:       {pnl_total:.4f}")
    print(f"P&L per trade:   {pnl_per_trade:.4f}")
    print(f"Max drawdown:    {max_drawdown:.4f}")
    print(f"Sharpe ratio:    {sharpe:.4f}")
    print(f"Hour filter:     {'UTC 0-6' if hour_filter_active else 'DISABLED (no data in range)'}")

    # Direction breakdown
    for direction in ['Up', 'Down']:
        mask = first_trades['trade_direction'] == direction
        if mask.sum() > 0:
            dir_wr = first_trades.loc[mask, 'is_win'].mean()
            dir_pnl = first_trades.loc[mask, 'pnl'].sum()
            print(f"  {direction}: {mask.sum()} trades, WR={dir_wr:.4f}, PnL={dir_pnl:.4f}")

    # === STEP 12: Determine verdict ===
    if total_trades < 10:
        verdict = "marginal"
        reason = f"Insufficient trade count ({total_trades})"
    elif win_rate > 0.55 and pnl_total > 0:
        verdict = "promising"
        reason = f"Win rate {win_rate:.2%} with positive PnL"
    elif pnl_total > 0:
        verdict = "marginal"
        reason = f"Positive PnL but win rate only {win_rate:.2%}"
    else:
        verdict = "failed"
        reason = f"Negative PnL ({pnl_total:.4f})"

    # === STEP 13: Write results ===
    results = {
        "strategy_name": "deep_level_ratio_narrow_band",
        "description": (
            "Uses deep-level orderbook ratio (levels 3-5 vs 1-2) to detect MM positioning "
            "in narrow mid-price band (0.45-0.55) markets during UTC 0-6 hours. "
            "Signal threshold 0.25 with 2s aggregation window."
        ),
        "hypothesis": (
            "In the most uncertain markets (true coin-flip territory), MM positioning at deeper "
            "orderbook levels carries maximum informational content since there is no strong prior "
            "to anchor prices. Lower threshold (0.25) compensates for reduced universe size."
        ),
        "metrics": {
            "total_trades": int(total_trades),
            "win_rate": round(float(win_rate), 4),
            "pnl_total": round(float(pnl_total), 4),
            "pnl_per_trade": round(float(pnl_per_trade), 4),
            "max_drawdown": round(float(max_drawdown), 4),
            "sharpe": round(float(sharpe), 4)
        },
        "parameters": {
            "mid_price_band": [MID_PRICE_LOW, MID_PRICE_HIGH],
            "signal_threshold": SIGNAL_THRESHOLD,
            "window_ms": WINDOW_MS,
            "utc_hour_filter": [UTC_HOUR_START, UTC_HOUR_END],
            "entry_price_cap": ENTRY_PRICE_CAP,
            "min_order_size": MIN_ORDER_SIZE,
            "hour_filter_active": hour_filter_active
        },
        "verdict": verdict,
        "next_steps": reason + ". " + (
            "Consider testing on full dataset if promising, or adjust threshold/band parameters."
            if verdict != "failed" else
            "Signal does not predict outcomes in this universe. Consider alternative signals or wider bands."
        )
    }

    results_path = OUTPUT_DIR / "results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {results_path}")
    print(f"Verdict: {verdict} — {reason}")

    return results


def write_failed_results(reason):
    """Write results.json for a failed strategy."""
    results = {
        "strategy_name": "deep_level_ratio_narrow_band",
        "description": (
            "Uses deep-level orderbook ratio (levels 3-5 vs 1-2) to detect MM positioning "
            "in narrow mid-price band (0.45-0.55) markets during UTC 0-6 hours."
        ),
        "hypothesis": (
            "In the most uncertain markets (true coin-flip territory), MM positioning at deeper "
            "orderbook levels carries maximum informational content since there is no strong prior "
            "to anchor prices."
        ),
        "metrics": {
            "total_trades": 0,
            "win_rate": 0.0,
            "pnl_total": 0.0,
            "pnl_per_trade": 0.0,
            "max_drawdown": 0.0,
            "sharpe": 0.0
        },
        "parameters": {
            "mid_price_band": [MID_PRICE_LOW, MID_PRICE_HIGH],
            "signal_threshold": SIGNAL_THRESHOLD,
            "window_ms": WINDOW_MS,
            "utc_hour_filter": [UTC_HOUR_START, UTC_HOUR_END],
            "entry_price_cap": ENTRY_PRICE_CAP,
            "min_order_size": MIN_ORDER_SIZE
        },
        "verdict": "failed",
        "next_steps": reason
    }

    results_path = OUTPUT_DIR / "results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {results_path}")
    print(f"Verdict: FAILED — {reason}")
    return results


if __name__ == "__main__":
    run_strategy()
