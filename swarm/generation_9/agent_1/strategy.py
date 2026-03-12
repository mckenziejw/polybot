"""
Strategy: Deep Level Ratio + L1 Imbalance Contrarian Filter

HYPOTHESIS: When deep liquidity (L3-5) and L1 book imbalance agree on direction,
market makers are layering visible depth to deceive — bet the OPPOSITE direction.
Restricted to first 2s window and mid-price 0.35-0.65 (uncertain markets).

The intuition: MMs layer visible depth opposite to their intended direction.
When both deep and shallow book agree, the signal is likely deceptive.
"""

import pandas as pd
import numpy as np
import json
import sys

# =============================================================================
# Configuration
# =============================================================================
SAMPLE_PATH = "data/sample_1k/book_snapshots_sample_1k.parquet"
RESOLUTION_PATH = "data/sample_1k/resolution_cache_sample.json"
OUTPUT_PATH = "swarm/generation_9/agent_1/results.json"

# Strategy parameters
ENTRY_WINDOW_MS = 2000       # Only trade in first 2 seconds
MID_PRICE_LOW = 0.35         # Uncertainty filter lower bound
MID_PRICE_HIGH = 0.65        # Uncertainty filter upper bound
MAX_ENTRY_PRICE = 0.85       # Never buy above this
MIN_ORDER_SIZE = 5           # Minimum 5 tokens
DEEP_RATIO_THRESHOLD = 0.0   # Deep ratio > 0 means deep levels favor one side
IMBALANCE_THRESHOLD = 0.0    # L1 imbalance > 0 means bids dominate

# Polymarket fee function
def calc_fee(p):
    """Fee = 0.0222 * p * 0.25 * (p*(1-p))^2"""
    return 0.0222 * p * 0.25 * (p * (1 - p)) ** 2


# =============================================================================
# Load Data
# =============================================================================
print("Loading data...")
df = pd.read_parquet(SAMPLE_PATH)
print(f"  Loaded {len(df):,} rows, {df['slug'].nunique()} markets")

with open(RESOLUTION_PATH) as f:
    resolution_cache = json.load(f)
print(f"  Loaded {len(resolution_cache)} resolutions")

# =============================================================================
# Compute Market Start Times (vectorized)
# =============================================================================
print("Computing market start times...")
# Get the earliest timestamp per slug
market_starts = df.groupby('slug')['exchange_timestamp'].min().rename('market_start')
df = df.merge(market_starts, on='slug', how='left')
df['time_since_start_ms'] = df['exchange_timestamp'] - df['market_start']

# =============================================================================
# Filter to entry window and Up tokens only
# =============================================================================
print("Filtering to entry window...")
# We analyze Up tokens — if we want to bet "Down", we buy Down tokens instead
mask_window = df['time_since_start_ms'] <= ENTRY_WINDOW_MS
mask_mid = (df['mid_price'] >= MID_PRICE_LOW) & (df['mid_price'] <= MID_PRICE_HIGH)
mask_up = df['token_label'] == 'Up'

df_up = df[mask_window & mask_mid & mask_up].copy()
print(f"  Up tokens in window with mid-price filter: {len(df_up):,} rows, {df_up['slug'].nunique()} markets")

# Also get Down tokens in same window for cross-referencing
mask_down = df['token_label'] == 'Down'
df_down = df[mask_window & mask_mid & mask_down].copy()
print(f"  Down tokens in window with mid-price filter: {len(df_down):,} rows, {df_down['slug'].nunique()} markets")

# =============================================================================
# Compute Signals (vectorized)
# =============================================================================
print("Computing signals...")

# Deep level ratio: compares deep (L3-5) bid size to deep ask size
# Positive = more deep bids (bullish depth), Negative = more deep asks (bearish depth)
for label, sub in [('up', df_up), ('down', df_down)]:
    deep_bid = sub['bid_size_3'].fillna(0) + sub['bid_size_4'].fillna(0) + sub['bid_size_5'].fillna(0)
    deep_ask = sub['ask_size_3'].fillna(0) + sub['ask_size_4'].fillna(0) + sub['ask_size_5'].fillna(0)
    deep_total = deep_bid + deep_ask
    deep_ratio = np.where(deep_total > 0, (deep_bid - deep_ask) / deep_total, 0.0)

    if label == 'up':
        df_up['deep_ratio'] = deep_ratio
        # L1 imbalance: positive = more L1 bids (bullish L1)
        df_up['l1_imbalance'] = sub['book_imbalance']
    else:
        df_down['deep_ratio'] = deep_ratio
        df_down['l1_imbalance'] = sub['book_imbalance']

# =============================================================================
# Identify Agreement Signals per Market (vectorized aggregation)
# =============================================================================
print("Aggregating signals per market...")

# For each market, compute the mean deep_ratio and mean l1_imbalance over the 2s window
# on Up tokens
up_agg = df_up.groupby('slug').agg(
    mean_deep_ratio=('deep_ratio', 'mean'),
    mean_l1_imbalance=('l1_imbalance', 'mean'),
    mean_mid_price=('mid_price', 'mean'),
    mean_ask_price_1=('ask_price_1', 'mean'),
    snapshot_count=('deep_ratio', 'count')
).reset_index()

# Similarly for Down tokens
down_agg = df_down.groupby('slug').agg(
    mean_deep_ratio_down=('deep_ratio', 'mean'),
    mean_l1_imbalance_down=('l1_imbalance', 'mean'),
    mean_mid_price_down=('mid_price', 'mean'),
    mean_ask_price_1_down=('ask_price_1', 'mean'),
    snapshot_count_down=('deep_ratio', 'count')
).reset_index()

print(f"  Up aggregated: {len(up_agg)} markets")
print(f"  Down aggregated: {len(down_agg)} markets")

# =============================================================================
# Signal Logic: Agreement = both deep_ratio and l1_imbalance point same direction
# Then CONTRARIAN: bet the opposite
# =============================================================================
# On the Up token book:
#   deep_ratio > 0 and l1_imbalance > 0 => both say "Up is strong" => contrarian says BUY DOWN
#   deep_ratio < 0 and l1_imbalance < 0 => both say "Up is weak" => contrarian says BUY UP

# Agreement: both signals have the same sign
up_agg['deep_bullish'] = up_agg['mean_deep_ratio'] > DEEP_RATIO_THRESHOLD
up_agg['l1_bullish'] = up_agg['mean_l1_imbalance'] > IMBALANCE_THRESHOLD
up_agg['agreement'] = up_agg['deep_bullish'] == up_agg['l1_bullish']

# Only trade when there's agreement
trades = up_agg[up_agg['agreement']].copy()
print(f"  Markets with deep/L1 agreement: {len(trades)}")

# Contrarian direction: if both bullish on Up => buy Down, if both bearish on Up => buy Up
trades['buy_token'] = np.where(trades['deep_bullish'], 'Down', 'Up')

# =============================================================================
# Merge Down token prices for Down bets
# =============================================================================
trades = trades.merge(down_agg[['slug', 'mean_ask_price_1_down']], on='slug', how='left')

# Entry price: ask_price_1 for the token we're buying
trades['entry_price'] = np.where(
    trades['buy_token'] == 'Up',
    trades['mean_ask_price_1'],
    trades['mean_ask_price_1_down']
)

# Filter: entry price must be <= MAX_ENTRY_PRICE and > 0
trades = trades[(trades['entry_price'] <= MAX_ENTRY_PRICE) & (trades['entry_price'] > 0)].copy()
print(f"  After entry price filter (<= {MAX_ENTRY_PRICE}): {len(trades)}")

# =============================================================================
# Resolve Outcomes
# =============================================================================
print("Resolving outcomes...")

def get_winner(slug):
    """Get winner label from resolution cache."""
    entry = resolution_cache.get(slug)
    if entry is None:
        return None
    if not isinstance(entry, dict):
        return None
    # Direct winnerLabel
    if entry.get('winnerLabel'):
        return entry['winnerLabel']
    # Resolve via winningTokenId + clobTokenIds + outcomes
    wt = entry.get('winningTokenId')
    outcomes = entry.get('outcomes')
    token_ids = entry.get('clobTokenIds')
    if wt and outcomes and token_ids and len(outcomes) == len(token_ids):
        for i, tid in enumerate(token_ids):
            if tid == wt:
                return outcomes[i]
    return None

trades['winner'] = trades['slug'].map(get_winner)
trades = trades.dropna(subset=['winner'])
print(f"  Markets with resolution: {len(trades)}")

# Did we win?
trades['won'] = trades['buy_token'] == trades['winner']

# =============================================================================
# PnL Calculation
# =============================================================================
print("Calculating PnL...")

# If we win: payout = 1.0, profit = 1.0 - entry_price - fee(entry_price)
# If we lose: payout = 0.0, profit = -entry_price - fee(entry_price)
# We buy MIN_ORDER_SIZE tokens at entry_price each
p = trades['entry_price'].values
trades['fee_per_token'] = 0.0222 * p * 0.25 * (p * (1 - p)) ** 2
trades['cost_per_token'] = trades['entry_price'] + trades['fee_per_token']
trades['pnl_per_token'] = np.where(
    trades['won'],
    1.0 - trades['cost_per_token'],
    -trades['cost_per_token']
)
trades['pnl'] = trades['pnl_per_token'] * MIN_ORDER_SIZE

# =============================================================================
# Metrics
# =============================================================================
total_trades = len(trades)
wins = trades['won'].sum()
win_rate = wins / total_trades if total_trades > 0 else 0.0
pnl_total = trades['pnl'].sum()
pnl_per_trade = pnl_total / total_trades if total_trades > 0 else 0.0

# Max drawdown
cumulative_pnl = trades['pnl'].cumsum()
running_max = cumulative_pnl.cummax()
drawdown = cumulative_pnl - running_max
max_drawdown = drawdown.min() if len(drawdown) > 0 else 0.0

# Sharpe (annualized assuming ~288 5-min periods per day)
if total_trades > 1:
    daily_pnl = trades['pnl'].values
    sharpe = (daily_pnl.mean() / daily_pnl.std()) * np.sqrt(288) if daily_pnl.std() > 0 else 0.0
else:
    sharpe = 0.0

# =============================================================================
# Analysis by Signal Strength
# =============================================================================
print("\n" + "=" * 70)
print("STRATEGY RESULTS: Deep Ratio + L1 Imbalance Contrarian")
print("=" * 70)
print(f"Total trades:    {total_trades}")
print(f"Wins:            {wins}")
print(f"Win rate:        {win_rate:.4f}")
print(f"Total PnL:       {pnl_total:.2f}")
print(f"PnL per trade:   {pnl_per_trade:.4f}")
print(f"Max drawdown:    {max_drawdown:.2f}")
print(f"Sharpe ratio:    {sharpe:.4f}")

# Breakdown by direction
print("\n--- By Bet Direction ---")
for direction in ['Up', 'Down']:
    sub = trades[trades['buy_token'] == direction]
    if len(sub) > 0:
        wr = sub['won'].mean()
        pnl = sub['pnl'].sum()
        print(f"  {direction}: {len(sub)} trades, win_rate={wr:.4f}, pnl={pnl:.2f}")

# Breakdown by signal strength (quintiles of deep_ratio magnitude)
print("\n--- By Deep Ratio Magnitude (quintiles) ---")
trades['deep_ratio_abs'] = trades['mean_deep_ratio'].abs()
if len(trades) >= 5:
    trades['dr_quintile'] = pd.qcut(trades['deep_ratio_abs'], 5, labels=False, duplicates='drop')
    for q in sorted(trades['dr_quintile'].unique()):
        sub = trades[trades['dr_quintile'] == q]
        wr = sub['won'].mean()
        pnl = sub['pnl'].sum()
        avg_dr = sub['deep_ratio_abs'].mean()
        print(f"  Q{q}: {len(sub)} trades, win_rate={wr:.4f}, pnl={pnl:.2f}, avg_|deep_ratio|={avg_dr:.4f}")

# Breakdown by mid-price bands
print("\n--- By Mid-Price Band ---")
bins = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]
trades['mp_band'] = pd.cut(trades['mean_mid_price'], bins=bins)
for band in trades['mp_band'].cat.categories:
    sub = trades[trades['mp_band'] == band]
    if len(sub) > 0:
        wr = sub['won'].mean()
        pnl = sub['pnl'].sum()
        print(f"  {band}: {len(sub)} trades, win_rate={wr:.4f}, pnl={pnl:.2f}")

# Entry price distribution
print(f"\n--- Entry Price Stats ---")
print(f"  Mean:   {trades['entry_price'].mean():.4f}")
print(f"  Median: {trades['entry_price'].median():.4f}")
print(f"  Std:    {trades['entry_price'].std():.4f}")

# =============================================================================
# Sensitivity: try different thresholds for deep_ratio
# =============================================================================
print("\n--- Sensitivity: Deep Ratio Magnitude Threshold ---")
for dr_thresh in [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]:
    sub = trades[trades['deep_ratio_abs'] >= dr_thresh]
    if len(sub) > 0:
        wr = sub['won'].mean()
        pnl = sub['pnl'].sum()
        ppt = pnl / len(sub)
        print(f"  |deep_ratio| >= {dr_thresh:.2f}: {len(sub)} trades, win_rate={wr:.4f}, pnl={pnl:.2f}, pnl/trade={ppt:.4f}")

# =============================================================================
# Determine verdict
# =============================================================================
if total_trades < 20:
    verdict = "failed"
    verdict_reason = "Insufficient trades for statistical significance"
elif win_rate > 0.55 and pnl_total > 0:
    verdict = "promising"
    verdict_reason = f"Win rate {win_rate:.1%} with positive PnL"
elif win_rate > 0.52 and pnl_total > 0:
    verdict = "marginal"
    verdict_reason = f"Modest edge: {win_rate:.1%} win rate"
else:
    verdict = "failed"
    verdict_reason = f"No edge detected: {win_rate:.1%} win rate, PnL={pnl_total:.2f}"

print(f"\nVERDICT: {verdict} — {verdict_reason}")

# =============================================================================
# Write Results
# =============================================================================
results = {
    "strategy_name": "deep_ratio_l1_contrarian",
    "description": (
        "Contrarian strategy combining deep-level ratio (L3-5 bid/ask imbalance) "
        "with L1 book imbalance. When both signals agree on direction (both bullish "
        "or both bearish), bet the OPPOSITE direction. Theory: MMs layer visible "
        "depth deceptively, so agreement between deep and shallow book indicates "
        "a trap. Restricted to first 2s and mid-price 0.35-0.65."
    ),
    "hypothesis": (
        "Agreement between deep liquidity (L3-5) and L1 imbalance is deceptive — "
        "MMs layer visible depth opposite to intended direction. Combining this "
        "contrarian insight with the uncertainty filter (mid 0.35-0.65) should "
        "isolate the highest-conviction subset of deceptive signals."
    ),
    "metrics": {
        "total_trades": int(total_trades),
        "win_rate": round(float(win_rate), 4),
        "pnl_total": round(float(pnl_total), 2),
        "pnl_per_trade": round(float(pnl_per_trade), 4),
        "max_drawdown": round(float(max_drawdown), 2),
        "sharpe": round(float(sharpe), 4)
    },
    "parameters": {
        "entry_window_ms": ENTRY_WINDOW_MS,
        "mid_price_low": MID_PRICE_LOW,
        "mid_price_high": MID_PRICE_HIGH,
        "max_entry_price": MAX_ENTRY_PRICE,
        "min_order_size": MIN_ORDER_SIZE,
        "deep_ratio_threshold": DEEP_RATIO_THRESHOLD,
        "imbalance_threshold": IMBALANCE_THRESHOLD
    },
    "verdict": verdict,
    "next_steps": (
        "If promising: test on full dataset, optimize deep_ratio threshold, "
        "explore time-weighting (later snapshots in 2s window may be more informative), "
        "test narrower mid-price bands around 0.50. "
        "If failed: the contrarian signal may not survive fees, or the 2s window "
        "may be too noisy for this signal combination."
    )
}

with open(OUTPUT_PATH, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nResults written to {OUTPUT_PATH}")
