"""
Strategy: Deep Level Ratio vs L1 Contrarian Disagreement

Hypothesis: When deep orderbook levels (L3-5) and L1 book imbalance DISAGREE
on direction, the deep level signal reflects genuine MM positioning while L1
is deceptive layering. Trade in the direction of deep levels when they
contradict surface liquidity.

Parameters:
- Window: 2 seconds (2000ms) for signal averaging
- Deep level threshold: 0.3 (|deep_ratio| must exceed this)
- Mid-price filter: 0.35-0.65 (avoid extreme prices)
- Entry price cap: 0.85
- Disagreement: deep_ratio and L1 imbalance must have opposite signs
"""

import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

# ── Config ──────────────────────────────────────────────────────────────
WINDOW_MS = 2000
DEEP_THRESHOLD = 0.3
MID_PRICE_LO = 0.35
MID_PRICE_HI = 0.65
MAX_ENTRY_PRICE = 0.85
MIN_ORDER_SIZE = 5

DATA_PATH = 'data/sample_1k/book_snapshots_sample_1k.parquet'
RESOLUTION_PATH = 'data/sample_1k/resolution_cache_sample.json'
OUTPUT_DIR = 'swarm/generation_11/agent_1'

# ── Fee calculation ─────────────────────────────────────────────────────
def calc_fee(p):
    """Polymarket fee: 0.0222 * p * 0.25 * (p*(1-p))^2"""
    return 0.0222 * p * 0.25 * (p * (1 - p)) ** 2


# ── Load resolution cache ──────────────────────────────────────────────
def load_resolutions(path):
    """Map slug -> winner label (Up/Down)"""
    with open(path) as f:
        rc = json.load(f)
    return rc


def resolve_winner_label(rc_entry, slug, asset_id_map):
    """Determine winner label from various resolution cache formats."""
    if 'winnerLabel' in rc_entry:
        return rc_entry['winnerLabel']

    winning_id = rc_entry.get('winningTokenId') or rc_entry.get('winning_token')
    if winning_id and slug in asset_id_map:
        return asset_id_map[slug].get(winning_id)

    return None


# ── Main strategy ───────────────────────────────────────────────────────
def run_strategy():
    print("Loading data...")
    df = pd.read_parquet(DATA_PATH)
    rc = load_resolutions(RESOLUTION_PATH)

    # Build asset_id -> token_label mapping per slug
    print("Building asset_id map...")
    id_map_df = df[['slug', 'asset_id', 'token_label']].drop_duplicates()
    asset_id_map = {}
    for slug in id_map_df['slug'].unique():
        sub = id_map_df[id_map_df['slug'] == slug]
        asset_id_map[slug] = dict(zip(sub['asset_id'], sub['token_label']))

    # Build slug -> winner mapping
    slug_winner = {}
    for slug, entry in rc.items():
        label = resolve_winner_label(entry, slug, asset_id_map)
        if label in ('Up', 'Down'):
            slug_winner[slug] = label

    print(f"Resolved {len(slug_winner)} / {len(rc)} markets")

    # Filter to Up tokens only (avoid double-counting)
    print("Filtering to Up tokens...")
    df = df[df['token_label'] == 'Up'].copy()

    # Filter markets with known resolution
    df = df[df['slug'].isin(slug_winner)]
    print(f"Working with {df['slug'].nunique()} markets, {len(df)} rows")

    # ── Compute signals (vectorized) ────────────────────────────────
    print("Computing signals...")

    # Deep level sizes (L3-5) for bid and ask
    df['deep_bid'] = df['bid_size_3'] + df['bid_size_4'] + df['bid_size_5']
    df['deep_ask'] = df['ask_size_3'] + df['ask_size_4'] + df['ask_size_5']
    df['deep_total'] = df['deep_bid'] + df['deep_ask']

    # Deep level ratio: positive = more deep bid support (bullish for Up)
    df['deep_ratio'] = np.where(
        df['deep_total'] > 0,
        (df['deep_bid'] - df['deep_ask']) / df['deep_total'],
        0.0
    )

    # L1 imbalance: positive = more bid at L1 (bullish for Up)
    df['l1_total'] = df['bid_size_1'] + df['ask_size_1']
    df['l1_imbalance'] = np.where(
        df['l1_total'] > 0,
        (df['bid_size_1'] - df['ask_size_1']) / df['l1_total'],
        0.0
    )

    # ── Apply mid-price filter ──────────────────────────────────────
    df = df[(df['mid_price'] >= MID_PRICE_LO) & (df['mid_price'] <= MID_PRICE_HI)]
    print(f"After mid-price filter: {df['slug'].nunique()} markets, {len(df)} rows")

    # ── Average signals over 2s window per market ───────────────────
    print("Computing 2s window averages...")

    # Sort by slug and timestamp
    df = df.sort_values(['slug', 'exchange_timestamp'])

    # Get the start timestamp per market for the 2s window
    market_start = df.groupby('slug')['exchange_timestamp'].transform('min')
    df['time_offset'] = df['exchange_timestamp'] - market_start

    # Keep only first 2s window
    df_window = df[df['time_offset'] <= WINDOW_MS].copy()
    print(f"2s window: {df_window['slug'].nunique()} markets, {len(df_window)} rows")

    # Compute average signals per market in the window
    signals = df_window.groupby('slug').agg(
        avg_deep_ratio=('deep_ratio', 'mean'),
        avg_l1_imbalance=('l1_imbalance', 'mean'),
        avg_mid_price=('mid_price', 'mean'),
        snapshot_count=('deep_ratio', 'count'),
    ).reset_index()

    print(f"Markets with signals: {len(signals)}")

    # ── Generate trade signals ──────────────────────────────────────
    # Disagreement: deep_ratio and l1_imbalance have opposite signs
    signals['disagree'] = (signals['avg_deep_ratio'] * signals['avg_l1_imbalance']) < 0

    # Deep ratio must exceed threshold
    signals['deep_strong'] = signals['avg_deep_ratio'].abs() > DEEP_THRESHOLD

    # Trade filter: disagreement AND strong deep signal
    signals['trade'] = signals['disagree'] & signals['deep_strong']

    # Direction: follow deep levels
    # deep_ratio > 0 means deep bids > deep asks -> bullish -> buy Up token
    # deep_ratio < 0 means deep asks > deep bids -> bearish -> buy Down token
    signals['buy_up'] = signals['avg_deep_ratio'] > 0

    trades = signals[signals['trade']].copy()
    print(f"\nTrade signals: {len(trades)}")
    print(f"  Buy Up: {trades['buy_up'].sum()}")
    print(f"  Buy Down: {(~trades['buy_up']).sum()}")

    if len(trades) == 0:
        print("No trades generated!")
        return write_results(0, 0, 0, 0, 0, 0, signals)

    # ── Determine entry prices and outcomes ─────────────────────────
    # Get entry price from the avg mid_price in the window
    # If buying Up, entry = avg_mid_price(Up)
    # If buying Down, entry = 1 - avg_mid_price(Up) [complement]
    trades['entry_price'] = np.where(
        trades['buy_up'],
        trades['avg_mid_price'],
        1 - trades['avg_mid_price']
    )

    # Apply entry price cap
    trades = trades[trades['entry_price'] <= MAX_ENTRY_PRICE]
    print(f"After price cap: {len(trades)} trades")

    if len(trades) == 0:
        print("No trades after price cap!")
        return write_results(0, 0, 0, 0, 0, 0, signals)

    # Determine outcome
    trades['winner'] = trades['slug'].map(slug_winner)
    trades['correct'] = np.where(
        trades['buy_up'],
        trades['winner'] == 'Up',
        trades['winner'] == 'Down'
    )

    # ── PnL calculation ─────────────────────────────────────────────
    # Buy at entry_price, payout is 1.00 if correct, 0.00 if wrong
    trades['fee'] = calc_fee(trades['entry_price'])
    trades['pnl'] = np.where(
        trades['correct'],
        1.0 - trades['entry_price'] - trades['fee'],   # win
        0.0 - trades['entry_price'] - trades['fee']     # lose
    )

    total_trades = len(trades)
    wins = trades['correct'].sum()
    win_rate = wins / total_trades
    pnl_total = trades['pnl'].sum()
    pnl_per_trade = trades['pnl'].mean()

    # Max drawdown
    cumulative = trades['pnl'].cumsum()
    running_max = cumulative.cummax()
    drawdown = cumulative - running_max
    max_drawdown = drawdown.min()

    # Sharpe (annualized assuming ~288 5-min periods per day)
    if trades['pnl'].std() > 0:
        sharpe = (trades['pnl'].mean() / trades['pnl'].std()) * np.sqrt(288)
    else:
        sharpe = 0.0

    print(f"\n{'='*50}")
    print(f"RESULTS")
    print(f"{'='*50}")
    print(f"Total trades:  {total_trades}")
    print(f"Win rate:      {win_rate:.4f}")
    print(f"PnL total:     {pnl_total:.4f}")
    print(f"PnL per trade: {pnl_per_trade:.4f}")
    print(f"Max drawdown:  {max_drawdown:.4f}")
    print(f"Sharpe:        {sharpe:.4f}")

    # Distribution analysis
    print(f"\nEntry price stats:")
    print(trades['entry_price'].describe())
    print(f"\nDeep ratio stats for trades:")
    print(trades['avg_deep_ratio'].describe())

    write_results(total_trades, win_rate, pnl_total, pnl_per_trade, max_drawdown, sharpe, signals)

    # ── Sensitivity analysis ────────────────────────────────────────
    print(f"\n{'='*50}")
    print("SENSITIVITY ANALYSIS")
    print(f"{'='*50}")

    for thresh in [0.1, 0.2, 0.3, 0.4, 0.5]:
        s = signals.copy()
        s['trade'] = s['disagree'] & (s['avg_deep_ratio'].abs() > thresh)
        s['buy_up'] = s['avg_deep_ratio'] > 0
        t = s[s['trade']].copy()
        if len(t) == 0:
            print(f"Threshold {thresh}: 0 trades")
            continue
        t['entry_price'] = np.where(t['buy_up'], t['avg_mid_price'], 1 - t['avg_mid_price'])
        t = t[t['entry_price'] <= MAX_ENTRY_PRICE]
        if len(t) == 0:
            print(f"Threshold {thresh}: 0 trades after price cap")
            continue
        t['winner'] = t['slug'].map(slug_winner)
        t['correct'] = np.where(t['buy_up'], t['winner'] == 'Up', t['winner'] == 'Down')
        t['fee'] = calc_fee(t['entry_price'])
        t['pnl'] = np.where(t['correct'], 1.0 - t['entry_price'] - t['fee'], -t['entry_price'] - t['fee'])
        wr = t['correct'].mean()
        ppt = t['pnl'].mean()
        print(f"Threshold {thresh:.1f}: {len(t)} trades, WR={wr:.4f}, PnL/trade={ppt:.4f}, total={t['pnl'].sum():.4f}")

    # Also test different windows
    print(f"\nWindow sensitivity (threshold={DEEP_THRESHOLD}):")
    for window_ms in [500, 1000, 2000, 5000, 10000]:
        dfw = df[df['time_offset'] <= window_ms]
        if len(dfw) == 0:
            continue
        sig = dfw.groupby('slug').agg(
            avg_deep_ratio=('deep_ratio', 'mean'),
            avg_l1_imbalance=('l1_imbalance', 'mean'),
            avg_mid_price=('mid_price', 'mean'),
        ).reset_index()
        sig['disagree'] = (sig['avg_deep_ratio'] * sig['avg_l1_imbalance']) < 0
        sig['deep_strong'] = sig['avg_deep_ratio'].abs() > DEEP_THRESHOLD
        sig['trade'] = sig['disagree'] & sig['deep_strong']
        sig['buy_up'] = sig['avg_deep_ratio'] > 0
        t = sig[sig['trade']].copy()
        if len(t) == 0:
            print(f"Window {window_ms}ms: 0 trades")
            continue
        t['entry_price'] = np.where(t['buy_up'], t['avg_mid_price'], 1 - t['avg_mid_price'])
        t = t[t['entry_price'] <= MAX_ENTRY_PRICE]
        if len(t) == 0:
            continue
        t['winner'] = t['slug'].map(slug_winner)
        t['correct'] = np.where(t['buy_up'], t['winner'] == 'Up', t['winner'] == 'Down')
        t['fee'] = calc_fee(t['entry_price'])
        t['pnl'] = np.where(t['correct'], 1.0 - t['entry_price'] - t['fee'], -t['entry_price'] - t['fee'])
        wr = t['correct'].mean()
        ppt = t['pnl'].mean()
        print(f"Window {window_ms}ms: {len(t)} trades, WR={wr:.4f}, PnL/trade={ppt:.4f}, total={t['pnl'].sum():.4f}")


def write_results(total_trades, win_rate, pnl_total, pnl_per_trade, max_drawdown, sharpe, signals):
    # Determine verdict
    if total_trades >= 20 and pnl_per_trade > 0.01 and win_rate > 0.52:
        verdict = "promising"
    elif total_trades >= 10 and pnl_per_trade > 0:
        verdict = "marginal"
    else:
        verdict = "failed"

    results = {
        "strategy_name": "deep_level_l1_contrarian_disagreement",
        "description": "Trade when deep orderbook levels (L3-5) disagree with L1 imbalance direction. Use deep level signal as trade direction, hypothesizing L1 is deceptive MM layering.",
        "hypothesis": "When deep levels (L3-5) and L1 book imbalance DISAGREE on direction, deep levels reflect genuine MM positioning while L1 is noise/deception. Disagreement isolates real signal from layering.",
        "metrics": {
            "total_trades": int(total_trades),
            "win_rate": round(float(win_rate), 4),
            "pnl_total": round(float(pnl_total), 4),
            "pnl_per_trade": round(float(pnl_per_trade), 4),
            "max_drawdown": round(float(max_drawdown), 4),
            "sharpe": round(float(sharpe), 4),
        },
        "parameters": {
            "window_ms": WINDOW_MS,
            "deep_threshold": DEEP_THRESHOLD,
            "mid_price_range": [MID_PRICE_LO, MID_PRICE_HI],
            "max_entry_price": MAX_ENTRY_PRICE,
            "signal": "deep_ratio = (deep_bid_L3-5 - deep_ask_L3-5) / total_deep",
            "filter": "disagreement between deep_ratio sign and L1 imbalance sign",
        },
        "verdict": verdict,
        "next_steps": "",
    }

    if verdict == "promising":
        results["next_steps"] = "Test on full dataset. Explore tighter thresholds and time-of-day effects."
    elif verdict == "marginal":
        results["next_steps"] = "Investigate why edge is thin. Try combining with spread filter or volume weighting."
    else:
        results["next_steps"] = "Disagreement between deep and L1 levels does not produce reliable alpha on this sample. Consider: 1) the disagreement signal may be noise, 2) L1 may not be deceptive in most markets, 3) deep levels may also be manipulated."

    with open(f'{OUTPUT_DIR}/results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {OUTPUT_DIR}/results.json")
    print(f"Verdict: {verdict}")


if __name__ == '__main__':
    run_strategy()
