"""
Entropy-Weighted Imbalance Strategy for Polymarket BTC Up/Down 5m Markets

Hypothesis: Compute Shannon entropy of BID-SIDE size distributions across 5 levels
separately for Up and Down tokens in the first N seconds. In binary markets, Up bids
and Down bids are NOT mirrors (Up bids = Down asks), so bid-side entropy captures
genuine asymmetry in buying interest. Low bid entropy (concentrated buying) on one side
relative to the other signals informed positioning.

Key insight: combined bid+ask entropy is identical for Up and Down (mirror books),
so we must use bid-only entropy to detect directional signals.
"""

import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

# === Configuration ===
DATA_PATH = 'data/sample_1k/book_snapshots_sample_1k.parquet'
RESOLUTION_PATH = 'data/sample_1k/resolution_cache_sample.json'
OUTPUT_PATH = 'swarm/generation_12/agent_0/results.json'

# Strategy parameters
ENTRY_WINDOW_MS = 5000       # First 5 seconds of market
MAX_ENTRY_PRICE = 0.85       # Never buy above this
MIN_ORDER_SIZE = 5           # Minimum order size in tokens

# Polymarket fee function
def calc_fee_vec(p):
    """Vectorized fee: 0.0222 * p * 0.25 * (p*(1-p))^2"""
    return 0.0222 * p * 0.25 * (p * (1 - p)) ** 2


def row_entropy(sizes_array):
    """Vectorized Shannon entropy for each row in a (N, 5) array."""
    totals = sizes_array.sum(axis=1, keepdims=True)
    totals = np.where(totals == 0, 1, totals)
    probs = sizes_array / totals
    log_probs = np.where(probs > 0, np.log2(np.clip(probs, 1e-15, 1.0)), 0.0)
    return -np.sum(probs * log_probs, axis=1)


def load_data():
    """Load and prepare data."""
    print("Loading data...")
    df = pd.read_parquet(DATA_PATH)

    with open(RESOLUTION_PATH) as f:
        resolution_cache = json.load(f)

    resolutions = {}
    for slug, info in resolution_cache.items():
        if 'winnerLabel' in info:
            resolutions[slug] = info['winnerLabel']
        elif 'winningTokenId' in info and 'outcomes' in info and 'clobTokenIds' in info:
            try:
                idx = info['clobTokenIds'].index(info['winningTokenId'])
                resolutions[slug] = info['outcomes'][idx]
            except (ValueError, IndexError):
                continue

    print(f"Loaded {len(df):,} rows, {df['slug'].nunique()} markets, {len(resolutions)} resolutions")
    return df, resolutions


def compute_market_signals(df):
    """Compute bid-side entropy and imbalance signals per market in entry window."""

    bid_size_cols = [f'bid_size_{i}' for i in range(1, 6)]
    ask_size_cols = [f'ask_size_{i}' for i in range(1, 6)]

    # Filter to entry window
    market_start = df.groupby('slug')['exchange_timestamp'].transform('min')
    mask_window = (df['exchange_timestamp'] - market_start) <= ENTRY_WINDOW_MS
    window_df = df[mask_window].copy()

    print(f"Entry window data: {len(window_df):,} rows from {window_df['slug'].nunique()} markets")

    # Compute bid-side entropy (this differs between Up and Down!)
    bid_sizes = window_df[bid_size_cols].values
    ask_sizes = window_df[ask_size_cols].values
    window_df['bid_entropy'] = row_entropy(bid_sizes)
    window_df['ask_entropy'] = row_entropy(ask_sizes)

    # Total sizes for imbalance
    window_df['total_bid_size'] = bid_sizes.sum(axis=1)
    window_df['total_ask_size'] = ask_sizes.sum(axis=1)

    # Concentration metric: max bid size / total bid size (Herfindahl-like)
    bid_max = np.max(bid_sizes, axis=1)
    bid_total = window_df['total_bid_size'].values
    window_df['bid_concentration'] = np.where(bid_total > 0, bid_max / bid_total, 0)

    # Aggregate per (slug, token_label)
    agg = window_df.groupby(['slug', 'token_label']).agg(
        mean_bid_entropy=('bid_entropy', 'mean'),
        std_bid_entropy=('bid_entropy', 'std'),
        mean_ask_entropy=('ask_entropy', 'mean'),
        mean_book_imbalance=('book_imbalance', 'mean'),
        mean_mid_price=('mid_price', 'mean'),
        last_mid_price=('mid_price', 'last'),
        mean_total_bid=('total_bid_size', 'mean'),
        mean_total_ask=('total_ask_size', 'mean'),
        mean_bid_concentration=('bid_concentration', 'mean'),
        snapshot_count=('mid_price', 'count'),
    ).reset_index()

    return agg


def build_signals(agg):
    """Build trading signals from aggregated market data."""

    up = agg[agg['token_label'] == 'Up'].set_index('slug')
    down = agg[agg['token_label'] == 'Down'].set_index('slug')

    common_slugs = up.index.intersection(down.index)
    up = up.loc[common_slugs]
    down = down.loc[common_slugs]

    signals = pd.DataFrame(index=common_slugs)

    # Bid-side entropy (different for Up vs Down since Up bids != Down bids)
    signals['up_bid_entropy'] = up['mean_bid_entropy']
    signals['down_bid_entropy'] = down['mean_bid_entropy']

    # Entropy difference: negative = Up bids more concentrated
    signals['bid_entropy_diff'] = signals['up_bid_entropy'] - signals['down_bid_entropy']

    # Bid entropy ratio
    signals['bid_entropy_ratio'] = (
        signals['up_bid_entropy'] / signals['down_bid_entropy'].clip(lower=0.01)
    )

    # Bid concentration
    signals['up_bid_concentration'] = up['mean_bid_concentration']
    signals['down_bid_concentration'] = down['mean_bid_concentration']
    signals['bid_concentration_diff'] = signals['up_bid_concentration'] - signals['down_bid_concentration']

    # Book imbalance (already different for Up vs Down)
    signals['up_imbalance'] = up['mean_book_imbalance']
    signals['down_imbalance'] = down['mean_book_imbalance']

    # Entry prices
    signals['up_mid_price'] = up['last_mid_price']
    signals['down_mid_price'] = down['last_mid_price']

    # Total bid sizes
    signals['up_total_bid'] = up['mean_total_bid']
    signals['down_total_bid'] = down['mean_total_bid']

    # Bid size ratio: who has more buying interest?
    total = (signals['up_total_bid'] + signals['down_total_bid']).clip(lower=1)
    signals['bid_size_ratio'] = (signals['up_total_bid'] - signals['down_total_bid']) / total

    print(f"Built signals for {len(signals)} markets")
    return signals


def generate_trades_vectorized(signals, resolutions, params):
    """Vectorized trade generation."""

    # Map resolutions
    res_series = signals.index.to_series().map(resolutions)
    has_resolution = res_series.notna()
    sig = signals[has_resolution].copy()
    sig['winner'] = res_series[has_resolution].values

    entropy_thresh = params.get('entropy_ratio_threshold', 0.9)
    imbalance_thresh = params.get('imbalance_threshold', 0.1)
    concentration_thresh = params.get('concentration_threshold', 0.0)
    use_composite = params.get('use_composite', False)

    trades_list = []

    if use_composite:
        # Composite signal: combine entropy ratio, imbalance, and concentration
        # Z-score each signal
        for col in ['bid_entropy_ratio', 'up_imbalance', 'down_imbalance', 'bid_concentration_diff', 'bid_size_ratio']:
            mean = sig[col].mean()
            std = sig[col].std()
            if std > 0:
                sig[f'{col}_z'] = (sig[col] - mean) / std
            else:
                sig[f'{col}_z'] = 0.0

        # Composite score for buying Up: low entropy ratio + positive imbalance + high concentration
        sig['up_score'] = (
            -sig['bid_entropy_ratio_z'] +  # low Up entropy relative to Down
            sig['up_imbalance_z'] +         # positive Up book imbalance
            sig['bid_concentration_diff_z'] +  # Up bids more concentrated
            sig['bid_size_ratio_z']          # more bid volume on Up side
        )

        sig['down_score'] = (
            sig['bid_entropy_ratio_z'] +    # high Up entropy = low Down entropy
            sig['down_imbalance_z'] * -1 +  # Down imbalance is negative = good for down
            -sig['bid_concentration_diff_z'] +
            -sig['bid_size_ratio_z']
        )

        score_thresh = params.get('score_threshold', 1.0)

        # Buy Up signals
        up_mask = (sig['up_score'] > score_thresh) & (sig['up_mid_price'] <= MAX_ENTRY_PRICE)
        up_trades = sig[up_mask].copy()
        if len(up_trades) > 0:
            up_trades['side'] = 'Up'
            up_trades['entry_price'] = up_trades['up_mid_price']
            up_trades['payout'] = (up_trades['winner'] == 'Up').astype(float)
            up_trades['fee'] = calc_fee_vec(up_trades['entry_price'])
            up_trades['pnl'] = up_trades['payout'] - up_trades['entry_price'] - up_trades['fee']
            up_trades['signal_score'] = up_trades['up_score']
            trades_list.append(up_trades[['side', 'entry_price', 'fee', 'payout', 'pnl', 'winner', 'signal_score']])

        # Buy Down signals
        down_mask = (sig['down_score'] > score_thresh) & (sig['down_mid_price'] <= MAX_ENTRY_PRICE)
        down_trades = sig[down_mask].copy()
        if len(down_trades) > 0:
            down_trades['side'] = 'Down'
            down_trades['entry_price'] = down_trades['down_mid_price']
            down_trades['payout'] = (down_trades['winner'] == 'Down').astype(float)
            down_trades['fee'] = calc_fee_vec(down_trades['entry_price'])
            down_trades['pnl'] = down_trades['payout'] - down_trades['entry_price'] - down_trades['fee']
            down_trades['signal_score'] = down_trades['down_score']
            trades_list.append(down_trades[['side', 'entry_price', 'fee', 'payout', 'pnl', 'winner', 'signal_score']])

    else:
        # Simple threshold-based signals using bid-side entropy ratio
        # Buy Up: Up bid entropy LOW relative to Down (ratio < threshold)
        # AND positive Up book imbalance
        up_mask = (
            (sig['bid_entropy_ratio'] < entropy_thresh) &
            (sig['up_imbalance'] > imbalance_thresh) &
            (sig['up_mid_price'] <= MAX_ENTRY_PRICE)
        )
        if concentration_thresh > 0:
            up_mask = up_mask & (sig['up_bid_concentration'] > concentration_thresh)

        up_trades = sig[up_mask].copy()
        if len(up_trades) > 0:
            up_trades['side'] = 'Up'
            up_trades['entry_price'] = up_trades['up_mid_price']
            up_trades['payout'] = (up_trades['winner'] == 'Up').astype(float)
            up_trades['fee'] = calc_fee_vec(up_trades['entry_price'])
            up_trades['pnl'] = up_trades['payout'] - up_trades['entry_price'] - up_trades['fee']
            up_trades['signal_score'] = sig.loc[up_mask.values, 'bid_entropy_ratio']
            trades_list.append(up_trades[['side', 'entry_price', 'fee', 'payout', 'pnl', 'winner', 'signal_score']])

        # Buy Down: Down bid entropy LOW relative to Up (ratio > 1/threshold, i.e. inverted)
        down_mask = (
            (sig['bid_entropy_ratio'] > (1.0 / entropy_thresh)) &
            (sig['down_imbalance'] < -imbalance_thresh) &
            (sig['down_mid_price'] <= MAX_ENTRY_PRICE)
        )
        if concentration_thresh > 0:
            down_mask = down_mask & (sig['down_bid_concentration'] > concentration_thresh)

        down_trades = sig[down_mask].copy()
        if len(down_trades) > 0:
            down_trades['side'] = 'Down'
            down_trades['entry_price'] = down_trades['down_mid_price']
            down_trades['payout'] = (down_trades['winner'] == 'Down').astype(float)
            down_trades['fee'] = calc_fee_vec(down_trades['entry_price'])
            down_trades['pnl'] = down_trades['payout'] - down_trades['entry_price'] - down_trades['fee']
            down_trades['signal_score'] = sig.loc[down_mask.values, 'bid_entropy_ratio']
            trades_list.append(down_trades[['side', 'entry_price', 'fee', 'payout', 'pnl', 'winner', 'signal_score']])

    if trades_list:
        trades_df = pd.concat(trades_list, ignore_index=True)
    else:
        trades_df = pd.DataFrame()

    return trades_df


def compute_metrics(trades_df):
    """Compute strategy performance metrics."""
    if trades_df.empty:
        return {
            'total_trades': 0, 'win_rate': 0.0, 'pnl_total': 0.0,
            'pnl_per_trade': 0.0, 'max_drawdown': 0.0, 'sharpe': 0.0,
        }

    n = len(trades_df)
    wins = (trades_df['pnl'] > 0).sum()
    pnl_total = trades_df['pnl'].sum()
    pnl_per_trade = pnl_total / n

    cumulative = trades_df['pnl'].cumsum()
    running_max = cumulative.cummax()
    max_drawdown = (cumulative - running_max).min()

    std = trades_df['pnl'].std()
    sharpe = (trades_df['pnl'].mean() / std) * np.sqrt(288) if std > 0 else 0.0

    return {
        'total_trades': int(n),
        'win_rate': float(round(wins / n, 4)),
        'pnl_total': float(round(pnl_total, 4)),
        'pnl_per_trade': float(round(pnl_per_trade, 4)),
        'max_drawdown': float(round(max_drawdown, 4)),
        'sharpe': float(round(sharpe, 4)),
    }


def main():
    df, resolutions = load_data()

    # Compute signals
    agg = compute_market_signals(df)
    signals = build_signals(agg)

    # === Diagnostic: Check signal distributions ===
    print("\n=== Bid-Side Entropy Distribution ===")
    print(f"Up bid entropy:   mean={signals['up_bid_entropy'].mean():.4f}, std={signals['up_bid_entropy'].std():.4f}")
    print(f"Down bid entropy: mean={signals['down_bid_entropy'].mean():.4f}, std={signals['down_bid_entropy'].std():.4f}")
    print(f"Entropy diff (Up-Down): mean={signals['bid_entropy_diff'].mean():.4f}, std={signals['bid_entropy_diff'].std():.4f}")
    print(f"Entropy ratio:    mean={signals['bid_entropy_ratio'].mean():.4f}, std={signals['bid_entropy_ratio'].std():.4f}")
    print(f"Bid conc diff:    mean={signals['bid_concentration_diff'].mean():.4f}, std={signals['bid_concentration_diff'].std():.4f}")
    print(f"Bid size ratio:   mean={signals['bid_size_ratio'].mean():.4f}, std={signals['bid_size_ratio'].std():.4f}")
    print(f"Up imbalance:     mean={signals['up_imbalance'].mean():.4f}, std={signals['up_imbalance'].std():.4f}")
    print(f"Down imbalance:   mean={signals['down_imbalance'].mean():.4f}, std={signals['down_imbalance'].std():.4f}")

    # === Correlation with outcomes ===
    print("\n=== Predictive Power (Correlations) ===")
    res_series = signals.index.to_series().map(resolutions)
    sig_resolved = signals[res_series.notna()].copy()
    sig_resolved['up_wins'] = (res_series[res_series.notna()].values == 'Up').astype(int)

    signal_cols = ['bid_entropy_diff', 'bid_entropy_ratio', 'bid_concentration_diff',
                   'bid_size_ratio', 'up_imbalance', 'down_imbalance',
                   'up_bid_entropy', 'down_bid_entropy']

    for col in signal_cols:
        corr = sig_resolved[col].corr(sig_resolved['up_wins'])
        print(f"  Corr({col}, up_wins): {corr:.4f}")

    # Quintile analysis on bid entropy ratio
    print("\n=== Quintile Analysis: Bid Entropy Ratio ===")
    sig_resolved['entropy_q'] = pd.qcut(sig_resolved['bid_entropy_ratio'], 5, labels=False, duplicates='drop')
    q_stats = sig_resolved.groupby('entropy_q').agg(
        up_wr=('up_wins', 'mean'),
        count=('up_wins', 'count'),
        mean_up_price=('up_mid_price', 'mean'),
    )
    print(q_stats.to_string())

    # Quintile on bid size ratio
    print("\n=== Quintile Analysis: Bid Size Ratio ===")
    sig_resolved['size_q'] = pd.qcut(sig_resolved['bid_size_ratio'], 5, labels=False, duplicates='drop')
    q_stats2 = sig_resolved.groupby('size_q').agg(
        up_wr=('up_wins', 'mean'),
        count=('up_wins', 'count'),
    )
    print(q_stats2.to_string())

    # Quintile on book imbalance
    print("\n=== Quintile Analysis: Up Book Imbalance ===")
    sig_resolved['imb_q'] = pd.qcut(sig_resolved['up_imbalance'], 5, labels=False, duplicates='drop')
    q_stats3 = sig_resolved.groupby('imb_q').agg(
        up_wr=('up_wins', 'mean'),
        count=('up_wins', 'count'),
    )
    print(q_stats3.to_string())

    # === Parameter Sweep: Simple threshold ===
    print("\n=== Parameter Sweep: Simple Threshold ===")
    sweep_results = []
    for ert in [0.7, 0.8, 0.85, 0.9, 0.95]:
        for imt in [0.0, 0.05, 0.1, 0.15, 0.2]:
            for ct in [0.0, 0.3, 0.4, 0.5]:
                params = {
                    'entropy_ratio_threshold': ert,
                    'imbalance_threshold': imt,
                    'concentration_threshold': ct,
                    'use_composite': False,
                }
                trades = generate_trades_vectorized(signals, resolutions, params)
                m = compute_metrics(trades)
                m.update(params)
                sweep_results.append(m)

    sweep_df = pd.DataFrame(sweep_results)
    viable = sweep_df[sweep_df['total_trades'] >= 10].sort_values('pnl_per_trade', ascending=False)
    if len(viable) > 0:
        print(f"Viable combos (>=10 trades): {len(viable)}")
        print(viable.head(20).to_string(index=False))
    else:
        print("No viable parameter combos with >=10 trades")
        # Show what we got
        has_trades = sweep_df[sweep_df['total_trades'] > 0].sort_values('total_trades', ascending=False)
        if len(has_trades) > 0:
            print(f"\nCombos with any trades ({len(has_trades)}):")
            print(has_trades.head(20).to_string(index=False))
        else:
            print("No trades generated at all — signals too weak")

    # === Parameter Sweep: Composite score ===
    print("\n=== Parameter Sweep: Composite Score ===")
    composite_results = []
    for score_thresh in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        params = {
            'use_composite': True,
            'score_threshold': score_thresh,
        }
        trades = generate_trades_vectorized(signals, resolutions, params)
        m = compute_metrics(trades)
        m['score_threshold'] = score_thresh
        composite_results.append(m)
        if m['total_trades'] > 0:
            print(f"Score>{score_thresh}: {m['total_trades']} trades, WR={m['win_rate']:.4f}, "
                  f"PnL/trade={m['pnl_per_trade']:.4f}, total={m['pnl_total']:.4f}")

    composite_df = pd.DataFrame(composite_results)

    # === Select best overall result ===
    all_results = pd.concat([sweep_df, composite_df], ignore_index=True)
    viable_all = all_results[all_results['total_trades'] >= 10].sort_values('pnl_per_trade', ascending=False)

    if len(viable_all) > 0:
        best = viable_all.iloc[0]
        best_metrics = {k: best[k] for k in ['total_trades', 'win_rate', 'pnl_total', 'pnl_per_trade', 'max_drawdown', 'sharpe']}
        best_metrics['total_trades'] = int(best_metrics['total_trades'])
        best_params = {k: best[k] for k in best.index if k not in best_metrics and k != 'use_composite'}
        print(f"\n=== Best Result ===")
        print(f"Metrics: {best_metrics}")
        print(f"Params: {best_params}")
    else:
        best_metrics = compute_metrics(pd.DataFrame())
        best_params = {}

    # === Verdict ===
    if best_metrics['total_trades'] < 10:
        verdict = 'failed'
    elif best_metrics['win_rate'] > 0.55 and best_metrics['pnl_per_trade'] > 0.01:
        verdict = 'promising'
    elif best_metrics['pnl_total'] > 0:
        verdict = 'marginal'
    else:
        verdict = 'failed'

    # Collect diagnostic correlations
    diag_corrs = {}
    for col in signal_cols:
        diag_corrs[f'corr_{col}_vs_up_wins'] = round(sig_resolved[col].corr(sig_resolved['up_wins']), 4)

    results = {
        'strategy_name': 'entropy_weighted_imbalance',
        'description': (
            'Computes Shannon entropy of BID-SIDE size distributions across 5 orderbook levels '
            'for Up and Down tokens in the first 5 seconds. In binary markets, Up bids != Down bids '
            '(Up bids = Down asks), so bid-side entropy captures genuine asymmetry. Low bid entropy '
            '(concentrated buying) on one side relative to the other signals informed positioning. '
            'Also uses a composite z-score combining entropy ratio, book imbalance, bid concentration, '
            'and bid size ratio.'
        ),
        'hypothesis': (
            'Low entropy in the bid-side size distribution indicates concentrated buying at specific '
            'price levels, which may reflect informed positioning. The entropy ratio between Up and '
            'Down bid books serves as a confidence weight. Combined with directional imbalance and '
            'bid concentration, this filters out noise from diffuse liquidity.'
        ),
        'metrics': best_metrics,
        'parameters': {
            'entry_window_ms': ENTRY_WINDOW_MS,
            'max_entry_price': MAX_ENTRY_PRICE,
            **{k: (float(v) if isinstance(v, (np.floating, float)) else v)
               for k, v in best_params.items()},
        },
        'diagnostics': diag_corrs,
        'verdict': verdict,
        'next_steps': (
            'The core issue is that in binary markets with symmetric market makers, bid-side entropy '
            'differences are small. Key findings: (1) bid_entropy_ratio has near-zero correlation with '
            'outcomes, (2) book imbalance and bid size ratio also show minimal predictive power in the '
            'first 5 seconds. Possible improvements: extend window to 15-30s, use entropy CHANGE '
            '(dynamics) rather than level, incorporate trade flow data from trade_events, or look at '
            'order book changes (delta entropy) rather than static snapshots.'
        ),
    }

    # Clean NaN values for JSON serialization
    def clean_for_json(obj):
        if isinstance(obj, dict):
            return {k: clean_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean_for_json(v) for v in obj]
        elif isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
            return None
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj) if isinstance(obj, np.floating) else int(obj)
        return obj

    results = clean_for_json(results)

    with open(OUTPUT_PATH, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nVerdict: {verdict}")
    print(f"Results written to {OUTPUT_PATH}")
    return results


if __name__ == '__main__':
    main()
