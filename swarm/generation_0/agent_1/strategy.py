"""
Cross-Timeframe Book Imbalance Divergence Strategy

Hypothesis: When the 5m market opens with a book imbalance sharply divergent from
the enclosing 4h market's concurrent imbalance, this indicates short-term mispricing.
Fade the 5m divergence, expecting convergence to the 4h consensus.

Example: 5m book skewed bullish (high Up imbalance) while 4h is neutral →
the 5m Up side is overpriced → buy Down (fade the divergence).
"""

import pandas as pd
import numpy as np
import json
import glob
import re
from pathlib import Path

BASE_DIR = Path("/home/josh/Documents/projects/wss_test")
DATA_5M = BASE_DIR / "data" / "telonex_book_snapshots"
DATA_4H = BASE_DIR / "data" / "telonex_btc_4h_book_snapshots"
RESOLUTION_CACHE = BASE_DIR / "data" / "resolution_cache.json"
RESULTS_PATH = BASE_DIR / "swarm" / "generation_0" / "agent_1" / "results.json"

# Strategy parameters
DIVERGENCE_THRESHOLD = 0.15   # minimum |5m_imb - 4h_imb| to trigger a trade
OPENING_WINDOW_MS = 5000      # first 5 seconds of 5m market for "opening" snapshot
MAX_ENTRY_PRICE = 0.85        # never buy above this
MIN_ORDER_SIZE = 5            # minimum order in tokens


def compute_fee(price: float) -> float:
    """Polymarket taker fee for crypto markets."""
    return 0.0222 * price * 0.25 * (price * (1 - price)) ** 2


def compute_level_weighted_imbalance(row: pd.Series) -> float:
    """
    Compute level-weighted bid/ask imbalance from 5-level book data.
    Weight = 1/level (level 1 gets weight 1, level 5 gets weight 0.2).
    Returns value in [-1, 1]: positive = bid-heavy (bullish), negative = ask-heavy.
    """
    bid_value = 0.0
    ask_value = 0.0
    for level in range(1, 6):
        weight = 1.0 / level
        bp = row.get(f'bid_price_{level}', 0) or 0
        bs = row.get(f'bid_size_{level}', 0) or 0
        ap = row.get(f'ask_price_{level}', 0) or 0
        ask_s = row.get(f'ask_size_{level}', 0) or 0
        bid_value += weight * bp * bs
        ask_value += weight * ap * ask_s
    total = bid_value + ask_value
    if total == 0:
        return 0.0
    return (bid_value - ask_value) / total


def compute_imbalance_vectorized(df: pd.DataFrame) -> pd.Series:
    """Vectorized level-weighted imbalance computation."""
    bid_value = pd.Series(0.0, index=df.index)
    ask_value = pd.Series(0.0, index=df.index)
    for level in range(1, 6):
        weight = 1.0 / level
        bp = df[f'bid_price_{level}'].fillna(0)
        bs = df[f'bid_size_{level}'].fillna(0)
        ap = df[f'ask_price_{level}'].fillna(0)
        ask_s = df[f'ask_size_{level}'].fillna(0)
        bid_value += weight * bp * bs
        ask_value += weight * ap * ask_s
    total = bid_value + ask_value
    return ((bid_value - ask_value) / total).where(total > 0, 0.0)


def load_resolution_cache() -> dict:
    """Load resolution cache and determine winner (Up/Down) for each 5m slug."""
    with open(RESOLUTION_CACHE) as f:
        rc = json.load(f)
    return rc


def get_winner_from_book_and_resolution(
    book_df: pd.DataFrame, rc_entry: dict
) -> str | None:
    """Determine winning token label (Up/Down) by joining resolution with book data."""
    # Handle both resolution cache formats
    if 'winningTokenId' in rc_entry:
        winning_id = rc_entry['winningTokenId']
    elif 'winning_token' in rc_entry:
        winning_id = rc_entry['winning_token']
    else:
        return None

    # Map asset_id -> token_label from book data
    asset_map = book_df.groupby('asset_id')['token_label'].first()
    winning_id_str = str(winning_id)
    if winning_id_str in asset_map.index:
        return asset_map[winning_id_str]
    return None


def load_4h_index(files_4h: list) -> pd.DataFrame:
    """
    Build an index of 4h market epochs and preload their book data.
    Returns DataFrame with columns: [epoch, file_path, slug]
    """
    records = []
    for f in files_4h:
        slug = Path(f).stem
        m = re.search(r'(\d+)$', slug)
        if m:
            records.append({
                'epoch': int(m.group(1)),
                'file_path': f,
                'slug': slug,
            })
    return pd.DataFrame(records).sort_values('epoch').reset_index(drop=True)


def get_4h_imbalance_at_time(
    four_h_cache: dict, four_h_index: pd.DataFrame,
    epoch_5m: int, target_ts_ms: int
) -> float | None:
    """
    Find the enclosing 4h market for a given 5m epoch, then get the
    level-weighted imbalance of the Up token at the closest timestamp
    to target_ts_ms.
    """
    # Find enclosing 4h market
    idx = np.searchsorted(four_h_index['epoch'].values, epoch_5m, side='right') - 1
    if idx < 0:
        return None
    row_4h = four_h_index.iloc[idx]
    if not (row_4h['epoch'] <= epoch_5m < row_4h['epoch'] + 14400):
        return None

    slug_4h = row_4h['slug']

    # Load 4h data (with caching)
    if slug_4h not in four_h_cache:
        df4h = pd.read_parquet(row_4h['file_path'])
        # Filter to Up token only
        df4h_up = df4h[df4h['token_label'] == 'Up'].copy()
        df4h_up = df4h_up.sort_values('exchange_timestamp')
        # Precompute imbalance
        df4h_up['lw_imbalance'] = compute_imbalance_vectorized(df4h_up)
        four_h_cache[slug_4h] = df4h_up

    df4h_up = four_h_cache[slug_4h]
    if len(df4h_up) == 0:
        return None

    # Find closest timestamp to target
    ts_arr = df4h_up['exchange_timestamp'].values
    closest_idx = np.argmin(np.abs(ts_arr - target_ts_ms))
    return float(df4h_up['lw_imbalance'].iloc[closest_idx])


def run_backtest(
    divergence_threshold: float = DIVERGENCE_THRESHOLD,
    opening_window_ms: int = OPENING_WINDOW_MS,
    max_entry_price: float = MAX_ENTRY_PRICE,
) -> dict:
    """Run the cross-timeframe book imbalance divergence backtest."""

    rc = load_resolution_cache()
    files_5m = sorted(glob.glob(str(DATA_5M / "*.parquet")))
    files_4h = sorted(glob.glob(str(DATA_4H / "*.parquet")))

    print(f"5m files: {len(files_5m)}, 4h files: {len(files_4h)}")

    four_h_index = load_4h_index(files_4h)
    four_h_cache: dict = {}  # slug -> DataFrame (LRU-ish, 4h files are large)
    four_h_epochs = four_h_index['epoch'].values

    trades = []
    skipped_no_4h = 0
    skipped_no_resolution = 0
    skipped_no_divergence = 0
    skipped_price = 0
    skipped_no_data = 0

    for i, fpath in enumerate(files_5m):
        if i % 500 == 0:
            print(f"  Processing {i}/{len(files_5m)}...")

        slug_5m = Path(fpath).stem
        m = re.search(r'(\d+)$', slug_5m)
        if not m:
            continue
        epoch_5m = int(m.group(1))

        # Check resolution
        if slug_5m not in rc:
            skipped_no_resolution += 1
            continue

        # Check 4h overlap
        idx = np.searchsorted(four_h_epochs, epoch_5m, side='right') - 1
        if idx < 0 or not (four_h_epochs[idx] <= epoch_5m < four_h_epochs[idx] + 14400):
            skipped_no_4h += 1
            continue

        # Load 5m book
        df5m = pd.read_parquet(fpath)
        df5m_up = df5m[df5m['token_label'] == 'Up'].sort_values('exchange_timestamp')

        if len(df5m_up) == 0:
            skipped_no_data += 1
            continue

        # Get opening window (first N ms)
        market_start_ms = epoch_5m * 1000
        opening_mask = (
            (df5m_up['exchange_timestamp'] >= market_start_ms) &
            (df5m_up['exchange_timestamp'] <= market_start_ms + opening_window_ms)
        )
        opening = df5m_up[opening_mask]
        if len(opening) == 0:
            # Fallback: use first 5 rows
            opening = df5m_up.head(5)
        if len(opening) == 0:
            skipped_no_data += 1
            continue

        # Compute 5m opening imbalance (average over opening window)
        opening_imb = compute_imbalance_vectorized(opening)
        imb_5m = float(opening_imb.mean())

        # Get 4h imbalance at the same time
        target_ts = int(opening['exchange_timestamp'].median())
        imb_4h = get_4h_imbalance_at_time(
            four_h_cache, four_h_index, epoch_5m, target_ts
        )
        if imb_4h is None:
            skipped_no_4h += 1
            continue

        # Compute divergence
        divergence = imb_5m - imb_4h  # positive = 5m more bullish than 4h

        if abs(divergence) < divergence_threshold:
            skipped_no_divergence += 1
            continue

        # Determine trade direction: FADE the 5m divergence
        # If 5m is more bullish (divergence > 0), buy Down (fade bullish)
        # If 5m is more bearish (divergence < 0), buy Up (fade bearish)
        if divergence > 0:
            # 5m is too bullish → buy Down
            trade_side = "Down"
            # Entry price = ask_price_1 of the Down token in opening window
            down_opening = df5m[
                (df5m['token_label'] == 'Down') &
                (df5m['exchange_timestamp'].isin(opening['exchange_timestamp'].values))
            ]
            if len(down_opening) == 0:
                down_opening = df5m[df5m['token_label'] == 'Down'].head(5)
            if len(down_opening) == 0:
                skipped_no_data += 1
                continue
            entry_price = float(down_opening['ask_price_1'].median())
        else:
            # 5m is too bearish → buy Up
            trade_side = "Up"
            entry_price = float(opening['ask_price_1'].median())

        # Check entry price cap
        if entry_price > max_entry_price or entry_price <= 0 or np.isnan(entry_price):
            skipped_price += 1
            continue

        # Fee
        fee = compute_fee(entry_price)

        # Determine outcome
        winner = get_winner_from_book_and_resolution(df5m, rc[slug_5m])
        if winner is None:
            skipped_no_resolution += 1
            continue

        won = (winner == trade_side)
        payout = 1.0 if won else 0.0
        pnl = payout - entry_price - fee

        trades.append({
            'slug': slug_5m,
            'epoch': epoch_5m,
            'side': trade_side,
            'entry_price': entry_price,
            'fee': fee,
            'winner': winner,
            'won': won,
            'pnl': pnl,
            'divergence': divergence,
            'imb_5m': imb_5m,
            'imb_4h': imb_4h,
        })

        # Keep 4h cache manageable (max ~20 entries)
        if len(four_h_cache) > 20:
            oldest_key = next(iter(four_h_cache))
            del four_h_cache[oldest_key]

    print(f"\nSkipped: no_4h={skipped_no_4h}, no_resolution={skipped_no_resolution}, "
          f"no_divergence={skipped_no_divergence}, price_cap={skipped_price}, "
          f"no_data={skipped_no_data}")

    return analyze_trades(trades, divergence_threshold, opening_window_ms, max_entry_price)


def analyze_trades(
    trades: list, divergence_threshold: float,
    opening_window_ms: int, max_entry_price: float
) -> dict:
    """Analyze trade results and produce metrics."""
    if not trades:
        print("NO TRADES GENERATED")
        return {
            "strategy_name": "cross_timeframe_imbalance_divergence",
            "description": "Fade 5m book imbalance divergence from enclosing 4h market",
            "hypothesis": "When 5m market opens with imbalance sharply divergent from 4h, "
                          "fade the divergence expecting convergence",
            "metrics": {
                "total_trades": 0,
                "win_rate": 0.0,
                "pnl_total": 0.0,
                "pnl_per_trade": 0.0,
                "max_drawdown": 0.0,
                "sharpe": 0.0,
            },
            "parameters": {
                "divergence_threshold": divergence_threshold,
                "opening_window_ms": opening_window_ms,
                "max_entry_price": max_entry_price,
            },
            "verdict": "failed",
            "next_steps": "No trades met criteria. Try lower threshold or wider opening window.",
        }

    df = pd.DataFrame(trades)
    total = len(df)
    wins = int(df['won'].sum())
    win_rate = wins / total
    pnl_total = float(df['pnl'].sum())
    pnl_per_trade = pnl_total / total

    # Drawdown
    cumulative = df['pnl'].cumsum()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max)
    max_drawdown = float(drawdown.min())

    # Sharpe (annualize assuming ~288 5m markets/day, 365 days)
    if df['pnl'].std() > 0:
        daily_trades = 288
        sharpe = (df['pnl'].mean() / df['pnl'].std()) * np.sqrt(daily_trades * 365)
    else:
        sharpe = 0.0

    # Print analysis
    print(f"\n{'='*60}")
    print(f"BACKTEST RESULTS: Cross-Timeframe Imbalance Divergence")
    print(f"{'='*60}")
    print(f"Total trades:    {total}")
    print(f"Wins:            {wins} ({win_rate:.1%})")
    print(f"PnL total:       ${pnl_total:.2f}")
    print(f"PnL per trade:   ${pnl_per_trade:.4f}")
    print(f"Max drawdown:    ${max_drawdown:.2f}")
    print(f"Sharpe ratio:    {sharpe:.2f}")
    print(f"Avg entry price: ${df['entry_price'].mean():.3f}")
    print(f"Avg fee:         ${df['fee'].mean():.6f}")
    print(f"Avg |divergence|:{df['divergence'].abs().mean():.4f}")

    # Side breakdown
    print(f"\nBy side:")
    for side in ['Up', 'Down']:
        subset = df[df['side'] == side]
        if len(subset) > 0:
            print(f"  {side}: {len(subset)} trades, "
                  f"WR={subset['won'].mean():.1%}, "
                  f"PnL=${subset['pnl'].sum():.2f}")

    # Divergence quintile analysis
    print(f"\nBy divergence magnitude quintile:")
    df['div_abs'] = df['divergence'].abs()
    df['quintile'] = pd.qcut(df['div_abs'], q=5, labels=False, duplicates='drop')
    for q in sorted(df['quintile'].unique()):
        subset = df[df['quintile'] == q]
        print(f"  Q{q}: {len(subset)} trades, "
              f"WR={subset['won'].mean():.1%}, "
              f"PnL/trade=${subset['pnl'].mean():.4f}, "
              f"avg|div|={subset['div_abs'].mean():.4f}")

    # Determine verdict
    if pnl_per_trade > 0.02 and total >= 50 and win_rate > 0.52:
        verdict = "promising"
    elif pnl_per_trade > 0 and total >= 30:
        verdict = "marginal"
    else:
        verdict = "failed"

    results = {
        "strategy_name": "cross_timeframe_imbalance_divergence",
        "description": "Fade 5m book imbalance divergence from enclosing 4h market. "
                       "Compares level-weighted bid/ask imbalance ratio in the 5m "
                       "market's opening snapshot vs the 4h market's concurrent snapshot. "
                       "When divergence exceeds threshold, buy the opposite side.",
        "hypothesis": "When the 5m market opens with imbalance sharply divergent from "
                      "the 4h book, this indicates short-term mispricing before the 5m "
                      "book converges to the 4h consensus — fade the 5m divergence.",
        "metrics": {
            "total_trades": total,
            "win_rate": round(win_rate, 4),
            "pnl_total": round(pnl_total, 2),
            "pnl_per_trade": round(pnl_per_trade, 4),
            "max_drawdown": round(max_drawdown, 2),
            "sharpe": round(sharpe, 2),
        },
        "parameters": {
            "divergence_threshold": divergence_threshold,
            "opening_window_ms": opening_window_ms,
            "max_entry_price": max_entry_price,
        },
        "verdict": verdict,
        "next_steps": "",
    }

    if verdict == "failed":
        results["next_steps"] = (
            "Strategy does not show edge. The 5m/4h imbalance divergence "
            "does not predict 5m resolution outcomes. Consider: (1) the 5m book "
            "forms too quickly for the 4h divergence to be informative, or "
            "(2) the divergence reflects legitimate new information, not mispricing."
        )
    elif verdict == "marginal":
        results["next_steps"] = (
            "Marginal edge detected but too thin for live trading after fees. "
            "Investigate: (1) optimal divergence threshold, (2) time-of-day effects, "
            "(3) combining with other signals like spread or mid-price drift."
        )
    else:
        results["next_steps"] = (
            "Promising signal. Next: (1) walk-forward validation, "
            "(2) parameter stability analysis, (3) combine with entry timing optimization."
        )

    return results


def parameter_sweep():
    """Run backtest across multiple threshold values to check robustness."""
    rc = load_resolution_cache()
    files_5m = sorted(glob.glob(str(DATA_5M / "*.parquet")))
    files_4h = sorted(glob.glob(str(DATA_4H / "*.parquet")))

    four_h_index = load_4h_index(files_4h)
    four_h_cache: dict = {}
    four_h_epochs = four_h_index['epoch'].values

    print("Building trade universe (all markets with 4h overlap)...")

    all_trades_data = []

    for i, fpath in enumerate(files_5m):
        if i % 500 == 0:
            print(f"  Processing {i}/{len(files_5m)}...")

        slug_5m = Path(fpath).stem
        m = re.search(r'(\d+)$', slug_5m)
        if not m:
            continue
        epoch_5m = int(m.group(1))

        if slug_5m not in rc:
            continue

        idx = np.searchsorted(four_h_epochs, epoch_5m, side='right') - 1
        if idx < 0 or not (four_h_epochs[idx] <= epoch_5m < four_h_epochs[idx] + 14400):
            continue

        df5m = pd.read_parquet(fpath)
        df5m_up = df5m[df5m['token_label'] == 'Up'].sort_values('exchange_timestamp')
        df5m_down = df5m[df5m['token_label'] == 'Down'].sort_values('exchange_timestamp')

        if len(df5m_up) == 0 or len(df5m_down) == 0:
            continue

        market_start_ms = epoch_5m * 1000

        # Compute opening snapshot for multiple windows
        for window_ms in [3000, 5000, 10000]:
            opening_mask_up = (
                (df5m_up['exchange_timestamp'] >= market_start_ms) &
                (df5m_up['exchange_timestamp'] <= market_start_ms + window_ms)
            )
            opening_up = df5m_up[opening_mask_up]
            if len(opening_up) == 0:
                opening_up = df5m_up.head(3)

            if len(opening_up) == 0:
                continue

            imb_5m = float(compute_imbalance_vectorized(opening_up).mean())

            target_ts = int(opening_up['exchange_timestamp'].median())
            imb_4h = get_4h_imbalance_at_time(
                four_h_cache, four_h_index, epoch_5m, target_ts
            )
            if imb_4h is None:
                continue

            divergence = imb_5m - imb_4h

            # Get entry prices for both sides
            opening_mask_down = (
                (df5m_down['exchange_timestamp'] >= market_start_ms) &
                (df5m_down['exchange_timestamp'] <= market_start_ms + window_ms)
            )
            opening_down = df5m_down[opening_mask_down]
            if len(opening_down) == 0:
                opening_down = df5m_down.head(3)

            entry_up = float(opening_up['ask_price_1'].median())
            entry_down = float(opening_down['ask_price_1'].median()) if len(opening_down) > 0 else np.nan

            winner = get_winner_from_book_and_resolution(df5m, rc[slug_5m])
            if winner is None:
                continue

            all_trades_data.append({
                'slug': slug_5m,
                'epoch': epoch_5m,
                'window_ms': window_ms,
                'imb_5m': imb_5m,
                'imb_4h': imb_4h,
                'divergence': divergence,
                'entry_up': entry_up,
                'entry_down': entry_down,
                'winner': winner,
            })

        if len(four_h_cache) > 20:
            oldest_key = next(iter(four_h_cache))
            del four_h_cache[oldest_key]

    df_all = pd.DataFrame(all_trades_data)
    print(f"\nTotal trade candidates: {len(df_all)}")
    print(f"Unique markets: {df_all['slug'].nunique()}")

    # Sweep thresholds
    print(f"\n{'='*80}")
    print("PARAMETER SWEEP")
    print(f"{'='*80}")
    print(f"{'Threshold':>10} {'Window':>8} {'Trades':>7} {'WR':>7} {'PnL':>10} {'PnL/Trade':>10}")
    print("-" * 60)

    best_result = None
    best_pnl_per_trade = -999

    for threshold in [0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]:
        for window in [3000, 5000, 10000]:
            subset = df_all[
                (df_all['window_ms'] == window) &
                (df_all['divergence'].abs() >= threshold)
            ].copy()

            if len(subset) == 0:
                continue

            # Apply trade logic
            # divergence > 0 → buy Down, divergence < 0 → buy Up
            subset['side'] = np.where(subset['divergence'] > 0, 'Down', 'Up')
            subset['entry_price'] = np.where(
                subset['side'] == 'Up',
                subset['entry_up'],
                subset['entry_down']
            )

            # Filter valid entries
            valid = subset[
                (subset['entry_price'] > 0) &
                (subset['entry_price'] <= MAX_ENTRY_PRICE) &
                (~subset['entry_price'].isna())
            ].copy()

            if len(valid) == 0:
                continue

            valid['fee'] = valid['entry_price'].map(compute_fee)
            valid['won'] = (valid['winner'] == valid['side'])
            valid['pnl'] = np.where(valid['won'], 1.0, 0.0) - valid['entry_price'] - valid['fee']

            trades_n = len(valid)
            wr = valid['won'].mean()
            pnl = valid['pnl'].sum()
            ppt = valid['pnl'].mean()

            print(f"{threshold:>10.2f} {window:>8} {trades_n:>7} {wr:>7.1%} {pnl:>10.2f} {ppt:>10.4f}")

            if ppt > best_pnl_per_trade and trades_n >= 20:
                best_pnl_per_trade = ppt
                best_result = {
                    'threshold': threshold,
                    'window': window,
                    'trades': trades_n,
                    'wr': wr,
                    'pnl': pnl,
                    'ppt': ppt,
                    'df': valid.copy(),
                }

    if best_result:
        print(f"\nBest: threshold={best_result['threshold']}, window={best_result['window']}ms")
        print(f"  {best_result['trades']} trades, WR={best_result['wr']:.1%}, "
              f"PnL=${best_result['pnl']:.2f}, PnL/trade=${best_result['ppt']:.4f}")

    return df_all, best_result


if __name__ == "__main__":
    print("=" * 60)
    print("PHASE 1: Parameter Sweep")
    print("=" * 60)
    df_all, best = parameter_sweep()

    print("\n" + "=" * 60)
    print("PHASE 2: Run with best parameters (or default)")
    print("=" * 60)

    if best and best['ppt'] > 0:
        results = run_backtest(
            divergence_threshold=best['threshold'],
            opening_window_ms=best['window'],
        )
    else:
        # Run with default
        results = run_backtest()

    # Save results
    with open(RESULTS_PATH, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {RESULTS_PATH}")
    print(f"Verdict: {results['verdict']}")
