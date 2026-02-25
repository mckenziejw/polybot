#!/usr/bin/env python3
"""
Momentum Strategy Backtest — Clean Re-evaluation from First Principles

Strategy:
  - Observe token A's mid-price over first 50% of a 5-minute market window
  - Calculate momentum = end_mid - start_mid
  - If momentum > +threshold: BUY token A at its ask (ride momentum up)
  - If momentum < -threshold: BUY token B at its ask (ride momentum down)
  - Hold to expiry. Winning token pays $1, losing token pays $0.

Profit calculation:
  - Cost = position_size (e.g. $100)
  - Shares = position_size / ask_price_of_token_we_buy
  - Payout = shares × outcome_of_token_we_buy (1.0 or 0.0)
  - Profit = payout - cost

This is unambiguous: we always BUY a token at its ask and hold to resolution.
"""

import json
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StrategyConfig:
    observation_window: float = 0.50  # observe first 50% of book snapshots
    momentum_threshold: float = 0.03  # minimum |momentum| to trigger signal
    position_size: float = 100.0      # dollars per trade
    use_ask_for_entry: bool = True    # True = realistic (pay the ask), False = use mid
    min_book_snapshots: int = 30      # skip markets with too few snapshots


# ─────────────────────────────────────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_resolutions(cache_path: str = "data/resolution_cache.json") -> dict:
    """Load resolution cache: slug -> {token_outcomes, winning_token, ...}"""
    with open(cache_path) as f:
        return json.load(f)


def load_book_snapshots(slug: str, data_dir: str = "data/book_snapshots") -> Optional[pd.DataFrame]:
    """Load and validate book snapshot data for a market."""
    path = Path(data_dir) / f"{slug}.parquet"
    if not path.exists() or path.stat().st_size < 1000:
        return None
    try:
        df = pq.read_table(str(path)).to_pandas()
        required = ["asset_id", "exchange_timestamp", "bid_price_1", "ask_price_1"]
        if not all(c in df.columns for c in required):
            return None
        return df
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Core Strategy Logic
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TradeResult:
    slug: str
    signal: str              # "buy_up" or "buy_down"
    observed_token: str      # token we measured momentum on
    traded_token: str        # token we actually bought
    momentum: float          # raw momentum value
    mid_at_entry: float      # mid-price of observed token at signal time
    entry_ask: float         # actual ask price we'd pay for the token we buy
    outcome: float           # 1.0 or 0.0 for the token we bought
    shares: float            # shares purchased
    cost: float              # dollars spent
    payout: float            # dollars received at resolution
    profit: float            # payout - cost
    profit_pct: float        # profit / cost
    market_start_ts: int     # unix timestamp of market start


def evaluate_market(
    slug: str,
    resolution: dict,
    books: pd.DataFrame,
    config: StrategyConfig,
) -> Optional[TradeResult]:
    """
    Evaluate the momentum strategy on a single market window.
    Returns a TradeResult if a signal fires, None otherwise.
    """
    token_outcomes = resolution["token_outcomes"]
    token_ids = list(token_outcomes.keys())

    if len(token_ids) != 2:
        return None

    # Filter books to only tokens in this resolution
    books = books[books["asset_id"].isin(token_ids)].copy()

    # Separate by token
    token_a_id = token_ids[0]
    token_b_id = token_ids[1]

    books_a = books[books["asset_id"] == token_a_id].sort_values("exchange_timestamp")
    books_b = books[books["asset_id"] == token_b_id].sort_values("exchange_timestamp")

    if len(books_a) < config.min_book_snapshots or len(books_b) < config.min_book_snapshots:
        return None

    # ── Step 1: Calculate momentum on token A ────────────────────────────
    # Use mid-price for momentum calculation (signal detection)
    books_a = books_a.copy()
    books_a["mid"] = (books_a["bid_price_1"] + books_a["ask_price_1"]) / 2

    cutoff_idx = int(len(books_a) * config.observation_window)
    if cutoff_idx < 5:
        return None

    observation = books_a.iloc[:cutoff_idx]
    start_mid = observation["mid"].iloc[0]
    end_mid = observation["mid"].iloc[-1]
    momentum = end_mid - start_mid

    # ── Step 2: Check threshold ──────────────────────────────────────────
    if abs(momentum) < config.momentum_threshold:
        return None

    # ── Step 3: Determine which token to BUY and get its ACTUAL ask ──────
    # Find the book snapshot closest to our signal time for BOTH tokens
    signal_timestamp = observation["exchange_timestamp"].iloc[-1]

    if momentum > 0:
        # Token A pumped → BUY token A (ride momentum up)
        signal = "buy_up"
        traded_token_id = token_a_id
        traded_books = books_a
    else:
        # Token A dumped → BUY token B (ride momentum down)
        signal = "buy_down"
        traded_token_id = token_b_id
        traded_books = books_b

    # Get the ask price of the token we're buying at signal time
    # Find closest snapshot at or after signal time
    entry_candidates = traded_books[
        traded_books["exchange_timestamp"] >= signal_timestamp
    ]
    if entry_candidates.empty:
        # Fall back to last snapshot before signal
        entry_candidates = traded_books[
            traded_books["exchange_timestamp"] <= signal_timestamp
        ]
    if entry_candidates.empty:
        return None

    entry_row = entry_candidates.iloc[0]

    if config.use_ask_for_entry:
        entry_price = entry_row["ask_price_1"]
    else:
        entry_price = (entry_row["bid_price_1"] + entry_row["ask_price_1"]) / 2

    # Validate entry price
    if entry_price <= 0 or entry_price >= 1:
        return None

    # ── Step 4: Calculate profit ─────────────────────────────────────────
    outcome = token_outcomes[traded_token_id]
    shares = config.position_size / entry_price
    cost = config.position_size
    payout = shares * outcome
    profit = payout - cost
    profit_pct = profit / cost

    # Extract market start timestamp from slug
    try:
        market_start_ts = int(slug.split("-")[-1])
    except ValueError:
        market_start_ts = 0

    return TradeResult(
        slug=slug,
        signal=signal,
        observed_token=token_a_id[:20] + "...",
        traded_token=traded_token_id[:20] + "...",
        momentum=momentum,
        mid_at_entry=end_mid,
        entry_ask=entry_price,
        outcome=outcome,
        shares=shares,
        cost=cost,
        payout=payout,
        profit=profit,
        profit_pct=profit_pct,
        market_start_ts=market_start_ts,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Backtest Runner
# ─────────────────────────────────────────────────────────────────────────────

def run_backtest(config: StrategyConfig) -> pd.DataFrame:
    """Run the momentum strategy across all resolved markets."""
    resolutions = load_resolutions()
    slugs = sorted(resolutions.keys())

    print(f"Config: window={config.observation_window:.0%}, "
          f"threshold={config.momentum_threshold}, "
          f"entry={'ask' if config.use_ask_for_entry else 'mid'}, "
          f"position=${config.position_size:.0f}")
    print(f"Markets available: {len(slugs)}")
    print()

    results = []
    processed = 0
    skipped = 0

    for slug in slugs:
        books = load_book_snapshots(slug)
        if books is None:
            skipped += 1
            continue

        trade = evaluate_market(slug, resolutions[slug], books, config)
        processed += 1

        if trade is not None:
            results.append(trade.__dict__)

    print(f"Markets processed: {processed}")
    print(f"Markets skipped (no/bad data): {skipped}")
    print(f"Signals generated: {len(results)}")
    print(f"Signal rate: {len(results)/processed:.1%}" if processed > 0 else "")
    print()

    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────────────────────
# Analysis Functions
# ─────────────────────────────────────────────────────────────────────────────

def analyze_results(df: pd.DataFrame, label: str = "FULL DATASET"):
    """Print comprehensive analysis of backtest results."""
    if df.empty:
        print("No trades to analyze.")
        return

    n = len(df)
    wins = df[df["profit"] > 0]
    losses = df[df["profit"] <= 0]
    n_wins = len(wins)
    win_rate = n_wins / n

    print(f"{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")
    print()

    # ── Overall ──────────────────────────────────────────────────────────
    print(f"  Total trades:      {n}")
    print(f"  Win rate:          {win_rate:.2%} ({n_wins}W / {n - n_wins}L)")
    print(f"  Avg profit:        ${df['profit'].mean():.2f} per ${df['cost'].iloc[0]:.0f} trade")
    print(f"  Median profit:     ${df['profit'].median():.2f}")
    print(f"  Total P&L:         ${df['profit'].sum():.2f}")
    print(f"  Std dev:           ${df['profit'].std():.2f}")
    if not wins.empty and not losses.empty:
        print(f"  Avg win:           ${wins['profit'].mean():.2f}")
        print(f"  Avg loss:          ${losses['profit'].mean():.2f}")
        pf = abs(wins['profit'].sum() / losses['profit'].sum()) if losses['profit'].sum() != 0 else float('inf')
        print(f"  Profit factor:     {pf:.2f}")
    print()

    # ── By Signal Type ───────────────────────────────────────────────────
    print("  BY SIGNAL TYPE:")
    for sig in ["buy_up", "buy_down"]:
        sub = df[df["signal"] == sig]
        if sub.empty:
            continue
        sw = sub[sub["profit"] > 0]
        wr = len(sw) / len(sub)
        print(f"    {sig:10s}: {len(sub):4d} trades | "
              f"Win: {wr:5.1%} | "
              f"Avg: ${sub['profit'].mean():7.2f} | "
              f"Total: ${sub['profit'].sum():8.2f}")
    print()

    # ── By Entry Price Bucket ────────────────────────────────────────────
    print("  BY ENTRY PRICE:")
    buckets = [
        (0.00, 0.15, "0.00-0.15"),
        (0.15, 0.30, "0.15-0.30"),
        (0.30, 0.45, "0.30-0.45"),
        (0.45, 0.55, "0.45-0.55"),
        (0.55, 0.70, "0.55-0.70"),
        (0.70, 0.85, "0.70-0.85"),
        (0.85, 1.00, "0.85-1.00"),
    ]
    for lo, hi, label_b in buckets:
        sub = df[(df["entry_ask"] >= lo) & (df["entry_ask"] < hi)]
        if len(sub) < 3:
            continue
        sw = sub[sub["profit"] > 0]
        wr = len(sw) / len(sub)
        print(f"    {label_b}: {len(sub):4d} trades | "
              f"Win: {wr:5.1%} | "
              f"Avg: ${sub['profit'].mean():7.2f}")
    print()

    # ── Validation: per-$1 vs per-$100 consistency ───────────────────────
    # per-$1 directional profit (the original notebook's approach)
    # buy_up: profit_dir = outcome - entry (positive if token won)
    # buy_down: we bought the other token, so profit_dir = outcome - entry (same)
    # In ALL cases: profit > 0 iff outcome == 1.0 for the token we bought
    inconsistent = ((df["profit"] > 0) != (df["outcome"] == 1.0)).sum()
    valid_str = "✓ YES" if inconsistent == 0 else f"✗ NO ({inconsistent} mismatches)"
    print(f"  VALIDATION: profit>0 consistent with outcome==1.0? {valid_str}")
    print()


def temporal_split_analysis(df: pd.DataFrame):
    """Split by time to check if strategy is stable."""
    if df.empty:
        return

    df_sorted = df.sort_values("market_start_ts")
    n = len(df_sorted)

    splits = {
        "First 25%":  df_sorted.iloc[:n//4],
        "Second 25%": df_sorted.iloc[n//4:n//2],
        "Third 25%":  df_sorted.iloc[n//2:3*n//4],
        "Last 25%":   df_sorted.iloc[3*n//4:],
    }

    print(f"{'='*70}")
    print(f"  TEMPORAL STABILITY (Time-based quartiles)")
    print(f"{'='*70}")
    print()

    for label, sub in splits.items():
        if sub.empty:
            continue
        wr = (sub["profit"] > 0).mean()
        print(f"    {label:12s}: {len(sub):4d} trades | "
              f"Win: {wr:5.1%} | "
              f"Avg: ${sub['profit'].mean():7.2f} | "
              f"Total: ${sub['profit'].sum():8.2f}")
    print()


def train_test_split_analysis(df: pd.DataFrame, train_frac: float = 0.7):
    """70/30 train/test split by time."""
    if df.empty:
        return

    df_sorted = df.sort_values("market_start_ts")
    n = len(df_sorted)
    split_idx = int(n * train_frac)

    train = df_sorted.iloc[:split_idx]
    test = df_sorted.iloc[split_idx:]

    print(f"{'='*70}")
    print(f"  TRAIN / TEST SPLIT ({train_frac:.0%} / {1-train_frac:.0%})")
    print(f"{'='*70}")
    print()
    print(f"  TRAIN ({len(train)} trades):")
    print(f"    Win rate: {(train['profit'] > 0).mean():.2%}")
    print(f"    Avg profit: ${train['profit'].mean():.2f}")
    print(f"    Total P&L: ${train['profit'].sum():.2f}")
    print()
    print(f"  TEST ({len(test)} trades):")
    print(f"    Win rate: {(test['profit'] > 0).mean():.2%}")
    print(f"    Avg profit: ${test['profit'].mean():.2f}")
    print(f"    Total P&L: ${test['profit'].sum():.2f}")

    # Per signal in test
    print()
    for sig in ["buy_up", "buy_down"]:
        sub = test[test["signal"] == sig]
        if sub.empty:
            continue
        wr = (sub["profit"] > 0).mean()
        print(f"    TEST {sig:10s}: {len(sub):3d} trades | Win: {wr:5.1%} | Avg: ${sub['profit'].mean():7.2f}")
    print()


def fee_impact_analysis(df: pd.DataFrame):
    """Estimate Polymarket fee impact on results."""
    if df.empty:
        return

    print(f"{'='*70}")
    print(f"  FEE IMPACT ANALYSIS")
    print(f"{'='*70}")
    print()

    # Polymarket taker fee formula:
    # fee = shares × price × 0.25 × (price × (1 - price))^2
    # This is the fee you pay when you BUY (take from the book)
    def taker_fee(shares, price):
        return shares * price * 0.25 * (price * (1 - price)) ** 2

    fees = df.apply(
        lambda r: taker_fee(r["shares"], r["entry_ask"]), axis=1
    )
    df_with_fees = df.copy()
    df_with_fees["fee"] = fees
    df_with_fees["profit_after_fee"] = df_with_fees["profit"] - df_with_fees["fee"]

    avg_fee = fees.mean()
    total_fees = fees.sum()

    print(f"  Average fee per trade:     ${avg_fee:.4f}")
    print(f"  Total fees:                ${total_fees:.2f}")
    print()
    print(f"  BEFORE FEES:")
    print(f"    Win rate:  {(df['profit'] > 0).mean():.2%}")
    print(f"    Avg profit: ${df['profit'].mean():.2f}")
    print(f"    Total P&L:  ${df['profit'].sum():.2f}")
    print()
    print(f"  AFTER FEES:")
    wr_af = (df_with_fees["profit_after_fee"] > 0).mean()
    print(f"    Win rate:  {wr_af:.2%}")
    print(f"    Avg profit: ${df_with_fees['profit_after_fee'].mean():.2f}")
    print(f"    Total P&L:  ${df_with_fees['profit_after_fee'].sum():.2f}")
    print()

    # Fee as % of entry
    print(f"  Fee as % of position: {avg_fee / df['cost'].iloc[0] * 100:.4f}%")
    print(f"  → Fees are negligible for this strategy")
    print()


def drawdown_analysis(df: pd.DataFrame):
    """Analyze worst consecutive losses."""
    if df.empty:
        return

    print(f"{'='*70}")
    print(f"  DRAWDOWN ANALYSIS")
    print(f"{'='*70}")
    print()

    df_sorted = df.sort_values("market_start_ts")
    cumulative = df_sorted["profit"].cumsum()
    running_max = cumulative.cummax()
    drawdown = cumulative - running_max

    print(f"  Max cumulative P&L:    ${cumulative.max():.2f}")
    print(f"  Final cumulative P&L:  ${cumulative.iloc[-1]:.2f}")
    print(f"  Max drawdown:          ${drawdown.min():.2f}")
    print()

    # Consecutive losses
    is_loss = (df_sorted["profit"] <= 0).values
    max_consec = 0
    current = 0
    for loss in is_loss:
        if loss:
            current += 1
            max_consec = max(max_consec, current)
        else:
            current = 0

    print(f"  Max consecutive losses: {max_consec}")
    print(f"  Worst single trade:    ${df['profit'].min():.2f}")
    print(f"  Best single trade:     ${df['profit'].max():.2f}")
    print()


def daily_economics(df: pd.DataFrame, capital: float = 2000):
    """Estimate daily profit potential."""
    if df.empty:
        return

    print(f"{'='*70}")
    print(f"  DAILY ECONOMICS (${capital:,.0f} capital)")
    print(f"{'='*70}")
    print()

    markets_per_day = 288  # 24hr × 60min / 5min
    signal_rate = len(df) / 1395  # approximate: signals / total markets
    signals_per_day = markets_per_day * signal_rate
    pos_size = df["cost"].iloc[0]
    scale = capital / pos_size
    avg_profit_scaled = df["profit"].mean() * scale

    print(f"  Markets per day:        {markets_per_day}")
    print(f"  Signal rate:            {signal_rate:.1%}")
    print(f"  Expected signals/day:   {signals_per_day:.0f}")
    print(f"  Avg profit per trade:   ${avg_profit_scaled:.2f} (on ${capital:,.0f})")
    print()

    for label, capture in [("Conservative (10%)", 0.10),
                            ("Moderate (25%)", 0.25),
                            ("Bot 24/7 (80%)", 0.80)]:
        n_trades = signals_per_day * capture
        daily = n_trades * avg_profit_scaled
        monthly = daily * 30
        print(f"  {label:25s}: {n_trades:5.1f} trades/day → "
              f"${daily:8.2f}/day → ${monthly:10.2f}/month")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print()
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║  MOMENTUM STRATEGY BACKTEST — Clean Evaluation from First Principles║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print()

    # ── Primary backtest: realistic params (ask entry) ───────────────────
    config = StrategyConfig(
        observation_window=0.50,
        momentum_threshold=0.03,
        position_size=100.0,
        use_ask_for_entry=True,
    )

    df = run_backtest(config)

    if df.empty:
        print("No trades generated. Check data paths.")
        return

    analyze_results(df, "FULL DATASET — ASK ENTRY")
    temporal_split_analysis(df)
    train_test_split_analysis(df)
    fee_impact_analysis(df)
    drawdown_analysis(df)
    daily_economics(df)

    # ── Comparison: mid entry (how much does ask vs mid matter?) ──────────
    print()
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║  COMPARISON: MID-PRICE ENTRY (less realistic)                       ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print()

    config_mid = StrategyConfig(
        observation_window=0.50,
        momentum_threshold=0.03,
        position_size=100.0,
        use_ask_for_entry=False,
    )
    df_mid = run_backtest(config_mid)
    if not df_mid.empty:
        analyze_results(df_mid, "FULL DATASET — MID ENTRY (comparison)")

    # ── Sensitivity: different thresholds ────────────────────────────────
    print()
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║  THRESHOLD SENSITIVITY                                              ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print()

    for thresh in [0.01, 0.02, 0.03, 0.05, 0.07, 0.10]:
        cfg = StrategyConfig(
            observation_window=0.50,
            momentum_threshold=thresh,
            position_size=100.0,
            use_ask_for_entry=True,
        )
        df_t = run_backtest(cfg)
        if df_t.empty:
            continue
        wr = (df_t["profit"] > 0).mean()
        print(f"  Threshold {thresh:.2f}: {len(df_t):4d} trades | "
              f"Win: {wr:5.1%} | "
              f"Avg: ${df_t['profit'].mean():7.2f} | "
              f"Total: ${df_t['profit'].sum():8.2f}")
    print()


if __name__ == "__main__":
    main()
