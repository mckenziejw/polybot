#!/usr/bin/env python3
"""
Backtest: Mid-price drift strategy with take-profit exit.

Same entry as backtest_midprice_drift.py (t=60s, drift>0.10, taker at ask),
but adds a take-profit exit: if the leading token's mid reaches TP threshold,
we sell at the bid (taker exit) instead of holding to expiry.

This protects against late reversals from oracle delay.

Sweeps TP thresholds from entry+0.05 to 0.99 to find optimal exit.
"""

import json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from dataclasses import dataclass

BOOK_DIR = Path("/home/josh/Documents/projects/wss_test/data/book_snapshots")
RESOLUTION_CACHE = Path("/home/josh/Documents/projects/wss_test/data/resolution_cache.json")

SIGNAL_TIME_S = 60
DRIFT_THRESHOLD = 0.10
BET_DOLLARS = 100


@dataclass
class TPResult:
    slug: str
    market_ts: int
    leading_token: str
    is_winner: bool
    drift: float
    entry_price: float       # taker ask at signal
    # Hold-to-expiry baseline
    hold_pnl: float
    # Per-TP-threshold results: {tp_level: (exited, exit_price, pnl)}
    tp_outcomes: dict


def parse_slug_timestamp(filename: str) -> int:
    return int(filename.replace("btc-updown-5m-", "").replace(".parquet", ""))


def load_resolutions() -> dict:
    return json.loads(RESOLUTION_CACHE.read_text())


def get_real_tokens(book_df: pd.DataFrame, market_ts: int) -> list[str]:
    open_ms = market_ts * 1000
    close_ms = open_ms + 300_000
    df = book_df[
        (book_df.exchange_timestamp >= open_ms - 5000) &
        (book_df.exchange_timestamp <= close_ms + 5000)
    ]
    real = []
    for aid in df.asset_id.unique():
        subset = df[df.asset_id == aid]
        if (subset.bid_size_1 > 0).any() or (subset.ask_size_1 > 0).any():
            real.append((aid, len(subset)))
    real.sort(key=lambda x: x[1], reverse=True)
    return [r[0] for r in real[:2]]


def simulate_market(book_df: pd.DataFrame, market_ts: int, resolution: dict,
                    tp_levels: list[float]) -> TPResult | None:
    open_ms = market_ts * 1000
    close_ms = open_ms + 300_000

    tokens = get_real_tokens(book_df, market_ts)
    if len(tokens) < 2:
        return None

    winning_token = resolution.get("winningTokenId") or resolution.get("winning_token")
    if not winning_token or winning_token not in tokens:
        return None

    token_a, token_b = tokens[0], tokens[1]

    books = book_df[
        (book_df.exchange_timestamp >= open_ms) &
        (book_df.exchange_timestamp <= close_ms)
    ].copy()
    books["elapsed_s"] = (books.exchange_timestamp - open_ms) / 1000

    # Get books at signal time
    books_a = books[(books.asset_id == token_a) & (books.elapsed_s <= SIGNAL_TIME_S)].sort_values("exchange_timestamp")
    books_b = books[(books.asset_id == token_b) & (books.elapsed_s <= SIGNAL_TIME_S)].sort_values("exchange_timestamp")

    if len(books_a) == 0 or len(books_b) == 0:
        return None

    snap_a = books_a.iloc[-1]
    snap_b = books_b.iloc[-1]
    mid_a = snap_a.mid_price
    mid_b = snap_b.mid_price
    drift_a = mid_a - 0.5
    drift_b = mid_b - 0.5

    # Identify leading token
    if abs(drift_a) > abs(drift_b):
        if drift_a > DRIFT_THRESHOLD:
            leading, leading_snap, drift = token_a, snap_a, drift_a
        elif drift_a < -DRIFT_THRESHOLD:
            leading, leading_snap, drift = token_b, snap_b, drift_b
        else:
            return None
    else:
        if drift_b > DRIFT_THRESHOLD:
            leading, leading_snap, drift = token_b, snap_b, drift_b
        elif drift_b < -DRIFT_THRESHOLD:
            leading, leading_snap, drift = token_a, snap_a, drift_a
        else:
            return None

    is_winner = (leading == winning_token)

    # Entry at ask
    entry_price = leading_snap.ask_price_1
    if entry_price <= 0 or entry_price >= 1:
        return None
    n_tokens = int(BET_DOLLARS / entry_price)
    if n_tokens < 5:
        return None

    hold_pnl = n_tokens * (1.0 - entry_price) if is_winner else -(n_tokens * entry_price)

    # Post-signal book snapshots for our token, sorted by time
    post = books[
        (books.asset_id == leading) &
        (books.elapsed_s > SIGNAL_TIME_S)
    ].sort_values("exchange_timestamp")

    # Evaluate each TP level
    tp_outcomes = {}
    for tp in tp_levels:
        if tp <= entry_price:
            # TP below entry makes no sense — just hold
            tp_outcomes[tp] = (False, None, hold_pnl)
            continue

        exited = False
        exit_price = None
        pnl = hold_pnl  # default: hold to expiry

        if len(post) > 0:
            # Check if mid ever reaches TP level
            hits = post[post.mid_price >= tp]
            if len(hits) > 0:
                # Exit at the bid at the first snapshot where mid >= TP
                exit_snap = hits.iloc[0]
                exit_price = exit_snap.bid_price_1
                if exit_price > 0:
                    exited = True
                    # Sell tokens: receive exit_price per token, paid entry_price per token
                    pnl = n_tokens * (exit_price - entry_price)

        tp_outcomes[tp] = (exited, exit_price, pnl)

    slug = f"btc-updown-5m-{market_ts}"
    return TPResult(
        slug=slug,
        market_ts=market_ts,
        leading_token=leading,
        is_winner=is_winner,
        drift=drift,
        entry_price=entry_price,
        hold_pnl=hold_pnl,
        tp_outcomes=tp_outcomes,
    )


def main():
    book_files = sorted(BOOK_DIR.glob("btc-updown-5m-*.parquet"))
    resolutions = load_resolutions()

    # TP levels: from 0.55 to 0.99
    tp_levels = [round(x, 2) for x in np.arange(0.55, 1.00, 0.01)]

    print(f"Running take-profit backtest: signal={SIGNAL_TIME_S}s, drift>{DRIFT_THRESHOLD}")
    print(f"TP levels: {tp_levels[0]} to {tp_levels[-1]} ({len(tp_levels)} levels)")
    print(f"Markets: {len(book_files)}")

    results: list[TPResult] = []
    for i, bf in enumerate(book_files):
        if i % 500 == 0 and i > 0:
            print(f"  {i}/{len(book_files)}...")

        stem = bf.stem
        market_ts = parse_slug_timestamp(stem + ".parquet")
        resolution = resolutions.get(stem)
        if resolution is None:
            continue

        try:
            book_df = pq.read_table(bf).to_pandas()
        except Exception:
            continue

        if len(book_df) < 20:
            continue

        result = simulate_market(book_df, market_ts, resolution, tp_levels)
        if result is not None:
            results.append(result)

    results.sort(key=lambda r: r.market_ts)
    n = len(results)
    wins = sum(1 for r in results if r.is_winner)

    print(f"\nTrades: {n} | Win rate: {wins}/{n} ({wins/n:.1%})")

    # Hold-to-expiry baseline
    hold_total = sum(r.hold_pnl for r in results)
    cum = peak = max_dd = 0.0
    for r in results:
        cum += r.hold_pnl
        peak = max(peak, cum)
        max_dd = max(max_dd, peak - cum)
    print(f"\nHOLD baseline: PnL={hold_total:+.0f}  MaxDD={max_dd:.0f}")

    # Summary table per TP level
    print(f"\n{'TP':>5s} {'Exits':>6s} {'Exit%':>6s} {'PnL':>8s} {'vs Hold':>8s} "
          f"{'MaxDD':>7s} {'AvgExit':>8s} {'Sharpe':>7s}")
    print("-" * 62)

    for tp in tp_levels:
        pnls = []
        exits = 0
        exit_prices = []
        for r in results:
            exited, exit_price, pnl = r.tp_outcomes[tp]
            pnls.append(pnl)
            if exited:
                exits += 1
                exit_prices.append(exit_price)

        total_pnl = sum(pnls)
        cum = peak = max_dd = 0.0
        for p in pnls:
            cum += p
            peak = max(peak, cum)
            max_dd = max(max_dd, peak - cum)

        avg_exit = np.mean(exit_prices) if exit_prices else 0
        sharpe = np.mean(pnls) / np.std(pnls) * np.sqrt(n) if np.std(pnls) > 0 else 0

        print(f"{tp:>5.2f} {exits:>6d} {exits/n:>5.1%} {total_pnl:>+8.0f} "
              f"{total_pnl - hold_total:>+8.0f} {max_dd:>7.0f} "
              f"{avg_exit:>8.3f} {sharpe:>7.2f}")

    # Detailed look at reversal trades (won signal but lost money holding)
    reversals = [r for r in results if r.is_winner and r.hold_pnl > 0]
    non_reversals_won = len(reversals)
    reversals_lost = [r for r in results if not r.is_winner]

    # How many losers could have been saved by TP?
    print(f"\n--- Reversal protection analysis ---")
    for tp in [0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
        saved = 0
        saved_pnl = 0.0
        missed_upside = 0.0
        for r in results:
            exited, exit_price, tp_pnl = r.tp_outcomes[tp]
            if exited:
                if not r.is_winner:
                    # Would have lost holding, but TP got us out with profit
                    saved += 1
                    saved_pnl += tp_pnl  # positive (exited before reversal)
                else:
                    # Winner — did we leave money on the table?
                    missed_upside += r.hold_pnl - tp_pnl

        print(f"  TP={tp:.2f}: saved {saved} losers (recovered {saved_pnl:+.0f}), "
              f"missed upside on winners: {missed_upside:.0f}")


if __name__ == "__main__":
    main()
