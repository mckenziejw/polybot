#!/usr/bin/env python3
"""
Backtest: Sweep sell price offsets for the mint+sell maker strategy.

Tests selling the loser side at ask, ask+1tick, ask+2ticks, ask+3ticks.
Higher sell prices mean more profit per fill but potentially lower fill rate.

For each offset, checks whether the loser bid ever reached our sell price
post-signal (i.e., would our maker order have been filled).
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
TICK_SIZE = 0.01
OFFSETS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10]  # ticks above ask


@dataclass
class TradeResult:
    slug: str
    is_winner: bool
    drift: float
    loser_ask: float
    # Per offset: (sell_price, effective_cost, filled, pnl)
    offset_results: list[tuple[float, float, bool, float]]


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


def simulate_market(book_df: pd.DataFrame, market_ts: int,
                    resolution: dict) -> TradeResult | None:
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

    # Identify winner candidate and loser
    if abs(drift_a) > abs(drift_b):
        if drift_a > DRIFT_THRESHOLD:
            loser_snap, winner_id, loser_id = snap_b, token_a, token_b
            drift = drift_a
        elif drift_a < -DRIFT_THRESHOLD:
            loser_snap, winner_id, loser_id = snap_a, token_b, token_a
            drift = drift_b
        else:
            return None
    else:
        if drift_b > DRIFT_THRESHOLD:
            loser_snap, winner_id, loser_id = snap_a, token_b, token_a
            drift = drift_b
        elif drift_b < -DRIFT_THRESHOLD:
            loser_snap, winner_id, loser_id = snap_b, token_a, token_b
            drift = drift_a
        else:
            return None

    is_winner = (winner_id == winning_token)
    loser_ask = loser_snap.ask_price_1
    loser_bid = loser_snap.bid_price_1

    if loser_ask <= 0 or loser_bid <= 0:
        return None

    # Post-signal loser books for fill checking
    post_signal_loser = books[
        (books.asset_id == loser_id) &
        (books.elapsed_s > SIGNAL_TIME_S)
    ]

    offset_results = []
    for offset in OFFSETS:
        sell_price = round(loser_ask + offset * TICK_SIZE, 2)
        if sell_price >= 1.0 or sell_price <= 0:
            offset_results.append((sell_price, 0.0, False, 0.0))
            continue

        effective_cost = 1.00 - sell_price
        if effective_cost <= 0 or effective_cost >= 1:
            offset_results.append((sell_price, effective_cost, False, 0.0))
            continue

        num_tokens = int(BET_DOLLARS / effective_cost)
        if num_tokens < 5:
            offset_results.append((sell_price, effective_cost, False, 0.0))
            continue

        # Check fill: did loser bid ever reach our sell price?
        if len(post_signal_loser) > 0:
            filled = (post_signal_loser.bid_price_1 >= sell_price).any()
            if not filled:
                filled = ((post_signal_loser.ask_price_1 >= sell_price) &
                          (post_signal_loser.mid_price >= sell_price)).any()
        else:
            filled = False

        if filled:
            pnl = num_tokens * (1.0 - effective_cost) if is_winner else -(num_tokens * effective_cost)
        else:
            pnl = 0.0

        offset_results.append((sell_price, effective_cost, filled, pnl))

    slug = f"btc-updown-5m-{market_ts}"
    return TradeResult(
        slug=slug,
        is_winner=is_winner,
        drift=drift,
        loser_ask=loser_ask,
        offset_results=offset_results,
    )


def main():
    book_files = sorted(BOOK_DIR.glob("btc-updown-5m-*.parquet"))
    resolutions = load_resolutions()

    print(f"Sell price offset sweep: signal={SIGNAL_TIME_S}s, drift>{DRIFT_THRESHOLD}")
    print(f"Tick size: {TICK_SIZE}, Offsets: {OFFSETS}")
    print(f"Markets: {len(book_files)}")

    results: list[TradeResult] = []
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

        result = simulate_market(book_df, market_ts, resolution)
        if result is not None:
            results.append(result)

    results.sort(key=lambda r: r.slug)
    n = len(results)

    print(f"\nTrades: {n}")
    print(f"\n{'='*80}")
    print("SELL PRICE OFFSET COMPARISON")
    print(f"{'='*80}")

    # Also compute direct buy baseline
    direct_pnls = []
    for r in results:
        # offset=0 effective_cost is 1-ask, direct buy cost is ask_winner
        # We approximate direct buy as the offset=0 case for comparison
        pass

    for oi, offset in enumerate(OFFSETS):
        filled_count = sum(1 for r in results if r.offset_results[oi][2])
        fill_rate = filled_count / n

        # PnL for filled trades only
        maker_pnl = sum(r.offset_results[oi][3] for r in results)

        # Hybrid: maker if filled, else direct buy (offset=0, taker-like fallback)
        # For unfilled, use the offset=0 result as direct equivalent
        hybrid_pnl = 0.0
        for r in results:
            sell_price, eff_cost, filled, pnl = r.offset_results[oi]
            if filled:
                hybrid_pnl += pnl
            else:
                # Fallback: direct buy at ask_winner ≈ offset=0 effective cost
                # Use the standard direct buy PnL
                base_cost = r.offset_results[0][1]  # offset=0 effective cost
                if base_cost > 0 and base_cost < 1:
                    base_tokens = int(BET_DOLLARS / base_cost)
                    if r.is_winner:
                        hybrid_pnl += base_tokens * (1.0 - base_cost)
                    else:
                        hybrid_pnl -= base_tokens * base_cost

        # Effective costs
        eff_costs = [r.offset_results[oi][1] for r in results
                     if r.offset_results[oi][1] > 0 and r.offset_results[oi][1] < 1]

        # Equity curve stats for hybrid
        cum = peak = max_dd = 0.0
        pnls_list = []
        for r in results:
            sell_price, eff_cost, filled, pnl = r.offset_results[oi]
            if filled:
                trade_pnl = pnl
            else:
                base_cost = r.offset_results[0][1]
                if base_cost > 0 and base_cost < 1:
                    base_tokens = int(BET_DOLLARS / base_cost)
                    trade_pnl = base_tokens * (1.0 - base_cost) if r.is_winner else -(base_tokens * base_cost)
                else:
                    trade_pnl = 0.0
            pnls_list.append(trade_pnl)
            cum += trade_pnl
            peak = max(peak, cum)
            max_dd = max(max_dd, peak - cum)

        sharpe = np.mean(pnls_list) / np.std(pnls_list) * np.sqrt(n) if np.std(pnls_list) > 0 else 0

        label = f"ask+{offset}" if offset > 0 else "ask (baseline)"
        sell_price_example = round(0.40 + offset * TICK_SIZE, 2)  # illustrative

        print(f"\n  {label}:")
        print(f"    Fill rate: {filled_count}/{n} ({fill_rate:.1%})")
        print(f"    Eff cost:  mean={np.mean(eff_costs):.4f}  median={np.median(eff_costs):.4f}")
        print(f"    Maker-only PnL: {maker_pnl:+.0f}")
        print(f"    Hybrid PnL:     {hybrid_pnl:+.0f}  MaxDD={max_dd:.0f}  Sharpe={sharpe:.2f}")

    # ── Breakdown by drift magnitude ──
    print(f"\n{'='*80}")
    print("FILL RATE BY DRIFT MAGNITUDE")
    print(f"{'='*80}")

    for lo, hi, label in [(0.10, 0.15, "10-15%"), (0.15, 0.20, "15-20%"),
                          (0.20, 0.30, "20-30%"), (0.30, 0.50, "30-50%")]:
        bucket = [r for r in results if lo <= abs(r.drift) < hi]
        if len(bucket) < 10:
            continue
        bn = len(bucket)
        row = f"  drift {label:>6s} (n={bn:4d}): "
        for oi, offset in enumerate(OFFSETS):
            filled = sum(1 for r in bucket if r.offset_results[oi][2])
            row += f"  +{offset}t={filled/bn:.1%}"
        print(row)

    # ── Look at actual fill depth: how far above ask did bids go? ──
    print(f"\n{'='*80}")
    print("BID OVERSHOOT ANALYSIS (how far above loser ask did bids reach post-signal)")
    print(f"{'='*80}")

    overshoots = []
    for r in results:
        loser_ask = r.loser_ask
        # We need the raw data for this... approximate from offset fill patterns
        max_offset_filled = -1
        for oi, offset in enumerate(OFFSETS):
            if r.offset_results[oi][2]:
                max_offset_filled = offset
        if max_offset_filled >= 0:
            overshoots.append(max_offset_filled)

    if overshoots:
        print(f"\n  Among trades where ask+0 fills ({len(overshoots)}/{n}):")
        for t in OFFSETS:
            pct = sum(1 for o in overshoots if o >= t) / len(overshoots)
            print(f"    Bid reached ask+{t}: {sum(1 for o in overshoots if o >= t)}/{len(overshoots)} ({pct:.1%})")


if __name__ == "__main__":
    main()
