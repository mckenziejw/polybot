#!/usr/bin/env python3
"""
Backtest: Greedy maker sell with taker fallback.

Post sell at ask+N ticks. If not filled within T seconds, cross the spread
and sell at the current bid (taker). Since we already hold minted tokens,
selling at any price > 0 reduces our effective cost below $1.

Compares:
  A) Maker-only at ask+0 (baseline)
  B) Greedy maker at ask+N, taker fallback after T seconds
  C) For reference: direct buy (no minting)

The taker fallback uses the loser's bid price at fallback time.
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

# Sweep parameters
OFFSETS = [0, 1, 2, 3, 4, 5]
FALLBACK_DELAYS = [10, 20, 30, 60]  # seconds after signal to switch to taker


@dataclass
class TradeResult:
    slug: str
    is_winner: bool
    drift: float
    loser_ask: float
    loser_bid_at_signal: float
    # For each (offset, delay) combo: (sell_price, filled_maker, fallback_bid, effective_cost, pnl)
    combo_results: dict


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

    # Post-signal loser books
    post_signal_loser = books[
        (books.asset_id == loser_id) &
        (books.elapsed_s > SIGNAL_TIME_S)
    ].sort_values("exchange_timestamp")

    combo_results = {}

    for offset in OFFSETS:
        sell_price = round(loser_ask + offset * TICK_SIZE, 2)
        if sell_price >= 1.0 or sell_price <= 0:
            for delay in FALLBACK_DELAYS:
                combo_results[(offset, delay)] = (sell_price, False, 0.0, 1.0, 0.0)
            continue

        for delay in FALLBACK_DELAYS:
            fallback_time = SIGNAL_TIME_S + delay

            # Check if maker fills before fallback time
            pre_fallback = post_signal_loser[post_signal_loser.elapsed_s <= fallback_time]
            if len(pre_fallback) > 0:
                filled_maker = (pre_fallback.bid_price_1 >= sell_price).any()
                if not filled_maker:
                    filled_maker = ((pre_fallback.ask_price_1 >= sell_price) &
                                    (pre_fallback.mid_price >= sell_price)).any()
            else:
                filled_maker = False

            if filled_maker:
                # Maker fill: effective cost = 1.00 - sell_price
                effective_cost = 1.00 - sell_price
                fallback_bid = 0.0
            else:
                # Taker fallback: sell at bid at fallback time
                at_fallback = post_signal_loser[post_signal_loser.elapsed_s >= fallback_time]
                if len(at_fallback) > 0:
                    fallback_bid = at_fallback.iloc[0].bid_price_1
                elif len(post_signal_loser) > 0:
                    # Market might end before fallback; use last available bid
                    fallback_bid = post_signal_loser.iloc[-1].bid_price_1
                else:
                    fallback_bid = loser_bid  # use signal-time bid

                if fallback_bid <= 0:
                    fallback_bid = 0.01  # minimum
                effective_cost = 1.00 - fallback_bid

            if effective_cost <= 0 or effective_cost >= 1:
                combo_results[(offset, delay)] = (sell_price, filled_maker, 0.0, 1.0, 0.0)
                continue

            num_tokens = int(BET_DOLLARS / effective_cost)
            if num_tokens < 5:
                combo_results[(offset, delay)] = (sell_price, filled_maker, 0.0, effective_cost, 0.0)
                continue

            pnl = num_tokens * (1.0 - effective_cost) if is_winner else -(num_tokens * effective_cost)
            combo_results[(offset, delay)] = (sell_price, filled_maker, fallback_bid if not filled_maker else 0.0, effective_cost, pnl)

    return TradeResult(
        slug=f"btc-updown-5m-{market_ts}",
        is_winner=is_winner,
        drift=drift,
        loser_ask=loser_ask,
        loser_bid_at_signal=loser_bid,
        combo_results=combo_results,
    )


def main():
    book_files = sorted(BOOK_DIR.glob("btc-updown-5m-*.parquet"))
    resolutions = load_resolutions()

    print(f"Greedy maker + taker fallback sweep")
    print(f"Signal={SIGNAL_TIME_S}s, drift>{DRIFT_THRESHOLD}, bet=${BET_DOLLARS}")
    print(f"Offsets: {OFFSETS}, Fallback delays: {FALLBACK_DELAYS}s")
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

    # ── Results table ──
    print(f"\n{'='*90}")
    print("GREEDY MAKER + TAKER FALLBACK")
    print(f"{'='*90}")
    print(f"\n{'Offset':>8s} {'Delay':>6s} | {'MakerFill':>9s} {'EffCost':>8s} | {'PnL':>8s} {'MaxDD':>6s} {'Sharpe':>7s}")
    print("-" * 70)

    for offset in OFFSETS:
        for delay in FALLBACK_DELAYS:
            maker_fills = sum(1 for r in results if r.combo_results[(offset, delay)][1])
            eff_costs = [r.combo_results[(offset, delay)][3] for r in results
                         if r.combo_results[(offset, delay)][3] > 0 and r.combo_results[(offset, delay)][3] < 1]

            # Equity curve
            cum = peak = max_dd = 0.0
            pnls = []
            for r in results:
                p = r.combo_results[(offset, delay)][4]
                pnls.append(p)
                cum += p
                peak = max(peak, cum)
                max_dd = max(max_dd, peak - cum)

            sharpe = np.mean(pnls) / np.std(pnls) * np.sqrt(n) if np.std(pnls) > 0 else 0
            total_pnl = sum(pnls)

            label = f"ask+{offset}" if offset > 0 else "ask"
            print(f"{label:>8s} {delay:>5d}s | {maker_fills:>4d}/{n} {np.mean(eff_costs):>7.4f} | "
                  f"{total_pnl:>+8.0f} {max_dd:>6.0f} {sharpe:>7.2f}")
        print()

    # ── Best combos ──
    print(f"\n{'='*90}")
    print("TOP 5 CONFIGURATIONS BY SHARPE")
    print(f"{'='*90}")

    all_combos = []
    for offset in OFFSETS:
        for delay in FALLBACK_DELAYS:
            pnls = [r.combo_results[(offset, delay)][4] for r in results]
            total = sum(pnls)
            sharpe = np.mean(pnls) / np.std(pnls) * np.sqrt(n) if np.std(pnls) > 0 else 0
            maker_fills = sum(1 for r in results if r.combo_results[(offset, delay)][1])

            cum = peak = max_dd = 0.0
            for p in pnls:
                cum += p
                peak = max(peak, cum)
                max_dd = max(max_dd, peak - cum)

            all_combos.append((offset, delay, total, max_dd, sharpe, maker_fills))

    all_combos.sort(key=lambda x: x[4], reverse=True)
    for offset, delay, total, max_dd, sharpe, maker_fills in all_combos[:5]:
        label = f"ask+{offset}" if offset > 0 else "ask"
        print(f"  {label} / {delay}s fallback: PnL={total:+.0f}  MaxDD={max_dd:.0f}  "
              f"Sharpe={sharpe:.2f}  MakerFill={maker_fills}/{n} ({maker_fills/n:.1%})")

    # ── Fallback cost analysis ──
    print(f"\n{'='*90}")
    print("FALLBACK COST: How much worse is the taker fallback vs maker fill?")
    print(f"{'='*90}")

    for offset in [2, 3, 5]:
        for delay in FALLBACK_DELAYS:
            maker_costs = []
            fallback_costs = []
            for r in results:
                sell_price, filled, fb_bid, eff_cost, pnl = r.combo_results[(offset, delay)]
                if filled:
                    maker_costs.append(eff_cost)
                else:
                    fallback_costs.append(eff_cost)

            if maker_costs and fallback_costs:
                label = f"ask+{offset}/{delay}s"
                print(f"  {label:>12s}: maker_cost={np.mean(maker_costs):.4f} ({len(maker_costs)})  "
                      f"fallback_cost={np.mean(fallback_costs):.4f} ({len(fallback_costs)})  "
                      f"delta={np.mean(fallback_costs) - np.mean(maker_costs):+.4f}")


if __name__ == "__main__":
    main()
