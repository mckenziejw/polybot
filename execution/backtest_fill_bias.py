#!/usr/bin/env python3
"""
Investigate: Are unfilled maker trades disproportionately winners?

If so, the maker strategy selectively misses the most profitable trades,
explaining the large PnL gap vs direct buy despite 92.4% fill rate.
"""

import json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path

BOOK_DIR = Path("/home/josh/Documents/projects/wss_test/data/book_snapshots")
RESOLUTION_CACHE = Path("/home/josh/Documents/projects/wss_test/data/resolution_cache.json")

SIGNAL_TIME_S = 60
DRIFT_THRESHOLD = 0.10
BET_DOLLARS = 100


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


def main():
    book_files = sorted(BOOK_DIR.glob("btc-updown-5m-*.parquet"))
    resolutions = load_resolutions()

    print(f"Fill bias analysis: signal={SIGNAL_TIME_S}s, drift>{DRIFT_THRESHOLD}")

    filled_wins = 0
    filled_losses = 0
    unfilled_wins = 0
    unfilled_losses = 0

    filled_pnls = []
    unfilled_direct_pnls = []
    filled_drifts = []
    unfilled_drifts = []

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

        open_ms = market_ts * 1000
        close_ms = open_ms + 300_000

        tokens = get_real_tokens(book_df, market_ts)
        if len(tokens) < 2:
            continue

        winning_token = resolution.get("winningTokenId") or resolution.get("winning_token")
        if not winning_token or winning_token not in tokens:
            continue

        token_a, token_b = tokens[0], tokens[1]

        books = book_df[
            (book_df.exchange_timestamp >= open_ms) &
            (book_df.exchange_timestamp <= close_ms)
        ].copy()
        books["elapsed_s"] = (books.exchange_timestamp - open_ms) / 1000

        books_a = books[(books.asset_id == token_a) & (books.elapsed_s <= SIGNAL_TIME_S)].sort_values("exchange_timestamp")
        books_b = books[(books.asset_id == token_b) & (books.elapsed_s <= SIGNAL_TIME_S)].sort_values("exchange_timestamp")

        if len(books_a) == 0 or len(books_b) == 0:
            continue

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
                continue
        else:
            if drift_b > DRIFT_THRESHOLD:
                loser_snap, winner_id, loser_id = snap_a, token_b, token_a
                drift = drift_b
            elif drift_b < -DRIFT_THRESHOLD:
                loser_snap, winner_id, loser_id = snap_b, token_a, token_b
                drift = drift_a
            else:
                continue

        is_winner = (winner_id == winning_token)
        loser_ask = loser_snap.ask_price_1
        loser_bid = loser_snap.bid_price_1
        winner_ask = snap_a.ask_price_1 if winner_id == token_a else snap_b.ask_price_1

        if loser_ask <= 0 or loser_bid <= 0 or winner_ask <= 0 or winner_ask >= 1:
            continue

        # Check maker fill at ask
        post_signal_loser = books[
            (books.asset_id == loser_id) &
            (books.elapsed_s > SIGNAL_TIME_S)
        ]

        if len(post_signal_loser) > 0:
            filled = (post_signal_loser.bid_price_1 >= loser_ask).any()
            if not filled:
                filled = ((post_signal_loser.ask_price_1 >= loser_ask) &
                          (post_signal_loser.mid_price >= loser_ask)).any()
        else:
            filled = False

        # Direct buy PnL (for comparison)
        direct_tokens = int(BET_DOLLARS / winner_ask)
        if direct_tokens < 5:
            continue
        direct_pnl = direct_tokens * (1.0 - winner_ask) if is_winner else -(direct_tokens * winner_ask)

        if filled:
            if is_winner:
                filled_wins += 1
            else:
                filled_losses += 1
            filled_drifts.append(abs(drift))
            # Maker PnL
            eff_cost = 1.0 - loser_ask
            if eff_cost > 0 and eff_cost < 1:
                maker_tokens = int(BET_DOLLARS / eff_cost)
                maker_pnl = maker_tokens * (1.0 - eff_cost) if is_winner else -(maker_tokens * eff_cost)
                filled_pnls.append(maker_pnl)
        else:
            if is_winner:
                unfilled_wins += 1
            else:
                unfilled_losses += 1
            unfilled_drifts.append(abs(drift))
            unfilled_direct_pnls.append(direct_pnl)

    total = filled_wins + filled_losses + unfilled_wins + unfilled_losses
    total_filled = filled_wins + filled_losses
    total_unfilled = unfilled_wins + unfilled_losses

    print(f"\nTotal trades: {total}")
    print(f"\n{'='*70}")
    print("FILL vs WIN RATE CORRELATION")
    print(f"{'='*70}")

    overall_wr = (filled_wins + unfilled_wins) / total
    filled_wr = filled_wins / total_filled if total_filled > 0 else 0
    unfilled_wr = unfilled_wins / total_unfilled if total_unfilled > 0 else 0

    print(f"\n  Overall win rate:   {filled_wins + unfilled_wins}/{total} ({overall_wr:.1%})")
    print(f"  Filled win rate:   {filled_wins}/{total_filled} ({filled_wr:.1%})")
    print(f"  Unfilled win rate: {unfilled_wins}/{total_unfilled} ({unfilled_wr:.1%})")

    print(f"\n  Unfilled breakdown: {unfilled_wins} wins + {unfilled_losses} losses = {total_unfilled} trades")

    print(f"\n{'='*70}")
    print("DRIFT MAGNITUDE: FILLED vs UNFILLED")
    print(f"{'='*70}")

    print(f"\n  Filled drift:   mean={np.mean(filled_drifts):.4f}  median={np.median(filled_drifts):.4f}")
    print(f"  Unfilled drift: mean={np.mean(unfilled_drifts):.4f}  median={np.median(unfilled_drifts):.4f}")

    print(f"\n{'='*70}")
    print("PNL IMPACT")
    print(f"{'='*70}")

    print(f"\n  Filled maker PnL:     {sum(filled_pnls):+.0f}  ({len(filled_pnls)} trades, avg={np.mean(filled_pnls):+.2f})")
    print(f"  Unfilled direct PnL:  {sum(unfilled_direct_pnls):+.0f}  ({len(unfilled_direct_pnls)} trades, avg={np.mean(unfilled_direct_pnls):+.2f})")
    print(f"  -> PnL left on table by not having fallback: {sum(unfilled_direct_pnls):+.0f}")

    # Win rate by drift bucket for filled vs unfilled
    print(f"\n{'='*70}")
    print("WIN RATE BY DRIFT: FILLED vs UNFILLED")
    print(f"{'='*70}")

    all_drifts = filled_drifts + unfilled_drifts
    for lo, hi, label in [(0.10, 0.15, "10-15%"), (0.15, 0.20, "15-20%"),
                          (0.20, 0.30, "20-30%"), (0.30, 0.50, "30-50%")]:
        # This is trickier - we need per-trade data
        pass

    print("\n  (See fill rate by drift in the offset sweep backtest)")


if __name__ == "__main__":
    main()
