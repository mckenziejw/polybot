#!/usr/bin/env python3
"""
Two analyses:

1) PnL by entry price bucket — are 0.90+ entries profitable despite thin margin?
2) Late-market loser activity — is there demand for loser tokens near expiry?
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

    print(f"Entry price analysis: signal={SIGNAL_TIME_S}s, drift>{DRIFT_THRESHOLD}")

    # Per-trade data
    trades = []  # (entry_ask, is_winner, drift, pnl, loser_late_bid, loser_late_ask, loser_late_volume)

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
                winner_snap, loser_snap = snap_a, snap_b
                winner_id, loser_id = token_a, token_b
                drift = drift_a
            elif drift_a < -DRIFT_THRESHOLD:
                winner_snap, loser_snap = snap_b, snap_a
                winner_id, loser_id = token_b, token_a
                drift = drift_b
            else:
                continue
        else:
            if drift_b > DRIFT_THRESHOLD:
                winner_snap, loser_snap = snap_b, snap_a
                winner_id, loser_id = token_b, token_a
                drift = drift_b
            elif drift_b < -DRIFT_THRESHOLD:
                winner_snap, loser_snap = snap_a, snap_b
                winner_id, loser_id = token_a, token_b
                drift = drift_a
            else:
                continue

        is_winner = (winner_id == winning_token)
        entry_ask = winner_snap.ask_price_1

        if entry_ask <= 0 or entry_ask >= 1:
            continue

        tokens_bought = int(BET_DOLLARS / entry_ask)
        if tokens_bought < 5:
            continue

        pnl = tokens_bought * (1.0 - entry_ask) if is_winner else -(tokens_bought * entry_ask)

        # Late-market loser activity (last 60s of the 300s market = elapsed 240-300s)
        late_loser = books[
            (books.asset_id == loser_id) &
            (books.elapsed_s >= 240)
        ]

        if len(late_loser) > 0:
            loser_late_bid = late_loser.bid_price_1.mean()
            loser_late_ask = late_loser.ask_price_1.mean()
            # Check if there's any bid activity (people wanting to buy the loser)
            loser_late_has_bids = (late_loser.bid_size_1 > 0).any()
            loser_late_max_bid = late_loser.bid_price_1.max()
            loser_late_snapshots = len(late_loser)
        else:
            loser_late_bid = 0
            loser_late_ask = 0
            loser_late_has_bids = False
            loser_late_max_bid = 0
            loser_late_snapshots = 0

        trades.append({
            "entry_ask": entry_ask,
            "is_winner": is_winner,
            "drift": abs(drift),
            "pnl": pnl,
            "tokens": tokens_bought,
            "loser_late_bid": loser_late_bid,
            "loser_late_ask": loser_late_ask,
            "loser_late_has_bids": loser_late_has_bids,
            "loser_late_max_bid": loser_late_max_bid,
            "loser_late_snapshots": loser_late_snapshots,
        })

    n = len(trades)
    print(f"\nTrades: {n}")

    # ══════════════════════════════════════════════════════════════
    # ANALYSIS 1: PnL by entry price
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("PNL BY ENTRY PRICE (winner ask at signal time)")
    print(f"{'='*80}")

    buckets = [
        (0.50, 0.55, "0.50-0.55"),
        (0.55, 0.60, "0.55-0.60"),
        (0.60, 0.65, "0.60-0.65"),
        (0.65, 0.70, "0.65-0.70"),
        (0.70, 0.75, "0.70-0.75"),
        (0.75, 0.80, "0.75-0.80"),
        (0.80, 0.85, "0.80-0.85"),
        (0.85, 0.90, "0.85-0.90"),
        (0.90, 0.95, "0.90-0.95"),
        (0.95, 1.00, "0.95-1.00"),
    ]

    print(f"\n{'Bucket':>12s} | {'N':>5s} {'WR':>6s} | {'PnL':>8s} {'AvgPnL':>8s} | {'AvgProfit':>9s} {'AvgLoss':>8s} | {'EV':>7s}")
    print("-" * 80)

    for lo, hi, label in buckets:
        bucket = [t for t in trades if lo <= t["entry_ask"] < hi]
        if len(bucket) == 0:
            continue
        bn = len(bucket)
        wins = sum(1 for t in bucket if t["is_winner"])
        wr = wins / bn
        total_pnl = sum(t["pnl"] for t in bucket)
        avg_pnl = total_pnl / bn

        win_pnls = [t["pnl"] for t in bucket if t["is_winner"]]
        loss_pnls = [t["pnl"] for t in bucket if not t["is_winner"]]
        avg_win = np.mean(win_pnls) if win_pnls else 0
        avg_loss = np.mean(loss_pnls) if loss_pnls else 0

        # Expected value per $100 bet
        ev = wr * avg_win + (1 - wr) * avg_loss

        print(f"{label:>12s} | {bn:>5d} {wr:>5.1%} | {total_pnl:>+8.0f} {avg_pnl:>+8.2f} | "
              f"{avg_win:>+9.2f} {avg_loss:>+8.2f} | {ev:>+7.2f}")

    # Cumulative PnL from dropping cheap entries
    print(f"\n{'='*80}")
    print("CUMULATIVE PNL — WHAT IF WE RAISE THE MINIMUM ENTRY PRICE?")
    print(f"{'='*80}")

    for min_ask in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]:
        filtered = [t for t in trades if t["entry_ask"] >= min_ask]
        if len(filtered) < 10:
            continue
        fn = len(filtered)
        fpnl = sum(t["pnl"] for t in filtered)
        fwr = sum(1 for t in filtered if t["is_winner"]) / fn

        # Sharpe
        pnls = [t["pnl"] for t in filtered]
        sharpe = np.mean(pnls) / np.std(pnls) * np.sqrt(fn) if np.std(pnls) > 0 else 0

        # Max DD
        cum = peak = max_dd = 0.0
        for t in sorted(filtered, key=lambda x: x["entry_ask"]):  # not time-sorted but gives rough idea
            cum += t["pnl"]
            peak = max(peak, cum)
            max_dd = max(max_dd, peak - cum)

        print(f"  ask >= {min_ask:.2f}: {fn:>5d} trades, WR={fwr:.1%}, PnL={fpnl:>+8.0f}, Sharpe={sharpe:.2f}")

    # ══════════════════════════════════════════════════════════════
    # ANALYSIS 2: Late-market loser activity
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("LATE-MARKET LOSER SIDE ACTIVITY (last 60s of market)")
    print(f"{'='*80}")

    has_late_bids = [t for t in trades if t["loser_late_has_bids"]]
    no_late_bids = [t for t in trades if not t["loser_late_has_bids"]]

    print(f"\n  Markets with loser bids in last 60s: {len(has_late_bids)}/{n} ({len(has_late_bids)/n:.1%})")
    print(f"  Markets without loser bids:          {len(no_late_bids)}/{n}")

    if has_late_bids:
        late_bids = [t["loser_late_bid"] for t in has_late_bids if t["loser_late_bid"] > 0]
        late_max_bids = [t["loser_late_max_bid"] for t in has_late_bids if t["loser_late_max_bid"] > 0]

        print(f"\n  Late loser bid (mean): mean={np.mean(late_bids):.4f}  median={np.median(late_bids):.4f}")
        print(f"  Late loser bid (max):  mean={np.mean(late_max_bids):.4f}  median={np.median(late_max_bids):.4f}")

        # How much could we sell losers for near expiry?
        print(f"\n  Distribution of late loser max bid:")
        for threshold in [0.01, 0.02, 0.03, 0.05, 0.10, 0.15, 0.20]:
            count = sum(1 for b in late_max_bids if b >= threshold)
            print(f"    bid >= {threshold:.2f}: {count}/{len(late_max_bids)} ({count/len(late_max_bids):.1%})")

    # What would selling losers near expiry add to PnL?
    print(f"\n{'='*80}")
    print("LOTTERY TICKET SELLING — POTENTIAL EXTRA PNL")
    print(f"{'='*80}")

    # For winning trades: we could sell loser tokens near expiry for extra profit
    # For losing trades: the loser is actually the winner, so its late bid would be high
    # We only want to sell when we correctly picked the winner
    winner_trades = [t for t in trades if t["is_winner"] and t["loser_late_max_bid"] > 0]
    loser_trades = [t for t in trades if not t["is_winner"] and t["loser_late_max_bid"] > 0]

    print(f"\n  Winning trades with loser bids near expiry: {len(winner_trades)}")
    if winner_trades:
        # If we minted and held losers, we could sell them for the late bid
        # But in the current strategy we only bought winners, we don't hold losers
        # This only applies if we ALSO mint pairs
        late_loser_bids = [t["loser_late_max_bid"] for t in winner_trades]
        print(f"  Late loser max bid when we're right: mean={np.mean(late_loser_bids):.4f}  median={np.median(late_loser_bids):.4f}")

        # Extra PnL from selling 10 minted loser tokens at late bid
        # (only on markets where signal fires AND we win)
        for mint_qty in [5, 10]:
            extra = sum(mint_qty * t["loser_late_max_bid"] for t in winner_trades)
            # But subtract the cost of minting: $mint_qty per market with signal
            signal_markets = len(trades)
            # Minted pairs that don't get signal = hedged, net $0
            # Minted pairs with signal, winner: hold winner (expires $1) + sell loser at bid
            # Net gain from mint on winners = mint_qty * (1 + loser_bid) - mint_qty = mint_qty * loser_bid
            # Minted pairs with signal, loser: hold winner (expires $0) + loser (is actually winner, expires $1)
            # But we sold the wrong one... actually we don't sell in the direct buy strategy
            # This analysis is for: "what if we ALSO mint pairs alongside buying winner"

            # Simpler framing: for each market, we mint N pairs ($N cost).
            # If signal fires and we're right: we hold N winner tokens (worth $N at expiry)
            #   AND N loser tokens we can sell near expiry for N * late_bid
            #   Net: N * (1 + late_bid) - N = N * late_bid (extra profit)
            # If signal fires and we're wrong: we hold N of our "winner" (worth $0)
            #   AND N of our "loser" (actually wins, worth $N)
            #   Net: 0 + N - N = $0 (break even on mint, but we also lost on the buy)
            # If no signal: N winner + N loser at expiry, one pays $N, net $0

            extra_winners = sum(mint_qty * t["loser_late_max_bid"] for t in winner_trades)
            # Losers break even on the mint (the "loser" we hold is actually the winner)
            extra_losers = 0  # break even

            print(f"\n  If we also mint {mint_qty} pairs per market:")
            print(f"    Extra from selling losers near expiry (winners only): +${extra_winners:.0f}")
            print(f"    Gas cost (~$0.01/split × {signal_markets} markets): -${signal_markets * 0.01:.0f}")
            print(f"    Net extra PnL: +${extra_winners - signal_markets * 0.01:.0f}")
            print(f"    Per-trade avg extra: +${extra_winners / len(trades):.2f}")


if __name__ == "__main__":
    main()
