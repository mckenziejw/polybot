#!/usr/bin/env python3
"""
Backtest: Stop-loss on mid-drift strategy.

After buying the winner at the ask at t=60s, monitor the winner's bid price.
If it drops below a stop-loss threshold, sell at the bid to cut the loss.

Sweep:
  - Absolute SL levels (0.30, 0.40, 0.50, 0.55, 0.60, 0.65)
  - Relative SL (drop from entry: -5%, -10%, -15%, -20%, -30%)
  - Time-based: only apply SL after N seconds, or only before N seconds remaining
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

    print(f"Stop-loss backtest: signal={SIGNAL_TIME_S}s, drift>{DRIFT_THRESHOLD}")

    # Collect per-trade data
    trades = []

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
                winner_id, loser_id = token_a, token_b
                drift = drift_a
            elif drift_a < -DRIFT_THRESHOLD:
                winner_id, loser_id = token_b, token_a
                drift = drift_b
            else:
                continue
        else:
            if drift_b > DRIFT_THRESHOLD:
                winner_id, loser_id = token_b, token_a
                drift = drift_b
            elif drift_b < -DRIFT_THRESHOLD:
                winner_id, loser_id = token_a, token_b
                drift = drift_a
            else:
                continue

        is_winner = (winner_id == winning_token)
        winner_snap = snap_a if winner_id == token_a else snap_b
        entry_ask = winner_snap.ask_price_1

        if entry_ask <= 0 or entry_ask >= 1:
            continue

        tokens_bought = int(BET_DOLLARS / entry_ask)
        if tokens_bought < 5:
            continue

        # Get post-signal winner book snapshots (bid prices for SL check)
        post_signal_winner = books[
            (books.asset_id == winner_id) &
            (books.elapsed_s > SIGNAL_TIME_S)
        ].sort_values("exchange_timestamp")

        # Build time series of (elapsed_s, bid_price) for SL simulation
        bid_series = []
        for _, row in post_signal_winner.iterrows():
            if row.bid_price_1 > 0:
                bid_series.append((row.elapsed_s, row.bid_price_1))

        trades.append({
            "is_winner": is_winner,
            "drift": abs(drift),
            "entry_ask": entry_ask,
            "tokens": tokens_bought,
            "bid_series": bid_series,
            "min_bid": min((b for _, b in bid_series), default=0) if bid_series else 0,
            "hold_pnl": tokens_bought * (1.0 - entry_ask) if is_winner else -(tokens_bought * entry_ask),
        })

    n = len(trades)
    print(f"\nTrades: {n}")

    hold_pnl = sum(t["hold_pnl"] for t in trades)
    hold_wins = sum(1 for t in trades if t["is_winner"])
    print(f"Hold to expiry: PnL={hold_pnl:+.0f}, WR={hold_wins}/{n} ({hold_wins/n:.1%})")

    def simulate_sl(trades, sl_fn, label):
        """Simulate stop-loss. sl_fn(entry_ask, elapsed, bid) -> True if should stop out."""
        pnls = []
        stopped = 0
        stopped_would_win = 0
        stopped_would_lose = 0

        for t in trades:
            entry = t["entry_ask"]
            tokens = t["tokens"]
            hit_sl = False

            for elapsed, bid in t["bid_series"]:
                if sl_fn(entry, elapsed, bid):
                    # Stop out: sell at bid
                    sl_pnl = tokens * (bid - entry)
                    pnls.append(sl_pnl)
                    stopped += 1
                    if t["is_winner"]:
                        stopped_would_win += 1
                    else:
                        stopped_would_lose += 1
                    hit_sl = True
                    break

            if not hit_sl:
                pnls.append(t["hold_pnl"])

        total = sum(pnls)
        sharpe = np.mean(pnls) / np.std(pnls) * np.sqrt(n) if np.std(pnls) > 0 else 0

        cum = peak = max_dd = 0.0
        for p in pnls:
            cum += p
            peak = max(peak, cum)
            max_dd = max(max_dd, peak - cum)

        return total, sharpe, max_dd, stopped, stopped_would_win, stopped_would_lose

    # ══════════════════════════════════════════════════════════════
    # ANALYSIS 1: Absolute stop-loss levels
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*90}")
    print("ABSOLUTE STOP-LOSS (sell if winner bid drops below level)")
    print(f"{'='*90}")

    print(f"\n{'SL Level':>10s} | {'PnL':>8s} {'Sharpe':>7s} {'MaxDD':>7s} | {'Stopped':>7s} {'SavedL':>6s} {'KilledW':>7s}")
    print("-" * 70)

    for sl_level in [0.20, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]:
        total, sharpe, max_dd, stopped, sw, sl_val = simulate_sl(
            trades,
            lambda entry, elapsed, bid, _sl=sl_level: bid < _sl,
            f"bid<{sl_level}"
        )
        print(f"  bid<{sl_level:.2f} | {total:>+8.0f} {sharpe:>7.2f} {max_dd:>7.0f} | "
              f"{stopped:>7d} {sl_val:>6d} {sw:>7d}")

    print(f"\n  (hold)   | {hold_pnl:>+8.0f}   1.43    2884 |       0      0       0")

    # ══════════════════════════════════════════════════════════════
    # ANALYSIS 2: Relative stop-loss (% drop from entry)
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*90}")
    print("RELATIVE STOP-LOSS (sell if bid drops X% below entry ask)")
    print(f"{'='*90}")

    print(f"\n{'Drop':>10s} | {'PnL':>8s} {'Sharpe':>7s} {'MaxDD':>7s} | {'Stopped':>7s} {'SavedL':>6s} {'KilledW':>7s}")
    print("-" * 70)

    for drop_pct in [5, 10, 15, 20, 25, 30, 40, 50]:
        total, sharpe, max_dd, stopped, sw, sl_val = simulate_sl(
            trades,
            lambda entry, elapsed, bid, _d=drop_pct: bid < entry * (1 - _d / 100),
            f"-{drop_pct}%"
        )
        print(f"  -{drop_pct:>3d}%    | {total:>+8.0f} {sharpe:>7.2f} {max_dd:>7.0f} | "
              f"{stopped:>7d} {sl_val:>6d} {sw:>7d}")

    # ══════════════════════════════════════════════════════════════
    # ANALYSIS 3: Time-gated stop-loss
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*90}")
    print("TIME-GATED STOP-LOSS (only check SL within a time window)")
    print(f"{'='*90}")
    print(f"\nAbsolute SL=0.45, but only active during specified window.\n")

    print(f"{'Window':>20s} | {'PnL':>8s} {'Sharpe':>7s} {'MaxDD':>7s} | {'Stopped':>7s} {'SavedL':>6s} {'KilledW':>7s}")
    print("-" * 75)

    sl_level = 0.45
    for window_label, time_fn in [
        ("60-120s", lambda e: 60 < e <= 120),
        ("60-180s", lambda e: 60 < e <= 180),
        ("60-240s", lambda e: 60 < e <= 240),
        ("120-240s", lambda e: 120 < e <= 240),
        ("180-300s", lambda e: 180 < e <= 300),
        ("60-300s (always)", lambda e: 60 < e <= 300),
    ]:
        total, sharpe, max_dd, stopped, sw, sl_val = simulate_sl(
            trades,
            lambda entry, elapsed, bid, _sl=sl_level, _tf=time_fn: _tf(elapsed) and bid < _sl,
            window_label
        )
        print(f"  {window_label:>18s} | {total:>+8.0f} {sharpe:>7.2f} {max_dd:>7.0f} | "
              f"{stopped:>7d} {sl_val:>6d} {sw:>7d}")

    # ══════════════════════════════════════════════════════════════
    # ANALYSIS 4: How many winners dip below various levels?
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*90}")
    print("WINNER DIP ANALYSIS — How often does the actual winner temporarily dip?")
    print(f"{'='*90}")

    winners = [t for t in trades if t["is_winner"]]
    losers = [t for t in trades if not t["is_winner"]]

    print(f"\n  Winners: {len(winners)}, Losers: {len(losers)}")

    for level in [0.30, 0.40, 0.50, 0.55, 0.60]:
        w_dips = sum(1 for t in winners if t["min_bid"] < level)
        l_dips = sum(1 for t in losers if t["min_bid"] < level)
        print(f"  Bid dips below {level:.2f}: {w_dips}/{len(winners)} winners ({w_dips/len(winners):.1%}), "
              f"{l_dips}/{len(losers)} losers ({l_dips/len(losers):.1%})")

    # ══════════════════════════════════════════════════════════════
    # ANALYSIS 5: Best combo sweep
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*90}")
    print("BEST CONFIGURATIONS BY SHARPE")
    print(f"{'='*90}")

    configs = []
    for sl_level in [0.30, 0.35, 0.40, 0.45, 0.50]:
        for drop_pct in [10, 15, 20, 25, 30]:
            # Combined: absolute AND relative
            total, sharpe, max_dd, stopped, sw, sl_val = simulate_sl(
                trades,
                lambda entry, elapsed, bid, _sl=sl_level, _d=drop_pct:
                    bid < _sl and bid < entry * (1 - _d / 100),
                f"abs<{sl_level} & rel<-{drop_pct}%"
            )
            configs.append((f"bid<{sl_level:.2f} AND -{drop_pct}%", total, sharpe, max_dd, stopped, sw, sl_val))

    configs.sort(key=lambda x: x[2], reverse=True)
    print(f"\n{'Config':>30s} | {'PnL':>8s} {'Sharpe':>7s} {'MaxDD':>7s} | {'Stopped':>7s} {'SavedL':>6s} {'KilledW':>7s}")
    print("-" * 85)
    for label, total, sharpe, max_dd, stopped, sw, sl_val in configs[:10]:
        print(f"  {label:>28s} | {total:>+8.0f} {sharpe:>7.2f} {max_dd:>7.0f} | "
              f"{stopped:>7d} {sl_val:>6d} {sw:>7d}")
    print(f"\n  {'(hold to expiry)':>28s} | {hold_pnl:>+8.0f}    1.43    2884 |       0      0       0")


if __name__ == "__main__":
    main()
