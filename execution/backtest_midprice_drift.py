#!/usr/bin/env python3
"""
Backtest: Mid-price drift strategy with two fill models.

For each market:
  1. Wait until time T after open
  2. Check both tokens' mid-prices; if one has drifted > threshold from 0.5, signal fires
  3. Buy the leading token

Fill models:
  A) TAKER — buy at ask_price_1 immediately (worst case, guaranteed fill)
  B) MAKER — post limit at bid_price_1, fill if price ever touches our level
     during the remaining market window (optimistic, but realistic for a passive order)

Payout: $1/token if winner, $0 if loser. PnL = payout - entry_price per token.
"""

import json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from dataclasses import dataclass

BOOK_DIR = Path("/home/josh/Documents/projects/wss_test/data/book_snapshots")
TRADE_DIR = Path("/home/josh/Documents/projects/wss_test/data/trade_events")
RESOLUTION_CACHE = Path("/home/josh/Documents/projects/wss_test/data/resolution_cache.json")


@dataclass
class TradeResult:
    slug: str
    market_ts: int
    signal_time_s: float
    leading_token: str
    is_winner: bool
    drift: float
    mid_at_signal: float
    # Taker model
    taker_entry: float
    taker_pnl: float
    # Maker model
    maker_entry: float | None
    maker_filled: bool
    maker_pnl: float


def parse_slug_timestamp(filename: str) -> int:
    return int(filename.replace("btc-updown-5m-", "").replace(".parquet", ""))


def load_resolutions() -> dict:
    cache = json.loads(RESOLUTION_CACHE.read_text())
    return cache


def get_real_tokens(book_df: pd.DataFrame, market_ts: int) -> list[str]:
    """Return the two real tradeable token IDs (those with actual orders)."""
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
                    signal_time_s: float, drift_threshold: float,
                    bet_dollars: float) -> TradeResult | None:
    """
    Simulate the mid-price drift strategy on a single market.
    """
    open_ms = market_ts * 1000
    close_ms = open_ms + 300_000

    # Get real tokens
    tokens = get_real_tokens(book_df, market_ts)
    if len(tokens) < 2:
        return None

    winning_token = resolution.get("winningTokenId") or resolution.get("winning_token")
    if not winning_token:
        return None
    if winning_token not in tokens:
        return None

    token_a, token_b = tokens[0], tokens[1]

    # Filter book data to market window
    books = book_df[
        (book_df.exchange_timestamp >= open_ms) &
        (book_df.exchange_timestamp <= close_ms)
    ].copy()
    books["elapsed_s"] = (books.exchange_timestamp - open_ms) / 1000

    # Get books at signal time for both tokens
    books_a = books[(books.asset_id == token_a) & (books.elapsed_s <= signal_time_s)].sort_values("exchange_timestamp")
    books_b = books[(books.asset_id == token_b) & (books.elapsed_s <= signal_time_s)].sort_values("exchange_timestamp")

    if len(books_a) == 0 or len(books_b) == 0:
        return None

    snap_a = books_a.iloc[-1]
    snap_b = books_b.iloc[-1]

    mid_a = snap_a.mid_price
    mid_b = snap_b.mid_price

    # Identify the leading token (the one with mid > 0.5)
    drift_a = mid_a - 0.5
    drift_b = mid_b - 0.5

    if abs(drift_a) > abs(drift_b):
        if drift_a > drift_threshold:
            leading = token_a
            leading_snap = snap_a
            drift = drift_a
        elif drift_a < -drift_threshold:
            # Token A drifted down — token B is the leader (complement)
            leading = token_b
            leading_snap = snap_b
            drift = drift_b
        else:
            return None  # No signal
    else:
        if drift_b > drift_threshold:
            leading = token_b
            leading_snap = snap_b
            drift = drift_b
        elif drift_b < -drift_threshold:
            leading = token_a
            leading_snap = snap_a
            drift = drift_a
        else:
            return None  # No signal

    is_winner = (leading == winning_token)
    mid_at_signal = leading_snap.mid_price

    # === TAKER MODEL: buy at ask ===
    taker_entry = leading_snap.ask_price_1
    if taker_entry <= 0 or taker_entry >= 1:
        return None  # Invalid book state
    taker_tokens = int(bet_dollars / taker_entry)
    if taker_tokens < 5:
        return None
    taker_pnl = taker_tokens * (1.0 - taker_entry) if is_winner else -(taker_tokens * taker_entry)

    # === MAKER MODEL: post at bid, check if filled ===
    maker_entry = leading_snap.bid_price_1
    if maker_entry <= 0 or maker_entry >= 1:
        maker_filled = False
        maker_pnl = 0.0
    else:
        # Check if price ever dips to our bid level after signal time
        # A fill happens if ask_price_1 <= our bid at any point, or if a trade occurs at our price
        post_signal_books = books[
            (books.asset_id == leading) &
            (books.elapsed_s > signal_time_s)
        ]
        # Optimistic fill: any snapshot where ask_price_1 <= our bid means we'd have filled
        # (someone crossed the spread and hit our level)
        if len(post_signal_books) > 0:
            maker_filled = (post_signal_books.ask_price_1 <= maker_entry).any()
            if not maker_filled:
                # Also check: did mid_price ever drop to our level?
                # This is more conservative — mid touching our bid means the ask was likely at our level
                maker_filled = (post_signal_books.mid_price <= maker_entry).any()
        else:
            maker_filled = False

        if maker_filled:
            maker_tokens = int(bet_dollars / maker_entry)
            if maker_tokens < 5:
                maker_filled = False
                maker_pnl = 0.0
            else:
                maker_pnl = maker_tokens * (1.0 - maker_entry) if is_winner else -(maker_tokens * maker_entry)
        else:
            maker_pnl = 0.0

    slug = f"btc-updown-5m-{market_ts}"
    return TradeResult(
        slug=slug,
        market_ts=market_ts,
        signal_time_s=signal_time_s,
        leading_token=leading,
        is_winner=is_winner,
        drift=drift,
        mid_at_signal=mid_at_signal,
        taker_entry=taker_entry,
        taker_pnl=taker_pnl,
        maker_entry=maker_entry,
        maker_filled=maker_filled,
        maker_pnl=maker_pnl if maker_filled else 0.0,
    )


def run_backtest(signal_time_s: float, drift_threshold: float, bet_dollars: float):
    """Run full backtest across all markets."""
    book_files = sorted(BOOK_DIR.glob("btc-updown-5m-*.parquet"))
    resolutions = load_resolutions()

    print(f"\n{'='*90}")
    print(f"BACKTEST: signal_time={signal_time_s}s  drift_threshold={drift_threshold}  bet=${bet_dollars}")
    print(f"{'='*90}")

    results: list[TradeResult] = []
    skipped = {"sparse": 0, "no_resolution": 0, "no_signal": 0, "bad_book": 0}

    for i, bf in enumerate(book_files):
        if i % 500 == 0 and i > 0:
            print(f"  {i}/{len(book_files)}...")

        stem = bf.stem
        market_ts = parse_slug_timestamp(stem + ".parquet")

        resolution = resolutions.get(stem)
        if resolution is None:
            skipped["no_resolution"] += 1
            continue

        try:
            book_df = pq.read_table(bf).to_pandas()
        except Exception:
            continue

        if len(book_df) < 20:
            skipped["sparse"] += 1
            continue

        result = simulate_market(book_df, market_ts, resolution, signal_time_s, drift_threshold, bet_dollars)
        if result is None:
            skipped["no_signal"] += 1
            continue

        results.append(result)

    if not results:
        print("No trades!")
        return results

    # Sort by time
    results.sort(key=lambda r: r.market_ts)

    # === Summary stats ===
    n = len(results)
    wins = sum(1 for r in results if r.is_winner)
    losses = n - wins

    print(f"\nMarkets: {len(book_files)} | Traded: {n} | Skipped: {skipped}")
    print(f"Win rate: {wins}/{n} ({wins/n:.1%})")

    # --- Taker model ---
    taker_total = sum(r.taker_pnl for r in results)
    taker_avg = np.mean([r.taker_entry for r in results])
    taker_pnls = [r.taker_pnl for r in results]

    print(f"\n--- TAKER MODEL (buy at ask) ---")
    print(f"  Avg entry price: {taker_avg:.3f}")
    print(f"  Total PnL: {taker_total:+.2f} USDC")
    print(f"  Avg PnL/trade: {np.mean(taker_pnls):+.2f}")

    # Equity curve
    cum = 0.0
    peak = 0.0
    max_dd = 0.0
    for r in results:
        cum += r.taker_pnl
        peak = max(peak, cum)
        max_dd = max(max_dd, peak - cum)
    print(f"  Final equity: {cum:+.2f}")
    print(f"  Max drawdown: {max_dd:.2f}")

    # Sharpe (per-trade)
    if np.std(taker_pnls) > 0:
        sharpe = np.mean(taker_pnls) / np.std(taker_pnls) * np.sqrt(len(taker_pnls))
        print(f"  Sharpe (annualized from per-trade): {sharpe:.2f}")

    # Loss streak
    max_streak = 0
    streak = 0
    for r in results:
        if not r.is_winner:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    print(f"  Max consecutive losses: {max_streak}")

    # --- Maker model ---
    maker_trades = [r for r in results if r.maker_filled]
    maker_unfilled = n - len(maker_trades)
    if maker_trades:
        maker_wins = sum(1 for r in maker_trades if r.is_winner)
        maker_total = sum(r.maker_pnl for r in maker_trades)
        maker_avg = np.mean([r.maker_entry for r in maker_trades])
        maker_pnls = [r.maker_pnl for r in maker_trades]

        print(f"\n--- MAKER MODEL (post at bid, fill if price crosses) ---")
        print(f"  Fill rate: {len(maker_trades)}/{n} ({len(maker_trades)/n:.1%})")
        print(f"  Win rate (filled only): {maker_wins}/{len(maker_trades)} ({maker_wins/len(maker_trades):.1%})")
        print(f"  Avg entry price: {maker_avg:.3f}")
        print(f"  Total PnL: {maker_total:+.2f} USDC")
        print(f"  Avg PnL/trade: {np.mean(maker_pnls):+.2f}")

        cum = 0.0
        peak = 0.0
        max_dd = 0.0
        for r in results:
            if r.maker_filled:
                cum += r.maker_pnl
                peak = max(peak, cum)
                max_dd = max(max_dd, peak - cum)
        print(f"  Final equity: {cum:+.2f}")
        print(f"  Max drawdown: {max_dd:.2f}")

        if np.std(maker_pnls) > 0:
            sharpe = np.mean(maker_pnls) / np.std(maker_pnls) * np.sqrt(len(maker_pnls))
            print(f"  Sharpe: {sharpe:.2f}")

        # Maker loss streak (among filled trades only)
        max_streak = 0
        streak = 0
        for r in results:
            if r.maker_filled:
                if not r.is_winner:
                    streak += 1
                    max_streak = max(max_streak, streak)
                else:
                    streak = 0
        print(f"  Max consecutive losses (filled): {max_streak}")
    else:
        print(f"\n--- MAKER MODEL: no fills ---")

    # === Entry price distribution ===
    print(f"\n--- Entry price distribution ---")
    taker_entries_win = [r.taker_entry for r in results if r.is_winner]
    taker_entries_lose = [r.taker_entry for r in results if not r.is_winner]
    print(f"  Taker (wins):  mean={np.mean(taker_entries_win):.3f}  median={np.median(taker_entries_win):.3f}")
    print(f"  Taker (losses): mean={np.mean(taker_entries_lose):.3f}  median={np.median(taker_entries_lose):.3f}")

    if maker_trades:
        maker_entries_win = [r.maker_entry for r in maker_trades if r.is_winner]
        maker_entries_lose = [r.maker_entry for r in maker_trades if not r.is_winner]
        if maker_entries_win:
            print(f"  Maker (wins):  mean={np.mean(maker_entries_win):.3f}  median={np.median(maker_entries_win):.3f}")
        if maker_entries_lose:
            print(f"  Maker (losses): mean={np.mean(maker_entries_lose):.3f}  median={np.median(maker_entries_lose):.3f}")

    # === Drift magnitude vs accuracy ===
    print(f"\n--- Accuracy by drift magnitude ---")
    for lo, hi, label in [(0.0, 0.02, "0-2%"), (0.02, 0.05, "2-5%"), (0.05, 0.10, "5-10%"),
                          (0.10, 0.20, "10-20%"), (0.20, 0.50, "20-50%")]:
        bucket = [r for r in results if lo <= abs(r.drift) < hi]
        if len(bucket) >= 10:
            bw = sum(1 for r in bucket if r.is_winner)
            avg_entry = np.mean([r.taker_entry for r in bucket])
            print(f"  drift {label:>6s}: {bw}/{len(bucket)} ({bw/len(bucket):.1%}) "
                  f"avg_ask={avg_entry:.3f} n={len(bucket)}")

    return results


def main():
    configs = [
        # (signal_time_s, drift_threshold, bet_dollars)
        (30, 0.02, 100),
        (30, 0.05, 100),
        (60, 0.02, 100),
        (60, 0.05, 100),
        (60, 0.10, 100),
        (120, 0.05, 100),
        (120, 0.10, 100),
    ]

    all_results = {}
    for signal_time, threshold, bet in configs:
        key = f"t={signal_time}s_d={threshold}"
        results = run_backtest(signal_time, threshold, bet)
        all_results[key] = results

    # === Comparison table ===
    print(f"\n{'='*90}")
    print("COMPARISON SUMMARY")
    print(f"{'='*90}")
    print(f"{'Config':<20s} {'Trades':>6s} {'WR':>6s} {'Taker PnL':>10s} {'Taker DD':>9s} "
          f"{'Maker Fill%':>11s} {'Maker PnL':>10s} {'Maker DD':>9s}")
    print("-" * 90)

    for key, results in all_results.items():
        if not results:
            continue
        n = len(results)
        wins = sum(1 for r in results if r.is_winner)
        taker_pnl = sum(r.taker_pnl for r in results)

        # Taker DD
        cum = peak = max_dd_t = 0.0
        for r in results:
            cum += r.taker_pnl
            peak = max(peak, cum)
            max_dd_t = max(max_dd_t, peak - cum)

        # Maker
        maker_filled = [r for r in results if r.maker_filled]
        fill_pct = len(maker_filled) / n if n > 0 else 0
        maker_pnl = sum(r.maker_pnl for r in maker_filled)
        cum = peak = max_dd_m = 0.0
        for r in results:
            if r.maker_filled:
                cum += r.maker_pnl
                peak = max(peak, cum)
                max_dd_m = max(max_dd_m, peak - cum)

        print(f"{key:<20s} {n:>6d} {wins/n:>5.1%} {taker_pnl:>+10.0f} {max_dd_t:>9.0f} "
              f"{fill_pct:>10.1%} {maker_pnl:>+10.0f} {max_dd_m:>9.0f}")


if __name__ == "__main__":
    main()
