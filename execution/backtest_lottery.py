#!/usr/bin/env python3
"""
Corrected lottery ticket backtest.

Scenario per market:
  1. Mint N pairs at open ($N cost) → hold N Up + N Down
  2. Buy winner at signal (separate, base strategy PnL)
  3. Near expiry: consider selling N "loser" tokens

Correct bookkeeping for the sell:
  - If we're RIGHT (72.8%): loser tokens expire worthless. Sell revenue = pure profit.
  - If we're WRONG (27.2%): "loser" tokens are actually worth $1. We sold $1 tokens
    for pennies → additional loss on top of the buy loss.

The sell is +EV when: WR * sell_price > (1-WR) * (1 - sell_price)
  i.e. sell_price > 1 - WR ≈ 0.272 (break-even price)

But conditional WR varies by market state at sell time. Low loser price near expiry
means high confidence we're right. Sweep price and time thresholds.
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
MINT_PAIRS = 10


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

    print(f"Lottery ticket backtest (corrected bookkeeping)")
    print(f"Signal={SIGNAL_TIME_S}s, drift>{DRIFT_THRESHOLD}, mint={MINT_PAIRS} pairs")

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

        base_pnl = tokens_bought * (1.0 - entry_ask) if is_winner else -(tokens_bought * entry_ask)

        # Get loser book snapshots at various time points for sell decisions
        loser_books = books[books.asset_id == loser_id].sort_values("exchange_timestamp")

        # For each time window, get the loser's bid price
        sell_opportunities = []
        for elapsed_target in range(60, 300, 5):  # every 5 seconds from signal to expiry
            remaining_s = 300 - elapsed_target
            nearby = loser_books[
                (loser_books.elapsed_s >= elapsed_target - 3) &
                (loser_books.elapsed_s <= elapsed_target + 3)
            ]
            if len(nearby) > 0:
                best = nearby.iloc[-1]
                bid = best.bid_price_1
                ask = best.ask_price_1
                if bid > 0:
                    sell_opportunities.append({
                        "elapsed": elapsed_target,
                        "remaining_s": remaining_s,
                        "bid": bid,
                        "ask": ask,
                        "mid": best.mid_price,
                    })

        trades.append({
            "is_winner": is_winner,
            "drift": abs(drift),
            "entry_ask": entry_ask,
            "base_pnl": base_pnl,
            "sell_opps": sell_opportunities,
        })

    n = len(trades)
    print(f"\nTrades: {n}")

    # ══════════════════════════════════════════════════════════════
    # ANALYSIS 1: Sell at fixed time, sweep price threshold
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*90}")
    print("SELL AT FIXED TIME — ONLY IF LOSER PRICE BELOW THRESHOLD")
    print(f"{'='*90}")
    print(f"\nSell N={MINT_PAIRS} loser tokens at bid. Only sell if bid < threshold.")
    print(f"Correct PnL: if right, +N*bid. If wrong, -N*(1-bid).\n")

    print(f"{'Time':>6s} {'Thresh':>7s} | {'Sells':>5s} {'WR':>6s} | {'Extra':>8s} {'AvgExtra':>9s} | {'Combined':>9s} {'Sharpe':>7s}")
    print("-" * 85)

    base_total = sum(t["base_pnl"] for t in trades)

    for sell_elapsed in [180, 210, 240, 270]:
        remaining = 300 - sell_elapsed
        for max_price in [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]:
            extra_pnls = []
            sells = 0
            sell_wins = 0

            for t in trades:
                # Find the sell opportunity closest to target time
                opp = None
                for o in t["sell_opps"]:
                    if abs(o["elapsed"] - sell_elapsed) <= 5:
                        opp = o
                        break

                if opp is None or opp["bid"] <= 0 or opp["bid"] >= max_price:
                    extra_pnls.append(0.0)  # don't sell, hold hedged
                    continue

                sells += 1
                sell_price = opp["bid"]

                if t["is_winner"]:
                    # Loser tokens expire worthless, we sold for sell_price each
                    extra = MINT_PAIRS * sell_price
                    sell_wins += 1
                else:
                    # "Loser" tokens are actually winners (worth $1 each)
                    # We sold $1 tokens for sell_price → lost (1 - sell_price) per token
                    extra = -MINT_PAIRS * (1.0 - sell_price)

                extra_pnls.append(extra)

            if sells < 10:
                continue

            total_extra = sum(extra_pnls)
            combined = base_total + total_extra
            sell_wr = sell_wins / sells if sells > 0 else 0

            # Sharpe of combined strategy
            combined_pnls = [t["base_pnl"] + e for t, e in zip(trades, extra_pnls)]
            sharpe = np.mean(combined_pnls) / np.std(combined_pnls) * np.sqrt(n) if np.std(combined_pnls) > 0 else 0

            print(f"{remaining:>4d}s  <{max_price:<5.2f} | {sells:>5d} {sell_wr:>5.1%} | "
                  f"{total_extra:>+8.0f} {total_extra/n:>+9.2f} | {combined:>+9.0f} {sharpe:>7.2f}")
        print()

    # ══════════════════════════════════════════════════════════════
    # ANALYSIS 2: Price-to-time ratio threshold
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*90}")
    print("PRICE/TIME RATIO — SELL WHEN loser_bid / remaining_seconds < threshold")
    print(f"{'='*90}")
    print(f"\nLow ratio = cheap loser with little time left = high confidence it's actually losing.\n")

    for ratio_threshold in [0.0005, 0.001, 0.002, 0.003, 0.005]:
        # For each trade, find the FIRST opportunity where ratio < threshold after signal
        extra_pnls = []
        sells = 0
        sell_wins = 0
        sell_times = []
        sell_prices = []

        for t in trades:
            sold = False
            for opp in t["sell_opps"]:
                if opp["remaining_s"] <= 0:
                    continue
                ratio = opp["bid"] / opp["remaining_s"]
                if ratio < ratio_threshold and opp["bid"] > 0.005:  # min bid to avoid dust
                    sells += 1
                    sell_price = opp["bid"]
                    sell_times.append(opp["elapsed"])
                    sell_prices.append(sell_price)

                    if t["is_winner"]:
                        extra = MINT_PAIRS * sell_price
                        sell_wins += 1
                    else:
                        extra = -MINT_PAIRS * (1.0 - sell_price)

                    extra_pnls.append(extra)
                    sold = True
                    break

            if not sold:
                extra_pnls.append(0.0)

        if sells < 10:
            continue

        total_extra = sum(extra_pnls)
        combined = base_total + total_extra
        sell_wr = sell_wins / sells if sells > 0 else 0

        combined_pnls = [t["base_pnl"] + e for t, e in zip(trades, extra_pnls)]
        sharpe = np.mean(combined_pnls) / np.std(combined_pnls) * np.sqrt(n) if np.std(combined_pnls) > 0 else 0

        print(f"  ratio < {ratio_threshold:.4f}: {sells:>5d} sells, WR={sell_wr:.1%}, "
              f"extra={total_extra:>+.0f}, combined={combined:>+.0f}, Sharpe={sharpe:.2f}")
        if sell_times:
            print(f"    avg sell time: {np.mean(sell_times):.0f}s (remaining: {300-np.mean(sell_times):.0f}s), "
                  f"avg sell price: {np.mean(sell_prices):.4f}")

    # ══════════════════════════════════════════════════════════════
    # ANALYSIS 3: Conditional win rate when loser is cheap near expiry
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*90}")
    print("CONDITIONAL WIN RATE — When loser bid is cheap near expiry, how often are we right?")
    print(f"{'='*90}")

    for sell_elapsed in [240, 270]:
        remaining = 300 - sell_elapsed
        print(f"\n  At {remaining}s remaining:")
        for max_price in [0.03, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]:
            right = 0
            wrong = 0
            for t in trades:
                opp = None
                for o in t["sell_opps"]:
                    if abs(o["elapsed"] - sell_elapsed) <= 5:
                        opp = o
                        break
                if opp is None or opp["bid"] <= 0 or opp["bid"] >= max_price:
                    continue
                if t["is_winner"]:
                    right += 1
                else:
                    wrong += 1

            total = right + wrong
            if total < 10:
                continue
            wr = right / total
            # Break-even: need WR > 1 - avg_sell_price
            # At low prices, break-even WR is very high
            print(f"    bid < {max_price:.2f}: {total:>5d} markets, WR={wr:.1%} "
                  f"(need >{1-max_price:.1%} to be +EV)")


if __name__ == "__main__":
    main()
