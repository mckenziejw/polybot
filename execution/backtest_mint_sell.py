#!/usr/bin/env python3
"""
Backtest: Compare direct buy vs mint+sell entry paths.

For each market where mid-drift signal fires (t=60s, d>0.10):
  A) DIRECT BUY: buy winner at ask_winner (current approach)
  B) MINT+SELL TAKER: mint pair for $1, sell loser at bid_loser
     effective cost = 1.00 - bid_loser
  C) MINT+SELL MAKER: mint pair for $1, post sell on loser at ask_loser
     effective cost = 1.00 - ask_loser (if filled)
     Fill check: did any buy order hit our ask level on the loser side post-signal?

Also checks: is there more buying pressure on the loser side (fills our maker sell)?
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

SIGNAL_TIME_S = 60
DRIFT_THRESHOLD = 0.10
BET_DOLLARS = 100


@dataclass
class ComparisonResult:
    slug: str
    market_ts: int
    is_winner: bool
    drift: float
    # Direct buy
    direct_ask: float           # ask_winner at signal
    direct_cost: float          # per-token cost
    direct_pnl: float
    # Mint+sell taker
    loser_bid: float            # bid_loser at signal
    mint_sell_taker_cost: float  # 1.00 - bid_loser
    mint_sell_taker_pnl: float
    # Mint+sell maker
    loser_ask: float            # ask_loser at signal (our posted sell price)
    mint_sell_maker_cost: float  # 1.00 - ask_loser
    mint_sell_maker_filled: bool
    mint_sell_maker_pnl: float
    # Spread info
    winner_spread: float        # ask_winner - bid_winner
    loser_spread: float         # ask_loser - bid_loser
    savings_taker: float        # direct_cost - mint_sell_taker_cost (positive = mint cheaper)
    savings_maker: float        # direct_cost - mint_sell_maker_cost


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
                    resolution: dict) -> ComparisonResult | None:
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

    # Identify leading (winner candidate) and trailing (loser candidate) tokens
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
            return None
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
            return None

    is_winner = (winner_id == winning_token)

    # Extract prices
    direct_ask = winner_snap.ask_price_1
    winner_bid = winner_snap.bid_price_1
    loser_bid = loser_snap.bid_price_1
    loser_ask = loser_snap.ask_price_1

    # Validate prices
    if direct_ask <= 0 or direct_ask >= 1:
        return None
    if loser_bid <= 0 or loser_ask <= 0:
        return None

    # ── Direct buy ──
    direct_tokens = int(BET_DOLLARS / direct_ask)
    if direct_tokens < 5:
        return None
    direct_pnl = direct_tokens * (1.0 - direct_ask) if is_winner else -(direct_tokens * direct_ask)

    # ── Mint+sell taker ──
    # Mint N pairs for $N, sell N loser tokens at bid_loser
    # Effective cost per winner token = 1.00 - bid_loser
    mint_sell_taker_cost = 1.00 - loser_bid
    if mint_sell_taker_cost <= 0 or mint_sell_taker_cost >= 1:
        return None
    taker_tokens = int(BET_DOLLARS / mint_sell_taker_cost)
    if taker_tokens < 5:
        return None
    mint_sell_taker_pnl = taker_tokens * (1.0 - mint_sell_taker_cost) if is_winner else -(taker_tokens * mint_sell_taker_cost)

    # ── Mint+sell maker ──
    # Mint N pairs, post sell on loser at ask_loser (maker)
    # Effective cost per winner token = 1.00 - ask_loser
    mint_sell_maker_cost = 1.00 - loser_ask
    if mint_sell_maker_cost <= 0 or mint_sell_maker_cost >= 1:
        mint_sell_maker_filled = False
        mint_sell_maker_pnl = 0.0
    else:
        maker_tokens = int(BET_DOLLARS / mint_sell_maker_cost)
        if maker_tokens < 5:
            mint_sell_maker_filled = False
            mint_sell_maker_pnl = 0.0
        else:
            # Check fill: did bid on loser side ever reach our ask level?
            post_signal_loser = books[
                (books.asset_id == loser_id) &
                (books.elapsed_s > SIGNAL_TIME_S)
            ]
            if len(post_signal_loser) > 0:
                # Our sell is at loser_ask. A fill happens if bid >= our price
                mint_sell_maker_filled = (post_signal_loser.bid_price_1 >= loser_ask).any()
                # Or more conservatively: did someone buy at our level?
                if not mint_sell_maker_filled:
                    mint_sell_maker_filled = (post_signal_loser.ask_price_1 >= loser_ask).any() and \
                                             (post_signal_loser.mid_price >= loser_ask).any()
            else:
                mint_sell_maker_filled = False

            if mint_sell_maker_filled:
                mint_sell_maker_pnl = maker_tokens * (1.0 - mint_sell_maker_cost) if is_winner else -(maker_tokens * mint_sell_maker_cost)
            else:
                mint_sell_maker_pnl = 0.0

    winner_spread = direct_ask - winner_bid if winner_bid > 0 else 0
    loser_spread = loser_ask - loser_bid

    slug = f"btc-updown-5m-{market_ts}"
    return ComparisonResult(
        slug=slug,
        market_ts=market_ts,
        is_winner=is_winner,
        drift=drift,
        direct_ask=direct_ask,
        direct_cost=direct_ask,
        direct_pnl=direct_pnl,
        loser_bid=loser_bid,
        mint_sell_taker_cost=mint_sell_taker_cost,
        mint_sell_taker_pnl=mint_sell_taker_pnl,
        loser_ask=loser_ask,
        mint_sell_maker_cost=mint_sell_maker_cost,
        mint_sell_maker_filled=mint_sell_maker_filled,
        mint_sell_maker_pnl=mint_sell_maker_pnl,
        winner_spread=winner_spread,
        loser_spread=loser_spread,
        savings_taker=direct_ask - mint_sell_taker_cost,
        savings_maker=direct_ask - mint_sell_maker_cost,
    )


def main():
    book_files = sorted(BOOK_DIR.glob("btc-updown-5m-*.parquet"))
    resolutions = load_resolutions()

    print(f"Running mint+sell comparison: signal={SIGNAL_TIME_S}s, drift>{DRIFT_THRESHOLD}")
    print(f"Markets: {len(book_files)}")

    results: list[ComparisonResult] = []
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

    results.sort(key=lambda r: r.market_ts)
    n = len(results)
    wins = sum(1 for r in results if r.is_winner)

    print(f"\nTrades: {n} | Win rate: {wins}/{n} ({wins/n:.1%})")

    # ── Entry cost comparison ──
    print(f"\n{'='*70}")
    print("ENTRY COST COMPARISON (per-token)")
    print(f"{'='*70}")

    direct_costs = [r.direct_cost for r in results]
    taker_costs = [r.mint_sell_taker_cost for r in results]
    maker_costs = [r.mint_sell_maker_cost for r in results if r.mint_sell_maker_cost > 0 and r.mint_sell_maker_cost < 1]

    print(f"\n  Direct buy (ask_winner):     mean={np.mean(direct_costs):.4f}  median={np.median(direct_costs):.4f}")
    print(f"  Mint+sell taker (1-bid_loser): mean={np.mean(taker_costs):.4f}  median={np.median(taker_costs):.4f}")
    if maker_costs:
        print(f"  Mint+sell maker (1-ask_loser): mean={np.mean(maker_costs):.4f}  median={np.median(maker_costs):.4f}")

    # ── Savings distribution ──
    savings_taker = [r.savings_taker for r in results]
    savings_maker = [r.savings_maker for r in results]

    print(f"\n--- Savings: direct_cost - mint_cost (positive = mint is cheaper) ---")
    print(f"  Taker savings:  mean={np.mean(savings_taker):+.4f}  median={np.median(savings_taker):+.4f}")
    print(f"    >0 (mint cheaper): {sum(1 for s in savings_taker if s > 0)}/{n} ({sum(1 for s in savings_taker if s > 0)/n:.1%})")
    print(f"    <0 (direct cheaper): {sum(1 for s in savings_taker if s < 0)}/{n} ({sum(1 for s in savings_taker if s < 0)/n:.1%})")
    print(f"    =0 (same): {sum(1 for s in savings_taker if s == 0)}/{n}")

    print(f"\n  Maker savings:  mean={np.mean(savings_maker):+.4f}  median={np.median(savings_maker):+.4f}")
    print(f"    >0 (mint cheaper): {sum(1 for s in savings_maker if s > 0)}/{n} ({sum(1 for s in savings_maker if s > 0)/n:.1%})")

    # ── PnL comparison ──
    print(f"\n{'='*70}")
    print("PNL COMPARISON ($100/trade)")
    print(f"{'='*70}")

    direct_pnl = sum(r.direct_pnl for r in results)
    taker_pnl = sum(r.mint_sell_taker_pnl for r in results)

    maker_filled = [r for r in results if r.mint_sell_maker_filled]
    maker_pnl = sum(r.mint_sell_maker_pnl for r in maker_filled)
    # For unfilled maker trades, fall back to direct buy
    maker_unfilled_pnl = sum(r.direct_pnl for r in results if not r.mint_sell_maker_filled)
    maker_hybrid_pnl = maker_pnl + maker_unfilled_pnl

    print(f"\n  A) Direct buy (taker):        PnL = {direct_pnl:+.0f}")
    print(f"  B) Mint+sell (taker):         PnL = {taker_pnl:+.0f}  (delta: {taker_pnl - direct_pnl:+.0f})")
    print(f"  C) Mint+sell (maker only):    PnL = {maker_pnl:+.0f}  ({len(maker_filled)}/{n} filled, {len(maker_filled)/n:.1%})")
    print(f"  D) Hybrid (maker if filled, else direct): PnL = {maker_hybrid_pnl:+.0f}  (delta: {maker_hybrid_pnl - direct_pnl:+.0f})")

    # ── Equity curves ──
    for label, pnl_fn in [
        ("Direct", lambda r: r.direct_pnl),
        ("Mint+sell taker", lambda r: r.mint_sell_taker_pnl),
        ("Hybrid maker", lambda r: r.mint_sell_maker_pnl if r.mint_sell_maker_filled else r.direct_pnl),
    ]:
        cum = peak = max_dd = 0.0
        for r in results:
            cum += pnl_fn(r)
            peak = max(peak, cum)
            max_dd = max(max_dd, peak - cum)
        pnls = [pnl_fn(r) for r in results]
        sharpe = np.mean(pnls) / np.std(pnls) * np.sqrt(n) if np.std(pnls) > 0 else 0
        print(f"\n  {label}: Final={cum:+.0f}  MaxDD={max_dd:.0f}  Sharpe={sharpe:.2f}")

    # ── Spread analysis ──
    print(f"\n{'='*70}")
    print("SPREAD ANALYSIS")
    print(f"{'='*70}")

    winner_spreads = [r.winner_spread for r in results]
    loser_spreads = [r.loser_spread for r in results]

    print(f"\n  Winner side spread: mean={np.mean(winner_spreads):.4f}  median={np.median(winner_spreads):.4f}")
    print(f"  Loser side spread:  mean={np.mean(loser_spreads):.4f}  median={np.median(loser_spreads):.4f}")

    # How often is loser spread tighter (better for our maker sell)?
    tighter_loser = sum(1 for r in results if r.loser_spread < r.winner_spread)
    print(f"  Loser spread tighter: {tighter_loser}/{n} ({tighter_loser/n:.1%})")

    # ── Savings by drift magnitude ──
    print(f"\n--- Savings by drift magnitude ---")
    for lo, hi, label in [(0.10, 0.15, "10-15%"), (0.15, 0.20, "15-20%"),
                          (0.20, 0.30, "20-30%"), (0.30, 0.50, "30-50%")]:
        bucket = [r for r in results if lo <= abs(r.drift) < hi]
        if len(bucket) >= 10:
            avg_save_t = np.mean([r.savings_taker for r in bucket])
            avg_save_m = np.mean([r.savings_maker for r in bucket])
            fill_rate = sum(1 for r in bucket if r.mint_sell_maker_filled) / len(bucket)
            print(f"  drift {label:>6s}: taker_save={avg_save_t:+.4f}  maker_save={avg_save_m:+.4f}  "
                  f"maker_fill={fill_rate:.1%}  n={len(bucket)}")


if __name__ == "__main__":
    main()
