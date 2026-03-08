"""
Momentum v5 — sequential bankroll simulation with two entry modes.

Two entry approaches:
  A) Market entry: buy at ask immediately at signal time
  B) Limit entry: post limit order at target_price, wait for fill in window

Both use the same signal (BTC momentum at delay) and hold to expiry.
Simulated sequentially through markets in chronological order with:
  - Running bankroll (starts at initial_bankroll)
  - Bet sizing as fraction of current bankroll (or fixed $)
  - Cumulative PnL / drawdown tracking
  - Optional session stop if drawdown exceeds threshold
"""

import csv
import glob
import math
import os
from collections import defaultdict
from dataclasses import dataclass

TICK_DIR = os.path.join(os.path.dirname(__file__), "data", "paper")


@dataclass
class MarketTick:
    timestamp: int
    elapsed_s: float
    remaining_s: float
    btc_price: float | None
    up_best_bid: float | None
    up_best_ask: float | None
    down_best_bid: float | None
    down_best_ask: float | None


@dataclass
class MarketData:
    slug: str
    ticks: list[MarketTick]


def parse_float(s: str) -> float | None:
    try:
        return float(s) if s else None
    except ValueError:
        return None


def load_all_markets() -> dict[str, MarketData]:
    markets: dict[str, list[MarketTick]] = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(TICK_DIR, "ticks_*.csv"))):
        with open(path, errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    slug = row.get("market_slug")
                    if not slug or "\x00" in slug or not slug.startswith("btc-"):
                        continue
                    tick = MarketTick(
                        timestamp=int(row["timestamp"]),
                        elapsed_s=float(row["time_elapsed_s"]),
                        remaining_s=float(row["time_remaining_s"]),
                        btc_price=parse_float(row["btc_price"]),
                        up_best_bid=parse_float(row["up_best_bid"]),
                        up_best_ask=parse_float(row["up_best_ask"]),
                        down_best_bid=parse_float(row["down_best_bid"]),
                        down_best_ask=parse_float(row["down_best_ask"]),
                    )
                    markets[slug].append(tick)
                except (ValueError, KeyError):
                    continue

    result = {}
    for slug, ticks in markets.items():
        ticks.sort(key=lambda t: t.timestamp)
        seen = set()
        deduped = []
        for t in ticks:
            if t.timestamp not in seen:
                seen.add(t.timestamp)
                deduped.append(t)
        result[slug] = MarketData(slug=slug, ticks=deduped)
    return result


def estimate_win_prob(btc_return_abs: float) -> float:
    anchors = [
        (0.000, 0.50), (0.005, 0.55), (0.010, 0.60),
        (0.015, 0.67), (0.020, 0.75), (0.030, 0.83), (0.050, 0.88),
    ]
    ret_pct = btc_return_abs * 100
    if ret_pct <= anchors[0][0]:
        return anchors[0][1]
    if ret_pct >= anchors[-1][0]:
        return min(anchors[-1][1], 0.90)
    for i in range(len(anchors) - 1):
        x0, y0 = anchors[i]
        x1, y1 = anchors[i + 1]
        if x0 <= ret_pct <= x1:
            t = (ret_pct - x0) / (x1 - x0)
            return y0 + t * (y1 - y0)
    return 0.50


def get_btc_at_time(ticks: list[MarketTick], target: float) -> float | None:
    for t in ticks:
        if t.elapsed_s >= target and t.btc_price is not None:
            return t.btc_price
    return None


def get_ask_at_time(ticks: list[MarketTick], target: float, direction: str) -> float | None:
    for t in ticks:
        if t.elapsed_s >= target:
            a = t.up_best_ask if direction == "Up" else t.down_best_ask
            if a is not None:
                return a
    return None


def find_limit_fill(ticks: list[MarketTick], start_s: float, end_s: float,
                    direction: str, limit_price: float) -> tuple[float, float] | None:
    """Find first tick in [start_s, end_s] where ask <= limit_price.
    Returns (elapsed_s, fill_price) or None."""
    for t in ticks:
        if t.elapsed_s < start_s:
            continue
        if t.elapsed_s > end_s:
            break
        ask = t.up_best_ask if direction == "Up" else t.down_best_ask
        if ask is not None and ask <= limit_price:
            return (t.elapsed_s, ask)
    return None


def determine_outcome(ticks: list[MarketTick]) -> str | None:
    for t in reversed(ticks):
        if t.up_best_bid is not None and t.up_best_bid >= 0.9:
            return "Up"
        if t.up_best_ask is not None and t.up_best_ask <= 0.1:
            return "Down"
        if t.down_best_bid is not None and t.down_best_bid >= 0.9:
            return "Down"
        if t.down_best_ask is not None and t.down_best_ask <= 0.1:
            return "Up"
    btc_prices = [t.btc_price for t in ticks if t.btc_price is not None]
    if len(btc_prices) >= 2:
        if btc_prices[-1] > btc_prices[0]:
            return "Up"
        elif btc_prices[-1] < btc_prices[0]:
            return "Down"
    return None


@dataclass
class TradeRecord:
    market_num: int
    slug: str
    btc_return_pct: float
    direction: str
    p_win: float
    target_price: float
    entry_price: float | None  # None if no fill
    entry_time_s: float | None
    tokens: float
    cost: float
    outcome: str
    won: bool | None  # None if skipped
    pnl: float
    bankroll_before: float
    bankroll_after: float
    cumulative_pnl: float
    peak_bankroll: float
    drawdown: float
    drawdown_pct: float
    action: str  # "BUY", "SKIP_NO_FILL", "SKIP_NO_SIGNAL", "SKIP_DRAWDOWN"


def run_session(
    markets: dict[str, MarketData],
    mode: str,  # "market" or "limit"
    signal_delay: float = 3.0,
    min_signal: float = 0.0001,
    desired_edge: float = 0.10,
    bet_dollars: float = 100.0,  # fixed $ per trade (or "bankroll_frac" mode)
    bet_mode: str = "fixed",  # "fixed" or "fraction"
    bet_fraction: float = 0.10,  # fraction of bankroll per trade
    initial_bankroll: float = 1000.0,
    limit_window: float = 60.0,  # seconds to wait for limit fill
    max_drawdown_pct: float = 1.0,  # stop session if drawdown > this (1.0 = no stop)
) -> list[TradeRecord]:

    records = []
    bankroll = initial_bankroll
    peak = initial_bankroll
    cumulative_pnl = 0.0
    market_num = 0
    session_stopped = False

    for slug in sorted(markets.keys()):
        m = markets[slug]
        ticks = m.ticks
        if len(ticks) < 10 or max(t.elapsed_s for t in ticks) < 200:
            continue

        market_num += 1

        # Check session drawdown
        dd = peak - bankroll
        dd_pct = dd / peak if peak > 0 else 0
        if dd_pct >= max_drawdown_pct:
            if not session_stopped:
                session_stopped = True
            records.append(TradeRecord(
                market_num=market_num, slug=slug,
                btc_return_pct=0, direction="", p_win=0, target_price=0,
                entry_price=None, entry_time_s=None, tokens=0, cost=0,
                outcome="", won=None, pnl=0,
                bankroll_before=bankroll, bankroll_after=bankroll,
                cumulative_pnl=cumulative_pnl, peak_bankroll=peak,
                drawdown=dd, drawdown_pct=dd_pct,
                action="SKIP_DRAWDOWN",
            ))
            continue

        # Get signal
        btc_ref = next((t.btc_price for t in ticks if t.btc_price), None)
        if not btc_ref:
            continue
        btc_sig = get_btc_at_time(ticks, signal_delay)
        if not btc_sig:
            continue

        btc_ret = (btc_sig - btc_ref) / btc_ref
        if abs(btc_ret) < min_signal:
            records.append(TradeRecord(
                market_num=market_num, slug=slug,
                btc_return_pct=btc_ret * 100, direction="", p_win=0, target_price=0,
                entry_price=None, entry_time_s=None, tokens=0, cost=0,
                outcome="", won=None, pnl=0,
                bankroll_before=bankroll, bankroll_after=bankroll,
                cumulative_pnl=cumulative_pnl, peak_bankroll=peak,
                drawdown=peak - bankroll, drawdown_pct=(peak - bankroll) / peak if peak > 0 else 0,
                action="SKIP_NO_SIGNAL",
            ))
            continue

        direction = "Up" if btc_ret > 0 else "Down"
        p_win = estimate_win_prob(abs(btc_ret))
        target = p_win - desired_edge

        outcome = determine_outcome(ticks)
        if not outcome:
            continue

        # Determine bet size
        if bet_mode == "fraction":
            dollars = bankroll * bet_fraction
        else:
            dollars = min(bet_dollars, bankroll)

        if dollars < 5:  # can't even buy minimum lot
            continue

        # Try to get fill
        entry_price = None
        entry_time = None
        sig_idx = next(i for i, t in enumerate(ticks) if t.elapsed_s >= signal_delay)

        if mode == "market":
            # Buy at ask immediately
            ask = get_ask_at_time(ticks, signal_delay, direction)
            if ask is not None and ask <= target:
                entry_price = ask
                entry_time = ticks[sig_idx].elapsed_s
        elif mode == "limit":
            # Wait for ask <= target within window
            fill = find_limit_fill(ticks, signal_delay, signal_delay + limit_window,
                                   direction, target)
            if fill:
                entry_time, entry_price = fill

        if entry_price is None:
            records.append(TradeRecord(
                market_num=market_num, slug=slug,
                btc_return_pct=btc_ret * 100, direction=direction,
                p_win=p_win, target_price=target,
                entry_price=None, entry_time_s=None, tokens=0, cost=0,
                outcome=outcome, won=None, pnl=0,
                bankroll_before=bankroll, bankroll_after=bankroll,
                cumulative_pnl=cumulative_pnl, peak_bankroll=peak,
                drawdown=peak - bankroll, drawdown_pct=(peak - bankroll) / peak if peak > 0 else 0,
                action="SKIP_NO_FILL",
            ))
            continue

        tokens = dollars / entry_price
        cost = tokens * entry_price  # = dollars

        won = direction == outcome
        pnl = tokens * (1.0 if won else 0.0) - cost

        bankroll_before = bankroll
        bankroll += pnl
        cumulative_pnl += pnl
        if bankroll > peak:
            peak = bankroll
        dd = peak - bankroll
        dd_pct = dd / peak if peak > 0 else 0

        records.append(TradeRecord(
            market_num=market_num, slug=slug,
            btc_return_pct=btc_ret * 100, direction=direction,
            p_win=p_win, target_price=target,
            entry_price=entry_price, entry_time_s=entry_time,
            tokens=tokens, cost=cost, outcome=outcome, won=won, pnl=pnl,
            bankroll_before=bankroll_before, bankroll_after=bankroll,
            cumulative_pnl=cumulative_pnl, peak_bankroll=peak,
            drawdown=dd, drawdown_pct=dd_pct,
            action="BUY",
        ))

    return records


def summarize(records: list[TradeRecord], label: str):
    trades = [r for r in records if r.action == "BUY"]
    skips_no_signal = sum(1 for r in records if r.action == "SKIP_NO_SIGNAL")
    skips_no_fill = sum(1 for r in records if r.action == "SKIP_NO_FILL")
    skips_dd = sum(1 for r in records if r.action == "SKIP_DRAWDOWN")

    if not trades:
        print(f"\n{label}: No trades (skipped: {skips_no_signal} no signal, {skips_no_fill} no fill)")
        return

    wins = sum(1 for t in trades if t.won)
    losses = len(trades) - wins
    total_pnl = sum(t.pnl for t in trades)
    total_cost = sum(t.cost for t in trades)

    pnls = [t.pnl for t in trades]
    ev = total_pnl / len(trades)
    variance = sum((p - ev) ** 2 for p in pnls) / len(pnls)
    stdev = math.sqrt(variance) if variance > 0 else 0
    sharpe = ev / stdev if stdev > 0 else float("inf")

    max_dd = max(r.drawdown for r in records)
    max_dd_pct = max(r.drawdown_pct for r in records)
    final_bankroll = records[-1].bankroll_after

    # Consecutive losses
    max_consec_loss = 0
    current_streak = 0
    for t in trades:
        if not t.won:
            current_streak += 1
            max_consec_loss = max(max_consec_loss, current_streak)
        else:
            current_streak = 0

    avg_entry = sum(t.entry_price for t in trades) / len(trades)
    avg_fill_time = sum(t.entry_time_s for t in trades) / len(trades)

    print(f"\n{label}")
    print(f"  Markets seen: {len(records)} | Traded: {len(trades)} | "
          f"Skip(no signal): {skips_no_signal} | Skip(no fill): {skips_no_fill} | Skip(DD): {skips_dd}")
    print(f"  Record: {wins}W / {losses}L ({wins/len(trades)*100:.0f}% WR) | "
          f"Max consec losses: {max_consec_loss}")
    print(f"  Total PnL: {total_pnl:+.2f} | Capital deployed: {total_cost:.2f} | "
          f"ROI: {total_pnl/total_cost*100:+.1f}%")
    print(f"  Final bankroll: {final_bankroll:.2f} | "
          f"Max drawdown: {max_dd:.2f} ({max_dd_pct*100:.1f}%)")
    print(f"  EV/trade: {ev:+.2f} | Stdev: {stdev:.2f} | Sharpe: {sharpe:.2f}")
    print(f"  Avg entry: {avg_entry:.3f} | Avg fill time: {avg_fill_time:.1f}s")


def print_equity_curve(records: list[TradeRecord], label: str):
    trades = [r for r in records if r.action == "BUY"]
    if not trades:
        return
    print(f"\n  Equity curve — {label}:")
    print(f"  {'#':>3} {'slug':<35} {'sig%':>7} {'p':>4} {'entry':>5} {'t_fill':>6} "
          f"{'dir':>4} {'W/L':>3} {'pnl':>8} {'bankroll':>9} {'DD':>6} {'DD%':>5}")
    for r in records:
        if r.action == "SKIP_NO_SIGNAL":
            continue
        if r.action == "SKIP_DRAWDOWN":
            print(f"  {r.market_num:3d} {r.slug:<35} {'':>7} {'':>4} {'':>5} {'':>6} "
                  f"{'':>4} {'':>3} {'':>8} {r.bankroll_after:9.2f} {r.drawdown:6.1f} "
                  f"{r.drawdown_pct*100:4.1f}%  ** SESSION STOPPED **")
            continue
        if r.action == "SKIP_NO_FILL":
            print(f"  {r.market_num:3d} {r.slug:<35} {r.btc_return_pct:+6.4f}% {r.p_win:.2f} "
                  f"{'—':>5} {'—':>6} {r.direction:>4} {'—':>3} {'—':>8} "
                  f"{r.bankroll_after:9.2f} {r.drawdown:6.1f} {r.drawdown_pct*100:4.1f}%  (no fill)")
            continue

        icon = "W" if r.won else "L"
        print(f"  {r.market_num:3d} {r.slug:<35} {r.btc_return_pct:+6.4f}% {r.p_win:.2f} "
              f"{r.entry_price:5.3f} {r.entry_time_s:5.1f}s {r.direction:>4} {icon:>3} "
              f"{r.pnl:+8.2f} {r.bankroll_after:9.2f} {r.drawdown:6.1f} {r.drawdown_pct*100:4.1f}%")


def main():
    print("Loading tick data...")
    markets = load_all_markets()
    full = {s: m for s, m in markets.items()
            if len(m.ticks) >= 10 and max(t.elapsed_s for t in m.ticks) >= 200}
    print(f"Loaded {len(markets)} markets ({len(full)} with full coverage)\n")

    BANKROLL = 1000.0

    # ================================================================
    print("=" * 110)
    print("MARKET vs LIMIT ENTRY — fixed $100/trade, $1000 bankroll")
    print("=" * 110)

    for sd in [3, 5]:
        for ms in [0.0001, 0.0002]:
            for edge in [0.05, 0.10, 0.15]:
                r_mkt = run_session(markets, "market", signal_delay=sd, min_signal=ms,
                                     desired_edge=edge, bet_dollars=100, initial_bankroll=BANKROLL)
                r_lim = run_session(markets, "limit", signal_delay=sd, min_signal=ms,
                                     desired_edge=edge, bet_dollars=100, initial_bankroll=BANKROLL,
                                     limit_window=60)

                t_mkt = [r for r in r_mkt if r.action == "BUY"]
                t_lim = [r for r in r_lim if r.action == "BUY"]
                if not t_mkt and not t_lim:
                    continue

                def quick(recs):
                    trades = [r for r in recs if r.action == "BUY"]
                    if not trades:
                        return "no trades"
                    w = sum(1 for t in trades if t.won)
                    pnl = sum(t.pnl for t in trades)
                    avg_e = sum(t.entry_price for t in trades) / len(trades)
                    max_dd = max(r.drawdown for r in recs)
                    return (f"{len(trades):2d}t {w/len(trades)*100:.0f}%WR "
                            f"PnL={pnl:+7.1f} entry={avg_e:.3f} maxDD={max_dd:.0f}")

                print(f"  {sd}s sig>={ms*100:.2f}% edge={edge:.2f}: "
                      f"market=[{quick(r_mkt)}]  limit=[{quick(r_lim)}]")

    # ================================================================
    print("\n" + "=" * 110)
    print("LIMIT WINDOW SWEEP — 3s delay, sig>=0.01%, edge=0.10")
    print("=" * 110)

    for window in [10, 20, 30, 45, 60, 90, 120, 180]:
        r = run_session(markets, "limit", signal_delay=3, min_signal=0.0001,
                        desired_edge=0.10, bet_dollars=100, initial_bankroll=BANKROLL,
                        limit_window=window)
        trades = [x for x in r if x.action == "BUY"]
        fills = sum(1 for x in r if x.action == "SKIP_NO_FILL")
        if trades:
            w = sum(1 for t in trades if t.won)
            pnl = sum(t.pnl for t in trades)
            avg_e = sum(t.entry_price for t in trades) / len(trades)
            max_dd = max(x.drawdown for x in r)
            print(f"  window={window:3d}s: {len(trades):2d} trades ({fills} missed), "
                  f"{w/len(trades)*100:.0f}%WR, PnL={pnl:+7.1f}, "
                  f"avg_entry={avg_e:.3f}, maxDD={max_dd:.0f}")

    # ================================================================
    print("\n" + "=" * 110)
    print("BET SIZING — fraction of bankroll vs fixed")
    print("=" * 110)

    for ms in [0.0001, 0.0002]:
        print(f"\n  --- sig >= {ms*100:.2f}% ---")
        # Fixed
        for bet in [50, 100, 150, 200]:
            r = run_session(markets, "market", signal_delay=3, min_signal=ms,
                            desired_edge=0.10, bet_dollars=bet, bet_mode="fixed",
                            initial_bankroll=BANKROLL)
            trades = [x for x in r if x.action == "BUY"]
            if trades:
                pnl = sum(t.pnl for t in trades)
                final = r[-1].bankroll_after
                max_dd = max(x.drawdown for x in r)
                max_dd_pct = max(x.drawdown_pct for x in r)
                print(f"    fixed ${bet:3d}: {len(trades):2d}t, PnL={pnl:+7.1f}, "
                      f"final={final:.0f}, maxDD={max_dd:.0f} ({max_dd_pct*100:.0f}%)")

        # Fraction
        for frac in [0.05, 0.10, 0.15, 0.20, 0.25]:
            r = run_session(markets, "market", signal_delay=3, min_signal=ms,
                            desired_edge=0.10, bet_mode="fraction", bet_fraction=frac,
                            initial_bankroll=BANKROLL)
            trades = [x for x in r if x.action == "BUY"]
            if trades:
                pnl = sum(t.pnl for t in trades)
                final = r[-1].bankroll_after
                max_dd = max(x.drawdown for x in r)
                max_dd_pct = max(x.drawdown_pct for x in r)
                print(f"    frac {frac:.0%}: {len(trades):2d}t, PnL={pnl:+7.1f}, "
                      f"final={final:.0f}, maxDD={max_dd:.0f} ({max_dd_pct*100:.0f}%)")

    # ================================================================
    print("\n" + "=" * 110)
    print("SESSION DRAWDOWN STOP — market entry, $100/trade, sig>=0.01%")
    print("=" * 110)

    for max_dd in [0.10, 0.15, 0.20, 0.25, 0.30, 0.50, 1.0]:
        r = run_session(markets, "market", signal_delay=3, min_signal=0.0001,
                        desired_edge=0.10, bet_dollars=100, initial_bankroll=BANKROLL,
                        max_drawdown_pct=max_dd)
        trades = [x for x in r if x.action == "BUY"]
        stopped = sum(1 for x in r if x.action == "SKIP_DRAWDOWN")
        if trades:
            w = sum(1 for t in trades if t.won)
            pnl = sum(t.pnl for t in trades)
            final = r[-1].bankroll_after
            actual_dd = max(x.drawdown for x in r)
            print(f"  stop@{max_dd*100:.0f}%DD: {len(trades):2d} trades, "
                  f"{w/len(trades)*100:.0f}%WR, PnL={pnl:+7.1f}, "
                  f"final={final:.0f}, peak_DD={actual_dd:.0f}, "
                  f"markets_skipped={stopped}")

    # ================================================================
    # Best configs — full equity curves
    # ================================================================

    configs = [
        ("A: Market, 3s, sig>=0.01%, edge=0.10, $100",
         dict(mode="market", signal_delay=3, min_signal=0.0001,
              desired_edge=0.10, bet_dollars=100, initial_bankroll=BANKROLL)),
        ("B: Limit 60s, 3s, sig>=0.01%, edge=0.10, $100",
         dict(mode="limit", signal_delay=3, min_signal=0.0001,
              desired_edge=0.10, bet_dollars=100, initial_bankroll=BANKROLL,
              limit_window=60)),
        ("C: Market, 3s, sig>=0.02%, edge=0.10, $100",
         dict(mode="market", signal_delay=3, min_signal=0.0002,
              desired_edge=0.10, bet_dollars=100, initial_bankroll=BANKROLL)),
        ("D: Market, 3s, sig>=0.01%, edge=0.10, 10% bankroll",
         dict(mode="market", signal_delay=3, min_signal=0.0001,
              desired_edge=0.10, bet_mode="fraction", bet_fraction=0.10,
              initial_bankroll=BANKROLL)),
    ]

    for label, kwargs in configs:
        print("\n" + "=" * 110)
        print(f"EQUITY CURVE — {label}")
        print("=" * 110)
        r = run_session(markets, **kwargs)
        summarize(r, label)
        print_equity_curve(r, label)


if __name__ == "__main__":
    main()
