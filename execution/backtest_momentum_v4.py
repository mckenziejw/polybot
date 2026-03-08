"""
Momentum v4 — head-to-head comparison of entry strategies.

Three strategies, all with $100 max capital per market:
  A) Single entry: buy once at signal time, spend up to $100
  B) Accumulate: buy throughout market when ask <= target, no stop
  C) Accumulate + drawdown stop: same as B, but stop buying if
     unrealized loss exceeds a threshold (and optionally dump position)

All hold to expiry. Compare on same set of markets.
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
class Fill:
    elapsed_s: float
    price: float
    size: float
    cost: float


@dataclass
class Result:
    slug: str
    strategy: str
    btc_return_pct: float
    direction: str
    p_win: float
    target_price: float
    fills: list[Fill]
    total_tokens: float
    total_cost: float
    vwap: float
    outcome: str
    won: bool
    pnl: float
    roi_pct: float
    stopped: bool = False
    stop_reason: str = ""
    exit_price: float | None = None  # if we dumped on stop


def get_signal(market: MarketData, signal_delay: float, min_signal: float):
    """Returns (btc_ret, direction, p_win, target_tick_idx) or None."""
    ticks = market.ticks
    if len(ticks) < 10 or max(t.elapsed_s for t in ticks) < 200:
        return None

    btc_ref = next((t.btc_price for t in ticks if t.btc_price), None)
    if not btc_ref:
        return None

    btc_sig = get_btc_at_time(ticks, signal_delay)
    if not btc_sig:
        return None

    btc_ret = (btc_sig - btc_ref) / btc_ref
    if abs(btc_ret) < min_signal:
        return None

    direction = "Up" if btc_ret > 0 else "Down"
    p_win = estimate_win_prob(abs(btc_ret))

    # Find the tick index at signal time
    sig_idx = 0
    for i, t in enumerate(ticks):
        if t.elapsed_s >= signal_delay:
            sig_idx = i
            break

    return btc_ret, direction, p_win, sig_idx


def get_ask(tick: MarketTick, direction: str) -> float | None:
    return tick.up_best_ask if direction == "Up" else tick.down_best_ask


def get_bid(tick: MarketTick, direction: str) -> float | None:
    return tick.up_best_bid if direction == "Up" else tick.down_best_bid


# ── Strategy A: Single entry ──

def run_single_entry(
    markets: dict[str, MarketData],
    signal_delay: float, min_signal: float, desired_edge: float,
    max_capital: float,
) -> list[Result]:
    results = []
    for slug in sorted(markets):
        m = markets[slug]
        sig = get_signal(m, signal_delay, min_signal)
        if not sig:
            continue
        btc_ret, direction, p_win, sig_idx = sig
        target = p_win - desired_edge

        outcome = determine_outcome(m.ticks)
        if not outcome:
            continue

        # Buy at signal time ask, up to max_capital
        ask = get_ask(m.ticks[sig_idx], direction)
        if ask is None or ask > target:
            # Try a few ticks after signal in case ask dips
            bought = False
            for t in m.ticks[sig_idx:sig_idx + 10]:
                a = get_ask(t, direction)
                if a is not None and a <= target:
                    ask = a
                    bought = True
                    break
            if not bought:
                continue

        tokens = min(max_capital / ask, max_capital / ask)  # spend up to max_capital
        cost = tokens * ask
        if cost > max_capital:
            tokens = max_capital / ask
            cost = max_capital

        won = direction == outcome
        pnl = tokens * (1.0 if won else 0.0) - cost

        results.append(Result(
            slug=slug, strategy="single_entry",
            btc_return_pct=btc_ret * 100, direction=direction,
            p_win=p_win, target_price=target,
            fills=[Fill(m.ticks[sig_idx].elapsed_s, ask, tokens, cost)],
            total_tokens=tokens, total_cost=cost, vwap=ask,
            outcome=outcome, won=won, pnl=pnl,
            roi_pct=(pnl / cost * 100) if cost > 0 else 0,
        ))
    return results


# ── Strategy B: Accumulate, no stop ──

def run_accumulate(
    markets: dict[str, MarketData],
    signal_delay: float, min_signal: float, desired_edge: float,
    order_lot: float, max_capital: float, stop_at_s: float = 30.0,
) -> list[Result]:
    results = []
    for slug in sorted(markets):
        m = markets[slug]
        sig = get_signal(m, signal_delay, min_signal)
        if not sig:
            continue
        btc_ret, direction, p_win, sig_idx = sig
        target = p_win - desired_edge
        if target <= 0.01:
            continue

        outcome = determine_outcome(m.ticks)
        if not outcome:
            continue

        fills = []
        total_cost = 0.0
        for t in m.ticks[sig_idx:]:
            if t.remaining_s < stop_at_s:
                break
            if total_cost >= max_capital:
                break

            ask = get_ask(t, direction)
            if ask is None or ask > target:
                continue

            remaining = max_capital - total_cost
            size = min(order_lot, remaining / ask)
            if size < 1:
                break
            cost = size * ask
            fills.append(Fill(t.elapsed_s, ask, size, cost))
            total_cost += cost

        if not fills:
            continue

        total_tokens = sum(f.size for f in fills)
        vwap = total_cost / total_tokens
        won = direction == outcome
        pnl = total_tokens * (1.0 if won else 0.0) - total_cost

        results.append(Result(
            slug=slug, strategy="accumulate",
            btc_return_pct=btc_ret * 100, direction=direction,
            p_win=p_win, target_price=target,
            fills=fills, total_tokens=total_tokens,
            total_cost=total_cost, vwap=vwap,
            outcome=outcome, won=won, pnl=pnl,
            roi_pct=(pnl / total_cost * 100) if total_cost > 0 else 0,
        ))
    return results


# ── Strategy C: Accumulate + drawdown stop ──

def run_accumulate_with_stop(
    markets: dict[str, MarketData],
    signal_delay: float, min_signal: float, desired_edge: float,
    order_lot: float, max_capital: float, stop_at_s: float = 30.0,
    drawdown_pct: float = 0.30,  # stop if unrealized loss > 30% of cost
    dump_on_stop: bool = False,  # sell position at market when stopped
) -> list[Result]:
    results = []
    for slug in sorted(markets):
        m = markets[slug]
        sig = get_signal(m, signal_delay, min_signal)
        if not sig:
            continue
        btc_ret, direction, p_win, sig_idx = sig
        target = p_win - desired_edge
        if target <= 0.01:
            continue

        outcome = determine_outcome(m.ticks)
        if not outcome:
            continue

        fills = []
        total_cost = 0.0
        total_tokens = 0.0
        stopped = False
        stop_reason = ""
        exit_price = None

        for t in m.ticks[sig_idx:]:
            if t.remaining_s < stop_at_s:
                break
            if total_cost >= max_capital:
                break
            if stopped:
                break

            # Check drawdown on existing position
            if total_tokens > 0:
                bid = get_bid(t, direction)
                if bid is not None:
                    mark_value = total_tokens * bid
                    unrealized_loss = total_cost - mark_value
                    loss_pct = unrealized_loss / total_cost
                    if loss_pct >= drawdown_pct:
                        stopped = True
                        stop_reason = f"drawdown {loss_pct:.0%} (bid={bid:.3f})"
                        if dump_on_stop:
                            exit_price = bid
                        break

            ask = get_ask(t, direction)
            if ask is None or ask > target:
                continue

            remaining = max_capital - total_cost
            size = min(order_lot, remaining / ask)
            if size < 1:
                break
            cost = size * ask
            fills.append(Fill(t.elapsed_s, ask, size, cost))
            total_cost += cost
            total_tokens += size

        if not fills:
            continue

        total_tokens = sum(f.size for f in fills)
        vwap = total_cost / total_tokens
        won = direction == outcome

        if dump_on_stop and stopped and exit_price is not None:
            # Sold at exit_price
            pnl = total_tokens * exit_price - total_cost
        else:
            # Held to expiry
            pnl = total_tokens * (1.0 if won else 0.0) - total_cost

        results.append(Result(
            slug=slug, strategy="accum+stop",
            btc_return_pct=btc_ret * 100, direction=direction,
            p_win=p_win, target_price=target,
            fills=fills, total_tokens=total_tokens,
            total_cost=total_cost, vwap=vwap,
            outcome=outcome, won=won, pnl=pnl,
            roi_pct=(pnl / total_cost * 100) if total_cost > 0 else 0,
            stopped=stopped, stop_reason=stop_reason,
            exit_price=exit_price,
        ))
    return results


# ── Metrics ──

def compute_metrics(results: list[Result]) -> dict:
    if not results:
        return {}
    wins = sum(1 for r in results if r.won)
    losses = len(results) - wins
    total_pnl = sum(r.pnl for r in results)
    total_cost = sum(r.total_cost for r in results)
    total_fills = sum(len(r.fills) for r in results)
    ev = total_pnl / len(results)
    pnls = [r.pnl for r in results]
    variance = sum((p - ev) ** 2 for p in pnls) / len(pnls)
    stdev = math.sqrt(variance) if variance > 0 else 0
    sharpe = ev / stdev if stdev > 0 else float("inf")
    stopped = sum(1 for r in results if r.stopped)

    # Max drawdown across markets (cumulative PnL)
    running = 0.0
    peak = 0.0
    max_dd = 0.0
    for r in results:
        running += r.pnl
        if running > peak:
            peak = running
        dd = peak - running
        if dd > max_dd:
            max_dd = dd

    return {
        "n": len(results), "wins": wins, "losses": losses,
        "wr": wins / len(results),
        "pnl": total_pnl, "cost": total_cost,
        "roi": total_pnl / total_cost * 100 if total_cost > 0 else 0,
        "fills": total_fills,
        "fills_per": total_fills / len(results),
        "ev": ev, "stdev": stdev, "sharpe": sharpe,
        "stopped": stopped,
        "max_dd": max_dd,
        "avg_vwap": sum(r.vwap for r in results) / len(results),
    }


def fmt(m: dict, label: str) -> str:
    if not m:
        return f"  {label}: no trades"
    stop_str = f" stopped={m['stopped']}" if m['stopped'] > 0 else ""
    return (
        f"  {label:<32s} {m['n']:3d} mkts  {m['wr']*100:4.0f}% WR  "
        f"PnL={m['pnl']:+8.2f}  ROI={m['roi']:+5.1f}%  "
        f"Sharpe={m['sharpe']:.2f}  maxDD={m['max_dd']:.0f}  "
        f"{m['fills_per']:.1f} fills/mkt  vwap={m['avg_vwap']:.3f}"
        f"{stop_str}"
    )


def print_detailed(results: list[Result]):
    print(f"\n  {'slug':<35} {'sig%':>7} {'p':>4} {'tgt':>5} {'fills':>5} "
          f"{'tokens':>6} {'cost':>7} {'vwap':>5} {'dir':>4} {'W/L':>3} "
          f"{'pnl':>8} {'roi':>6} {'stop':>6}")
    for r in results:
        icon = "W" if r.won else "L"
        stop = "STOP" if r.stopped else ""
        print(
            f"  {r.slug:<35} {r.btc_return_pct:+6.4f}% {r.p_win:.2f} "
            f"{r.target_price:.3f} {len(r.fills):5d} {r.total_tokens:6.1f} "
            f"{r.total_cost:7.2f} {r.vwap:.3f} {r.direction:>4} {icon:>3} "
            f"{r.pnl:+8.2f} {r.roi_pct:+5.1f}% {stop:>6}"
        )


def main():
    print("Loading tick data...")
    markets = load_all_markets()
    full = {s: m for s, m in markets.items()
            if len(m.ticks) >= 10 and max(t.elapsed_s for t in m.ticks) >= 200}
    print(f"Loaded {len(markets)} markets ({len(full)} with full coverage)\n")

    # Best config from v3: 3s delay, 0.01% min signal
    SD = 3.0
    MS = 0.0001

    # ================================================================
    print("=" * 100)
    print(f"HEAD-TO-HEAD COMPARISON — signal_delay={SD:.0f}s, min_signal={MS*100:.2f}%, max_capital=$100")
    print("=" * 100)

    # Sweep edge for each strategy
    print(f"\n--- Edge sweep ---")
    for edge in [0.05, 0.08, 0.10, 0.12, 0.15]:
        r_single = run_single_entry(markets, SD, MS, edge, 100)
        r_accum = run_accumulate(markets, SD, MS, edge, 5, 100)
        r_stop30 = run_accumulate_with_stop(markets, SD, MS, edge, 5, 100, drawdown_pct=0.30)
        r_stop50 = run_accumulate_with_stop(markets, SD, MS, edge, 5, 100, drawdown_pct=0.50)
        r_dump30 = run_accumulate_with_stop(markets, SD, MS, edge, 5, 100, drawdown_pct=0.30, dump_on_stop=True)

        print(f"\n  edge={edge:.2f}:")
        print(fmt(compute_metrics(r_single), "A: Single entry"))
        print(fmt(compute_metrics(r_accum), "B: Accumulate (no stop)"))
        print(fmt(compute_metrics(r_stop30), "C1: Accum + 30% DD stop (hold)"))
        print(fmt(compute_metrics(r_stop50), "C2: Accum + 50% DD stop (hold)"))
        print(fmt(compute_metrics(r_dump30), "C3: Accum + 30% DD stop (dump)"))

    # ================================================================
    print("\n" + "=" * 100)
    print("DRAWDOWN STOP SWEEP — edge=0.10, accumulate")
    print("=" * 100)

    print(f"\n  {'strategy':<40} {'mkts':>4} {'WR':>5} {'PnL':>9} {'ROI':>6} {'Sharpe':>7} {'maxDD':>6} {'stops':>5}")
    for dd_pct in [0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75, 1.0]:
        for dump in [False, True]:
            r = run_accumulate_with_stop(markets, SD, MS, 0.10, 5, 100,
                                         drawdown_pct=dd_pct, dump_on_stop=dump)
            m = compute_metrics(r)
            if not m:
                continue
            label = f"DD={dd_pct:.0%} {'dump' if dump else 'hold'}"
            print(
                f"  {label:<40} {m['n']:4d} {m['wr']*100:4.0f}% {m['pnl']:+8.2f} "
                f"{m['roi']:+5.1f}% {m['sharpe']:6.2f} {m['max_dd']:6.0f} {m['stopped']:5d}"
            )

    r_no_stop = run_accumulate(markets, SD, MS, 0.10, 5, 100)
    m_ns = compute_metrics(r_no_stop)
    print(
        f"  {'No stop':<40} {m_ns['n']:4d} {m_ns['wr']*100:4.0f}% {m_ns['pnl']:+8.2f} "
        f"{m_ns['roi']:+5.1f}% {m_ns['sharpe']:6.2f} {m_ns['max_dd']:6.0f}     0"
    )

    # ================================================================
    print("\n" + "=" * 100)
    print("DETAILED — edge=0.10, all three strategies side by side")
    print("=" * 100)

    r_a = run_single_entry(markets, SD, MS, 0.10, 100)
    r_b = run_accumulate(markets, SD, MS, 0.10, 5, 100)
    r_c = run_accumulate_with_stop(markets, SD, MS, 0.10, 5, 100, drawdown_pct=0.30)

    # Build lookup by slug
    a_by_slug = {r.slug: r for r in r_a}
    b_by_slug = {r.slug: r for r in r_b}
    c_by_slug = {r.slug: r for r in r_c}

    all_slugs = sorted(set(list(a_by_slug) + list(b_by_slug) + list(c_by_slug)))

    print(f"\n  {'slug':<35} {'sig%':>7} {'p':>4} {'out':>4}  "
          f"{'A_pnl':>7} {'A_roi':>6}  "
          f"{'B_pnl':>7} {'B_roi':>6} {'B_fills':>7}  "
          f"{'C_pnl':>7} {'C_roi':>6} {'C_stop':>6}")

    total_a = total_b = total_c = 0.0
    for slug in all_slugs:
        a = a_by_slug.get(slug)
        b = b_by_slug.get(slug)
        c = c_by_slug.get(slug)

        # Use whichever exists for common fields
        ref = a or b or c
        if not ref:
            continue

        icon = "W" if ref.won else "L"

        a_pnl = f"{a.pnl:+7.2f}" if a else "    n/a"
        a_roi = f"{a.roi_pct:+5.1f}%" if a else "   n/a"
        b_pnl = f"{b.pnl:+7.2f}" if b else "    n/a"
        b_roi = f"{b.roi_pct:+5.1f}%" if b else "   n/a"
        b_fills = f"{len(b.fills):7d}" if b else "    n/a"
        c_pnl = f"{c.pnl:+7.2f}" if c else "    n/a"
        c_roi = f"{c.roi_pct:+5.1f}%" if c else "   n/a"
        c_stop = "STOP" if c and c.stopped else ""

        if a: total_a += a.pnl
        if b: total_b += b.pnl
        if c: total_c += c.pnl

        print(
            f"  {slug:<35} {ref.btc_return_pct:+6.4f}% {ref.p_win:.2f} {icon:>4}  "
            f"{a_pnl} {a_roi}  {b_pnl} {b_roi} {b_fills}  {c_pnl} {c_roi} {c_stop:>6}"
        )

    print(f"\n  {'TOTALS':<35} {'':>7} {'':>4} {'':>4}  "
          f"{total_a:+7.2f} {'':>6}  {total_b:+7.2f} {'':>6} {'':>7}  "
          f"{total_c:+7.2f}")

    # ================================================================
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(fmt(compute_metrics(r_a), "A: Single entry ($100)"))
    print(fmt(compute_metrics(r_b), "B: Accumulate (no stop)"))
    print(fmt(compute_metrics(r_c), "C: Accum + 30% DD stop"))

    # ================================================================
    # What about stop + signal threshold interaction?
    print("\n" + "=" * 100)
    print("SIGNAL THRESHOLD × STOP INTERACTION — edge=0.10, cap=$100")
    print("=" * 100)

    print(f"\n  {'config':<45} {'mkts':>4} {'WR':>5} {'PnL':>9} {'ROI':>6} {'Sharpe':>7} {'maxDD':>6}")
    for ms in [0.0, 0.00005, 0.0001, 0.0002, 0.0003]:
        for stop_type, dd, dump in [("no stop", 999, False), ("30% hold", 0.30, False), ("30% dump", 0.30, True)]:
            if stop_type == "no stop":
                r = run_accumulate(markets, SD, ms, 0.10, 5, 100)
            else:
                r = run_accumulate_with_stop(markets, SD, ms, 0.10, 5, 100,
                                              drawdown_pct=dd, dump_on_stop=dump)
            m = compute_metrics(r)
            if not m:
                continue
            label = f"sig>={ms*100:.3f}% {stop_type}"
            print(
                f"  {label:<45} {m['n']:4d} {m['wr']*100:4.0f}% {m['pnl']:+8.2f} "
                f"{m['roi']:+5.1f}% {m['sharpe']:6.2f} {m['max_dd']:6.0f}"
            )


if __name__ == "__main__":
    main()
