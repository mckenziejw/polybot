"""
Momentum strategy v3 backtest — accumulation through the market.

Strategy:
  1. At signal_delay, read BTC return → decide direction, estimate win probability
  2. Compute target price = estimated_prob - desired_edge (our max buy price)
  3. Walk forward through all remaining ticks:
     - Whenever best ask <= target price, simulate a fill at the ask
     - After each fill, either:
       a) Place next order at target_price, OR
       b) Place at min(target_price, current_ask) if ask is running below target
     - Continue until max_capital is reached or market closes
  4. All positions held to expiry

This simulates patient limit-order accumulation with no look-ahead bias.
Each "fill" is modeled as buying order_lot tokens at the ask when ask <= target.
"""

import csv
import glob
import math
import os
from collections import defaultdict
from dataclasses import dataclass, field

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
    """Piecewise linear estimate of P(signal correct) given |BTC return|."""
    anchors = [
        (0.000, 0.50),
        (0.005, 0.55),
        (0.010, 0.60),
        (0.015, 0.67),
        (0.020, 0.75),
        (0.030, 0.83),
        (0.050, 0.88),
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
    cost: float  # price * size


@dataclass
class MarketResult:
    slug: str
    btc_return_pct: float
    direction: str
    estimated_prob: float
    target_price: float
    fills: list[Fill]
    total_tokens: float
    total_cost: float
    vwap: float
    outcome: str
    won: bool
    pnl: float
    roi_pct: float


def simulate_market(
    market: MarketData,
    signal_delay: float,
    min_signal: float,
    desired_edge: float,
    order_lot: float,
    max_capital: float,
    stop_accumulating_s: float,  # stop buying this many seconds before close
) -> MarketResult | None:
    ticks = market.ticks
    if len(ticks) < 10:
        return None

    max_elapsed = max(t.elapsed_s for t in ticks)
    if max_elapsed < 200:
        return None

    # BTC reference price
    btc_ref = next((t.btc_price for t in ticks if t.btc_price), None)
    if not btc_ref:
        return None

    # BTC at signal time
    btc_sig = get_btc_at_time(ticks, signal_delay)
    if not btc_sig:
        return None

    btc_ret = (btc_sig - btc_ref) / btc_ref
    if abs(btc_ret) < min_signal:
        return None

    direction = "Up" if btc_ret > 0 else "Down"
    p_win = estimate_win_prob(abs(btc_ret))
    target_price = p_win - desired_edge

    if target_price <= 0.01:
        return None

    outcome = determine_outcome(ticks)
    if not outcome:
        return None

    # Walk forward: accumulate whenever ask <= target_price
    fills: list[Fill] = []
    total_cost = 0.0

    for t in ticks:
        if t.elapsed_s < signal_delay:
            continue

        # Stop accumulating near market close
        if t.remaining_s < stop_accumulating_s:
            break

        # Check if we've hit capital limit
        if total_cost >= max_capital:
            break

        # Get current ask for our direction
        ask = t.up_best_ask if direction == "Up" else t.down_best_ask
        if ask is None:
            continue

        # Fill condition: ask is at or below our target
        if ask <= target_price:
            # How many tokens can we still afford?
            remaining_capital = max_capital - total_cost
            affordable = remaining_capital / ask
            size = min(order_lot, affordable)
            if size < 1:  # minimum viable order
                break

            cost = size * ask
            fills.append(Fill(
                elapsed_s=t.elapsed_s,
                price=ask,
                size=size,
                cost=cost,
            ))
            total_cost += cost

    if not fills:
        return None

    total_tokens = sum(f.size for f in fills)
    vwap = total_cost / total_tokens

    won = direction == outcome
    payout = total_tokens * 1.0 if won else 0.0
    pnl = payout - total_cost

    return MarketResult(
        slug=market.slug,
        btc_return_pct=btc_ret * 100,
        direction=direction,
        estimated_prob=p_win,
        target_price=target_price,
        fills=fills,
        total_tokens=total_tokens,
        total_cost=total_cost,
        vwap=vwap,
        outcome=outcome,
        won=won,
        pnl=pnl,
        roi_pct=(pnl / total_cost * 100) if total_cost > 0 else 0,
    )


def run_simulation(
    markets: dict[str, MarketData],
    signal_delay: float = 5.0,
    min_signal: float = 0.00005,
    desired_edge: float = 0.10,
    order_lot: float = 5.0,
    max_capital: float = 100.0,
    stop_accumulating_s: float = 30.0,
) -> list[MarketResult]:
    results = []
    for slug in sorted(markets):
        r = simulate_market(
            markets[slug], signal_delay, min_signal, desired_edge,
            order_lot, max_capital, stop_accumulating_s,
        )
        if r:
            results.append(r)
    return results


def compute_metrics(results: list[MarketResult]) -> dict:
    if not results:
        return {}
    wins = sum(1 for r in results if r.won)
    losses = len(results) - wins
    total_pnl = sum(r.pnl for r in results)
    total_cost = sum(r.total_cost for r in results)
    total_fills = sum(len(r.fills) for r in results)
    total_tokens = sum(r.total_tokens for r in results)
    avg_vwap = sum(r.vwap for r in results) / len(results)
    ev = total_pnl / len(results)

    pnls = [r.pnl for r in results]
    variance = sum((p - ev) ** 2 for p in pnls) / len(pnls)
    stdev = math.sqrt(variance) if variance > 0 else 0
    sharpe = ev / stdev if stdev > 0 else float("inf")

    return {
        "trades": len(results),
        "wins": wins,
        "losses": losses,
        "win_rate": wins / len(results),
        "total_pnl": total_pnl,
        "total_cost": total_cost,
        "roi": total_pnl / total_cost * 100 if total_cost > 0 else 0,
        "total_fills": total_fills,
        "avg_fills_per_market": total_fills / len(results),
        "total_tokens": total_tokens,
        "avg_vwap": avg_vwap,
        "ev_per_market": ev,
        "pnl_stdev": stdev,
        "sharpe": sharpe,
    }


def print_metrics(m: dict, label: str):
    if not m:
        print(f"\n{label}: No trades")
        return
    print(f"\n{label}")
    print(f"  Markets: {m['trades']} ({m['wins']}W / {m['losses']}L) | Win rate: {m['win_rate']*100:.1f}%")
    print(f"  Total PnL: {m['total_pnl']:+.2f} USDC | Capital deployed: {m['total_cost']:.2f}")
    print(f"  ROI on capital: {m['roi']:+.1f}%")
    print(f"  Total fills: {m['total_fills']} ({m['avg_fills_per_market']:.1f}/market)")
    print(f"  Total tokens: {m['total_tokens']:.1f} | Avg VWAP: {m['avg_vwap']:.3f}")
    print(f"  EV/market: {m['ev_per_market']:+.2f} USDC")
    print(f"  PnL stdev: {m['pnl_stdev']:.2f} | Sharpe: {m['sharpe']:.2f}")


def print_detailed(results: list[MarketResult]):
    print(f"\n  {'slug':<35} {'sig%':>7} {'p':>4} {'tgt':>5} {'fills':>5} {'tokens':>6} "
          f"{'cost':>6} {'vwap':>5} {'dir':>4} {'out':>4} {'pnl':>7} {'roi':>6}")
    for r in results:
        icon = "W" if r.won else "L"
        print(
            f"  {r.slug:<35} {r.btc_return_pct:+6.4f}% {r.estimated_prob:.2f} "
            f"{r.target_price:.3f} {len(r.fills):5d} {r.total_tokens:6.1f} "
            f"{r.total_cost:6.2f} {r.vwap:.3f} {r.direction:>4} {icon:>4} "
            f"{r.pnl:+7.2f} {r.roi_pct:+5.1f}%"
        )


def print_fill_detail(results: list[MarketResult], max_markets: int = 5):
    """Show individual fills for a few markets."""
    shown = 0
    for r in results:
        if shown >= max_markets:
            break
        if len(r.fills) < 2:
            continue
        icon = "W" if r.won else "L"
        print(f"\n  {r.slug} [{icon}] — {r.direction} target={r.target_price:.3f} p_win={r.estimated_prob:.2f}")
        for i, f in enumerate(r.fills):
            print(f"    fill {i+1:2d}: t={f.elapsed_s:5.1f}s  price={f.price:.3f}  size={f.size:.1f}  cost={f.cost:.2f}")
        print(f"    Total: {r.total_tokens:.1f} tokens @ VWAP {r.vwap:.3f}, cost={r.total_cost:.2f} → pnl={r.pnl:+.2f}")
        shown += 1


def main():
    print("Loading tick data...")
    markets = load_all_markets()
    full = {s: m for s, m in markets.items()
            if len(m.ticks) >= 10 and max(t.elapsed_s for t in m.ticks) >= 200}
    print(f"Loaded {len(markets)} markets ({len(full)} with full coverage)\n")

    # ================================================================
    print("=" * 80)
    print("ACCUMULATION STRATEGY — PARAMETER SWEEP")
    print("=" * 80)

    # Edge sweep
    print("\n--- Desired edge sweep (5s signal, lot=5, cap=$100, stop@30s) ---")
    for edge in [0.0, 0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25]:
        r = run_simulation(markets, desired_edge=edge)
        m = compute_metrics(r)
        if m:
            print(
                f"  edge={edge:.2f}: {m['trades']:3d} mkts, {m['win_rate']*100:4.0f}% WR, "
                f"PnL={m['total_pnl']:+7.2f}, {m['avg_fills_per_market']:4.1f} fills/mkt, "
                f"vwap={m['avg_vwap']:.3f}, ROI={m['roi']:+.1f}%, Sharpe={m['sharpe']:.2f}"
            )
        else:
            print(f"  edge={edge:.2f}: no trades")

    # Capital cap sweep
    print("\n--- Max capital sweep (5s signal, edge=0.10, lot=5, stop@30s) ---")
    for cap in [10, 25, 50, 100, 200, 500]:
        r = run_simulation(markets, desired_edge=0.10, max_capital=cap)
        m = compute_metrics(r)
        if m:
            print(
                f"  cap=${cap:3d}: {m['trades']:3d} mkts, PnL={m['total_pnl']:+7.2f}, "
                f"deployed={m['total_cost']:7.2f}, {m['avg_fills_per_market']:4.1f} fills/mkt, "
                f"ROI={m['roi']:+.1f}%"
            )

    # Order lot size sweep
    print("\n--- Order lot sweep (5s signal, edge=0.10, cap=$100, stop@30s) ---")
    for lot in [5, 10, 15, 20, 50]:
        r = run_simulation(markets, desired_edge=0.10, order_lot=lot)
        m = compute_metrics(r)
        if m:
            print(
                f"  lot={lot:2d}: {m['trades']:3d} mkts, PnL={m['total_pnl']:+7.2f}, "
                f"{m['avg_fills_per_market']:4.1f} fills/mkt, avg_tokens={m['total_tokens']/m['trades']:.1f}, "
                f"ROI={m['roi']:+.1f}%"
            )

    # Signal delay sweep
    print("\n--- Signal delay sweep (edge=0.10, lot=5, cap=$100, stop@30s) ---")
    for delay in [3, 5, 7, 10, 15]:
        r = run_simulation(markets, signal_delay=delay, desired_edge=0.10)
        m = compute_metrics(r)
        if m:
            print(
                f"  delay={delay:2d}s: {m['trades']:3d} mkts, {m['win_rate']*100:4.0f}% WR, "
                f"PnL={m['total_pnl']:+7.2f}, {m['avg_fills_per_market']:4.1f} fills/mkt, "
                f"ROI={m['roi']:+.1f}%, Sharpe={m['sharpe']:.2f}"
            )

    # Stop accumulating time sweep
    print("\n--- Stop-accumulating sweep (5s signal, edge=0.10, lot=5, cap=$100) ---")
    for stop in [0, 10, 30, 60, 120]:
        r = run_simulation(markets, desired_edge=0.10, stop_accumulating_s=stop)
        m = compute_metrics(r)
        if m:
            print(
                f"  stop@{stop:3d}s: {m['trades']:3d} mkts, {m['win_rate']*100:4.0f}% WR, "
                f"PnL={m['total_pnl']:+7.2f}, {m['avg_fills_per_market']:4.1f} fills/mkt, "
                f"vwap={m['avg_vwap']:.3f}, ROI={m['roi']:+.1f}%"
            )

    # Min signal strength sweep
    print("\n--- Min signal sweep (edge=0.10, lot=5, cap=$100, stop@30s) ---")
    for ms in [0.0, 0.00005, 0.0001, 0.0002, 0.0003]:
        r = run_simulation(markets, desired_edge=0.10, min_signal=ms)
        m = compute_metrics(r)
        if m:
            print(
                f"  min_sig={ms*100:.3f}%: {m['trades']:3d} mkts, {m['win_rate']*100:4.0f}% WR, "
                f"PnL={m['total_pnl']:+7.2f}, ROI={m['roi']:+.1f}%, Sharpe={m['sharpe']:.2f}"
            )

    # ================================================================
    print("\n" + "=" * 80)
    print("BEST CONFIG CANDIDATES — combined sweep (min 10 markets)")
    print("=" * 80)

    combos = []
    for delay in [3, 5, 7]:
        for edge in [0.05, 0.08, 0.10, 0.12, 0.15]:
            for ms in [0.0, 0.0001, 0.0002]:
                for stop in [10, 30, 60]:
                    r = run_simulation(
                        markets, signal_delay=delay, desired_edge=edge,
                        min_signal=ms, stop_accumulating_s=stop,
                    )
                    m = compute_metrics(r)
                    if m and m["trades"] >= 10:
                        combos.append((delay, edge, ms, stop, m))

    # Sort by total PnL
    combos.sort(key=lambda x: x[4]["total_pnl"], reverse=True)
    print(f"\n  {'delay':>5} {'edge':>5} {'sig%':>6} {'stop':>4} {'mkts':>4} "
          f"{'WR':>5} {'PnL':>8} {'ROI':>6} {'fills':>5} {'vwap':>5} {'Sharpe':>6}")
    print(f"  {'-'*5} {'-'*5} {'-'*6} {'-'*4} {'-'*4} {'-'*5} {'-'*8} {'-'*6} {'-'*5} {'-'*5} {'-'*6}")
    for delay, edge, ms, stop, m in combos[:20]:
        print(
            f"  {delay:4d}s {edge:5.2f} {ms*100:5.3f}% {stop:3d}s {m['trades']:4d} "
            f"{m['win_rate']*100:4.0f}% {m['total_pnl']:+7.2f} {m['roi']:+5.1f}% "
            f"{m['avg_fills_per_market']:5.1f} {m['avg_vwap']:.3f} {m['sharpe']:5.2f}"
        )

    # ================================================================
    # Pick a good config for detailed view
    print("\n" + "=" * 80)
    print("DETAILED RESULTS — edge=0.10, 5s signal, lot=5, cap=$100, stop@30s")
    print("=" * 80)

    results = run_simulation(markets, desired_edge=0.10)
    print_detailed(results)
    print_metrics(compute_metrics(results), "Summary")

    # Show fill-level detail for interesting markets
    print("\n--- Fill-level detail (markets with multiple fills) ---")
    # Show some winners and losers
    multi_fill = [r for r in results if len(r.fills) >= 3]
    print_fill_detail(multi_fill, max_markets=8)

    # ================================================================
    print("\n" + "=" * 80)
    print("ACCUMULATION DYNAMICS")
    print("=" * 80)

    # When do fills happen? Time distribution
    all_fill_times = []
    for r in results:
        for f in r.fills:
            all_fill_times.append(f.elapsed_s)

    if all_fill_times:
        print(f"\n  Total fills across all markets: {len(all_fill_times)}")
        print(f"  Fill time distribution:")
        for lo, hi in [(0, 10), (10, 30), (30, 60), (60, 120), (120, 180), (180, 270)]:
            count = sum(1 for t in all_fill_times if lo <= t < hi)
            if count:
                print(f"    {lo:3d}-{hi:3d}s: {count:3d} fills ({count/len(all_fill_times)*100:.0f}%)")

    # Comparison: how much better is accumulation vs single entry?
    print("\n--- Accumulation vs single-entry comparison ---")
    for r in results[:15]:
        first_fill = r.fills[0]
        single_pnl = first_fill.size * (1.0 - first_fill.price) if r.won else first_fill.size * (-first_fill.price)
        improvement = r.pnl - single_pnl
        print(
            f"  {r.slug}: single={single_pnl:+5.2f} (1x{first_fill.size:.0f}@{first_fill.price:.3f}) "
            f"vs accum={r.pnl:+6.2f} ({len(r.fills)}x fills, vwap={r.vwap:.3f}) "
            f"delta={improvement:+5.2f}"
        )


if __name__ == "__main__":
    main()
