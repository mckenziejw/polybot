"""
Momentum strategy v2 backtest — value-hunting and position sizing.

Three improvements over v1:
  1. Avoid the dead zone (ask near 0.50 where signal is noise)
  2. Scale position size by entry price (cheaper = larger bet)
  3. Estimate win probability from signal strength, only buy when ask < p_win
     (the market is offering us a price below our estimated edge)

The key insight: our BTC momentum signal gives us a *conviction* about market
resolution. We can estimate the probability the signal is correct as a function
of its magnitude. Then we only enter when the market price is below that
probability — i.e., when we're getting +EV odds.

Kelly sizing: f* = (p * b - q) / b where b = (1-ask)/ask, p = win_prob, q = 1-p
Simplified for binary: f* = p - ask (fraction of bankroll)
We use fractional Kelly (0.25x) to be conservative.
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


# ── Signal → win probability estimation ──

def estimate_win_prob(btc_return_abs: float) -> float:
    """Estimate probability our directional signal is correct, given |BTC return|.

    Based on empirical win rates from v1 backtest (60 markets):
      |ret| < 0.005%:  ~50% (coin flip — BTC barely moved)
      |ret| 0.005-0.01%: ~55%
      |ret| 0.01-0.02%: ~65%
      |ret| 0.02-0.03%: ~75%
      |ret| >= 0.03%:   ~85%

    We use piecewise linear interpolation between these anchors.
    Capped at 0.90 — we're not *that* confident.
    """
    # (threshold_pct, win_prob) anchor points
    anchors = [
        (0.000, 0.50),
        (0.005, 0.55),
        (0.010, 0.60),
        (0.015, 0.67),
        (0.020, 0.75),
        (0.030, 0.83),
        (0.050, 0.88),
    ]

    ret_pct = btc_return_abs * 100  # convert to percentage

    if ret_pct <= anchors[0][0]:
        return anchors[0][1]
    if ret_pct >= anchors[-1][0]:
        return min(anchors[-1][1], 0.90)

    # Linear interpolation
    for i in range(len(anchors) - 1):
        x0, y0 = anchors[i]
        x1, y1 = anchors[i + 1]
        if x0 <= ret_pct <= x1:
            t = (ret_pct - x0) / (x1 - x0)
            return y0 + t * (y1 - y0)

    return 0.50


# ── Helpers ──

def get_btc_at_time(ticks: list[MarketTick], target: float) -> float | None:
    for t in ticks:
        if t.elapsed_s >= target and t.btc_price is not None:
            return t.btc_price
    return None


def get_ask_at_time(ticks: list[MarketTick], target: float, direction: str) -> float | None:
    for t in ticks:
        if t.elapsed_s >= target:
            return t.up_best_ask if direction == "Up" else t.down_best_ask
    return None


def get_best_ask_in_window(
    ticks: list[MarketTick], start: float, end: float, direction: str
) -> tuple[float, float] | None:
    """Find the lowest ask price in [start, end] for direction.
    Returns (elapsed_s, price) or None."""
    best_price = None
    best_time = None
    for t in ticks:
        if t.elapsed_s < start:
            continue
        if t.elapsed_s > end:
            break
        price = t.up_best_ask if direction == "Up" else t.down_best_ask
        if price is not None and (best_price is None or price < best_price):
            best_price = price
            best_time = t.elapsed_s
    if best_price is None:
        return None
    return (best_time, best_price)


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
    # Fallback
    btc_prices = [t.btc_price for t in ticks if t.btc_price is not None]
    if len(btc_prices) >= 2:
        if btc_prices[-1] > btc_prices[0]:
            return "Up"
        elif btc_prices[-1] < btc_prices[0]:
            return "Down"
    return None


# ── Trade result ──

@dataclass
class TradeResult:
    slug: str
    btc_return_pct: float
    direction: str
    entry_price: float
    entry_time_s: float
    estimated_prob: float
    edge: float            # estimated_prob - entry_price
    size: float
    outcome: str
    won: bool
    pnl: float
    roi_pct: float
    strategy: str


# ── Strategy variants ──

def run_v1_baseline(
    markets: dict[str, MarketData],
    signal_delay: float = 5.0,
    min_threshold: float = 0.0002,
    max_ask: float = 0.65,
    order_size: float = 5.0,
) -> list[TradeResult]:
    """Original v1: snapshot signal, fixed size, maxAsk cap."""
    results = []
    for slug in sorted(markets):
        m = markets[slug]
        ticks = m.ticks
        if len(ticks) < 10 or max(t.elapsed_s for t in ticks) < 200:
            continue

        btc_ref = next((t.btc_price for t in ticks if t.btc_price), None)
        if not btc_ref:
            continue

        btc_sig = get_btc_at_time(ticks, signal_delay)
        if not btc_sig:
            continue

        btc_ret = (btc_sig - btc_ref) / btc_ref
        if abs(btc_ret) < min_threshold:
            continue

        direction = "Up" if btc_ret > 0 else "Down"
        ask = get_ask_at_time(ticks, signal_delay, direction)
        if not ask or ask > max_ask:
            continue

        outcome = determine_outcome(ticks)
        if not outcome:
            continue

        won = direction == outcome
        pnl = order_size * (1.0 - ask) if won else order_size * (-ask)

        results.append(TradeResult(
            slug=slug, btc_return_pct=btc_ret * 100, direction=direction,
            entry_price=ask, entry_time_s=signal_delay,
            estimated_prob=0.0, edge=0.0, size=order_size,
            outcome=outcome, won=won, pnl=pnl,
            roi_pct=((1.0 - ask) / ask * 100) if won else -100.0,
            strategy="v1_baseline",
        ))
    return results


def run_v2_avoid_middle(
    markets: dict[str, MarketData],
    signal_delay: float = 5.0,
    min_threshold: float = 0.0002,
    max_ask: float = 0.65,
    dead_zone: tuple[float, float] = (0.48, 0.55),
    order_size: float = 5.0,
) -> list[TradeResult]:
    """v1 + skip the dead zone where ask is near 0.50."""
    results = []
    for slug in sorted(markets):
        m = markets[slug]
        ticks = m.ticks
        if len(ticks) < 10 or max(t.elapsed_s for t in ticks) < 200:
            continue

        btc_ref = next((t.btc_price for t in ticks if t.btc_price), None)
        if not btc_ref:
            continue

        btc_sig = get_btc_at_time(ticks, signal_delay)
        if not btc_sig:
            continue

        btc_ret = (btc_sig - btc_ref) / btc_ref
        if abs(btc_ret) < min_threshold:
            continue

        direction = "Up" if btc_ret > 0 else "Down"
        ask = get_ask_at_time(ticks, signal_delay, direction)
        if not ask or ask > max_ask:
            continue

        # Skip dead zone
        if dead_zone[0] <= ask <= dead_zone[1]:
            continue

        outcome = determine_outcome(ticks)
        if not outcome:
            continue

        won = direction == outcome
        pnl = order_size * (1.0 - ask) if won else order_size * (-ask)

        results.append(TradeResult(
            slug=slug, btc_return_pct=btc_ret * 100, direction=direction,
            entry_price=ask, entry_time_s=signal_delay,
            estimated_prob=0.0, edge=0.0, size=order_size,
            outcome=outcome, won=won, pnl=pnl,
            roi_pct=((1.0 - ask) / ask * 100) if won else -100.0,
            strategy="v2_no_middle",
        ))
    return results


def run_v3_scaled_size(
    markets: dict[str, MarketData],
    signal_delay: float = 5.0,
    min_threshold: float = 0.0002,
    max_ask: float = 0.65,
    dead_zone: tuple[float, float] = (0.48, 0.55),
    base_size: float = 5.0,
    max_size: float = 20.0,
) -> list[TradeResult]:
    """v2 + scale size inversely with ask price.
    Cheaper entries get bigger bets (better risk/reward)."""
    results = []
    for slug in sorted(markets):
        m = markets[slug]
        ticks = m.ticks
        if len(ticks) < 10 or max(t.elapsed_s for t in ticks) < 200:
            continue

        btc_ref = next((t.btc_price for t in ticks if t.btc_price), None)
        if not btc_ref:
            continue

        btc_sig = get_btc_at_time(ticks, signal_delay)
        if not btc_sig:
            continue

        btc_ret = (btc_sig - btc_ref) / btc_ref
        if abs(btc_ret) < min_threshold:
            continue

        direction = "Up" if btc_ret > 0 else "Down"
        ask = get_ask_at_time(ticks, signal_delay, direction)
        if not ask or ask > max_ask:
            continue

        if dead_zone[0] <= ask <= dead_zone[1]:
            continue

        outcome = determine_outcome(ticks)
        if not outcome:
            continue

        # Scale: at ask=0.40, size=base*1.5; at ask=0.65, size=base*0.8
        # Linear scale: size = base * (1 + (0.55 - ask) * 2)
        scale = max(0.5, min(2.0, 1.0 + (0.55 - ask) * 3))
        size = min(max_size, max(base_size, base_size * scale))

        won = direction == outcome
        pnl = size * (1.0 - ask) if won else size * (-ask)

        results.append(TradeResult(
            slug=slug, btc_return_pct=btc_ret * 100, direction=direction,
            entry_price=ask, entry_time_s=signal_delay,
            estimated_prob=0.0, edge=0.0, size=size,
            outcome=outcome, won=won, pnl=pnl,
            roi_pct=((1.0 - ask) / ask * 100) if won else -100.0,
            strategy="v3_scaled",
        ))
    return results


def run_v4_value_hunt(
    markets: dict[str, MarketData],
    signal_delay: float = 5.0,
    min_signal: float = 0.00005,   # very low — let the edge filter do the work
    min_edge: float = 0.05,        # only enter if estimated_prob - ask >= this
    hunt_window: float = 30.0,     # seconds after signal to hunt for best price
    base_size: float = 5.0,
    max_size: float = 20.0,
    kelly_fraction: float = 0.25,
) -> list[TradeResult]:
    """The full value-hunting strategy:
    1. At signal_delay, read BTC return → estimate win probability
    2. Hunt for best ask in [signal_delay, signal_delay + hunt_window]
    3. Only enter if ask < estimated_prob - min_edge
    4. Size using fractional Kelly: size = kelly * (p - ask) / (1 - ask) * bankroll
       Simplified: size proportional to edge.
    """
    results = []
    for slug in sorted(markets):
        m = markets[slug]
        ticks = m.ticks
        if len(ticks) < 10 or max(t.elapsed_s for t in ticks) < 200:
            continue

        btc_ref = next((t.btc_price for t in ticks if t.btc_price), None)
        if not btc_ref:
            continue

        btc_sig = get_btc_at_time(ticks, signal_delay)
        if not btc_sig:
            continue

        btc_ret = (btc_sig - btc_ref) / btc_ref
        if abs(btc_ret) < min_signal:
            continue

        direction = "Up" if btc_ret > 0 else "Down"
        p_win = estimate_win_prob(abs(btc_ret))

        # Hunt for best ask in window
        hunt_end = signal_delay + hunt_window
        best = get_best_ask_in_window(ticks, signal_delay, hunt_end, direction)
        if best is None:
            continue
        entry_time, ask = best

        # Check edge
        edge = p_win - ask
        if edge < min_edge:
            continue

        outcome = determine_outcome(ticks)
        if not outcome:
            continue

        # Kelly sizing: f* = edge / (1 - ask) for binary outcome
        # Then scale by kelly_fraction and base_size
        kelly_raw = edge / (1.0 - ask) if ask < 1.0 else 0
        size = max(base_size, min(max_size, base_size * (1 + kelly_raw * 10 * kelly_fraction)))

        won = direction == outcome
        pnl = size * (1.0 - ask) if won else size * (-ask)

        results.append(TradeResult(
            slug=slug, btc_return_pct=btc_ret * 100, direction=direction,
            entry_price=ask, entry_time_s=entry_time,
            estimated_prob=p_win, edge=edge, size=size,
            outcome=outcome, won=won, pnl=pnl,
            roi_pct=((1.0 - ask) / ask * 100) if won else -100.0,
            strategy="v4_value_hunt",
        ))
    return results


# ── Metrics ──

def compute_metrics(results: list[TradeResult]) -> dict:
    if not results:
        return {}

    wins = sum(1 for r in results if r.won)
    losses = len(results) - wins
    total_pnl = sum(r.pnl for r in results)
    total_risked = sum(r.size * r.entry_price for r in results)
    avg_entry = sum(r.entry_price for r in results) / len(results)
    avg_size = sum(r.size for r in results) / len(results)
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
        "total_risked": total_risked,
        "roi_total": total_pnl / total_risked * 100 if total_risked > 0 else 0,
        "ev_per_trade": ev,
        "avg_entry": avg_entry,
        "avg_size": avg_size,
        "pnl_stdev": stdev,
        "sharpe": sharpe,
    }


def print_metrics(m: dict, label: str):
    if not m:
        print(f"\n{label}: No trades")
        return
    print(f"\n{label}")
    print(f"  Trades: {m['trades']} ({m['wins']}W / {m['losses']}L)")
    print(f"  Win rate: {m['win_rate']*100:.1f}%")
    print(f"  Total PnL: {m['total_pnl']:+.2f} USDC | Total risked: {m['total_risked']:.2f}")
    print(f"  ROI on capital: {m['roi_total']:+.1f}%")
    print(f"  EV/trade: {m['ev_per_trade']:+.3f} USDC")
    print(f"  Avg entry: {m['avg_entry']:.3f} | Avg size: {m['avg_size']:.1f} tokens")
    print(f"  PnL stdev: {m['pnl_stdev']:.3f} | Sharpe: {m['sharpe']:.2f}")


def print_trades(results: list[TradeResult]):
    print(f"  {'slug':<35} {'sig%':>7} {'p_win':>5} {'ask':>5} {'edge':>5} {'size':>5} {'dir':>4} {'out':>4} {'pnl':>7}")
    for r in results:
        icon = "W" if r.won else "L"
        p_str = f"{r.estimated_prob:.2f}" if r.estimated_prob > 0 else "  -  "
        e_str = f"{r.edge:.3f}" if r.edge > 0 else "  -  "
        print(
            f"  {r.slug:<35} {r.btc_return_pct:+6.4f}% {p_str} {r.entry_price:5.3f} "
            f"{e_str} {r.size:5.1f} {r.direction:>4} {icon:>4} {r.pnl:+6.2f}"
        )


def main():
    print("Loading tick data...")
    markets = load_all_markets()
    print(f"Loaded {len(markets)} unique markets\n")

    full = {s: m for s, m in markets.items()
            if len(m.ticks) >= 10 and max(t.elapsed_s for t in m.ticks) >= 200}
    print(f"Markets with full coverage: {len(full)}\n")

    # ================================================================
    print("=" * 75)
    print("STRATEGY COMPARISON (5s signal delay)")
    print("=" * 75)

    r1 = run_v1_baseline(markets)
    print_metrics(compute_metrics(r1), "v1: Baseline (5s snapshot, 0.02% thresh, size=5)")

    r2 = run_v2_avoid_middle(markets)
    print_metrics(compute_metrics(r2), "v2: + Avoid dead zone (0.48-0.55)")

    r3 = run_v3_scaled_size(markets)
    print_metrics(compute_metrics(r3), "v3: + Scaled size (cheaper → bigger bet)")

    r4 = run_v4_value_hunt(markets, min_edge=0.05, hunt_window=30)
    print_metrics(compute_metrics(r4), "v4: Value hunt (edge>=0.05, 30s window, Kelly 0.25x)")

    # ================================================================
    print("\n" + "=" * 75)
    print("v4 VALUE HUNT — PARAMETER SWEEP")
    print("=" * 75)

    print("\n--- Min edge sweep (5s signal, 30s hunt window) ---")
    for min_edge in [0.0, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20]:
        r = run_v4_value_hunt(markets, min_edge=min_edge)
        m = compute_metrics(r)
        if m:
            print(
                f"  edge>={min_edge:.2f}: {m['trades']:3d}t, {m['win_rate']*100:4.0f}% WR, "
                f"PnL={m['total_pnl']:+7.2f}, EV/t={m['ev_per_trade']:+.2f}, "
                f"ROI={m['roi_total']:+.1f}%, Sharpe={m['sharpe']:.2f}"
            )
        else:
            print(f"  edge>={min_edge:.2f}: no trades")

    print("\n--- Hunt window sweep (5s signal, edge>=0.05) ---")
    for window in [0, 5, 10, 15, 30, 60, 90]:
        r = run_v4_value_hunt(markets, hunt_window=window, min_edge=0.05)
        m = compute_metrics(r)
        if m:
            print(
                f"  hunt={window:2d}s: {m['trades']:3d}t, {m['win_rate']*100:4.0f}% WR, "
                f"PnL={m['total_pnl']:+7.2f}, EV/t={m['ev_per_trade']:+.2f}, "
                f"avg_entry={m['avg_entry']:.3f}, Sharpe={m['sharpe']:.2f}"
            )
        else:
            print(f"  hunt={window:2d}s: no trades")

    print("\n--- Signal delay sweep (edge>=0.05, 30s hunt) ---")
    for delay in [3, 5, 7, 10, 15]:
        r = run_v4_value_hunt(markets, signal_delay=delay, min_edge=0.05)
        m = compute_metrics(r)
        if m:
            print(
                f"  delay={delay:2d}s: {m['trades']:3d}t, {m['win_rate']*100:4.0f}% WR, "
                f"PnL={m['total_pnl']:+7.2f}, EV/t={m['ev_per_trade']:+.2f}, Sharpe={m['sharpe']:.2f}"
            )
        else:
            print(f"  delay={delay:2d}s: no trades")

    print("\n--- Kelly fraction sweep (5s signal, edge>=0.05, 30s hunt) ---")
    for kf in [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]:
        r = run_v4_value_hunt(markets, min_edge=0.05, kelly_fraction=kf)
        m = compute_metrics(r)
        if m:
            print(
                f"  kelly={kf:.2f}: {m['trades']:3d}t, PnL={m['total_pnl']:+7.2f}, "
                f"avg_size={m['avg_size']:.1f}, risked={m['total_risked']:.1f}, "
                f"ROI={m['roi_total']:+.1f}%"
            )

    # ================================================================
    print("\n" + "=" * 75)
    print("v4 VALUE HUNT — DETAILED TRADES (edge>=0.05, 30s hunt)")
    print("=" * 75)

    r4 = run_v4_value_hunt(markets, min_edge=0.05, hunt_window=30)
    print_trades(r4)
    print_metrics(compute_metrics(r4), "Summary")

    # ================================================================
    print("\n" + "=" * 75)
    print("WIN PROBABILITY CALIBRATION CHECK")
    print("=" * 75)
    print("\nDoes estimate_win_prob match actual outcomes?")
    print("(Using all markets, no threshold, snapshot at 5s)\n")

    # Run with very low threshold to get all signals
    all_trades = run_v1_baseline(markets, min_threshold=0.0, max_ask=1.0)
    prob_buckets: dict[str, list[TradeResult]] = defaultdict(list)
    for r in all_trades:
        p = estimate_win_prob(abs(r.btc_return_pct / 100))
        bucket = f"{p:.2f}"
        prob_buckets[bucket].append(r)

    print(f"  {'est_prob':>8} {'trades':>6} {'actual_WR':>10} {'calibration':>12}")
    for bucket_key in sorted(prob_buckets.keys()):
        trades = prob_buckets[bucket_key]
        actual_wr = sum(1 for t in trades if t.won) / len(trades)
        est_p = float(bucket_key)
        cal = actual_wr - est_p
        print(f"  {est_p:8.2f} {len(trades):6d} {actual_wr:10.2f} {cal:+11.2f}")

    # ================================================================
    print("\n" + "=" * 75)
    print("EDGE DISTRIBUTION — all markets with signal")
    print("=" * 75)
    print("\nFor every market, compute estimated_prob and best ask in 30s window:\n")
    print(f"  {'slug':<35} {'sig%':>7} {'p_win':>5} {'ask_0':>5} {'ask_best':>8} {'edge':>6} {'out':>4} {'correct':>7}")

    for slug in sorted(markets):
        m = markets[slug]
        ticks = m.ticks
        if len(ticks) < 10 or max(t.elapsed_s for t in ticks) < 200:
            continue

        btc_ref = next((t.btc_price for t in ticks if t.btc_price), None)
        if not btc_ref:
            continue

        btc_sig = get_btc_at_time(ticks, 5.0)
        if not btc_sig:
            continue

        btc_ret = (btc_sig - btc_ref) / btc_ref
        if abs(btc_ret) < 1e-7:
            continue

        direction = "Up" if btc_ret > 0 else "Down"
        p_win = estimate_win_prob(abs(btc_ret))

        ask_now = get_ask_at_time(ticks, 5.0, direction)
        best = get_best_ask_in_window(ticks, 5.0, 35.0, direction)
        ask_best = best[1] if best else ask_now

        outcome = determine_outcome(ticks)
        if not outcome:
            continue

        correct = "Y" if direction == outcome else "N"
        edge = p_win - ask_best if ask_best else 0

        ask_now_str = f"{ask_now:.3f}" if ask_now else "  n/a"
        ask_best_str = f"{ask_best:.3f}" if ask_best else "     n/a"

        print(
            f"  {slug:<35} {btc_ret*100:+6.4f}% {p_win:.2f} {ask_now_str} "
            f"{ask_best_str} {edge:+5.3f} {outcome:>4} {correct:>7}"
        )


if __name__ == "__main__":
    main()
