"""
Backtest the BTC momentum strategy across all accumulated paper trading tick data.

Two signal modes:
  "snapshot" — BTC return at exactly signalDelaySec (current live strategy)
  "max_in_window" — max |BTC return| during [0, signalDelaySec], take direction of extreme

For each market:
  1. Record BTC price at first available tick (ref price)
  2. Compute signal per chosen mode
  3. If |return| >= threshold, "buy" the direction token at best ask
  4. Hold to expiry — determine outcome from final book prices / BTC
  5. Track win/loss and simulated PnL

Fill assumption: immediate fill at best ask at signal time. No slippage.
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
    """Load all tick CSVs, deduplicate markets by slug, return dict."""
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

    # Sort ticks by timestamp and deduplicate
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


def get_btc_at_time(ticks: list[MarketTick], target_elapsed: float) -> float | None:
    """Get BTC price at or just after target_elapsed seconds."""
    for t in ticks:
        if t.elapsed_s >= target_elapsed and t.btc_price is not None:
            return t.btc_price
    return None


def get_max_btc_return_in_window(
    ticks: list[MarketTick], btc_ref: float, window_end: float
) -> tuple[float, float] | None:
    """Get the BTC return with max absolute value during [0, window_end].
    Returns (btc_price_at_extreme, return_value) or None."""
    best_abs = 0.0
    best_price = None
    best_return = 0.0

    for t in ticks:
        if t.elapsed_s > window_end:
            break
        if t.btc_price is None:
            continue
        ret = (t.btc_price - btc_ref) / btc_ref
        if abs(ret) > best_abs:
            best_abs = abs(ret)
            best_price = t.btc_price
            best_return = ret

    if best_price is None:
        return None
    return (best_price, best_return)


def get_ask_at_time(
    ticks: list[MarketTick], target_elapsed: float, direction: str
) -> float | None:
    """Get the best ask for up/down token at or just after target_elapsed."""
    for t in ticks:
        if t.elapsed_s >= target_elapsed:
            if direction == "Up":
                return t.up_best_ask
            else:
                return t.down_best_ask
    return None


def determine_outcome(ticks: list[MarketTick]) -> str | None:
    """Determine market outcome from final tick prices."""
    for t in reversed(ticks):
        if t.up_best_bid is not None and t.up_best_bid >= 0.9:
            return "Up"
        if t.up_best_ask is not None and t.up_best_ask <= 0.1:
            return "Down"
        if t.down_best_bid is not None and t.down_best_bid >= 0.9:
            return "Down"
        if t.down_best_ask is not None and t.down_best_ask <= 0.1:
            return "Up"
    return None


def determine_outcome_from_btc(ticks: list[MarketTick]) -> str | None:
    """Fallback: determine outcome from BTC price movement."""
    btc_prices = [t.btc_price for t in ticks if t.btc_price is not None]
    if len(btc_prices) < 2:
        return None
    if btc_prices[-1] > btc_prices[0]:
        return "Up"
    elif btc_prices[-1] < btc_prices[0]:
        return "Down"
    return None


@dataclass
class TradeResult:
    slug: str
    signal_delay: float
    signal_mode: str
    btc_ref: float
    btc_at_signal: float
    btc_return_pct: float
    direction: str
    entry_price: float
    outcome: str
    pnl: float
    won: bool


def backtest_market(
    market: MarketData,
    signal_delay: float,
    min_threshold: float,
    max_ask: float,
    order_size: float,
    signal_mode: str = "snapshot",
) -> TradeResult | None:
    """Run momentum strategy on a single market. Returns None if no trade."""
    ticks = market.ticks

    if len(ticks) < 10:
        return None

    # Get BTC ref (first available price)
    btc_ref = None
    for t in ticks:
        if t.btc_price is not None:
            btc_ref = t.btc_price
            break
    if btc_ref is None:
        return None

    # Compute signal
    if signal_mode == "max_in_window":
        result = get_max_btc_return_in_window(ticks, btc_ref, signal_delay)
        if result is None:
            return None
        btc_signal, btc_return = result
    else:  # "snapshot"
        btc_signal = get_btc_at_time(ticks, signal_delay)
        if btc_signal is None:
            return None
        btc_return = (btc_signal - btc_ref) / btc_ref

    # Check threshold
    if abs(btc_return) < min_threshold:
        return None

    direction = "Up" if btc_return > 0 else "Down"

    # Get ask price at signal time
    ask = get_ask_at_time(ticks, signal_delay, direction)
    if ask is None or ask > max_ask:
        return None

    # Determine outcome
    outcome = determine_outcome(ticks)
    if outcome is None:
        outcome = determine_outcome_from_btc(ticks)
    if outcome is None:
        return None

    # Check if market has enough time coverage (at least 200s for 5-min markets)
    max_elapsed = max(t.elapsed_s for t in ticks)
    if max_elapsed < 200:
        return None

    won = direction == outcome
    pnl = order_size * (1.0 - ask) if won else order_size * (-ask)

    return TradeResult(
        slug=market.slug,
        signal_delay=signal_delay,
        signal_mode=signal_mode,
        btc_ref=btc_ref,
        btc_at_signal=btc_signal,
        btc_return_pct=btc_return * 100,
        direction=direction,
        entry_price=ask,
        outcome=outcome,
        pnl=pnl,
        won=won,
    )


def run_backtest(
    markets: dict[str, MarketData],
    signal_delay: float = 5.0,
    min_threshold: float = 0.0002,
    max_ask: float = 0.65,
    order_size: float = 5.0,
    signal_mode: str = "snapshot",
    verbose: bool = False,
) -> list[TradeResult]:
    results = []

    for slug in sorted(markets.keys()):
        market = markets[slug]
        result = backtest_market(
            market, signal_delay, min_threshold, max_ask, order_size, signal_mode
        )
        if result is None:
            continue
        results.append(result)

        if verbose:
            icon = "W" if result.won else "L"
            print(
                f"  [{icon}] {slug}: BTC {result.btc_return_pct:+.4f}% → {result.direction} "
                f"@ {result.entry_price:.3f} → {result.outcome} | pnl={result.pnl:+.3f}"
            )

    return results


def compute_metrics(results: list[TradeResult]) -> dict:
    """Compute summary metrics for a set of trade results."""
    if not results:
        return {}

    wins = sum(1 for r in results if r.won)
    losses = len(results) - wins
    total_pnl = sum(r.pnl for r in results)
    avg_entry = sum(r.entry_price for r in results) / len(results)
    avg_win = sum(r.pnl for r in results if r.won) / wins if wins > 0 else 0
    avg_loss = sum(r.pnl for r in results if not r.won) / losses if losses > 0 else 0
    win_rate = wins / len(results)

    # EV per trade
    ev = total_pnl / len(results)

    # Capital per trade = order_size * entry_price
    costs = [abs(r.entry_price * 5) for r in results]
    avg_cost = sum(costs) / len(costs)
    roi_per_trade = ev / avg_cost if avg_cost > 0 else 0

    # PnL stdev and Sharpe-like ratio (per-trade)
    pnls = [r.pnl for r in results]
    mean_pnl = total_pnl / len(pnls)
    variance = sum((p - mean_pnl) ** 2 for p in pnls) / len(pnls)
    stdev = math.sqrt(variance) if variance > 0 else 0
    sharpe = mean_pnl / stdev if stdev > 0 else float("inf")

    return {
        "trades": len(results),
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "ev_per_trade": ev,
        "avg_entry": avg_entry,
        "avg_cost": avg_cost,
        "roi_per_trade": roi_per_trade,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "pnl_stdev": stdev,
        "sharpe": sharpe,
    }


def print_metrics(m: dict, label: str = ""):
    if not m:
        print(f"{label}: No trades")
        return

    print(f"\n{label}")
    print(f"  Trades: {m['trades']} ({m['wins']}W / {m['losses']}L)")
    print(f"  Win rate: {m['win_rate']*100:.1f}%")
    print(f"  Total PnL: {m['total_pnl']:+.2f} USDC")
    print(f"  EV/trade: {m['ev_per_trade']:+.3f} USDC ({m['roi_per_trade']*100:+.1f}% ROI)")
    print(f"  Avg entry: {m['avg_entry']:.3f} | Avg cost: {m['avg_cost']:.2f} USDC")
    print(f"  Avg win: {m['avg_win']:+.3f} | Avg loss: {m['avg_loss']:+.3f}")
    print(f"  PnL stdev: {m['pnl_stdev']:.3f} | Sharpe (per-trade): {m['sharpe']:.2f}")


def print_sweep_row(label: str, results: list[TradeResult]):
    m = compute_metrics(results)
    if not m:
        print(f"  {label}: no trades")
        return
    print(
        f"  {label}: {m['trades']:3d} trades, {m['win_rate']*100:4.0f}% WR, "
        f"PnL={m['total_pnl']:+7.2f}, EV/trade={m['ev_per_trade']:+.2f}, "
        f"Sharpe={m['sharpe']:.2f}"
    )


def main():
    print("Loading tick data...")
    markets = load_all_markets()
    print(f"Loaded {len(markets)} unique markets\n")

    valid = {s: m for s, m in markets.items() if len(m.ticks) >= 10}
    btc_available = {
        s: m for s, m in valid.items()
        if any(t.btc_price is not None for t in m.ticks)
    }
    full_coverage = {
        s: m for s, m in btc_available.items()
        if max(t.elapsed_s for t in m.ticks) >= 200
    }
    print(f"Markets with >=10 ticks: {len(valid)}")
    print(f"Markets with BTC data: {len(btc_available)}")
    print(f"Markets with full coverage (>=200s): {len(full_coverage)}")

    # ================================================================
    # SNAPSHOT MODE — current live strategy logic
    # ================================================================
    print("\n" + "=" * 70)
    print("SNAPSHOT MODE (BTC price at exact delay time)")
    print("=" * 70)

    print("\n--- Signal delay sweep (threshold=0.02%, maxAsk=0.65) ---")
    for delay in [3, 5, 7, 10, 15, 20, 30]:
        results = run_backtest(markets, signal_delay=delay, min_threshold=0.0002)
        print_sweep_row(f"delay={delay:2d}s", results)

    print("\n--- Threshold sweep (delay=5s, maxAsk=0.65) ---")
    for thresh in [0.0, 0.00005, 0.0001, 0.00015, 0.0002, 0.0003, 0.0005, 0.001]:
        results = run_backtest(markets, signal_delay=5, min_threshold=thresh)
        print_sweep_row(f"thresh={thresh*100:.3f}%", results)

    print("\n--- Max ask price sweep (delay=5s, threshold=0.02%) ---")
    for max_ask in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        results = run_backtest(markets, signal_delay=5, min_threshold=0.0002, max_ask=max_ask)
        m = compute_metrics(results)
        if m:
            print(
                f"  maxAsk={max_ask:.2f}: {m['trades']:3d} trades, {m['win_rate']*100:4.0f}% WR, "
                f"PnL={m['total_pnl']:+7.2f}, avg_entry={m['avg_entry']:.3f}"
            )

    # ================================================================
    # MAX-IN-WINDOW MODE — use strongest signal in the window
    # ================================================================
    print("\n" + "=" * 70)
    print("MAX-IN-WINDOW MODE (max |BTC return| during [0, delay])")
    print("=" * 70)

    print("\n--- Signal window sweep (threshold=0.02%, maxAsk=0.65) ---")
    for delay in [3, 5, 7, 10, 15, 20, 30]:
        results = run_backtest(
            markets, signal_delay=delay, min_threshold=0.0002, signal_mode="max_in_window"
        )
        print_sweep_row(f"window={delay:2d}s", results)

    print("\n--- Threshold sweep, max_in_window (window=5s, maxAsk=0.65) ---")
    for thresh in [0.0, 0.00005, 0.0001, 0.00015, 0.0002, 0.0003, 0.0005, 0.001]:
        results = run_backtest(
            markets, signal_delay=5, min_threshold=thresh, signal_mode="max_in_window"
        )
        print_sweep_row(f"thresh={thresh*100:.3f}%", results)

    # ================================================================
    # HEAD-TO-HEAD: snapshot vs max_in_window at same params
    # ================================================================
    print("\n" + "=" * 70)
    print("HEAD-TO-HEAD COMPARISON")
    print("=" * 70)

    for delay in [3, 5, 7, 10]:
        for thresh in [0.0001, 0.0002, 0.0003]:
            r_snap = run_backtest(markets, signal_delay=delay, min_threshold=thresh, signal_mode="snapshot")
            r_maxw = run_backtest(markets, signal_delay=delay, min_threshold=thresh, signal_mode="max_in_window")
            m_snap = compute_metrics(r_snap)
            m_maxw = compute_metrics(r_maxw)
            if m_snap and m_maxw and (m_snap["trades"] >= 5 or m_maxw["trades"] >= 5):
                snap_str = f"{m_snap['trades']:2d}t {m_snap['win_rate']*100:.0f}%WR PnL={m_snap['total_pnl']:+.1f}"
                maxw_str = f"{m_maxw['trades']:2d}t {m_maxw['win_rate']*100:.0f}%WR PnL={m_maxw['total_pnl']:+.1f}"
                print(f"  {delay:2d}s/{thresh*100:.2f}%: snapshot=[{snap_str}]  max_window=[{maxw_str}]")

    # ================================================================
    # TOP COMBOS — ranked by total PnL, min 5 trades
    # ================================================================
    print("\n" + "=" * 70)
    print("TOP 20 CONFIGS BY TOTAL PnL (min 5 trades)")
    print("=" * 70)

    combos = []
    for mode in ["snapshot", "max_in_window"]:
        for delay in [3, 5, 7, 10, 15, 20, 30]:
            for thresh in [0.0, 0.00005, 0.0001, 0.00015, 0.0002, 0.0003, 0.0005]:
                results = run_backtest(
                    markets, signal_delay=delay, min_threshold=thresh, signal_mode=mode
                )
                m = compute_metrics(results)
                if m and m["trades"] >= 5:
                    combos.append((mode, delay, thresh, m))

    combos.sort(key=lambda x: x[3]["total_pnl"], reverse=True)
    print(f"\n  {'Mode':<12} {'Delay':>5} {'Thresh':>8} {'Trades':>6} {'WR':>5} {'PnL':>8} {'EV/t':>7} {'Sharpe':>7}")
    print(f"  {'-'*12} {'-'*5} {'-'*8} {'-'*6} {'-'*5} {'-'*8} {'-'*7} {'-'*7}")
    for mode, delay, thresh, m in combos[:20]:
        print(
            f"  {mode:<12} {delay:4d}s {thresh*100:7.3f}% {m['trades']:6d} "
            f"{m['win_rate']*100:4.0f}% {m['total_pnl']:+7.2f} {m['ev_per_trade']:+6.2f} "
            f"{m['sharpe']:6.2f}"
        )

    # ================================================================
    # DETAILED RESULTS — current config
    # ================================================================
    print("\n" + "=" * 70)
    print("DETAILED RESULTS — Current config (snapshot, delay=5s, threshold=0.02%)")
    print("=" * 70)

    results = run_backtest(markets, signal_delay=5, min_threshold=0.0002, verbose=True)
    print_metrics(compute_metrics(results), "Summary")

    # Breakdown by signal strength
    if results:
        print("\n--- Breakdown by signal magnitude ---")
        for cutoff in [0.02, 0.03, 0.05, 0.10]:
            subset = [r for r in results if abs(r.btc_return_pct) >= cutoff]
            if subset:
                m = compute_metrics(subset)
                print(
                    f"  |signal| >= {cutoff:.2f}%: {m['trades']} trades, "
                    f"{m['win_rate']*100:.0f}% WR, PnL={m['total_pnl']:+.2f}, "
                    f"EV/t={m['ev_per_trade']:+.2f}"
                )

    # ================================================================
    # MARKET-BY-MARKET: show what each signal mode would have done
    # ================================================================
    print("\n" + "=" * 70)
    print("ALL MARKETS — signal comparison (5s window, no threshold)")
    print("=" * 70)
    print(f"  {'slug':<35} {'snap_ret':>9} {'maxw_ret':>9} {'outcome':>7} {'snap':>5} {'maxw':>5}")

    for slug in sorted(markets.keys()):
        market = markets[slug]
        ticks = market.ticks
        if len(ticks) < 10 or max(t.elapsed_s for t in ticks) < 200:
            continue

        btc_ref = None
        for t in ticks:
            if t.btc_price is not None:
                btc_ref = t.btc_price
                break
        if btc_ref is None:
            continue

        outcome = determine_outcome(ticks)
        if outcome is None:
            outcome = determine_outcome_from_btc(ticks)
        if outcome is None:
            continue

        # Snapshot at 5s
        btc_5s = get_btc_at_time(ticks, 5.0)
        snap_ret = ((btc_5s - btc_ref) / btc_ref * 100) if btc_5s else None

        # Max in window [0, 5s]
        maxw = get_max_btc_return_in_window(ticks, btc_ref, 5.0)
        maxw_ret = maxw[1] * 100 if maxw else None

        snap_dir = "Up" if snap_ret and snap_ret > 0 else "Down" if snap_ret and snap_ret < 0 else "?"
        maxw_dir = "Up" if maxw_ret and maxw_ret > 0 else "Down" if maxw_ret and maxw_ret < 0 else "?"

        snap_ok = "Y" if snap_dir == outcome else "N" if snap_dir != "?" else "-"
        maxw_ok = "Y" if maxw_dir == outcome else "N" if maxw_dir != "?" else "-"

        snap_str = f"{snap_ret:+.4f}%" if snap_ret is not None else "     n/a"
        maxw_str = f"{maxw_ret:+.4f}%" if maxw_ret is not None else "     n/a"

        print(f"  {slug:<35} {snap_str:>9} {maxw_str:>9} {outcome:>7} {snap_ok:>5} {maxw_ok:>5}")


if __name__ == "__main__":
    main()
