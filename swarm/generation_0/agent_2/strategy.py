"""
Strategy: Early Liquidity Asymmetry Decay Rate

Hypothesis: In the first 30 seconds after a 5m BTC Up/Down market opens,
the bid-ask size asymmetry (top-3 bid size vs top-3 ask size on the Up token)
either normalizes quickly (noise) or persists/widens (informed flow).
Trade in the direction of persistent asymmetry.

- Heavy bids on Up token that persist → buy Up
- Heavy asks on Up token that persist → buy Down (via Up token being cheap)
"""

import pandas as pd
import numpy as np
import json
import glob
import os
import sys

# ─── Config ───────────────────────────────────────────────────────────────────
DATA_DIR = "data/telonex_book_snapshots"
RESOLUTION_CACHE = "data/resolution_cache.json"
OUTPUT_DIR = "swarm/generation_0/agent_2"

# Strategy parameters
WINDOW_MS = 10_000       # 10-second bins for measuring asymmetry
N_WINDOWS = 3            # First 3 windows (0-10s, 10-20s, 20-30s)
ENTRY_PRICE_CAP = 0.85   # Never buy above this
MIN_ORDER_SIZE = 5       # Minimum 5 tokens
PERSISTENCE_THRESHOLD = 0.15  # Min |log(ratio)| to consider asymmetric
DECAY_THRESHOLD = 0.5    # If asymmetry decays by >50%, it's noise

# Fee function: fee = C * p * 0.25 * (p*(1-p))^2
FEE_C = 0.0222


def compute_fee(price: np.ndarray) -> np.ndarray:
    """Vectorized Polymarket fee calculation."""
    return FEE_C * price * 0.25 * (price * (1 - price)) ** 2


def load_resolution_cache() -> dict:
    """Load resolution cache and build slug -> winning_label mapping."""
    with open(RESOLUTION_CACHE) as f:
        cache = json.load(f)

    slug_to_winner = {}
    for slug, info in cache.items():
        if not slug.startswith("btc-updown-5m-"):
            continue
        winning_id = info.get("winningTokenId", "")
        outcomes = info.get("outcomes", [])
        clob_ids = info.get("clobTokenIds", [])
        if winning_id and outcomes and clob_ids and winning_id in clob_ids:
            idx = clob_ids.index(winning_id)
            slug_to_winner[slug] = outcomes[idx]  # "Up" or "Down"
    return slug_to_winner


def process_single_market(filepath: str) -> dict | None:
    """
    Process a single market file. Returns dict with asymmetry metrics
    or None if insufficient data.
    """
    df = pd.read_parquet(filepath)
    slug = os.path.basename(filepath).replace(".parquet", "")

    # Extract market open time from slug
    try:
        market_open_ms = int(slug.split("-")[-1]) * 1000
    except (ValueError, IndexError):
        return None

    # Filter to Up token only (Down is complementary)
    up = df[df["token_label"] == "Up"].copy()
    if len(up) < 5:
        return None

    # Compute relative time from market open
    up_ts = up["exchange_timestamp"].values
    rel_time = up_ts - market_open_ms

    # Compute top-3 bid and ask sizes
    bid_top3 = (
        up["bid_size_1"].values + up["bid_size_2"].values + up["bid_size_3"].values
    )
    ask_top3 = (
        up["ask_size_1"].values + up["ask_size_2"].values + up["ask_size_3"].values
    )

    # Avoid division by zero
    valid = (bid_top3 > 0) & (ask_top3 > 0)

    # Compute asymmetry ratios per window
    window_ratios = []
    window_counts = []
    for w in range(N_WINDOWS):
        t_start = w * WINDOW_MS
        t_end = (w + 1) * WINDOW_MS
        mask = valid & (rel_time >= t_start) & (rel_time < t_end)
        n = mask.sum()
        if n < 2:
            window_ratios.append(np.nan)
            window_counts.append(n)
            continue
        # Log ratio: positive = bids heavy, negative = asks heavy
        log_ratios = np.log(bid_top3[mask] / ask_top3[mask])
        window_ratios.append(np.median(log_ratios))
        window_counts.append(n)

    if any(np.isnan(r) for r in window_ratios):
        return None

    # Get entry price (best ask in first window for Up, or best bid for Down)
    first_window = valid & (rel_time >= 0) & (rel_time < WINDOW_MS)
    if first_window.sum() == 0:
        return None
    entry_ask_up = np.median(up["ask_price_1"].values[first_window])
    entry_bid_up = np.median(up["bid_price_1"].values[first_window])

    return {
        "slug": slug,
        "window_ratios": window_ratios,  # log(bid/ask) per window
        "window_counts": window_counts,
        "entry_ask_up": entry_ask_up,
        "entry_bid_up": entry_bid_up,
        "mid_price": np.median(up["mid_price"].values[first_window]),
    }


def run_backtest():
    """Main backtest loop."""
    print("Loading resolution cache...")
    slug_to_winner = load_resolution_cache()
    print(f"  {len(slug_to_winner)} resolved markets")

    # Get all market files
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.parquet")))
    print(f"  {len(files)} orderbook files")

    # Process all markets
    print("Processing markets...")
    results = []
    skipped = 0
    for filepath in files:
        slug = os.path.basename(filepath).replace(".parquet", "")
        if slug not in slug_to_winner:
            skipped += 1
            continue
        market_data = process_single_market(filepath)
        if market_data is not None:
            market_data["winner"] = slug_to_winner[slug]
            results.append(market_data)

    print(f"  {len(results)} markets processed, {skipped} skipped (no resolution)")

    # Convert to DataFrame for vectorized analysis
    df = pd.DataFrame(results)
    df["r0"] = df["window_ratios"].apply(lambda x: x[0])
    df["r1"] = df["window_ratios"].apply(lambda x: x[1])
    df["r2"] = df["window_ratios"].apply(lambda x: x[2])

    # ─── Signal Construction ──────────────────────────────────────────────
    # Measure asymmetry persistence:
    # - Initial asymmetry magnitude (|r0|)
    # - Decay rate: how much asymmetry changes from window 0 to window 2
    # - Direction consistency: do all windows agree on direction?

    df["abs_r0"] = np.abs(df["r0"])
    df["abs_r2"] = np.abs(df["r2"])

    # Direction: sign of initial asymmetry
    df["direction"] = np.sign(df["r0"])

    # Persistence: ratio of final to initial asymmetry (same-sign)
    # >1 means widening, <1 means decaying, negative means flipped
    df["persistence"] = np.where(
        df["abs_r0"] > 0.01,
        df["r2"] / df["r0"],  # preserves sign flip detection
        0.0,
    )

    # All 3 windows agree on direction
    df["consistent_direction"] = (np.sign(df["r0"]) == np.sign(df["r1"])) & (
        np.sign(df["r1"]) == np.sign(df["r2"])
    )

    # ─── Trading Signal ───────────────────────────────────────────────────
    # Trade when:
    # 1. Initial asymmetry is meaningful (|r0| > threshold)
    # 2. Asymmetry persists or widens (persistence > decay_threshold)
    # 3. Direction is consistent across all windows

    df["has_signal"] = (
        (df["abs_r0"] >= PERSISTENCE_THRESHOLD)
        & (df["persistence"] >= DECAY_THRESHOLD)
        & df["consistent_direction"]
    )

    # Trade direction: positive r0 = bids heavy on Up token = buy Up
    #                  negative r0 = asks heavy on Up token = buy Down
    df["trade_up"] = df["direction"] > 0  # True = buy Up, False = buy Down

    # Entry price: if buying Up, use ask_price; if buying Down, use (1 - bid_up)
    df["entry_price"] = np.where(
        df["trade_up"], df["entry_ask_up"], 1 - df["entry_bid_up"]
    )

    # Apply entry price cap
    df["tradeable"] = df["has_signal"] & (df["entry_price"] <= ENTRY_PRICE_CAP)

    # ─── P&L Calculation ─────────────────────────────────────────────────
    trades = df[df["tradeable"]].copy()
    print(f"\n{'='*60}")
    print(f"SIGNAL ANALYSIS")
    print(f"{'='*60}")
    print(f"Markets with signal: {df['has_signal'].sum()} / {len(df)}")
    print(f"Tradeable (after price cap): {len(trades)} / {len(df)}")

    if len(trades) == 0:
        print("No trades generated!")
        return df, trades

    # Determine if trade won
    trades["trade_label"] = np.where(trades["trade_up"], "Up", "Down")
    trades["won"] = trades["trade_label"] == trades["winner"]

    # P&L per trade (1 token each for simplicity, can scale later)
    # Win: payout 1.0 - entry_price - fee
    # Lose: -entry_price - fee
    fees = compute_fee(trades["entry_price"].values)
    trades["fee"] = fees
    trades["pnl"] = np.where(
        trades["won"],
        1.0 - trades["entry_price"] - fees,
        -trades["entry_price"] - fees,
    )

    # Scale to minimum order size
    trades["pnl_scaled"] = trades["pnl"] * MIN_ORDER_SIZE

    # ─── Results ──────────────────────────────────────────────────────────
    total_trades = len(trades)
    wins = trades["won"].sum()
    win_rate = wins / total_trades
    pnl_total = trades["pnl_scaled"].sum()
    pnl_per_trade = trades["pnl_scaled"].mean()

    # Drawdown
    cumulative = trades["pnl_scaled"].cumsum()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max).min()

    # Sharpe (annualized, assuming ~288 5m markets/day)
    daily_pnl = trades["pnl_scaled"].values
    sharpe = (
        np.sqrt(288) * daily_pnl.mean() / daily_pnl.std()
        if daily_pnl.std() > 0
        else 0.0
    )

    print(f"\n{'='*60}")
    print(f"BACKTEST RESULTS")
    print(f"{'='*60}")
    print(f"Total trades:    {total_trades}")
    print(f"Win rate:        {win_rate:.1%}")
    print(f"Total P&L:       ${pnl_total:.2f}")
    print(f"P&L per trade:   ${pnl_per_trade:.4f}")
    print(f"Max drawdown:    ${drawdown:.2f}")
    print(f"Sharpe ratio:    {sharpe:.3f}")
    print(f"Avg entry price: {trades['entry_price'].mean():.4f}")
    print(f"Avg fee:         {trades['fee'].mean():.6f}")

    # Direction breakdown
    print(f"\nDirection breakdown:")
    for direction in ["Up", "Down"]:
        mask = trades["trade_label"] == direction
        if mask.sum() > 0:
            d_trades = trades[mask]
            d_wr = d_trades["won"].mean()
            d_pnl = d_trades["pnl_scaled"].sum()
            print(f"  {direction}: {mask.sum()} trades, {d_wr:.1%} WR, ${d_pnl:.2f} PnL")

    # ─── Parameter sensitivity ────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"PARAMETER SENSITIVITY")
    print(f"{'='*60}")

    # Vary persistence threshold
    print("\nPersistence threshold sweep:")
    for pt in [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]:
        mask = (
            (df["abs_r0"] >= pt)
            & (df["persistence"] >= DECAY_THRESHOLD)
            & df["consistent_direction"]
            & (df["entry_price"] <= ENTRY_PRICE_CAP)
        )
        subset = df[mask]
        if len(subset) == 0:
            continue
        trade_label = np.where(subset["direction"] > 0, "Up", "Down")
        won = trade_label == subset["winner"]
        p = subset["entry_price"].values
        f = compute_fee(p)
        pnl = np.where(won, 1.0 - p - f, -p - f) * MIN_ORDER_SIZE
        wr = won.mean()
        print(
            f"  threshold={pt:.2f}: {len(subset):4d} trades, "
            f"WR={wr:.1%}, PnL=${pnl.sum():.2f} (${pnl.mean():.4f}/trade)"
        )

    # Vary decay threshold
    print("\nDecay threshold sweep:")
    for dt in [0.0, 0.3, 0.5, 0.7, 1.0, 1.5]:
        mask = (
            (df["abs_r0"] >= PERSISTENCE_THRESHOLD)
            & (df["persistence"] >= dt)
            & df["consistent_direction"]
            & (df["entry_price"] <= ENTRY_PRICE_CAP)
        )
        subset = df[mask]
        if len(subset) == 0:
            print(f"  decay_threshold={dt:.1f}: 0 trades")
            continue
        trade_label = np.where(subset["direction"] > 0, "Up", "Down")
        won = trade_label == subset["winner"]
        p = subset["entry_price"].values
        f = compute_fee(p)
        pnl = np.where(won, 1.0 - p - f, -p - f) * MIN_ORDER_SIZE
        wr = won.mean()
        print(
            f"  decay_threshold={dt:.1f}: {len(subset):4d} trades, "
            f"WR={wr:.1%}, PnL=${pnl.sum():.2f} (${pnl.mean():.4f}/trade)"
        )

    # ─── Baseline: random trading at same prices ──────────────────────────
    print(f"\n{'='*60}")
    print(f"BASELINE COMPARISON")
    print(f"{'='*60}")

    # What if we just traded every market at the same entry prices?
    all_tradeable = df[df["entry_price"] <= ENTRY_PRICE_CAP]
    if len(all_tradeable) > 0:
        # Random: 50% win rate expectation
        p_all = all_tradeable["entry_price"].values
        f_all = compute_fee(p_all)
        # Expected PnL at 50% WR
        expected_pnl_win = (1.0 - p_all - f_all) * MIN_ORDER_SIZE
        expected_pnl_lose = (-p_all - f_all) * MIN_ORDER_SIZE
        expected_per_trade = 0.5 * expected_pnl_win.mean() + 0.5 * expected_pnl_lose.mean()
        print(f"Random baseline (50% WR) at avg price {p_all.mean():.4f}:")
        print(f"  Expected P&L/trade: ${expected_per_trade:.4f}")
        print(f"  (Break-even WR at avg price: {p_all.mean()/(1.0):.1%})")

    return df, trades


def write_results(df, trades):
    """Write results.json."""
    if len(trades) == 0:
        metrics = {
            "total_trades": 0,
            "win_rate": 0.0,
            "pnl_total": 0.0,
            "pnl_per_trade": 0.0,
            "max_drawdown": 0.0,
            "sharpe": 0.0,
        }
        verdict = "failed"
    else:
        fees = compute_fee(trades["entry_price"].values)
        pnl = np.where(
            trades["won"],
            1.0 - trades["entry_price"] - fees,
            -trades["entry_price"] - fees,
        ) * MIN_ORDER_SIZE
        cumulative = np.cumsum(pnl)
        running_max = np.maximum.accumulate(cumulative)
        max_dd = float(np.min(cumulative - running_max))
        sharpe = (
            float(np.sqrt(288) * np.mean(pnl) / np.std(pnl))
            if np.std(pnl) > 0
            else 0.0
        )

        total_trades = len(trades)
        win_rate = float(trades["won"].mean())
        pnl_total = float(np.sum(pnl))
        pnl_per_trade = float(np.mean(pnl))

        metrics = {
            "total_trades": total_trades,
            "win_rate": round(win_rate, 4),
            "pnl_total": round(pnl_total, 2),
            "pnl_per_trade": round(pnl_per_trade, 4),
            "max_drawdown": round(max_dd, 2),
            "sharpe": round(sharpe, 3),
        }

        if pnl_per_trade > 0.02 and win_rate > 0.52 and total_trades >= 50:
            verdict = "promising"
        elif pnl_per_trade > 0 and win_rate > 0.50:
            verdict = "marginal"
        else:
            verdict = "failed"

    results = {
        "strategy_name": "early_liquidity_asymmetry_decay",
        "description": (
            "Measures bid-ask size asymmetry decay rate in first 30s of 5m BTC markets. "
            "Trades in direction of persistent asymmetry (non-decaying imbalance = informed flow)."
        ),
        "hypothesis": (
            "Markets where top-3 bid/ask size asymmetry persists or widens across "
            "the first 3 ten-second windows reflect informed flow rather than noise. "
            "Trading in the direction of persistent asymmetry captures this edge."
        ),
        "metrics": metrics,
        "parameters": {
            "window_ms": WINDOW_MS,
            "n_windows": N_WINDOWS,
            "persistence_threshold": PERSISTENCE_THRESHOLD,
            "decay_threshold": DECAY_THRESHOLD,
            "entry_price_cap": ENTRY_PRICE_CAP,
            "min_order_size": MIN_ORDER_SIZE,
            "fee_constant": FEE_C,
        },
        "verdict": verdict,
        "next_steps": "",
    }

    # Fill next_steps based on verdict
    if verdict == "promising":
        results["next_steps"] = (
            "Walk-forward validation on held-out data. "
            "Test with realistic slippage and queue position modeling. "
            "Consider combining with mid-price momentum for confirmation."
        )
    elif verdict == "marginal":
        results["next_steps"] = (
            "Investigate whether combining with other features (spread, trade flow) "
            "improves edge. Check if signal decays over time (structural alpha erosion). "
            "May not survive live execution costs."
        )
    else:
        results["next_steps"] = (
            "Asymmetry decay rate does not reliably predict 5m market outcomes. "
            "Consider: (1) the signal may be too noisy at 5m timescale, "
            "(2) market makers may be symmetric by design, "
            "(3) informed flow may not manifest in top-of-book sizes."
        )

    output_path = os.path.join(OUTPUT_DIR, "results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {output_path}")
    print(f"Verdict: {verdict}")


if __name__ == "__main__":
    os.chdir("/home/josh/Documents/projects/wss_test")
    df, trades = run_backtest()
    write_results(df, trades)
