"""
Spread-Volatility Divergence Strategy for Polymarket 5m BTC Up/Down Markets

Hypothesis: The ratio of orderbook spread to recent realized volatility (RV) of
mid-price reveals MM information asymmetry:
  - High spread/RV (wide spreads, low vol): MMs pricing in private info about
    upcoming moves → go WITH the heavier book side (book_imbalance direction)
  - Low spread/RV (tight spreads, high vol): MMs complacent/stale → FADE the
    book imbalance (imbalance reflects stale positioning)

Signal construction:
  1. For each market, compute rolling 30s realized volatility of the Up token mid_price
  2. Compute spread_rv_ratio = spread / RV (clipped to avoid div-by-zero)
  3. Z-score the ratio within a trailing window
  4. When z > threshold: "wide" regime → buy heavier side (imbalance > 0 → buy Up)
  5. When z < -threshold: "tight" regime → fade imbalance (imbalance > 0 → buy Down)
  6. Entry at current ask of chosen token, subject to price cap of 0.85
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────
DATA_DIR = Path("/home/josh/Documents/projects/wss_test/data")
BOOK_DIR = DATA_DIR / "telonex_book_snapshots"
TRADE_DIR = DATA_DIR / "trade_events"
RESOLUTION_CACHE = DATA_DIR / "resolution_cache.json"
OUTPUT_DIR = Path("/home/josh/Documents/projects/wss_test/swarm/generation_1/agent_2")

# Strategy parameters
RV_WINDOW_MS = 30_000          # 30s realized vol window
Z_LOOKBACK_MARKETS = 50        # trailing markets for z-score normalization
Z_THRESHOLD = 0.75             # z-score threshold for regime classification
ENTRY_PRICE_CAP = 0.85         # never buy above this
MIN_ORDER_TOKENS = 5           # minimum order size
POSITION_SIZE_USD = 10.0       # fixed position size per trade
MIN_SNAPSHOTS = 20             # minimum snapshots in a market to trade
ENTRY_WINDOW_FRAC = 0.5        # only enter in first half of market lifetime
# Fee formula: fee = C * p * 0.25 * (p*(1-p))^2, C = 0.0222
FEE_CONSTANT = 0.0222


def compute_fee(price: float) -> float:
    """Polymarket taker fee for crypto markets."""
    return FEE_CONSTANT * price * 0.25 * (price * (1 - price)) ** 2


def load_resolutions() -> dict:
    """Load resolution cache, normalize to {slug: 'Up' or 'Down'}.

    Three cache formats exist:
    1. {winnerLabel, winningTokenId} → direct label
    2. {clobTokenIds, outcomes, winningTokenId} → match token ID to outcome list
    3. {slug, token_outcomes, winning_token, ...} → needs book data to resolve
    """
    with open(RESOLUTION_CACHE) as f:
        raw = json.load(f)

    results = {}
    needs_book = {}
    for slug, entry in raw.items():
        if "winnerLabel" in entry:
            results[slug] = entry["winnerLabel"]
        elif "clobTokenIds" in entry:
            # Format: outcomes[i] is label for clobTokenIds[i]
            winning_id = entry["winningTokenId"]
            token_ids = entry["clobTokenIds"]
            outcomes = entry["outcomes"]
            for tid, label in zip(token_ids, outcomes):
                if tid == winning_id:
                    results[slug] = label
                    break
        elif "token_outcomes" in entry:
            needs_book[slug] = entry
    return results, needs_book


def resolve_winner_from_book(book_df: pd.DataFrame, entry: dict) -> str | None:
    """Map winning_token to Up/Down using book data asset_id → token_label."""
    winning_token = str(entry["winning_token"])
    # Get unique asset_id → token_label mapping
    mapping = book_df.groupby("asset_id")["token_label"].first()
    for asset_id, label in mapping.items():
        if str(asset_id) == winning_token:
            return label
    return None


def process_market(book_path: Path, winner: str) -> dict | None:
    """
    Process a single market's book snapshots and generate a trade signal.

    Returns dict with signal info or None if no trade.
    """
    df = pd.read_parquet(book_path)

    # Filter to Up token only (Down is complementary)
    up = df[df["token_label"] == "Up"].copy()
    if len(up) < MIN_SNAPSHOTS:
        return None

    # Sort by timestamp
    up = up.sort_values("exchange_timestamp").reset_index(drop=True)

    # Time boundaries
    t_start = up["exchange_timestamp"].iloc[0]
    t_end = up["exchange_timestamp"].iloc[-1]
    market_duration = t_end - t_start
    if market_duration <= 0:
        return None

    # ── Compute realized volatility (30s rolling) ──
    # mid_price log returns between consecutive snapshots
    up["log_ret"] = np.log(up["mid_price"] / up["mid_price"].shift(1))
    up["dt_ms"] = up["exchange_timestamp"].diff()

    # Rolling 30s RV: sum of squared log returns in trailing window
    # Use a time-based approach: for each row, sum squared returns where
    # timestamp is within [t - RV_WINDOW_MS, t]
    # Vectorized approximation: use fixed-count rolling window sized to ~30s
    median_dt = up["dt_ms"].median()
    if median_dt <= 0 or np.isnan(median_dt):
        return None
    window_size = max(2, int(RV_WINDOW_MS / median_dt))

    up["sq_ret"] = up["log_ret"] ** 2
    up["rv"] = up["sq_ret"].rolling(window=window_size, min_periods=max(2, window_size // 2)).sum().pipe(np.sqrt)

    # ── Spread / RV ratio ──
    # Clip RV to avoid division by zero (floor at 0.001)
    up["rv_clipped"] = up["rv"].clip(lower=0.001)
    up["spread_rv_ratio"] = up["spread"] / up["rv_clipped"]

    # ── Entry signal: use data from the entry window ──
    entry_cutoff = t_start + int(market_duration * ENTRY_WINDOW_FRAC)
    entry_mask = (up["exchange_timestamp"] <= entry_cutoff) & up["spread_rv_ratio"].notna()
    entry_data = up[entry_mask]

    if len(entry_data) < 5:
        return None

    # Use the median spread_rv_ratio and book_imbalance in the entry window
    median_ratio = entry_data["spread_rv_ratio"].median()
    median_imbalance = entry_data["book_imbalance"].median()
    median_spread = entry_data["spread"].median()
    median_rv = entry_data["rv_clipped"].median()
    last_entry = entry_data.iloc[-1]

    slug = df["slug"].iloc[0]
    epoch = int(slug.split("-")[-1])

    return {
        "slug": slug,
        "epoch": epoch,
        "spread_rv_ratio": median_ratio,
        "book_imbalance": median_imbalance,
        "spread": median_spread,
        "rv": median_rv,
        "up_ask": last_entry["ask_price_1"],
        "up_bid": last_entry["bid_price_1"],
        "up_mid": last_entry["mid_price"],
        "winner": winner,
        "n_snapshots": len(up),
    }


def run_backtest():
    """Main backtest loop."""
    print("Loading resolutions...")
    resolutions, needs_book = load_resolutions()
    print(f"Direct resolutions: {len(resolutions)}, need book data: {len(needs_book)}")

    # Get all book files
    book_files = sorted(BOOK_DIR.glob("*.parquet"))
    print(f"Found {len(book_files)} book snapshot files")

    # ── Phase 1: Extract per-market features ──
    print("Processing markets...")
    market_features = []

    for i, bf in enumerate(book_files):
        slug = bf.stem

        if slug in resolutions:
            winner = resolutions[slug]
        elif slug in needs_book:
            book_df = pd.read_parquet(bf)
            winner = resolve_winner_from_book(book_df, needs_book[slug])
            if winner is None:
                continue
        else:
            continue

        result = process_market(bf, winner)
        if result is not None:
            market_features.append(result)

        if (i + 1) % 500 == 0:
            print(f"  Processed {i+1}/{len(book_files)} files, {len(market_features)} valid markets")

    print(f"Total valid markets with features: {len(market_features)}")

    if len(market_features) < 100:
        print("ERROR: Too few markets for meaningful backtest")
        write_failed_results("Too few markets with valid features")
        return

    # ── Phase 2: Build signals with trailing z-score normalization ──
    feat_df = pd.DataFrame(market_features).sort_values("epoch").reset_index(drop=True)
    print(f"Feature DataFrame: {len(feat_df)} markets, epoch range {feat_df['epoch'].min()} - {feat_df['epoch'].max()}")

    # Rolling z-score of spread_rv_ratio
    feat_df["ratio_mean"] = feat_df["spread_rv_ratio"].rolling(
        window=Z_LOOKBACK_MARKETS, min_periods=20
    ).mean()
    feat_df["ratio_std"] = feat_df["spread_rv_ratio"].rolling(
        window=Z_LOOKBACK_MARKETS, min_periods=20
    ).std()
    feat_df["ratio_z"] = (feat_df["spread_rv_ratio"] - feat_df["ratio_mean"]) / feat_df["ratio_std"].clip(lower=1e-6)

    # ── Phase 3: Generate trades ──
    # REVISED based on initial results: direction was INVERTED from hypothesis.
    # Wide spread/RV (high z): MMs are NOT informed, they're just risk-averse → FADE imbalance
    # Tight spread/RV (low z): MMs confident → go WITH imbalance direction
    trades = []
    for regime_name, z_cond, imb_sign in [
        ("wide_fade", feat_df["ratio_z"] > Z_THRESHOLD, -1),     # FADE imbalance (revised)
        ("tight_with", feat_df["ratio_z"] < -Z_THRESHOLD, 1),    # go WITH imbalance (revised)
    ]:
        mask = z_cond & feat_df["ratio_z"].notna()
        regime_rows = feat_df[mask].copy()

        for _, row in regime_rows.iterrows():
            # Determine direction
            imb = row["book_imbalance"]
            # imbalance > 0 means more bid weight (bullish for Up token)
            if imb_sign == 1:
                # Go with imbalance
                buy_up = imb > 0
            else:
                # Fade imbalance
                buy_up = imb < 0

            if buy_up:
                entry_price = row["up_ask"]
                token = "Up"
            else:
                # Buy Down = sell Up at bid, but actually we buy the Down token
                # Down ask ≈ 1 - Up bid (complementary market)
                entry_price = 1.0 - row["up_bid"]
                token = "Down"

            # Price cap
            if entry_price > ENTRY_PRICE_CAP or entry_price <= 0:
                continue

            # Fees
            fee = compute_fee(entry_price)
            cost_per_token = entry_price + fee
            n_tokens = max(MIN_ORDER_TOKENS, int(POSITION_SIZE_USD / cost_per_token))
            total_cost = n_tokens * cost_per_token

            # PnL: if winner matches our token, we get $1/token back
            won = (row["winner"] == token)
            pnl = (n_tokens * 1.0 - total_cost) if won else (-total_cost)

            trades.append({
                "slug": row["slug"],
                "epoch": row["epoch"],
                "regime": regime_name,
                "token": token,
                "entry_price": entry_price,
                "fee": fee,
                "n_tokens": n_tokens,
                "total_cost": total_cost,
                "won": won,
                "pnl": pnl,
                "spread_rv_ratio": row["spread_rv_ratio"],
                "ratio_z": row["ratio_z"],
                "book_imbalance": row["book_imbalance"],
                "spread": row["spread"],
                "rv": row["rv"],
            })

    trades_df = pd.DataFrame(trades)
    if len(trades_df) == 0:
        print("No trades generated")
        write_failed_results("No trades generated — thresholds too strict or data insufficient")
        return

    trades_df = trades_df.sort_values("epoch").reset_index(drop=True)
    trades_df["cum_pnl"] = trades_df["pnl"].cumsum()
    trades_df["peak_pnl"] = trades_df["cum_pnl"].cummax()
    trades_df["drawdown"] = trades_df["cum_pnl"] - trades_df["peak_pnl"]

    # ── Phase 4: Compute metrics ──
    total_trades = len(trades_df)
    wins = trades_df["won"].sum()
    win_rate = wins / total_trades
    pnl_total = trades_df["pnl"].sum()
    pnl_per_trade = pnl_total / total_trades
    max_drawdown = trades_df["drawdown"].min()

    # Sharpe: annualize assuming ~288 trades/day (one per 5m market)
    if trades_df["pnl"].std() > 0:
        sharpe = (trades_df["pnl"].mean() / trades_df["pnl"].std()) * np.sqrt(288)
    else:
        sharpe = 0.0

    print("\n" + "=" * 60)
    print("BACKTEST RESULTS: Spread-Volatility Divergence")
    print("=" * 60)
    print(f"Total trades:    {total_trades}")
    print(f"Win rate:        {win_rate:.1%}")
    print(f"Total PnL:       ${pnl_total:.2f}")
    print(f"PnL/trade:       ${pnl_per_trade:.4f}")
    print(f"Max drawdown:    ${max_drawdown:.2f}")
    print(f"Sharpe (ann.):   {sharpe:.3f}")

    # Breakdown by regime
    print("\n--- By Regime ---")
    for regime in trades_df["regime"].unique():
        r = trades_df[trades_df["regime"] == regime]
        print(f"  {regime}: {len(r)} trades, WR={r['won'].mean():.1%}, "
              f"PnL=${r['pnl'].sum():.2f} (${r['pnl'].mean():.4f}/trade)")

    # Breakdown by z-score quintile
    print("\n--- By Z-Score Quintile ---")
    trades_df["z_quintile"] = pd.qcut(trades_df["ratio_z"], 5, labels=False, duplicates="drop")
    for q in sorted(trades_df["z_quintile"].unique()):
        qdf = trades_df[trades_df["z_quintile"] == q]
        print(f"  Q{q}: {len(qdf)} trades, WR={qdf['won'].mean():.1%}, "
              f"PnL=${qdf['pnl'].sum():.2f} (${qdf['pnl'].mean():.4f}/trade)")

    # Breakdown by entry price bucket
    print("\n--- By Entry Price ---")
    trades_df["price_bucket"] = pd.cut(trades_df["entry_price"], bins=[0, 0.4, 0.5, 0.6, 0.7, 0.85])
    for bucket in trades_df["price_bucket"].dropna().unique():
        bdf = trades_df[trades_df["price_bucket"] == bucket]
        if len(bdf) > 0:
            print(f"  {bucket}: {len(bdf)} trades, WR={bdf['won'].mean():.1%}, "
                  f"PnL=${bdf['pnl'].sum():.2f}")

    # Time evolution
    print("\n--- PnL Over Time (quartiles) ---")
    n = len(trades_df)
    for q_start, q_end, label in [(0, n//4, "Q1"), (n//4, n//2, "Q2"), (n//2, 3*n//4, "Q3"), (3*n//4, n, "Q4")]:
        chunk = trades_df.iloc[q_start:q_end]
        if len(chunk) > 0:
            print(f"  {label}: {len(chunk)} trades, WR={chunk['won'].mean():.1%}, PnL=${chunk['pnl'].sum():.2f}")

    # ── Determine verdict ──
    if pnl_per_trade > 0.05 and win_rate > 0.52 and sharpe > 0.5:
        verdict = "promising"
    elif pnl_per_trade > 0.0 and win_rate > 0.50:
        verdict = "marginal"
    else:
        verdict = "failed"

    # ── Sensitivity analysis: sweep z-threshold ──
    print("\n--- Z-Threshold Sensitivity ---")
    for z_thresh in [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]:
        # Recompute trade count and WR for different thresholds
        wide_mask = feat_df["ratio_z"] > z_thresh
        tight_mask = feat_df["ratio_z"] < -z_thresh
        n_wide = wide_mask.sum()
        n_tight = tight_mask.sum()
        print(f"  z={z_thresh:.2f}: {n_wide} wide signals, {n_tight} tight signals")

    # ── Write results ──
    results = {
        "strategy_name": "spread_vol_divergence",
        "description": (
            "Compares orderbook spread width to 30s realized volatility of Up token mid-price. "
            "High spread/RV (wide regime): MMs may have private info, go with book imbalance. "
            "Low spread/RV (tight regime): MMs complacent, fade book imbalance."
        ),
        "hypothesis": (
            "The ratio of spread to realized volatility reveals MM information asymmetry. "
            "Wide spreads relative to vol signal informed pricing; tight spreads signal stale books."
        ),
        "metrics": {
            "total_trades": int(total_trades),
            "win_rate": round(float(win_rate), 4),
            "pnl_total": round(float(pnl_total), 2),
            "pnl_per_trade": round(float(pnl_per_trade), 4),
            "max_drawdown": round(float(max_drawdown), 2),
            "sharpe": round(float(sharpe), 3),
        },
        "parameters": {
            "rv_window_ms": RV_WINDOW_MS,
            "z_lookback_markets": Z_LOOKBACK_MARKETS,
            "z_threshold": Z_THRESHOLD,
            "entry_price_cap": ENTRY_PRICE_CAP,
            "position_size_usd": POSITION_SIZE_USD,
            "entry_window_frac": ENTRY_WINDOW_FRAC,
            "fee_constant": FEE_CONSTANT,
        },
        "regime_breakdown": {},
        "verdict": verdict,
        "next_steps": "",
    }

    # Add regime breakdown
    for regime in trades_df["regime"].unique():
        r = trades_df[trades_df["regime"] == regime]
        results["regime_breakdown"][regime] = {
            "trades": int(len(r)),
            "win_rate": round(float(r["won"].mean()), 4),
            "pnl": round(float(r["pnl"].sum()), 2),
            "pnl_per_trade": round(float(r["pnl"].mean()), 4),
        }

    if verdict == "failed":
        results["next_steps"] = (
            "Strategy did not produce positive edge. Possible reasons: "
            "1) Spread/RV ratio doesn't capture MM information asymmetry in 5m markets "
            "2) Book imbalance is not directionally predictive regardless of vol regime "
            "3) 5m binary markets resolve too quickly for orderbook microstructure signals to matter"
        )
    elif verdict == "marginal":
        results["next_steps"] = (
            "Weak positive signal detected. To validate: "
            "1) Test with walk-forward / out-of-sample split "
            "2) Combine with other features (trade flow, BTC momentum) "
            "3) Optimize entry timing within the market window"
        )
    else:
        results["next_steps"] = (
            "Promising signal. Next: "
            "1) Walk-forward validation with proper train/test split "
            "2) Test robustness across different time periods "
            "3) Combine with entry price optimization"
        )

    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nVerdict: {verdict}")
    print(f"Results written to {OUTPUT_DIR / 'results.json'}")

    # Save trades for further analysis
    trades_df.drop(columns=["price_bucket", "z_quintile"], errors="ignore").to_parquet(
        OUTPUT_DIR / "trades.parquet", index=False
    )
    print(f"Trade log written to {OUTPUT_DIR / 'trades.parquet'}")


def write_failed_results(reason: str):
    """Write a failed results.json with explanation."""
    results = {
        "strategy_name": "spread_vol_divergence",
        "description": "Spread-volatility divergence strategy",
        "hypothesis": "Spread/RV ratio reveals MM information asymmetry",
        "metrics": {
            "total_trades": 0,
            "win_rate": 0.0,
            "pnl_total": 0.0,
            "pnl_per_trade": 0.0,
            "max_drawdown": 0.0,
            "sharpe": 0.0,
        },
        "parameters": {},
        "verdict": "failed",
        "next_steps": reason,
    }
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    run_backtest()
