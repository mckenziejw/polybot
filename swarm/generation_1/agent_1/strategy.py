"""
Cross-Timeframe Liquidity Regime Transfer Strategy
====================================================
Hypothesis: The 4h BTC Up/Down market's orderbook state (25-75% through)
reveals a "regime" that persists into subsequent 5m markets.

- Consensus regime (tight spread, balanced depth, mid near 0.50):
  → Fade extreme 5m opening prices (mean-reversion)
- Discovery regime (wide spread, skewed depth, mid far from 0.50):
  → Follow early directional signal in 5m markets (momentum)

Only trades 5m markets that start AFTER a 4h market ends (causal ordering).
Uses ~138 overlapping 4h markets (Feb 12 – Mar 6, 2026).
"""

import json
import glob
import re
import numpy as np
import pandas as pd
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
BASE = Path("/home/josh/Documents/projects/wss_test")
DIR_4H = BASE / "data" / "telonex_btc_4h_book_snapshots"
DIR_5M = BASE / "data" / "telonex_book_snapshots"
RESOLUTION_CACHE = BASE / "data" / "resolution_cache.json"
OUTPUT_DIR = BASE / "swarm" / "generation_1" / "agent_1"

# ── Constants ──────────────────────────────────────────────────────────
MARKET_4H_DURATION = 14400  # seconds
MARKET_5M_DURATION = 300    # seconds
ENTRY_PRICE_CAP = 0.85
POSITION_SIZE = 10  # tokens per trade


def polymarket_fee(price: np.ndarray | float) -> np.ndarray | float:
    """Polymarket crypto taker fee (vectorized)."""
    return 0.0222 * price * 0.25 * (price * (1 - price)) ** 2


def extract_epoch(filename: str) -> int:
    return int(re.search(r"(\d+)\.parquet$", filename).group(1))


# ══════════════════════════════════════════════════════════════════════
# PHASE 1: Pre-load all data into memory
# ══════════════════════════════════════════════════════════════════════

def load_all_4h_features() -> pd.DataFrame:
    """Load all 4h markets in overlap period, extract regime features."""
    files_4h = sorted(glob.glob(str(DIR_4H / "*.parquet")))
    feb12_epoch = 1770854400
    files_4h = [f for f in files_4h if extract_epoch(f) >= feb12_epoch]
    print(f"  Loading {len(files_4h)} 4h files...")

    records = []
    for f in files_4h:
        df = pd.read_parquet(f)
        slug = df["slug"].iloc[0]
        start_epoch = int(re.search(r"(\d+)$", slug).group(1))

        up = df[df["token_label"] == "Up"]
        if len(up) < 10:
            continue

        # Vectorized: compute pct through market
        pct = (up["exchange_timestamp"].values / 1000 - start_epoch) / MARKET_4H_DURATION

        # Window: 25-75% of market
        mask = (pct >= 0.25) & (pct <= 0.75)
        window = up[mask]
        if len(window) < 5:
            continue

        # Aggregate features
        records.append({
            "slug_4h": slug,
            "start_epoch_4h": start_epoch,
            "end_epoch_4h": start_epoch + MARKET_4H_DURATION,
            "median_spread": window["spread"].median(),
            "median_imbalance": window["book_imbalance"].median(),
            "median_mid": window["mid_price"].median(),
            "spread_std": window["spread"].std(),
            "imbalance_std": window["book_imbalance"].std(),
            "total_depth": (window["bid_size_1"] + window["ask_size_1"]).median(),
            "mid_distance_from_half": abs(window["mid_price"].median() - 0.5),
        })

    result = pd.DataFrame(records)
    print(f"  {len(result)} 4h markets with valid features")
    return result


def load_all_5m_signals() -> pd.DataFrame:
    """Load all 5m markets, extract early opening signals + resolution."""
    with open(RESOLUTION_CACHE) as f:
        rc = json.load(f)

    files_5m = sorted(glob.glob(str(DIR_5M / "*.parquet")))
    print(f"  Loading {len(files_5m)} 5m files...")

    records = []
    for i, fpath in enumerate(files_5m):
        if i % 500 == 0:
            print(f"    ...{i}/{len(files_5m)}")

        df = pd.read_parquet(fpath)
        slug = df["slug"].iloc[0]
        start_epoch = int(re.search(r"(\d+)$", slug).group(1))

        up = df[df["token_label"] == "Up"]
        down = df[df["token_label"] == "Down"]
        if len(up) < 3 or len(down) < 3:
            continue

        # First 20% of snapshots by time
        pct_up = (up["exchange_timestamp"].values / 1000 - start_epoch) / MARKET_5M_DURATION
        early_up = up[pct_up <= 0.20]
        pct_down = (down["exchange_timestamp"].values / 1000 - start_epoch) / MARKET_5M_DURATION
        early_down = down[pct_down <= 0.20]

        if len(early_up) < 2 or len(early_down) < 2:
            continue

        # Opening features
        opening_mid_up = early_up["mid_price"].iloc[0]
        opening_imbalance = early_up["book_imbalance"].median()
        entry_price_up = early_up["ask_price_1"].median()
        entry_price_down = early_down["ask_price_1"].median()

        # Resolve winner
        if slug not in rc:
            continue
        entry = rc[slug]
        winning_token = entry.get("winning_token")
        if winning_token is None:
            continue

        up_asset_id = up["asset_id"].iloc[0]
        down_asset_id = down["asset_id"].iloc[0]

        if winning_token == up_asset_id:
            winner = "Up"
        elif winning_token == down_asset_id:
            winner = "Down"
        else:
            continue

        records.append({
            "slug_5m": slug,
            "start_epoch_5m": start_epoch,
            "opening_mid_up": opening_mid_up,
            "opening_imbalance": opening_imbalance,
            "opening_spread": early_up["spread"].iloc[0],
            "entry_price_up": entry_price_up,
            "entry_price_down": entry_price_down,
            "winner": winner,
        })

    result = pd.DataFrame(records)
    print(f"  {len(result)} 5m markets with valid signals + resolution")
    return result


# ══════════════════════════════════════════════════════════════════════
# PHASE 2: Vectorized trade generation + parameter sweep
# ══════════════════════════════════════════════════════════════════════

def link_4h_to_5m(df_4h: pd.DataFrame, df_5m: pd.DataFrame) -> pd.DataFrame:
    """
    For each 5m market, find which 4h market's regime applies.
    A 5m market uses the regime of the PREVIOUS 4h market (causal).
    """
    # Sort both
    df_4h = df_4h.sort_values("end_epoch_4h").reset_index(drop=True)
    df_5m = df_5m.sort_values("start_epoch_5m").reset_index(drop=True)

    # For each 5m start_epoch, find the most recent 4h end_epoch that is <= it
    # Use numpy searchsorted for vectorized lookup
    ends_4h = df_4h["end_epoch_4h"].values
    starts_5m = df_5m["start_epoch_5m"].values

    # searchsorted: find insertion point, then go back one to get the most recent 4h that ended
    idx = np.searchsorted(ends_4h, starts_5m, side="right") - 1

    # Filter: only keep 5m markets that have a valid prior 4h market
    # AND that start within the next 4h window after the prior 4h ended
    valid_mask = idx >= 0
    linked = df_5m[valid_mask].copy()
    valid_idx = idx[valid_mask]

    # Attach 4h features
    for col in df_4h.columns:
        linked[col] = df_4h[col].values[valid_idx]

    # Filter: 5m must start within MARKET_4H_DURATION of the 4h end
    linked = linked[
        linked["start_epoch_5m"] < linked["end_epoch_4h"] + MARKET_4H_DURATION
    ].copy()

    print(f"  {len(linked)} 5m markets linked to prior 4h regime")
    return linked


def generate_trades(
    linked: pd.DataFrame,
    spread_threshold: float,
    mid_threshold: float,
    fade_entry_threshold: float,
    follow_imbalance_threshold: float,
) -> pd.DataFrame:
    """Vectorized trade generation from linked data."""
    df = linked.copy()

    # Classify regime
    is_tight = df["median_spread"] <= spread_threshold
    is_centered = df["mid_distance_from_half"] <= mid_threshold
    df["regime"] = np.where(is_tight & is_centered, "consensus", "discovery")

    # ── Consensus trades: fade extremes ──
    cons = df["regime"] == "consensus"
    fade_down = cons & (df["opening_mid_up"] > fade_entry_threshold)
    fade_up = cons & (df["opening_mid_up"] < (1 - fade_entry_threshold))

    # ── Discovery trades: follow imbalance ──
    disc = df["regime"] == "discovery"
    follow_up = disc & (df["opening_imbalance"] > follow_imbalance_threshold)
    follow_down = disc & (df["opening_imbalance"] < -follow_imbalance_threshold)

    # Assign side and entry price
    df["side"] = None
    df["entry_price"] = np.nan

    df.loc[fade_down, "side"] = "Down"
    df.loc[fade_down, "entry_price"] = df.loc[fade_down, "entry_price_down"]

    df.loc[fade_up, "side"] = "Up"
    df.loc[fade_up, "entry_price"] = df.loc[fade_up, "entry_price_up"]

    df.loc[follow_up, "side"] = "Up"
    df.loc[follow_up, "entry_price"] = df.loc[follow_up, "entry_price_up"]

    df.loc[follow_down, "side"] = "Down"
    df.loc[follow_down, "entry_price"] = df.loc[follow_down, "entry_price_down"]

    # Filter to actual trades
    trades = df[df["side"].notna()].copy()
    trades = trades[
        (trades["entry_price"] <= ENTRY_PRICE_CAP)
        & (trades["entry_price"] > 0)
        & trades["entry_price"].notna()
    ].copy()

    if len(trades) == 0:
        return trades

    # Vectorized PnL
    trades["fee"] = polymarket_fee(trades["entry_price"].values)
    trades["cost"] = trades["entry_price"] + trades["fee"]
    trades["won"] = trades["side"] == trades["winner"]
    trades["payout"] = np.where(trades["won"], 1.0, 0.0)
    trades["pnl_per_token"] = trades["payout"] - trades["cost"]
    trades["total_pnl"] = trades["pnl_per_token"] * POSITION_SIZE

    return trades


def compute_metrics(trades: pd.DataFrame) -> dict:
    if len(trades) == 0:
        return {
            "total_trades": 0, "win_rate": 0.0, "pnl_total": 0.0,
            "pnl_per_trade": 0.0, "max_drawdown": 0.0, "sharpe": 0.0,
        }

    cum_pnl = trades["total_pnl"].cumsum()
    max_dd = (cum_pnl - cum_pnl.cummax()).min()

    pnl_mean = trades["total_pnl"].mean()
    pnl_std = trades["total_pnl"].std()
    sharpe = (pnl_mean / pnl_std * np.sqrt(len(trades))) if pnl_std > 0 else 0.0

    return {
        "total_trades": int(len(trades)),
        "win_rate": float(trades["won"].mean()),
        "pnl_total": float(trades["total_pnl"].sum()),
        "pnl_per_trade": float(pnl_mean),
        "max_drawdown": float(max_dd),
        "sharpe": float(sharpe),
    }


def print_breakdown(trades: pd.DataFrame) -> dict:
    print("\n" + "=" * 60)
    print("OVERALL RESULTS")
    print("=" * 60)
    metrics = compute_metrics(trades)
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    if len(trades) == 0:
        return metrics

    for regime in ["consensus", "discovery"]:
        sub = trades[trades["regime"] == regime]
        if len(sub) == 0:
            continue
        print(f"\n--- {regime.upper()} ({len(sub)} trades) ---")
        m = compute_metrics(sub)
        for k, v in m.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    for side in ["Up", "Down"]:
        sub = trades[trades["side"] == side]
        if len(sub) == 0:
            continue
        print(f"\n--- Side={side} ({len(sub)} trades) ---")
        print(f"  Win rate: {sub['won'].mean():.4f}, PnL: ${sub['total_pnl'].sum():.2f}")

    print(f"\n--- Entry price stats ---")
    print(f"  Mean: {trades['entry_price'].mean():.4f}, Median: {trades['entry_price'].median():.4f}")
    print(f"  Range: {trades['entry_price'].min():.4f} – {trades['entry_price'].max():.4f}")

    return metrics


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Cross-Timeframe Liquidity Regime Transfer Strategy")
    print("=" * 60)

    # ── Phase 1: Load all data ──
    print("\n[Phase 1] Loading data...")
    df_4h = load_all_4h_features()
    df_5m = load_all_5m_signals()

    # ── Link 4h → 5m ──
    print("\n[Phase 2] Linking 4h regimes to 5m markets...")
    linked = link_4h_to_5m(df_4h, df_5m)

    # ── Phase 2: Default parameters ──
    print("\n[Phase 3] Running default backtest...")
    default_params = {
        "spread_threshold": 0.04,
        "mid_threshold": 0.15,
        "fade_entry_threshold": 0.58,
        "follow_imbalance_threshold": 0.15,
    }
    trades = generate_trades(linked, **default_params)
    metrics = print_breakdown(trades)

    if len(trades) > 0:
        trades.to_csv(OUTPUT_DIR / "trades.csv", index=False)

    # ── Phase 3: Parameter sweep (all in-memory, fast) ──
    print("\n" + "=" * 60)
    print("PARAMETER SENSITIVITY ANALYSIS")
    print("=" * 60)

    sweep_results = []
    for spread_t in [0.02, 0.03, 0.04, 0.05, 0.06, 0.08]:
        for mid_t in [0.08, 0.10, 0.15, 0.20, 0.25]:
            for fade_t in [0.53, 0.55, 0.58, 0.60, 0.65]:
                for follow_t in [0.05, 0.10, 0.15, 0.20, 0.30]:
                    t = generate_trades(linked, spread_t, mid_t, fade_t, follow_t)
                    m = compute_metrics(t)
                    if m["total_trades"] >= 10:
                        sweep_results.append({
                            "spread_t": spread_t, "mid_t": mid_t,
                            "fade_t": fade_t, "follow_t": follow_t, **m,
                        })

    sweep_df = pd.DataFrame(sweep_results) if sweep_results else pd.DataFrame()

    if len(sweep_df) > 0:
        sweep_df = sweep_df.sort_values("pnl_total", ascending=False)
        print(f"\nTotal combos with >=10 trades: {len(sweep_df)}")
        pct_prof = (sweep_df["pnl_total"] > 0).mean()
        print(f"Profitable combos: {(sweep_df['pnl_total'] > 0).sum()} ({pct_prof:.1%})")
        print(f"PnL range: ${sweep_df['pnl_total'].min():.2f} to ${sweep_df['pnl_total'].max():.2f}")
        print(f"Win rate range: {sweep_df['win_rate'].min():.3f} to {sweep_df['win_rate'].max():.3f}")

        print("\nTop 5 by PnL:")
        cols = ["spread_t", "mid_t", "fade_t", "follow_t", "total_trades", "win_rate", "pnl_total", "pnl_per_trade", "sharpe"]
        print(sweep_df[cols].head(5).to_string(index=False))

        print("\nBottom 5 by PnL:")
        print(sweep_df[cols].tail(5).to_string(index=False))

        # Regime breakdown for best params
        best = sweep_df.iloc[0]
        print(f"\n--- Best params breakdown ---")
        best_trades = generate_trades(
            linked, best["spread_t"], best["mid_t"], best["fade_t"], best["follow_t"]
        )
        for regime in ["consensus", "discovery"]:
            sub = best_trades[best_trades["regime"] == regime]
            if len(sub) > 0:
                print(f"  {regime}: {len(sub)} trades, WR={sub['won'].mean():.3f}, PnL=${sub['total_pnl'].sum():.2f}")
    else:
        print("No parameter combinations produced >= 10 trades")
        pct_prof = 0.0

    # ── Also check: does regime classification matter at all? ──
    print("\n" + "=" * 60)
    print("REGIME VALIDITY CHECK")
    print("=" * 60)
    print("4h feature distributions:")
    for col in ["median_spread", "median_mid", "mid_distance_from_half", "median_imbalance"]:
        vals = linked[col]
        print(f"  {col}: median={vals.median():.4f}, mean={vals.mean():.4f}, std={vals.std():.4f}")

    # Check if 5m win rate differs by 4h regime at all (regime-agnostic)
    for threshold in [0.04, 0.06, 0.08]:
        tight = linked[linked["median_spread"] <= threshold]
        wide = linked[linked["median_spread"] > threshold]
        # For tight-spread 4h: what's the 5m base rate of Up winning?
        if len(tight) > 0 and len(wide) > 0:
            up_rate_tight = (tight["winner"] == "Up").mean()
            up_rate_wide = (wide["winner"] == "Up").mean()
            print(f"\n  Spread threshold {threshold}:")
            print(f"    Tight ({len(tight)}): Up wins {up_rate_tight:.3f}")
            print(f"    Wide  ({len(wide)}): Up wins {up_rate_wide:.3f}")

    # ── Determine verdict ──
    if len(trades) == 0:
        verdict = "failed"
        description = "No trades generated with default parameters"
    elif metrics["pnl_total"] > 0 and metrics["win_rate"] > 0.52:
        if len(sweep_df) > 0 and pct_prof > 0.6:
            verdict = "promising"
        else:
            verdict = "marginal"
        description = (
            f"{metrics['total_trades']} trades, {metrics['win_rate']:.1%} WR, "
            f"${metrics['pnl_total']:.2f} PnL"
        )
    else:
        verdict = "failed"
        description = (
            f"Unprofitable: {metrics['total_trades']} trades, "
            f"{metrics['win_rate']:.1%} WR, ${metrics['pnl_total']:.2f} PnL"
        )

    best_params_sweep = {}
    if len(sweep_df) > 0:
        b = sweep_df.iloc[0]
        best_params_sweep = {
            "spread_threshold": float(b["spread_t"]),
            "mid_threshold": float(b["mid_t"]),
            "fade_entry_threshold": float(b["fade_t"]),
            "follow_imbalance_threshold": float(b["follow_t"]),
        }

    # ── Write results.json ──
    if verdict == "promising":
        next_steps = (
            "Walk-forward validation on out-of-sample 5m data. "
            "Test with realistic execution delays. "
            "Check regime stability across market conditions."
        )
    elif verdict == "marginal":
        next_steps = (
            "Signal may exist but too weak for fees. "
            "Try combining with BTC momentum or other features. "
            "Check if specific regime transitions have stronger signal."
        )
    else:
        next_steps = (
            "4h regime does not transfer meaningfully to 5m markets. "
            "5m microstructure appears independent of 4h timeframe. "
            "Consider within-5m features instead of cross-timeframe transfer."
        )

    results = {
        "strategy_name": "cross_timeframe_liquidity_regime",
        "description": description,
        "hypothesis": (
            "The 4h market's orderbook regime (consensus vs discovery) persists into "
            "subsequent 5m markets. Consensus → fade extreme openings; Discovery → follow "
            "early directional signals."
        ),
        "metrics": metrics,
        "parameters": {
            "default": default_params,
            "best_sweep": best_params_sweep,
        },
        "sweep_summary": {
            "total_combos": len(sweep_df),
            "pct_profitable": float(pct_prof),
        },
        "verdict": verdict,
        "next_steps": next_steps,
    }

    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"VERDICT: {verdict.upper()}")
    print(f"Results → {OUTPUT_DIR / 'results.json'}")
