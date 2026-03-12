"""
Cross-Asset Book Imbalance Divergence Strategy

Hypothesis: When BTC and ETH 5m orderbook imbalances agree in direction but SOL or XRP
imbalance diverges (opposite sign), bet that the BTC/ETH consensus is correct. The divergent
alt imbalance reflects stale/noise liquidity rather than genuine contrarian information.

We trade BTC up/down prediction markets based on the BTC/ETH consensus direction when
alt divergence is detected.
"""

import glob
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE = "/home/ubuntu/wss_test"
DATA = f"{BASE}/data"
OUTPUT = f"{BASE}/swarm/generation_10/agent_0"


def polymarket_fee(p: np.ndarray) -> np.ndarray:
    """Vectorized Polymarket fee calculation."""
    return 0.0222 * p * 0.25 * (p * (1 - p)) ** 2


def load_resolution_cache() -> dict:
    with open(f"{DATA}/resolution_cache.json") as f:
        return json.load(f)


def extract_epoch_from_slug(slug: str) -> int:
    return int(slug.rsplit("-", 1)[1])


def load_imbalance_for_asset(asset: str, epochs: set) -> pd.DataFrame:
    """Load book imbalance data for an asset, filtering to relevant epochs.

    Returns DataFrame with columns: [epoch, token_label, book_imbalance, mid_price]
    Aggregated per market (epoch) and token_label using the median imbalance.
    Memory-efficient: processes in small batches, aggregating immediately.
    """
    pattern = f"{DATA}/telonex_{asset}_5m_book_snapshots/*.parquet"
    files = glob.glob(pattern)

    # Filter files to only those matching target epochs
    relevant_files = []
    file_epochs = []
    for f in files:
        epoch = int(os.path.basename(f).replace(".parquet", "").rsplit("-", 1)[1])
        if epoch in epochs:
            relevant_files.append(f)
            file_epochs.append(epoch)

    if not relevant_files:
        return pd.DataFrame()

    # Process in small batches, aggregate each batch immediately to save memory
    batch_size = 200
    agg_parts = []
    cols = ["token_label", "book_imbalance", "mid_price"]

    for i in range(0, len(relevant_files), batch_size):
        batch_files = relevant_files[i : i + batch_size]
        batch_epochs = file_epochs[i : i + batch_size]

        dfs = []
        for f, ep in zip(batch_files, batch_epochs):
            try:
                small = pd.read_parquet(f, columns=cols)
                small["epoch"] = ep
                dfs.append(small)
            except Exception:
                continue

        if not dfs:
            continue

        df = pd.concat(dfs, ignore_index=True)
        agg = (
            df.groupby(["epoch", "token_label"])
            .agg(
                book_imbalance=("book_imbalance", "median"),
                mid_price=("mid_price", "median"),
            )
            .reset_index()
        )
        agg_parts.append(agg)
        del df, dfs

    if not agg_parts:
        return pd.DataFrame()

    return pd.concat(agg_parts, ignore_index=True)


def build_cross_asset_signals(epochs: set) -> pd.DataFrame:
    """Build a DataFrame with imbalance signals across all 4 assets for matching epochs."""

    print("Loading BTC imbalances...")
    btc = load_imbalance_for_asset("btc", epochs)
    print(f"  BTC: {len(btc)} rows, {btc['epoch'].nunique()} epochs")

    print("Loading ETH imbalances...")
    eth = load_imbalance_for_asset("eth", epochs)
    print(f"  ETH: {len(eth)} rows, {eth['epoch'].nunique()} epochs")

    print("Loading SOL imbalances...")
    sol = load_imbalance_for_asset("sol", epochs)
    print(f"  SOL: {len(sol)} rows, {sol['epoch'].nunique()} epochs")

    print("Loading XRP imbalances...")
    xrp = load_imbalance_for_asset("xrp", epochs)
    print(f"  XRP: {len(xrp)} rows, {xrp['epoch'].nunique()} epochs")

    # Focus on "Up" token imbalance (Down is just the complement)
    btc_up = btc[btc["token_label"] == "Up"][["epoch", "book_imbalance", "mid_price"]].rename(
        columns={"book_imbalance": "btc_imb", "mid_price": "btc_mid"}
    )
    eth_up = eth[eth["token_label"] == "Up"][["epoch", "book_imbalance"]].rename(
        columns={"book_imbalance": "eth_imb"}
    )
    sol_up = sol[sol["token_label"] == "Up"][["epoch", "book_imbalance"]].rename(
        columns={"book_imbalance": "sol_imb"}
    )
    xrp_up = xrp[xrp["token_label"] == "Up"][["epoch", "book_imbalance"]].rename(
        columns={"book_imbalance": "xrp_imb"}
    )

    # Merge all on epoch
    merged = btc_up.merge(eth_up, on="epoch", how="inner")
    merged = merged.merge(sol_up, on="epoch", how="inner")
    merged = merged.merge(xrp_up, on="epoch", how="inner")

    print(f"\nMerged dataset: {len(merged)} epochs with all 4 assets")
    return merged


def generate_signals(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Generate trading signals based on cross-asset imbalance divergence.

    Signal logic:
    1. BTC and ETH imbalances must agree in sign (consensus direction)
    2. At least one of SOL/XRP must diverge (opposite sign from consensus)
    3. The divergence magnitude must exceed a threshold
    4. BTC mid_price must be away from 0.50 (for payoff) but below 0.85 (cap)
    """
    imb_threshold = params["imb_threshold"]
    divergence_threshold = params["divergence_threshold"]
    min_mid = params["min_mid"]
    max_mid = params["max_mid"]

    df = df.copy()

    # Consensus: BTC and ETH agree in sign
    df["btc_sign"] = np.sign(df["btc_imb"])
    df["eth_sign"] = np.sign(df["eth_imb"])
    df["consensus"] = (df["btc_sign"] == df["eth_sign"]) & (df["btc_sign"] != 0)

    # Consensus direction: positive imbalance = bullish (Up), negative = bearish (Down)
    df["consensus_dir"] = df["btc_sign"]

    # Minimum imbalance magnitude for BTC and ETH
    df["btc_mag"] = np.abs(df["btc_imb"])
    df["eth_mag"] = np.abs(df["eth_imb"])
    df["major_imb_ok"] = (df["btc_mag"] >= imb_threshold) & (df["eth_mag"] >= imb_threshold)

    # Divergence: SOL or XRP has opposite sign from consensus
    df["sol_diverges"] = (np.sign(df["sol_imb"]) != df["btc_sign"]) & (np.abs(df["sol_imb"]) >= divergence_threshold)
    df["xrp_diverges"] = (np.sign(df["xrp_imb"]) != df["btc_sign"]) & (np.abs(df["xrp_imb"]) >= divergence_threshold)
    df["any_divergence"] = df["sol_diverges"] | df["xrp_diverges"]

    # Price filter: mid_price between bounds and below 0.85 cap
    df["price_ok"] = (df["btc_mid"] >= min_mid) & (df["btc_mid"] <= max_mid)

    # Signal: consensus + divergence + price ok
    df["signal"] = df["consensus"] & df["major_imb_ok"] & df["any_divergence"] & df["price_ok"]

    # Trade direction: if consensus is bullish, buy Up token; if bearish, buy Down token
    # consensus_dir > 0 => buy Up, consensus_dir < 0 => buy Down
    df["trade_token"] = np.where(df["consensus_dir"] > 0, "Up", "Down")

    # Entry price: if buying Up, entry = btc_mid; if buying Down, entry = 1 - btc_mid
    df["entry_price"] = np.where(
        df["trade_token"] == "Up",
        df["btc_mid"],
        1.0 - df["btc_mid"],
    )

    return df


def backtest(signals: pd.DataFrame, resolutions: dict, params: dict) -> dict:
    """Backtest signal DataFrame against resolution outcomes."""

    trades = signals[signals["signal"]].copy()
    print(f"\nTotal signals: {len(trades)}")

    if len(trades) == 0:
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "pnl_total": 0.0,
            "pnl_per_trade": 0.0,
            "max_drawdown": 0.0,
            "sharpe": 0.0,
        }

    # Map resolution outcomes
    trades["slug"] = "btc-updown-5m-" + trades["epoch"].astype(str)
    trades["winner"] = trades["slug"].map(
        lambda s: resolutions.get(s, {}).get("winnerLabel")
    )

    # Drop unresolved
    trades = trades[trades["winner"].notna()].copy()
    print(f"Resolved trades: {len(trades)}")

    if len(trades) == 0:
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "pnl_total": 0.0,
            "pnl_per_trade": 0.0,
            "max_drawdown": 0.0,
            "sharpe": 0.0,
        }

    # Determine win/loss
    trades["won"] = trades["trade_token"] == trades["winner"]

    # PnL calculation
    # Buy token at entry_price, pays 1.0 if win, 0.0 if lose
    # Fee on entry
    entry_fees = polymarket_fee(trades["entry_price"].values)
    cost = trades["entry_price"].values + entry_fees

    # Payout: 1.0 if won (minus exit fee on winnings), 0.0 if lost
    exit_fees = np.where(trades["won"].values, polymarket_fee(np.ones(len(trades))), 0.0)
    payout = np.where(trades["won"].values, 1.0 - exit_fees, 0.0)

    trades["pnl"] = payout - cost

    # Position size: 5 tokens (minimum)
    position_size = params.get("position_size", 5)
    trades["pnl_sized"] = trades["pnl"] * position_size

    # Metrics
    total_trades = len(trades)
    wins = trades["won"].sum()
    win_rate = wins / total_trades
    pnl_total = trades["pnl_sized"].sum()
    pnl_per_trade = pnl_total / total_trades

    # Max drawdown
    cumulative = trades["pnl_sized"].cumsum()
    running_max = cumulative.cummax()
    drawdown = cumulative - running_max
    max_drawdown = drawdown.min()

    # Sharpe (annualized, assuming ~288 5-min windows per day)
    if trades["pnl"].std() > 0:
        sharpe = (trades["pnl"].mean() / trades["pnl"].std()) * np.sqrt(288)
    else:
        sharpe = 0.0

    print(f"\n=== BACKTEST RESULTS ===")
    print(f"Total trades:    {total_trades}")
    print(f"Win rate:        {win_rate:.4f}")
    print(f"PnL total:       {pnl_total:.4f}")
    print(f"PnL per trade:   {pnl_per_trade:.4f}")
    print(f"Max drawdown:    {max_drawdown:.4f}")
    print(f"Sharpe ratio:    {sharpe:.4f}")

    # Distribution analysis
    print(f"\n--- Trade Distribution ---")
    print(f"Up bets: {(trades['trade_token'] == 'Up').sum()}, Down bets: {(trades['trade_token'] == 'Down').sum()}")
    print(f"Mean entry price: {trades['entry_price'].mean():.4f}")
    print(f"Mean PnL per token: {trades['pnl'].mean():.4f}")

    # Win rate by direction
    for direction in ["Up", "Down"]:
        mask = trades["trade_token"] == direction
        if mask.sum() > 0:
            wr = trades.loc[mask, "won"].mean()
            print(f"  {direction} win rate: {wr:.4f} (n={mask.sum()})")

    return {
        "total_trades": int(total_trades),
        "win_rate": round(float(win_rate), 4),
        "pnl_total": round(float(pnl_total), 4),
        "pnl_per_trade": round(float(pnl_per_trade), 4),
        "max_drawdown": round(float(max_drawdown), 4),
        "sharpe": round(float(sharpe), 4),
    }


def run_parameter_sweep(df: pd.DataFrame, resolutions: dict) -> list:
    """Sweep parameters to find robust signals (not to cherry-pick, but to assess robustness)."""
    results = []

    param_grid = [
        {"imb_threshold": t, "divergence_threshold": d, "min_mid": mn, "max_mid": mx}
        for t in [0.05, 0.10, 0.15, 0.20, 0.30]
        for d in [0.05, 0.10, 0.15, 0.20]
        for mn in [0.20, 0.30]
        for mx in [0.70, 0.80, 0.85]
    ]

    for params in param_grid:
        params["position_size"] = 5
        signals = generate_signals(df, params)
        trades = signals[signals["signal"]].copy()

        if len(trades) == 0:
            continue

        trades["slug"] = "btc-updown-5m-" + trades["epoch"].astype(str)
        trades["winner"] = trades["slug"].map(
            lambda s: resolutions.get(s, {}).get("winnerLabel")
        )
        trades = trades[trades["winner"].notna()]

        if len(trades) < 10:
            continue

        trades["won"] = trades["trade_token"] == trades["winner"]
        entry_fees = polymarket_fee(trades["entry_price"].values)
        cost = trades["entry_price"].values + entry_fees
        exit_fees = np.where(trades["won"].values, polymarket_fee(np.ones(len(trades))), 0.0)
        payout = np.where(trades["won"].values, 1.0 - exit_fees, 0.0)
        trades["pnl"] = payout - cost

        wr = trades["won"].mean()
        pnl_mean = trades["pnl"].mean()
        pnl_std = trades["pnl"].std()
        sharpe = (pnl_mean / pnl_std * np.sqrt(288)) if pnl_std > 0 else 0.0

        results.append({
            **params,
            "n_trades": len(trades),
            "win_rate": round(wr, 4),
            "pnl_mean": round(pnl_mean, 4),
            "sharpe": round(sharpe, 4),
        })

    return sorted(results, key=lambda x: x["sharpe"], reverse=True)


def main():
    print("=" * 60)
    print("Cross-Asset Book Imbalance Divergence Strategy")
    print("=" * 60)

    # Load resolutions
    resolutions = load_resolution_cache()
    resolved_epochs = set()
    for slug, info in resolutions.items():
        if info.get("winnerLabel") is not None:
            resolved_epochs.add(extract_epoch_from_slug(slug))
    print(f"Resolved BTC epochs: {len(resolved_epochs)}")

    # Build cross-asset signals using all available data
    signals_df = build_cross_asset_signals(resolved_epochs)

    if len(signals_df) == 0:
        print("No cross-asset data found!")
        write_results(None, {})
        return

    # Parameter sweep for robustness assessment
    print("\n" + "=" * 60)
    print("Parameter Sweep (robustness check)")
    print("=" * 60)
    sweep = run_parameter_sweep(signals_df, resolutions)

    print(f"\nTop 10 parameter sets by Sharpe:")
    print(f"{'imb_th':>8} {'div_th':>8} {'min_mid':>8} {'max_mid':>8} {'n_trades':>8} {'win_rate':>8} {'pnl_mean':>9} {'sharpe':>8}")
    for r in sweep[:10]:
        print(f"{r['imb_threshold']:8.2f} {r['divergence_threshold']:8.2f} {r['min_mid']:8.2f} {r['max_mid']:8.2f} {r['n_trades']:8d} {r['win_rate']:8.4f} {r['pnl_mean']:9.4f} {r['sharpe']:8.4f}")

    print(f"\nBottom 10 parameter sets by Sharpe:")
    for r in sweep[-10:]:
        print(f"{r['imb_threshold']:8.2f} {r['divergence_threshold']:8.2f} {r['min_mid']:8.2f} {r['max_mid']:8.2f} {r['n_trades']:8d} {r['win_rate']:8.4f} {r['pnl_mean']:9.4f} {r['sharpe']:8.4f}")

    # Assess overall robustness
    positive_sharpe = sum(1 for r in sweep if r["sharpe"] > 0)
    total_configs = len(sweep)
    print(f"\nRobustness: {positive_sharpe}/{total_configs} configs have positive Sharpe")

    avg_win_rate = np.mean([r["win_rate"] for r in sweep]) if sweep else 0
    avg_sharpe = np.mean([r["sharpe"] for r in sweep]) if sweep else 0
    print(f"Average win rate across all configs: {avg_win_rate:.4f}")
    print(f"Average Sharpe across all configs: {avg_sharpe:.4f}")

    # Run with default/median parameters for the main result
    print("\n" + "=" * 60)
    print("Main Backtest (default parameters)")
    print("=" * 60)

    default_params = {
        "imb_threshold": 0.10,
        "divergence_threshold": 0.10,
        "min_mid": 0.20,
        "max_mid": 0.80,
        "position_size": 5,
    }

    signals = generate_signals(signals_df, default_params)
    metrics = backtest(signals, resolutions, default_params)

    # Write results
    write_results(metrics, default_params, sweep)


def write_results(metrics, params, sweep=None):
    """Write standardized results JSON."""

    if metrics is None or metrics.get("total_trades", 0) == 0:
        verdict = "failed"
        description = "No tradeable signals generated."
    elif metrics["sharpe"] > 1.0 and metrics["pnl_total"] > 0:
        verdict = "promising"
        description = "Strategy shows positive risk-adjusted returns."
    elif metrics["sharpe"] > 0 and metrics["pnl_total"] > 0:
        verdict = "marginal"
        description = "Strategy shows weakly positive returns but low Sharpe."
    else:
        verdict = "failed"
        description = "Strategy does not generate positive returns after fees."

    # Assess robustness from sweep
    if sweep:
        positive_sharpe = sum(1 for r in sweep if r["sharpe"] > 0)
        total_configs = len(sweep)
        robustness_pct = positive_sharpe / total_configs if total_configs > 0 else 0
        robustness_note = f" {positive_sharpe}/{total_configs} ({robustness_pct:.0%}) parameter configs show positive Sharpe."
    else:
        robustness_note = ""

    results = {
        "strategy_name": "cross_asset_book_imbalance_divergence",
        "description": (
            "When BTC and ETH 5m orderbook imbalances agree in direction but SOL or XRP "
            "imbalance diverges (opposite sign), bet that BTC resolves in the BTC/ETH consensus direction. "
            "Hypothesis: informed flow hits correlated majors first, alt divergence is noise."
        ),
        "hypothesis": (
            "Divergent alt-coin book imbalance (SOL/XRP vs BTC/ETH consensus) reflects stale "
            "or noise liquidity rather than genuine contrarian information. Fading the outlier "
            "captures a reversion premium."
        ),
        "metrics": metrics if metrics else {
            "total_trades": 0, "win_rate": 0.0, "pnl_total": 0.0,
            "pnl_per_trade": 0.0, "max_drawdown": 0.0, "sharpe": 0.0,
        },
        "parameters": params,
        "verdict": verdict,
        "next_steps": (
            f"{description}{robustness_note} "
            "Consider: (1) time-weighting snapshots closer to market open, "
            "(2) adding spread/liquidity filters, (3) using weighted imbalance by size, "
            "(4) testing with trade flow data for confirmation."
        ),
    }

    output_path = f"{OUTPUT}/results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {output_path}")


if __name__ == "__main__":
    main()
