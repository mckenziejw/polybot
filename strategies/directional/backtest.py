"""Backtest for BTC directional classifier (v2).

Simulates trading on walk-forward predictions with realistic costs
(Bybit VIP0 fees + slippage). Uses variable holding periods from
triple-barrier labels for exit timing.

Usage:
    python -m strategies.directional.backtest
    python -m strategies.directional.backtest --results-dir data/directional_results_v2
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DirectionalConfig


def _apply_cooldown(
    timestamps: np.ndarray,
    holding_periods: np.ndarray,
    snapshot_interval_ms: int,
    cooldown_ms: int,
) -> np.ndarray:
    """Return boolean mask keeping only rows that respect cooldown from exit time.

    Cooldown starts from exit_ts = entry_ts + holding_period * interval_ms.
    """
    mask = np.zeros(len(timestamps), dtype=bool)
    next_allowed_ts = -cooldown_ms
    for i in range(len(timestamps)):
        if timestamps[i] >= next_allowed_ts:
            mask[i] = True
            hp = holding_periods[i] if np.isfinite(holding_periods[i]) else 0
            exit_ts = timestamps[i] + hp * snapshot_interval_ms
            next_allowed_ts = exit_ts + cooldown_ms
    return mask


def simulate_trades(
    predictions_df: pd.DataFrame,
    book_df: pd.DataFrame,
    cfg: DirectionalConfig,
    confidence: float = 0.60,
) -> pd.DataFrame:
    """Simulate trades from predictions — vectorized with cooldown scan.

    Uses variable holding periods from triple-barrier labels for exit timing.

    Parameters
    ----------
    predictions_df : DataFrame with timestamp_ms, y_true, y_pred, y_std,
                     holding_period, window
    book_df : DataFrame with timestamp_ms index, bid_price_1, ask_price_1
    cfg : DirectionalConfig

    Returns DataFrame of trades with PnL.
    """
    cooldown_ms = cfg.cooldown_snapshots * cfg.snapshot_interval_ms
    interval_ms = cfg.snapshot_interval_ms
    notional = cfg.notional_per_trade
    entry_cost_rate = cfg.taker_fee + cfg.slippage_bps / 10_000
    exit_cost_rate = cfg.maker_fee
    tolerance = cfg.snapshot_interval_ms

    preds = predictions_df.sort_values("timestamp_ms").reset_index(drop=True)

    # Vectorized: determine direction for all predictions
    prob = preds["y_pred"].values
    direction = np.where(prob > confidence, 1,
                         np.where(prob < (1 - confidence), -1, 0))

    # Filter to only confident predictions
    confident_mask = direction != 0
    candidates = preds[confident_mask].copy()
    candidates["direction"] = direction[confident_mask]

    if candidates.empty:
        return pd.DataFrame()

    # Get holding periods (fall back to primary_horizon if column missing)
    if "holding_period" in candidates.columns:
        hp_values = candidates["holding_period"].values.astype(np.float64)
        # Fill NaN with max_barrier_horizon
        hp_values = np.where(np.isfinite(hp_values), hp_values, cfg.max_barrier_horizon)
    else:
        hp_values = np.full(len(candidates), cfg.primary_horizon, dtype=np.float64)

    # Apply cooldown (forward scan on small subset, cooldown from exit time)
    ts_values = candidates["timestamp_ms"].values.astype(np.int64)
    cd_mask = _apply_cooldown(ts_values, hp_values, interval_ms, cooldown_ms)
    candidates = candidates[cd_mask].reset_index(drop=True)
    hp_values = hp_values[cd_mask]

    if candidates.empty:
        return pd.DataFrame()

    # Vectorized price lookup via merge_asof
    entry_ts = candidates["timestamp_ms"].astype(np.int64)
    exit_ts = entry_ts + (hp_values * interval_ms).astype(np.int64)

    # Look up entry prices
    entry_lookup = pd.DataFrame({"timestamp_ms": entry_ts})
    entry_merged = pd.merge_asof(
        entry_lookup.sort_values("timestamp_ms"),
        book_df.reset_index().sort_values("timestamp_ms"),
        on="timestamp_ms",
        direction="nearest",
        tolerance=tolerance,
    )

    # Look up exit prices
    exit_lookup = pd.DataFrame({"timestamp_ms": exit_ts})
    exit_merged = pd.merge_asof(
        exit_lookup.sort_values("timestamp_ms"),
        book_df.reset_index().sort_values("timestamp_ms"),
        on="timestamp_ms",
        direction="nearest",
        tolerance=tolerance,
    )

    # Build result arrays
    d = candidates["direction"].values
    entry_bid = entry_merged["bid_price_1"].values
    entry_ask = entry_merged["ask_price_1"].values
    exit_bid = exit_merged["bid_price_1"].values
    exit_ask = exit_merged["ask_price_1"].values

    # Entry: long buys at ask, short sells at bid
    entry_price = np.where(d == 1, entry_ask, entry_bid)
    # Exit: long sells at bid, short buys at ask
    exit_price = np.where(d == 1, exit_bid, exit_ask)

    # Filter out rows where price lookup failed (NaN from merge_asof)
    valid = (~np.isnan(entry_price)) & (~np.isnan(exit_price)) & \
            (entry_price > 0) & (exit_price > 0)

    candidates = candidates[valid].reset_index(drop=True)
    entry_price = entry_price[valid]
    exit_price = exit_price[valid]
    d = d[valid]

    if len(candidates) == 0:
        return pd.DataFrame()

    # Vectorized PnL
    position_size = notional / entry_price
    gross_pnl = position_size * (exit_price - entry_price) * d
    entry_cost = notional * entry_cost_rate
    exit_cost = notional * exit_cost_rate
    net_pnl = gross_pnl - entry_cost - exit_cost

    return pd.DataFrame({
        "timestamp_ms": candidates["timestamp_ms"].values,
        "direction": d,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "holding_period": hp_values[valid],
        "prob": candidates["y_pred"].values,
        "y_true": candidates["y_true"].values,
        "gross_pnl": gross_pnl,
        "fees": entry_cost + exit_cost,
        "net_pnl": net_pnl,
        "window": candidates["window"].values if "window" in candidates.columns
                  else np.full(len(candidates), -1),
    })


def load_predictions_and_book(
    results_dir: Path, orderbook_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load predictions + orderbook data once for reuse across thresholds."""
    pred_path = results_dir / "predictions.parquet"
    if not pred_path.exists():
        raise FileNotFoundError(f"No predictions at {pred_path}")

    preds = pd.read_parquet(pred_path)
    print(f"Loaded {len(preds):,} predictions")

    # Load only bid/ask from orderbook files that overlap prediction dates
    pred_dates = set(
        pd.to_datetime(preds["timestamp_ms"].astype(np.int64), unit="ms", utc=True)
        .dt.strftime("%Y-%m-%d")
    )

    ob_frames = []
    parquet_files = sorted(orderbook_dir.glob("*.parquet"))
    for f in parquet_files:
        if f.stem in pred_dates:
            ob_frames.append(
                pd.read_parquet(f, columns=[
                    "timestamp_ms", "bid_price_1", "ask_price_1"
                ])
            )

    if not ob_frames:
        raise FileNotFoundError(f"No orderbook data found in {orderbook_dir}")

    ob_df = pd.concat(ob_frames, ignore_index=True)
    ob_df["timestamp_ms"] = ob_df["timestamp_ms"].astype(np.int64)
    ob_df = ob_df.drop_duplicates(subset=["timestamp_ms"], keep="last")
    ob_df = ob_df.set_index("timestamp_ms").sort_index()
    print(f"Loaded orderbook data: {len(ob_df):,} snapshots")

    return preds, ob_df


def load_predictions_and_candles(
    results_dir: Path,
    candles_dir: Path,
    symbol: str = "BTC-USDT-PERP",
    spread_bps: float = 5.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load predictions + candle data, synthesizing bid/ask from close + spread.

    For assets without separate orderbook data, we use 1-minute candles
    and apply a fixed spread to simulate bid/ask prices.

    Parameters
    ----------
    spread_bps : float
        Half-spread in basis points applied to close price.
        E.g., 5 bps means bid = close * (1 - 0.0005), ask = close * (1 + 0.0005)
    """
    pred_path = results_dir / "predictions.parquet"
    if not pred_path.exists():
        raise FileNotFoundError(f"No predictions at {pred_path}")

    preds = pd.read_parquet(pred_path)
    print(f"Loaded {len(preds):,} predictions")

    pred_dates = set(
        pd.to_datetime(preds["timestamp_ms"].astype(np.int64), unit="ms", utc=True)
        .dt.strftime("%Y-%m-%d")
    )

    candle_path = candles_dir / "candles" / "exchange=BINANCE_FUTURES" / f"symbol={symbol}"
    if not candle_path.exists():
        raise FileNotFoundError(f"No candle data at {candle_path}")

    frames = []
    for f in sorted(candle_path.glob("*.parquet")):
        if f.stem in pred_dates:
            df = pd.read_parquet(f)
            frames.append(df)

    if not frames:
        raise FileNotFoundError(f"No candle data found for prediction dates")

    candles = pd.concat(frames, ignore_index=True)
    candles["timestamp_ms"] = (candles["timestamp"] // 1_000_000).astype("int64")

    # Synthesize bid/ask from close price + spread
    half_spread = spread_bps / 10_000
    candles["bid_price_1"] = candles["close"] * (1 - half_spread)
    candles["ask_price_1"] = candles["close"] * (1 + half_spread)

    ob_df = candles[["timestamp_ms", "bid_price_1", "ask_price_1"]].copy()
    ob_df = ob_df.drop_duplicates(subset=["timestamp_ms"], keep="last")
    ob_df = ob_df.set_index("timestamp_ms").sort_index()
    print(f"Loaded candle data: {len(ob_df):,} bars (spread={spread_bps:.1f}bps)")

    return preds, ob_df


def compute_metrics(trades_df: pd.DataFrame, cfg: DirectionalConfig) -> dict:
    """Compute backtest performance metrics."""
    if trades_df.empty:
        return {"n_trades": 0}

    net = trades_df["net_pnl"]
    gross = trades_df["gross_pnl"]
    n = len(trades_df)

    total_pnl = net.sum()
    avg_pnl = net.mean()
    win_rate = (net > 0).mean()

    # Sharpe — aggregate to daily PnL, then annualize (avoids HFT inflation)
    ts = trades_df["timestamp_ms"].astype(np.int64)
    dates = pd.to_datetime(ts, unit="ms", utc=True).dt.date
    daily_pnl = trades_df.groupby(dates)["net_pnl"].sum()
    if len(daily_pnl) >= 2 and daily_pnl.std() > 0:
        sharpe = (daily_pnl.mean() / daily_pnl.std()) * np.sqrt(252)
    else:
        sharpe = 0.0

    # Drawdown
    cum = net.cumsum()
    peak = cum.cummax()
    dd = cum - peak
    max_dd = dd.min()

    # Profit factor
    wins = net[net > 0].sum()
    losses = abs(net[net < 0].sum())
    pf = wins / losses if losses > 0 else 1e6

    # Fees as % of notional turnover
    total_fees = trades_df["fees"].sum()
    total_turnover = cfg.notional_per_trade * n
    fee_pct = total_fees / total_turnover if total_turnover > 0 else 0

    # Trades per day
    n_days = len(daily_pnl)
    trades_per_day = n / n_days if n_days > 0 else 0

    # Holding period stats
    if "holding_period" in trades_df.columns:
        hp = trades_df["holding_period"]
        mean_hp = float(hp.mean())
        median_hp = float(hp.median())
    else:
        mean_hp = 0.0
        median_hp = 0.0

    return {
        "n_trades": n,
        "n_days": n_days,
        "trades_per_day": float(trades_per_day),
        "total_pnl": float(total_pnl),
        "avg_pnl": float(avg_pnl),
        "win_rate": float(win_rate),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_dd),
        "profit_factor": float(pf),
        "total_fees": float(total_fees),
        "fee_impact_pct": float(fee_pct),
        "avg_gross_pnl": float(gross.mean()),
        "mean_holding_period": mean_hp,
        "median_holding_period": median_hp,
    }


def run_backtest(args):
    """Full backtest with threshold sweep and per-window breakdown."""
    cfg = DirectionalConfig.config_1m() if getattr(args, "resolution", "1m") == "1m" else DirectionalConfig()
    results_dir = Path(args.results_dir)

    symbol = getattr(args, "symbol", "BTC-USDT-PERP")
    use_candles = getattr(args, "candles", False)

    # Override costs for Binance perps (not Bybit)
    if hasattr(args, "taker_fee") and args.taker_fee is not None:
        cfg.taker_fee = args.taker_fee
    if hasattr(args, "maker_fee") and args.maker_fee is not None:
        cfg.maker_fee = args.maker_fee

    print(f"{'='*60}")
    print(f"DIRECTIONAL CLASSIFIER — BACKTEST ({symbol})")
    print(f"{'='*60}")
    print(f"Results dir: {results_dir}")
    print(f"Primary horizon: {cfg.primary_horizon} snapshots "
          f"({cfg.horizon_seconds(cfg.primary_horizon):.1f}s)")
    print(f"Notional per trade: ${cfg.notional_per_trade:.0f}")
    print(f"Costs: taker={cfg.taker_fee:.4f}, maker={cfg.maker_fee:.4f}, "
          f"slippage={cfg.slippage_bps:.1f}bps")

    # Load data once
    if use_candles:
        candles_dir = Path(args.data_dir)
        spread_bps = getattr(args, "spread_bps", 5.0)
        preds, ob_df = load_predictions_and_candles(
            results_dir, candles_dir, symbol=symbol, spread_bps=spread_bps,
        )
    else:
        orderbook_dir = Path(args.data_dir)
        preds, ob_df = load_predictions_and_book(results_dir, orderbook_dir)

    # Confidence threshold sweep
    print(f"\n{'='*60}")
    print("CONFIDENCE THRESHOLD SWEEP")
    print(f"{'='*60}")
    print(f"{'Conf':>6} {'Trades':>8} {'Tr/day':>8} {'WR':>8} {'PnL':>10} "
          f"{'Avg PnL':>10} {'Sharpe':>8} {'MaxDD':>10} {'PF':>8}")

    best_sharpe = -999
    best_conf = 0.5
    best_trades = pd.DataFrame()
    sweep_results = []

    for conf in cfg.confidence_sweep:
        trades = simulate_trades(preds, ob_df, cfg, conf)
        metrics = compute_metrics(trades, cfg)

        n = metrics["n_trades"]
        if n == 0:
            print(f"{conf:>6.2f} {'—':>8}")
            continue

        print(
            f"{conf:>6.2f} {n:>8,} {metrics['trades_per_day']:>8.1f} "
            f"{metrics['win_rate']:>8.3f} "
            f"${metrics['total_pnl']:>9.2f} ${metrics['avg_pnl']:>9.4f} "
            f"{metrics['sharpe']:>8.2f} ${metrics['max_drawdown']:>9.2f} "
            f"{metrics['profit_factor']:>8.2f}"
        )

        sweep_results.append({"confidence": conf, **metrics})

        if metrics["sharpe"] > best_sharpe:
            best_sharpe = metrics["sharpe"]
            best_conf = conf
            best_trades = trades

    # Detailed analysis at best confidence
    print(f"\n{'='*60}")
    print(f"DETAILED ANALYSIS (confidence={best_conf:.2f})")
    print(f"{'='*60}")

    trades = best_trades
    if trades.empty:
        print("No trades at best confidence level.")
        return

    metrics = compute_metrics(trades, cfg)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    # Per-window breakdown
    if "window" in trades.columns:
        print(f"\nPer-window breakdown:")
        print(f"{'Window':>8} {'Trades':>8} {'WR':>8} {'PnL':>10} {'Avg':>10}")
        for w in sorted(trades["window"].unique()):
            if w == -1:
                continue
            wt = trades[trades["window"] == w]
            wn = len(wt)
            wwr = (wt["net_pnl"] > 0).mean()
            wpnl = wt["net_pnl"].sum()
            wavg = wt["net_pnl"].mean()
            print(f"{w:>8} {wn:>8,} {wwr:>8.3f} ${wpnl:>9.2f} ${wavg:>9.4f}")

    # Direction breakdown
    print(f"\nDirection breakdown:")
    for d, label in [(1, "Long"), (-1, "Short")]:
        dt = trades[trades["direction"] == d]
        if dt.empty:
            continue
        dn = len(dt)
        dwr = (dt["net_pnl"] > 0).mean()
        dpnl = dt["net_pnl"].sum()
        print(f"  {label}: {dn:,} trades, WR={dwr:.3f}, PnL=${dpnl:.2f}")

    # Equity curve
    trades = trades.copy()
    trades["cum_pnl"] = trades["net_pnl"].cumsum()
    print(f"\nEquity curve: ${trades['cum_pnl'].iloc[0]:.2f} → "
          f"${trades['cum_pnl'].iloc[-1]:.2f}")

    # Save detailed results
    output_dir = results_dir / "backtest"
    output_dir.mkdir(parents=True, exist_ok=True)
    trades.to_parquet(output_dir / "trades.parquet", index=False)
    with open(output_dir / "metrics.json", "w") as f:
        json.dump({"best_confidence": best_conf, "metrics": metrics,
                    "sweep": sweep_results}, f, indent=2)

    print(f"\nBacktest results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Backtest directional classifier predictions"
    )
    parser.add_argument(
        "--results-dir", default="data/directional_results_v2",
        help="Directory with predictions.parquet from training",
    )
    parser.add_argument(
        "--data-dir", default="data/bybit_orderbook_raw",
        help="Orderbook or candles data dir for fill prices",
    )
    parser.add_argument("--candles", action="store_true",
                        help="Use candle data instead of orderbook for price simulation")
    parser.add_argument("--symbol", default="BTC-USDT-PERP",
                        help="Symbol for candle data lookup")
    parser.add_argument("--spread-bps", type=float, default=5.0,
                        help="Half-spread in bps for candle-based simulation (default 5)")
    parser.add_argument("--resolution", default="1m", choices=["200ms", "1m"],
                        help="Data resolution for config")
    parser.add_argument("--taker-fee", type=float, default=None,
                        help="Override taker fee rate")
    parser.add_argument("--maker-fee", type=float, default=None,
                        help="Override maker fee rate")
    args = parser.parse_args()
    run_backtest(args)


if __name__ == "__main__":
    main()
