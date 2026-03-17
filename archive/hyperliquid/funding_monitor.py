"""
Enhanced funding rate monitor for Hyperliquid.

Continuously monitors funding rates via WebSocket mid-price updates and
periodic REST polling. Tracks predicted vs actual rate accuracy, alerts
on high-yield opportunities, and logs historical rates to CSV.

Builds on the patterns from funding_arb.py but adds real-time monitoring.

Usage:
    python -m hyperliquid.funding_monitor
    python -m hyperliquid.funding_monitor --threshold 30 --coins BTC ETH SOL
"""

import argparse
import csv
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .client import HyperliquidClient
from .websocket_client import HyperliquidWS
from .config import Config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FUNDING_PAYMENTS_PER_DAY = 3          # every 8 hours
DAYS_PER_YEAR = 365
ANNUALIZE_MULT = FUNDING_PAYMENTS_PER_DAY * DAYS_PER_YEAR  # 1095


def annualized_yield(funding_rate: float) -> float:
    """Convert a single 8-hour funding rate to annualized percent."""
    return funding_rate * ANNUALIZE_MULT * 100


def net_yield_after_fees(
    annual_yield_pct: float,
    holding_days: int = 30,
    maker_fee_bps: float = 1.0,
    taker_fee_bps: float = 3.5,
) -> float:
    """
    Estimate net annualized yield after entry/exit fees.

    Assumes maker entry on perp + taker entry on spot hedge.
    """
    # One-time cost: (maker + taker) for entry, same for exit = 2x
    one_time_cost_pct = 2 * (maker_fee_bps + taker_fee_bps) / 100
    annualized_cost = one_time_cost_pct * (DAYS_PER_YEAR / holding_days)
    return annual_yield_pct - annualized_cost


# ---------------------------------------------------------------------------
# Funding Monitor
# ---------------------------------------------------------------------------


class FundingMonitor:
    """
    Real-time funding rate monitor.

    Combines WebSocket mid-price streaming with periodic REST polling
    of funding rates. Logs to CSV and prints alerts.
    """

    def __init__(self, config: Config, coins: list[str] | None = None):
        self.config = config
        self.coins = coins  # None = monitor all
        self.client = HyperliquidClient(config=config)

        # State
        self._mids: dict[str, float] = {}
        self._current_rates: dict[str, dict] = {}
        self._predicted_rates: dict[str, dict] = {}
        self._prediction_history: list[dict] = []  # predicted vs actual

        # CSV logging
        self._csv_path = Path(config.funding_csv_path)
        self._csv_initialized = False

    # ------------------------------------------------------------------
    # CSV logging
    # ------------------------------------------------------------------

    def _init_csv(self) -> None:
        """Create CSV file with headers if it doesn't exist."""
        if self._csv_initialized:
            return

        if not self._csv_path.exists():
            with open(self._csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp",
                    "coin",
                    "funding_rate_8h",
                    "annualized_pct",
                    "predicted_rate_8h",
                    "predicted_annual_pct",
                    "mid_price",
                    "open_interest",
                    "mark_price",
                ])
        self._csv_initialized = True

    def _log_to_csv(self, records: list[dict]) -> None:
        """Append funding rate records to CSV."""
        self._init_csv()
        with open(self._csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            for r in records:
                writer.writerow([
                    r["timestamp"],
                    r["coin"],
                    r["funding_rate"],
                    r["annual_pct"],
                    r.get("predicted_rate", ""),
                    r.get("predicted_annual", ""),
                    r.get("mid_price", ""),
                    r.get("open_interest", ""),
                    r.get("mark_price", ""),
                ])

    # ------------------------------------------------------------------
    # REST polling
    # ------------------------------------------------------------------

    def poll_funding_rates(self) -> list[dict]:
        """
        Fetch current funding rates via REST and return records.

        Also checks predicted vs actual accuracy.
        """
        try:
            meta, asset_ctxs = self.client.get_funding_rates()
        except Exception as e:
            logger.error("Failed to fetch funding rates: %s", e)
            return []

        universe = meta["universe"]
        now = datetime.now(timezone.utc).isoformat()
        records = []

        for asset_meta, ctx in zip(universe, asset_ctxs):
            coin = asset_meta["name"]

            # Filter to requested coins if specified
            if self.coins and coin not in self.coins:
                continue

            try:
                funding_rate = float(ctx.get("funding", 0))
                mark_px = float(ctx.get("markPx", 0))
                oi = float(ctx.get("openInterest", 0))
            except (ValueError, TypeError):
                continue

            if mark_px == 0:
                continue

            annual = annualized_yield(funding_rate)
            mid = self._mids.get(coin, mark_px)

            # Check if we had a prediction for this coin
            prev_predicted = self._predicted_rates.get(coin)
            if prev_predicted:
                self._prediction_history.append({
                    "coin": coin,
                    "predicted": prev_predicted.get("rate"),
                    "actual": funding_rate,
                    "timestamp": now,
                })

            record = {
                "timestamp": now,
                "coin": coin,
                "funding_rate": funding_rate,
                "annual_pct": annual,
                "mid_price": mid,
                "open_interest": oi * mid,
                "mark_price": mark_px,
            }

            # Store current rate
            self._current_rates[coin] = {
                "rate": funding_rate,
                "annual": annual,
                "timestamp": now,
            }

            records.append(record)

            # Alert if above threshold
            if abs(annual) >= self.config.funding_alert_threshold_annual_pct:
                direction = "SHORT perp" if funding_rate > 0 else "LONG perp"
                net = net_yield_after_fees(
                    annual,
                    holding_days=30,
                    maker_fee_bps=self.config.maker_fee_bps,
                    taker_fee_bps=self.config.taker_fee_bps,
                )
                logger.warning(
                    "ALERT: %s funding %.4f%% (%.1f%% ann, ~%.1f%% net/30d) -> %s",
                    coin, funding_rate * 100, annual, net, direction,
                )

        # Log to CSV
        if records:
            self._log_to_csv(records)

        return records

    # ------------------------------------------------------------------
    # Prediction tracking
    # ------------------------------------------------------------------

    def get_prediction_accuracy(self) -> dict:
        """
        Compute how well predicted funding rates match actuals.

        Returns stats on prediction error.
        """
        if not self._prediction_history:
            return {"n_samples": 0}

        errors = []
        for entry in self._prediction_history:
            pred = entry.get("predicted")
            actual = entry.get("actual")
            if pred is not None and actual is not None:
                errors.append(abs(pred - actual))

        if not errors:
            return {"n_samples": 0}

        n = len(errors)
        avg_error = sum(errors) / n
        max_error = max(errors)

        return {
            "n_samples": n,
            "avg_abs_error": avg_error,
            "avg_abs_error_annual_pct": annualized_yield(avg_error),
            "max_abs_error": max_error,
            "max_abs_error_annual_pct": annualized_yield(max_error),
        }

    # ------------------------------------------------------------------
    # WebSocket mid-price handler
    # ------------------------------------------------------------------

    def _handle_all_mids(self, data: Any) -> None:
        """Callback for allMids WebSocket channel."""
        if isinstance(data, dict) and "mids" in data:
            mids = data["mids"]
        elif isinstance(data, dict):
            mids = data
        else:
            return

        for coin, price_str in mids.items():
            try:
                self._mids[coin] = float(price_str)
            except (ValueError, TypeError):
                pass

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """
        Start the funding monitor.

        Subscribes to allMids via WebSocket for live prices,
        polls funding rates via REST at configured intervals.
        """
        logger.info(
            "Starting funding monitor (threshold=%.1f%% ann, poll=%ds, csv=%s)",
            self.config.funding_alert_threshold_annual_pct,
            self.config.funding_poll_interval_s,
            self._csv_path,
        )
        if self.coins:
            logger.info("Monitoring coins: %s", ", ".join(self.coins))
        else:
            logger.info("Monitoring all coins")

        # Start WebSocket for live mid prices
        ws = HyperliquidWS(config=self.config)
        ws.on("allMids", self._handle_all_mids)
        ws.subscribe_all_mids()
        ws.run_in_background()

        # Wait for WebSocket connection
        if not ws.wait_connected(timeout=10):
            logger.warning("WebSocket did not connect in time, continuing with REST only")

        # Initial poll
        records = self.poll_funding_rates()
        self._print_summary(records)

        # Polling loop
        try:
            while True:
                time.sleep(self.config.funding_poll_interval_s)
                records = self.poll_funding_rates()
                self._print_summary(records)
        except KeyboardInterrupt:
            logger.info("Shutting down funding monitor")
            ws.stop()

    def _print_summary(self, records: list[dict]) -> None:
        """Print a compact summary of current funding rates."""
        if not records:
            print("[no data]")
            return

        # Sort by absolute annualized yield
        records.sort(key=lambda r: abs(r["annual_pct"]), reverse=True)

        now = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
        print(f"\n--- Funding Rates @ {now} ---")
        print(f"{'Coin':<10} {'Rate (8h)':>12} {'Annual':>10} {'Mid Price':>12} {'OI ($M)':>10}")
        print("-" * 58)

        for r in records[:30]:
            oi_m = r["open_interest"] / 1e6
            print(
                f"{r['coin']:<10} {r['funding_rate']:>12.6f} "
                f"{r['annual_pct']:>9.1f}% "
                f"${r['mid_price']:>10,.2f} "
                f"{oi_m:>9.1f}"
            )

        # Prediction accuracy
        acc = self.get_prediction_accuracy()
        if acc["n_samples"] > 0:
            print(
                f"\nPrediction accuracy ({acc['n_samples']} samples): "
                f"avg error {acc['avg_abs_error_annual_pct']:.1f}% ann"
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hyperliquid real-time funding rate monitor"
    )
    parser.add_argument(
        "--threshold", type=float, default=20.0,
        help="Alert threshold for annualized funding %% (default: 20)"
    )
    parser.add_argument(
        "--poll-interval", type=int, default=300,
        help="REST polling interval in seconds (default: 300)"
    )
    parser.add_argument(
        "--coins", nargs="+", default=None,
        help="Specific coins to monitor (default: all)"
    )
    parser.add_argument(
        "--csv", type=str, default="funding_history.csv",
        help="CSV output path (default: funding_history.csv)"
    )
    parser.add_argument(
        "--testnet", action="store_true", default=False,
        help="Use testnet (default: mainnet for monitoring)"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config = Config.from_env()
    # Override with CLI args (monitoring is read-only, safe on mainnet)
    config.testnet = args.testnet
    config.funding_alert_threshold_annual_pct = args.threshold
    config.funding_poll_interval_s = args.poll_interval
    config.funding_csv_path = args.csv

    monitor = FundingMonitor(config=config, coins=args.coins)
    monitor.run()


if __name__ == "__main__":
    main()
