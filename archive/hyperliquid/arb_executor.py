"""
Delta-neutral funding rate arbitrage executor.

Opens long spot + short perp positions to collect funding payments while
remaining market-neutral. Monitors position delta and rebalances as needed.

SAFETY: Defaults to DRY-RUN mode. All execution methods log what they
WOULD do without actually placing orders. Pass --live to enable real trading.

Usage:
    # Dry-run (default) — logs intended actions
    python -m hyperliquid.arb_executor --coin BTC --size 1000

    # Live trading (requires explicit flag + private key)
    python -m hyperliquid.arb_executor --coin BTC --size 1000 --live
"""

import argparse
import logging
import time
from datetime import datetime, timezone
from typing import Any

from .client import HyperliquidClient
from .websocket_client import HyperliquidWS
from .config import Config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Position tracking
# ---------------------------------------------------------------------------


class PositionTracker:
    """Tracks open positions and cumulative PnL from funding."""

    def __init__(self):
        self.perp_size: float = 0.0       # negative = short
        self.perp_entry_price: float = 0.0
        self.spot_size: float = 0.0        # positive = long
        self.spot_entry_price: float = 0.0

        self.cumulative_funding: float = 0.0
        self.cumulative_price_pnl: float = 0.0
        self.total_fees_paid: float = 0.0
        self.n_funding_payments: int = 0

        self.opened_at: str | None = None
        self.last_funding_at: str | None = None

    @property
    def net_delta(self) -> float:
        """Net position delta (should be ~0 for delta-neutral)."""
        return self.perp_size + self.spot_size

    @property
    def delta_pct(self) -> float:
        """Delta as percentage of total position size."""
        total = abs(self.perp_size) + abs(self.spot_size)
        if total == 0:
            return 0.0
        return abs(self.net_delta) / total * 100

    @property
    def total_pnl(self) -> float:
        """Total PnL = funding collected - fees - price delta losses."""
        return self.cumulative_funding - self.total_fees_paid + self.cumulative_price_pnl

    @property
    def is_open(self) -> bool:
        return abs(self.perp_size) > 0 or abs(self.spot_size) > 0

    def record_funding_payment(self, amount: float) -> None:
        """Record a funding payment received/paid."""
        self.cumulative_funding += amount
        self.n_funding_payments += 1
        self.last_funding_at = datetime.now(timezone.utc).isoformat()

    def update_price_pnl(self, current_price: float) -> None:
        """Update unrealized PnL from price movement."""
        if not self.is_open:
            self.cumulative_price_pnl = 0.0
            return

        perp_pnl = self.perp_size * (current_price - self.perp_entry_price)
        spot_pnl = self.spot_size * (current_price - self.spot_entry_price)
        self.cumulative_price_pnl = perp_pnl + spot_pnl

    def summary(self, current_price: float | None = None) -> str:
        """Return a human-readable position summary."""
        lines = []
        lines.append(f"  Perp:    {self.perp_size:+.6f} @ ${self.perp_entry_price:,.2f}")
        lines.append(f"  Spot:    {self.spot_size:+.6f} @ ${self.spot_entry_price:,.2f}")
        lines.append(f"  Delta:   {self.net_delta:+.6f} ({self.delta_pct:.2f}%)")
        lines.append(f"  Funding: ${self.cumulative_funding:+.4f} ({self.n_funding_payments} payments)")
        lines.append(f"  Fees:    ${self.total_fees_paid:.4f}")

        if current_price is not None:
            self.update_price_pnl(current_price)
            lines.append(f"  Price PnL: ${self.cumulative_price_pnl:+.4f}")
            lines.append(f"  Total PnL: ${self.total_pnl:+.4f}")

        if self.opened_at:
            lines.append(f"  Opened:  {self.opened_at}")
        if self.last_funding_at:
            lines.append(f"  Last funding: {self.last_funding_at}")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Arb Executor
# ---------------------------------------------------------------------------


class ArbExecutor:
    """
    Delta-neutral funding rate arbitrage executor.

    Strategy: when funding rate is positive (longs pay shorts),
    go short perp + long spot to collect funding while staying neutral.
    Vice versa when funding is negative.

    Default mode is DRY-RUN — logs intended actions without executing.
    """

    def __init__(self, config: Config, coin: str, target_size_usd: float):
        self.config = config
        self.coin = coin
        self.target_size_usd = target_size_usd
        self.dry_run = config.dry_run
        self.client = HyperliquidClient(config=config)
        self.position = PositionTracker()

        self._mid_price: float = 0.0
        self._current_funding_rate: float = 0.0
        self._last_poll: float = 0.0

    @property
    def mode_str(self) -> str:
        return "DRY-RUN" if self.dry_run else "LIVE"

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------

    def _update_market_data(self) -> None:
        """Fetch current mid price and funding rate."""
        try:
            mids = self.client.get_mids()
            if self.coin in mids:
                self._mid_price = mids[self.coin]
        except Exception as e:
            logger.error("Failed to fetch mids: %s", e)

        try:
            meta, ctxs = self.client.get_funding_rates()
            universe = meta["universe"]
            for asset_meta, ctx in zip(universe, ctxs):
                if asset_meta["name"] == self.coin:
                    self._current_funding_rate = float(ctx.get("funding", 0))
                    break
        except Exception as e:
            logger.error("Failed to fetch funding rates: %s", e)

    def _handle_mid_update(self, data: Any) -> None:
        """WebSocket callback for mid-price updates."""
        if isinstance(data, dict) and "mids" in data:
            mids = data["mids"]
        elif isinstance(data, dict):
            mids = data
        else:
            return

        price_str = mids.get(self.coin)
        if price_str:
            try:
                self._mid_price = float(price_str)
            except (ValueError, TypeError):
                pass

    # ------------------------------------------------------------------
    # Order execution (dry-run aware)
    # ------------------------------------------------------------------

    def _execute_order(
        self,
        is_buy: bool,
        price: float,
        size: float,
        order_type: str = "limit",
        reduce_only: bool = False,
        label: str = "",
    ) -> dict | None:
        """
        Place an order, or log what would be placed in dry-run mode.

        Returns the order response (or a mock response in dry-run).
        """
        side = "BUY" if is_buy else "SELL"
        action_str = (
            f"[{self.mode_str}] {label} {side} {size:.6f} {self.coin} "
            f"@ ${price:,.2f} ({order_type})"
        )

        if self.dry_run:
            logger.info("%s — NOT EXECUTED (dry-run)", action_str)
            return {"status": "dry_run", "action": action_str}

        logger.info("%s — EXECUTING", action_str)
        try:
            result = self.client.place_order(
                coin=self.coin,
                is_buy=is_buy,
                price=price,
                size=size,
                order_type=order_type,
                reduce_only=reduce_only,
            )
            logger.info("Order result: %s", result)
            return result
        except Exception as e:
            logger.error("Order failed: %s", e)
            return None

    def _cancel_all_orders(self) -> None:
        """Cancel all open orders for the coin."""
        if self.dry_run:
            logger.info("[DRY-RUN] Would cancel all %s orders", self.coin)
            return

        try:
            result = self.client.cancel_all(self.coin)
            logger.info("Cancelled %d orders", len(result) if result else 0)
        except Exception as e:
            logger.error("Cancel all failed: %s", e)

    # ------------------------------------------------------------------
    # Position management
    # ------------------------------------------------------------------

    def open_position(self) -> None:
        """
        Open a delta-neutral position: short perp + long spot
        (when funding is positive) or long perp + short spot
        (when funding is negative).
        """
        if self.position.is_open:
            logger.warning("Position already open, skipping open_position()")
            return

        if self._mid_price <= 0:
            logger.error("No valid mid price, cannot open position")
            return

        size = self.target_size_usd / self._mid_price
        price = self._mid_price

        if self._current_funding_rate > 0:
            # Longs pay shorts -> we short perp, long spot
            logger.info(
                "Funding rate positive (%.6f) -> SHORT perp + LONG spot",
                self._current_funding_rate,
            )
            self._execute_order(
                is_buy=False, price=price, size=size,
                label="PERP SHORT",
            )
            self._execute_order(
                is_buy=True, price=price, size=size,
                label="SPOT LONG",
            )
        else:
            # Shorts pay longs -> we long perp, short spot
            logger.info(
                "Funding rate negative (%.6f) -> LONG perp + SHORT spot",
                self._current_funding_rate,
            )
            self._execute_order(
                is_buy=True, price=price, size=size,
                label="PERP LONG",
            )
            self._execute_order(
                is_buy=False, price=price, size=size,
                label="SPOT SHORT",
            )

        # Update tracker
        self.position.perp_size = -size if self._current_funding_rate > 0 else size
        self.position.perp_entry_price = price
        self.position.spot_size = size if self._current_funding_rate > 0 else -size
        self.position.spot_entry_price = price
        self.position.opened_at = datetime.now(timezone.utc).isoformat()

        logger.info("Position opened:\n%s", self.position.summary(price))

    def close_position(self) -> None:
        """Close both legs of the position."""
        if not self.position.is_open:
            logger.info("No position to close")
            return

        price = self._mid_price

        # Close perp
        if self.position.perp_size != 0:
            is_buy = self.position.perp_size < 0  # buy to close short
            self._execute_order(
                is_buy=is_buy,
                price=price,
                size=abs(self.position.perp_size),
                reduce_only=True,
                label="CLOSE PERP",
            )

        # Close spot
        if self.position.spot_size != 0:
            is_buy = self.position.spot_size < 0  # buy to close short
            self._execute_order(
                is_buy=is_buy,
                price=price,
                size=abs(self.position.spot_size),
                label="CLOSE SPOT",
            )

        # Estimate fees
        notional = abs(self.position.perp_size) * price * 2  # both legs
        fee_rate = self.config.taker_fee_bps / 10_000
        self.position.total_fees_paid += notional * fee_rate

        self.position.update_price_pnl(price)
        logger.info("Position closed:\n%s", self.position.summary(price))

        # Reset sizes
        self.position.perp_size = 0.0
        self.position.spot_size = 0.0

    def check_rebalance(self) -> None:
        """Rebalance if position delta exceeds threshold."""
        if not self.position.is_open:
            return

        if self.position.delta_pct > self.config.rebalance_threshold_pct:
            logger.warning(
                "Delta drift: %.2f%% (threshold: %.2f%%) — rebalancing",
                self.position.delta_pct,
                self.config.rebalance_threshold_pct,
            )
            # Simple rebalance: close and reopen
            # A real implementation would adjust individual legs
            self.close_position()
            time.sleep(1)
            self.open_position()

    def check_funding_flip(self) -> bool:
        """
        Check if funding rate has flipped sign.

        If we're short perp (collecting positive funding) but funding goes
        negative, we should close and potentially reverse.
        """
        if not self.position.is_open:
            return False

        # We're short perp when funding was positive
        was_positive = self.position.perp_size < 0
        is_positive = self._current_funding_rate > 0

        if was_positive != is_positive:
            logger.warning(
                "Funding rate flipped! Was %s, now %s (%.6f). Closing position.",
                "positive" if was_positive else "negative",
                "positive" if is_positive else "negative",
                self._current_funding_rate,
            )
            return True

        return False

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """
        Main execution loop.

        1. Fetch initial market data
        2. Open delta-neutral position
        3. Monitor funding + delta, rebalance as needed
        4. Close on funding flip or KeyboardInterrupt
        """
        logger.info(
            "Starting arb executor [%s] coin=%s size=$%.0f",
            self.mode_str, self.coin, self.target_size_usd,
        )

        if self.dry_run:
            logger.info(
                "DRY-RUN MODE: No orders will be placed. "
                "Use --live to enable real trading."
            )

        # Start WebSocket for live mid prices
        ws = HyperliquidWS(config=self.config)
        ws.on("allMids", self._handle_mid_update)
        ws.subscribe_all_mids()
        ws.run_in_background()

        if not ws.wait_connected(timeout=10):
            logger.warning("WebSocket did not connect, falling back to REST")

        # Initial data fetch
        self._update_market_data()
        if self._mid_price <= 0:
            logger.error("Could not get price for %s, aborting", self.coin)
            ws.stop()
            return

        logger.info(
            "%s mid price: $%.2f, funding rate: %.6f (%.1f%% ann)",
            self.coin,
            self._mid_price,
            self._current_funding_rate,
            self._current_funding_rate * 1095 * 100,
        )

        # Open position
        self.open_position()

        # Monitor loop
        poll_interval = 60  # seconds between REST polls
        try:
            while True:
                time.sleep(poll_interval)

                # Periodic REST data refresh
                self._update_market_data()

                # Check for funding flip
                if self.check_funding_flip():
                    self.close_position()
                    logger.info("Funding flipped — waiting for next opportunity")
                    # Could re-enter in opposite direction here
                    break

                # Check delta drift
                self.check_rebalance()

                # Log status
                if self.position.is_open:
                    self.position.update_price_pnl(self._mid_price)
                    logger.info(
                        "Status: price=$%.2f funding=%.6f delta=%.2f%% total_pnl=$%.4f",
                        self._mid_price,
                        self._current_funding_rate,
                        self.position.delta_pct,
                        self.position.total_pnl,
                    )

        except KeyboardInterrupt:
            logger.info("Shutting down — closing position")
            self._update_market_data()
            self.close_position()
        finally:
            ws.stop()

        # Final summary
        print(f"\n=== Final Position Summary ({self.mode_str}) ===")
        print(self.position.summary(self._mid_price))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Delta-neutral funding rate arbitrage executor"
    )
    parser.add_argument(
        "--coin", type=str, required=True,
        help="Asset to trade (e.g. BTC, ETH)"
    )
    parser.add_argument(
        "--size", type=float, default=1000.0,
        help="Target position size in USD (default: 1000)"
    )
    parser.add_argument(
        "--live", action="store_true", default=False,
        help="Enable LIVE trading (default: dry-run)"
    )
    parser.add_argument(
        "--testnet", action="store_true", default=True,
        help="Use testnet (default: True)"
    )
    parser.add_argument(
        "--mainnet", action="store_true", default=False,
        help="Use mainnet (overrides --testnet)"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config = Config.from_env()
    config.testnet = not args.mainnet
    config.dry_run = not args.live
    config.max_position_usd = args.size

    if args.live and not config.private_key:
        logger.error(
            "HYPERLIQUID_PRIVATE_KEY environment variable required for live trading"
        )
        return

    if args.live and args.mainnet:
        logger.warning("=" * 60)
        logger.warning("  LIVE TRADING ON MAINNET — REAL MONEY AT RISK")
        logger.warning("=" * 60)

    executor = ArbExecutor(
        config=config,
        coin=args.coin,
        target_size_usd=args.size,
    )
    executor.run()


if __name__ == "__main__":
    main()
