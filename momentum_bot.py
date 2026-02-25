#!/usr/bin/env python3
"""
Momentum Trading Bot — Live trading on Polymarket BTC 5-min markets.

Usage:
    python momentum_bot.py                  # uses bot_config.json + config.json
    python momentum_bot.py --dry-run        # force dry-run mode
    python momentum_bot.py --live           # force live mode (careful!)

The bot:
  1. Connects to Polymarket WebSocket for live book data
  2. Observes first 50% of each 5-min market window
  3. Calculates momentum and generates buy signals
  4. Executes trades via the CLOB API
  5. Waits for resolution, redeems contracts, notifies results
  6. Repeats for the next market window
"""

import json
import logging
import signal
import sys
import time
import threading
from datetime import datetime, timezone
from pathlib import Path

from bot.engine import TradingEngine
from bot.strategy import MomentumStrategy
from bot.notifications import NotificationService
from ingest.src.market_client import PolymarketClient, MarketInfo

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("momentum_bot")


class MomentumBot:
    """
    Main bot orchestrator. Connects WebSocket feed to the momentum strategy,
    executes trades via the engine, and handles the full market lifecycle.
    """

    def __init__(self, poly_config: dict, bot_config: dict):
        self.poly_config = poly_config
        self.bot_config = bot_config

        # Initialize components
        self.strategy = MomentumStrategy(bot_config["strategy"])
        self.engine = TradingEngine(poly_config, bot_config["trading"])
        self.notifications = NotificationService(bot_config.get("notifications", {}))

        # Current market state
        self.current_market: MarketInfo = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._signal_check_thread: threading.Thread = None

    def start(self):
        """Start the bot."""
        logger.info("=" * 60)
        logger.info("  MOMENTUM BOT STARTING")
        logger.info("=" * 60)

        # Load persisted state
        self.engine.load_state()
        self.engine.check_daily_reset()

        # Fetch initial balance
        balance = self.engine.fetch_balance()

        mode = "DRY RUN" if self.engine.dry_run else "LIVE"
        config_summary = (
            f"Mode:       {mode}\n"
            f"Capital:    ${balance:.2f}\n"
            f"Pos size:   ${self.engine.get_position_size():.2f}\n"
            f"Autoscale:  {self.engine.autoscale_enabled} "
            f"({self.engine.autoscale_fraction:.0%})\n"
            f"Max daily loss: ${self.engine.max_daily_loss:.2f}\n"
            f"Momentum:   {self.strategy.momentum_threshold}\n"
            f"Obs window: {self.strategy.observation_window:.0%}\n"
            f"Price range: [{self.strategy.min_entry_price}, "
            f"{self.strategy.max_entry_price}]"
        )
        self.notifications.bot_started(config_summary)

        # Check for unresolved positions from crash recovery
        if self.engine.positions:
            logger.info(f"Found {len(self.engine.positions)} open position(s) from previous session")
            self._resolve_existing_positions()

        # Start the WebSocket client (blocks until stopped)
        client = PolymarketClient(
            on_event=self._on_ws_event,
            on_market_open=self._on_market_open,
            on_market_close=self._on_market_close,
        )

        # Handle shutdown
        def shutdown(sig, frame):
            logger.info(f"Received signal {sig}, shutting down")
            self._stop_event.set()
            client.stop()
            self.engine.save_state()
            sys.exit(0)

        signal.signal(signal.SIGINT, shutdown)
        signal.signal(signal.SIGTERM, shutdown)

        try:
            client.start()  # blocks
        except KeyboardInterrupt:
            shutdown(2, None)

    # ------------------------------------------------------------------
    # WebSocket Event Handlers
    # ------------------------------------------------------------------

    def _on_ws_event(self, data: bytes):
        """Handle incoming WebSocket event (book or trade)."""
        from shared.polymarket_pb2 import MarketEvent

        try:
            event = MarketEvent()
            event.ParseFromString(data)

            if event.HasField("book"):
                self._handle_book_event(event.book)
        except Exception as e:
            logger.error(f"Failed to process event: {e}")

    def _handle_book_event(self, book):
        """Process a book snapshot for the strategy."""
        if not self.current_market:
            return

        slug = self.current_market.slug
        asset_id = book.asset_id

        # Only process ticks for tokens in the current market
        if asset_id not in self.current_market.token_ids:
            return

        # Extract best bid/ask from the protobuf
        # IMPORTANT: bids arrive ascending (worst to best) from WS,
        # so best bid = max price, best ask = min price
        n_bids = len(book.bids)
        n_asks = len(book.asks)
        bid_price = max((float(b.price) for b in book.bids), default=0.0) if n_bids > 0 else 0.0
        ask_price = min((float(a.price) for a in book.asks), default=0.0) if n_asks > 0 else 0.0

        # Log first tick per market for visibility
        tick_count = len(self.strategy._ticks.get(slug, []))
        if tick_count < 2:
            is_up = asset_id == self.current_market.up_token_id
            label = "UP" if is_up else "DN"
            logger.info(
                f"[{slug}] First tick {label}: "
                f"bid={bid_price:.4f} ask={ask_price:.4f} "
                f"mid={((bid_price+ask_price)/2):.4f}"
            )

        if bid_price <= 0 or ask_price <= 0:
            return

        timestamp = int(book.exchange_timestamp)

        self.strategy.on_book_tick(slug, asset_id, timestamp, bid_price, ask_price)

    def _on_market_open(self, market: MarketInfo):
        """Called when a new 5-min market window opens."""
        with self._lock:
            self.current_market = market

        logger.info(f"Market opened: {market.slug}")

        # Parse start/end timestamps
        slug_ts = int(market.slug.split("-")[-1])
        start_ts_ms = slug_ts * 1000
        end_ts_str = market.end_time.replace("Z", "+00:00")
        end_ts = int(datetime.fromisoformat(end_ts_str).timestamp())
        end_ts_ms = end_ts * 1000

        # Reset strategy for new market
        self.strategy.reset_market(
            slug=market.slug,
            start_ts_ms=start_ts_ms,
            end_ts_ms=end_ts_ms,
            up_token_id=market.up_token_id,
            down_token_id=market.down_token_id,
        )

        # Start signal checking thread
        self._start_signal_checker(market)

    def _on_market_close(self, market: MarketInfo):
        """Called when a market window expires."""
        logger.info(f"Market closed: {market.slug}")

        # Handle resolution in a background thread so the bot keeps trading
        if self.engine.has_position(market.slug):
            t = threading.Thread(
                target=self._handle_resolution,
                args=(market,),
                daemon=True,
            )
            t.start()

        # Cleanup strategy state
        self.strategy.cleanup_market(market.slug)

    # ------------------------------------------------------------------
    # Signal Checking Loop
    # ------------------------------------------------------------------

    def _start_signal_checker(self, market: MarketInfo):
        """Background thread that checks for signals at the observation cutoff."""

        def checker():
            slug = market.slug
            up = market.up_token_id
            down = market.down_token_id

            while not self._stop_event.is_set():
                time.sleep(1)  # check every second

                # Check if we should trade
                self.engine.check_daily_reset()
                if self.engine.is_daily_loss_exceeded():
                    return

                signal_result = self.strategy.check_signal(slug, up, down)

                if signal_result is not None:
                    self._execute_signal(market, signal_result)
                    return

                # If signal already generated (even if None), stop checking
                if self.strategy._signal_generated.get(slug, False):
                    return

        self._signal_check_thread = threading.Thread(target=checker, daemon=True)
        self._signal_check_thread.start()

    def _execute_signal(self, market: MarketInfo, sig):
        """Execute a trading signal."""
        # Check daily loss again
        if self.engine.is_daily_loss_exceeded():
            self.notifications.daily_loss_hit(
                abs(self.engine.daily_stats.pnl),
                self.engine.max_daily_loss,
            )
            return

        # Don't open a duplicate position for the same market
        if self.engine.has_position(market.slug):
            logger.warning(f"Already have position for {market.slug}, skipping")
            return

        # Calculate position size
        size = self.engine.get_position_size()
        balance = self.engine.fetch_balance()

        if size > balance:
            logger.warning(f"Position size ${size:.2f} > balance ${balance:.2f}")
            size = balance * 0.9  # use 90% of remaining

        if size < 1.0:
            logger.warning("Insufficient balance to trade")
            return

        # Check price bounds against the SIGNAL's observed price
        # (the price at observation time, not the current CLOB price)
        if sig.entry_ask < self.strategy.min_entry_price:
            logger.info(f"Signal ask {sig.entry_ask:.4f} below min "
                        f"{self.strategy.min_entry_price}, skipping")
            return
        if sig.entry_ask > self.strategy.max_entry_price:
            logger.info(f"Signal ask {sig.entry_ask:.4f} above max "
                        f"{self.strategy.max_entry_price}, skipping")
            return

        # In live mode, get fresh ask from CLOB for actual execution price
        # and apply a slippage guard (max 20% above signal price)
        max_slippage = 0.20
        if not self.engine.dry_run:
            fresh_ask = self.engine.get_best_ask(sig.token_to_buy)
            if fresh_ask and fresh_ask > 0:
                slippage = (fresh_ask - sig.entry_ask) / sig.entry_ask
                if slippage > max_slippage:
                    logger.info(
                        f"Slippage too high: signal={sig.entry_ask:.4f} "
                        f"fresh={fresh_ask:.4f} ({slippage:.1%}), skipping")
                    return
                entry_price = fresh_ask
                logger.info(f"Fresh ask: {fresh_ask:.4f} (slippage: {slippage:+.1%})")
            else:
                entry_price = sig.entry_ask
        else:
            entry_price = sig.entry_ask

        shares = size / entry_price

        # Place the order (pass entry_price for the limit order)
        order_id = self.engine.place_market_buy(sig.token_to_buy, size, entry_price)

        if order_id is None:
            logger.error("Order placement failed!")
            self.notifications.bot_error(
                f"Order failed for {market.slug}: {sig.direction}")
            return

        # Record position
        self.engine.open_position(
            slug=market.slug,
            condition_id=market.condition_id,
            signal=sig.direction,
            token_id=sig.token_to_buy,
            entry_price=entry_price,
            shares=shares,
            cost=size,
            order_id=order_id,
            up_token_id=market.up_token_id,
            down_token_id=market.down_token_id,
        )

        self.notifications.trade_opened(
            slug=market.slug,
            signal=sig.direction,
            token_id=sig.token_to_buy,
            price=entry_price,
            size=size,
            shares=shares,
        )

    # ------------------------------------------------------------------
    # Resolution & Redemption
    # ------------------------------------------------------------------

    def _handle_resolution(self, market: MarketInfo):
        """Wait for market resolution and close position (runs in background thread)."""
        slug = market.slug
        pos = self.engine.get_position(slug)
        if not pos:
            return

        logger.info(f"Waiting for resolution of {slug}...")

        # Poll for resolution — BTC 5-min markets can take 5-10 min to
        # show as resolved in the Gamma API
        resolution = None
        for attempt in range(180):  # up to 15 minutes of polling
            if self._stop_event.is_set():
                return
            time.sleep(5)
            resolution = self.engine.check_resolution(slug)
            if resolution:
                break
            if attempt > 0 and attempt % 12 == 0:
                logger.info(f"Still waiting for {slug} resolution... "
                            f"({attempt * 5}s elapsed)")

        if not resolution:
            logger.error(f"Resolution timeout for {slug} after 15 min")
            self.notifications.bot_error(
                f"Resolution timeout: {slug}. Position still open!")
            return

        # Determine outcome for our token
        outcome = resolution["token_outcomes"].get(pos.token_id, 0.0)

        # Close position
        record = self.engine.close_position(slug, outcome)

        # Redeem contracts
        redeem_success = self.engine.redeem_positions(pos.condition_id)
        if outcome > 0:
            self.notifications.redemption_result(
                slug, record.payout, redeem_success)

        # Notify result
        self.notifications.trade_result(
            slug=record.slug,
            signal=record.signal,
            outcome="WIN" if record.profit > 0 else "LOSS",
            profit=record.profit,
            capital=self.engine.fetch_balance(),
            daily_pnl=self.engine.daily_stats.pnl,
            total_pnl=self.engine.total_pnl,
        )

        # Check daily loss
        if self.engine.is_daily_loss_exceeded():
            self.notifications.daily_loss_hit(
                abs(self.engine.daily_stats.pnl),
                self.engine.max_daily_loss,
            )

    def _resolve_existing_positions(self):
        """Handle positions left open from a previous session."""
        # Take a snapshot of slugs to resolve (dict may change during iteration)
        slugs = list(self.engine.positions.keys())
        for slug in slugs:
            pos = self.engine.get_position(slug)
            if not pos:
                continue

            resolution = self.engine.check_resolution(slug)
            if resolution:
                outcome = resolution["token_outcomes"].get(pos.token_id, 0.0)
                record = self.engine.close_position(slug, outcome)
                self.engine.redeem_positions(pos.condition_id)
                self.notifications.trade_result(
                    slug=record.slug,
                    signal=record.signal,
                    outcome="WIN" if record.profit > 0 else "LOSS",
                    profit=record.profit,
                    capital=self.engine.fetch_balance(),
                    daily_pnl=self.engine.daily_stats.pnl,
                    total_pnl=self.engine.total_pnl,
                )
            else:
                logger.warning(f"Position {slug} still unresolved — "
                               f"spawning background resolver")
                # Spawn background thread with a fake MarketInfo for resolution
                import types
                fake_market = types.SimpleNamespace(
                    slug=slug, condition_id=pos.condition_id)
                t = threading.Thread(
                    target=self._handle_resolution,
                    args=(fake_market,),
                    daemon=True,
                )
                t.start()


# ------------------------------------------------------------------
# Entry Point
# ------------------------------------------------------------------

def load_configs():
    """Load polymarket credentials and bot configuration."""
    # Polymarket API config
    with open("config.json") as f:
        poly_config = json.load(f)["polymarket"]

    # Bot-specific config
    bot_config_path = Path("bot_config.json")
    if bot_config_path.exists():
        with open(bot_config_path) as f:
            bot_config = json.load(f)
    else:
        # Defaults
        bot_config = {
            "strategy": {
                "observation_window": 0.50,
                "momentum_threshold": 0.03,
                "min_entry_price": 0.0,
                "max_entry_price": 0.84,
                "min_book_snapshots": 20,
            },
            "trading": {
                "position_size": 10.0,
                "autoscale_enabled": False,
                "autoscale_fraction": 0.05,
                "max_daily_loss": 200.0,
                "dry_run": True,
            },
            "notifications": {"backend": "console"},
        }

    return poly_config, bot_config


def main():
    poly_config, bot_config = load_configs()

    # CLI overrides
    if "--dry-run" in sys.argv:
        bot_config["trading"]["dry_run"] = True
    elif "--live" in sys.argv:
        bot_config["trading"]["dry_run"] = False

    bot = MomentumBot(poly_config, bot_config)
    bot.start()


if __name__ == "__main__":
    main()
