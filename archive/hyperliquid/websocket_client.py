"""
Hyperliquid WebSocket subscription manager.

Maintains a persistent connection to the Hyperliquid WebSocket API with
auto-reconnect and keepalive pings. Dispatches parsed messages to
user-provided callbacks by channel type.

Usage:
    from hyperliquid.websocket_client import HyperliquidWS

    def on_l2(data):
        print(f"L2 update: {data['coin']}")

    ws = HyperliquidWS()
    ws.on("l2Book", on_l2)
    ws.subscribe_l2("BTC")
    ws.subscribe_l2("ETH")
    ws.run_forever()  # blocks
"""

import json
import logging
import threading
import time
from collections import defaultdict
from typing import Any, Callable

import websocket

from .config import Config, MAINNET_WS, TESTNET_WS

logger = logging.getLogger(__name__)


class HyperliquidWS:
    """
    WebSocket client for Hyperliquid real-time data.

    Features:
    - Subscribe/unsubscribe to channels (l2Book, trades, allMids, etc.)
    - Callback dispatch by channel type
    - Auto-reconnect on disconnect
    - Keepalive ping every 15 seconds
    """

    PING_INTERVAL = 15  # seconds

    def __init__(
        self,
        on_message: Callable[[dict], None] | None = None,
        testnet: bool = True,
        config: Config | None = None,
    ):
        """
        Args:
            on_message: Optional catch-all callback for every message.
            testnet: Use testnet endpoint (default True).
            config: Full Config object (overrides testnet param).
        """
        if config:
            self._ws_url = config.ws_url
        else:
            self._ws_url = TESTNET_WS if testnet else MAINNET_WS

        self._on_message_all = on_message
        self._callbacks: dict[str, list[Callable]] = defaultdict(list)
        self._subscriptions: list[dict] = []
        self._ws: websocket.WebSocketApp | None = None
        self._thread: threading.Thread | None = None
        self._ping_thread: threading.Thread | None = None
        self._connected = threading.Event()
        self._should_run = False
        self._reconnect_delay = 1.0  # seconds, doubles on repeated failure
        self._max_reconnect_delay = 60.0

    # ------------------------------------------------------------------
    # Public API — callback registration
    # ------------------------------------------------------------------

    def on(self, channel: str, callback: Callable[[dict], None]) -> None:
        """
        Register a callback for a specific channel type.

        Channel types: "l2Book", "trades", "allMids", "userFills",
        "orderUpdates", "candle", "pong"
        """
        self._callbacks[channel].append(callback)

    # ------------------------------------------------------------------
    # Public API — subscriptions
    # ------------------------------------------------------------------

    def subscribe_l2(self, coin: str) -> None:
        """Subscribe to L2 orderbook updates for a coin."""
        self._subscribe({"type": "l2Book", "coin": coin})

    def subscribe_trades(self, coin: str) -> None:
        """Subscribe to trade feed for a coin."""
        self._subscribe({"type": "trades", "coin": coin})

    def subscribe_all_mids(self) -> None:
        """Subscribe to all mid-price updates."""
        self._subscribe({"type": "allMids"})

    def subscribe_candle(self, coin: str, interval: str = "1m") -> None:
        """Subscribe to candle updates for a coin."""
        self._subscribe({"type": "candle", "coin": coin, "interval": interval})

    def subscribe_user_fills(self, address: str) -> None:
        """Subscribe to fill notifications for a user address."""
        self._subscribe({"type": "userFills", "user": address})

    def subscribe_order_updates(self, address: str) -> None:
        """Subscribe to order status updates for a user address."""
        self._subscribe({"type": "orderUpdates", "user": address})

    def unsubscribe_l2(self, coin: str) -> None:
        """Unsubscribe from L2 orderbook updates for a coin."""
        self._unsubscribe({"type": "l2Book", "coin": coin})

    def unsubscribe_trades(self, coin: str) -> None:
        """Unsubscribe from trade feed for a coin."""
        self._unsubscribe({"type": "trades", "coin": coin})

    def unsubscribe_all_mids(self) -> None:
        """Unsubscribe from all mid-price updates."""
        self._unsubscribe({"type": "allMids"})

    # ------------------------------------------------------------------
    # Public API — lifecycle
    # ------------------------------------------------------------------

    def run_forever(self) -> None:
        """
        Connect and block forever, reconnecting on disconnection.

        Call this from your main thread. Use Ctrl+C to stop.
        """
        self._should_run = True
        while self._should_run:
            try:
                self._connect_and_run()
            except Exception:
                logger.exception("WebSocket connection failed")

            if not self._should_run:
                break

            logger.info(
                "Reconnecting in %.1f seconds...", self._reconnect_delay
            )
            time.sleep(self._reconnect_delay)
            self._reconnect_delay = min(
                self._reconnect_delay * 2, self._max_reconnect_delay
            )

    def run_in_background(self) -> threading.Thread:
        """
        Start the WebSocket connection in a daemon thread.

        Returns the thread handle.
        """
        self._should_run = True
        self._thread = threading.Thread(target=self.run_forever, daemon=True)
        self._thread.start()
        return self._thread

    def stop(self) -> None:
        """Gracefully stop the WebSocket connection."""
        self._should_run = False
        if self._ws:
            self._ws.close()

    def wait_connected(self, timeout: float = 10.0) -> bool:
        """Block until connected or timeout. Returns True if connected."""
        return self._connected.wait(timeout)

    # ------------------------------------------------------------------
    # Internal — connection management
    # ------------------------------------------------------------------

    def _connect_and_run(self) -> None:
        """Create a WebSocketApp and run it (blocks until disconnect)."""
        self._connected.clear()

        self._ws = websocket.WebSocketApp(
            self._ws_url,
            on_open=self._handle_open,
            on_message=self._handle_message,
            on_error=self._handle_error,
            on_close=self._handle_close,
        )

        # run_forever blocks until the connection drops
        self._ws.run_forever(
            ping_interval=0,  # we handle our own pings
        )

    def _handle_open(self, ws: websocket.WebSocketApp) -> None:
        """Called when WebSocket connects. Resubscribes and starts pings."""
        logger.info("WebSocket connected to %s", self._ws_url)
        self._reconnect_delay = 1.0  # reset backoff
        self._connected.set()

        # Resubscribe to all active subscriptions
        for sub in self._subscriptions:
            self._send({"method": "subscribe", "subscription": sub})

        # Start keepalive ping thread
        self._ping_thread = threading.Thread(target=self._ping_loop, daemon=True)
        self._ping_thread.start()

    def _handle_message(self, ws: websocket.WebSocketApp, raw: str) -> None:
        """Parse and dispatch an incoming message."""
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Received non-JSON message: %s", raw[:200])
            return

        channel = msg.get("channel")

        # Dispatch to catch-all callback
        if self._on_message_all:
            try:
                self._on_message_all(msg)
            except Exception:
                logger.exception("Error in catch-all message handler")

        # Dispatch to channel-specific callbacks
        if channel and channel in self._callbacks:
            data = msg.get("data", msg)
            for cb in self._callbacks[channel]:
                try:
                    cb(data)
                except Exception:
                    logger.exception(
                        "Error in callback for channel '%s'", channel
                    )

    def _handle_error(self, ws: websocket.WebSocketApp, error: Exception) -> None:
        """Called on WebSocket error."""
        logger.error("WebSocket error: %s", error)

    def _handle_close(
        self,
        ws: websocket.WebSocketApp,
        close_status: int | None,
        close_msg: str | None,
    ) -> None:
        """Called when WebSocket closes."""
        self._connected.clear()
        logger.info(
            "WebSocket closed (status=%s, msg=%s)", close_status, close_msg
        )

    # ------------------------------------------------------------------
    # Internal — messaging
    # ------------------------------------------------------------------

    def _send(self, msg: dict) -> None:
        """Send a JSON message if connected."""
        if self._ws and self._connected.is_set():
            self._ws.send(json.dumps(msg))

    def _subscribe(self, subscription: dict) -> None:
        """Track and send a subscription message."""
        # Avoid duplicates
        if subscription not in self._subscriptions:
            self._subscriptions.append(subscription)

        self._send({"method": "subscribe", "subscription": subscription})

    def _unsubscribe(self, subscription: dict) -> None:
        """Remove tracking and send an unsubscribe message."""
        if subscription in self._subscriptions:
            self._subscriptions.remove(subscription)

        self._send({"method": "unsubscribe", "subscription": subscription})

    def _ping_loop(self) -> None:
        """Send pings at regular intervals to keep the connection alive."""
        while self._should_run and self._connected.is_set():
            try:
                self._send({"method": "ping"})
            except Exception:
                logger.debug("Ping failed (connection may be closing)")
                break
            time.sleep(self.PING_INTERVAL)
