"""
user_channel.py

WebSocket connection to the Polymarket user channel.
Delivers fill notifications in real time so the market maker
can react to order state changes without polling.

The user channel requires L2 authentication headers on the
initial subscription message, not on the HTTP upgrade itself.
"""

import json
import logging
import threading
import time
from typing import Callable

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import ApiCreds
from websocket import WebSocketApp

logger = logging.getLogger(__name__)

WS_URL       = "wss://ws-subscriptions-clob.polymarket.com/ws/user"
PING_INTERVAL = 10


class UserChannel:
    """
    Subscribes to the Polymarket user channel and fires callbacks
    on order fill events.

    Runs in a background thread — non-blocking from the caller's perspective.
    """

    def __init__(
        self,
        private_key: str,
        api_key: str,
        api_secret: str,
        api_passphrase: str,
        on_order_update: Callable[[dict], None],
        condition_ids: list[str] | None = None,
    ):
        """
        Args:
            on_order_update: called with the raw event dict on any order
                             status change (trade, order update, etc.)
            condition_ids:   markets to subscribe to; can be updated via
                             subscribe() after start()
        """
        _client = ClobClient(
            host="https://clob.polymarket.com",
            chain_id=137,
            key=private_key,
            creds=ApiCreds(
                api_key=api_key,
                api_secret=api_secret,
                api_passphrase=api_passphrase,
            ),
        )
        _client.set_api_creds(_client.create_or_derive_api_creds())
        self._creds         = _client.creds
        self._condition_ids = condition_ids or []
        self.on_order_update = on_order_update

        self._ws: WebSocketApp | None = None
        self._stop_event  = threading.Event()
        self._ws_thread:  threading.Thread | None = None
        self._ping_thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def start(self):
        """Start the user channel in a background thread."""
        self._ws_thread = threading.Thread(
            target=self._run, daemon=True, name="user-channel"
        )
        self._ws_thread.start()
        logger.info("User channel thread started")

    def stop(self):
        """Shut down the websocket cleanly."""
        self._stop_event.set()
        if self._ws:
            self._ws.close()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run(self):
        while not self._stop_event.is_set():
            self._ws = WebSocketApp(
                WS_URL,
                on_open=self._on_open,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close,
            )
            self._ws.run_forever()

            if self._stop_event.is_set():
                break

            logger.warning("User channel disconnected, reconnecting in 5s...")
            time.sleep(5)

    def _build_auth_message(self, condition_ids: list[str]) -> dict:
        """
        Build the user channel subscription message.
        Auth is plain credentials — no HMAC required.
        Subscribes by condition ID, not asset ID.
        """
        return {
            "auth": {
                "apiKey":     self._creds.api_key,
                "secret":     self._creds.api_secret,
                "passphrase": self._creds.api_passphrase,
            },
            "markets": condition_ids,
            "type": "user",
        }

    def subscribe(self, condition_ids: list[str]):
        """Update the list of markets to receive events for."""
        self._condition_ids = condition_ids
        if self._ws:
            msg = {"markets": condition_ids, "operation": "subscribe"}
            self._ws.send(json.dumps(msg))

    def _on_open(self, ws):
        logger.info(f"User channel connected, subscribing to {len(self._condition_ids)} markets")
        auth_msg = self._build_auth_message(self._condition_ids)
        ws.send(json.dumps(auth_msg))
        self._start_ping(ws)

    def _on_message(self, ws, message):
        if message == "PONG":
            return

        try:
            raw = json.loads(message)
            events = raw if isinstance(raw, list) else [raw]

            for event in events:
                event_type = event.get("event_type", "")

                if event_type in ("trade", "order"):
                    self.on_order_update(event)
                else:
                    logger.debug(f"User channel ignored event_type={event_type!r}: {str(event)[:120]}")

        except Exception as e:
            logger.error(f"Failed to parse user channel message: {e}\nRaw: {message[:200]}")

    def _on_error(self, ws, error):
        logger.error(f"User channel error: {error}")

    def _on_close(self, ws, code, msg):
        logger.info(f"User channel closed: {code} {msg}")

    def _start_ping(self, ws):
        def ping():
            while not self._stop_event.is_set():
                try:
                    ws.send("PING")
                except Exception:
                    break
                time.sleep(PING_INTERVAL)

        self._ping_thread = threading.Thread(target=ping, daemon=True, name="user-ping")
        self._ping_thread.start()