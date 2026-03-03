"""
market_channel.py

Subscribes to the Polymarket market channel for real-time orderbook
updates. Maintains current mid-price for mark-to-market valuation.
Only tracks the YES token -- NO is always 1 - YES in a binary market.
"""

import json
import logging
import threading
import time
from typing import Callable

from websocket import WebSocketApp
from execution.market_info import MarketInfo

logger = logging.getLogger(__name__)

WS_URL        = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
PING_INTERVAL = 10
MAX_SPREAD    = 0.10


class MarketChannel:
    def __init__(self):
        self._mid_price: float | None = None
        self._best_bid:  float | None = None
        self._best_ask:  float | None = None
        self._asset_id:  str | None   = None
        self._lock       = threading.Lock()
        self._stop_event = threading.Event()
        self._ws:          WebSocketApp | None = None
        self._ws_thread:   threading.Thread | None = None
        self._ping_thread: threading.Thread | None = None

    @property
    def mid_price(self) -> float | None:
        with self._lock:
            return self._mid_price

    @property
    def best_bid(self) -> float | None:
        with self._lock:
            return self._best_bid

    @property
    def best_ask(self) -> float | None:
        with self._lock:
            return self._best_ask

    def start(self, market: MarketInfo):
        self._asset_id = market.up_token_id
        self._ws_thread = threading.Thread(
            target=self._run, daemon=True, name="market-channel"
        )
        self._ws_thread.start()
        logger.info(f"Market channel started for {market.slug}")

    def rotate(self, market: MarketInfo):
        with self._lock:
            self._asset_id  = market.up_token_id
            self._mid_price = None
            self._best_bid  = None
            self._best_ask  = None

        # Only send if the socket is actually open — if not, _on_open will
        # pick up the new _asset_id when the connection completes
        ws = self._ws
        if ws and ws.sock and ws.sock.connected:
            ws.send(json.dumps({
                "assets_ids": [market.up_token_id],
                "type": "market",
            }))
        logger.info(f"Market channel rotated to {market.slug}")

    def stop(self):
        self._stop_event.set()
        if self._ws:
            self._ws.close()

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
            logger.warning("Market channel disconnected, reconnecting in 3s...")
            time.sleep(3)

    def _on_open(self, ws):
        asset_id = self._asset_id
        if not asset_id:
            return
        ws.send(json.dumps({"assets_ids": [asset_id], "type": "market"}))
        self._start_ping(ws)
        logger.info(f"Market channel subscribed to {asset_id[:20]}...")

    def _on_message(self, ws, message):
        if message == "PONG":
            return
        try:
            raw    = json.loads(message)
            events = raw if isinstance(raw, list) else [raw]
            for event in events:
                et = event.get("event_type", "")
                if et == "book":
                    self._handle_book(event)
                elif et == "price_change":
                    self._handle_price_change(event)
        except Exception as e:
            logger.error(f"Market channel parse error: {e}")

    def _handle_book(self, event: dict):
        bids = event.get("bids", [])
        asks = event.get("asks", [])
        best_bid = max((float(b["price"]) for b in bids if float(b["size"]) > 0), default=None)
        best_ask = min((float(a["price"]) for a in asks if float(a["size"]) > 0), default=None)
        self._update_mid(best_bid, best_ask)

    def _handle_price_change(self, event: dict):
        for pc in event.get("price_changes", []):
            if pc.get("asset_id") != self._asset_id:
                continue
            bid = float(pc["best_bid"]) if pc.get("best_bid") else None
            ask = float(pc["best_ask"]) if pc.get("best_ask") else None
            self._update_mid(bid, ask)

    def _update_mid(self, bid: float | None, ask: float | None):
        if bid is None or ask is None or ask <= bid:
            return
        if ask - bid > MAX_SPREAD:
            return
        with self._lock:
            self._best_bid  = bid
            self._best_ask  = ask
            self._mid_price = (bid + ask) / 2.0

    def _on_error(self, ws, error):
        logger.error(f"Market channel error: {error}")

    def _on_close(self, ws, code, msg):
        logger.info(f"Market channel closed: {code} {msg}")

    def _start_ping(self, ws):
        def ping():
            while not self._stop_event.is_set():
                try:
                    ws.send("PING")
                except Exception:
                    break
                time.sleep(PING_INTERVAL)
        self._ping_thread = threading.Thread(target=ping, daemon=True, name="market-ping")
        self._ping_thread.start()