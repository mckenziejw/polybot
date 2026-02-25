import json
import time
import logging
import threading
from datetime import datetime, timezone

import requests
from websocket import WebSocketApp

from polymarket_pb2 import (
    MarketEvent,
    BookSnapshot,
    TradeEvent,
    PriceChange,
    PriceLevel,
    Side,
)

logger = logging.getLogger(__name__)


GAMMA_API = "https://gamma-api.polymarket.com"
WS_URL = "wss://ws-subscriptions-clob.polymarket.com"
PING_INTERVAL = 10  # seconds
MARKET_CHANNEL = "market"


class MarketInfo:
    """Holds metadata for the current active market window."""

    def __init__(self, slug: str, condition_id: str, up_token_id: str, down_token_id: str, end_time: str):
        self.slug = slug
        self.condition_id = condition_id
        self.up_token_id = up_token_id
        self.down_token_id = down_token_id
        self.end_time = end_time

    @property
    def token_ids(self) -> list[str]:
        return [self.up_token_id, self.down_token_id]

    def __str__(self):
        return f"MarketInfo(slug={self.slug}, end_time={self.end_time})"


class PolymarketClient:
    """
    Manages the WebSocket connection to Polymarket CLOB.
    Parses incoming messages into MarketEvent protobufs and
    delivers them via callback. Handles market lifecycle internally.
    """

    def __init__(
        self,
        on_event: callable,
        on_market_open: callable,
        on_market_close: callable,
    ):
        """
        Args:
            on_event: called with a serialized MarketEvent bytes on each message
            on_market_open: called with MarketInfo when a new market window opens
            on_market_close: called with MarketInfo when a market window closes
        """
        self.on_event = on_event
        self.on_market_open = on_market_open
        self.on_market_close = on_market_close

        self.current_market: MarketInfo | None = None
        self.ws: WebSocketApp | None = None
        self._ping_thread: threading.Thread | None = None
        self._market_monitor_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._rotating = threading.Lock()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def start(self):
        """Fetch the current active market and open the WebSocket connection."""
        self.current_market = self._fetch_active_market()
        logger.info(f"Starting with market: {self.current_market}")
        self.on_market_open(self.current_market)
        self._connect(self.current_market)

    def stop(self):
        """Cleanly shut down the client."""
        logger.info("Stopping client")
        self._stop_event.set()
        if self.ws:
            self.ws.close()

    # ------------------------------------------------------------------
    # Market fetching
    # ------------------------------------------------------------------

    def _fetch_active_market(self) -> MarketInfo:
        """Fetch the current active BTC 5-minute market from Gamma API."""
        now_utc = int(datetime.now(timezone.utc).timestamp())
        remainder = now_utc % 300
        slug_time = now_utc - remainder
        slug = f"btc-updown-5m-{slug_time}"

        logger.info(f"Fetching market: {slug}")
        resp = requests.get(f"{GAMMA_API}/markets/slug/{slug}", timeout=10)
        resp.raise_for_status()
        result = resp.json()

        assert result.get("active"), f"Market {slug} is not active"
        assert not result.get("closed"), f"Market {slug} is already closed"

        clob_token_ids = json.loads(result["clobTokenIds"])
        assert len(clob_token_ids) == 2, f"Expected 2 token IDs, got {len(clob_token_ids)}"

        # Determine which token is UP vs DOWN from the outcomes array
        outcomes = json.loads(result.get("outcomes", '["Up", "Down"]'))
        if outcomes[0].lower() == "up":
            up_token_id = clob_token_ids[0]
            down_token_id = clob_token_ids[1]
        else:
            up_token_id = clob_token_ids[1]
            down_token_id = clob_token_ids[0]

        logger.info(f"Token mapping: UP={up_token_id[:20]}..., DN={down_token_id[:20]}...")

        return MarketInfo(
            slug=slug,
            condition_id=result["conditionId"],
            up_token_id=up_token_id,
            down_token_id=down_token_id,
            end_time=result["endDate"],
        )

    def _fetch_next_market(self) -> MarketInfo:
        """Fetch the next market window based on current market's end_time."""
        # Derive next slug time from current market end_time rather than wall clock
        end_ts = int(datetime.fromisoformat(
            self.current_market.end_time.replace("Z", "+00:00")
        ).timestamp())
        next_slug_time = end_ts  # next window starts exactly when current one ends
        slug = f"btc-updown-5m-{next_slug_time}"
        
        logger.info(f"Fetching next market: {slug}")
        for attempt in range(10):
            try:
                resp = requests.get(f"{GAMMA_API}/markets/slug/{slug}", timeout=10)
                resp.raise_for_status()
                result = resp.json()
                if result.get("active") and not result.get("closed"):
                    clob_token_ids = json.loads(result["clobTokenIds"])
                    outcomes = json.loads(result.get("outcomes", '["Up", "Down"]'))
                    if outcomes[0].lower() == "up":
                        up_token_id = clob_token_ids[0]
                        down_token_id = clob_token_ids[1]
                    else:
                        up_token_id = clob_token_ids[1]
                        down_token_id = clob_token_ids[0]
                    return MarketInfo(
                        slug=slug,
                        condition_id=result["conditionId"],
                        up_token_id=up_token_id,
                        down_token_id=down_token_id,
                        end_time=result["endDate"],
                    )
                else:
                    logger.warning(f"Attempt {attempt + 1}: market {slug} not active yet")
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed fetching next market: {e}")
            time.sleep(3)

        raise RuntimeError(f"Could not fetch next active market after 10 attempts: {slug}")

    # ------------------------------------------------------------------
    # WebSocket connection
    # ------------------------------------------------------------------

    def _connect(self, market: MarketInfo):
        """Open a WebSocket connection and subscribe to the given market."""
        url = f"{WS_URL}/ws/{MARKET_CHANNEL}"
        self.ws = WebSocketApp(
            url,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )
        self.ws.run_forever()

    def _on_open(self, ws):
        logger.info(f"WebSocket connected, subscribing to {self.current_market.slug}")
        ws.send(json.dumps({
            "assets_ids": self.current_market.token_ids,
            "type": MARKET_CHANNEL,
        }))
        self._start_ping_thread(ws)
        self._start_market_monitor_thread()

    def _on_message(self, ws, message):
        if message == "PONG":
            return

        received_ts = int(time.time() * 1000)
        try:
            raw = json.loads(message)

            # Messages can arrive as a single object or a batch array
            events = raw if isinstance(raw, list) else [raw]

            for event in events:
                event_type = event.get("event_type")

                if event_type == "book":
                    parsed = self._parse_book(event, received_ts)
                elif event_type == "price_change":
                    parsed = self._parse_trade(event, received_ts)
                elif event_type in ("last_trade_price", "tick_size_change"):
                    # Not needed for order book analysis — silently ignore
                    continue
                else:
                    logger.warning(f"Unknown event_type: {event_type}")
                    continue

                self.on_event(parsed.SerializeToString())

        except Exception as e:
            logger.error(f"Failed to parse message: {e}\nRaw: {message[:200]}")

    def _on_error(self, ws, error):
        logger.error(f"WebSocket error: {error}")

    def _on_close(self, ws, close_status_code, close_msg):
        logger.info(f"WebSocket closed: {close_status_code} {close_msg}")
        self._stop_event.set()

    # ------------------------------------------------------------------
    # Background threads
    # ------------------------------------------------------------------

    def _start_ping_thread(self, ws):
        def ping():
            while not self._stop_event.is_set():
                try:
                    ws.send("PING")
                except Exception as e:
                    logger.warning(f"Ping failed: {e}")
                    break
                time.sleep(PING_INTERVAL)

        self._ping_thread = threading.Thread(target=ping, daemon=True)
        self._ping_thread.start()

    def _start_market_monitor_thread(self):
        """Monitor for market window expiry and handle rotation."""
        def monitor():
            while not self._stop_event.is_set():
                time.sleep(5)
                if self._is_market_expired():
                    if not self._rotating.acquire(blocking=False):
                        # Another rotation is already in progress
                        logger.debug("Rotation already in progress, skipping")
                        return
                    try:
                        logger.info(f"Market window closing: {self.current_market.slug}")
                        self.on_market_close(self.current_market)
                        self._rotate_market()
                    finally:
                        self._rotating.release()
                    return

        self._market_monitor_thread = threading.Thread(target=monitor, daemon=True)
        self._market_monitor_thread.start()

    def _is_market_expired(self) -> bool:
        if not self.current_market:
            return False
        now_utc = int(datetime.now(timezone.utc).timestamp())
        # Parse end_time e.g. "2026-02-17T13:35:00Z"
        end_ts = int(datetime.fromisoformat(
            self.current_market.end_time.replace("Z", "+00:00")
        ).timestamp())
        return now_utc >= end_ts

    def _rotate_market(self):
        """Fetch next market, resubscribe, and notify listeners."""
        try:
            next_market = self._fetch_next_market()
            self.current_market = next_market
            logger.info(f"Rotating to: {next_market}")
            self.on_market_open(next_market)

            # Resubscribe on existing connection
            if self.ws:
                self.ws.send(json.dumps({
                    "assets_ids": next_market.token_ids,
                    "type": MARKET_CHANNEL,
                    "operation": "subscribe",
                }))
            self._start_market_monitor_thread()

        except Exception as e:
            logger.error(f"Market rotation failed: {e}")
            self.stop()

    # ------------------------------------------------------------------
    # Protobuf parsing
    # ------------------------------------------------------------------

    def _parse_book(self, raw: dict, received_ts: int) -> MarketEvent:
        snapshot = BookSnapshot(
            market=raw["market"],
            asset_id=raw["asset_id"],
            exchange_timestamp=int(raw["timestamp"]),
            hash=raw["hash"],
            bids=[
                PriceLevel(price=float(b["price"]), size=float(b["size"]))
                for b in raw.get("bids", [])
            ],
            asks=[
                PriceLevel(price=float(a["price"]), size=float(a["size"]))
                for a in raw.get("asks", [])
            ],
            tick_size=float(raw.get("tick_size", 0)),
            last_trade_price=float(raw.get("last_trade_price", 0)),
        )
        return MarketEvent(received_timestamp=received_ts, book=snapshot)

    def _parse_trade(self, raw: dict, received_ts: int) -> MarketEvent:
        def parse_side(side_str: str) -> Side:
            return Side.BUY if side_str == "BUY" else Side.SELL if side_str == "SELL" else Side.SIDE_UNKNOWN

        price_changes = [
            PriceChange(
                asset_id=pc["asset_id"],
                price=float(pc["price"]),
                size=float(pc["size"]),
                side=parse_side(pc["side"]),
                hash=pc["hash"],
                best_bid=float(pc["best_bid"]),
                best_ask=float(pc["best_ask"]),
            )
            for pc in raw.get("price_changes", [])
        ]
        trade = TradeEvent(
            market=raw["market"],
            exchange_timestamp=int(raw["timestamp"]),
            price_changes=price_changes,
        )
        return MarketEvent(received_timestamp=received_ts, trade=trade)