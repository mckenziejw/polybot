import json
import logging
import time

import redis

from polymarket_pb2 import MarketEvent
from src.writer import ParquetWriter

logger = logging.getLogger(__name__)

MARKET_EVENTS_STREAM = "polymarket:market_events"
CONTROL_STREAM       = "polymarket:control"
CONSUMER_GROUP       = "polymarket:persistence"
CONSUMER_NAME        = "persistence-1"
BLOCK_MS             = 1000


class PersistenceConsumer:
    """
    Consumes from Redis Streams and writes to Parquet via ParquetWriter.
    Reads from both market_events and control streams.
    """

    def __init__(self, host: str = "localhost", port: int = 6379):
        self.client = redis.Redis(host=host, port=port, decode_responses=False)
        self.writer = ParquetWriter()
        self._running = False
        self._verify_connection()

    def _verify_connection(self):
        try:
            self.client.ping()
            logger.info("Redis connection established")
        except redis.ConnectionError as e:
            raise RuntimeError(f"Could not connect to Redis: {e}")

    def start(self):
        """Bootstrap from current market metadata and start consuming."""
        self._running = True
        self._bootstrap_from_control_stream()
        logger.info("Starting consume loop")
        self._consume_loop()

    def stop(self):
        logger.info("Stopping persistence consumer")
        self._running = False
        self.writer.close()

    # ------------------------------------------------------------------
    # Bootstrap
    # ------------------------------------------------------------------

    def _bootstrap_from_control_stream(self):
        """
        Read the control stream history to determine current market state.
        Handles the case where persistence restarts mid-window.
        """
        try:
            events = self.client.xrange(CONTROL_STREAM, "-", "+")
            last_open_slug   = None
            last_open_fields = None
            last_close_slug  = None

            for _, fields in events:
                event_type = fields.get(b"event_type", b"").decode()
                slug       = fields.get(b"slug", b"").decode()
                if event_type == "market_open":
                    last_open_slug   = slug
                    last_open_fields = fields
                elif event_type == "market_close":
                    last_close_slug = slug

            if last_open_slug and last_open_slug != last_close_slug:
                logger.info(f"Resuming mid-window: {last_open_slug}")
                token_label_map = _parse_token_label_map(last_open_fields) if last_open_fields else {}
                self.writer.open(last_open_slug, token_label_map)
            else:
                logger.info("No active market window found — waiting for market_open")

        except redis.RedisError as e:
            logger.error(f"Failed to bootstrap from control stream: {e}")

    # ------------------------------------------------------------------
    # Consume loop
    # ------------------------------------------------------------------

    def _consume_loop(self):
        """Main loop — reads from both streams using XREADGROUP."""
        streams   = {MARKET_EVENTS_STREAM: ">", CONTROL_STREAM: ">"}
        iteration = 0

        while self._running:
            iteration += 1
            if iteration % 60 == 0:
                try:
                    self.client.ping()
                    logger.info(
                        f"Heartbeat {iteration} — Redis: OK, "
                        f"writer: {self.writer._current_slug}"
                    )
                except redis.RedisError as e:
                    logger.error(f"Redis health check failed: {e}")

            try:
                results = self.client.xreadgroup(
                    groupname=CONSUMER_GROUP,
                    consumername=CONSUMER_NAME,
                    streams=streams,
                    count=100,
                    block=BLOCK_MS,
                )
                if not results:
                    continue

                for stream_name, messages in results:
                    stream = stream_name.decode()
                    for msg_id, fields in messages:
                        try:
                            self._handle_message(stream, fields)
                            self.client.xack(stream_name, CONSUMER_GROUP, msg_id)
                        except Exception as e:
                            logger.error(f"Failed to process message {msg_id}: {e}")

            except redis.RedisError as e:
                logger.error(f"Redis error in consume loop: {e}")
                time.sleep(1)
            except Exception as e:
                logger.error(f"Unexpected error in consume loop: {e}", exc_info=True)
                time.sleep(1)

    # ------------------------------------------------------------------
    # Message handling
    # ------------------------------------------------------------------

    def _handle_message(self, stream: str, fields: dict):
        if stream == CONTROL_STREAM:
            self._handle_control(fields)
        elif stream == MARKET_EVENTS_STREAM:
            self._handle_event(fields)

    def _handle_control(self, fields: dict):
        event_type = fields.get(b"event_type", b"").decode()
        slug       = fields.get(b"slug", b"").decode()

        if event_type == "market_open":
            logger.info(f"market_open: {slug}")
            if self.writer._current_slug:
                logger.info(f"Closing previous window: {self.writer._current_slug}")
                self.writer.close()
            token_label_map = _parse_token_label_map(fields)
            self.writer.open(slug, token_label_map)

        elif event_type == "market_close":
            # Keep writer open — in-flight messages from the closing window
            # may still arrive. Writer rotates on the next market_open.
            logger.info(f"market_close: {slug} — writer stays open until next market_open")

    def _handle_event(self, fields: dict):
        data = fields.get(b"data")
        if not data:
            return

        event = MarketEvent()
        event.ParseFromString(data)

        if event.HasField("book"):
            self.writer.write_book(event.book, event.received_timestamp)
        elif event.HasField("trade"):
            for pc in event.trade.price_changes:
                self.writer.write_trade(event.trade, event.received_timestamp, pc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_token_label_map(fields: dict) -> dict[str, str]:
    """
    Extract {asset_id: "Yes"/"No"} from a market_open control message.

    Tries two encodings in order:
      1. b"token_label_map" — JSON-encoded dict (preferred, most explicit)
      2. b"up_token_id" / b"down_token_id" — individual fields

    Returns empty dict if neither is present (legacy messages without labels).
    token_label will be "" in that case — not a crash, just missing data.
    """
    raw_map = fields.get(b"token_label_map")
    if raw_map:
        try:
            return json.loads(raw_map.decode())
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.warning(f"Failed to decode token_label_map: {e}")

    up_token   = fields.get(b"up_token_id",   b"").decode()
    down_token = fields.get(b"down_token_id", b"").decode()
    result = {}
    if up_token:
        result[up_token]   = "Yes"
    if down_token:
        result[down_token] = "No"

    if not result:
        logger.warning("market_open message has no token label fields — token_label will be empty")

    return result