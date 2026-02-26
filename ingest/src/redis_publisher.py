import logging
import redis

logger = logging.getLogger(__name__)

MARKET_EVENTS_STREAM = "polymarket:market_events"
CONTROL_STREAM = "polymarket:control"
MARKET_META_PREFIX = "market_meta"
STREAM_MAXLEN = 50000


class RedisPublisher:
    """
    Handles all Redis interactions for the Ingest component.
    - Publishes MarketEvent protobuf bytes to the market events stream
    - Publishes control events (market_open, market_close) to the control stream
    - Maintains market metadata in Redis hashes
    """

    def __init__(self, host: str = "localhost", port: int = 6379):
        self.client = redis.Redis(host=host, port=port, decode_responses=False)
        self._verify_connection()

    def _verify_connection(self):
        try:
            self.client.ping()
            logger.info("Redis connection established")
        except redis.ConnectionError as e:
            raise RuntimeError(f"Could not connect to Redis: {e}")

    # ------------------------------------------------------------------
    # Stream publishing
    # ------------------------------------------------------------------

    def publish_event(self, event_type: str, market: str, data: bytes):
        """
        Write a serialized MarketEvent to the market events stream.

        Args:
            event_type: "book" or "price_change" — plaintext for easy debugging
            market:     condition ID of the market
            data:       serialized protobuf MarketEvent bytes
        """
        try:
            self.client.xadd(
                MARKET_EVENTS_STREAM,
                {
                    "event_type": event_type,
                    "market": market,
                    "data": data,
                },
                maxlen=STREAM_MAXLEN,
                approximate=True,
            )
        except redis.RedisError as e:
            logger.error(f"Failed to publish event to stream: {e}")

    def publish_market_open(self, slug: str, condition_id: str, up_token_id: str, down_token_id: str, end_time: str):
        """Publish a market_open control event and update market metadata."""
        try:
            # Write control event
            self.client.xadd(
                CONTROL_STREAM,
                {
                    "event_type": "market_open",
                    "market": condition_id,
                    "slug": slug,
                    "end_time": end_time,
                    "up_token_id": up_token_id,
                    "down_token_id": down_token_id,
                },
            )

            # Update market metadata hash
            self.client.hset(
                f"{MARKET_META_PREFIX}:{condition_id}",
                mapping={
                    "slug": slug,
                    "condition_id": condition_id,
                    "up_token_id": up_token_id,
                    "down_token_id": down_token_id,
                    "end_time": end_time,
                },
            )

            logger.info(f"Published market_open: {slug}")

        except redis.RedisError as e:
            logger.error(f"Failed to publish market_open: {e}")

    def publish_market_close(self, slug: str, condition_id: str):
        """Publish a market_close control event."""
        try:
            self.client.xadd(
                CONTROL_STREAM,
                {
                    "event_type": "market_close",
                    "market": condition_id,
                    "slug": slug,
                },
            )
            logger.info(f"Published market_close: {slug}")

        except redis.RedisError as e:
            logger.error(f"Failed to publish market_close: {e}")

    # ------------------------------------------------------------------
    # Consumer group management
    # ------------------------------------------------------------------

    def ensure_consumer_groups(self):
        """
        Create consumer groups if they don't already exist.
        Safe to call on every startup — silently ignores already-exists errors.
        """
        groups = [
            (MARKET_EVENTS_STREAM, "polymarket:persistence"),
            (MARKET_EVENTS_STREAM, "polymarket:hot_analysis"),
            (CONTROL_STREAM, "polymarket:persistence"),
            (CONTROL_STREAM, "polymarket:hot_analysis"),
        ]

        for stream, group in groups:
            try:
                # Create stream if it doesn't exist yet via MKSTREAM
                self.client.xgroup_create(stream, group, id="0", mkstream=True)
                logger.info(f"Created consumer group: {group} on {stream}")
            except redis.exceptions.ResponseError as e:
                if "BUSYGROUP" in str(e):
                    # Group already exists — this is fine
                    pass
                else:
                    raise