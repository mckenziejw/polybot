import logging
import signal
import sys
from polymarket_pb2 import MarketEvent
from src.market_client import PolymarketClient, MarketInfo
from src.redis_publisher import RedisPublisher

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    publisher = RedisPublisher()
    publisher.ensure_consumer_groups()

    def on_event(data: bytes):
        # Deserialize just enough to extract routing metadata
        # without passing raw dicts around
        event = MarketEvent()
        event.ParseFromString(data)

        if event.HasField("book"):
            publisher.publish_event("book", event.book.market, data)
        elif event.HasField("trade"):
            publisher.publish_event("price_change", event.trade.market, data)

    def on_market_open(market: MarketInfo):
        publisher.publish_market_open(
            slug=market.slug,
            condition_id=market.condition_id,
            up_token_id=market.up_token_id,
            down_token_id=market.down_token_id,
            end_time=market.end_time,
        )

    def on_market_close(market: MarketInfo):
        publisher.publish_market_close(
            slug=market.slug,
            condition_id=market.condition_id,
        )

    client = PolymarketClient(
        on_event=on_event,
        on_market_open=on_market_open,
        on_market_close=on_market_close,
    )

    # Handle clean shutdown on SIGINT/SIGTERM
    def handle_signal(sig, frame):
        logger.info(f"Received signal {sig}, shutting down")
        client.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    logger.info("Starting ingest")
    client.start()


if __name__ == "__main__":
    main()