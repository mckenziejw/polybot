import logging
import signal
import sys

from src.consumer import PersistenceConsumer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    consumer = PersistenceConsumer()

    def handle_signal(sig, frame):
        logger.info(f"Received signal {sig}, shutting down")
        consumer.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    logger.info("Starting persistence")
    consumer.start()


if __name__ == "__main__":
    main()