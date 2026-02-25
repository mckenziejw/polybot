import json
import logging
import os
from datetime import datetime, timezone
import threading
import time

import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

DATA_DIR = os.environ.get("DATA_DIR", "./data")
FLUSH_INTERVAL_ROWS = 500


class ParquetWriter:
    """
    Writes BookSnapshot and TradeEvent records to per-window Parquet files.
    Buffers rows in memory and flushes in batches for efficiency.
    Rotates files on market_close / market_open control events.
    """

    BOOK_SCHEMA = pa.schema([
        pa.field("exchange_timestamp", pa.int64()),
        pa.field("received_timestamp", pa.int64()),
        pa.field("market", pa.string()),
        pa.field("asset_id", pa.string()),
        pa.field("mid_price", pa.float64()),
        pa.field("spread", pa.float64()),
        pa.field("book_imbalance", pa.float64()),
        pa.field("bid_price_1", pa.float64()),
        pa.field("bid_size_1", pa.float64()),
        pa.field("bid_price_2", pa.float64()),
        pa.field("bid_size_2", pa.float64()),
        pa.field("bid_price_3", pa.float64()),
        pa.field("bid_size_3", pa.float64()),
        pa.field("bid_price_4", pa.float64()),
        pa.field("bid_size_4", pa.float64()),
        pa.field("bid_price_5", pa.float64()),
        pa.field("bid_size_5", pa.float64()),
        pa.field("bid_price_6", pa.float64()),
        pa.field("bid_size_6", pa.float64()),
        pa.field("bid_price_7", pa.float64()),
        pa.field("bid_size_7", pa.float64()),
        pa.field("bid_price_8", pa.float64()),
        pa.field("bid_size_8", pa.float64()),
        pa.field("bid_price_9", pa.float64()),
        pa.field("bid_size_9", pa.float64()),
        pa.field("bid_price_10", pa.float64()),
        pa.field("bid_size_10", pa.float64()),
        pa.field("ask_price_1", pa.float64()),
        pa.field("ask_size_1", pa.float64()),
        pa.field("ask_price_2", pa.float64()),
        pa.field("ask_size_2", pa.float64()),
        pa.field("ask_price_3", pa.float64()),
        pa.field("ask_size_3", pa.float64()),
        pa.field("ask_price_4", pa.float64()),
        pa.field("ask_size_4", pa.float64()),
        pa.field("ask_price_5", pa.float64()),
        pa.field("ask_size_5", pa.float64()),
        pa.field("ask_price_6", pa.float64()),
        pa.field("ask_size_6", pa.float64()),
        pa.field("ask_price_7", pa.float64()),
        pa.field("ask_size_7", pa.float64()),
        pa.field("ask_price_8", pa.float64()),
        pa.field("ask_size_8", pa.float64()),
        pa.field("ask_price_9", pa.float64()),
        pa.field("ask_size_9", pa.float64()),
        pa.field("ask_price_10", pa.float64()),
        pa.field("ask_size_10", pa.float64()),
        pa.field("hash", pa.string()),
        pa.field("full_bids", pa.string()),
        pa.field("full_asks", pa.string()),
    ])

    TRADE_SCHEMA = pa.schema([
        pa.field("exchange_timestamp", pa.int64()),
        pa.field("received_timestamp", pa.int64()),
        pa.field("market", pa.string()),
        pa.field("asset_id", pa.string()),
        pa.field("side", pa.string()),
        pa.field("trade_price", pa.float64()),
        pa.field("size", pa.float64()),
        pa.field("best_bid", pa.float64()),
        pa.field("best_ask", pa.float64()),
        pa.field("mid_price", pa.float64()),
        pa.field("spread", pa.float64()),
        pa.field("hash", pa.string()),
    ])

    def __init__(self):
        self._current_slug: str | None = None
        self._book_buffer: list[dict] = []
        self._trade_buffer: list[dict] = []
        self._book_writer: pq.ParquetWriter | None = None
        self._trade_writer: pq.ParquetWriter | None = None
        os.makedirs(f"{DATA_DIR}/book_snapshots", exist_ok=True)
        os.makedirs(f"{DATA_DIR}/trade_events", exist_ok=True)
        self._flush_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open(self, slug: str):
        """Open Parquet writers for a new market window."""
        if self._current_slug:
            logger.warning(f"open() called while {self._current_slug} still active — closing first")
            self.close()

        self._current_slug = slug
        book_path = f"{DATA_DIR}/book_snapshots/{slug}.parquet"
        trade_path = f"{DATA_DIR}/trade_events/{slug}.parquet"

        self._book_writer = pq.ParquetWriter(
            book_path,
            self.BOOK_SCHEMA,
            compression="snappy",
        )
        self._trade_writer = pq.ParquetWriter(
            trade_path,
            self.TRADE_SCHEMA,
            compression="snappy",
        )
        logger.info(f"Opened Parquet writers for {slug}")
        self._start_flush_timer()

    def close(self):
        """Flush buffers and close Parquet writers for the current window."""
        if not self._current_slug:
            return

        self._flush_book()
        self._flush_trade()

        if self._book_writer:
            self._book_writer.close()
            self._book_writer = None

        if self._trade_writer:
            self._trade_writer.close()
            self._trade_writer = None

        logger.info(f"Closed Parquet writers for {self._current_slug}")
        self._current_slug = None

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def write_book(self, event, received_timestamp: int):
        """Process a BookSnapshot and append to the book buffer."""
        if not self._book_writer:
            logger.warning("write_book called but no writer open")
            return

        # Bids arrive ascending (worst to best) — reverse to get best first
        bids = [(pl.price, pl.size) for pl in event.bids]
        asks = [(pl.price, pl.size) for pl in event.asks]

        # Sort: bids descending (highest first), asks ascending (lowest first)
        bids_sorted = sorted(bids, key=lambda x: x[0], reverse=True)
        asks_sorted = sorted(asks, key=lambda x: x[0])

        best_bid = bids_sorted[0][0] if bids_sorted else 0.0
        best_ask = asks_sorted[0][0] if asks_sorted else 0.0
        mid_price = (best_bid + best_ask) / 2
        spread = best_ask - best_bid

        top_bids = bids_sorted[:10]
        top_asks = asks_sorted[:10]

        total_bid_size = sum(s for _, s in top_bids)
        total_ask_size = sum(s for _, s in top_asks)
        denom = total_bid_size + total_ask_size
        book_imbalance = (total_bid_size - total_ask_size) / denom if denom else 0.0

        row = {
            "exchange_timestamp": event.exchange_timestamp,
            "received_timestamp": received_timestamp,
            "market": event.market,
            "asset_id": event.asset_id,
            "mid_price": mid_price,
            "spread": spread,
            "book_imbalance": book_imbalance,
            "hash": event.hash,
            "full_bids": json.dumps([[p, s] for p, s in bids]),
            "full_asks": json.dumps([[p, s] for p, s in asks]),
        }

        # Flatten top 10 levels
        for i in range(10):
            row[f"bid_price_{i+1}"] = top_bids[i][0] if i < len(top_bids) else 0.0
            row[f"bid_size_{i+1}"] = top_bids[i][1] if i < len(top_bids) else 0.0
            row[f"ask_price_{i+1}"] = top_asks[i][0] if i < len(top_asks) else 0.0
            row[f"ask_size_{i+1}"] = top_asks[i][1] if i < len(top_asks) else 0.0

        self._book_buffer.append(row)
        if len(self._book_buffer) >= FLUSH_INTERVAL_ROWS:
            self._flush_book()

    def write_trade(self, event, received_timestamp: int, price_change):
        """Process a single PriceChange and append to the trade buffer."""
        if not self._trade_writer:
            logger.warning("write_trade called but no writer open")
            return

        mid_price = (price_change.best_bid + price_change.best_ask) / 2
        spread = price_change.best_ask - price_change.best_bid

        row = {
            "exchange_timestamp": event.exchange_timestamp,
            "received_timestamp": received_timestamp,
            "market": event.market,
            "asset_id": price_change.asset_id,
            "side": "BUY" if price_change.side == 1 else "SELL",
            "trade_price": price_change.price,
            "size": price_change.size,
            "best_bid": price_change.best_bid,
            "best_ask": price_change.best_ask,
            "mid_price": mid_price,
            "spread": spread,
            "hash": price_change.hash,
        }

        self._trade_buffer.append(row)
        if len(self._trade_buffer) >= FLUSH_INTERVAL_ROWS:
            self._flush_trade()

    # ------------------------------------------------------------------
    # Flushing
    # ------------------------------------------------------------------
    def _start_flush_timer(self):
        """Flush buffers every 10 seconds to prevent unbounded growth."""
        def flush_periodically():
            try:
                while self._current_slug:
                    time.sleep(10)
                    logger.debug("Periodic flush triggered")
                    self._flush_book()
                    self._flush_trade()
            except Exception as e:
                logger.error(f"Flush timer thread crashed: {e}", exc_info=True)
        
        t = threading.Thread(target=flush_periodically, daemon=True)
        t.start()

    def _flush_book(self):
        with self._flush_lock:
            if not self._book_buffer or not self._book_writer:
                return
            table = pa.Table.from_pylist(self._book_buffer, schema=self.BOOK_SCHEMA)
            self._book_writer.write_table(table)
            logger.debug(f"Flushed {len(self._book_buffer)} book rows")
            self._book_buffer.clear()

    def _flush_trade(self):
        with self._flush_lock:
            if not self._trade_buffer or not self._trade_writer:
                return
            table = pa.Table.from_pylist(self._trade_buffer, schema=self.TRADE_SCHEMA)
            self._trade_writer.write_table(table)
            logger.debug(f"Flushed {len(self._trade_buffer)} trade rows")
            self._trade_buffer.clear()