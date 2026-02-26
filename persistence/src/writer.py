import json
import logging
import os
import threading
import time

import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

DATA_DIR = os.environ.get("DATA_DIR", "./data")
FLUSH_INTERVAL_ROWS = 500

# ---------------------------------------------------------------------------
# Canonical schemas — module-level so backfill and analysis code can import
# them without instantiating a ParquetWriter.
#
# Schema v2 additions vs. v1:
#   + slug        (string) — market slug; inferred from filename for legacy files
#   + token_label (string) — "Yes" / "No" / "" for live data missing the map
# ---------------------------------------------------------------------------

BOOK_SCHEMA = pa.schema([
    pa.field("exchange_timestamp", pa.int64()),    # milliseconds UTC
    pa.field("received_timestamp", pa.int64()),    # milliseconds UTC; 0 for backfill
    pa.field("slug",               pa.string()),   # e.g. btc-updown-5m-1771346400
    pa.field("market",             pa.string()),   # condition_id (0x...)
    pa.field("asset_id",           pa.string()),   # token_id
    pa.field("token_label",        pa.string()),   # "Yes" / "No" / ""
    pa.field("mid_price",          pa.float64()),
    pa.field("spread",             pa.float64()),
    pa.field("book_imbalance",     pa.float64()),
    pa.field("bid_price_1",        pa.float64()),
    pa.field("bid_size_1",         pa.float64()),
    pa.field("bid_price_2",        pa.float64()),
    pa.field("bid_size_2",         pa.float64()),
    pa.field("bid_price_3",        pa.float64()),
    pa.field("bid_size_3",         pa.float64()),
    pa.field("bid_price_4",        pa.float64()),
    pa.field("bid_size_4",         pa.float64()),
    pa.field("bid_price_5",        pa.float64()),
    pa.field("bid_size_5",         pa.float64()),
    pa.field("bid_price_6",        pa.float64()),
    pa.field("bid_size_6",         pa.float64()),
    pa.field("bid_price_7",        pa.float64()),
    pa.field("bid_size_7",         pa.float64()),
    pa.field("bid_price_8",        pa.float64()),
    pa.field("bid_size_8",         pa.float64()),
    pa.field("bid_price_9",        pa.float64()),
    pa.field("bid_size_9",         pa.float64()),
    pa.field("bid_price_10",       pa.float64()),
    pa.field("bid_size_10",        pa.float64()),
    pa.field("ask_price_1",        pa.float64()),
    pa.field("ask_size_1",         pa.float64()),
    pa.field("ask_price_2",        pa.float64()),
    pa.field("ask_size_2",         pa.float64()),
    pa.field("ask_price_3",        pa.float64()),
    pa.field("ask_size_3",         pa.float64()),
    pa.field("ask_price_4",        pa.float64()),
    pa.field("ask_size_4",         pa.float64()),
    pa.field("ask_price_5",        pa.float64()),
    pa.field("ask_size_5",         pa.float64()),
    pa.field("ask_price_6",        pa.float64()),
    pa.field("ask_size_6",         pa.float64()),
    pa.field("ask_price_7",        pa.float64()),
    pa.field("ask_size_7",         pa.float64()),
    pa.field("ask_price_8",        pa.float64()),
    pa.field("ask_size_8",         pa.float64()),
    pa.field("ask_price_9",        pa.float64()),
    pa.field("ask_size_9",         pa.float64()),
    pa.field("ask_price_10",       pa.float64()),
    pa.field("ask_size_10",        pa.float64()),
    pa.field("hash",               pa.string()),
    pa.field("full_bids",          pa.string()),   # JSON [[price, size], ...]
    pa.field("full_asks",          pa.string()),   # JSON [[price, size], ...]
])

TRADE_SCHEMA = pa.schema([
    pa.field("exchange_timestamp", pa.int64()),
    pa.field("received_timestamp", pa.int64()),
    pa.field("slug",               pa.string()),
    pa.field("market",             pa.string()),
    pa.field("asset_id",           pa.string()),
    pa.field("token_label",        pa.string()),
    pa.field("side",               pa.string()),
    pa.field("trade_price",        pa.float64()),
    pa.field("size",               pa.float64()),
    pa.field("best_bid",           pa.float64()),
    pa.field("best_ask",           pa.float64()),
    pa.field("mid_price",          pa.float64()),
    pa.field("spread",             pa.float64()),
    pa.field("hash",               pa.string()),
])


# ---------------------------------------------------------------------------
# Helpers — shared by ParquetWriter (live) and Telonex backfill
# ---------------------------------------------------------------------------

def compute_book_derived(
    bids_raw: list[tuple[float, float]],
    asks_raw: list[tuple[float, float]],
    n: int = 10,
) -> dict:
    """
    Sort bid/ask levels and compute derived fields.

    Args:
        bids_raw: [(price, size), ...] in any order
        asks_raw: [(price, size), ...] in any order
        n: number of levels to return in top_bids / top_asks

    Returns dict with:
        mid_price, spread, book_imbalance,
        top_bids: [(price, size), ...] best-first, length <= n
        top_asks: [(price, size), ...] best-first, length <= n
    """
    bids_sorted = sorted(bids_raw, key=lambda x: x[0], reverse=True)
    asks_sorted = sorted(asks_raw, key=lambda x: x[0])

    best_bid  = bids_sorted[0][0] if bids_sorted else 0.0
    best_ask  = asks_sorted[0][0] if asks_sorted else 0.0
    mid_price = (best_bid + best_ask) / 2
    spread    = best_ask - best_bid

    top_bids = bids_sorted[:n]
    top_asks = asks_sorted[:n]

    total_bid = sum(s for _, s in top_bids)
    total_ask = sum(s for _, s in top_asks)
    denom     = total_bid + total_ask
    imbalance = (total_bid - total_ask) / denom if denom else 0.0

    return {
        "mid_price":      mid_price,
        "spread":         spread,
        "book_imbalance": imbalance,
        "top_bids":       top_bids,
        "top_asks":       top_asks,
    }


def flatten_book_levels(row: dict, top_bids: list, top_asks: list, n: int = 10):
    """Write flattened 1-indexed price/size columns into row dict in-place."""
    for i in range(n):
        row[f"bid_price_{i+1}"] = top_bids[i][0] if i < len(top_bids) else 0.0
        row[f"bid_size_{i+1}"]  = top_bids[i][1] if i < len(top_bids) else 0.0
        row[f"ask_price_{i+1}"] = top_asks[i][0] if i < len(top_asks) else 0.0
        row[f"ask_size_{i+1}"]  = top_asks[i][1] if i < len(top_asks) else 0.0


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

class ParquetWriter:
    """
    Writes BookSnapshot and TradeEvent records to per-window Parquet files.
    Buffers rows in memory and flushes in batches for efficiency.
    Rotates files on market_close / market_open control events.

    token_label_map: {asset_id: "Yes"/"No"} populated from the market_open
    control event. Empty for legacy live data — token_label will be "".
    """

    # Mirror module-level schemas on the class for import convenience
    BOOK_SCHEMA  = BOOK_SCHEMA
    TRADE_SCHEMA = TRADE_SCHEMA

    def __init__(self):
        self._current_slug:    str | None        = None
        self._token_label_map: dict[str, str]    = {}
        self._book_buffer:     list[dict]        = []
        self._trade_buffer:    list[dict]        = []
        self._book_writer:     pq.ParquetWriter | None = None
        self._trade_writer:    pq.ParquetWriter | None = None
        self._flush_lock = threading.Lock()
        os.makedirs(f"{DATA_DIR}/book_snapshots", exist_ok=True)
        os.makedirs(f"{DATA_DIR}/trade_events",   exist_ok=True)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open(self, slug: str, token_label_map: dict[str, str] | None = None):
        """
        Open Parquet writers for a new market window.

        Args:
            slug: market slug, used as the output filename.
            token_label_map: optional {asset_id: "Yes"/"No"} for this market.
                             Absent for bootstrap/legacy — token_label will be "".
        """
        if self._current_slug:
            logger.warning(f"open() called while {self._current_slug} still active — closing first")
            self.close()

        self._current_slug    = slug
        self._token_label_map = token_label_map or {}

        book_path  = f"{DATA_DIR}/book_snapshots/{slug}.parquet"
        trade_path = f"{DATA_DIR}/trade_events/{slug}.parquet"

        self._book_writer  = pq.ParquetWriter(book_path,  BOOK_SCHEMA,  compression="snappy")
        self._trade_writer = pq.ParquetWriter(trade_path, TRADE_SCHEMA, compression="snappy")
        logger.info(f"Opened Parquet writers for {slug} (labels: {self._token_label_map})")
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
        self._current_slug    = None
        self._token_label_map = {}

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def write_book(self, event, received_timestamp: int):
        """Process a BookSnapshot protobuf and append to the book buffer."""
        if not self._book_writer:
            logger.warning("write_book called but no writer open")
            return

        bids_raw = [(pl.price, pl.size) for pl in event.bids]
        asks_raw = [(pl.price, pl.size) for pl in event.asks]
        derived  = compute_book_derived(bids_raw, asks_raw)

        row = {
            "exchange_timestamp": event.exchange_timestamp,
            "received_timestamp": received_timestamp,
            "slug":               self._current_slug or "",
            "market":             event.market,
            "asset_id":           event.asset_id,
            "token_label":        self._token_label_map.get(event.asset_id, ""),
            "mid_price":          derived["mid_price"],
            "spread":             derived["spread"],
            "book_imbalance":     derived["book_imbalance"],
            "hash":               event.hash,
            "full_bids":          json.dumps([[p, s] for p, s in bids_raw]),
            "full_asks":          json.dumps([[p, s] for p, s in asks_raw]),
        }
        flatten_book_levels(row, derived["top_bids"], derived["top_asks"])

        self._book_buffer.append(row)
        if len(self._book_buffer) >= FLUSH_INTERVAL_ROWS:
            self._flush_book()

    def write_trade(self, event, received_timestamp: int, price_change):
        """Process a single PriceChange protobuf and append to the trade buffer."""
        if not self._trade_writer:
            logger.warning("write_trade called but no writer open")
            return

        mid_price = (price_change.best_bid + price_change.best_ask) / 2
        spread    = price_change.best_ask - price_change.best_bid

        row = {
            "exchange_timestamp": event.exchange_timestamp,
            "received_timestamp": received_timestamp,
            "slug":               self._current_slug or "",
            "market":             event.market,
            "asset_id":           price_change.asset_id,
            "token_label":        self._token_label_map.get(price_change.asset_id, ""),
            "side":               "BUY" if price_change.side == 1 else "SELL",
            "trade_price":        price_change.price,
            "size":               price_change.size,
            "best_bid":           price_change.best_bid,
            "best_ask":           price_change.best_ask,
            "mid_price":          mid_price,
            "spread":             spread,
            "hash":               price_change.hash,
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

        threading.Thread(target=flush_periodically, daemon=True).start()

    def _flush_book(self):
        with self._flush_lock:
            if not self._book_buffer or not self._book_writer:
                return
            table = pa.Table.from_pylist(self._book_buffer, schema=BOOK_SCHEMA)
            self._book_writer.write_table(table)
            logger.debug(f"Flushed {len(self._book_buffer)} book rows")
            self._book_buffer.clear()

    def _flush_trade(self):
        with self._flush_lock:
            if not self._trade_buffer or not self._trade_writer:
                return
            table = pa.Table.from_pylist(self._trade_buffer, schema=TRADE_SCHEMA)
            self._trade_writer.write_table(table)
            logger.debug(f"Flushed {len(self._trade_buffer)} trade rows")
            self._trade_buffer.clear()