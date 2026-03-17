# Persistence Service

Consumes protobuf-serialized market events from Redis Streams and writes them as Parquet files to disk. One pair of files (book snapshots + trade events) per 5-minute market window.

## Architecture

```
Redis Streams
  polymarket:market_events   (protobuf bytes)
  polymarket:control         (market_open / market_close)
        |
        v
  PersistenceConsumer        (consumer.py)
  - XREADGROUP consumer in group "polymarket:persistence"
  - Deserializes MarketEvent protobufs
  - Routes to ParquetWriter based on event type
  - Bootstraps from control stream history on restart
        |
        v
  ParquetWriter              (writer.py)
  - Buffers rows in memory
  - Flushes every 500 rows or 10 seconds
  - Writes Snappy-compressed Parquet files
  - Rotates files on market_open control events
        |
        v
  data/book_snapshots/{slug}.parquet
  data/trade_events/{slug}.parquet
```

## Data Flow

1. On startup, reads the full `polymarket:control` stream history to determine if a market window is currently active (handles mid-window restarts).
2. Enters the main consume loop, reading from both `polymarket:market_events` and `polymarket:control` streams via `XREADGROUP` (batch size 100, 1s block timeout).
3. Control events:
   - `market_open` -- closes any previous writer, opens new Parquet files for the new slug, sets up the token label map (`{asset_id: "Yes"/"No"}`).
   - `market_close` -- logged but writer stays open to catch in-flight messages. Actual file rotation happens on the next `market_open`.
4. Market events: protobuf bytes are deserialized back into `MarketEvent`. Book snapshots go through `compute_book_derived()` which sorts levels, computes mid price, spread, and book imbalance. Trade events are flattened (one row per `PriceChange`).
5. Messages are acknowledged (`XACK`) after successful processing.
6. Heartbeat logged every 60 iterations (~60 seconds).

## Output Schema

### Book Snapshots (`data/book_snapshots/{slug}.parquet`)

| Column | Type | Description |
|---|---|---|
| `exchange_timestamp` | int64 | Exchange timestamp (ms UTC) |
| `received_timestamp` | int64 | Local receive timestamp (ms UTC) |
| `slug` | string | Market slug (e.g., `btc-updown-5m-1771346400`) |
| `market` | string | Condition ID |
| `asset_id` | string | Token ID |
| `token_label` | string | `"Yes"` / `"No"` / `""` |
| `mid_price` | float64 | (best_bid + best_ask) / 2 |
| `spread` | float64 | best_ask - best_bid |
| `book_imbalance` | float64 | (bid_vol - ask_vol) / (bid_vol + ask_vol) over top 10 levels |
| `bid_price_1`..`bid_price_10` | float64 | Bid prices, best-first |
| `bid_size_1`..`bid_size_10` | float64 | Bid sizes |
| `ask_price_1`..`ask_price_10` | float64 | Ask prices, best-first |
| `ask_size_1`..`ask_size_10` | float64 | Ask sizes |
| `hash` | string | Exchange-provided snapshot hash |
| `full_bids` | string | JSON array of all bid levels `[[price, size], ...]` |
| `full_asks` | string | JSON array of all ask levels |

### Trade Events (`data/trade_events/{slug}.parquet`)

| Column | Type | Description |
|---|---|---|
| `exchange_timestamp` | int64 | Exchange timestamp (ms UTC) |
| `received_timestamp` | int64 | Local receive timestamp (ms UTC) |
| `slug` | string | Market slug |
| `market` | string | Condition ID |
| `asset_id` | string | Token ID |
| `token_label` | string | `"Yes"` / `"No"` / `""` |
| `side` | string | `"BUY"` or `"SELL"` |
| `trade_price` | float64 | Execution price |
| `size` | float64 | Trade size (tokens) |
| `best_bid` | float64 | Best bid at time of trade |
| `best_ask` | float64 | Best ask at time of trade |
| `mid_price` | float64 | (best_bid + best_ask) / 2 |
| `spread` | float64 | best_ask - best_bid |
| `hash` | string | Trade hash |

## Configuration

| Variable | Default | Description |
|---|---|---|
| `DATA_DIR` (env) | `./data` | Output directory for Parquet files |
| `FLUSH_INTERVAL_ROWS` | 500 | Buffer size before flushing to disk |
| Periodic flush | 10 seconds | Background thread flush interval |
| Redis host | `localhost:6379` | Redis connection |

## Running

Part of the root `docker-compose.yml`:

```bash
# Start all services
docker compose up -d

# View persistence logs
docker compose logs -f persistence

# Verify output files
python persistence/verify_parquet.py
```

The service uses `network_mode: host`, mounts `./data` to `/app/data`, and runs as the host user (`${UID}:${GID}`) so output files have correct ownership. Depends on the `redis` service.

## Key Files

| File | Purpose |
|---|---|
| `src/main.py` | Entry point. Signal handling and consumer startup. |
| `src/consumer.py` | `PersistenceConsumer` -- Redis Stream consumer loop, control event handling, bootstrap logic. |
| `src/writer.py` | `ParquetWriter` -- buffered Parquet writing, schema definitions, derived field computation. Also exports `BOOK_SCHEMA`, `TRADE_SCHEMA`, `compute_book_derived()`, and `flatten_book_levels()` for use by analysis code. |
| `verify_parquet.py` | Utility script to inspect the most recent output files. |
| `Dockerfile` | Python 3.12 slim image. Copies `shared/polymarket_pb2.py` for protobuf definitions. |

## Dependencies

- `redis` -- Redis client
- `protobuf` -- protobuf deserialization
- `pyarrow` -- Parquet writing
- `pandas` -- listed in requirements (used by downstream analysis importing from writer)
