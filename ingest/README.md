# Ingest Service

Live data collection from the Polymarket CLOB WebSocket. Receives orderbook snapshots and trade events for BTC 5-minute Up/Down markets, serializes them as protobuf, and publishes to Redis Streams.

## Architecture

```
Polymarket CLOB WebSocket
        |
        v
  PolymarketClient          (market_client.py)
  - Connects to wss://ws-subscriptions-clob.polymarket.com
  - Subscribes to active BTC 5m market tokens (Up + Down)
  - Parses JSON into MarketEvent protobufs
  - Handles market rotation (5m windows) automatically
        |
        v
  RedisPublisher             (redis_publisher.py)
  - Publishes serialized protobuf to "polymarket:market_events" stream
  - Publishes market_open/market_close to "polymarket:control" stream
  - Stores market metadata in Redis hashes (market_meta:<condition_id>)
  - Creates consumer groups on startup (persistence, hot_analysis)
```

## Data Flow

1. On startup, fetches the current active market from the Gamma API (`https://gamma-api.polymarket.com/markets/slug/btc-updown-5m-{timestamp}`).
2. Opens a WebSocket connection to the `market` channel and subscribes to both token IDs (Up and Down).
3. Incoming messages are parsed by event type:
   - `book` -- full orderbook snapshot -> `BookSnapshot` protobuf
   - `price_change` -- trade event -> `TradeEvent` protobuf (contains one or more `PriceChange` entries)
   - `last_trade_price`, `tick_size_change` -- silently ignored
4. Each parsed event is wrapped in a `MarketEvent` envelope (adds `received_timestamp`) and serialized to bytes.
5. Serialized bytes are published to the `polymarket:market_events` Redis Stream with fields: `event_type`, `market` (condition ID), and `data` (protobuf bytes).
6. A background monitor thread checks every 5 seconds for market window expiry. On expiry, it publishes `market_close`, fetches the next market from Gamma API (up to 10 retries), publishes `market_open`, and resubscribes on the existing WebSocket connection.

## Configuration

No config file needed by the ingest service itself. It derives the market slug from the current UTC time (floored to 5-minute boundary).

Redis connection defaults to `localhost:6379` (host networking in Docker).

## Redis Streams

| Stream | Fields | Purpose |
|---|---|---|
| `polymarket:market_events` | `event_type`, `market`, `data` (protobuf bytes) | Book snapshots and trades |
| `polymarket:control` | `event_type`, `market`, `slug`, `end_time`, `up_token_id`, `down_token_id` | Market lifecycle events |

The market events stream is capped at 50,000 entries (`STREAM_MAXLEN`). The control stream is uncapped (small volume).

Consumer groups created on startup: `polymarket:persistence`, `polymarket:hot_analysis`.

## Running

Part of the root `docker-compose.yml`:

```bash
# Start all services (redis, ingest, persistence)
docker compose up -d

# View ingest logs
docker compose logs -f ingest

# Restart just ingest
docker compose restart ingest
```

The service uses `network_mode: host` and mounts `./config.json` (though ingest only needs Redis, not API credentials). Depends on the `redis` service.

## Key Files

| File | Purpose |
|---|---|
| `src/main.py` | Entry point. Wires up callbacks between client and publisher. |
| `src/market_client.py` | `PolymarketClient` -- WebSocket connection, JSON-to-protobuf parsing, market rotation. |
| `src/redis_publisher.py` | `RedisPublisher` -- Redis Stream writes, consumer group setup, market metadata hashes. |
| `Dockerfile` | Python 3.12 slim image. Copies `shared/polymarket_pb2.py` for protobuf definitions. |

## Protobuf Schema

Defined in `polymarket.proto` at the repo root. Generated code lives in `shared/polymarket_pb2.py`.

Key messages:
- `MarketEvent` -- envelope with `received_timestamp` + oneof `BookSnapshot` | `TradeEvent`
- `BookSnapshot` -- full book with `market`, `asset_id`, bids/asks as `PriceLevel` arrays
- `TradeEvent` -- one or more `PriceChange` entries with price, size, side, best bid/ask

## Dependencies

- `websocket-client` -- WebSocket connection
- `redis` -- Redis client
- `protobuf` -- protobuf serialization
- `requests` -- Gamma API calls for market discovery
