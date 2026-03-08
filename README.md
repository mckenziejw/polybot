# polybot

RL trading framework for Polymarket's BTC Up/Down binary prediction markets. Covers the full pipeline: historical data collection, preprocessing, PPO+LSTM training, and live execution.

## Project Status

**Current phase: Live execution engine built, testing in progress**

1. ~~Momentum strategy~~ — falsified. Buying momentum loses money.
2. **Calibration analysis** — complete. Market is broadly efficient in aggregate; no exploitable edge in price buckets, time windows, or volatility regimes.
3. **RL training** — operational. PPO+LSTM agent trained on 15-minute markets via CleanRL-style pure PyTorch.
4. **Live execution** — built (TypeScript/Bun). Pluggable strategy system with dual data source support. 253 tests passing (215 unit + 38 integration).

## Market Structure

Polymarket runs continuous BTC Up/Down binary markets. Each market has two tokens:

- **Yes (Up)** — pays $1 if BTC is higher at close than at open
- **No (Down)** — pays $1 if BTC is lower or unchanged

Tokens trade on Polymarket's CLOB between $0.00 and $1.00. Resolution uses Chainlink's BTC/USD price feed on Polygon.

Key structural properties:
- Hard episode boundaries (each market is fully isolated)
- Known termination time
- Prices are probability estimates, not asset values
- Dual-token sum constraint (Yes + No ~ 1 minus platform take)
- Liquidity collapses in the final ~10 seconds before close

## Repository Structure

```
polybot/
├── execution/                   # Live trading engine (TypeScript/Bun)
│   ├── src/
│   │   ├── index.ts             #   Orchestrator — wires all components
│   │   ├── config.ts            #   Config loader (snake_case → camelCase)
│   │   ├── types.ts             #   All shared interfaces
│   │   ├── shutdown.ts          #   Graceful shutdown handler
│   │   ├── data/
│   │   │   ├── ws-market-data.ts    # Direct Polymarket WS connection
│   │   │   ├── redis-market-data.ts # Redis Streams consumer
│   │   │   ├── market-data-source.ts# Interface + factory
│   │   │   ├── orderbook.ts     #   L2 book (snapshot + incremental)
│   │   │   ├── resampler.ts     #   Raw ticks → fixed-interval bars
│   │   │   ├── indicators.ts    #   Rolling MA, EMA, vol, RSI
│   │   │   ├── proto.ts         #   Protobuf decode helper
│   │   │   ├── external-feed.ts #   External data feed interface
│   │   │   └── binance-feed.ts  #   Binance BTC/USDT bookTicker
│   │   ├── orders/
│   │   │   ├── order-manager.ts #   CLOB client wrapper (place/cancel)
│   │   │   └── nonce-cancel.ts  #   On-chain incrementNonce() via viem
│   │   ├── positions/
│   │   │   ├── position-store.ts#   In-memory position + order tracking
│   │   │   └── user-channel.ts  #   WS user channel (fills/cancels)
│   │   ├── strategy/
│   │   │   ├── strategy.ts      #   Strategy interface + registry
│   │   │   ├── market-maker.ts  #   Two-sided market maker
│   │   │   └── signal-receiver.ts#  RL signal adapter stub
│   │   ├── rotation/
│   │   │   └── market-rotation.ts#  Gamma API polling
│   │   └── logging/
│   │       └── csv-logger.ts    #   CSV writers (orders, fills, PnL)
│   └── test/                    #   Unit + integration tests (bun:test)
│       ├── orderbook.test.ts
│       ├── resampler.test.ts
│       ├── indicators.test.ts
│       ├── position-store.test.ts
│       ├── csv-logger.test.ts
│       ├── config.test.ts
│       ├── ws-market-data.test.ts
│       ├── user-channel.test.ts
│       ├── market-rotation.test.ts
│       ├── strategy.test.ts
│       └── integration/
│           ├── orderbook-replay.test.ts  # Replay parquet data, verify book reconstruction
│           ├── resampler-parity.test.ts  # Compare TS vs Python resampler output
│           └── schema-validation.test.ts # Cross-source schema checks
├── ingest/                      # Docker service: WebSocket → Redis
│   └── src/
│       ├── main.py
│       ├── market_client.py     #   Polymarket WS client + market rotation
│       └── redis_publisher.py   #   Redis Streams publisher (protobuf)
├── persistence/                 # Docker service: Redis → Parquet
│   └── src/
│       ├── main.py
│       ├── consumer.py          #   Redis Streams consumer
│       └── writer.py            #   Parquet writer (book + trade schemas)
├── shared/
│   └── polymarket_pb2.py        # Generated protobuf bindings
├── polymarket_env.py            # Gymnasium RL environment (Discrete(6), maskable)
├── train_cleanrl.py             # PPO+LSTM training (CleanRL style, pure PyTorch)
├── train.py                     # SB3-based training (WIP alternative)
├── replay_cleanrl.py            # Model evaluation / replay
├── replay.py                    # Replay (WIP alternative)
├── ppo_lstm.py                  # PPO+LSTM network definition (WIP)
├── preprocess_markets.py        # Resample raw data → 100ms bars
├── telonex_pipeline.py          # Download + normalize Telonex historical data
├── btc_pipeline.py              # Download + normalize Binance BTC quotes
├── market_analysis.py           # MarketLoader, ResolutionStore, CalibrationAnalyser
├── chainlink_oracle.py          # Chainlink BTC/USD round fetcher (Polygon RPC)
├── settlement_latency.py        # CLOB match → on-chain settlement gap analysis
├── price_move_windows.py        # Price movement within 3-6s windows
├── batch_cadence.py             # Trade batch timing analysis
├── window_summariser.py         # Analysis helper
├── resolution_fetcher.py        # Standalone resolution fetch utility
├── mispricing.py                # Calibration / edge analysis
├── docker-compose.yml           # Redis + ingest + persistence services
├── config.example.json          # Credentials template
├── polymarket.proto             # Protobuf schema
├── POLYMARKET_ARCHITECTURE.md   # Polymarket technical reference
└── requirements.txt
```

## Architecture

### Data Flow (Training)

```
Telonex API → telonex_pipeline.py → datasets/telonex_15m_raw/
Binance API → btc_pipeline.py     → datasets/btcusdt_quotes.parquet
                    ↓
          preprocess_markets.py → data/telonex_15m_100ms/ (100ms resampled)
                    ↓
          polymarket_env.py (Gymnasium env, 63-dim obs, Discrete(6) actions)
                    ↓
          train_cleanrl.py (PPO+LSTM, 8 parallel envs) → models/checkpoints/
```

### Data Flow (Live Ingest)

```
Polymarket WS → ingest/ → Redis Streams (protobuf) → persistence/ → data/book_snapshots/
```

### Data Flow (Live Execution)

```
MarketDataSource (WS or Redis)
    │
    ├── OrderBook (snapshot + incremental updates)
    ├── Resampler → Indicators (MA, EMA, vol, RSI)
    └── Strategy.evaluate(MarketState) → TradeAction[]
                                            │
                                            └── OrderManager (CLOB API)

UserChannel (ws/user) → PositionStore → CsvLogger
Binance WS → ExternalData → MarketState.externalData
```

## Execution Engine

The `execution/` directory is a standalone TypeScript/Bun application for live trading on Polymarket.

### Key Features

- **Pluggable strategies**: Implement the `Strategy` interface for custom trading logic. Built-in: `market-maker` (two-sided), `signal-receiver` (RL adapter)
- **Dual data source**: Direct WebSocket to Polymarket CLOB, or Redis Streams consumer (reuses ingest pipeline)
- **Incremental orderbook**: Full L2 book maintained from snapshots + price_change events (10-level depth from live, 5-level from Telonex)
- **Resampled bars**: 100ms fixed-interval bars with forward-fill, staleness tracking, and rolling indicators (parity-tested against Python `preprocess_markets.py`)
- **External feeds**: Binance BTC/USDT bookTicker (extensible to other feeds via `ExternalFeed` interface)
- **Two cancellation paths**: CLOB API soft cancel + on-chain `incrementNonce()` nuclear option (via viem)
- **Graceful shutdown**: Cancels all orders on SIGINT/SIGTERM, falls back to nonce invalidation if soft cancel fails
- **CSV logging**: Orders, fills, and PnL logged for offline analysis
- **Market rotation**: Gamma API polling detects new 5-minute market windows (WebSocket mode) or Redis control stream events (Redis mode)

### Strategy Interface

Strategies are pure functions of state → actions. They don't execute orders directly; the orchestrator dispatches actions.

```typescript
interface Strategy {
  readonly id: string;
  readonly dataMode: "tick" | "resampled";
  onMarketOpen(state: MarketState): Promise<void>;
  evaluate(state: MarketState): Promise<TradeAction[]>;
  onMarketClose(state: MarketState): Promise<TradeAction[]>;
}
```

**`dataMode`** controls how the strategy receives data:
- `"tick"` — called on every raw market data event (book snapshot / price change). Lowest latency.
- `"resampled"` — called only when a new resampled bar is emitted (every 100ms by default). Gets `bars` and `indicators` in `MarketState`.

**Built-in strategies:**
- **`market-maker`** — Two-sided market maker. Posts simultaneous YES buy and NO buy to earn the spread. Cancels stale one-sided fills after a configurable window. Tick mode.
- **`signal-receiver`** — Bridge for external signals (RL model, rules engine). External code pushes actions via `pushSignal()`, which are drained on the next `evaluate()` call. Resampled mode.

Register custom strategies:
```typescript
import { registerStrategy } from "./strategy/strategy.ts";
registerStrategy("my-strategy", () => new MyStrategy());
```

### MarketState

Every `evaluate()` call receives a complete snapshot:

```typescript
interface MarketState {
  market: MarketInfo;                    // slug, conditionId, token IDs, endTime
  books: Record<string, OrderBookState>; // keyed by assetId (up + down tokens)
  positions: Position[];
  openOrders: OpenOrder[];
  timeRemainingMs: number;
  timestamp: number;
  bars?: ResampledBar[];                 // present when dataMode = "resampled"
  indicators?: IndicatorSnapshot;        // MA, EMA, vol, RSI
  externalData?: Record<string, ExternalDataPoint>;  // e.g. "binance-btcusdt"
}
```

### TradeAction

```typescript
type TradeAction =
  | { type: "PLACE_ORDER"; assetId: string; side: Side; price: number; size: number; orderType: OrderType }
  | { type: "CANCEL_ORDER"; orderId: string }
  | { type: "CANCEL_ALL" }
  | { type: "NONCE_INVALIDATE" }
  | { type: "NOOP" };
```

### Running

```bash
cd execution
bun install
bun run start
```

### Testing

```bash
cd execution
bun test              # all tests (253 tests, ~5s)
bun test:unit         # unit tests only (215 tests)
bun test:integration  # integration tests only (38 tests, requires data/ parquets)
```

Integration tests replay captured Parquet data (via pyarrow subprocess) and validate:
- **orderbook-replay** — L2 book reconstruction from captured snapshots, verifies top-5 levels and derived fields
- **resampler-parity** — bar-for-bar comparison of TS resampler output vs Python `preprocess_markets.py` output (compares raw bid/ask level prices)
- **schema-validation** — validates schemas across all data sources (live 10-level, Telonex 5-level, raw Telonex 0-indexed string prices), documents differences

### Configuration

Set the `execution` section in `config.json`:

```json
{
  "execution": {
    "data_source": "websocket",
    "strategy_id": "market-maker",
    "rpc_url": "https://polygon-rpc.com",
    "redis_url": "redis://localhost:6379",
    "metrics_dir": "data/metrics",
    "resample_interval_ms": 100,
    "external_feeds": ["binance-btcusdt"]
  }
}
```

| Field | Default | Description |
|-------|---------|-------------|
| `data_source` | `"websocket"` | `"websocket"` (direct WS) or `"redis"` (Redis Streams consumer) |
| `strategy_id` | `"market-maker"` | Strategy to run (must be registered in strategy registry) |
| `rpc_url` | `"https://polygon-rpc.com"` | Polygon RPC for on-chain nonce invalidation |
| `redis_url` | `"redis://localhost:6379"` | Redis URL (only used when `data_source = "redis"`) |
| `metrics_dir` | `"data/metrics"` | Directory for CSV logs (orders, fills, PnL) |
| `resample_interval_ms` | `100` | Bar interval for resampled-mode strategies |
| `external_feeds` | `[]` | External data feeds to connect (e.g. `"binance-btcusdt"`) |

Market maker parameters in the `market_maker` section:

| Field | Default | Description |
|-------|---------|-------------|
| `order_size` | `5.0` | USDC size per order |
| `cancel_window_s` | `6.0` | Seconds before cancelling a stale one-sided fill |
| `min_remaining_s` | `30.0` | Stop trading when less than this many seconds remain |
| `max_positions` | `1` | Maximum concurrent positions |
| `price_offset` | `0.0` | Offset from best bid/ask for order placement |

### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `@polymarket/clob-client` | ^4 | CLOB API (place/cancel orders, EIP-712 auth) |
| `viem` | ^2 | On-chain `incrementNonce()` for nuclear order cancellation |
| `ioredis` | ^5 | Redis Streams consumer (Redis data source mode) |
| `protobufjs` | ^7 | Decode protobuf-encoded market events from Redis |

## RL Environment

`polymarket_env.py` implements a Gymnasium environment for binary prediction markets:

- **Observation space** (63 dims): 46 book features (Yes+No, 5 levels), 6 position features, 1 time remaining, 8 BTC features, 2 staleness features
- **Action space**: Discrete(6) — Hold, Buy Yes Small/Large, Buy No Small/Large, Sell. Invalid actions masked based on capital and positions.
- **Reward**: Mark-to-market portfolio delta per step; binary settlement at terminal step
- **Episodes**: Each 15-minute market is one episode, resampled to 100ms steps

## Setup

### Prerequisites

- Python 3.12+
- [Bun](https://bun.sh) (for the execution engine)
- Docker & Docker Compose (for live ingest)
- Redis (runs via Docker)

### Installation

```bash
# Python (analysis + training)
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Execution engine
cd execution
bun install
```

### Configuration

Copy `config.example.json` to `config.json` and fill in your credentials:

```bash
cp config.example.json config.json
```

Required keys:
- `polymarket.private_key` — wallet private key
- `polymarket.api_key/secret/passphrase` — CLOB API credentials (or leave blank to auto-derive)
- `polymarket.proxy_wallet` — proxy wallet address
- `telonex.api_key` — for historical data downloads
- `etherscan.api_key` — for Chainlink oracle analysis

## Usage

### Live Ingest

```bash
docker-compose up -d
```

### Download Historical Data

```bash
python telonex_pipeline.py    # Polymarket book snapshots
python btc_pipeline.py        # Binance BTC quotes
```

### Preprocess for Training

```bash
python preprocess_markets.py
```

### Train

```bash
python train_cleanrl.py
```

Training logs are written to `runs/` for TensorBoard:

```bash
tensorboard --logdir runs
```

### Live Trading

```bash
cd execution
bun run start
```

## Known Issues

- **Token label mismatch**: Live ingest uses `"Yes"/"No"`, Telonex data uses `"Up"/"Down"`. The RL env filters on `"Up"/"Down"`, so it won't work directly with live data without normalization.
- **Mid price convention**: Telonex computes `mid = ask/2` when the bid side is empty; the TS OrderBook returns `mid = 0`. Integration tests compare raw level data to avoid false mismatches.

## Key Findings (Calibration Analysis)

Analysis of 9.78M observations across 1,394 markets:

- Tail contracts (< 0.35) are systematically overpriced. Buying cheap contracts loses money.
- High-probability contracts (> 0.65) show mild positive edge, but disaggregates into noise.
- The market is broadly efficient — no exploitable aggregate edge found.
- All volatility regimes show the same pattern.

## Data

- `data/` is gitignored (Parquet files not committed)
- `config.json` is gitignored (never commit credentials)
- `models/` contains trained checkpoints
- `datasets/` contains raw downloaded data
