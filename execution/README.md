# Execution Engine

TypeScript/Bun execution engine for live trading on Polymarket 5-minute crypto binary prediction markets (BTC/ETH/SOL/XRP Up/Down).

## Architecture

Data flow through the system:

```
MarketDataSource (WS/Redis)
  -> OrderBookManager -> Resampler -> IndicatorEngine -> MarketState
  -> Strategy.evaluate() -> TradeAction[]
  -> dispatchActions() -> OrderManager / Minter / Redeemer / NonceCanceller
```

Supporting subsystems:

- **MarketRotation** -- Polls Gamma API for the current market window, schedules open/close via `setTimeout` aligned to `market.endTime`.
- **UserChannel** -- WebSocket subscription to Polymarket user channel for real-time order and fill events. Feeds back into PositionStore.
- **External feeds** -- Binance WebSocket for spot prices (BTC, ETH, SOL, XRP). Injected into MarketState as `externalData`.
- **Dashboard** -- HTTP/SSE server on port 3456 (configurable). Exposes live state and a kill switch that cancels all orders and market-sells remaining inventory.
- **CsvLogger** -- Writes per-trade, per-order, and per-fill CSVs to `metrics_dir`.

## Strategy Interface

```typescript
interface Strategy {
  readonly id: string;
  readonly dataMode: DataMode; // "tick" | "resampled"
  onMarketOpen(state: MarketState): Promise<void>;
  evaluate(state: MarketState): Promise<TradeAction[]>;
  onMarketClose(state: MarketState): Promise<TradeAction[]>;
  onFill?(assetId: string, side: Side, size: number, price: number): void;
  onMintComplete?(conditionId: string, pairsMinted: number, success: boolean): void;
  onInventorySync?(upBalance: number, downBalance: number, openOrders: { assetId: string; side: Side; price: number; size: number }[]): void;
  getState?(): Record<string, unknown>;
}
```

Strategies register themselves via `registerStrategy(id, factory)` and are resolved at startup by `config.execution.strategy_id`.

## Available Strategies

Registered (imported in `index.ts`):

| ID | Description |
|----|-------------|
| `entropy-binance` | Conditional entropy regime detection + Binance BTC price signal |
| `market-maker` | Dual-mode MM (mint-and-sell / traditional) with inventory-skew or whale-front pricing |
| `xgb-confidence` | XGBoost model server integration (120s accumulate, then predict) |
| `signal-receiver` | Generic HTTP model server interface |
| `smoke-test` | Simple spread-capture validation |
| `mid-drift` | Mid-price drift detector |
| `mid-drift-mint` | Mid-drift variant with minting |

Additional strategy files exist in `src/strategy/` (momentum, gap-trader, accumulator) but are not imported by default.

## TradeAction Types

| Action | Description |
|--------|-------------|
| `PLACE_ORDER` | Submit a limit order (GTC/GTD/FOK) |
| `CANCEL_ORDER` | Cancel a single order by ID |
| `CANCEL_ALL` | Cancel all open orders |
| `SPLIT` | Mint YES+NO token pairs from USDC.e (on-chain) |
| `MERGE_PAIRS` | Burn YES+NO pairs back to USDC.e |
| `REDEEM` | Redeem winning tokens after market resolution |
| `NONCE_INVALIDATE` | On-chain nonce bump to invalidate all outstanding orders |
| `NOOP` | Do nothing |

## Configuration

The engine reads `config.json` from the repository root (or `$CONFIG` env var). Snake_case JSON keys are mapped to camelCase internally.

```json
{
  "polymarket": {
    "host": "https://clob.polymarket.com",
    "chain_id": 137,
    "private_key": "",
    "api_key": "",
    "api_secret": "",
    "api_passphrase": "",
    "proxy_wallet": ""
  },
  "market_maker": {
    "mode": "mint-and-sell",
    "pricing": "inventory-skew",
    "initial_pairs": 50,
    "max_total_inventory": 200,
    "max_unhedged_exposure": 50,
    "skew_k": 0.005,
    "requote_threshold": 0.005,
    "replenish_threshold": 10,
    "whale_threshold": 1000,
    "whale_pull_cooldown_ms": 3000,
    "whale_move_ticks": 3,
    "order_size": 5.0,
    "cancel_window_s": 6.0,
    "min_remaining_s": 30.0,
    "max_positions": 1,
    "price_offset": 0.0
  },
  "execution": {
    "asset": "btc",
    "data_source": "websocket",
    "redis_url": "redis://localhost:6379",
    "strategy_id": "entropy-binance",
    "metrics_dir": "data/metrics",
    "rpc_url": "https://polygon-rpc.com",
    "resample_interval_ms": 100,
    "external_feeds": ["binance-btcusdt"],
    "bet_dollars": 20,
    "model_server_url": "http://127.0.0.1:8000",
    "market_duration": "5m",
    "dashboard_port": 3456
  }
}
```

Key execution fields:

- `asset` -- Which crypto asset to trade (`btc`, `eth`, `sol`, `xrp`).
- `data_source` -- `websocket` (Polymarket WS) or `redis` (Redis Streams with protobuf).
- `strategy_id` -- Which registered strategy to run.
- `external_feeds` -- Array of external feed names (e.g. `binance-btcusdt`, `binance-ethusdt`).
- `bet_dollars` -- Position size in USD per trade.
- `market_duration` -- Market window length (`5m`, `4h`, etc.).

## Running

```bash
cd execution && bun run src/index.ts
```

Requires Bun runtime. Config is loaded from `../config.json` by default.

## Key Files

```
src/
  index.ts                  Main orchestrator (wiring, event loop, dispatch)
  types.ts                  All type definitions
  config.ts                 Config loading and validation
  shutdown.ts               Graceful shutdown handler

  strategy/
    strategy.ts             Strategy interface + registry
    entropy-binance.ts      Entropy/Binance strategy
    market-maker.ts         Market maker strategy
    xgb-confidence.ts       XGBoost confidence strategy
    signal-receiver.ts      Model server interface
    smoke-test.ts           Smoke test strategy
    mid-drift.ts            Mid-drift strategy
    mid-drift-mint.ts       Mid-drift with minting

  data/
    market-data-source.ts   Factory for WS/Redis data sources
    ws-market-data.ts       Polymarket WebSocket data source
    redis-market-data.ts    Redis Streams data source
    orderbook.ts            OrderBookManager (L2 book state)
    resampler.ts            Tick-to-bar resampling
    indicators.ts           Technical indicators (SMA, EMA, RSI, vol)
    external-feed.ts        External feed factory
    binance-feed.ts         Binance WebSocket price feed
    proto.ts                Protobuf init for Redis mode

  orders/
    order-manager.ts        CLOB order placement, cancellation, API cred derivation
    nonce-cancel.ts         On-chain nonce invalidation

  positions/
    position-store.ts       Local position/order tracking with fill reconciliation
    user-channel.ts         Polymarket user channel WebSocket
    minter.ts               On-chain split/merge (Gnosis Safe, mutex, timeout)
    redemption.ts           Post-resolution token redemption sweep

  rotation/
    market-rotation.ts      Gamma API polling + market window scheduling

  dashboard/
    server.ts               HTTP/SSE dashboard server
    html.ts                 Dashboard HTML template

  logging/
    csv-logger.ts           CSV trade/order/fill logging
```

## Adding a New Strategy

1. Create `src/strategy/my-strategy.ts` implementing the `Strategy` interface.
2. Call `registerStrategy("my-strategy", factory)` at module level.
3. Add a side-effect import in `src/index.ts`: `import "./strategy/my-strategy.ts";`
4. Set `execution.strategy_id = "my-strategy"` in `config.json`.

## Testing

```bash
cd execution && bun test
```

Tests live in `execution/test/`.
