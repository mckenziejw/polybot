# TODO — polybot

Outstanding work, ranked by priority.

---

## Blockers — Before Live Trading

### Token Label Normalization (Yes/No vs Up/Down)
Live ingest labels tokens `"Yes"` / `"No"` (Polymarket Gamma API). Telonex historical data and the RL env use `"Up"` / `"Down"`. The execution engine currently passes through whatever label the data source provides.

**Impact:** If the RL model expects `"Up"/"Down"` but live data says `"Yes"/"No"`, observation construction will break.

**Fix:** Normalize at the data source boundary. Options:
1. Map `"Yes"→"Up"`, `"No"→"Down"` in `ws-market-data.ts` and `redis-market-data.ts`
2. Map in `MarketRotation.fetchCurrentMarket()` where token labels are first assigned

### Live Smoke Test
The execution engine compiles and passes 253 offline tests, but has not been tested against live Polymarket infrastructure. Need to verify:
- [ ] WebSocket connection + subscription to a live market
- [ ] Book snapshot and price_change parsing from live data
- [ ] User channel authentication (derived L2 credentials)
- [ ] Order placement (small test order, e.g. $0.01)
- [ ] Order cancellation via CLOB API
- [ ] Graceful shutdown (Ctrl+C cancels open orders)
- [ ] Market rotation detection across window boundaries
- [ ] Binance feed connection and data flow

### Model Server Latency Budget
The signal-receiver strategy calls the Python model server via HTTP on every 100ms resampled bar. The round-trip (serialize MarketState → HTTP → model forward pass → deserialize actions) must fit within the 100ms interval. Localhost HTTP overhead is ~1-2ms, but the LSTM forward pass time on CPU is unknown. If inference exceeds the budget, bars will be skipped. **Monitor this during first live tests.** Mitigations: GPU inference, batching, increasing resample interval, or switching to Unix socket.

---

## Improvements — Execution Engine

### Retry / Error Handling for Order Placement
`OrderManager.placeOrder()` currently fires once and reports success/failure. Production needs:
- Retry with backoff on transient failures (rate limits, network errors)
- Dead-letter logging for permanently failed orders
- Idempotency (don't double-place if retry is ambiguous)

### PnL Tracking at Market Resolution
`PositionStore` tracks mark-to-market PnL but doesn't handle market resolution (binary settlement at $0 or $1). Need to:
- Subscribe to resolution events (Gamma API or on-chain)
- Compute final realized PnL per market
- Log to `pnl.csv`

### Redis Data Source Testing
`redis-market-data.ts` implements the Redis Streams consumer but hasn't been integration-tested against the actual ingest pipeline. Need to run with `docker-compose up` and verify protobuf decoding + event flow matches the WebSocket path.

### Indicator Extensibility
The `IndicatorEngine` computes a fixed set (SMA20, EMA20, vol, RSI14, spreadSMA10). Strategies may want custom indicators. Consider:
- Strategy-registered indicator callbacks
- Configurable window sizes

---

## Improvements — RL Training

### Chainlink Oracle Features
The Chainlink oracle price (not the CEX mid) determines Polymarket resolution. The CEX-oracle divergence in the final seconds is the primary hypothesized edge for smart money. Without oracle price as a feature, the agent has no direct signal.

**Plan:** Fetch Chainlink BTC/USD price feed history, align to market timestamps, add as observation features: `oracle_price`, `cex_oracle_divergence`, `seconds_since_last_oracle_update`.

### Slippage / Fill Model
Currently uses mid-price immediate fill. Real fills cross the spread. In thin Polymarket books the spread can be 5-10 cents wide. The current fill model is systematically optimistic.

**Options:** Fill at ask for buys / bid for sells, or probabilistic fill model based on order size vs available depth at each level.

### 15-Minute Market Training
Current training targets 15-minute markets (9000 bars per episode). The Telonex 5-minute data is denser and more plentiful. Consider:
- Training on 5-minute markets first (faster iteration, 3000 bars per episode)
- Transfer learning from 5m to 15m

---

## Infrastructure

### Cloud GPU Training (VastAI)
- Training script should offload compute to Vast
- Need cloud storage for market data (S3 or equivalent)
- Startup script: clone repo → pull data → start training → push logs

### Live Ingest Resampling
The ingest pipeline (`ingest/` + `persistence/`) writes raw tick-level snapshots. For live RL inference, need a resampling buffer that produces 100ms bars matching the training schema. The execution engine's `Resampler` class does this, but only in-process — it's not exposed as a standalone service.
