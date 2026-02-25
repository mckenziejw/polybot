# Polymarket BTC 5-Min Momentum Trading Bot

A live trading bot that trades Polymarket's **BTC Up/Down 5-minute** binary markets using a momentum strategy. The bot observes order book movement during the first half of each 5-minute window, then places a directional bet if momentum exceeds a threshold.

## Architecture Overview

```
momentum_bot.py          ← Main orchestrator (entry point)
├── bot/engine.py         ← Trading engine: orders, positions, balance, resolution
├── bot/strategy.py       ← Momentum signal generation from book ticks
├── bot/notifications.py  ← Console/webhook notifications
├── ingest/src/market_client.py ← WebSocket client + market rotation
├── shared/polymarket_pb2.py    ← Protobuf definitions for WS messages
├── bot_config.json       ← Strategy & trading parameters
└── config.json           ← Polymarket API credentials
```

### Data Flow

```
Polymarket WebSocket (protobuf book snapshots)
       │
       ▼
PolymarketClient (market_client.py)
  - Fetches current market from Gamma API
  - Subscribes to WS channel for UP + DOWN tokens
  - Handles market rotation every 5 minutes
  - Fires callbacks: on_event, on_market_open, on_market_close
       │
       ▼
MomentumBot (momentum_bot.py)
  - Parses protobuf book events
  - Feeds bid/ask ticks to MomentumStrategy
  - Executes signals via TradingEngine
  - Spawns background threads for resolution polling
       │
       ▼
MomentumStrategy (bot/strategy.py)
  - Collects book ticks during observation window (first 50% of market)
  - Computes momentum = (latest_mid - first_mid)
  - Generates buy_up or buy_down signal if |momentum| > threshold
       │
       ▼
TradingEngine (bot/engine.py)
  - Places limit buy orders via py-clob-client
  - Tracks multiple concurrent positions (dict keyed by slug)
  - Polls Gamma API for resolution (background threads)
  - Persists state to data/bot_state_live.json
```

## How Each 5-Minute Market Works

1. **Market opens** — Polymarket creates a new binary market: "Will BTC go up in the next 5 minutes?"
   - Two tokens: UP token and DOWN token
   - Each token pays $1 if correct, $0 if wrong
   - Tokens trade between $0 and $1 on the CLOB

2. **Observation phase** (first 2.5 min) — Bot watches order book movement
   - Records bid/ask midpoints for both UP and DOWN tokens
   - Computes momentum from first to latest midpoint

3. **Signal generation** (at 50% mark) — If momentum exceeds threshold:
   - Positive momentum → buy UP token
   - Negative momentum → buy DOWN token
   - Checks price bounds, slippage, daily loss limits

4. **Order execution** — Places a GTC limit buy at the best ask
   - Uses `create_order` (not `create_market_order`) to avoid py-clob-client rounding bugs
   - Price rounded to 2dp, shares floored to 2dp

5. **Market closes** (at 5 min mark) — Bot rotates to next market
   - Spawns background thread to poll for resolution

6. **Resolution** (5-10 min after close) — Gamma API shows outcome
   - `outcomePrices` goes to `["0","1"]` or `["1","0"]`
   - Position is closed, P&L calculated
   - Redemption logged (currently placeholder — see Known Issues)

## Files & Modules

### `momentum_bot.py` — Main Orchestrator
- Entry point: `python momentum_bot.py --live` or `--dry-run`
- Loads configs from `config.json` and `bot_config.json`
- Creates `TradingEngine`, `MomentumStrategy`, `NotificationService`, `PolymarketClient`
- Runs WebSocket event loop (blocking)
- Key callbacks:
  - `_on_ws_event(data)` — Parses protobuf, routes to `_handle_book_event`
  - `_on_market_open(market)` — Resets strategy, starts signal checker thread
  - `_on_market_close(market)` — Spawns resolution thread if position exists
  - `_execute_signal(market, sig)` — Validates price/balance/slippage, places order
  - `_handle_resolution(market)` — Background thread: polls Gamma API up to 15 min

### `bot/engine.py` — Trading Engine
- **Position management**: `positions: dict[str, Position]` (thread-safe with `_positions_lock`)
  - `open_position()`, `close_position(slug, outcome)`, `get_position()`, `has_position()`
- **Order execution**: `place_market_buy(token_id, amount, price)` — limit order via `OrderArgs`
- **Balance**: `fetch_balance()` from CLOB API, `get_position_size()` with optional autoscaling
- **Resolution**: `check_resolution(slug)` — polls Gamma API for `umaResolutionStatus`, `closed`, `outcomePrices`
- **Daily loss tracking**: `DailyStats` with auto-reset, `is_daily_loss_exceeded()`
- **State persistence**: `save_state()` / `load_state()` to `data/bot_state_{live|dry}.json`
- **CLOB auth**: Derives fresh API keys on startup (`signature_type=2` for proxy wallet)

### `bot/strategy.py` — Momentum Strategy
- `on_book_tick(slug, asset_id, timestamp, bid, ask)` — Records tick with midpoint
- `check_signal(slug, up_token, down_token)` — Called every second by checker thread
  - Returns `Signal(direction, token_to_buy, entry_ask, momentum)` or `None`
  - Only fires once per market (tracked by `_signal_generated` dict)
- Configuration:
  - `observation_window` (0.50 = first 50% of market)
  - `momentum_threshold` (0.03 = 3 cents of midpoint movement)
  - `min_entry_price` / `max_entry_price` — skip signals outside price bounds
  - `min_book_snapshots` — minimum ticks before generating a signal

### `bot/notifications.py` — Notification Service
- `bot_started()`, `trade_opened()`, `trade_result()`, `bot_error()`, etc.
- Currently: console-only (prints emoji-formatted messages)
- Designed for extension with webhooks (Discord, Telegram, etc.)

### `ingest/src/market_client.py` — WebSocket Client
- `PolymarketClient` — connects to `wss://ws-subscriptions-clob.polymarket.com/ws/market`
- Discovers current market via Gamma API slug patterns (`btc-updown-5m-{timestamp}`)
- Token mapping: resolves UP/DOWN token IDs from `clobTokenIds` + `outcomes`
- Market rotation: detects when current market ends, fetches next, resubscribes
- Fires callbacks: `on_event(data: bytes)`, `on_market_open(MarketInfo)`, `on_market_close(MarketInfo)`

### `shared/polymarket_pb2.py` — Protobuf Definitions
- Generated from `polymarket.proto`
- `MarketEvent` with `book` field containing `BookSnapshot`
- `BookSnapshot.bids` / `BookSnapshot.asks` — price levels with size

## Configuration

### `config.json` — API Credentials
```json
{
  "polymarket": {
    "host": "https://clob.polymarket.com",
    "private_key": "0x...",
    "proxy_wallet": "0x...",
    "api_key": "...",
    "api_secret": "...",
    "api_passphrase": "...",
    "chain_id": 137
  }
}
```

### `bot_config.json` — Trading Parameters
```json
{
  "strategy": {
    "observation_window": 0.50,
    "momentum_threshold": 0.03,
    "min_entry_price": 0.25,
    "max_entry_price": 0.84,
    "min_book_snapshots": 20
  },
  "trading": {
    "position_size": 5.0,
    "autoscale_enabled": false,
    "autoscale_fraction": 0.05,
    "max_daily_loss": 50.0,
    "dry_run": false
  },
  "notifications": {
    "backend": "console"
  }
}
```

**Parameter notes:**
- `position_size`: Fixed dollar amount per trade ($5 currently)
- `observation_window`: 0.50 means observe for 2.5 min of a 5-min market
- `momentum_threshold`: 0.03 means require 3-cent midpoint movement
- `min_entry_price`/`max_entry_price`: Skip trades where the ask is outside this range (avoids extreme favorites/longshots)
- `max_daily_loss`: Stop trading after this cumulative daily loss

## Running

```bash
# Live trading
.venv/bin/python3 momentum_bot.py --live

# Paper trading (dry run)
.venv/bin/python3 momentum_bot.py --dry-run

# Backtesting (separate script)
.venv/bin/python3 momentum_backtest.py
```

## Key Design Decisions & Bug Fixes

### 1. Limit Orders Instead of Market Orders
**Problem**: `py-clob-client`'s `create_market_order` has a rounding bug — the `ROUNDING_CONFIG` allows 4 decimal places for taker amounts, but the CLOB API only accepts 2.

**Fix**: Use `create_order` with `OrderArgs` (limit order at the best ask). We control rounding explicitly:
- Price: `round(price, 2)` 
- Size (shares): `math.floor(amount / price * 100) / 100`

### 2. Multiple Concurrent Positions
**Problem**: Originally only supported one open position — bot blocked waiting for resolution.

**Fix**: `positions: dict[str, Position]` keyed by market slug. Thread-safe with `threading.Lock`. Allows trading new markets while previous positions await resolution.

### 3. Non-Blocking Resolution
**Problem**: Resolution polling blocked the main loop, preventing new trades.

**Fix**: Each market's resolution runs in a separate daemon thread. Polls Gamma API every 5 seconds for up to 15 minutes. Main loop continues uninterrupted.

### 4. Resolution Detection for BTC 5-Min Markets
**Problem**: These markets don't use standard UMA resolution immediately. The Gamma API takes 5-10 minutes to update.

**Fix**: Check multiple indicators:
- `umaResolutionStatus == "resolved"` (primary, but delayed)
- `closed == True`
- `outcomePrices` contains exact `"0"` and `"1"`
- `max(outcomePrices) >= 0.99` (snap to 0/1)

### 5. Order Book Sorting
**Problem**: WebSocket book snapshots arrive with bids in ascending order (worst to best).

**Fix**: Use `max()` for best bid, `min()` for best ask instead of assuming sorted order.

### 6. Auth with Proxy Wallet
**Problem**: Polymarket uses proxy wallets (`signature_type=2`). Orders failed with auth errors.

**Fix**: Set `signature_type=2` and `funder=proxy_wallet` in `ClobClient`. Derive fresh API keys on startup.

## Known Issues & Future Work

### Resolution Timing
- The Gamma API can take **5-12 minutes** to show resolution for BTC 5-min markets
- One market (`btc-updown-5m-1771862400`) timed out even at 15 minutes
- **Possible fix**: Increase timeout to 20-30 min, or use on-chain event monitoring instead of API polling

### Redemption (Placeholder)
- `redeem_positions()` in `engine.py` is a **placeholder** — it logs but doesn't actually redeem on-chain
- Polymarket appears to auto-redeem winning tokens after some time
- For faster capital turnover, implement proper on-chain redemption via `CTFExchange.redeemPositions()`
- Would need `web3.py` and the exchange contract ABI

### Strategy Refinements
- **Momentum threshold tuning**: 0.03 catches many signals, but win rate is ~50%. Consider backtesting higher thresholds
- **Price range tuning**: Currently [0.25, 0.84]. Many strong momentum signals get filtered at extremes (>0.84). Consider widening
- **Position sizing**: Currently fixed $5. The `autoscale_enabled` feature scales by balance percentage
- **Slippage guard**: Currently 20% max. Could be tightened
- **Signal timing**: Currently fires at exactly 50% of window. Could experiment with earlier/later
- **Multiple signals per market**: Currently one signal per market. Could add a "double down" if momentum strengthens

### Tick Processing
- The bot currently gets duplicate "First tick" logs because ticks from the previous market's WS channel arrive during rotation
- Not harmful but noisy — could filter by checking if the asset_id belongs to the *current* market only

### State Persistence
- State file tracks positions and daily P&L
- On crash, positions are restored and resolution checked on startup
- Unresolved positions spawn background resolver threads

### Balance Tracking
- In live mode, balance is fetched from the CLOB API
- Capital display may lag behind actual balance (auto-redemption timing)
- The bot doesn't yet account for pending position value in balance checks

## Dependencies

```
py-clob-client     # Polymarket CLOB API client
protobuf           # WebSocket message parsing
websocket-client   # WebSocket connection
requests           # Gamma API HTTP calls
httpx              # Used internally by py-clob-client
```

Install: `pip install -r requirements.txt`

## File Structure
```
wss_test/
├── momentum_bot.py          # Main bot entry point
├── momentum_backtest.py     # Backtesting script
├── bot_config.json          # Strategy & trading config
├── config.json              # API credentials
├── bot/
│   ├── __init__.py
│   ├── engine.py            # Trading engine
│   ├── strategy.py          # Momentum strategy
│   └── notifications.py     # Alert system
├── ingest/
│   └── src/
│       └── market_client.py # WebSocket + market discovery
├── shared/
│   ├── __init__.py
│   └── polymarket_pb2.py    # Protobuf definitions
├── data/
│   ├── bot_state_live.json  # Live state persistence
│   ├── bot_state_dry.json   # Dry-run state persistence
│   └── ...                  # Historical data
└── .env                     # Environment variables (optional)
```
