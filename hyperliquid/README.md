# Hyperliquid Trading Framework

Infrastructure for real-time market data, order execution, and delta-neutral funding rate arbitrage on Hyperliquid.

## Components

### `config.py`
Configuration via environment variables or dict. Defaults to **testnet** for safety.

```bash
export HYPERLIQUID_PRIVATE_KEY="0x..."   # only needed for trading
export HYPERLIQUID_ADDRESS="0x..."       # for read-only queries
export HYPERLIQUID_TESTNET=1             # default: testnet
```

### `client.py`
REST API client with EIP-712 signing for authenticated exchange operations.

```python
from hyperliquid.client import HyperliquidClient

# Read-only
client = HyperliquidClient()
mids = client.get_mids()
book = client.get_l2("BTC")

# With trading
client = HyperliquidClient(private_key="0x...", testnet=True)
client.place_order("BTC", is_buy=True, price=60000.0, size=0.01)
client.cancel_order("BTC", order_id=12345)
```

### `websocket_client.py`
WebSocket subscription manager with auto-reconnect and keepalive.

```python
from hyperliquid.websocket_client import HyperliquidWS

ws = HyperliquidWS()
ws.on("l2Book", lambda data: print(data))
ws.subscribe_l2("BTC")
ws.run_forever()
```

### `funding_monitor.py`
Real-time funding rate monitor. Streams mid prices via WebSocket, polls funding rates via REST, logs to CSV, alerts on high-yield opportunities.

```bash
python -m hyperliquid.funding_monitor --threshold 30 --coins BTC ETH SOL
```

### `arb_executor.py`
Delta-neutral funding arb execution. Opens long spot + short perp (or vice versa) to collect funding while staying market-neutral.

**Defaults to DRY-RUN mode** -- logs intended actions without executing.

```bash
# Dry-run (default)
python -m hyperliquid.arb_executor --coin BTC --size 1000

# Live trading (requires --live flag + private key)
python -m hyperliquid.arb_executor --coin BTC --size 1000 --live --testnet
```

## Dependencies

- `requests` -- HTTP client
- `websocket-client` -- WebSocket
- `eth_account` -- EIP-712 signing
