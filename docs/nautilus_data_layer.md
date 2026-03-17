# NautilusTrader Data Layer — Technical Reference

NautilusTrader v1.224.0 on Binance Spot. This document covers how to get data in and out of
the engine for both backtesting (historical parquet) and live/paper trading (Binance WebSocket).

---

## 1. Historical Data Loading (Parquet → QuoteTick)

### Our parquet format

```
data/btc_quotes/btcusdt_quotes.parquet  — 2.5M rows, 1-second snapshots
Columns: timestamp_ms (int64), bid_price (float64), ask_price (float64), ...
```

No bid/ask *size* columns — we'll use a constant placeholder (1.0).

### QuoteTick constructor

```python
QuoteTick(
    instrument_id: InstrumentId,   # e.g. BTCUSDT.BINANCE
    bid_price: Price,              # Price.from_str("62911.43")
    ask_price: Price,              # Price.from_str("62911.44")
    bid_size: Quantity,            # Quantity.from_str("1.00000")
    ask_size: Quantity,            # Quantity.from_str("1.00000")
    ts_event: uint64,              # UNIX nanoseconds
    ts_init: uint64,               # UNIX nanoseconds
)
```

**Precision constraints:**
- `bid_price.precision` must equal `ask_price.precision`
- `bid_size.precision` must equal `ask_size.precision`
- Timestamps are **nanoseconds**, not milliseconds

### Bulk loading — the fast path

`QuoteTick.from_raw_arrays_to_list()` is a Cython-optimized batch constructor.
Despite the parameter name `*_raw`, it takes **double arrays** (actual float prices),
not fixed-point integers. Internally it calls `Price(double_value, precision)`.

```python
import numpy as np
import pandas as pd
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue

df = pd.read_parquet("data/btc_quotes/btcusdt_quotes.parquet")
instrument_id = InstrumentId(Symbol("BTCUSDT"), Venue("BINANCE"))
n = len(df)

# milliseconds → nanoseconds
ts_nanos = (df["timestamp_ms"].values * 1_000_000).astype(np.uint64)

quotes = QuoteTick.from_raw_arrays_to_list(
    instrument_id=instrument_id,
    price_prec=2,          # BTCUSDT tick size = 0.01
    size_prec=5,           # BTCUSDT step size = 0.00001
    bid_prices_raw=np.ascontiguousarray(df["bid_price"].values, dtype=np.float64),
    ask_prices_raw=np.ascontiguousarray(df["ask_price"].values, dtype=np.float64),
    bid_sizes_raw=np.ascontiguousarray(np.ones(n), dtype=np.float64),  # placeholder
    ask_sizes_raw=np.ascontiguousarray(np.ones(n), dtype=np.float64),  # placeholder
    ts_events=np.ascontiguousarray(ts_nanos),
    ts_inits=np.ascontiguousarray(ts_nanos),
)
# 2.5M quotes created in ~0.8 seconds
```

**Critical:** all arrays must be `np.ascontiguousarray` — Cython typed memoryviews
require C-contiguous memory layout.

### Adding data to BacktestEngine

```python
engine.add_data(quotes)                    # sorts after every call (safe but slow)

# Faster for multi-asset loading:
engine.add_data(btc_quotes, sort=False)
engine.add_data(eth_quotes, sort=False)
engine.add_data(sol_quotes, sort=False)
engine.add_data(xrp_quotes, sort=False)
engine.sort_data()                         # sort once at the end
```

The `validate=True` default checks that the instrument exists in the cache before
accepting data. You must call `add_instrument()` before `add_data()`.

### Streaming large datasets

For datasets too large to fit in memory, use `add_data_iterator()`:

```python
def quote_generator():
    for chunk_file in sorted(parquet_files):
        df = pd.read_parquet(chunk_file)
        quotes = convert_to_quote_ticks(df)
        yield quotes

engine.add_data_iterator("btc_stream", quote_generator())
```

---

## 2. Instrument Creation

Binance spot pairs use the `CurrencyPair` instrument class. For backtesting, instruments
must be created manually (no live API calls). The instrument defines the precision that
all QuoteTick/Bar data must match.

```python
from decimal import Decimal
from nautilus_trader.model.instruments.currency_pair import CurrencyPair
from nautilus_trader.model.objects import Price, Quantity, Money, Currency

BINANCE = Venue("BINANCE")

btcusdt = CurrencyPair(
    instrument_id=InstrumentId(Symbol("BTCUSDT"), BINANCE),
    raw_symbol=Symbol("BTCUSDT"),
    base_currency=Currency.from_str("BTC"),     # built-in, precision=8
    quote_currency=Currency.from_str("USDT"),   # built-in, precision=8
    price_precision=2,                           # tick size = 0.01
    size_precision=5,                            # step size = 0.00001
    price_increment=Price.from_str("0.01"),
    size_increment=Quantity.from_str("0.00001"),
    ts_event=0,
    ts_init=0,
    lot_size=Quantity.from_str("0.00001"),
    max_quantity=Quantity.from_str("9000.00000"),
    min_quantity=Quantity.from_str("0.00001"),
    min_notional=Money(5.0, Currency.from_str("USDT")),
    maker_fee=Decimal("0.001"),                  # 0.1%
    taker_fee=Decimal("0.001"),                  # 0.1%
)
```

### Reference: Binance precision values

| Symbol   | price_precision | size_precision | tick_size | step_size |
|----------|----------------|---------------|-----------|-----------|
| BTCUSDT  | 2              | 5             | 0.01      | 0.00001   |
| ETHUSDT  | 2              | 4             | 0.01      | 0.0001    |
| SOLUSDT  | 2              | 2             | 0.01      | 0.01      |
| XRPUSDT  | 4              | 1             | 0.0001    | 0.1       |

Built-in currencies via `Currency.from_str()`: BTC (8), ETH (8), SOL (8), XRP (6), USDT (8).

### Venue setup

```python
engine.add_venue(
    venue=BINANCE,
    oms_type=OmsType.NETTING,
    account_type=AccountType.CASH,          # spot = CASH, not MARGIN
    starting_balances=[Money(10_000, Currency.from_str("USDT"))],
    base_currency=None,                     # multi-currency account
)
```

**Important:** `CurrencyPair` instruments require either:
- `account_type=AccountType.MARGIN`, OR
- `base_currency=None` (multi-currency cash account)

Setting `base_currency` to a specific currency with a CASH account will reject
CurrencyPair instruments with an `InvalidConfiguration` error.

---

## 3. Multi-Asset Support

All instruments share the same venue. Load data in any order, sort once.

```python
ASSETS = {
    "BTCUSDT": {"file": "data/btc_quotes/btcusdt_quotes.parquet", "pp": 2, "sp": 5},
    "ETHUSDT": {"file": "data/eth_quotes/ethusdt_quotes.parquet", "pp": 2, "sp": 4},
    "SOLUSDT": {"file": "data/sol_quotes/solusdt_quotes.parquet", "pp": 2, "sp": 2},
    "XRPUSDT": {"file": "data/xrp_quotes/xrpusdt_quotes.parquet", "pp": 4, "sp": 1},
}

for sym, cfg in ASSETS.items():
    # 1. Create and add instrument (see Section 2)
    engine.add_instrument(make_instrument(sym, cfg))

    # 2. Load quotes with sort=False for speed
    df = pd.read_parquet(cfg["file"])
    quotes = load_quotes(df, sym, cfg["pp"], cfg["sp"])
    engine.add_data(quotes, sort=False)

engine.sort_data()  # merge-sorts all streams by ts_init
```

The engine interleaves all data types (QuoteTicks, Bars, etc.) from all instruments
into a single time-ordered stream. Events are replayed in timestamp order.

---

## 4. Live Data Bridge (Binance Adapter)

### How live QuoteTicks are produced

The Binance adapter subscribes to the `@bookTicker` WebSocket stream. The message
contains best bid/ask price and size:

```json
{"s": "BTCUSDT", "u": 123, "b": "62911.43", "B": "1.50000", "a": "62911.44", "A": "2.00000", "T": 1770336001000}
```

This is parsed via `BinanceQuoteData.parse_to_quote_tick()`:
```python
QuoteTick(
    instrument_id=instrument_id,
    bid_price=Price.from_str(self.b),     # "62911.43" → Price(62911.43, precision=2)
    ask_price=Price.from_str(self.a),
    bid_size=Quantity.from_str(self.B),   # "1.50000" → Quantity(1.5, precision=5)
    ask_size=Quantity.from_str(self.A),
    ts_event=millis_to_nanos(self.T),     # milliseconds → nanoseconds
    ts_init=clock.timestamp_ns(),
)
```

### Config for live/paper trading

```python
from nautilus_trader.adapters.binance.config import BinanceDataClientConfig

data_config = BinanceDataClientConfig(
    api_key=None,       # None = public market data only (no auth needed)
    api_secret=None,    # None = public market data only
    account_type=BinanceAccountType.SPOT,
    # For paper trading on testnet:
    # environment=BinanceEnvironment.TESTNET,
)
```

**API keys are optional for public market data** (quotes, trades, order book).
The docstring says: *"If None, the client will work for public market data only."*
Keys are only needed for execution (placing orders) and querying fee tiers.

### Live node setup (for future reference)

```python
from nautilus_trader.live.node import TradingNode
from nautilus_trader.live.config import TradingNodeConfig
from nautilus_trader.adapters.binance.factories import BinanceLiveDataClientFactory

config = TradingNodeConfig(
    data_clients={
        "BINANCE": BinanceDataClientConfig(api_key=None, api_secret=None),
    },
    # exec_clients would go here for live trading
)
node = TradingNode(config=config)
node.add_data_client_factory("BINANCE", BinanceLiveDataClientFactory)
# node.build() / node.run()
```

---

## 5. Data Alignment: Backtest vs Live

### What's identical

| Property | Backtest (parquet) | Live (WebSocket) |
|----------|-------------------|------------------|
| Type | `QuoteTick` | `QuoteTick` |
| `instrument_id` | `BTCUSDT.BINANCE` | `BTCUSDT.BINANCE` |
| Price precision | 2 (from instrument) | 2 (from exchange info) |
| Size precision | 5 (from instrument) | 5 (from exchange info) |
| Timestamp unit | nanoseconds | nanoseconds |

### What differs

| Property | Backtest | Live |
|----------|---------|------|
| `bid_size` / `ask_size` | Constant 1.0 (no depth data) | Actual best level size |
| `ts_event` vs `ts_init` | Same value (historical) | `ts_event` = exchange time, `ts_init` = local recv time |
| Frequency | Fixed 1-second intervals | Variable (every book ticker update) |

### Impact on strategy behavior

- **Strategies that only use bid/ask prices** (most momentum/signal strategies) will
  behave identically in backtest and live.
- **Strategies using bid/ask sizes** (e.g., order imbalance) will see different values.
  To fix: obtain historical depth data from Binance's data dumps or collect via WebSocket.
- **Strategies sensitive to data frequency** may see different behavior since live data
  arrives at variable intervals (every ticker update, often sub-second) while historical
  data is fixed 1-second.

### Precision alignment guarantee

The instrument's `price_precision` and `size_precision` define the fixed-point scale
for all data. Since both backtest and live use the same instrument definition (either
manually created or loaded from exchange info), precisions are guaranteed to match.

---

## 6. Bar Aggregation

NautilusTrader can aggregate raw QuoteTick data into OHLCV bars natively via two modes:

### INTERNAL aggregation (recommended for our use case)

The DataEngine subscribes to quote ticks and aggregates them into bars on the fly.
This happens automatically when you subscribe to a bar type with `AggregationSource.INTERNAL`.

```python
from nautilus_trader.model.data import BarType, BarSpecification
from nautilus_trader.model.enums import BarAggregation, PriceType, AggregationSource

# In your strategy's on_start():
bar_type = BarType(
    instrument_id=InstrumentId.from_str("BTCUSDT.BINANCE"),
    bar_spec=BarSpecification(1, BarAggregation.MINUTE, PriceType.MID),
    aggregation_source=AggregationSource.INTERNAL,
)
self.subscribe_bars(bar_type)
# → DataEngine creates a TimeBarAggregator
# → Feeds QuoteTicks through it
# → Emits Bar objects to your on_bar() callback
```

**Supported time aggregations:** MILLISECOND, SECOND, MINUTE, HOUR, DAY, WEEK, MONTH, YEAR

**Non-time aggregations also supported:** TICK, VOLUME, VALUE (and their imbalance/runs variants), RENKO

**Price types:** BID, ASK, MID, LAST

This works identically in backtest and live — the same DataEngine code runs in both modes.
The aggregator subscribes to the underlying tick data and produces bars regardless of source.

### EXTERNAL aggregation

Load pre-computed bars directly. Bars must have `aggregation_source=EXTERNAL`:

```python
bar_type = BarType(
    instrument_id=instrument_id,
    bar_spec=BarSpecification(1, BarAggregation.MINUTE, PriceType.LAST),
    aggregation_source=AggregationSource.EXTERNAL,
)

bars = Bar.from_raw_arrays_to_list(
    bar_type=bar_type,
    price_prec=2,
    size_prec=5,
    opens=np.array([...], dtype=np.float64),
    highs=np.array([...], dtype=np.float64),
    lows=np.array([...], dtype=np.float64),
    closes=np.array([...], dtype=np.float64),
    volumes=np.array([...], dtype=np.float64),
    ts_events=np.array([...], dtype=np.uint64),
    ts_inits=np.array([...], dtype=np.uint64),
)

engine.add_data(bars)
```

For live, Binance natively provides kline/candlestick streams:

```python
# The adapter maps BarSpecification to Binance kline intervals:
# 1-SECOND, 1-MINUTE, 3-MINUTE, 5-MINUTE, 15-MINUTE, 30-MINUTE,
# 1-HOUR, 2-HOUR, 4-HOUR, 6-HOUR, 8-HOUR, 12-HOUR, 1-DAY, 3-DAY, 1-WEEK, 1-MONTH
```

### Recommendation

Use **INTERNAL aggregation** for the strategy evolution system. Benefits:
- Same code path in backtest and live
- No need to pre-compute bars from parquet
- Flexible: add new bar timeframes without reprocessing data
- Indicators (e.g., RSI on 5m bars) subscribe to the aggregated bars, not raw ticks

---

## 7. BacktestEngine.reset() Behavior

```python
engine.reset()
# Preserves: loaded instruments, loaded data (QuoteTick/Bar lists)
# Resets: strategies, actors, portfolio, account balances, clock, data iterator position
```

This enables the strategy evolution loop:
```python
engine = setup_engine_once()       # instruments + data loaded once

for params in param_sweep:
    engine.reset()
    engine.add_strategy(MyStrategy(params))
    engine.run()
    results = engine.get_result()
```

`clear_data()` and `clear_strategies()` are available for more granular control.

---

## 8. Complete Minimal Example

```python
from decimal import Decimal
import numpy as np
import pandas as pd
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.config import BacktestEngineConfig, LoggingConfig
from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue
from nautilus_trader.model.objects import Price, Quantity, Money, Currency
from nautilus_trader.model.instruments.currency_pair import CurrencyPair
from nautilus_trader.model.enums import AccountType, OmsType
from nautilus_trader.model.data import QuoteTick

# --- Engine setup (once) ---
engine = BacktestEngine(config=BacktestEngineConfig(
    logging=LoggingConfig(log_level="WARNING"),
))

BINANCE = Venue("BINANCE")
engine.add_venue(
    venue=BINANCE,
    oms_type=OmsType.NETTING,
    account_type=AccountType.CASH,
    starting_balances=[Money(100_000, Currency.from_str("USDT"))],
    base_currency=None,
)

# --- Instruments ---
btcusdt = CurrencyPair(
    instrument_id=InstrumentId(Symbol("BTCUSDT"), BINANCE),
    raw_symbol=Symbol("BTCUSDT"),
    base_currency=Currency.from_str("BTC"),
    quote_currency=Currency.from_str("USDT"),
    price_precision=2,
    size_precision=5,
    price_increment=Price.from_str("0.01"),
    size_increment=Quantity.from_str("0.00001"),
    ts_event=0, ts_init=0,
    maker_fee=Decimal("0.001"),
    taker_fee=Decimal("0.001"),
)
engine.add_instrument(btcusdt)

# --- Data loading ---
df = pd.read_parquet("data/btc_quotes/btcusdt_quotes.parquet")
n = len(df)
ts_nanos = (df["timestamp_ms"].values * 1_000_000).astype(np.uint64)

quotes = QuoteTick.from_raw_arrays_to_list(
    instrument_id=btcusdt.id,
    price_prec=2,
    size_prec=5,
    bid_prices_raw=np.ascontiguousarray(df["bid_price"].values, dtype=np.float64),
    ask_prices_raw=np.ascontiguousarray(df["ask_price"].values, dtype=np.float64),
    bid_sizes_raw=np.ascontiguousarray(np.ones(n), dtype=np.float64),
    ask_sizes_raw=np.ascontiguousarray(np.ones(n), dtype=np.float64),
    ts_events=np.ascontiguousarray(ts_nanos),
    ts_inits=np.ascontiguousarray(ts_nanos),
)
engine.add_data(quotes)

# --- Run backtest (add strategy first) ---
# engine.add_strategy(MyStrategy(config))
# engine.run()
```
