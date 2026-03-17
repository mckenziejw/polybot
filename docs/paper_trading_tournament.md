# Paper Trading Tournament Coordinator — Technical Design

## 1. NautilusTrader Architecture for Paper Trading

### How Paper Trading Works

NautilusTrader does not have a dedicated "paper trading" mode as a separate system state. Instead, it offers three `Environment` values:

- `BACKTEST` — fully simulated, historical data replay
- `SANDBOX` — live data feed, simulated execution (this is "paper trading")
- `LIVE` — live data feed, live execution

For paper trading, the approach is:

1. Connect **live data clients** (real Binance WebSocket feeds for market data)
2. Connect a **sandbox execution client** (orders are simulated locally, never sent to the exchange)
3. Set `environment = Environment.SANDBOX` in `TradingNodeConfig`

The Binance adapter supports `BinanceEnvironment.DEMO` which provides live market data but sandbox execution. Alternatively, `BinanceEnvironment.TESTNET` connects to Binance's testnet (separate fake-money exchange with its own API keys).

**Recommendation for POC**: Use `BinanceEnvironment.DEMO` (sandbox) — it gives real market data with simulated fills, no testnet API keys needed. The `BinanceDataClientConfig` only needs `api_key`/`api_secret` for private data; public market data works without credentials.

### Multiple Strategies on One TradingNode

Yes, NautilusTrader natively supports multiple strategies on a single `TradingNode`. The `Trader` class maintains an internal dict `_strategies: dict[StrategyId, Strategy]` and methods `add_strategy()`, `add_strategies()`, `remove_strategy()`, `start_strategy()`, `stop_strategy()`.

**Critical discovery**: The `Trader.add_strategy()` method has a guard:

```python
if self.is_running and not self._has_controller:
    self._log.error("Cannot add a strategy to a running trader")
    return
```

This means strategies **can only be added/removed at runtime if a Controller is registered**. Without a controller, strategies must be added before starting the node.

### The Controller Pattern

NautilusTrader provides `Controller` (in `nautilus_trader.trading.controller`), which is an `Actor` subclass that holds a reference to the `Trader` and can dynamically manage strategies at runtime. It provides:

- `create_strategy(strategy, start=True)` — add and optionally start a strategy
- `remove_strategy(strategy)` — stop and remove a strategy
- `start_strategy(strategy)` / `stop_strategy(strategy)`
- `market_exit_strategy(strategy)` — force-close all positions for a strategy
- `create_strategy_from_config(config, start=True)` — create from importable config

The Controller is the intended mechanism for runtime strategy management. The tournament coordinator should be implemented as a `Controller` subclass.

## 2. Strategy Lifecycle Management

### Runtime Add/Remove Flow

```
TournamentController (extends Controller)
    |
    ├── on_start():
    |       Load persisted tournament state
    |       Create initial strategies from genomes
    |       Schedule evaluation timer
    |
    ├── _evaluate_tournament() [timer callback, e.g. every 1h]:
    |       Collect per-strategy metrics
    |       Rank strategies by fitness
    |       Prune bottom N%
    |       Mutate top N%
    |       Inject new candidates from backtest lab
    |
    ├── _add_contestant(genome):
    |       strategy = GenomeStrategy(genome)
    |       self.create_strategy(strategy, start=True)
    |
    ├── _remove_contestant(strategy_id):
    |       Record final metrics
    |       self.remove_strategy(strategy)
    |
    └── on_stop():
            Persist tournament state
            Save all strategy metrics
```

### Strategy Identity

Each strategy needs a unique `StrategyId` and `order_id_tag`. The `order_id_tag` must be unique across all running strategies (enforced by `Trader.add_strategy()`). For tournament strategies, use a scheme like:

- `StrategyId`: `"Genome-{generation}_{index}"` (e.g., `Genome-012_003`)
- `order_id_tag`: `"G{gen:03d}{idx:03d}"` (e.g., `"G012003"`)

When a strategy is auto-assigned, `Trader.add_strategy()` will generate sequential tags. But for the tournament, explicit `order_id_tag` values are better because they survive restarts and can be traced back to specific genomes.

### GenomeStrategy Class

A single `Strategy` subclass that interprets a genome dict to drive its trading logic:

```
class GenomeStrategyConfig(StrategyConfig):
    genome: dict           # The genome parameters
    generation: int        # GA generation number
    parent_ids: list[str]  # Parent genome IDs for genealogy
    oms_type: str = "HEDGING"  # Each strategy tracks its own positions

class GenomeStrategy(Strategy):
    def __init__(self, config: GenomeStrategyConfig):
        super().__init__(config)
        self.genome = config.genome
        self.generation = config.generation
        self.parent_ids = config.parent_ids
        # Build indicators from genome
        # Build entry/exit rules from genome
```

## 3. Per-Strategy Performance Tracking

### Cache-Level Filtering

The NautilusTrader `Cache` (Cython) supports filtering orders and positions by `strategy_id`. Key methods:

```python
cache.orders(strategy_id=strategy_id)           # All orders for a strategy
cache.orders_open(strategy_id=strategy_id)       # Open orders
cache.orders_closed(strategy_id=strategy_id)     # Closed orders
cache.positions(strategy_id=strategy_id)         # All positions
cache.positions_open(strategy_id=strategy_id)    # Open positions
cache.positions_closed(strategy_id=strategy_id)  # Closed positions
cache.orders_total_count(strategy_id=strategy_id)
cache.positions_open_count(strategy_id=strategy_id)
```

This means per-strategy performance can be computed at any time by querying the cache with the strategy's ID.

### Per-Strategy Analyzer

NautilusTrader's `PortfolioAnalyzer` works at the portfolio level, not per-strategy. For per-strategy metrics, the tournament coordinator creates **one `PortfolioAnalyzer` per strategy** and feeds it the strategy-filtered positions:

```python
def _compute_strategy_metrics(self, strategy_id: StrategyId) -> dict:
    positions = self.cache.positions(strategy_id=strategy_id)
    closed_positions = self.cache.positions_closed(strategy_id=strategy_id)

    analyzer = PortfolioAnalyzer()
    # Register desired statistics (Sharpe, drawdown, etc.)
    analyzer.add_positions(closed_positions)

    return {
        "strategy_id": str(strategy_id),
        "total_trades": len(closed_positions),
        "open_positions": len(positions) - len(closed_positions),
        "realized_pnls": analyzer.realized_pnls(),
        "returns": analyzer.returns(),
        "pnl_stats": analyzer.get_performance_stats_pnls(),
        "return_stats": analyzer.get_performance_stats_returns(),
        "general_stats": analyzer.get_performance_stats_general(),
    }
```

### Custom Statistics

The `PortfolioStatistic` base class allows plugging in custom metrics. Implement these for the tournament fitness function:

| Statistic | Source | Method |
|-----------|--------|--------|
| Sharpe ratio | Returns series | `calculate_from_returns()` |
| Profit factor | Realized PnLs | `calculate_from_realized_pnls()` |
| Max drawdown | Returns series | `calculate_from_returns()` |
| Win rate | Positions | `calculate_from_positions()` |
| Avg trade duration | Positions | `calculate_from_positions()` |
| Trade count | Positions | `calculate_from_positions()` |

### Metrics Snapshot Timing

The coordinator collects metrics snapshots on a timer (e.g., every 15 minutes) and stores them in a time-indexed structure. This allows computing metrics over any rolling window (1h, 24h, 7d).

```python
@dataclass
class MetricsSnapshot:
    timestamp: datetime
    strategy_id: str
    genome_id: str
    generation: int
    n_trades: int
    n_open: int
    realized_pnl: float
    sharpe: float | None
    max_drawdown: float | None
    win_rate: float | None
    profit_factor: float | None
```

## 4. Tournament Cycle Design

### Timing

| Window | Duration | Purpose |
|--------|----------|---------|
| Warm-up | 2 hours | Strategy subscribes to data, builds indicator state. No evaluation. |
| Snapshot interval | 15 minutes | Collect metrics snapshot for each strategy |
| Short eval | 6 hours | Kill strategies that have zero trades or are catastrophically losing |
| Primary eval | 24 hours | Main ranking cycle — prune + mutate + inject |
| Confirmation | 72 hours | Sustained performance required for promotion flagging |

These are shorter than the original design doc's 24h/1w/2w windows because Binance spot gives far more trading opportunities per unit time than Polymarket binary markets.

### Fitness Function

```python
def compute_fitness(metrics: MetricsSnapshot, window_hours: float) -> float:
    # Minimum trade count gate
    min_trades = max(5, int(window_hours / 4))  # ~1 trade per 4 hours minimum
    if metrics.n_trades < min_trades:
        return -1000.0  # Insufficient data penalty

    # Primary: Sharpe ratio (annualized from returns)
    sharpe = metrics.sharpe or 0.0

    # Secondary adjustments
    drawdown_penalty = max(0, (metrics.max_drawdown or 0) - 0.05) * 10  # Penalize >5% DD
    trade_diversity = min(1.0, metrics.n_trades / (min_trades * 3))  # Bonus for more trades

    fitness = sharpe - drawdown_penalty + trade_diversity * 0.5

    return fitness
```

### Tournament Cycle (runs every `primary_eval` hours)

```
1. SNAPSHOT
   For each active strategy:
     - Query cache for positions/orders filtered by strategy_id
     - Compute fitness over the evaluation window
     - Store snapshot

2. RANK
   Sort strategies by fitness (descending)

3. PRUNE (bottom 25%)
   For each strategy in bottom quartile:
     - If strategy.age < warm_up_period: skip (grace period)
     - Call self.market_exit_strategy(strategy)  # Close positions
     - Call self.remove_strategy(strategy)        # Deregister
     - Record in graveyard (genome + final metrics)

4. MUTATE (top 25%)
   For each strategy in top quartile:
     - Clone genome
     - Apply mutation operators (from GA engine)
     - Create new GenomeStrategy with mutated genome
     - Call self.create_strategy(new_strategy, start=True)

5. INJECT (1-2 from backtest lab)
   - Read top candidates from backtest_lab_queue (file/Redis/SQLite)
   - Create GenomeStrategy instances
   - Call self.create_strategy(injected_strategy, start=True)

6. PERSIST
   - Write tournament state to disk (JSON/SQLite)
   - Write metrics history to Parquet
```

### Population Sizing

Target: 20-30 active strategies at any time on one TradingNode. Rationale:

- Each strategy subscribes to the same data feeds (shared, not duplicated)
- Each strategy maintains its own indicator instances (memory cost)
- 30 strategies with ~5 indicators each = ~150 indicator instances — well within memory budget
- Order submission is simulated (sandbox mode), so no rate limit concerns

### State Persistence (Survive Restarts)

Tournament state is persisted to `data/tournament/` directory:

```
data/tournament/
├── state.json              # Current tournament state
├── genomes/                # All genome definitions
│   ├── G001_000.json
│   ├── G001_001.json
│   └── ...
├── metrics.parquet         # Time-series metrics snapshots
├── genealogy.json          # Parent-child relationships
└── graveyard.parquet       # Pruned strategies + final metrics
```

**`state.json`** contains:
```json
{
    "tournament_id": "t_20260312_001",
    "started_at": "2026-03-12T00:00:00Z",
    "generation": 5,
    "active_strategies": [
        {"genome_id": "G005_000", "strategy_id": "Genome-005_000", "added_at": "..."},
        ...
    ],
    "evaluation_count": 12,
    "last_eval_at": "2026-03-12T18:00:00Z"
}
```

On restart:
1. Controller `on_start()` reads `state.json`
2. Recreates `GenomeStrategy` instances from `genomes/*.json`
3. Re-adds them to the trader via `create_strategy()`
4. Strategies warm up naturally as data flows in
5. Metrics history is preserved in Parquet (appended to, not overwritten)

## 5. Strategy Isolation / Sandboxing

### Position Isolation via OmsType.HEDGING

Set `oms_type = "HEDGING"` in every `GenomeStrategyConfig`. In hedging mode, the execution engine assigns a unique `PositionId` per order, so each strategy's positions are tracked independently. Two strategies can hold opposing positions on the same instrument without netting.

With `OmsType.NETTING`, positions on the same instrument would be aggregated across strategies, which is exactly what we do NOT want.

### Order ID Isolation

Each strategy has a unique `order_id_tag`, which is embedded in every `ClientOrderId` it generates. This provides:
- Order attribution: `cache.orders(strategy_id=X)` filters by the strategy's order index
- No cross-contamination: even in sandbox mode, the execution engine routes fills only to the strategy that submitted the order

### Data Feed Sharing (No Isolation Needed)

All strategies subscribe to the same instruments via the same data client. NautilusTrader's `DataEngine` handles fan-out — when a `QuoteTick` or `TradeTick` arrives, it is published on the `MessageBus` and all subscribed strategies receive it. This is efficient (data is not duplicated) and correct (all strategies see the same market data).

### Resource Limits

To prevent a single rogue strategy from consuming excessive resources:
- **Max open orders per strategy**: Enforced in `GenomeStrategy.on_order_accepted()` — if count exceeds limit, cancel oldest
- **Max open positions per strategy**: Part of the genome (`max_positions` field), enforced in the strategy's entry logic
- **Position size cap**: Genome's `position_size` is bounded during mutation (e.g., max 1% of simulated account)

### What Sandbox Mode Does NOT Do

Sandbox mode simulates order fills locally (instant fill at current price, or with a configurable fill model). It does NOT:
- Send any orders to Binance
- Interact with any real account balances
- Provide realistic slippage or partial fills (without custom fill model)

This is fine for tournament evaluation (relative ranking), but promotion thresholds must account for live execution degradation, which the existing design doc already addresses.

## 6. Dashboard / Monitoring

### Data to Surface

#### Strategy Rankings Table (primary view)

| Rank | Strategy ID | Gen | Age | Trades | WR% | PnL | Sharpe | DD% | Fitness | Status |
|------|-------------|-----|-----|--------|-----|-----|--------|-----|---------|--------|
| 1 | Genome-005_002 | 5 | 47h | 23 | 68.2 | +$142 | 1.83 | 2.1 | 2.34 | RUNNING |
| 2 | Genome-004_007 | 4 | 71h | 31 | 61.3 | +$98 | 1.52 | 3.4 | 1.87 | RUNNING |
| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |
| 20 | Genome-005_011 | 5 | 4h | 1 | 100 | +$3 | N/A | 0.0 | -999 | WARM_UP |

#### Genealogy Graph

Track parent-child relationships:
```json
{
    "G005_002": {
        "parents": ["G004_003", "G004_007"],
        "operator": "crossover+mutation",
        "mutation_params": {"indicator_perturb": 0.08},
        "created_at": "2026-03-12T10:00:00Z"
    }
}
```

Visualize as a tree/DAG showing which lineages produce winning strategies.

#### Performance Over Time

Time-series chart per strategy showing:
- Cumulative PnL curve
- Rolling Sharpe (e.g., 6h rolling window)
- Trade markers (entry/exit points)

#### Tournament Activity Log

Event stream:
```
[2026-03-12 18:00:01] EVAL #12: Ranked 24 strategies
[2026-03-12 18:00:02] PRUNE: Removed Genome-003_011 (fitness=-2.3, 0 trades in 24h)
[2026-03-12 18:00:02] PRUNE: Removed Genome-004_001 (fitness=-0.8, Sharpe=-1.2)
[2026-03-12 18:00:03] MUTATE: Created Genome-006_000 from Genome-005_002 (perturb RSI period 14→16)
[2026-03-12 18:00:03] INJECT: Added Genome-006_001 from backtest lab (backtest Sharpe=2.1)
[2026-03-12 18:00:04] Population: 24 → 22 (pruned 6, added 4)
```

### Implementation

The dashboard should use the same HTTP+SSE pattern as the existing execution dashboard (port 3456). Add a separate endpoint:

- `GET /tournament` — HTML page with rankings table + charts
- `GET /tournament/api/rankings` — JSON rankings
- `GET /tournament/api/metrics/:strategy_id` — JSON metrics time series
- `GET /tournament/api/genealogy` — JSON genealogy tree
- `GET /tournament/api/events` — SSE stream of tournament events
- `POST /tournament/api/pause` — Pause evaluation cycles
- `POST /tournament/api/inject` — Manually inject a genome

## 7. Concrete Implementation Plan

### File Structure

```
evolution/
├── __init__.py
├── genome.py                 # Genome dataclass + serialization
├── genome_strategy.py        # GenomeStrategy(Strategy) + GenomeStrategyConfig
├── tournament_controller.py  # TournamentController(Controller) — main coordinator
├── fitness.py                # Fitness function + per-strategy analyzer wrapper
├── mutation.py               # GA operators (mutate, crossover, select)
├── persistence.py            # Save/load tournament state (JSON + Parquet)
├── dashboard.py              # HTTP/SSE dashboard actor
└── config.py                 # TournamentConfig dataclass
```

### Phase 2 Implementation Steps

1. **GenomeStrategy** — Strategy subclass that reads a genome dict and configures indicators + entry/exit rules. Start with a minimal genome (EMA crossover + RSI filter + configurable TP/SL) to validate the pipeline.

2. **TournamentController** — Controller subclass with:
   - `on_start()`: load state, create initial population, schedule eval timer
   - `_evaluate_tournament()`: timer callback implementing the full cycle
   - `_compute_strategy_fitness()`: query cache, run PortfolioAnalyzer
   - `on_stop()`: persist state

3. **Binance paper trading setup** — TradingNode config with:
   ```python
   config = TradingNodeConfig(
       trader_id="TOURNAMENT-001",
       environment=Environment.SANDBOX,
       data_clients={
           "BINANCE": BinanceDataClientConfig(
               account_type=BinanceAccountType.SPOT,
               environment=BinanceEnvironment.DEMO,
               instrument_provider=BinanceInstrumentProviderConfig(
                   load_ids=frozenset([
                       InstrumentId.from_str("BTCUSDT.BINANCE"),
                       InstrumentId.from_str("ETHUSDT.BINANCE"),
                       InstrumentId.from_str("SOLUSDT.BINANCE"),
                       InstrumentId.from_str("XRPUSDT.BINANCE"),
                   ]),
               ),
           ),
       },
       exec_clients={
           "BINANCE": BinanceExecClientConfig(
               account_type=BinanceAccountType.SPOT,
               environment=BinanceEnvironment.DEMO,
           ),
       },
   )
   ```

4. **GA operators** — Port mutation/crossover from the backtest GA (Phase 1) to work on live genomes.

5. **Persistence** — JSON for state, Parquet for metrics, survive restarts.

6. **Dashboard** — SSE-based tournament monitor.

### Key Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Memory growth with many strategies + indicators | Cap population at 30; periodically purge closed orders/positions via `LiveExecEngineConfig.purge_closed_*` settings |
| Strategies interfering via shared cache | OmsType.HEDGING + unique order_id_tag ensures isolation at the engine level |
| Controller timer drift / missed evaluations | Use NautilusTrader's built-in timer (precise, drift-compensated). Log every evaluation cycle. |
| Tournament state corruption on crash | Atomic writes (write to temp file, then rename). Keep last 3 state snapshots. |
| Strategies that never trade (dead weight) | Short eval window (6h) kills zero-trade strategies. Also penalize in fitness function. |
| Sandbox fills unrealistically optimistic | Tournament is for relative ranking only. Promotion threshold (already defined in design doc) applies a 1.5-2x multiplier to account for live degradation. |

### Configuration

```python
@dataclass
class TournamentConfig:
    # Population
    initial_population: int = 20
    max_population: int = 30
    min_population: int = 10

    # Timing
    warm_up_hours: float = 2.0
    snapshot_interval_minutes: int = 15
    short_eval_hours: float = 6.0
    primary_eval_hours: float = 24.0
    confirmation_hours: float = 72.0

    # Tournament operations
    prune_fraction: float = 0.25       # Bottom 25% pruned each cycle
    mutate_fraction: float = 0.25      # Top 25% cloned + mutated
    inject_count: int = 2              # New genomes from backtest lab per cycle

    # Fitness
    min_trades_per_4h: int = 1
    max_drawdown_pct: float = 10.0     # Hard kill above this

    # Persistence
    state_dir: str = "data/tournament"
    metrics_flush_interval_minutes: int = 60

    # Instruments
    instruments: list[str] = field(default_factory=lambda: [
        "BTCUSDT.BINANCE",
        "ETHUSDT.BINANCE",
        "SOLUSDT.BINANCE",
        "XRPUSDT.BINANCE",
    ])
```
