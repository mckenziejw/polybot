# Strategy Evolution System

Continuous strategy optimization using genetic algorithms, paper trading tournaments, and automatic live promotion. Built on NautilusTrader's unified backtest/paper/live architecture.

## Status

**Phase 1 POC complete** (March 12, 2026). All modules implemented and tested end-to-end:
- Data loading (Binance parquet → NautilusTrader QuoteTicks)
- Genome encoding/decoding with full serialization
- GA operators (mutation, crossover, tournament selection, evolution)
- Fitness evaluation via NautilusTrader BacktestEngine
- GA runner with persistence and resume capability
- Tournament coordinator with pruning, mutation, injection
- GenomeStrategy adapter (genome → live NautilusTrader Strategy)

### POC Test Results
- **Single genome evaluation**: RSI mean-reversion on 6h BTC data → 3 trades, 66.7% WR, $2.30 PnL, Sharpe 6.70
- **Mini evolution**: 3 genomes × 2 generations on 3h BTC data → best genome: 42 trades, 47.6% WR, Sharpe 0.87
- **Tournament lifecycle**: prune/mutate/inject/leaderboard all functional

## Core Concept

Three layers running simultaneously:

1. **Live bot** — one active strategy trading real money
2. **Paper tournament** — N paper-trading strategies on the same live data feed
3. **Backtest lab** — genetic algorithm exploring new strategy variants offline

Strategies flow upward: backtest lab → paper tournament → live (if they beat the threshold).

## File Map

```
evolution/
├── __init__.py
├── data/
│   ├── __init__.py
│   └── binance_loader.py      # Parquet → QuoteTick, engine setup, BarType creation
├── ga/
│   ├── __init__.py
│   ├── genome.py               # Genome, IndicatorGene, EntryRule, ExitRule dataclasses
│   ├── operators.py            # random_genome, mutate, crossover, tournament_select, evolve_population
│   ├── fitness.py              # evaluate_genome, compute_fitness, FitnessResult
│   └── runner.py               # run_evolution loop, EvolutionState persistence, CLI
├── strategies/
│   ├── __init__.py
│   └── genome_strategy.py      # GenomeStrategy(Strategy) — genome → NautilusTrader bridge
├── tournament/
│   ├── __init__.py
│   └── coordinator.py          # TournamentCoordinator, StrategyRecord, TournamentConfig
└── tests/
    ├── __init__.py
    └── test_poc.py             # End-to-end POC tests (run: python -m evolution.tests.test_poc)
```

## Genome Representation

A genome encodes a complete trading strategy. Implemented in `evolution/ga/genome.py`.

```python
Genome(
    symbols=["BTCUSDT"],
    bar_period_minutes=5,              # 1, 5, or 15
    indicators=[                        # 2-6 indicators
        IndicatorGene(type=IndicatorType.RSI, params={"period": 14}),
        IndicatorGene(type=IndicatorType.BB, params={"period": 20, "k": 2.0}),
    ],
    entry_rules=[                       # 1-3 rules (any triggers a trade)
        EntryRule(indicator_id="RSI_14", condition=Condition.LESS_THAN, value=0.30, side=Side.BUY),
    ],
    exit_rules=[                        # 1-3 rules + mandatory stop_loss + time_limit
        ExitRule(type="take_profit", value=2.0),   # percent
        ExitRule(type="stop_loss", value=1.0),      # percent
        ExitRule(type="time_limit", value=3600),    # seconds
        ExitRule(type="indicator", value=0.70, indicator_id="RSI_14", condition=Condition.CROSSES_ABOVE),
    ],
    position_size_usd=100.0,
    max_open_positions=1,
    max_drawdown_pct=15.0,
)
```

### Available Indicators (15 types)

| Category | Indicators |
|----------|-----------|
| Averages | EMA, SMA, HMA, DEMA, WMA |
| Momentum | RSI, Stochastics, ROC, CCI, CMO |
| Volatility | BollingerBands, ATR, DonchianChannel, KeltnerChannel, VHF |

**Important**: NautilusTrader indicators output **0-1 range** (not 0-100 like traditional). All thresholds in entry/exit rules use this scale.

### Entry Conditions

- `GREATER_THAN` / `LESS_THAN` — threshold comparison
- `CROSSES_ABOVE` / `CROSSES_BELOW` — cross detection (requires prev bar)
- `CROSSOVER_ABOVE` / `CROSSOVER_BELOW` — two-indicator crossover

### Exit Types

- `take_profit` — close when PnL% exceeds value
- `stop_loss` — close when PnL% drops below -value
- `time_limit` — close after N seconds (converted to bars)
- `indicator` — close on indicator condition

## Fitness Function

Primary: **Sharpe ratio** (annualized). Implemented in `evolution/ga/fitness.py`.

```
fitness = sharpe
    + 0.1 * log(1 + pnl)        if pnl > 0
    + 0.2 * (win_rate - 0.5)    if win_rate > 55%
    - 0.5 * (dd - 15) / 10      if max_drawdown > 15%
    = -2 * (1 - trades/min)     if trades < min_trades  (penalty)
```

Metrics extracted from NautilusTrader's positions report: realized PnL, win rate, Sharpe, Sortino, profit factor, max drawdown.

## GA Operators

Implemented in `evolution/ga/operators.py`.

| Operator | Details |
|----------|---------|
| **Selection** | Tournament (k=3 default) |
| **Crossover** | Uniform on indicators, filter rules for valid refs |
| **Mutation** | Gaussian perturbation (10% of range), add/remove indicators (rare) |
| **Elitism** | Top 10% survive unchanged |
| **Population** | Configurable, default 30 per generation |

### Evolution Loop

```
for each generation:
    evaluate all genomes (BacktestEngine per genome)
    sort by fitness
    save best genome
    next_gen = elite (10%) + crossover (60%) + mutation (30%)
```

State is persisted to `evolution_state.json` after each generation for crash recovery.

## Paper Trading Tournament

Implemented in `evolution/tournament/coordinator.py`.

### Lifecycle

Each cycle:
1. **Rank** all strategies by composite fitness
2. **Prune** bottom 25% (if older than 24h minimum)
3. **Clone + mutate** top 25% of survivors
4. **Inject** top genomes from GA backtest lab
5. **Fill** remaining slots with random genomes

### Configuration

```python
TournamentConfig(
    max_strategies=20,      # max concurrent paper traders
    min_strategies=10,      # refill threshold
    eval_interval_s=3600,   # evaluate every hour
    min_age_hours=24,       # minimum age before prune-eligible
    prune_pct=0.25,         # prune bottom 25%
    mutation_rate=0.15,     # mutation rate for offspring
    inject_count=2,         # inject N from GA lab per cycle
)
```

### Fitness for Tournament Ranking

```
fitness = sharpe + 0.1 (if pnl > 0) - 0.5 (if drawdown > 15%)
fitness = -10.0 if n_trades < 5  (not enough data)
```

## Promotion Threshold (Future)

Based on observed paper-to-live degradation:

```
Historical degradation estimates:
- Win rate: -5 to -17 percentage points
- PnL/trade: -30% to -100% (spread/slippage costs)
- Sharpe: ~50% reduction

Promotion threshold:
  paper_sharpe > live_sharpe * 1.5  AND
  paper_pnl_per_trade > live_pnl_per_trade * 2.0  AND
  paper_n_trades > 50  AND
  paper_max_drawdown < account_risk_limit
```

## Running

### Quick Test (POC)
```bash
.venv/bin/python3 -m evolution.tests.test_poc
```

### GA Evolution Run
```bash
.venv/bin/python3 -m evolution.ga.runner \
    --data-dir data/ \
    --output-dir evolution/runs/ \
    --symbols BTCUSDT \
    --pop-size 30 \
    --generations 20

# Resume from checkpoint:
.venv/bin/python3 -m evolution.ga.runner \
    --resume evolution/runs/evolution_state.json
```

### Data Requirements

Binance 1-second quote data in parquet format:
```
data/
├── btc_quotes/btcusdt_quotes.parquet   # 2.6M rows, 720 hours
├── eth_quotes/ethusdt_quotes.parquet
├── sol_quotes/solusdt_quotes.parquet
└── xrp_quotes/xrpusdt_quotes.parquet
```

Required columns: `timestamp_ms`, `bid_price`, `ask_price`

## Known Issues & Next Steps

### Known Issues
- **Cash account short-selling**: NautilusTrader CASH account type doesn't support short positions well (errors on deducting base currency PnL). Switch to MARGIN account or restrict to long-only for spot.
- **No parallelism yet**: Genome evaluations run sequentially. Could use multiprocessing for 10-30x speedup.
- **Indicator value range**: Some indicators (CCI, ROC) can output outside 0-1. Entry rules need indicator-aware thresholds.

### Phase 2: Paper Tournament on Live Data
- Connect NautilusTrader to live Binance WebSocket feed
- Run N GenomeStrategy instances on shared TradingNode
- Wire tournament coordinator to periodic evaluation timer
- Dashboard showing tournament standings + strategy genealogy

### Phase 3: Live Integration
- Implement promotion threshold + swap protocol
- Risk guardrails (daily loss limit, position limits, correlation check)
- Initially: manual approval for promotions
- Eventually: fully autonomous with rollback triggers

### Phase 4: Multi-Venue + Polymarket
- Port data loader to Polymarket BinaryOption format
- Cross-venue strategies (hedge binary with perp)
- Expand genome to support multi-leg positions
