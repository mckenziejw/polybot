# Genetic Algorithm Engine — Technical Design

## 1. Genome Representation

The genome encodes a complete trading strategy as a nested dataclass tree. Every field is either a numeric value (evolvable via mutation) or a categorical choice (evolvable via swap). The design prioritizes composability: any combination of indicators and rules should produce a *valid* (if not profitable) strategy.

### 1.1 Core Dataclasses

```python
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Literal
import random
import uuid


# ── Asset selection ──────────────────────────────────────────────────

class Asset(Enum):
    BTCUSDT = "BTCUSDT.BINANCE"
    ETHUSDT = "ETHUSDT.BINANCE"
    SOLUSDT = "SOLUSDT.BINANCE"
    XRPUSDT = "XRPUSDT.BINANCE"


# ── Bar aggregation ──────────────────────────────────────────────────

class BarPeriod(Enum):
    S10  = "10-SECOND"
    S30  = "30-SECOND"
    M1   = "1-MINUTE"
    M5   = "5-MINUTE"
    M15  = "15-MINUTE"
    H1   = "1-HOUR"


# ── Moving average type (maps to NautilusTrader MovingAverageType) ───

class MAType(Enum):
    SIMPLE = auto()
    EXPONENTIAL = auto()
    WILDER = auto()
    HULL = auto()
    DOUBLE_EXPONENTIAL = auto()


# ── Indicator genes ──────────────────────────────────────────────────

@dataclass
class IndicatorGene:
    """Base for all indicator genes. Each produces one or more named output values."""
    enabled: bool = True


@dataclass
class SMAGene(IndicatorGene):
    period: int = 20          # range: [5, 200]
    # Output: "sma_{period}" → float (the moving average value)

@dataclass
class EMAGene(IndicatorGene):
    period: int = 12          # range: [5, 200]
    # Output: "ema_{period}" → float

@dataclass
class RSIGene(IndicatorGene):
    period: int = 14          # range: [5, 50]
    ma_type: MAType = MAType.EXPONENTIAL
    # Output: "rsi" → float [0, 1]  (NautilusTrader RSI is 0–1, not 0–100)

@dataclass
class MACDGene(IndicatorGene):
    fast_period: int = 12     # range: [5, 30]
    slow_period: int = 26     # range: [15, 60], must be > fast_period
    ma_type: MAType = MAType.EXPONENTIAL
    # Output: "macd" → float (fast_ma - slow_ma)

@dataclass
class BollingerGene(IndicatorGene):
    period: int = 20          # range: [10, 50]
    k: float = 2.0            # range: [1.0, 3.5]
    ma_type: MAType = MAType.SIMPLE
    # Outputs: "bb_upper", "bb_middle", "bb_lower" → float

@dataclass
class ATRGene(IndicatorGene):
    period: int = 14          # range: [5, 50]
    ma_type: MAType = MAType.SIMPLE
    # Output: "atr" → float

@dataclass
class StochasticsGene(IndicatorGene):
    period_k: int = 14        # range: [5, 30]
    period_d: int = 3         # range: [2, 10]
    # Outputs: "stoch_k", "stoch_d" → float [0, 100]

@dataclass
class DonchianGene(IndicatorGene):
    period: int = 20          # range: [10, 60]
    # Outputs: "dc_upper", "dc_middle", "dc_lower" → float

@dataclass
class KeltnerGene(IndicatorGene):
    period: int = 20          # range: [10, 50]
    k_multiplier: float = 1.5 # range: [0.5, 3.0]
    ma_type: MAType = MAType.EXPONENTIAL
    # Outputs: "kc_upper", "kc_middle", "kc_lower" → float

@dataclass
class AroonGene(IndicatorGene):
    period: int = 25          # range: [10, 50]
    # Outputs: "aroon_up", "aroon_down", "aroon_osc" → float

@dataclass
class LinearRegressionGene(IndicatorGene):
    period: int = 20          # range: [10, 100]
    # Outputs: "linreg_value", "linreg_slope", "linreg_r2" → float

@dataclass
class ROCGene(IndicatorGene):
    period: int = 10          # range: [2, 50]
    # Output: "roc" → float (rate of change)

@dataclass
class CMOGene(IndicatorGene):
    period: int = 14          # range: [5, 50]
    ma_type: MAType = MAType.WILDER
    # Output: "cmo" → float [-100, 100]

@dataclass
class CCIGene(IndicatorGene):
    period: int = 20          # range: [10, 50]
    scalar: float = 0.015     # fixed, standard CCI scalar
    # Output: "cci" → float

@dataclass
class OBVGene(IndicatorGene):
    period: int = 0           # range: [0, 50], 0 = no windowing
    # Output: "obv" → float

@dataclass
class PressureGene(IndicatorGene):
    period: int = 14          # range: [5, 50]
    # Output: "pressure" → float


# ── Condition/Rule genes ─────────────────────────────────────────────

class Comparator(Enum):
    GT = ">"       # greater than
    LT = "<"       # less than
    CROSS_ABOVE = "cross_above"
    CROSS_BELOW = "cross_below"


class ConditionRHS(Enum):
    """What the indicator output is compared against."""
    CONSTANT = auto()      # compare to a fixed threshold
    INDICATOR = auto()     # compare to another indicator's output
    PRICE = auto()         # compare to current close price


@dataclass
class ConditionGene:
    """A single boolean condition: left_output COMPARATOR right_value."""
    left_output: str         # name of indicator output, e.g. "rsi", "ema_12"
    comparator: Comparator = Comparator.GT
    rhs_type: ConditionRHS = ConditionRHS.CONSTANT
    constant_value: float = 0.5    # used when rhs_type == CONSTANT, range varies by indicator
    right_output: str = ""         # used when rhs_type == INDICATOR, e.g. "sma_50"


class RuleCombinator(Enum):
    AND = auto()    # all conditions must be true
    OR  = auto()    # any condition must be true


@dataclass
class RuleGene:
    """A composable entry or exit rule: combinator over 1-4 conditions."""
    conditions: list[ConditionGene] = field(default_factory=list)
    combinator: RuleCombinator = RuleCombinator.AND


# ── Exit mechanism genes ─────────────────────────────────────────────

@dataclass
class ExitGene:
    """Fixed exit mechanisms (always active, independent of indicator rules)."""
    take_profit_pct: float = 0.02     # range: [0.005, 0.10]
    stop_loss_pct: float = 0.01       # range: [0.003, 0.05]
    trailing_stop_pct: float = 0.0    # range: [0.0, 0.05], 0 = disabled
    time_limit_bars: int = 0          # range: [0, 500], 0 = disabled
    exit_rule: RuleGene | None = None # optional indicator-based exit


# ── Position sizing genes ────────────────────────────────────────────

class SizingMethod(Enum):
    FIXED_QUANTITY = auto()      # trade N units
    FIXED_FRACTION = auto()     # trade X% of account equity
    ATR_SCALED = auto()          # size inversely proportional to ATR


@dataclass
class SizingGene:
    method: SizingMethod = SizingMethod.FIXED_FRACTION
    fixed_quantity: float = 0.001       # range: [0.0001, 0.1] BTC-scale
    fixed_fraction: float = 0.02        # range: [0.005, 0.10] of equity
    atr_risk_pct: float = 0.01          # range: [0.005, 0.05] of equity risked per ATR
    max_positions: int = 1              # range: [1, 5]


# ── Top-level genome ─────────────────────────────────────────────────

@dataclass
class Genome:
    genome_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    generation: int = 0
    parent_ids: list[str] = field(default_factory=list)

    # What to trade and at what resolution
    assets: list[Asset] = field(default_factory=lambda: [Asset.BTCUSDT])
    bar_period: BarPeriod = BarPeriod.M5

    # Indicator bank (all possible indicators; enabled flag controls which are active)
    sma_fast: SMAGene = field(default_factory=lambda: SMAGene(period=10))
    sma_slow: SMAGene = field(default_factory=lambda: SMAGene(period=50))
    ema_fast: EMAGene = field(default_factory=lambda: EMAGene(period=12))
    ema_slow: EMAGene = field(default_factory=lambda: EMAGene(period=26))
    rsi: RSIGene = field(default_factory=RSIGene)
    macd: MACDGene = field(default_factory=MACDGene)
    bollinger: BollingerGene = field(default_factory=BollingerGene)
    atr: ATRGene = field(default_factory=ATRGene)
    stochastics: StochasticsGene = field(default_factory=StochasticsGene)
    donchian: DonchianGene = field(default_factory=DonchianGene)
    keltner: KeltnerGene = field(default_factory=KeltnerGene)
    aroon: AroonGene = field(default_factory=AroonGene)
    linreg: LinearRegressionGene = field(default_factory=LinearRegressionGene)
    roc: ROCGene = field(default_factory=ROCGene)
    cmo: CMOGene = field(default_factory=CMOGene)
    cci: CCIGene = field(default_factory=CCIGene)
    obv: OBVGene = field(default_factory=OBVGene)
    pressure: PressureGene = field(default_factory=PressureGene)

    # Entry rules (long and short)
    entry_long: RuleGene = field(default_factory=RuleGene)
    entry_short: RuleGene = field(default_factory=RuleGene)

    # Exit configuration
    exit_config: ExitGene = field(default_factory=ExitGene)

    # Position sizing
    sizing: SizingGene = field(default_factory=SizingGene)
```

### 1.2 Parameter Ranges Reference Table

| Indicator Gene | Parameter | Min | Max | Type | Mutation Step |
|---|---|---|---|---|---|
| SMAGene | period | 5 | 200 | int | gaussian(0, 15) |
| EMAGene | period | 5 | 200 | int | gaussian(0, 15) |
| RSIGene | period | 5 | 50 | int | gaussian(0, 5) |
| MACDGene | fast_period | 5 | 30 | int | gaussian(0, 3) |
| MACDGene | slow_period | 15 | 60 | int | gaussian(0, 5) |
| BollingerGene | period | 10 | 50 | int | gaussian(0, 5) |
| BollingerGene | k | 1.0 | 3.5 | float | gaussian(0, 0.3) |
| ATRGene | period | 5 | 50 | int | gaussian(0, 5) |
| StochasticsGene | period_k | 5 | 30 | int | gaussian(0, 3) |
| StochasticsGene | period_d | 2 | 10 | int | gaussian(0, 1) |
| DonchianGene | period | 10 | 60 | int | gaussian(0, 5) |
| KeltnerGene | period | 10 | 50 | int | gaussian(0, 5) |
| KeltnerGene | k_multiplier | 0.5 | 3.0 | float | gaussian(0, 0.3) |
| AroonGene | period | 10 | 50 | int | gaussian(0, 5) |
| LinearRegressionGene | period | 10 | 100 | int | gaussian(0, 10) |
| ROCGene | period | 2 | 50 | int | gaussian(0, 5) |
| CMOGene | period | 5 | 50 | int | gaussian(0, 5) |
| CCIGene | period | 10 | 50 | int | gaussian(0, 5) |
| OBVGene | period | 0 | 50 | int | gaussian(0, 5) |
| PressureGene | period | 5 | 50 | int | gaussian(0, 5) |
| ExitGene | take_profit_pct | 0.005 | 0.10 | float | gaussian(0, 0.005) |
| ExitGene | stop_loss_pct | 0.003 | 0.05 | float | gaussian(0, 0.003) |
| ExitGene | trailing_stop_pct | 0.0 | 0.05 | float | gaussian(0, 0.005) |
| ExitGene | time_limit_bars | 0 | 500 | int | gaussian(0, 30) |
| SizingGene | fixed_fraction | 0.005 | 0.10 | float | gaussian(0, 0.01) |
| SizingGene | atr_risk_pct | 0.005 | 0.05 | float | gaussian(0, 0.005) |
| SizingGene | max_positions | 1 | 5 | int | uniform_int(-1, 1) |

### 1.3 Condition Constant Ranges by Indicator

Conditions compare indicator outputs to thresholds. The valid range for `constant_value` depends on which indicator output is being compared:

| left_output | Semantic range | Typical threshold range |
|---|---|---|
| `rsi` | [0, 1] | [0.2, 0.8] |
| `stoch_k`, `stoch_d` | [0, 100] | [10, 90] |
| `cmo` | [-100, 100] | [-50, 50] |
| `cci` | unbounded, centered ~0 | [-200, 200] |
| `aroon_up`, `aroon_down` | [0, 100] | [20, 80] |
| `aroon_osc` | [-100, 100] | [-50, 50] |
| `macd` | unbounded, centered ~0 | [-price*0.01, price*0.01] |
| `linreg_slope` | unbounded | [-1, 1] |
| `linreg_r2` | [0, 1] | [0.3, 0.9] |
| `roc` | unbounded, centered ~0 | [-0.05, 0.05] |
| `pressure` | unbounded, centered ~0 | [-2, 2] |
| MA/BB/KC/DC outputs | price-scale | compared to other indicators, not constants |

---

## 2. Genome-to-Strategy Adapter

The adapter translates a `Genome` into a NautilusTrader `Strategy` subclass instance. Key design decisions:

### 2.1 Architecture

```python
class GeneticStrategy(Strategy):
    """
    A NautilusTrader Strategy that is fully parameterized by a Genome.
    One class, many instances with different genomes.
    """

    def __init__(self, config: GeneticStrategyConfig):
        super().__init__(config)
        self.genome: Genome = config.genome
        self._indicators: dict[str, Indicator] = {}   # name → indicator instance
        self._prev_values: dict[str, float] = {}       # for cross_above/cross_below detection
        self._bar_count: int = 0
        self._entry_bar: dict[str, int] = {}           # position_id → bar count at entry

    def on_start(self):
        # 1. Build indicator instances from genome
        # 2. Register each enabled indicator for the appropriate bar type
        # 3. Subscribe to bar data for each asset
        for asset in self.genome.assets:
            instrument_id = InstrumentId.from_str(asset.value)
            bar_type = BarType.from_str(f"{asset.value}-{self.genome.bar_period.value}-BID-EXTERNAL")

            if self.genome.rsi.enabled:
                rsi = RelativeStrengthIndex(self.genome.rsi.period)
                self._indicators["rsi"] = rsi
                self.register_indicator_for_bars(bar_type, rsi)

            if self.genome.bollinger.enabled:
                bb = BollingerBands(self.genome.bollinger.period, self.genome.bollinger.k)
                self._indicators["bb"] = bb
                self.register_indicator_for_bars(bar_type, bb)

            # ... repeat for all indicator genes ...

            self.subscribe_bars(bar_type)

    def on_bar(self, bar: Bar):
        self._bar_count += 1

        # Skip until all indicators initialized
        if not all(ind.initialized for ind in self._indicators.values()):
            return

        # Collect current indicator output values into a dict
        values = self._collect_indicator_values()

        # Check fixed exits first (TP/SL/trailing/time)
        self._check_fixed_exits(bar)

        # Check indicator-based exit rule
        if self.genome.exit_config.exit_rule:
            self._check_rule_exit(values, bar)

        # Check entry rules
        if self._can_open_position():
            if self._evaluate_rule(self.genome.entry_long, values):
                self._enter_long(bar)
            elif self._evaluate_rule(self.genome.entry_short, values):
                self._enter_short(bar)

        # Store previous values for crossover detection
        self._prev_values = values.copy()

    def _collect_indicator_values(self) -> dict[str, float]:
        """Read all indicator outputs into a flat dict."""
        values = {}
        if "rsi" in self._indicators:
            values["rsi"] = self._indicators["rsi"].value
        if "bb" in self._indicators:
            bb = self._indicators["bb"]
            values["bb_upper"] = bb.upper
            values["bb_middle"] = bb.middle
            values["bb_lower"] = bb.lower
        # ... etc for all multi-output indicators ...
        return values

    def _evaluate_rule(self, rule: RuleGene, values: dict[str, float]) -> bool:
        """Evaluate a rule gene against current indicator values."""
        if not rule.conditions:
            return False

        results = []
        for cond in rule.conditions:
            left = values.get(cond.left_output)
            if left is None:
                results.append(False)
                continue

            # Determine right-hand side value
            if cond.rhs_type == ConditionRHS.CONSTANT:
                right = cond.constant_value
            elif cond.rhs_type == ConditionRHS.INDICATOR:
                right = values.get(cond.right_output, 0.0)
            elif cond.rhs_type == ConditionRHS.PRICE:
                right = self._last_close
            else:
                results.append(False)
                continue

            # Evaluate comparison
            if cond.comparator == Comparator.GT:
                results.append(left > right)
            elif cond.comparator == Comparator.LT:
                results.append(left < right)
            elif cond.comparator == Comparator.CROSS_ABOVE:
                prev_left = self._prev_values.get(cond.left_output, left)
                results.append(prev_left <= right and left > right)
            elif cond.comparator == Comparator.CROSS_BELOW:
                prev_left = self._prev_values.get(cond.left_output, left)
                results.append(prev_left >= right and left < right)

        if rule.combinator == RuleCombinator.AND:
            return all(results)
        else:  # OR
            return any(results)
```

### 2.2 Indicator Output Name Registry

Each indicator gene produces named outputs. The adapter maps gene field names to NautilusTrader indicator instances using a registry:

```python
INDICATOR_REGISTRY = {
    "sma_fast":   lambda g: ("sma_fast",  SimpleMovingAverage(g.sma_fast.period),  [("sma_fast", "value")]),
    "sma_slow":   lambda g: ("sma_slow",  SimpleMovingAverage(g.sma_slow.period),  [("sma_slow", "value")]),
    "ema_fast":   lambda g: ("ema_fast",  ExponentialMovingAverage(g.ema_fast.period), [("ema_fast", "value")]),
    "ema_slow":   lambda g: ("ema_slow",  ExponentialMovingAverage(g.ema_slow.period), [("ema_slow", "value")]),
    "rsi":        lambda g: ("rsi",       RelativeStrengthIndex(g.rsi.period),     [("rsi", "value")]),
    "macd":       lambda g: ("macd",      MovingAverageConvergenceDivergence(g.macd.fast_period, g.macd.slow_period), [("macd", "value")]),
    "bollinger":  lambda g: ("bb",        BollingerBands(g.bollinger.period, g.bollinger.k), [("bb_upper","upper"),("bb_middle","middle"),("bb_lower","lower")]),
    "atr":        lambda g: ("atr",       AverageTrueRange(g.atr.period),          [("atr", "value")]),
    "stochastics":lambda g: ("stoch",     Stochastics(g.stochastics.period_k, g.stochastics.period_d), [("stoch_k","value_k"),("stoch_d","value_d")]),
    "donchian":   lambda g: ("dc",        DonchianChannel(g.donchian.period),      [("dc_upper","upper"),("dc_middle","middle"),("dc_lower","lower")]),
    "keltner":    lambda g: ("kc",        KeltnerChannel(g.keltner.period, g.keltner.k_multiplier), [("kc_upper","upper"),("kc_middle","middle"),("kc_lower","lower")]),
    "aroon":      lambda g: ("aroon",     AroonOscillator(g.aroon.period),         [("aroon_up","aroon_up"),("aroon_down","aroon_down"),("aroon_osc","value")]),
    "linreg":     lambda g: ("linreg",    LinearRegression(g.linreg.period),       [("linreg_value","value"),("linreg_slope","slope"),("linreg_r2","R2")]),
    "roc":        lambda g: ("roc",       RateOfChange(g.roc.period),              [("roc", "value")]),
    "cmo":        lambda g: ("cmo",       ChandeMomentumOscillator(g.cmo.period),  [("cmo", "value")]),
    "cci":        lambda g: ("cci",       CommodityChannelIndex(g.cci.period),     [("cci", "value")]),
    "obv":        lambda g: ("obv",       OnBalanceVolume(g.obv.period),           [("obv", "value")]),
    "pressure":   lambda g: ("pressure",  Pressure(g.pressure.period),             [("pressure", "value")]),
}
```

### 2.3 Order Execution

The adapter uses NautilusTrader's order factory for all execution:

- **Entry**: `MarketOrder` via `self.order_factory.market(...)` then `self.submit_order(order)`
- **TP/SL**: Submitted as `OrderList` with `contingency_type=OUO` (one-updates-other) at entry time
- **Trailing stop**: `StopMarketOrder` with `trigger_type=LAST_TRADE` and `trailing_offset`
- **Time-based exit**: `self.clock.set_timer(...)` callback that cancels + market-exits

### 2.4 Validity Constraints

The adapter enforces constraints that the GA operators can violate:

1. **MACD fast < slow**: if mutated to fast >= slow, swap them
2. **Conditions reference enabled indicators only**: if a condition references a disabled indicator's output, the condition evaluates to `False` (never fires)
3. **At least one entry condition**: if all conditions are removed via mutation, generate one random condition
4. **stop_loss < take_profit**: if violated, set stop_loss = take_profit * 0.5

---

## 3. Fitness Function

### 3.1 Primary Metric: Risk-Adjusted Composite

NautilusTrader's `PortfolioAnalyzer` provides these stats after each run (via `engine.get_result().stats_pnls` and `engine.get_result().stats_returns`):

- **Sharpe Ratio** (252 days annualized) — from `stats_returns`
- **Sortino Ratio** (252 days) — from `stats_returns`
- **Profit Factor** — from `stats_returns`
- **Win Rate** — from `stats_pnls`
- **Expectancy** — from `stats_pnls`
- **Max Drawdown** — from account balances via analyzer
- **Total PnL** — from `stats_pnls`

### 3.2 Composite Fitness Score

```python
@dataclass
class FitnessResult:
    sharpe: float = 0.0
    sortino: float = 0.0
    profit_factor: float = 0.0
    total_pnl: float = 0.0
    win_rate: float = 0.0
    expectancy: float = 0.0
    max_drawdown_pct: float = 0.0
    num_trades: int = 0
    train_sharpe: float = 0.0   # for overfitting detection
    test_sharpe: float = 0.0

    @property
    def composite_score(self) -> float:
        """
        Single scalar fitness value for ranking genomes.
        Penalize insufficient trades and excessive drawdown.
        """
        # Minimum trade count gate: need statistical significance
        if self.num_trades < 30:
            return -10.0 + (self.num_trades / 30.0)  # gradient toward trading

        # Drawdown death penalty
        if self.max_drawdown_pct > 0.25:
            return -5.0

        # Core score: weighted combination
        score = (
            0.40 * self.sharpe
            + 0.20 * self.sortino
            + 0.15 * min(self.profit_factor, 5.0)  # cap outliers
            + 0.10 * (self.win_rate * 10.0)         # scale [0,1] → [0,10]
            + 0.15 * min(self.expectancy * 100, 5.0)
        )

        # Overfitting penalty: if test performance degrades > 50% vs train
        if self.train_sharpe > 0 and self.test_sharpe < self.train_sharpe * 0.5:
            overfit_ratio = self.test_sharpe / max(self.train_sharpe, 0.001)
            score *= max(overfit_ratio, 0.1)  # severe penalty but don't zero out

        # Drawdown drag (continuous, not just death penalty)
        if self.max_drawdown_pct > 0.10:
            score *= (1.0 - (self.max_drawdown_pct - 0.10))

        return score
```

### 3.3 Extracting Metrics from NautilusTrader

```python
def extract_fitness(engine: BacktestEngine) -> FitnessResult:
    result = engine.get_result()
    analyzer = engine.kernel.portfolio.analyzer

    # PnL stats are keyed by currency code
    pnl_stats = result.stats_pnls.get("USDT", {})
    ret_stats = result.stats_returns

    return FitnessResult(
        sharpe=ret_stats.get("Sharpe Ratio (252 days)", 0.0),
        sortino=ret_stats.get("Sortino Ratio (252 days)", 0.0),
        profit_factor=ret_stats.get("Profit Factor", 0.0),
        total_pnl=pnl_stats.get("PnL (total)", 0.0),
        win_rate=pnl_stats.get("Win Rate", 0.0),
        expectancy=pnl_stats.get("Expectancy", 0.0),
        max_drawdown_pct=_compute_max_drawdown(analyzer),
        num_trades=result.total_positions,
    )
```

---

## 4. Evolution Operators

### 4.1 Selection: Tournament Selection

```python
def tournament_select(population: list[Genome], fitnesses: dict[str, FitnessResult], k: int = 3) -> Genome:
    """Select one genome via tournament selection (k contestants)."""
    contestants = random.sample(list(zip(population, [fitnesses[g.genome_id] for g in population])), k)
    winner = max(contestants, key=lambda x: x[1].composite_score)
    return winner[0]
```

**Parameters**: k=3 (standard). Larger k increases selection pressure; smaller k maintains diversity.

### 4.2 Crossover: Structured Uniform Crossover

Since the genome is a nested dataclass (not a flat vector), crossover operates at the gene-block level:

```python
def crossover(parent_a: Genome, parent_b: Genome) -> tuple[Genome, Genome]:
    """
    Uniform crossover at the gene-block level.
    Each top-level gene block (indicator, rule, exit, sizing) is inherited
    from one parent with 50% probability.
    """
    child_a = Genome(parent_ids=[parent_a.genome_id, parent_b.genome_id])
    child_b = Genome(parent_ids=[parent_a.genome_id, parent_b.genome_id])

    gene_fields = [
        "assets", "bar_period",
        "sma_fast", "sma_slow", "ema_fast", "ema_slow",
        "rsi", "macd", "bollinger", "atr", "stochastics",
        "donchian", "keltner", "aroon", "linreg", "roc", "cmo", "cci", "obv", "pressure",
        "entry_long", "entry_short", "exit_config", "sizing",
    ]

    for field_name in gene_fields:
        if random.random() < 0.5:
            setattr(child_a, field_name, deepcopy(getattr(parent_a, field_name)))
            setattr(child_b, field_name, deepcopy(getattr(parent_b, field_name)))
        else:
            setattr(child_a, field_name, deepcopy(getattr(parent_b, field_name)))
            setattr(child_b, field_name, deepcopy(getattr(parent_a, field_name)))

    return child_a, child_b
```

For **RuleGene crossover**, conditions are mixed from both parents:

```python
def crossover_rules(rule_a: RuleGene, rule_b: RuleGene) -> RuleGene:
    """Mix conditions from two rules."""
    all_conditions = rule_a.conditions + rule_b.conditions
    n = max(1, min(len(all_conditions), 4))  # 1-4 conditions
    selected = random.sample(all_conditions, min(n, len(all_conditions)))
    combinator = random.choice([rule_a.combinator, rule_b.combinator])
    return RuleGene(conditions=deepcopy(selected), combinator=combinator)
```

### 4.3 Mutation Operators

Mutation rates and specific operators per gene type:

#### 4.3.1 Numeric Parameter Mutation (per-gene probability: 15%)

```python
def mutate_int(value: int, min_val: int, max_val: int, sigma: float) -> int:
    """Gaussian perturbation, clamped to valid range."""
    return max(min_val, min(max_val, value + int(random.gauss(0, sigma))))

def mutate_float(value: float, min_val: float, max_val: float, sigma: float) -> float:
    return max(min_val, min(max_val, value + random.gauss(0, sigma)))
```

#### 4.3.2 Categorical Mutation (per-gene probability: 10%)

```python
def mutate_enum(value: Enum, enum_class: type) -> Enum:
    """Swap to a random different enum value."""
    options = [e for e in enum_class if e != value]
    return random.choice(options) if options else value
```

#### 4.3.3 Indicator Enable/Disable (per-indicator probability: 8%)

```python
def mutate_indicator_toggle(gene: IndicatorGene) -> None:
    gene.enabled = not gene.enabled
```

#### 4.3.4 Rule Mutation (per-rule probability: 12%)

Three sub-operations chosen uniformly:

1. **Add condition** (if < 4 conditions): generate a random condition referencing an enabled indicator
2. **Remove condition** (if > 1 condition): remove a random condition
3. **Mutate condition**: change comparator, threshold, or referenced indicator output

```python
def mutate_rule(rule: RuleGene, available_outputs: list[str]) -> None:
    if not rule.conditions:
        rule.conditions.append(random_condition(available_outputs))
        return

    op = random.choice(["add", "remove", "mutate", "swap_combinator"])

    if op == "add" and len(rule.conditions) < 4:
        rule.conditions.append(random_condition(available_outputs))
    elif op == "remove" and len(rule.conditions) > 1:
        rule.conditions.pop(random.randrange(len(rule.conditions)))
    elif op == "mutate":
        cond = random.choice(rule.conditions)
        sub_op = random.choice(["comparator", "threshold", "output"])
        if sub_op == "comparator":
            cond.comparator = mutate_enum(cond.comparator, Comparator)
        elif sub_op == "threshold":
            cond.constant_value = mutate_float(cond.constant_value, -200, 200, cond.constant_value * 0.15)
        elif sub_op == "output":
            cond.left_output = random.choice(available_outputs)
    elif op == "swap_combinator":
        rule.combinator = RuleCombinator.OR if rule.combinator == RuleCombinator.AND else RuleCombinator.AND
```

#### 4.3.5 Asset/Bar Period Mutation (probability: 5%)

```python
def mutate_assets(genome: Genome) -> None:
    op = random.choice(["add", "remove", "swap"])
    if op == "add" and len(genome.assets) < 4:
        new_asset = random.choice([a for a in Asset if a not in genome.assets])
        genome.assets.append(new_asset)
    elif op == "remove" and len(genome.assets) > 1:
        genome.assets.pop(random.randrange(len(genome.assets)))
    elif op == "swap":
        genome.bar_period = mutate_enum(genome.bar_period, BarPeriod)
```

### 4.4 Mutation Rate Summary

| Component | Probability per genome | Details |
|---|---|---|
| Each numeric parameter | 15% | Gaussian perturbation (10% relative sigma) |
| Each categorical (MA type, etc.) | 10% | Swap to random other value |
| Each indicator enable/disable | 8% | Toggle on/off |
| Entry long rule | 12% | Add/remove/mutate condition |
| Entry short rule | 12% | Add/remove/mutate condition |
| Exit rule | 10% | Add/remove/mutate condition |
| Exit fixed params (TP/SL) | 15% | Gaussian perturbation |
| Sizing method | 5% | Swap to random other method |
| Assets / bar period | 5% | Add/remove asset or swap period |
| Combinator (AND/OR) | 5% | Flip |

### 4.5 Elitism

Top 10% of the population survive unchanged to the next generation. These are deep-copied, not shared references.

---

## 5. Population Management & Engine Reuse

### 5.1 Evaluation Strategy: Sequential per Genome

**Key finding from NautilusTrader source**: `BacktestEngine.add_strategy()` calls `self._kernel.trader.add_strategy(strategy)`, and `reset()` preserves loaded data but clears all strategies' state. Multiple strategies CAN run simultaneously on the same engine — they all receive the same data events. However, for the GA use case, **running one strategy per engine run is cleaner**:

1. Strategies share the same account, which creates PnL attribution problems
2. Concurrent strategies' orders can interfere (same instrument, same venue)
3. There is no per-strategy account isolation in the backtest engine

**Decision: One genome per engine run, clear between runs.**

### 5.2 Evaluation Loop Pseudocode

```python
def evaluate_generation(
    engine: BacktestEngine,
    population: list[Genome],
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
) -> dict[str, FitnessResult]:
    """
    Evaluate all genomes in a generation.
    Data stays loaded across runs via engine.reset().
    """
    results = {}

    for genome in population:
        # ── Train period ─────────────────────────────
        engine.reset()               # clears strategies, accounts, positions; keeps data
        engine.clear_strategies()

        strategy = build_strategy(genome)
        engine.add_strategy(strategy)
        engine.run(start=train_start, end=train_end)

        train_fitness = extract_fitness(engine)

        # ── Test period (walk-forward) ───────────────
        engine.reset()
        engine.clear_strategies()

        strategy = build_strategy(genome)  # fresh instance
        engine.add_strategy(strategy)
        engine.run(start=test_start, end=test_end)

        test_fitness = extract_fitness(engine)

        # ── Combine ──────────────────────────────────
        combined = FitnessResult(
            sharpe=test_fitness.sharpe,         # fitness is on TEST data
            sortino=test_fitness.sortino,
            profit_factor=test_fitness.profit_factor,
            total_pnl=test_fitness.total_pnl,
            win_rate=test_fitness.win_rate,
            expectancy=test_fitness.expectancy,
            max_drawdown_pct=test_fitness.max_drawdown_pct,
            num_trades=test_fitness.num_trades,
            train_sharpe=train_fitness.sharpe,  # for overfit penalty
            test_sharpe=test_fitness.sharpe,
        )
        results[genome.genome_id] = combined

    return results
```

### 5.3 Performance Estimate

With ~2.5M rows per asset and 1-second quote data:
- 30 days of data for 1 asset: ~2.5M events
- `engine.run()` throughput: ~500K-1M events/sec (NautilusTrader benchmark)
- Per genome evaluation (train + test): ~5-10 seconds
- Population of 50: ~4-8 minutes per generation
- 100 generations: ~7-13 hours total

This is acceptable for overnight runs. For faster iteration, reduce to M5 bars only (data size drops ~300x).

### 5.4 Data Loading

Data is loaded once before the evolution loop starts:

```python
# Load data once
engine = BacktestEngine()
engine.add_venue(...)           # BINANCE venue config
engine.add_instrument(...)      # BTCUSDT instrument spec

# Load 1-second quote ticks from parquet
quotes = load_quote_ticks_from_parquet("data/binance_btcusdt_quotes.parquet")
engine.add_data(quotes)

# This data persists across all engine.reset() calls
```

---

## 6. Overfitting Protection

### 6.1 Walk-Forward Validation

The 30-day dataset is split into overlapping train/test windows:

```
Window 1: Train days 1-20,  Test days 21-25
Window 2: Train days 5-25,  Test days 26-30
Window 3: Train days 1-15,  Test days 16-20  (different regime)
```

**Primary evaluation** uses Window 1 (test on out-of-sample data). The overfit penalty in the fitness function uses `train_sharpe` vs `test_sharpe`.

### 6.2 Multiple Walk-Forward Folds (for top candidates)

For the top 20% of each generation, run all 3 windows and average the test scores. This is 3x more expensive but only applies to ~10 genomes. A genome must be profitable on at least 2 of 3 folds to survive.

### 6.3 Minimum Trade Count

The fitness function assigns a heavily negative score to genomes with < 30 trades. This prevents the GA from converging on "do nothing" or "trade once with perfect timing" strategies.

### 6.4 Complexity Penalty

Count the number of enabled indicators and conditions. Penalize genomes with > 8 enabled indicators or > 6 total conditions (Occam's razor — simpler strategies generalize better):

```python
def complexity_penalty(genome: Genome) -> float:
    n_indicators = sum(1 for gene in get_indicator_genes(genome) if gene.enabled)
    n_conditions = len(genome.entry_long.conditions) + len(genome.entry_short.conditions)
    if genome.exit_config.exit_rule:
        n_conditions += len(genome.exit_config.exit_rule.conditions)

    penalty = 0.0
    if n_indicators > 8:
        penalty += 0.05 * (n_indicators - 8)
    if n_conditions > 6:
        penalty += 0.10 * (n_conditions - 6)
    return penalty
```

### 6.5 Regime Diversity

Track the correlation between a genome's equity curve and BTC price. Strategies that are simply "buy and hold with leverage" (correlation > 0.95) are penalized, since they add no alpha.

### 6.6 Population Diversity Maintenance

Monitor Hamming distance across genomes (enabled indicator bitmask). If diversity drops below threshold (average pairwise distance < 20% of genome length), inject 5 completely random genomes.

---

## 7. Full Evolution Loop Pseudocode

```python
def run_evolution(
    data_path: str,
    n_generations: int = 100,
    population_size: int = 50,
    elitism_pct: float = 0.10,
    crossover_rate: float = 0.70,
    mutation_rate: float = 1.0,  # mutations applied to all non-elites
):
    # ── Setup ────────────────────────────────────────
    engine = setup_engine(data_path)
    population = [random_genome() for _ in range(population_size)]
    history: list[dict] = []   # generation-level stats for monitoring

    for gen in range(n_generations):
        # ── Evaluate ─────────────────────────────────
        fitnesses = evaluate_generation(engine, population, TRAIN_START, TRAIN_END, TEST_START, TEST_END)

        # ── Rank ─────────────────────────────────────
        ranked = sorted(population, key=lambda g: fitnesses[g.genome_id].composite_score, reverse=True)

        # ── Log generation stats ─────────────────────
        scores = [fitnesses[g.genome_id].composite_score for g in ranked]
        history.append({
            "generation": gen,
            "best_score": scores[0],
            "median_score": scores[len(scores)//2],
            "worst_score": scores[-1],
            "best_genome": ranked[0].genome_id,
            "best_sharpe": fitnesses[ranked[0].genome_id].sharpe,
            "best_trades": fitnesses[ranked[0].genome_id].num_trades,
            "avg_diversity": compute_diversity(population),
        })
        print(f"Gen {gen}: best={scores[0]:.3f} median={scores[len(scores)//2]:.3f} "
              f"sharpe={fitnesses[ranked[0].genome_id].sharpe:.2f} "
              f"trades={fitnesses[ranked[0].genome_id].num_trades}")

        # ── Multi-fold validation for top candidates ─
        if gen % 5 == 0:  # every 5th generation
            for genome in ranked[:int(population_size * 0.2)]:
                multi_fold_validate(engine, genome)

        # ── Elitism ──────────────────────────────────
        n_elite = max(1, int(population_size * elitism_pct))
        next_gen = [deepcopy(g) for g in ranked[:n_elite]]

        # ── Fill rest via selection + crossover + mutation
        while len(next_gen) < population_size:
            if random.random() < crossover_rate and len(next_gen) < population_size - 1:
                parent_a = tournament_select(ranked, fitnesses, k=3)
                parent_b = tournament_select(ranked, fitnesses, k=3)
                child_a, child_b = crossover(parent_a, parent_b)
                mutate(child_a)
                mutate(child_b)
                next_gen.extend([child_a, child_b])
            else:
                parent = tournament_select(ranked, fitnesses, k=3)
                child = deepcopy(parent)
                mutate(child)
                next_gen.append(child)

        # ── Diversity injection ──────────────────────
        if compute_diversity(next_gen) < 0.2:
            for _ in range(5):
                next_gen[-(random.randrange(n_elite, len(next_gen)))] = random_genome()

        # ── Trim to population size ──────────────────
        population = next_gen[:population_size]
        for g in population:
            g.generation = gen + 1

    # ── Return best genome ───────────────────────────
    return ranked[0], history
```

---

## 8. Output & Persistence

### 8.1 Genome Serialization

Genomes are serialized to JSON for storage and analysis:

```python
import json
from dataclasses import asdict

def save_genome(genome: Genome, path: str):
    d = asdict(genome)
    # Convert enums to strings
    with open(path, "w") as f:
        json.dump(d, f, indent=2, default=str)
```

### 8.2 Leaderboard

Each generation's top 10 genomes are persisted to `data/ga_results/gen_{N}.json` with their fitness scores. A separate leaderboard file tracks the all-time best genomes across runs.

### 8.3 Genealogy Tracking

Each genome records `parent_ids`, enabling visualization of which lineages produce the best strategies. This helps identify whether convergence is from one dominant lineage (premature convergence) or diverse recombination.

---

## 9. Implementation Phases

### Phase 1 (MVP): Single-asset, bar-only, basic indicators
- BTC/USDT only, M5 bars
- 6 indicators: SMA, EMA, RSI, MACD, Bollinger, ATR
- Fixed TP/SL exits only
- Population 30, 50 generations
- Single train/test split

### Phase 2: Full indicator set + rule complexity
- All 18 indicators enabled
- Composable conditions with crossover detection
- Trailing stops and time-based exits
- Walk-forward with 3 folds
- Population 50, 100 generations

### Phase 3: Multi-asset + quote-tick strategies
- All 4 Binance pairs
- Quote tick handlers (bid/ask spread strategies)
- ATR-scaled position sizing
- Complexity penalty and diversity injection

### Phase 4: Paper tournament integration
- Top GA genomes automatically injected into paper tournament
- Shared `GeneticStrategy` class used in both backtest and live contexts
