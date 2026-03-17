"""
Fitness evaluation for the genetic algorithm.

Runs a genome through the NautilusTrader backtest engine and computes
fitness metrics (Sharpe, PnL, win rate, drawdown, etc.).

Fitness functions are pluggable via the FitnessFunction protocol.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.config import BacktestEngineConfig, LoggingConfig
from nautilus_trader.model.identifiers import Venue

from evolution.data.binance_loader import BINANCE, make_bar_type, setup_backtest_engine
from evolution.ga.genome import Genome
from evolution.strategies.genome_strategy import GenomeStrategy, GenomeStrategyConfig


@dataclass
class FitnessResult:
    """Complete fitness evaluation result for a genome."""
    genome_id: str
    fitness: float          # composite fitness score
    sharpe: float           # Sharpe ratio
    total_pnl: float        # total PnL in USD
    n_trades: int           # number of completed trades
    win_rate: float         # fraction of winning trades
    avg_pnl: float          # average PnL per trade
    max_drawdown_pct: float # maximum drawdown percentage
    profit_factor: float    # gross profit / gross loss
    sortino: float          # Sortino ratio
    avg_bars_held: float    # average holding period in bars
    error: str = ""         # non-empty if evaluation failed

    def to_dict(self) -> dict:
        return {
            "genome_id": self.genome_id,
            "fitness": self.fitness,
            "sharpe": self.sharpe,
            "total_pnl": self.total_pnl,
            "n_trades": self.n_trades,
            "win_rate": self.win_rate,
            "avg_pnl": self.avg_pnl,
            "max_drawdown_pct": self.max_drawdown_pct,
            "profit_factor": self.profit_factor,
            "sortino": self.sortino,
            "avg_bars_held": self.avg_bars_held,
            "error": self.error,
        }

    @property
    def summary(self) -> str:
        if self.error:
            return f"{self.genome_id}: ERROR - {self.error}"
        return (
            f"{self.genome_id}: fitness={self.fitness:.3f}, "
            f"sharpe={self.sharpe:.2f}, pnl=${self.total_pnl:.2f}, "
            f"trades={self.n_trades}, wr={self.win_rate:.1%}, "
            f"dd={self.max_drawdown_pct:.1f}%"
        )


class FitnessFunction(ABC):
    """
    Abstract base class for fitness functions.

    Subclass this to create custom fitness scoring. The evaluate() method
    receives raw metrics and returns a composite fitness score.
    """

    @abstractmethod
    def compute(
        self,
        sharpe: float,
        sortino: float,
        total_pnl: float,
        n_trades: int,
        win_rate: float,
        max_drawdown_pct: float,
        profit_factor: float,
        avg_pnl: float,
    ) -> float:
        """Compute composite fitness from raw metrics. Higher is better."""
        ...

    @property
    def name(self) -> str:
        return self.__class__.__name__


class SharpeWeightedFitness(FitnessFunction):
    """
    Balanced fitness: Sharpe capped + PnL + volume.

    Sharpe is capped at 3.0 to prevent the GA from over-optimizing for
    ultra-smooth-but-tiny returns. PnL and trade volume are weighted more
    heavily so the GA favors strategies that actually make money.
    """

    def __init__(self, min_trades: int = 50):
        self.min_trades = min_trades

    def compute(
        self,
        sharpe: float,
        sortino: float,
        total_pnl: float,
        n_trades: int,
        win_rate: float,
        max_drawdown_pct: float,
        profit_factor: float,
        avg_pnl: float,
    ) -> float:
        if n_trades < self.min_trades:
            trade_penalty = -2.0 * (1 - n_trades / self.min_trades)
            return trade_penalty

        # Sharpe capped at 3.0 — beyond that, more money matters more than smoother returns
        fitness = min(sharpe, 3.0)

        # PnL bonus — scaled so $100 PnL ≈ +1.5 fitness, $500 ≈ +2.0
        # This is the main lever to prevent penny-scalping genomes from dominating
        if total_pnl > 0:
            fitness += 0.3 * math.log1p(total_pnl)

        # Volume bonus ONLY for profitable strategies
        # log2 scaling: 50→1.7, 200→2.3, 1000→3.0 (with 0.3 coeff)
        if sharpe > 0:
            fitness += 0.3 * math.log2(max(n_trades, 1))

        # Penalize negative avg PnL — losing more per trade is worse
        if avg_pnl < 0:
            fitness += avg_pnl * 0.5  # e.g., -$1/trade → -0.5 fitness

        if max_drawdown_pct > 15:
            fitness -= 0.5 * (max_drawdown_pct - 15) / 10

        # Win rate bonus for genuinely selective strategies (not for always-on)
        if win_rate > 0.45:
            fitness += 0.3 * (win_rate - 0.45)

        # Penalize extreme win rates — likely degenerate (always betting one side)
        if win_rate < 0.25 or win_rate > 0.85:
            fitness -= 1.0

        return fitness


class PnLFocusedFitness(FitnessFunction):
    """Fitness primarily driven by total PnL, with drawdown penalty."""

    def __init__(self, min_trades: int = 50):
        self.min_trades = min_trades

    def compute(
        self,
        sharpe: float,
        sortino: float,
        total_pnl: float,
        n_trades: int,
        win_rate: float,
        max_drawdown_pct: float,
        profit_factor: float,
        avg_pnl: float,
    ) -> float:
        if n_trades < self.min_trades:
            return -2.0 * (1 - n_trades / self.min_trades)

        fitness = avg_pnl * 10  # scale so $1/trade avg = 10 fitness

        if avg_pnl > 0:
            fitness += 0.1 * math.log2(n_trades)

        if max_drawdown_pct > 20:
            fitness -= (max_drawdown_pct - 20) * 0.5

        if profit_factor > 1.5:
            fitness += 1.0

        return fitness


class SortinoFitness(FitnessFunction):
    """Fitness driven by Sortino ratio (penalizes downside risk only)."""

    def __init__(self, min_trades: int = 50):
        self.min_trades = min_trades

    def compute(
        self,
        sharpe: float,
        sortino: float,
        total_pnl: float,
        n_trades: int,
        win_rate: float,
        max_drawdown_pct: float,
        profit_factor: float,
        avg_pnl: float,
    ) -> float:
        if n_trades < self.min_trades:
            return -2.0 * (1 - n_trades / self.min_trades)

        fitness = sortino

        if sortino > 0:
            fitness += 0.1 * math.log2(n_trades)

        if total_pnl > 0:
            fitness += 0.1 * math.log1p(total_pnl)

        if avg_pnl < 0:
            fitness += avg_pnl * 0.5

        if max_drawdown_pct > 15:
            fitness -= 0.5 * (max_drawdown_pct - 15) / 10

        return fitness


class EdgeScaledFitness(FitnessFunction):
    """
    Fitness = edge_per_trade * sqrt(n_trades) * drawdown_tolerance.

    The intuition: a fund manager evaluates a strategy by asking:
    1. What's the expected dollar edge per trade? (avg_pnl)
    2. How many opportunities are there? (n_trades)
    3. Is the drawdown survivable? (soft penalty above threshold)

    The sqrt(n_trades) term is deliberate — it means doubling trade count
    adds ~41% to fitness, not 100%. This rewards activity without making
    the GA chase pure volume. It also mirrors the statistical intuition
    that significance of an edge grows with sqrt(N).

    Losing strategies get negative fitness proportional to how much they
    lose, so the GA has gradient signal to climb out of money-losing regions.
    """

    def __init__(self, min_trades: int = 30, max_dd_pct: float = 25.0):
        self.min_trades = min_trades
        self.max_dd_pct = max_dd_pct

    def compute(
        self,
        sharpe: float,
        sortino: float,
        total_pnl: float,
        n_trades: int,
        win_rate: float,
        max_drawdown_pct: float,
        profit_factor: float,
        avg_pnl: float,
    ) -> float:
        # Hard floor: not enough trades to evaluate
        if n_trades < self.min_trades:
            return -10.0 + (n_trades / self.min_trades) * 5.0  # gradient toward more trades

        # Core: edge * sqrt(opportunities)
        # avg_pnl carries the sign — losing strategies naturally get negative fitness
        fitness = avg_pnl * math.sqrt(n_trades)

        # Drawdown tolerance: no penalty below threshold, soft penalty above
        # A 12% DD on $10k is fine. 25%+ starts to hurt.
        if max_drawdown_pct > self.max_dd_pct:
            excess = max_drawdown_pct - self.max_dd_pct
            fitness *= max(0.1, 1.0 - 0.03 * excess)  # 3% fitness reduction per 1% excess DD

        return fitness


class TStatFitness(FitnessFunction):
    """
    Fitness = t-statistic of per-trade PnL * total_pnl sign.

    The t-stat directly answers "is this edge statistically real?" and
    naturally penalizes low trade counts (t = mean / (std / sqrt(n))).
    Multiplying by sgn(total_pnl) ensures losing strategies rank below
    zero even if their t-stat magnitude is high.

    Bonus: total PnL scale factor so a $500 strategy beats a $50 strategy
    when both have significant t-stats.

    This is the "quant's fitness function" — it's what you'd use if you
    were submitting a paper and needed to prove the edge was real.
    """

    def __init__(self, min_trades: int = 30, max_dd_pct: float = 25.0):
        self.min_trades = min_trades
        self.max_dd_pct = max_dd_pct

    def compute(
        self,
        sharpe: float,
        sortino: float,
        total_pnl: float,
        n_trades: int,
        win_rate: float,
        max_drawdown_pct: float,
        profit_factor: float,
        avg_pnl: float,
    ) -> float:
        if n_trades < self.min_trades:
            return -10.0 + (n_trades / self.min_trades) * 5.0

        # t-stat ≈ sharpe * sqrt(n_trades) / sqrt(annualization_factor)
        # But sharpe is already annualized, so we de-annualize first.
        # Simpler: just compute it from avg_pnl and profit_factor.
        # t = avg_pnl / (std_pnl / sqrt(n))
        # We don't have std_pnl directly, but we can approximate:
        # For win_rate w with avg_win/avg_loss ratio = profit_factor * (1-w)/w:
        #   std ≈ |avg_pnl| * sqrt(w*(1-w)) * (1 + profit_factor) / profit_factor
        # This is rough but directionally correct. The exact t-stat would
        # need the raw returns array, which we should add later.

        # For now, use the Sharpe-based proxy: t ≈ sharpe_per_trade * sqrt(n)
        # sharpe_per_trade = sharpe / sqrt(trades_per_year)
        # But we don't know trades_per_year here. Approximate from 90-day window:
        trades_per_year = n_trades * (365.0 / 90.0)
        sharpe_per_trade = sharpe / math.sqrt(trades_per_year) if trades_per_year > 0 else 0.0
        t_stat = sharpe_per_trade * math.sqrt(n_trades)

        # Ensure sign matches PnL direction
        if total_pnl < 0:
            t_stat = -abs(t_stat)

        # Scale by log of absolute PnL so $500 beats $50 at same significance
        if total_pnl > 0:
            pnl_scale = 1.0 + 0.3 * math.log1p(total_pnl)
        elif total_pnl < 0:
            pnl_scale = 1.0 + 0.3 * math.log1p(abs(total_pnl))
            t_stat *= pnl_scale
            return t_stat  # negative, scaled by loss magnitude
        else:
            return 0.0

        fitness = t_stat * pnl_scale

        # Drawdown: soft penalty above threshold
        if max_drawdown_pct > self.max_dd_pct:
            excess = max_drawdown_pct - self.max_dd_pct
            fitness *= max(0.1, 1.0 - 0.03 * excess)

        return fitness


class ExpectedValueFitness(FitnessFunction):
    """
    Fitness = total_pnl * confidence_multiplier * drawdown_penalty.

    The simplest function that does the right thing. Start with total PnL
    (the thing you actually care about), then adjust:

    1. Confidence multiplier: saturating bonus for trade count.
       min(1.0, sqrt(n_trades / target_trades)). Below target_trades,
       fitness is discounted. Above, no extra bonus — PnL speaks for itself.
    2. Drawdown penalty: only kicks in above threshold.
    3. Losing strategies: fitness = total_pnl (negative), no multipliers.

    Why this works: the GA can only improve fitness by making more money.
    It can't game the function by trading less, because the confidence
    multiplier punishes low N. It can't game it by being reckless, because
    drawdown penalty kicks in. The only winning move is to find actual edge
    and exploit it repeatedly.

    This is the one I'd actually use.
    """

    def __init__(self, min_trades: int = 30, target_trades: int = 100, max_dd_pct: float = 20.0):
        self.min_trades = min_trades
        self.target_trades = target_trades
        self.max_dd_pct = max_dd_pct

    def compute(
        self,
        sharpe: float,
        sortino: float,
        total_pnl: float,
        n_trades: int,
        win_rate: float,
        max_drawdown_pct: float,
        profit_factor: float,
        avg_pnl: float,
    ) -> float:
        if n_trades < self.min_trades:
            return -10.0 + (n_trades / self.min_trades) * 5.0

        # Losing strategies: just return total_pnl (negative)
        # No multipliers — don't accidentally make a -$100 strategy look good
        if total_pnl <= 0:
            return total_pnl

        # Confidence: sqrt ramp to target, capped at 1.0
        confidence = min(1.0, math.sqrt(n_trades / self.target_trades))

        # Drawdown: no penalty below threshold, linear decay above
        dd_factor = 1.0
        if max_drawdown_pct > self.max_dd_pct:
            excess = max_drawdown_pct - self.max_dd_pct
            dd_factor = max(0.2, 1.0 - 0.02 * excess)  # 2% per 1% excess DD

        fitness = total_pnl * confidence * dd_factor

        return fitness


# Default fitness function
DEFAULT_FITNESS = ExpectedValueFitness(min_trades=30, target_trades=100)


def _failed_result(genome_id: str, error: str) -> FitnessResult:
    return FitnessResult(
        genome_id=genome_id,
        fitness=-999.0,
        sharpe=0.0,
        total_pnl=0.0,
        n_trades=0,
        win_rate=0.0,
        avg_pnl=0.0,
        max_drawdown_pct=0.0,
        profit_factor=0.0,
        sortino=0.0,
        avg_bars_held=0.0,
        error=error,
    )


def _extract_results(
    engine: BacktestEngine,
    genome: Genome,
    fitness_fn: FitnessFunction,
    start_ms: int | None,
    end_ms: int | None,
    starting_balance: float,
) -> FitnessResult:
    """Extract fitness metrics from a completed backtest engine run."""
    positions_report = engine.trader.generate_positions_report()

    # Parse realized PnL from positions report
    realized_pnls = []
    if not positions_report.empty and "realized_pnl" in positions_report.columns:
        for pnl_str in positions_report["realized_pnl"]:
            try:
                # PnL format is like "123.45 USDT"
                pnl_val = float(str(pnl_str).split()[0].replace("_", ""))
                realized_pnls.append(pnl_val)
            except (ValueError, IndexError):
                pass

    n_trades = len(realized_pnls)

    if n_trades == 0:
        return _failed_result(genome.genome_id, "No trades generated")

    total_pnl = sum(realized_pnls)
    avg_pnl = total_pnl / n_trades
    wins = sum(1 for p in realized_pnls if p > 0)
    win_rate = wins / n_trades

    # Profit factor
    gross_profit = sum(p for p in realized_pnls if p > 0)
    gross_loss = abs(sum(p for p in realized_pnls if p < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Sharpe and Sortino — compute per-trade returns relative to position size
    if n_trades > 1:
        returns = np.array(realized_pnls) / genome.position_size_usd
        mean_ret = np.mean(returns)
        std_ret = np.std(returns, ddof=1)

        # Estimate trades per day from backtest duration
        if start_ms is not None and end_ms is not None:
            duration_days = max(1, (end_ms - start_ms) / (86_400_000))
        else:
            duration_days = 1.0
        trades_per_day = n_trades / duration_days

        # Annualize: scale by sqrt(trades_per_year) — 365 for 24/7 crypto markets
        trades_per_year = trades_per_day * 365
        annualization = math.sqrt(trades_per_year) if trades_per_year > 0 else 1.0

        sharpe = (mean_ret / std_ret) * annualization if std_ret > 0 else 0.0

        # Sortino
        downside_returns = returns[returns < 0]
        downside_std = np.std(downside_returns, ddof=1) if len(downside_returns) > 1 else 0.0
        sortino = (mean_ret / downside_std) * annualization if downside_std > 0 else 0.0
    else:
        sharpe = 0.0
        sortino = 0.0

    # Max drawdown
    cumulative = np.cumsum(realized_pnls)
    peak = np.maximum.accumulate(cumulative + starting_balance)
    drawdown = (peak - (cumulative + starting_balance)) / peak * 100
    max_drawdown_pct = float(np.max(drawdown)) if len(drawdown) > 0 else 0.0

    # Average holding period (bars)
    avg_bars_held = 0.0  # TODO: extract from strategy state

    # Composite fitness via pluggable function
    fitness = fitness_fn.compute(
        sharpe=sharpe,
        sortino=sortino,
        total_pnl=total_pnl,
        n_trades=n_trades,
        win_rate=win_rate,
        max_drawdown_pct=max_drawdown_pct,
        profit_factor=profit_factor,
        avg_pnl=avg_pnl,
    )

    return FitnessResult(
        genome_id=genome.genome_id,
        fitness=fitness,
        sharpe=sharpe,
        total_pnl=total_pnl,
        n_trades=n_trades,
        win_rate=win_rate,
        avg_pnl=avg_pnl,
        max_drawdown_pct=max_drawdown_pct,
        profit_factor=profit_factor,
        sortino=sortino,
        avg_bars_held=avg_bars_held,
    )


def evaluate_genome(
    genome: Genome,
    data_dir: Path,
    start_ms: int | None = None,
    end_ms: int | None = None,
    starting_balance: float = 10_000.0,
    fitness_fn: FitnessFunction | None = None,
    min_trades: int = 50,
) -> FitnessResult:
    """
    Evaluate a single genome by creating a fresh backtest engine.

    This is the simple path — creates and disposes an engine per call.
    For batch evaluation, use evaluate_genome_on_engine() instead.
    """
    if fitness_fn is None:
        fitness_fn = ExpectedValueFitness(min_trades=min_trades)

    engine = None
    try:
        engine = BacktestEngine(
            config=BacktestEngineConfig(
                logging=LoggingConfig(log_level="ERROR"),
            ),
        )
        instruments = setup_backtest_engine(
            engine=engine,
            data_dir=data_dir,
            symbols=genome.symbols,
            start_ms=start_ms,
            end_ms=end_ms,
            starting_balance=starting_balance,
        )

        if not instruments:
            return _failed_result(genome.genome_id, "No instruments loaded")

        symbol = genome.symbols[0]
        bar_type = make_bar_type(symbol, genome.bar_period_minutes)
        strategy = GenomeStrategy(
            config=GenomeStrategyConfig(
                genome_dict=genome.to_dict(),
                bar_type_str=str(bar_type),
            ),
        )
        engine.add_strategy(strategy)
        engine.run()

        return _extract_results(engine, genome, fitness_fn, start_ms, end_ms, starting_balance)

    except Exception as e:
        return _failed_result(genome.genome_id, str(e))
    finally:
        if engine is not None:
            try:
                engine.dispose()
            except Exception:
                pass


def evaluate_genome_on_engine(
    genome: Genome,
    engine: BacktestEngine,
    fitness_fn: FitnessFunction,
    start_ms: int | None = None,
    end_ms: int | None = None,
    starting_balance: float = 10_000.0,
) -> FitnessResult:
    """
    Evaluate a genome on a pre-loaded engine using engine.reset().

    The engine must already have data and instruments loaded.
    This avoids reloading parquet data for each evaluation.
    """
    try:
        engine.reset()
        engine.clear_strategies()
        symbol = genome.symbols[0]
        bar_type = make_bar_type(symbol, genome.bar_period_minutes)
        strategy = GenomeStrategy(
            config=GenomeStrategyConfig(
                genome_dict=genome.to_dict(),
                bar_type_str=str(bar_type),
            ),
        )
        engine.add_strategy(strategy)
        engine.run()

        return _extract_results(engine, genome, fitness_fn, start_ms, end_ms, starting_balance)

    except Exception as e:
        return _failed_result(genome.genome_id, str(e))


def create_engine(
    data_dir: Path,
    symbols: list[str],
    start_ms: int | None = None,
    end_ms: int | None = None,
    starting_balance: float = 10_000.0,
) -> BacktestEngine:
    """Create a BacktestEngine with data pre-loaded, ready for repeated use."""
    engine = BacktestEngine(
        config=BacktestEngineConfig(
            logging=LoggingConfig(log_level="ERROR"),
        ),
    )
    instruments = setup_backtest_engine(
        engine=engine,
        data_dir=data_dir,
        symbols=symbols,
        start_ms=start_ms,
        end_ms=end_ms,
        starting_balance=starting_balance,
    )
    if not instruments:
        raise RuntimeError(f"No instruments loaded for {symbols}")
    return engine
