"""
GA Runner: orchestrates the genetic algorithm evolution loop.

Usage:
    python -m evolution.ga.runner --data-dir data/ --generations 20 --pop-size 30
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from .fitness import (
    FitnessFunction,
    FitnessResult,
    ExpectedValueFitness,
    _extract_results,
    _failed_result,
    create_engine,
    evaluate_genome,
    evaluate_genome_on_engine,
)
from .genome import Genome
from .operators import evolve_population, random_genome


@dataclass
class GenerationResult:
    """Results from one generation of evolution."""
    generation: int
    results: list[FitnessResult]
    best: FitnessResult
    avg_fitness: float
    elapsed_s: float

    @property
    def summary(self) -> str:
        lines = [
            f"\n{'='*60}",
            f"Generation {self.generation} — {self.elapsed_s:.1f}s",
            f"{'='*60}",
            f"  Best:  {self.best.summary}",
            f"  Avg fitness: {self.avg_fitness:.3f}",
            f"  Trades range: {min(r.n_trades for r in self.results)}-{max(r.n_trades for r in self.results)}",
            f"  Errors: {sum(1 for r in self.results if r.error)}",
        ]
        return "\n".join(lines)


@dataclass
class EvolutionState:
    """Persisted state of an evolution run."""
    run_id: str
    current_generation: int = 0
    best_fitness_ever: float = -999.0
    best_genome_ever: dict | None = None
    generation_history: list[dict] = field(default_factory=list)
    population: list[dict] = field(default_factory=list)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({
            "run_id": self.run_id,
            "current_generation": self.current_generation,
            "best_fitness_ever": self.best_fitness_ever,
            "best_genome_ever": self.best_genome_ever,
            "generation_history": self.generation_history,
            "population": self.population,
        }, indent=2))

    @classmethod
    def load(cls, path: Path) -> EvolutionState:
        d = json.loads(path.read_text())
        return cls(**d)


# Per-worker cached data — loaded once per process, used to build fresh engines
_worker_data = None  # dict[str, tuple[instrument, quotes]]
_worker_data_key = None


def _init_worker_data(data_dir: str, symbols: list[str],
                      start_ms: int | None, end_ms: int | None) -> None:
    """Load data once per worker process (cached across genome evaluations)."""
    global _worker_data, _worker_data_key
    from evolution.data.binance_loader import load_multi_asset
    key = (data_dir, tuple(symbols), start_ms, end_ms)
    if _worker_data is not None and _worker_data_key == key:
        return
    _worker_data = load_multi_asset(Path(data_dir), symbols, start_ms, end_ms)
    _worker_data_key = key


def _eval_worker(args: dict) -> dict:
    """Worker function for parallel genome evaluation.

    Each worker process loads data once and creates a fresh engine per genome.
    """
    from nautilus_trader.backtest.engine import BacktestEngine
    from nautilus_trader.config import BacktestEngineConfig, LoggingConfig
    from nautilus_trader.model.currencies import Currency
    from nautilus_trader.model.enums import AccountType, OmsType
    from nautilus_trader.model.objects import Money
    from evolution.data.binance_loader import BINANCE, make_bar_type
    from evolution.ga.genome import Genome
    from evolution.strategies.genome_strategy import GenomeStrategy, GenomeStrategyConfig

    # Ensure data is loaded (no-op after first call in this process)
    _init_worker_data(
        args["data_dir"], args["symbols"],
        args.get("start_ms"), args.get("end_ms"),
    )

    genome = Genome.from_dict(args["genome_dict"])
    genome.generation = args["generation"]
    starting_balance = args["starting_balance"]

    fitness_fn = ExpectedValueFitness(min_trades=args["min_trades"])

    engine = None
    try:
        engine = BacktestEngine(
            config=BacktestEngineConfig(
                logging=LoggingConfig(log_level="ERROR"),
            ),
        )
        usdt = Currency.from_str("USDT")
        engine.add_venue(
            venue=BINANCE,
            oms_type=OmsType.NETTING,
            account_type=AccountType.MARGIN,
            starting_balances=[Money(starting_balance, usdt)],
        )
        for symbol, (instrument, quotes) in _worker_data.items():
            engine.add_instrument(instrument)
            engine.add_data(quotes, sort=False)
        engine.sort_data()

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

        result = _extract_results(engine, genome, fitness_fn,
                                  args.get("start_ms"), args.get("end_ms"),
                                  starting_balance)
    except Exception as e:
        result = _failed_result(genome.genome_id, str(e))
    finally:
        if engine is not None:
            try:
                engine.dispose()
            except Exception:
                pass

    return result.to_dict()


def run_evolution(
    data_dir: Path,
    output_dir: Path,
    symbols: list[str] | None = None,
    pop_size: int = 30,
    n_generations: int = 20,
    start_ms: int | None = None,
    end_ms: int | None = None,
    starting_balance: float = 10_000.0,
    min_trades: int = 50,
    fitness_fn: FitnessFunction | None = None,
    elitism_pct: float = 0.05,
    crossover_pct: float = 0.5,
    mutation_rate: float = 0.2,
    immigration_pct: float = 0.2,
    resume_state: EvolutionState | None = None,
    n_workers: int | None = None,
) -> EvolutionState:
    """
    Run the full genetic algorithm evolution loop.

    Parameters
    ----------
    data_dir : Path
        Directory containing Binance quote parquet files.
    output_dir : Path
        Directory to save results, genomes, and state.
    symbols : list[str], optional
        Trading pairs to include. Default: ["BTCUSDT"].
    pop_size : int
        Population size per generation.
    n_generations : int
        Number of generations to evolve.
    start_ms, end_ms : int, optional
        Time range for backtesting.
    starting_balance : float
        Initial account balance.
    min_trades : int
        Minimum trades for valid fitness.
    elitism_pct, crossover_pct, mutation_rate : float
        GA operator parameters.
    resume_state : EvolutionState, optional
        Resume from a previous run.
    n_workers : int, optional
        Number of parallel worker processes. Default: min(8, cpu_count).

    Returns
    -------
    EvolutionState
        Final state of the evolution run.
    """
    if symbols is None:
        symbols = ["BTCUSDT"]

    if n_workers is None:
        n_workers = os.cpu_count() or 1

    output_dir.mkdir(parents=True, exist_ok=True)
    genomes_dir = output_dir / "genomes"
    genomes_dir.mkdir(exist_ok=True)

    # Cap indicator periods based on data window to ensure indicators can initialize
    max_period = None
    if start_ms is not None and end_ms is not None:
        bar_minutes = 1  # conservative estimate
        total_bars = (end_ms - start_ms) / (bar_minutes * 60_000)
        max_period = max(5, int(total_bars * 0.3))  # use at most 30% of bars for warmup

    # Initialize or resume state
    if resume_state:
        state = resume_state
        population = [Genome.from_dict(d) for d in state.population]
        start_gen = state.current_generation + 1
        print(f"Resuming from generation {start_gen}")
    else:
        import uuid
        state = EvolutionState(run_id=uuid.uuid4().hex[:8])
        population = [random_genome(symbols=symbols, max_period=max_period) for _ in range(pop_size)]
        start_gen = 0
        print(f"Starting new evolution run: {state.run_id}")

    parallel = n_workers > 1
    print(f"Population: {pop_size}, Generations: {n_generations}, Workers: {n_workers}")
    print(f"Symbols: {symbols}")
    print(f"Data dir: {data_dir}")

    # For sequential mode, create engine once and reuse
    seq_engine = None
    executor = None
    if not parallel:
        if fitness_fn is None:
            fitness_fn = ExpectedValueFitness(min_trades=min_trades)
        seq_engine = create_engine(data_dir, symbols, start_ms, end_ms, starting_balance)
        print("Engine loaded (sequential mode, data cached)")
    else:
        # Keep pool alive across generations so workers retain cached data
        executor = ProcessPoolExecutor(max_workers=n_workers)
        print(f"Worker pool started ({n_workers} processes, data cached per-worker)")

    try:
        for gen in range(start_gen, n_generations):
            gen_start = time.time()
            print(f"\n--- Generation {gen} ({len(population)} genomes) ---")

            # Set generation on all genomes
            for genome in population:
                genome.generation = gen

            if parallel:
                # Work queue: submit all, process as they complete
                # This avoids idle cores when some genomes finish early (killed or few trades)
                futures = {}
                for i, genome in enumerate(population):
                    args = {
                        "genome_dict": genome.to_dict(),
                        "generation": gen,
                        "data_dir": str(data_dir),
                        "symbols": symbols,
                        "start_ms": start_ms,
                        "end_ms": end_ms,
                        "starting_balance": starting_balance,
                        "min_trades": min_trades,
                    }
                    future = executor.submit(_eval_worker, args)
                    futures[future] = i

                # Collect results in completion order (fast genomes report first)
                results_by_idx: dict[int, FitnessResult] = {}
                completed = 0
                for future in as_completed(futures):
                    idx = futures[future]
                    rd = future.result()
                    result = FitnessResult(**rd)
                    results_by_idx[idx] = result
                    completed += 1
                    status = f"trades={result.n_trades}, fit={result.fitness:.2f}"
                    if result.error:
                        status = f"ERROR: {result.error[:50]}"
                    print(f"  [{completed}/{len(population)}] {population[idx].genome_id}: {status}")

                # Reassemble in original order for consistent genome-result pairing
                results = [results_by_idx[i] for i in range(len(population))]
            else:
                # Sequential evaluation with engine reuse
                results: list[FitnessResult] = []
                for i, genome in enumerate(population):
                    result = evaluate_genome_on_engine(
                        genome=genome,
                        engine=seq_engine,
                        fitness_fn=fitness_fn,
                        start_ms=start_ms,
                        end_ms=end_ms,
                        starting_balance=starting_balance,
                    )
                    results.append(result)

                    status = f"trades={result.n_trades}, fit={result.fitness:.2f}"
                    if result.error:
                        status = f"ERROR: {result.error[:50]}"
                    print(f"  [{i+1}/{len(population)}] {genome.genome_id}: {status}")

            # Sort by fitness
            scored_population = list(zip(population, [r.fitness for r in results]))
            scored_population.sort(key=lambda x: x[1], reverse=True)
            results.sort(key=lambda r: r.fitness, reverse=True)

            elapsed = time.time() - gen_start
            avg_fitness = sum(r.fitness for r in results) / len(results)

            gen_result = GenerationResult(
                generation=gen,
                results=results,
                best=results[0],
                avg_fitness=avg_fitness,
                elapsed_s=elapsed,
            )
            print(gen_result.summary)

            # Save best genome of this generation
            best_genome = scored_population[0][0]
            best_genome.save(genomes_dir / f"gen{gen:03d}_best_{best_genome.genome_id}.json")

            # Update state
            state.current_generation = gen
            state.generation_history.append({
                "generation": gen,
                "best_fitness": results[0].fitness,
                "avg_fitness": avg_fitness,
                "best_genome_id": results[0].genome_id,
                "n_trades_best": results[0].n_trades,
                "total_pnl_best": results[0].total_pnl,
                "elapsed_s": elapsed,
            })

            if results[0].fitness > state.best_fitness_ever:
                state.best_fitness_ever = results[0].fitness
                state.best_genome_ever = best_genome.to_dict()
                best_genome.save(genomes_dir / f"best_ever_{best_genome.genome_id}.json")
                print(f"  *** New best ever: fitness={results[0].fitness:.3f} ***")

            # Evolve to next generation
            if gen < n_generations - 1:
                population = evolve_population(
                    population=scored_population,
                    pop_size=pop_size,
                    elitism_pct=elitism_pct,
                    crossover_pct=crossover_pct,
                    mutation_rate=mutation_rate,
                    immigration_pct=immigration_pct,
                    max_period=max_period,
                )

            # Save state for resume
            state.population = [g.to_dict() for g in population]
            state.save(output_dir / "evolution_state.json")

    finally:
        if seq_engine is not None:
            try:
                seq_engine.dispose()
            except Exception:
                pass
        if executor is not None:
            executor.shutdown(wait=False)

    # Final summary
    print(f"\n{'='*60}")
    print(f"Evolution complete: {n_generations} generations")
    print(f"Best ever: fitness={state.best_fitness_ever:.3f}")
    if state.best_genome_ever:
        best = Genome.from_dict(state.best_genome_ever)
        print(best.description)
    print(f"Results saved to: {output_dir}")

    return state


if __name__ == "__main__":
    import argparse
    from datetime import datetime, timezone

    def date_to_ms(s: str) -> int:
        return int(datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)

    parser = argparse.ArgumentParser(description="Run GA evolution")
    parser.add_argument("--data-dir", type=Path, default=Path("data/"))
    parser.add_argument("--output-dir", type=Path, default=Path("evolution/runs/"))
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT"])
    parser.add_argument("--pop-size", type=int, default=30)
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--starting-balance", type=float, default=10_000.0)
    parser.add_argument("--min-trades", type=int, default=50)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--mutation-rate", type=float, default=0.2)
    parser.add_argument("--elitism-pct", type=float, default=0.05)
    parser.add_argument("--immigration-pct", type=float, default=0.2)
    parser.add_argument("--crossover-pct", type=float, default=0.5)
    parser.add_argument("--start-date", type=str, default=None,
                        help="Training start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default=None,
                        help="Training end date (YYYY-MM-DD)")
    parser.add_argument("--resume", type=Path, default=None,
                        help="Path to evolution_state.json to resume")
    args = parser.parse_args()

    start_ms = date_to_ms(args.start_date) if args.start_date else None
    end_ms = date_to_ms(args.end_date) if args.end_date else None

    resume = None
    if args.resume:
        resume = EvolutionState.load(args.resume)

    run_evolution(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        symbols=args.symbols,
        pop_size=args.pop_size,
        n_generations=args.generations,
        start_ms=start_ms,
        end_ms=end_ms,
        starting_balance=args.starting_balance,
        min_trades=args.min_trades,
        mutation_rate=args.mutation_rate,
        elitism_pct=args.elitism_pct,
        crossover_pct=args.crossover_pct,
        immigration_pct=args.immigration_pct,
        n_workers=args.workers,
        resume_state=resume,
    )
