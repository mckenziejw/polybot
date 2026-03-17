"""
Visualize evolution results — genome tearsheets + evolution progress charts.

Usage:
    # Tearsheet for the best genome from a run
    python -m evolution.visualize tearsheet evolution/runs/prod_v1/

    # Evolution progress charts (fitness, trades, timing over generations)
    python -m evolution.visualize progress evolution/runs/prod_v1/

    # Both
    python -m evolution.visualize all evolution/runs/prod_v1/

    # Specific genome file
    python -m evolution.visualize tearsheet --genome evolution/runs/prod_v1/genomes/best_ever_abc123.json

    # OOS validation tearsheet (train on one window, test on another)
    python -m evolution.visualize tearsheet evolution/runs/prod_v1/ --oos-start 2025-12-01 --oos-end 2026-03-01
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from nautilus_trader.analysis import (
    TearsheetBarsWithFillsChart,
    TearsheetConfig,
    TearsheetDrawdownChart,
    TearsheetEquityChart,
    TearsheetStatsTableChart,
    create_tearsheet,
)
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.config import BacktestEngineConfig, LoggingConfig

from evolution.data.binance_loader import make_bar_type, setup_backtest_engine
from evolution.ga.fitness import SharpeWeightedFitness, _extract_results
from evolution.ga.genome import Genome
from evolution.strategies.genome_strategy import GenomeStrategy, GenomeStrategyConfig


def date_to_ms(s: str) -> int:
    return int(datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)


def load_best_genome(run_dir: Path) -> Genome:
    """Load the best-ever genome from an evolution run."""
    state_path = run_dir / "evolution_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state.get("best_genome_ever"):
            return Genome.from_dict(state["best_genome_ever"])

    # Fallback: find best_ever_*.json in genomes/
    genomes_dir = run_dir / "genomes"
    best_files = sorted(genomes_dir.glob("best_ever_*.json"))
    if best_files:
        return Genome.load(best_files[-1])

    raise FileNotFoundError(f"No best genome found in {run_dir}")


def run_backtest_for_tearsheet(
    genome: Genome,
    data_dir: Path,
    start_ms: int | None = None,
    end_ms: int | None = None,
    starting_balance: float = 10_000.0,
) -> BacktestEngine:
    """Run a genome through backtest and return the engine (for tearsheet generation)."""
    engine = BacktestEngine(
        config=BacktestEngineConfig(
            logging=LoggingConfig(log_level="WARNING"),
        ),
    )
    setup_backtest_engine(
        engine=engine,
        data_dir=data_dir,
        symbols=genome.symbols,
        start_ms=start_ms,
        end_ms=end_ms,
        starting_balance=starting_balance,
    )

    bar_type = make_bar_type(genome.symbols[0], genome.bar_period_minutes)
    strategy = GenomeStrategy(
        config=GenomeStrategyConfig(
            genome_dict=genome.to_dict(),
            bar_type_str=str(bar_type),
        ),
    )
    engine.add_strategy(strategy)
    engine.run()
    return engine


def generate_tearsheet(
    genome: Genome,
    data_dir: Path,
    output_path: Path,
    start_ms: int | None = None,
    end_ms: int | None = None,
    starting_balance: float = 10_000.0,
    title: str | None = None,
) -> None:
    """Generate a NautilusTrader tearsheet for a genome."""
    if title is None:
        title = f"Genome {genome.genome_id} (gen {genome.generation})"

    print(f"Running backtest for {genome.genome_id}...")
    engine = run_backtest_for_tearsheet(
        genome=genome,
        data_dir=data_dir,
        start_ms=start_ms,
        end_ms=end_ms,
        starting_balance=starting_balance,
    )

    # Extract quick stats
    fitness_fn = SharpeWeightedFitness(min_trades=20)
    result = _extract_results(engine, genome, fitness_fn, start_ms, end_ms, starting_balance)
    print(f"  Result: {result.summary}")

    bar_type = make_bar_type(genome.symbols[0], genome.bar_period_minutes)
    config = TearsheetConfig(
        charts=[
            TearsheetEquityChart(),
            TearsheetDrawdownChart(),
            TearsheetStatsTableChart(),
            TearsheetBarsWithFillsChart(bar_type=str(bar_type)),
        ],
        theme="nautilus_dark",
        height=2400,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    create_tearsheet(
        engine=engine,
        output_path=str(output_path),
        title=title,
        config=config,
    )
    print(f"  Tearsheet saved: {output_path}")

    engine.dispose()
    return result


def generate_progress_charts(run_dir: Path, output_path: Path) -> None:
    """Generate evolution progress charts from evolution_state.json."""
    state_path = run_dir / "evolution_state.json"
    if not state_path.exists():
        print(f"No evolution_state.json found in {run_dir}")
        return

    state = json.loads(state_path.read_text())
    history = state.get("generation_history", [])
    if not history:
        print("No generation history found")
        return

    gens = [h["generation"] for h in history]
    best_fitness = [h["best_fitness"] for h in history]
    avg_fitness = [h["avg_fitness"] for h in history]
    n_trades = [h.get("n_trades_best", 0) for h in history]
    total_pnl = [h.get("total_pnl_best", 0) for h in history]
    elapsed = [h.get("elapsed_s", 0) for h in history]

    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=[
            "Best & Avg Fitness", "Best Genome Trades",
            "Best Genome PnL ($)", "Generation Time (s)",
            "Fitness Distribution", "Cumulative Compute Time",
        ],
        vertical_spacing=0.08,
        horizontal_spacing=0.1,
    )

    # 1. Fitness over generations
    fig.add_trace(
        go.Scatter(x=gens, y=best_fitness, mode="lines+markers",
                   name="Best Fitness", line=dict(color="#00d4aa", width=2)),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=gens, y=avg_fitness, mode="lines",
                   name="Avg Fitness", line=dict(color="#ff6b6b", width=1, dash="dash")),
        row=1, col=1,
    )

    # 2. Trades per generation (best genome)
    fig.add_trace(
        go.Bar(x=gens, y=n_trades, name="Trades (best)",
               marker_color="#4ecdc4"),
        row=1, col=2,
    )

    # 3. PnL over generations
    fig.add_trace(
        go.Scatter(x=gens, y=total_pnl, mode="lines+markers",
                   name="PnL (best)", line=dict(color="#45b7d1", width=2)),
        row=2, col=1,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)

    # 4. Generation timing
    fig.add_trace(
        go.Bar(x=gens, y=elapsed, name="Gen Time (s)",
               marker_color="#96ceb4"),
        row=2, col=2,
    )

    # 5. Fitness distribution (best vs avg gap)
    gap = [b - a for b, a in zip(best_fitness, avg_fitness)]
    fig.add_trace(
        go.Scatter(x=gens, y=gap, mode="lines+markers",
                   name="Best - Avg Gap", line=dict(color="#ffa07a", width=2)),
        row=3, col=1,
    )

    # 6. Cumulative compute time
    cumulative_time = np.cumsum(elapsed)
    fig.add_trace(
        go.Scatter(x=gens, y=cumulative_time / 60, mode="lines",
                   name="Cumulative (min)", line=dict(color="#dda0dd", width=2)),
        row=3, col=2,
    )

    fig.update_layout(
        title=f"Evolution Progress — Run {state.get('run_id', 'unknown')}",
        template="plotly_dark",
        height=900,
        showlegend=False,
        font=dict(size=11),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path))
    print(f"Progress charts saved: {output_path}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"Evolution Summary — Run {state.get('run_id', 'unknown')}")
    print(f"{'='*60}")
    print(f"  Generations: {len(history)}")
    print(f"  Best fitness: {state.get('best_fitness_ever', 0):.3f}")
    print(f"  Total compute: {sum(elapsed)/60:.1f} min")
    print(f"  Avg gen time: {np.mean(elapsed):.1f}s")
    if state.get("best_genome_ever"):
        genome = Genome.from_dict(state["best_genome_ever"])
        print(f"\n{genome.description}")


def main():
    parser = argparse.ArgumentParser(description="Visualize evolution results")
    parser.add_argument("mode", choices=["tearsheet", "progress", "all"],
                        help="What to generate")
    parser.add_argument("run_dir", type=Path,
                        help="Evolution run directory (e.g., evolution/runs/prod_v1/)")
    parser.add_argument("--genome", type=Path, default=None,
                        help="Specific genome JSON file (overrides run_dir best)")
    parser.add_argument("--data-dir", type=Path, default=Path("data/"),
                        help="Data directory for backtesting")
    parser.add_argument("--start-date", type=str, default=None,
                        help="Backtest start date (YYYY-MM-DD, default: from evolution state)")
    parser.add_argument("--end-date", type=str, default=None,
                        help="Backtest end date (YYYY-MM-DD, default: from evolution state)")
    parser.add_argument("--oos-start", type=str, default=None,
                        help="Out-of-sample start date (YYYY-MM-DD)")
    parser.add_argument("--oos-end", type=str, default=None,
                        help="Out-of-sample end date (YYYY-MM-DD)")
    parser.add_argument("--balance", type=float, default=10_000.0,
                        help="Starting balance")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory (default: run_dir/charts/)")
    args = parser.parse_args()

    output_dir = args.output_dir or args.run_dir / "charts"

    # Load dates from evolution state if not specified
    start_ms = date_to_ms(args.start_date) if args.start_date else None
    end_ms = date_to_ms(args.end_date) if args.end_date else None

    if start_ms is None or end_ms is None:
        state_path = args.run_dir / "evolution_state.json"
        if state_path.exists():
            state = json.loads(state_path.read_text())
            # Try to infer from run — these aren't stored in state currently
            # User must provide dates for tearsheet if not in state

    if args.mode in ("progress", "all"):
        generate_progress_charts(args.run_dir, output_dir / "evolution_progress.html")

    if args.mode in ("tearsheet", "all"):
        # Load genome
        if args.genome:
            genome = Genome.load(args.genome)
        else:
            genome = load_best_genome(args.run_dir)

        # In-sample tearsheet
        generate_tearsheet(
            genome=genome,
            data_dir=args.data_dir,
            output_path=output_dir / f"tearsheet_is_{genome.genome_id}.html",
            start_ms=start_ms,
            end_ms=end_ms,
            starting_balance=args.balance,
            title=f"IN-SAMPLE: {genome.genome_id} (gen {genome.generation})",
        )

        # OOS tearsheet if dates provided
        if args.oos_start and args.oos_end:
            oos_start_ms = date_to_ms(args.oos_start)
            oos_end_ms = date_to_ms(args.oos_end)
            generate_tearsheet(
                genome=genome,
                data_dir=args.data_dir,
                output_path=output_dir / f"tearsheet_oos_{genome.genome_id}.html",
                start_ms=oos_start_ms,
                end_ms=oos_end_ms,
                starting_balance=args.balance,
                title=f"OUT-OF-SAMPLE: {genome.genome_id} (gen {genome.generation})",
            )


if __name__ == "__main__":
    main()
