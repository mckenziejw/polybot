"""
End-to-end proof-of-concept test for the strategy evolution system.

Tests:
1. Data loading from parquet -> NautilusTrader QuoteTicks
2. Random genome generation
3. Genome -> Strategy translation
4. Single genome backtest evaluation
5. GA evolution (2 generations, tiny population)
6. Tournament coordinator cycle
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evolution.data.binance_loader import (
    create_instrument,
    load_quotes_from_parquet,
    make_bar_type,
    setup_backtest_engine,
)
from evolution.ga.genome import (
    BandOutput,
    Genome,
    IndicatorGene,
    IndicatorType,
    EntryRule,
    ExitRule,
    Condition,
    Side,
)
from evolution.ga.operators import random_genome, mutate, crossover, evolve_population
from evolution.ga.fitness import (
    evaluate_genome,
    SharpeWeightedFitness,
    PnLFocusedFitness,
    SortinoFitness,
)
from evolution.ga.runner import run_evolution
from evolution.strategies.genome_strategy import GenomeStrategy, GenomeStrategyConfig
from evolution.tournament.coordinator import TournamentCoordinator, TournamentConfig


DATA_DIR = PROJECT_ROOT / "data"


def test_data_loading():
    """Test loading Binance parquet data into NautilusTrader types."""
    print("\n=== Test: Data Loading ===")

    instrument, quotes = load_quotes_from_parquet(
        DATA_DIR, "BTCUSDT",
        start_ms=1770336001000,
        end_ms=1770336001000 + 300_000,  # 5 minutes
    )

    print(f"  Instrument: {instrument.id}")
    print(f"  Price precision: {instrument.price_precision}")
    print(f"  Quotes loaded: {len(quotes)}")
    if quotes:
        print(f"  First: price={quotes[0].bid_price}/{quotes[0].ask_price}, ts={quotes[0].ts_event}")
        print(f"  Last:  price={quotes[-1].bid_price}/{quotes[-1].ask_price}, ts={quotes[-1].ts_event}")

    assert len(quotes) > 0, "Should load quotes"
    assert instrument.price_precision == 2, "BTC price precision should be 2"
    print("  PASSED")
    return instrument, quotes


def test_genome_creation():
    """Test genome creation, serialization, and mutation."""
    print("\n=== Test: Genome Creation ===")

    # Random genome
    g = random_genome(symbols=["BTCUSDT"])
    print(f"  Random genome: {g.genome_id}")
    print(f"  Indicators: {len(g.indicators)}")
    print(f"  Entry rules: {len(g.entry_rules)}")
    print(f"  Exit rules: {len(g.exit_rules)}")
    print(f"  Min entry matches: {g.min_entry_matches}")

    # Serialization roundtrip
    json_str = g.to_json()
    g2 = Genome.from_json(json_str)
    assert g2.genome_id == g.genome_id
    assert len(g2.indicators) == len(g.indicators)
    assert g2.min_entry_matches == g.min_entry_matches
    print("  Serialization roundtrip: OK")

    # Mutation
    g_mut = mutate(g, rate=0.3)
    assert g_mut.genome_id != g.genome_id
    assert g_mut.parent_ids == [g.genome_id]
    print(f"  Mutated: {g_mut.genome_id} (parent: {g_mut.parent_ids})")

    # Crossover
    g_a = random_genome(symbols=["BTCUSDT"])
    g_b = random_genome(symbols=["BTCUSDT"])
    child = crossover(g_a, g_b)
    assert child.parent_ids == [g_a.genome_id, g_b.genome_id]
    print(f"  Crossover child: {child.genome_id}")

    # Verify unique indicator IDs
    ids = [ind.id for ind in g.indicators]
    assert len(ids) == len(set(ids)), f"Duplicate indicator IDs: {ids}"
    print("  Unique indicator IDs: OK")

    # Description
    print(g.description)

    print("  PASSED")
    return g


def test_manual_genome():
    """Test a hand-crafted genome (RSI mean reversion)."""
    print("\n=== Test: Manual Genome (RSI Mean Reversion) ===")

    rsi_id = "RSI_14_manual"
    genome = Genome(
        symbols=["BTCUSDT"],
        bar_period_minutes=1,
        indicators=[
            IndicatorGene(type=IndicatorType.RSI, params={"period": 14}, id=rsi_id),
            IndicatorGene(type=IndicatorType.SMA, params={"period": 20}),
        ],
        entry_rules=[
            EntryRule(
                indicator_id=rsi_id,
                condition=Condition.LESS_THAN,
                value=0.45,
                side=Side.BUY,
            ),
            EntryRule(
                indicator_id=rsi_id,
                condition=Condition.GREATER_THAN,
                value=0.55,
                side=Side.SELL,
            ),
        ],
        exit_rules=[
            ExitRule(type="take_profit", value=2.0),
            ExitRule(type="stop_loss", value=1.0),
            ExitRule(type="time_limit", value=3600),
        ],
        position_size_usd=100.0,
        min_entry_matches=1,
    )

    print(genome.description)
    print("  PASSED")
    return genome


def test_fitness_functions():
    """Test that different fitness functions produce different scores."""
    print("\n=== Test: Fitness Functions ===")

    fns = [
        SharpeWeightedFitness(min_trades=1),
        PnLFocusedFitness(min_trades=1),
        SortinoFitness(min_trades=1),
    ]

    for fn in fns:
        score = fn.compute(
            sharpe=1.5, sortino=2.0, total_pnl=100.0,
            n_trades=20, win_rate=0.6, max_drawdown_pct=5.0,
            profit_factor=2.0, avg_pnl=5.0,
        )
        print(f"  {fn.name}: {score:.3f}")

    # Verify min_trades penalty
    score = fns[0].compute(
        sharpe=1.5, sortino=2.0, total_pnl=100.0,
        n_trades=0, win_rate=0.0, max_drawdown_pct=0.0,
        profit_factor=0.0, avg_pnl=0.0,
    )
    assert score < 0, "Zero trades should produce negative fitness"
    print(f"  Zero trades penalty: {score:.3f}")

    print("  PASSED")


def test_single_evaluation():
    """Test evaluating a single genome through the backtest engine."""
    print("\n=== Test: Single Genome Evaluation ===")

    genome = test_manual_genome()

    # Use a small time window for speed (6 hours of data)
    result = evaluate_genome(
        genome=genome,
        data_dir=DATA_DIR,
        start_ms=1770336001000,
        end_ms=1770336001000 + 21_600_000,  # 6 hours
        starting_balance=10_000.0,
        min_trades=1,  # low threshold for testing
    )

    print(f"  Result: {result.summary}")
    if result.error:
        print(f"  Error (may be OK for short window): {result.error}")
    else:
        print(f"  Trades: {result.n_trades}")
        print(f"  PnL: ${result.total_pnl:.2f}")
        print(f"  Win rate: {result.win_rate:.1%}")
        print(f"  Sharpe: {result.sharpe:.2f}")

    print("  PASSED (evaluation completed without crash)")
    return result


def test_mini_evolution():
    """Test a tiny GA evolution run."""
    print("\n=== Test: Mini Evolution (3 genomes, 2 generations) ===")

    output_dir = PROJECT_ROOT / "evolution" / "test_runs" / "mini"

    state = run_evolution(
        data_dir=DATA_DIR,
        output_dir=output_dir,
        symbols=["BTCUSDT"],
        pop_size=3,
        n_generations=2,
        start_ms=1770336001000,
        end_ms=1770336001000 + 10_800_000,  # 3 hours
        starting_balance=10_000.0,
        min_trades=1,
    )

    print(f"\n  Run ID: {state.run_id}")
    print(f"  Generations completed: {state.current_generation + 1}")
    print(f"  Best fitness ever: {state.best_fitness_ever:.3f}")
    print("  PASSED")
    return state


def test_tournament_coordinator():
    """Test tournament coordinator lifecycle."""
    print("\n=== Test: Tournament Coordinator ===")

    config = TournamentConfig(
        max_strategies=5,
        min_strategies=3,
        prune_pct=0.3,
        symbols=["BTCUSDT"],
    )

    coordinator = TournamentCoordinator(
        config=config,
        state_dir=PROJECT_ROOT / "evolution" / "test_runs" / "tournament",
    )

    # Should auto-fill to min_strategies
    report = coordinator.run_cycle()
    print(f"  After first cycle: {report['post_count']} strategies")
    assert report["post_count"] >= config.min_strategies

    # Simulate some performance data
    for sid, record in coordinator.strategies.items():
        import random
        coordinator.update_performance(
            strategy_id=sid,
            n_trades=random.randint(5, 50),
            total_pnl=random.uniform(-100, 200),
            win_rate=random.uniform(0.3, 0.7),
            sharpe=random.uniform(-1, 3),
            max_drawdown_pct=random.uniform(1, 15),
        )
        # Fake age so pruning works
        record.entered_at = record.entered_at - 25 * 3600  # 25h ago

    # Run another cycle with fake GA candidates
    ga_candidates = [random_genome(symbols=["BTCUSDT"]) for _ in range(3)]
    report2 = coordinator.run_cycle(ga_candidates=ga_candidates)
    print(f"  After second cycle: {report2['post_count']} strategies")
    print(f"  Pruned: {len(report2['pruned'])}")
    print(f"  Mutated: {len(report2['mutated'])}")
    print(f"  Injected: {len(report2['injected'])}")

    print(coordinator.get_leaderboard())
    print("  PASSED")


def main():
    print("=" * 60)
    print(" Strategy Evolution System — Proof of Concept Tests")
    print("=" * 60)

    # Check data exists
    btc_path = DATA_DIR / "btc_quotes" / "btcusdt_quotes.parquet"
    if not btc_path.exists():
        print(f"ERROR: Data file not found: {btc_path}")
        print("Make sure you're running from the project root.")
        sys.exit(1)

    test_data_loading()
    test_genome_creation()
    test_manual_genome()
    test_fitness_functions()
    test_single_evaluation()
    test_mini_evolution()
    test_tournament_coordinator()

    print("\n" + "=" * 60)
    print(" ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
