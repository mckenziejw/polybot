"""
Paper Trading Tournament Coordinator.

Manages a population of paper-trading strategies on live data feeds.
Periodically evaluates performance, prunes losers, mutates winners,
and injects new candidates from the GA backtest lab.

This module handles the tournament lifecycle but delegates the actual
NautilusTrader node management to the runner module.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evolution.ga.genome import Genome
from evolution.ga.operators import mutate, random_genome


@dataclass
class StrategyRecord:
    """Track a strategy's lifecycle in the tournament."""
    genome: Genome
    strategy_id: str  # NautilusTrader strategy ID
    entered_at: float  # timestamp when added to tournament
    generation: int = 0

    # Performance tracking (updated by coordinator)
    n_trades: int = 0
    total_pnl: float = 0.0
    win_rate: float = 0.0
    sharpe: float = 0.0
    max_drawdown_pct: float = 0.0
    last_evaluated: float = 0.0

    # Genealogy
    parent_ids: list[str] = field(default_factory=list)
    origin: str = "random"  # "random", "mutation", "injection", "crossover"

    @property
    def fitness(self) -> float:
        """Composite fitness for tournament ranking."""
        if self.n_trades < 5:
            return -10.0  # Not enough data
        score = self.sharpe
        if self.total_pnl > 0:
            score += 0.1
        if self.max_drawdown_pct > 15:
            score -= 0.5
        return score

    @property
    def age_hours(self) -> float:
        return (time.time() - self.entered_at) / 3600

    def to_dict(self) -> dict:
        return {
            "genome": self.genome.to_dict(),
            "strategy_id": self.strategy_id,
            "entered_at": self.entered_at,
            "generation": self.generation,
            "n_trades": self.n_trades,
            "total_pnl": self.total_pnl,
            "win_rate": self.win_rate,
            "sharpe": self.sharpe,
            "max_drawdown_pct": self.max_drawdown_pct,
            "last_evaluated": self.last_evaluated,
            "parent_ids": self.parent_ids,
            "origin": self.origin,
        }

    @classmethod
    def from_dict(cls, d: dict) -> StrategyRecord:
        return cls(
            genome=Genome.from_dict(d["genome"]),
            strategy_id=d["strategy_id"],
            entered_at=d["entered_at"],
            generation=d.get("generation", 0),
            n_trades=d.get("n_trades", 0),
            total_pnl=d.get("total_pnl", 0.0),
            win_rate=d.get("win_rate", 0.0),
            sharpe=d.get("sharpe", 0.0),
            max_drawdown_pct=d.get("max_drawdown_pct", 0.0),
            last_evaluated=d.get("last_evaluated", 0.0),
            parent_ids=d.get("parent_ids", []),
            origin=d.get("origin", "unknown"),
        )


@dataclass
class TournamentConfig:
    """Configuration for the paper trading tournament."""
    max_strategies: int = 20           # max concurrent paper traders
    min_strategies: int = 10           # refill if below this
    eval_interval_s: float = 3600      # evaluate every hour
    min_age_hours: float = 24          # minimum age before pruning eligible
    prune_pct: float = 0.25            # prune bottom 25%
    mutation_rate: float = 0.15        # mutation rate for offspring
    inject_count: int = 2              # inject N from GA lab per cycle
    symbols: list[str] = field(default_factory=lambda: ["BTCUSDT"])


class TournamentCoordinator:
    """
    Manages the paper trading tournament lifecycle.

    Responsibilities:
    - Maintain a population of paper-trading strategies
    - Periodically evaluate and rank strategies
    - Prune underperformers
    - Clone and mutate top performers
    - Inject new candidates from the GA lab
    - Persist state for crash recovery
    """

    def __init__(
        self,
        config: TournamentConfig,
        state_dir: Path,
    ) -> None:
        self.config = config
        self.state_dir = state_dir
        self.state_dir.mkdir(parents=True, exist_ok=True)

        self.strategies: dict[str, StrategyRecord] = {}
        self.history: list[dict] = []
        self._cycle_count = 0

        # Load state if exists
        state_path = self.state_dir / "tournament_state.json"
        if state_path.exists():
            self._load_state(state_path)

    def add_strategy(
        self,
        genome: Genome,
        origin: str = "random",
    ) -> str:
        """
        Add a new strategy to the tournament.

        Returns the strategy ID for NautilusTrader registration.
        """
        strategy_id = f"EVO-{genome.genome_id}"

        record = StrategyRecord(
            genome=genome,
            strategy_id=strategy_id,
            entered_at=time.time(),
            generation=genome.generation,
            parent_ids=genome.parent_ids,
            origin=origin,
        )
        self.strategies[strategy_id] = record
        return strategy_id

    def update_performance(
        self,
        strategy_id: str,
        n_trades: int,
        total_pnl: float,
        win_rate: float,
        sharpe: float,
        max_drawdown_pct: float,
    ) -> None:
        """Update a strategy's performance metrics (called during evaluation)."""
        if strategy_id not in self.strategies:
            return
        record = self.strategies[strategy_id]
        record.n_trades = n_trades
        record.total_pnl = total_pnl
        record.win_rate = win_rate
        record.sharpe = sharpe
        record.max_drawdown_pct = max_drawdown_pct
        record.last_evaluated = time.time()

    def run_cycle(
        self,
        ga_candidates: list[Genome] | None = None,
    ) -> dict[str, Any]:
        """
        Run one tournament cycle: evaluate → prune → mutate → inject.

        Parameters
        ----------
        ga_candidates : list[Genome], optional
            Top genomes from the GA backtest lab to inject.

        Returns
        -------
        dict
            Cycle report with actions taken.
        """
        self._cycle_count += 1
        report: dict[str, Any] = {
            "cycle": self._cycle_count,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "pre_count": len(self.strategies),
            "pruned": [],
            "mutated": [],
            "injected": [],
        }

        # 1. Rank all strategies
        ranked = self._rank_strategies()
        report["rankings"] = [
            {
                "id": r.strategy_id,
                "fitness": r.fitness,
                "trades": r.n_trades,
                "pnl": r.total_pnl,
                "wr": r.win_rate,
                "age_h": round(r.age_hours, 1),
                "origin": r.origin,
            }
            for r in ranked
        ]

        # 2. Prune bottom performers (if old enough)
        pruned_ids = self._prune(ranked)
        report["pruned"] = pruned_ids

        # Re-rank after pruning so mutate_winners only uses surviving strategies
        ranked_after_prune = self._rank_strategies()

        # 3. Clone and mutate top performers
        mutated_ids = self._mutate_winners(ranked_after_prune)
        report["mutated"] = mutated_ids

        # 4. Inject GA candidates
        injected_ids = self._inject(ga_candidates)
        report["injected"] = injected_ids

        # 5. Fill remaining slots with random genomes
        filled_ids = self._fill_population()
        report["random_fill"] = filled_ids

        report["post_count"] = len(self.strategies)

        # Save state and history
        self.history.append(report)
        if len(self.history) > 100:
            self.history = self.history[-100:]
        self._save_state()

        return report

    def _rank_strategies(self) -> list[StrategyRecord]:
        """Rank strategies by fitness, best first."""
        return sorted(
            self.strategies.values(),
            key=lambda r: r.fitness,
            reverse=True,
        )

    def _prune(self, ranked: list[StrategyRecord]) -> list[str]:
        """Remove bottom performers that are old enough."""
        if len(ranked) <= self.config.min_strategies:
            return []

        n_prune = max(1, int(len(ranked) * self.config.prune_pct))
        pruned = []

        # Prune from the bottom
        for record in reversed(ranked):
            if len(pruned) >= n_prune:
                break
            if record.age_hours < self.config.min_age_hours:
                continue  # Too young to prune
            if len(self.strategies) - len(pruned) <= self.config.min_strategies:
                break
            if record.n_trades < 3:
                # Definitely prune strategies that aren't trading
                pruned.append(record.strategy_id)
                continue
            pruned.append(record.strategy_id)

        for sid in pruned:
            del self.strategies[sid]

        return pruned

    def _mutate_winners(self, ranked: list[StrategyRecord]) -> list[str]:
        """Clone and mutate top performers to fill gaps."""
        if not ranked:
            return []

        # Take top 25% as parents
        n_parents = max(1, int(len(ranked) * 0.25))
        parents = ranked[:n_parents]

        mutated = []
        slots_available = self.config.max_strategies - len(self.strategies)
        n_mutate = min(n_parents, slots_available)

        for i in range(n_mutate):
            parent = parents[i % len(parents)]
            child_genome = mutate(parent.genome, rate=self.config.mutation_rate)
            sid = self.add_strategy(child_genome, origin="mutation")
            mutated.append(sid)

        return mutated

    def _inject(self, candidates: list[Genome] | None) -> list[str]:
        """Inject top GA lab candidates."""
        if not candidates:
            return []

        injected = []
        slots_available = self.config.max_strategies - len(self.strategies)
        n_inject = min(self.config.inject_count, len(candidates), slots_available)

        for i in range(n_inject):
            sid = self.add_strategy(candidates[i], origin="injection")
            injected.append(sid)

        return injected

    def _fill_population(self) -> list[str]:
        """Fill remaining slots with random genomes."""
        filled = []
        while len(self.strategies) < self.config.min_strategies:
            genome = random_genome(symbols=self.config.symbols)
            sid = self.add_strategy(genome, origin="random")
            filled.append(sid)
        return filled

    def get_leaderboard(self, top_n: int = 10) -> str:
        """Get formatted leaderboard string."""
        ranked = self._rank_strategies()
        lines = [
            f"\n{'='*70}",
            f" Tournament Leaderboard — {len(ranked)} strategies",
            f"{'='*70}",
            f" {'Rank':<5} {'ID':<18} {'Fitness':>8} {'Trades':>7} {'PnL':>10} {'WR':>6} {'Age(h)':>7} {'Origin':<10}",
            f" {'-'*5} {'-'*18} {'-'*8} {'-'*7} {'-'*10} {'-'*6} {'-'*7} {'-'*10}",
        ]

        for i, r in enumerate(ranked[:top_n]):
            lines.append(
                f" {i+1:<5} {r.strategy_id:<18} {r.fitness:>8.3f} "
                f"{r.n_trades:>7} {r.total_pnl:>10.2f} "
                f"{r.win_rate:>5.1%} {r.age_hours:>7.1f} {r.origin:<10}"
            )

        return "\n".join(lines)

    def _save_state(self) -> None:
        """Persist tournament state."""
        state = {
            "cycle_count": self._cycle_count,
            "strategies": {
                sid: record.to_dict()
                for sid, record in self.strategies.items()
            },
            "config": {
                "max_strategies": self.config.max_strategies,
                "min_strategies": self.config.min_strategies,
                "eval_interval_s": self.config.eval_interval_s,
                "prune_pct": self.config.prune_pct,
                "symbols": self.config.symbols,
            },
            "history_last_10": self.history[-10:],
        }
        path = self.state_dir / "tournament_state.json"
        path.write_text(json.dumps(state, indent=2))

    def _load_state(self, path: Path) -> None:
        """Load persisted state."""
        data = json.loads(path.read_text())
        self._cycle_count = data.get("cycle_count", 0)
        for sid, record_dict in data.get("strategies", {}).items():
            self.strategies[sid] = StrategyRecord.from_dict(record_dict)
        self.history = data.get("history_last_10", [])
