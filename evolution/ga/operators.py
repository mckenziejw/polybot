"""
Genetic algorithm operators: random genome generation, mutation, crossover, selection.
"""

from __future__ import annotations

import copy
import random
from typing import Sequence

from .genome import (
    BandOutput,
    Condition,
    EntryRule,
    ExitRule,
    FilterRule,
    Genome,
    IndicatorGene,
    IndicatorOutputScale,
    IndicatorType,
    INDICATOR_PARAM_RANGES,
    INDICATOR_SCALE,
    THRESHOLD_COMPATIBLE,
    THRESHOLD_RANGES,
    THRESHOLD_MUTATION_SCALE,
    Side,
)


def random_indicator(max_period: int | None = None) -> IndicatorGene:
    """Generate a random indicator with valid parameters.

    Parameters
    ----------
    max_period : int, optional
        Cap period parameters to this value (useful for short data windows).
    """
    ind_type = random.choice(list(IndicatorType))
    param_ranges = INDICATOR_PARAM_RANGES[ind_type.value]
    params = {}
    for param_name, (lo, hi) in param_ranges.items():
        if param_name == "period" or param_name.startswith("period"):
            effective_hi = min(int(hi), max_period) if max_period else int(hi)
            effective_hi = max(effective_hi, int(lo))  # ensure hi >= lo
            params[param_name] = random.randint(int(lo), effective_hi)
        else:
            params[param_name] = round(random.uniform(lo, hi), 2)

    # MACD constraint: fast_period < slow_period
    if ind_type == IndicatorType.MACD:
        if params["fast_period"] >= params["slow_period"]:
            params["fast_period"], params["slow_period"] = params["slow_period"], params["fast_period"]
            if params["fast_period"] == params["slow_period"]:
                params["slow_period"] = params["fast_period"] + 5

    # For band indicators (BB, DC, KC), randomly select which band to output
    band = None
    if ind_type in (IndicatorType.BB, IndicatorType.DC, IndicatorType.KC):
        band = random.choice(list(BandOutput))

    return IndicatorGene(type=ind_type, params=params, band=band)


def _get_indicator_scale(indicator_id: str, indicators: list[IndicatorGene]) -> IndicatorOutputScale:
    """Look up the output scale for an indicator by its ID."""
    for ind in indicators:
        if ind.id == indicator_id:
            return ind.output_scale
    return IndicatorOutputScale.NORMALIZED  # fallback


def random_entry_rule(indicators: list[IndicatorGene]) -> EntryRule:
    """Generate a random entry rule referencing existing indicators."""
    if not indicators:
        raise ValueError("Need at least one indicator to create entry rules")

    side = random.choice(list(Side))

    # Separate indicators by type for smart pairing
    threshold_inds = [ind for ind in indicators if ind.output_scale in THRESHOLD_COMPATIBLE]
    price_inds = [ind for ind in indicators if ind.output_scale == IndicatorOutputScale.PRICE]

    # Strategy: prefer threshold-based rules (they actually fire reliably)
    # Only use crossover when we have 2+ price-scale indicators to compare
    use_crossover = (
        len(price_inds) >= 2
        and (not threshold_inds or random.random() < 0.3)
    )

    if use_crossover:
        # Crossover between two SAME-SCALE indicators (both price-scale)
        pair = random.sample(price_inds, 2)
        condition = random.choice([Condition.CROSSOVER_ABOVE, Condition.CROSSOVER_BELOW])
        return EntryRule(
            indicator_id=pair[0].id,
            condition=condition,
            value=0.0,
            side=side,
            compare_indicator_id=pair[1].id,
        )

    # Threshold-based on a compatible indicator
    if threshold_inds:
        ind = random.choice(threshold_inds)
        scale = ind.output_scale
        lo, hi = THRESHOLD_RANGES[scale]
        condition = random.choice([
            Condition.CROSSES_ABOVE,
            Condition.CROSSES_BELOW,
            Condition.GREATER_THAN,
            Condition.LESS_THAN,
        ])
        value = round(random.uniform(lo, hi), 3)
        return EntryRule(
            indicator_id=ind.id,
            condition=condition,
            value=value,
            side=side,
        )

    # Last resort: crossover between any two price-scale indicators
    if len(price_inds) >= 2:
        pair = random.sample(price_inds, 2)
        condition = random.choice([Condition.CROSSOVER_ABOVE, Condition.CROSSOVER_BELOW])
        return EntryRule(
            indicator_id=pair[0].id,
            condition=condition,
            value=0.0,
            side=side,
            compare_indicator_id=pair[1].id,
        )

    # Single price-scale indicator: use crossover with itself at different scales
    # (shouldn't normally reach here due to THRESHOLD_TYPES guarantee in random_genome)
    ind = random.choice(indicators)
    if ind.output_scale in THRESHOLD_COMPATIBLE:
        lo, hi = THRESHOLD_RANGES[ind.output_scale]
        value = round(random.uniform(lo, hi), 3)
    else:
        value = 0.0
    return EntryRule(
        indicator_id=ind.id,
        condition=random.choice([Condition.GREATER_THAN, Condition.LESS_THAN]),
        value=value,
        side=side,
    )


def random_exit_rule(indicators: list[IndicatorGene]) -> ExitRule:
    """Generate a random exit rule."""
    has_atr = any(ind.type == IndicatorType.ATR for ind in indicators)
    exit_types = ["take_profit", "stop_loss", "time_limit", "indicator", "trailing_stop"]
    if has_atr:
        exit_types.extend(["atr_stop_loss", "atr_take_profit"])
    exit_type = random.choice(exit_types)

    if exit_type == "take_profit":
        return ExitRule(type="take_profit", value=round(random.uniform(0.5, 10.0), 2))
    elif exit_type == "stop_loss":
        return ExitRule(type="stop_loss", value=round(random.uniform(0.5, 5.0), 2))
    elif exit_type == "trailing_stop":
        return ExitRule(type="trailing_stop", value=round(random.uniform(0.5, 5.0), 2))
    elif exit_type == "atr_stop_loss":
        return ExitRule(type="atr_stop_loss", value=round(random.uniform(1.0, 4.0), 2))
    elif exit_type == "atr_take_profit":
        return ExitRule(type="atr_take_profit", value=round(random.uniform(1.0, 6.0), 2))
    elif exit_type == "time_limit":
        return ExitRule(type="time_limit", value=random.choice([60, 120, 300, 600, 1800, 3600]))
    else:
        # Indicator-based exit — only use normalized/unbounded indicators
        if not indicators:
            return ExitRule(type="stop_loss", value=round(random.uniform(0.5, 5.0), 2))

        # Filter to threshold-compatible indicators
        compatible = [ind for ind in indicators if ind.output_scale in THRESHOLD_COMPATIBLE]
        if not compatible:
            return ExitRule(type="stop_loss", value=round(random.uniform(0.5, 5.0), 2))

        ind = random.choice(compatible)
        scale = ind.output_scale
        lo, hi = THRESHOLD_RANGES[scale]
        return ExitRule(
            type="indicator",
            value=round(random.uniform(lo, hi), 3),
            indicator_id=ind.id,
            condition=random.choice([
                Condition.CROSSES_ABOVE,
                Condition.CROSSES_BELOW,
                Condition.GREATER_THAN,
                Condition.LESS_THAN,
            ]),
        )


def random_filter_rule(indicators: list[IndicatorGene]) -> FilterRule | None:
    """Generate a random filter rule referencing a threshold-compatible indicator."""
    compatible = [ind for ind in indicators if ind.output_scale in THRESHOLD_COMPATIBLE]
    if not compatible:
        return None
    ind = random.choice(compatible)
    scale = ind.output_scale
    lo, hi = THRESHOLD_RANGES[scale]
    condition = random.choice([Condition.GREATER_THAN, Condition.LESS_THAN])
    value = round(random.uniform(lo, hi), 3)
    return FilterRule(indicator_id=ind.id, condition=condition, value=value)


def random_genome(
    symbols: list[str] | None = None,
    n_indicators: tuple[int, int] = (2, 6),
    n_entry_rules: tuple[int, int] = (1, 3),
    n_exit_rules: tuple[int, int] = (1, 3),
    max_period: int | None = None,
) -> Genome:
    """
    Generate a completely random genome.

    Parameters
    ----------
    symbols : list[str], optional
        Asset symbols. Defaults to ["BTCUSDT"].
    n_indicators : tuple[int, int]
        Min/max number of indicators.
    n_entry_rules : tuple[int, int]
        Min/max number of entry rules.
    n_exit_rules : tuple[int, int]
        Min/max number of exit rules.

    Returns
    -------
    Genome
    """
    if symbols is None:
        symbols = ["BTCUSDT"]

    # Generate indicators — ensure at least one threshold-compatible (non-price) indicator
    n_ind = random.randint(*n_indicators)
    indicators = [random_indicator(max_period=max_period) for _ in range(n_ind)]

    # Ensure at least one momentum indicator for threshold-based rules
    THRESHOLD_TYPES = [IndicatorType.RSI, IndicatorType.STOCH, IndicatorType.CMO]
    has_normalized = any(ind.type in THRESHOLD_TYPES for ind in indicators)
    if not has_normalized:
        norm_type = random.choice(THRESHOLD_TYPES)
        param_ranges = INDICATOR_PARAM_RANGES[norm_type.value]
        params = {}
        for param_name, (lo, hi) in param_ranges.items():
            effective_hi = min(int(hi), max_period) if max_period else int(hi)
            effective_hi = max(effective_hi, int(lo))
            if param_name.startswith("period"):
                params[param_name] = random.randint(int(lo), effective_hi)
            else:
                params[param_name] = round(random.uniform(lo, hi), 2)
        indicators.append(IndicatorGene(type=norm_type, params=params))

    # Generate entry rules
    n_entry = random.randint(*n_entry_rules)
    entry_rules = [random_entry_rule(indicators) for _ in range(n_entry)]

    # Generate exit rules
    n_exit = random.randint(*n_exit_rules)
    exit_rules = [random_exit_rule(indicators) for _ in range(n_exit)]

    # Always include at least one stop loss and time limit
    has_sl = any(r.type in ("stop_loss", "trailing_stop", "atr_stop_loss") for r in exit_rules)
    has_tl = any(r.type == "time_limit" for r in exit_rules)
    if not has_sl:
        exit_rules.append(ExitRule(type="stop_loss", value=round(random.uniform(1.0, 5.0), 2)))
    if not has_tl:
        exit_rules.append(ExitRule(type="time_limit", value=random.choice([300, 600, 1800, 3600])))

    # Filter rules (optional, ~30% of genomes)
    filter_rules = []
    if random.random() < 0.3:
        n_filters = random.randint(1, 2)
        for _ in range(n_filters):
            fr = random_filter_rule(indicators)
            if fr:
                filter_rules.append(fr)

    # min_entry_matches: usually 1 (OR), sometimes higher
    min_matches = 1
    if n_entry > 1 and random.random() < 0.3:
        min_matches = random.randint(1, n_entry)

    return Genome(
        symbols=symbols,
        bar_period_minutes=random.choice([1, 5, 15, 30, 60, 240]),
        indicators=indicators,
        entry_rules=entry_rules,
        min_entry_matches=min_matches,
        exit_rules=exit_rules,
        filter_rules=filter_rules,
        position_size_usd=round(random.uniform(50, 500), 0),
        max_open_positions=random.randint(1, 3),
        max_drawdown_pct=round(random.uniform(5.0, 20.0), 1),
        cooldown_bars=random.choice([0, 0, 0, 1, 2, 5]),
        direction_bias=random.choice(["both", "both", "both", "long_only", "short_only"]),
    )


def mutate(genome: Genome, rate: float = 0.1, max_period: int | None = None) -> Genome:
    """
    Mutate a genome by randomly perturbing its parameters.

    Parameters
    ----------
    genome : Genome
        The genome to mutate.
    rate : float
        Probability of each mutation occurring (0-1).
    max_period : int, optional
        Cap period parameters for newly added indicators.

    Returns
    -------
    Genome
        A new mutated genome (original is not modified).
    """
    g = copy.deepcopy(genome)
    g.parent_ids = [genome.genome_id]
    g.genome_id = Genome().genome_id  # new ID

    # Mutate indicator parameters
    for ind in g.indicators:
        if random.random() < rate:
            param_ranges = INDICATOR_PARAM_RANGES[ind.type.value]
            for param_name, (lo, hi) in param_ranges.items():
                if random.random() < 0.5:
                    current = ind.params[param_name]
                    # Cap period to max_period if set
                    effective_hi = hi
                    if max_period and param_name.startswith("period"):
                        effective_hi = min(hi, max_period)
                    # Gaussian perturbation, 10% of range
                    noise = random.gauss(0, (effective_hi - lo) * 0.1)
                    new_val = max(lo, min(effective_hi, current + noise))
                    if param_name.startswith("period"):
                        new_val = max(int(lo), int(round(new_val)))
                    else:
                        new_val = round(new_val, 2)
                    ind.params[param_name] = new_val

    # Mutate band selection for BB/DC/KC (rare)
    for ind in g.indicators:
        if ind.type in (IndicatorType.BB, IndicatorType.DC, IndicatorType.KC):
            if random.random() < rate * 0.3:
                ind.band = random.choice(list(BandOutput))

    # Add/remove indicator (rare)
    if random.random() < rate * 0.3:
        if len(g.indicators) > 1 and random.random() < 0.5:
            # Remove a random indicator
            idx = random.randrange(len(g.indicators))
            removed_id = g.indicators[idx].id
            g.indicators.pop(idx)
            # Remove rules referencing it (both indicator_id and compare_indicator_id)
            g.entry_rules = [
                r for r in g.entry_rules
                if r.indicator_id != removed_id and r.compare_indicator_id != removed_id
            ]
            g.exit_rules = [
                r for r in g.exit_rules
                if r.type != "indicator" or r.indicator_id != removed_id
            ]
        else:
            # Add a new indicator
            g.indicators.append(random_indicator(max_period=max_period))

    # Mutate entry rule thresholds (scale-aware)
    for rule in g.entry_rules:
        if random.random() < rate:
            scale = _get_indicator_scale(rule.indicator_id, g.indicators)
            if scale in THRESHOLD_COMPATIBLE:
                lo, hi = THRESHOLD_RANGES[scale]
                noise_scale = THRESHOLD_MUTATION_SCALE[scale]
                rule.value = round(rule.value + random.gauss(0, noise_scale), 3)
                rule.value = max(lo, min(hi, rule.value))

    # Mutate exit rule values
    for rule in g.exit_rules:
        if random.random() < rate:
            if rule.type in ("take_profit", "stop_loss", "trailing_stop"):
                rule.value = round(rule.value + random.gauss(0, 0.5), 2)
                rule.value = max(0.1, min(20.0, rule.value))
            elif rule.type in ("atr_stop_loss", "atr_take_profit"):
                rule.value = round(rule.value + random.gauss(0, 0.3), 2)
                rule.value = max(0.5, min(6.0, rule.value))
            elif rule.type == "time_limit":
                choices = [60, 120, 300, 600, 1800, 3600]
                rule.value = random.choice(choices)
            elif rule.type == "indicator":
                scale = _get_indicator_scale(rule.indicator_id, g.indicators)
                if scale in THRESHOLD_COMPATIBLE:
                    lo, hi = THRESHOLD_RANGES[scale]
                    noise_scale = THRESHOLD_MUTATION_SCALE[scale]
                    rule.value = round(rule.value + random.gauss(0, noise_scale), 3)
                    rule.value = max(lo, min(hi, rule.value))

    # Mutate bar period (rare)
    if random.random() < rate * 0.2:
        g.bar_period_minutes = random.choice([1, 5, 15, 30, 60, 240])

    # Mutate position sizing
    if random.random() < rate:
        g.position_size_usd = round(
            max(50, min(1000, g.position_size_usd + random.gauss(0, 50))), 0
        )

    # Mutate min_entry_matches (rare)
    if random.random() < rate * 0.2 and len(g.entry_rules) > 1:
        g.min_entry_matches = random.randint(1, len(g.entry_rules))

    # Mutate cooldown (rare)
    if random.random() < rate * 0.2:
        g.cooldown_bars = random.choice([0, 0, 1, 2, 5, 10])

    # Mutate direction bias (rare)
    if random.random() < rate * 0.1:
        g.direction_bias = random.choice(["both", "long_only", "short_only"])

    # Mutate filter rules (rare)
    for rule in g.filter_rules:
        if random.random() < rate:
            scale = _get_indicator_scale(rule.indicator_id, g.indicators)
            if scale in THRESHOLD_COMPATIBLE:
                lo, hi = THRESHOLD_RANGES[scale]
                noise_scale = THRESHOLD_MUTATION_SCALE[scale]
                rule.value = round(rule.value + random.gauss(0, noise_scale), 3)
                rule.value = max(lo, min(hi, rule.value))

    # Add/remove filter rule (rare)
    if random.random() < rate * 0.2:
        if g.filter_rules and random.random() < 0.5:
            g.filter_rules.pop(random.randrange(len(g.filter_rules)))
        elif len(g.filter_rules) < 2:
            fr = random_filter_rule(g.indicators)
            if fr:
                g.filter_rules.append(fr)

    # Dedup identical indicators and prune unused ones
    g.indicators = _dedup_indicators(g.indicators)
    _prune_unused_indicators(g)

    return g


def _dedup_indicators(indicators: list[IndicatorGene]) -> list[IndicatorGene]:
    """Remove duplicate indicators (same type + params). Keeps the first occurrence."""
    seen: set[str] = set()
    result: list[IndicatorGene] = []
    for ind in indicators:
        # Canonical key: type + sorted params (ignoring ID)
        key = (ind.type.value, tuple(sorted(ind.params.items())),
               ind.band.value if ind.band else "")
        if key not in seen:
            seen.add(key)
            result.append(ind)
    return result


def _prune_unused_indicators(genome: Genome) -> None:
    """Remove indicators not referenced by any entry, exit, or filter rule."""
    used_ids: set[str] = set()
    for r in genome.entry_rules:
        used_ids.add(r.indicator_id)
        if r.compare_indicator_id:
            used_ids.add(r.compare_indicator_id)
    for r in genome.exit_rules:
        if r.indicator_id:
            used_ids.add(r.indicator_id)
    for r in genome.filter_rules:
        used_ids.add(r.indicator_id)
    # Keep ATR even if not directly referenced (used by atr_stop_loss/atr_take_profit)
    if any(r.type in ("atr_stop_loss", "atr_take_profit") for r in genome.exit_rules):
        for ind in genome.indicators:
            if ind.type == IndicatorType.ATR:
                used_ids.add(ind.id)
    genome.indicators = [ind for ind in genome.indicators if ind.id in used_ids]
    # Ensure at least one indicator remains
    if not genome.indicators:
        genome.indicators.append(random_indicator())


def crossover(parent_a: Genome, parent_b: Genome) -> Genome:
    """
    Create offspring by combining two parent genomes.

    Uses uniform crossover on indicators and rules.

    Parameters
    ----------
    parent_a, parent_b : Genome
        Parent genomes.

    Returns
    -------
    Genome
        New offspring genome.
    """
    child = Genome(
        parent_ids=[parent_a.genome_id, parent_b.genome_id],
        symbols=random.choice([parent_a.symbols, parent_b.symbols]),
        bar_period_minutes=random.choice([parent_a.bar_period_minutes, parent_b.bar_period_minutes]),
        position_size_usd=round((parent_a.position_size_usd + parent_b.position_size_usd) / 2, 0),
        max_open_positions=random.choice([parent_a.max_open_positions, parent_b.max_open_positions]),
        max_drawdown_pct=round((parent_a.max_drawdown_pct + parent_b.max_drawdown_pct) / 2, 1),
        min_entry_matches=random.choice([parent_a.min_entry_matches, parent_b.min_entry_matches]),
        cooldown_bars=random.choice([parent_a.cooldown_bars, parent_b.cooldown_bars]),
        direction_bias=random.choice([parent_a.direction_bias, parent_b.direction_bias]),
    )

    # Uniform crossover on indicators: pick from each parent randomly
    all_indicators = parent_a.indicators + parent_b.indicators
    n_indicators = random.randint(
        min(len(parent_a.indicators), len(parent_b.indicators)),
        max(len(parent_a.indicators), len(parent_b.indicators)),
    )
    n_indicators = max(2, min(n_indicators, len(all_indicators)))
    sampled = copy.deepcopy(random.sample(all_indicators, n_indicators))

    # Deduplicate: first by ID, then by type+params
    seen_ids: set[str] = set()
    deduped: list[IndicatorGene] = []
    for ind in sampled:
        if ind.id in seen_ids:
            import uuid as _uuid
            period = int(ind.params.get("period", ind.params.get("period_k", 0)))
            ind.id = f"{ind.type.value}_{period}_{_uuid.uuid4().hex[:4]}"
        seen_ids.add(ind.id)
        deduped.append(ind)
    child.indicators = _dedup_indicators(deduped)

    child_indicator_ids = {ind.id for ind in child.indicators}

    # Crossover entry rules — pick from parents, filter for valid indicator refs
    all_entry_rules = parent_a.entry_rules + parent_b.entry_rules
    valid_entry_rules = [
        copy.deepcopy(r) for r in all_entry_rules
        if r.indicator_id in child_indicator_ids
        and (not r.compare_indicator_id or r.compare_indicator_id in child_indicator_ids)
    ]
    if valid_entry_rules:
        n_entry = random.randint(1, min(3, len(valid_entry_rules)))
        child.entry_rules = random.sample(valid_entry_rules, n_entry)
    else:
        # Generate a new entry rule if crossover produced no valid ones
        child.entry_rules = [random_entry_rule(child.indicators)]

    # Clamp min_entry_matches to actual rule count
    child.min_entry_matches = min(child.min_entry_matches, len(child.entry_rules))

    # Crossover exit rules
    all_exit_rules = parent_a.exit_rules + parent_b.exit_rules
    valid_exit_rules = []
    for r in all_exit_rules:
        r = copy.deepcopy(r)
        if r.type == "indicator" and r.indicator_id not in child_indicator_ids:
            continue
        valid_exit_rules.append(r)
    if valid_exit_rules:
        n_exit = random.randint(1, min(4, len(valid_exit_rules)))
        child.exit_rules = random.sample(valid_exit_rules, n_exit)
    else:
        child.exit_rules = [ExitRule(type="stop_loss", value=3.0)]

    # Ensure at least one stop loss
    if not any(r.type in ("stop_loss", "trailing_stop", "atr_stop_loss") for r in child.exit_rules):
        child.exit_rules.append(ExitRule(
            type="stop_loss",
            value=round(random.uniform(1.0, 5.0), 2),
        ))

    # Crossover filter rules
    all_filter_rules = parent_a.filter_rules + parent_b.filter_rules
    valid_filter_rules = [
        copy.deepcopy(r) for r in all_filter_rules
        if r.indicator_id in child_indicator_ids
    ]
    if valid_filter_rules:
        n_filters = random.randint(0, min(2, len(valid_filter_rules)))
        child.filter_rules = random.sample(valid_filter_rules, n_filters) if n_filters > 0 else []

    # Prune unused indicators to keep genomes lean
    _prune_unused_indicators(child)

    return child


def tournament_select(
    population: list[tuple[Genome, float]],
    k: int = 3,
) -> Genome:
    """
    Tournament selection: pick k random individuals, return the fittest.

    Parameters
    ----------
    population : list[tuple[Genome, float]]
        List of (genome, fitness) tuples.
    k : int
        Tournament size.

    Returns
    -------
    Genome
        The winner's genome.
    """
    if not population:
        raise ValueError("Cannot select from empty population")
    contestants = random.sample(population, min(k, len(population)))
    winner = max(contestants, key=lambda x: x[1])
    return winner[0]


def evolve_population(
    population: list[tuple[Genome, float]],
    pop_size: int,
    elitism_pct: float = 0.05,
    crossover_pct: float = 0.5,
    mutation_rate: float = 0.2,
    tournament_k: int = 3,
    max_period: int | None = None,
    immigration_pct: float = 0.2,
) -> list[Genome]:
    """
    Produce the next generation from a scored population.

    Parameters
    ----------
    population : list[tuple[Genome, float]]
        Current generation with fitness scores, sorted best-first.
    pop_size : int
        Target population size.
    elitism_pct : float
        Fraction of top performers to keep unchanged.
    crossover_pct : float
        Fraction produced by crossover (remainder by mutation).
    mutation_rate : float
        Mutation probability per gene.
    tournament_k : int
        Tournament selection group size.
    immigration_pct : float
        Fraction of fresh random genomes injected each generation.

    Returns
    -------
    list[Genome]
        Next generation of genomes.
    """
    population = sorted(population, key=lambda x: x[1], reverse=True)
    next_gen: list[Genome] = []

    # Elitism: keep top performers unchanged
    n_elite = max(1, int(pop_size * elitism_pct))
    for genome, _ in population[:n_elite]:
        elite = copy.deepcopy(genome)
        elite.generation += 1
        next_gen.append(elite)

    # Immigration: inject fresh random genomes to maintain diversity
    n_immigrants = max(1, int(pop_size * immigration_pct))
    symbols = population[0][0].symbols if population else ["BTCUSDT"]
    gen_num = population[0][0].generation + 1 if population else 0
    for _ in range(n_immigrants):
        immigrant = random_genome(symbols=symbols, max_period=max_period)
        immigrant.generation = gen_num
        next_gen.append(immigrant)

    # Fill remainder with crossover and mutation
    remaining = pop_size - n_elite - n_immigrants
    n_crossover = int(remaining * crossover_pct)
    n_mutation = remaining - n_crossover

    # Crossover offspring
    for _ in range(n_crossover):
        parent_a = tournament_select(population, tournament_k)
        parent_b = tournament_select(population, tournament_k)
        child = crossover(parent_a, parent_b)
        child.generation = gen_num
        # Light mutation on offspring — preserve crossover parent_ids
        if random.random() < 0.5:
            crossover_parents = child.parent_ids
            child = mutate(child, rate=mutation_rate * 0.5, max_period=max_period)
            child.parent_ids = crossover_parents
        next_gen.append(child)

    # Mutation offspring
    for _ in range(n_mutation):
        parent = tournament_select(population, tournament_k)
        child = mutate(parent, rate=mutation_rate, max_period=max_period)
        child.generation = gen_num
        next_gen.append(child)

    return next_gen
