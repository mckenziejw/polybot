"""
Genome representation for the strategy evolution system.

A genome encodes a complete trading strategy as a structured parameter set:
- Which indicators to compute
- Indicator parameters (period, thresholds, etc.)
- Entry/exit rules (composable conditions on indicator values)
- Position sizing and risk management
- Asset scope

Genomes are serializable (JSON) for persistence and can be mutated/crossed over
by the GA operators.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class IndicatorType(str, Enum):
    """Available NautilusTrader indicators."""
    # Averages
    EMA = "EMA"
    SMA = "SMA"
    HMA = "HMA"     # Hull Moving Average
    DEMA = "DEMA"    # Double Exponential Moving Average
    WMA = "WMA"      # Weighted Moving Average

    # Momentum (output 0-1 in NautilusTrader)
    RSI = "RSI"
    STOCH = "STOCH"  # Stochastics
    CMO = "CMO"      # Chande Momentum Oscillator

    # Momentum (unbounded output)
    ROC = "ROC"      # Rate of Change (percentage)
    CCI = "CCI"      # Commodity Channel Index (typically -200 to +200)

    # Volatility
    BB = "BB"        # Bollinger Bands (outputs: upper, middle, lower)
    ATR = "ATR"      # Average True Range (price-scale)
    DC = "DC"        # Donchian Channel (outputs: upper, middle, lower)
    KC = "KC"        # Keltner Channel (outputs: upper, middle, lower)
    VHF = "VHF"      # Vertical Horizontal Filter

    # Trend
    MACD = "MACD"    # Moving Average Convergence Divergence (unbounded)
    ADX = "ADX"      # Average Directional Index (0-100, via DirectionalMovement)
    AROON = "AROON"  # Aroon Oscillator (-100 to +100)


class IndicatorOutputScale(str, Enum):
    """Scale of indicator output, used for generating appropriate thresholds."""
    NORMALIZED = "normalized"    # 0-1 range (RSI)
    PERCENTAGE = "percentage"    # 0-100 range (Stochastics)
    CENTERED = "centered"        # -100 to +100 range (CMO)
    PRICE = "price"              # Price-scale (EMA, SMA, HMA, DEMA, WMA, BB, DC, KC)
    UNBOUNDED = "unbounded"      # Can be any value (CCI, ROC, ATR, VHF)


# Map each indicator to its output scale
INDICATOR_SCALE: dict[str, IndicatorOutputScale] = {
    "EMA": IndicatorOutputScale.PRICE,
    "SMA": IndicatorOutputScale.PRICE,
    "HMA": IndicatorOutputScale.PRICE,
    "DEMA": IndicatorOutputScale.PRICE,
    "WMA": IndicatorOutputScale.PRICE,
    "RSI": IndicatorOutputScale.NORMALIZED,
    "STOCH": IndicatorOutputScale.PERCENTAGE,
    "CMO": IndicatorOutputScale.CENTERED,
    "ROC": IndicatorOutputScale.UNBOUNDED,
    "CCI": IndicatorOutputScale.UNBOUNDED,
    "BB": IndicatorOutputScale.PRICE,
    "ATR": IndicatorOutputScale.UNBOUNDED,
    "DC": IndicatorOutputScale.PRICE,
    "KC": IndicatorOutputScale.PRICE,
    "VHF": IndicatorOutputScale.UNBOUNDED,
    "MACD": IndicatorOutputScale.UNBOUNDED,
    "ADX": IndicatorOutputScale.UNBOUNDED,
    "AROON": IndicatorOutputScale.CENTERED,
}

# Which indicators support threshold-based entry rules (not price-scale ones)
# Price-scale indicators should use crossover conditions instead
THRESHOLD_COMPATIBLE = {
    IndicatorOutputScale.NORMALIZED,  # 0-1 range: threshold comparison makes sense
    IndicatorOutputScale.PERCENTAGE,  # 0-100 range: threshold comparison makes sense
    IndicatorOutputScale.CENTERED,    # -100 to +100: threshold comparison makes sense
    IndicatorOutputScale.UNBOUNDED,   # CCI, ROC: threshold comparison makes sense
}

# Parameter ranges for each indicator type
INDICATOR_PARAM_RANGES: dict[str, dict[str, tuple[float, float]]] = {
    "EMA": {"period": (2, 200)},
    "SMA": {"period": (2, 200)},
    "HMA": {"period": (2, 200)},
    "DEMA": {"period": (2, 200)},
    "WMA": {"period": (2, 200)},
    "RSI": {"period": (2, 100)},
    "STOCH": {"period_k": (5, 50), "period_d": (3, 20)},
    "ROC": {"period": (2, 100)},
    "CCI": {"period": (5, 100)},
    "CMO": {"period": (2, 100)},
    "BB": {"period": (5, 100), "k": (0.5, 4.0)},
    "ATR": {"period": (2, 100)},
    "DC": {"period": (5, 100)},
    "KC": {"period": (5, 100), "k": (0.5, 4.0)},
    "VHF": {"period": (5, 100)},
    "MACD": {"fast_period": (5, 50), "slow_period": (10, 100)},
    "ADX": {"period": (5, 100)},
    "AROON": {"period": (5, 100)},
}

# Threshold ranges for entry rules by output scale
THRESHOLD_RANGES: dict[IndicatorOutputScale, tuple[float, float]] = {
    IndicatorOutputScale.NORMALIZED: (0.25, 0.75),
    IndicatorOutputScale.PERCENTAGE: (10.0, 90.0),
    IndicatorOutputScale.CENTERED: (-80.0, 80.0),
    IndicatorOutputScale.UNBOUNDED: (-100.0, 100.0),
    # PRICE scale should not use thresholds (use crossover instead)
}

# Mutation noise scale per output scale
THRESHOLD_MUTATION_SCALE: dict[IndicatorOutputScale, float] = {
    IndicatorOutputScale.NORMALIZED: 0.05,
    IndicatorOutputScale.PERCENTAGE: 5.0,
    IndicatorOutputScale.CENTERED: 10.0,
    IndicatorOutputScale.UNBOUNDED: 10.0,
}


class Condition(str, Enum):
    """Conditions for entry/exit rules."""
    CROSSES_ABOVE = "crosses_above"
    CROSSES_BELOW = "crosses_below"
    GREATER_THAN = "greater_than"
    LESS_THAN = "less_than"
    # For crossover between two indicators
    CROSSOVER_ABOVE = "crossover_above"  # ind_a crosses above ind_b
    CROSSOVER_BELOW = "crossover_below"  # ind_a crosses below ind_b


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"


# Band output selection for BB, DC, KC
class BandOutput(str, Enum):
    UPPER = "upper"
    MIDDLE = "middle"
    LOWER = "lower"


@dataclass
class IndicatorGene:
    """A single indicator configuration."""
    type: IndicatorType
    params: dict[str, float]
    id: str = ""  # auto-generated identifier for referencing in rules
    band: BandOutput | None = None  # for BB, DC, KC: which band to output

    def __post_init__(self):
        if not self.id:
            # Use uuid suffix to guarantee uniqueness
            suffix = uuid.uuid4().hex[:4]
            period = int(self.params.get("period", self.params.get("period_k", self.params.get("fast_period", 0))))
            self.id = f"{self.type.value}_{period}_{suffix}"

    @property
    def output_scale(self) -> IndicatorOutputScale:
        return INDICATOR_SCALE[self.type.value]

    def to_dict(self) -> dict:
        d = {"type": self.type.value, "params": self.params, "id": self.id}
        if self.band is not None:
            d["band"] = self.band.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> IndicatorGene:
        band = BandOutput(d["band"]) if d.get("band") else None
        return cls(
            type=IndicatorType(d["type"]),
            params=d["params"],
            id=d.get("id", ""),
            band=band,
        )


@dataclass
class EntryRule:
    """A condition that triggers a trade entry."""
    indicator_id: str               # references IndicatorGene.id
    condition: Condition
    value: float                    # threshold value (or 0.0 for crossover)
    side: Side
    compare_indicator_id: str = ""  # for crossover conditions

    def to_dict(self) -> dict:
        return {
            "indicator_id": self.indicator_id,
            "condition": self.condition.value,
            "value": self.value,
            "side": self.side.value,
            "compare_indicator_id": self.compare_indicator_id,
        }

    @classmethod
    def from_dict(cls, d: dict) -> EntryRule:
        return cls(
            indicator_id=d["indicator_id"],
            condition=Condition(d["condition"]),
            value=d["value"],
            side=Side(d["side"]),
            compare_indicator_id=d.get("compare_indicator_id", ""),
        )


@dataclass
class ExitRule:
    """A condition that triggers position exit."""
    type: str  # "take_profit", "stop_loss", "time_limit", "indicator"
    value: float  # percentage for TP/SL, seconds for time, threshold for indicator
    indicator_id: str = ""  # for indicator-based exits
    condition: Condition | None = None

    def to_dict(self) -> dict:
        d: dict[str, Any] = {"type": self.type, "value": self.value}
        if self.indicator_id:
            d["indicator_id"] = self.indicator_id
        if self.condition is not None:
            d["condition"] = self.condition.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> ExitRule:
        return cls(
            type=d["type"],
            value=d["value"],
            indicator_id=d.get("indicator_id", ""),
            condition=Condition(d["condition"]) if d.get("condition") else None,
        )


@dataclass
class FilterRule:
    """A condition that gates ALL trading — must be true for any entry to occur."""
    indicator_id: str
    condition: Condition  # typically GREATER_THAN or LESS_THAN
    value: float

    def to_dict(self) -> dict:
        return {
            "indicator_id": self.indicator_id,
            "condition": self.condition.value,
            "value": self.value,
        }

    @classmethod
    def from_dict(cls, d: dict) -> FilterRule:
        return cls(
            indicator_id=d["indicator_id"],
            condition=Condition(d["condition"]),
            value=d["value"],
        )


@dataclass
class Genome:
    """
    Complete strategy genome.

    Encodes all parameters needed to instantiate a NautilusTrader Strategy.
    Serializable to/from JSON for persistence.
    """
    # Identity
    genome_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    generation: int = 0
    parent_ids: list[str] = field(default_factory=list)

    # Asset scope
    symbols: list[str] = field(default_factory=lambda: ["BTCUSDT"])

    # Bar period for indicator computation (minutes)
    bar_period_minutes: int = 1

    # Indicators
    indicators: list[IndicatorGene] = field(default_factory=list)

    # Entry rules
    entry_rules: list[EntryRule] = field(default_factory=list)

    # How many entry rules must match to trigger (1 = OR / any, len = AND / all)
    min_entry_matches: int = 1

    # Exit rules (ANY rule triggering = exit signal)
    exit_rules: list[ExitRule] = field(default_factory=list)

    # Filter rules — ALL must be true for trading to be enabled
    filter_rules: list[FilterRule] = field(default_factory=list)

    # Position sizing
    position_size_usd: float = 100.0     # USD per trade
    max_open_positions: int = 1           # concurrent positions per asset

    # Risk
    max_drawdown_pct: float = 10.0       # kill switch

    # Trade management
    cooldown_bars: int = 0               # minimum bars between trades (0 = disabled)
    direction_bias: str = "both"         # "both", "long_only", "short_only"

    def to_dict(self) -> dict:
        return {
            "genome_id": self.genome_id,
            "generation": self.generation,
            "parent_ids": self.parent_ids,
            "symbols": self.symbols,
            "bar_period_minutes": self.bar_period_minutes,
            "indicators": [ind.to_dict() for ind in self.indicators],
            "entry_rules": [rule.to_dict() for rule in self.entry_rules],
            "min_entry_matches": self.min_entry_matches,
            "exit_rules": [rule.to_dict() for rule in self.exit_rules],
            "filter_rules": [rule.to_dict() for rule in self.filter_rules],
            "position_size_usd": self.position_size_usd,
            "max_open_positions": self.max_open_positions,
            "max_drawdown_pct": self.max_drawdown_pct,
            "cooldown_bars": self.cooldown_bars,
            "direction_bias": self.direction_bias,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Genome:
        return cls(
            genome_id=d.get("genome_id", uuid.uuid4().hex[:12]),
            generation=d.get("generation", 0),
            parent_ids=list(d.get("parent_ids", [])),
            symbols=list(d.get("symbols", ["BTCUSDT"])),
            bar_period_minutes=d.get("bar_period_minutes", 1),
            indicators=[IndicatorGene.from_dict(i) for i in d.get("indicators", [])],
            entry_rules=[EntryRule.from_dict(r) for r in d.get("entry_rules", [])],
            min_entry_matches=d.get("min_entry_matches", 1),
            exit_rules=[ExitRule.from_dict(r) for r in d.get("exit_rules", [])],
            filter_rules=[FilterRule.from_dict(r) for r in d.get("filter_rules", [])],
            position_size_usd=d.get("position_size_usd", 100.0),
            max_open_positions=d.get("max_open_positions", 1),
            max_drawdown_pct=d.get("max_drawdown_pct", 10.0),
            cooldown_bars=d.get("cooldown_bars", 0),
            direction_bias=d.get("direction_bias", "both"),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, s: str) -> Genome:
        return cls.from_dict(json.loads(s))

    def save(self, path: Path) -> None:
        path.write_text(self.to_json())

    @classmethod
    def load(cls, path: Path) -> Genome:
        return cls.from_json(path.read_text())

    @property
    def description(self) -> str:
        """Human-readable summary."""
        parts = [f"Genome {self.genome_id} (gen {self.generation})"]
        parts.append(f"  Assets: {', '.join(self.symbols)}")
        parts.append(f"  Bar: {self.bar_period_minutes}m")
        parts.append(f"  Indicators: {len(self.indicators)}")
        for ind in self.indicators:
            params_str = ", ".join(f"{k}={v}" for k, v in ind.params.items())
            band_str = f" [{ind.band.value}]" if ind.band else ""
            parts.append(f"    {ind.id}: {ind.type.value}({params_str}){band_str}")
        trigger = f"any" if self.min_entry_matches <= 1 else f">={self.min_entry_matches}"
        parts.append(f"  Entry rules: {len(self.entry_rules)} (trigger: {trigger})")
        for rule in self.entry_rules:
            parts.append(f"    {rule.indicator_id} {rule.condition.value} {rule.value} -> {rule.side.value}")
        parts.append(f"  Exit rules: {len(self.exit_rules)}")
        for rule in self.exit_rules:
            parts.append(f"    {rule.type}: {rule.value}")
        if self.filter_rules:
            parts.append(f"  Filter rules: {len(self.filter_rules)}")
            for rule in self.filter_rules:
                parts.append(f"    {rule.indicator_id} {rule.condition.value} {rule.value}")
        parts.append(f"  Size: ${self.position_size_usd}, max positions: {self.max_open_positions}")
        if self.cooldown_bars > 0:
            parts.append(f"  Cooldown: {self.cooldown_bars} bars")
        if self.direction_bias != "both":
            parts.append(f"  Direction: {self.direction_bias}")
        return "\n".join(parts)
