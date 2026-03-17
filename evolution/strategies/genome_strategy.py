"""
GenomeStrategy: translates a Genome into a live NautilusTrader Strategy.

This is the bridge between the GA's parameter space and NautilusTrader's
event-driven strategy interface. A Genome specifies which indicators to
compute, what conditions trigger entries/exits, and position sizing rules.
GenomeStrategy instantiates the indicators, registers them for bar updates,
and evaluates rules on each new bar.
"""

from __future__ import annotations

from nautilus_trader.config import StrategyConfig
from nautilus_trader.indicators.averages import (
    DoubleExponentialMovingAverage,
    ExponentialMovingAverage,
    HullMovingAverage,
    SimpleMovingAverage,
    WeightedMovingAverage,
)
from nautilus_trader.indicators.momentum import (
    ChandeMomentumOscillator,
    CommodityChannelIndex,
    RateOfChange,
    RelativeStrengthIndex,
    Stochastics,
)
from nautilus_trader.indicators.trend import (
    AroonOscillator,
    DirectionalMovement,
    MovingAverageConvergenceDivergence,
)
from nautilus_trader.indicators.volatility import (
    AverageTrueRange,
    BollingerBands,
    DonchianChannel,
    KeltnerChannel,
    VerticalHorizontalFilter,
)
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import OrderSide, TimeInForce
from nautilus_trader.model.events import OrderFilled
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Quantity
from nautilus_trader.trading.strategy import Strategy

from evolution.ga.genome import (
    BandOutput,
    Condition,
    FilterRule,
    Genome,
    IndicatorGene,
    IndicatorType,
    Side,
)


class GenomeStrategyConfig(StrategyConfig, frozen=True):
    """Config for GenomeStrategy — wraps a genome dict."""
    genome_dict: dict
    bar_type_str: str  # e.g., "BTCUSDT.BINANCE-1-MINUTE-MID-INTERNAL"


def _create_indicator(gene: IndicatorGene):
    """Instantiate a NautilusTrader indicator from a genome IndicatorGene."""
    t = gene.type
    p = gene.params

    if t == IndicatorType.EMA:
        return ExponentialMovingAverage(int(p["period"]))
    elif t == IndicatorType.SMA:
        return SimpleMovingAverage(int(p["period"]))
    elif t == IndicatorType.HMA:
        return HullMovingAverage(int(p["period"]))
    elif t == IndicatorType.DEMA:
        return DoubleExponentialMovingAverage(int(p["period"]))
    elif t == IndicatorType.WMA:
        return WeightedMovingAverage(int(p["period"]))
    elif t == IndicatorType.RSI:
        return RelativeStrengthIndex(int(p["period"]))
    elif t == IndicatorType.STOCH:
        return Stochastics(int(p.get("period_k", 14)), int(p.get("period_d", 3)))
    elif t == IndicatorType.ROC:
        return RateOfChange(int(p["period"]))
    elif t == IndicatorType.CCI:
        return CommodityChannelIndex(int(p["period"]))
    elif t == IndicatorType.CMO:
        return ChandeMomentumOscillator(int(p["period"]))
    elif t == IndicatorType.BB:
        return BollingerBands(int(p["period"]), float(p.get("k", 2.0)))
    elif t == IndicatorType.ATR:
        return AverageTrueRange(int(p["period"]))
    elif t == IndicatorType.DC:
        return DonchianChannel(int(p["period"]))
    elif t == IndicatorType.KC:
        return KeltnerChannel(int(p["period"]), float(p.get("k", 2.0)))
    elif t == IndicatorType.VHF:
        return VerticalHorizontalFilter(int(p["period"]))
    elif t == IndicatorType.MACD:
        # NautilusTrader MACD only takes fast_period and slow_period (no signal line)
        return MovingAverageConvergenceDivergence(
            int(p["fast_period"]), int(p["slow_period"])
        )
    elif t == IndicatorType.ADX:
        return DirectionalMovement(int(p["period"]))
    elif t == IndicatorType.AROON:
        return AroonOscillator(int(p["period"]))
    else:
        raise ValueError(f"Unknown indicator type: {t}")


def _get_indicator_value(indicator, band: BandOutput | None = None) -> float | None:
    """Extract a value from an indicator, optionally selecting a band."""
    if not indicator.initialized:
        return None

    # Band selection for multi-output indicators (BB, DC, KC)
    if band is not None:
        if band == BandOutput.UPPER and hasattr(indicator, "upper"):
            return float(indicator.upper)
        elif band == BandOutput.LOWER and hasattr(indicator, "lower"):
            return float(indicator.lower)
        elif band == BandOutput.MIDDLE and hasattr(indicator, "middle"):
            return float(indicator.middle)
        # Fall through to default extraction if band attr not found

    # Most indicators have a .value property
    if hasattr(indicator, "value"):
        return float(indicator.value)

    # Stochastics has .value_k and .value_d
    if hasattr(indicator, "value_k"):
        return float(indicator.value_k)

    # Bollinger Bands / channels have .middle as default
    if hasattr(indicator, "middle"):
        return float(indicator.middle)

    return None


class GenomeStrategy(Strategy):
    """
    A NautilusTrader Strategy driven by a Genome.

    Translates genome indicator/rule specifications into live indicator
    computation and order generation.
    """

    def __init__(self, config: GenomeStrategyConfig) -> None:
        super().__init__(config)
        self.genome = Genome.from_dict(config.genome_dict)
        self.bar_type = BarType.from_str(config.bar_type_str)

        # Will be populated in on_start
        self._indicators: dict[str, object] = {}  # gene_id -> indicator instance
        self._indicator_bands: dict[str, BandOutput | None] = {}  # gene_id -> band
        self._prev_values: dict[str, float | None] = {}  # for crossover detection

        # Position tracking — updated via on_order_filled, not submit_order
        self._position_open = False
        self._position_side: OrderSide | None = None
        self._bars_since_entry = 0
        self._entry_price: float | None = None
        self._entry_quantity: float | None = None
        self._pending_entry = False
        self._pending_exit = False

        # Kill switch — running stats for early termination
        self._killed = False
        self._n_completed_trades = 0
        self._running_pnl = 0.0
        self._peak_balance = 0.0
        self._bar_count = 0
        self._bars_since_last_trade = 999  # high initial value so first trade isn't blocked

        # Trailing/ATR exit state
        self._trailing_stop_price: float | None = None
        self._atr_stop_price: float | None = None
        self._atr_tp_price: float | None = None

    def on_start(self) -> None:
        # Create and register indicators
        for gene in self.genome.indicators:
            indicator = _create_indicator(gene)
            self._indicators[gene.id] = indicator
            self._indicator_bands[gene.id] = gene.band
            self._prev_values[gene.id] = None
            self.register_indicator_for_bars(self.bar_type, indicator)

        # Subscribe to bars
        self.subscribe_bars(self.bar_type)

        self.log.info(
            f"GenomeStrategy started: {self.genome.genome_id} "
            f"({len(self._indicators)} indicators, "
            f"{len(self.genome.entry_rules)} entry rules [{self.genome.min_entry_matches} needed], "
            f"{len(self.genome.exit_rules)} exit rules)"
        )

    def on_bar(self, bar: Bar) -> None:
        # Kill switch — skip all processing once killed
        if self._killed:
            return

        self._bar_count += 1

        # Get current indicator values
        current_values: dict[str, float | None] = {}
        for gene_id, indicator in self._indicators.items():
            band = self._indicator_bands.get(gene_id)
            current_values[gene_id] = _get_indicator_value(indicator, band)

        # Check if all indicators are initialized
        if any(v is None for v in current_values.values()):
            self._prev_values = current_values
            return

        # Kill switch checks (after warmup period)
        if self._bar_count > 100 and self._should_kill():
            self._killed = True
            if self._position_open:
                self._exit_position(bar, "kill_switch")
            return

        self._bars_since_last_trade += 1

        if self._position_open:
            self._bars_since_entry += 1
            if not self._pending_exit:
                self._check_exit_rules(bar, current_values)
        elif not self._pending_entry:
            # Check cooldown
            if self.genome.cooldown_bars > 0 and self._bars_since_last_trade < self.genome.cooldown_bars:
                self._prev_values = current_values
                return
            # Check filter rules (all must pass)
            if self._check_filters(current_values):
                self._check_entry_rules(bar, current_values)

        self._prev_values = current_values

    def _should_kill(self) -> bool:
        """Check if this genome is hopeless and should stop trading."""
        n = self._n_completed_trades
        bars = self._bar_count

        # Degenerate: trading more than once every 3 bars on average
        # (e.g., 1m bars: >480 trades/day, 5m bars: >96/day)
        if bars > 0 and n > max(100, bars // 3):
            return True

        # Enough data to judge: 50+ trades and hemorrhaging money
        if n >= 50:
            avg_pnl = self._running_pnl / n
            # Losing >$1/trade on average with 50+ trades — not recovering
            if avg_pnl < -1.0:
                return True

        # Severe drawdown (>30% of starting balance)
        if n >= 20 and self._peak_balance > 0:
            drawdown = (self._peak_balance - (self._peak_balance + self._running_pnl))
            dd_pct = drawdown / self._peak_balance * 100
            if dd_pct > 30:
                return True

        return False

    def _check_filters(self, current_values: dict[str, float | None]) -> bool:
        """Check if all filter rules pass (AND logic). Returns True if trading is allowed."""
        for rule in self.genome.filter_rules:
            current = current_values.get(rule.indicator_id)
            if current is None:
                return False
            if rule.condition == Condition.GREATER_THAN:
                if current <= rule.value:
                    return False
            elif rule.condition == Condition.LESS_THAN:
                if current >= rule.value:
                    return False
            else:
                return False  # only GT/LT supported for filters
        return True

    def _get_current_atr(self) -> float | None:
        """Get current ATR value from indicators, if an ATR indicator exists."""
        for gene_id, indicator in self._indicators.items():
            if isinstance(indicator, AverageTrueRange) and indicator.initialized:
                return float(indicator.value)
        return None

    def on_order_filled(self, event: OrderFilled) -> None:
        """Update position state when an order is actually filled."""
        if self._pending_entry:
            self._pending_entry = False
            self._position_open = True
            self._position_side = event.order_side
            self._bars_since_entry = 0
            self._entry_price = float(event.last_px)
            self._entry_quantity = float(event.last_qty)
            # Initialize trailing/ATR exit levels
            atr_val = self._get_current_atr()
            for rule in self.genome.exit_rules:
                if rule.type == "trailing_stop":
                    if self._position_side == OrderSide.BUY:
                        self._trailing_stop_price = self._entry_price * (1 - rule.value / 100)
                    else:
                        self._trailing_stop_price = self._entry_price * (1 + rule.value / 100)
                elif rule.type == "atr_stop_loss" and atr_val is not None:
                    offset = atr_val * rule.value
                    if self._position_side == OrderSide.BUY:
                        self._atr_stop_price = self._entry_price - offset
                    else:
                        self._atr_stop_price = self._entry_price + offset
                elif rule.type == "atr_take_profit" and atr_val is not None:
                    offset = atr_val * rule.value
                    if self._position_side == OrderSide.BUY:
                        self._atr_tp_price = self._entry_price + offset
                    else:
                        self._atr_tp_price = self._entry_price - offset

            self.log.info(
                f"ENTRY FILLED {event.order_side.name} "
                f"qty={event.last_qty} @ {event.last_px} "
                f"(genome={self.genome.genome_id})"
            )
        elif self._pending_exit:
            self._pending_exit = False
            price = float(event.last_px)
            pnl_pct = 0.0
            pnl_usd = 0.0
            if self._entry_price and self._entry_price > 0 and self._entry_quantity:
                if self._position_side == OrderSide.BUY:
                    pnl_pct = ((price - self._entry_price) / self._entry_price) * 100
                    pnl_usd = (price - self._entry_price) * self._entry_quantity
                else:
                    pnl_pct = ((self._entry_price - price) / self._entry_price) * 100
                    pnl_usd = (self._entry_price - price) * self._entry_quantity
            # Track running stats for kill switch
            self._n_completed_trades += 1
            self._running_pnl += pnl_usd
            balance = self.genome.position_size_usd * self.genome.max_open_positions + self._running_pnl
            if balance > self._peak_balance:
                self._peak_balance = balance
            self.log.info(
                f"EXIT FILLED @ {price:.2f}, PnL: {pnl_pct:+.2f}% "
                f"(genome={self.genome.genome_id}, bars={self._bars_since_entry})"
            )
            self._position_open = False
            self._position_side = None
            self._bars_since_entry = 0
            self._entry_price = None
            self._entry_quantity = None
            self._trailing_stop_price = None
            self._atr_stop_price = None
            self._atr_tp_price = None
            self._bars_since_last_trade = 0

    def _check_condition(
        self,
        condition: Condition,
        current: float,
        prev: float | None,
        value: float,
        current_values: dict[str, float | None],
        compare_indicator_id: str = "",
    ) -> bool:
        """Evaluate a single condition."""
        if condition == Condition.GREATER_THAN:
            return current > value
        elif condition == Condition.LESS_THAN:
            return current < value
        elif condition == Condition.CROSSES_ABOVE:
            return prev is not None and prev <= value and current > value
        elif condition == Condition.CROSSES_BELOW:
            return prev is not None and prev >= value and current < value
        elif condition in (Condition.CROSSOVER_ABOVE, Condition.CROSSOVER_BELOW):
            compare_current = current_values.get(compare_indicator_id)
            compare_prev = self._prev_values.get(compare_indicator_id)
            if compare_current is not None and compare_prev is not None and prev is not None:
                if condition == Condition.CROSSOVER_ABOVE:
                    return prev <= compare_prev and current > compare_current
                else:
                    return prev >= compare_prev and current < compare_current
        return False

    def _check_entry_rules(
        self,
        bar: Bar,
        current_values: dict[str, float | None],
    ) -> None:
        """Evaluate entry rules — trigger when >= min_entry_matches rules match on the same side."""
        # Count matches per side to avoid conflicting BUY+SELL triggers
        side_matches: dict[Side, int] = {Side.BUY: 0, Side.SELL: 0}

        for rule in self.genome.entry_rules:
            # Direction bias filtering
            if self.genome.direction_bias == "long_only" and rule.side == Side.SELL:
                continue
            if self.genome.direction_bias == "short_only" and rule.side == Side.BUY:
                continue

            if rule.indicator_id not in current_values:
                continue

            current = current_values[rule.indicator_id]
            prev = self._prev_values.get(rule.indicator_id)

            if current is None:
                continue

            if self._check_condition(
                rule.condition, current, prev, rule.value,
                current_values, rule.compare_indicator_id,
            ):
                side_matches[rule.side] += 1
                if side_matches[rule.side] >= self.genome.min_entry_matches:
                    self._enter_position(bar, rule.side)
                    return

    def _check_exit_rules(
        self,
        bar: Bar,
        current_values: dict[str, float | None],
    ) -> None:
        """Evaluate exit rules — any matching rule closes the position."""
        if self._entry_price is None:
            return

        current_price = float(bar.close)
        pnl_pct = 0.0
        if self._position_side == OrderSide.BUY:
            pnl_pct = ((current_price - self._entry_price) / self._entry_price) * 100
        elif self._position_side == OrderSide.SELL:
            pnl_pct = ((self._entry_price - current_price) / self._entry_price) * 100

        for rule in self.genome.exit_rules:
            triggered = False

            if rule.type == "take_profit":
                triggered = pnl_pct >= rule.value
            elif rule.type == "stop_loss":
                triggered = pnl_pct <= -rule.value
            elif rule.type == "time_limit":
                # Convert seconds to bars based on bar period
                bar_seconds = self.genome.bar_period_minutes * 60
                max_bars = max(1, int(rule.value / bar_seconds))
                triggered = self._bars_since_entry >= max_bars
            elif rule.type == "indicator":
                if rule.indicator_id in current_values and rule.condition is not None:
                    current = current_values[rule.indicator_id]
                    prev = self._prev_values.get(rule.indicator_id)
                    if current is not None:
                        triggered = self._check_condition(
                            rule.condition, current, prev, rule.value,
                            current_values,
                        )
            elif rule.type == "trailing_stop":
                if self._trailing_stop_price is not None:
                    # Update trailing price (tighten only, never loosen)
                    if self._position_side == OrderSide.BUY:
                        new_stop = current_price * (1 - rule.value / 100)
                        if new_stop > self._trailing_stop_price:
                            self._trailing_stop_price = new_stop
                        triggered = current_price <= self._trailing_stop_price
                    else:
                        new_stop = current_price * (1 + rule.value / 100)
                        if new_stop < self._trailing_stop_price:
                            self._trailing_stop_price = new_stop
                        triggered = current_price >= self._trailing_stop_price
            elif rule.type == "atr_stop_loss":
                if self._atr_stop_price is not None:
                    if self._position_side == OrderSide.BUY:
                        triggered = current_price <= self._atr_stop_price
                    else:
                        triggered = current_price >= self._atr_stop_price
            elif rule.type == "atr_take_profit":
                if self._atr_tp_price is not None:
                    if self._position_side == OrderSide.BUY:
                        triggered = current_price >= self._atr_tp_price
                    else:
                        triggered = current_price <= self._atr_tp_price

            if triggered:
                self._exit_position(bar, rule.type)
                return

    def _enter_position(self, bar: Bar, side: Side) -> None:
        """Place entry order."""
        instrument = self.cache.instrument(self.bar_type.instrument_id)
        if instrument is None:
            return

        order_side = OrderSide.BUY if side == Side.BUY else OrderSide.SELL
        price = float(bar.close)

        # Calculate quantity from USD size
        qty = self.genome.position_size_usd / price
        min_qty = float(instrument.min_quantity) if instrument.min_quantity else 0.0
        qty = max(qty, min_qty)

        order = self.order_factory.market(
            instrument_id=self.bar_type.instrument_id,
            order_side=order_side,
            quantity=Quantity(qty, precision=instrument.size_precision),
            time_in_force=TimeInForce.IOC,
        )

        self._pending_entry = True
        self.submit_order(order)

    def _exit_position(self, bar: Bar, reason: str) -> None:
        """Close current position."""
        if not self._position_open:
            return

        instrument = self.cache.instrument(self.bar_type.instrument_id)
        if instrument is None:
            return

        # Use tracked entry quantity for exact exit
        exit_side = OrderSide.SELL if self._position_side == OrderSide.BUY else OrderSide.BUY
        qty = self._entry_quantity
        if qty is None:
            price = float(bar.close)
            qty = self.genome.position_size_usd / (self._entry_price or price)

        min_qty = float(instrument.min_quantity) if instrument.min_quantity else 0.0
        qty = max(qty, min_qty)

        order = self.order_factory.market(
            instrument_id=self.bar_type.instrument_id,
            order_side=exit_side,
            quantity=Quantity(qty, precision=instrument.size_precision),
            time_in_force=TimeInForce.IOC,
            reduce_only=True,
        )

        self._pending_exit = True
        self.submit_order(order)

    def on_stop(self) -> None:
        # Close any open position
        if self._position_open:
            self.log.warning(f"Strategy stopping with open position (genome={self.genome.genome_id})")
            self.cancel_all_orders(self.bar_type.instrument_id)
            self.close_all_positions(self.bar_type.instrument_id)
            self._position_open = False
            self._pending_entry = False
            self._pending_exit = False
