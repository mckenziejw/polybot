"""NautilusTrader strategy for Polymarket 5-minute BTC UpDown sniper.

Subscribes to Binance BTCUSDT 1-minute bars, accumulates a rolling buffer,
and at each new 5-minute Polymarket market window:
  1. Computes features from the candle buffer
  2. Runs XGBoost ensemble inference
  3. If confident, buys the predicted outcome token at the ask

Holds to binary settlement ($1 or $0).
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from nautilus_trader.config import StrategyConfig
from nautilus_trader.model.data import Bar, BarType, QuoteTick
from nautilus_trader.model.enums import OrderSide, TimeInForce
from nautilus_trader.model.events import OrderFilled
from nautilus_trader.model.identifiers import InstrumentId, Venue
from nautilus_trader.model.instruments import BinaryOption
from nautilus_trader.model.objects import Quantity
from nautilus_trader.trading.strategy import Strategy

from .features import assign_windows, compute_features, get_feature_columns

logger = logging.getLogger(__name__)

POLYMARKET_VENUE = Venue("POLYMARKET")


class SniperStrategyConfig(StrategyConfig, frozen=True):
    """Configuration for the 5-minute sniper strategy."""

    # Binance bar subscription
    binance_bar_type: str = "BTCUSDT-PERP.BINANCE-1-MINUTE-LAST-INTERNAL"

    # Model
    model_dir: str = "data/polymarket_5m_models_v2"
    model_window: int = 31  # walk-forward window index (latest trained)
    n_seeds: int = 5

    # Trading
    confidence_threshold: float = 0.55
    max_entry_price: float = 0.60
    max_spread: float = 0.04
    bet_dollars: float = 10.0
    daily_loss_limit: float = 50.0

    # Buffer
    candle_buffer_size: int = 300  # ~5 hours of 1m candles (plenty for all lookback features)
    min_warmup_candles: int = 60  # minimum candles before trading

    # Dry run: log predictions without placing orders
    dry_run: bool = True


@dataclass
class _CandleRow:
    """Lightweight struct for a single 1-minute candle."""
    timestamp_ns: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    trades: int


class SniperStrategy(Strategy):
    """Minute-0 sniper for Polymarket BTC UpDown 5-minute markets."""

    def __init__(self, config: SniperStrategyConfig) -> None:
        super().__init__(config)
        self._config = config
        self._bar_type = BarType.from_str(config.binance_bar_type)

        # Candle buffer (deque with max size)
        self._candles: deque[_CandleRow] = deque(maxlen=config.candle_buffer_size)

        # XGBoost models (loaded on start)
        self._models: list[xgb.Booster] = []
        self._feature_cols: list[str] | None = None

        # Track the last traded window (by aligned timestamp ms) — only need to
        # prevent double-trading the current window, so a single int suffices.
        self._last_traded_window_ms: int = -1

        # Track active Polymarket instruments we've subscribed to
        self._subscribed_pm_instruments: set[InstrumentId] = set()

        # PnL tracking for daily loss limit
        self._daily_pnl: float = 0.0
        self._daily_pnl_reset_day: int = -1  # monotonic UTC day counter

        # Track cost basis per fill for PnL calculation on settlement
        self._open_positions: dict[InstrumentId, float] = {}  # inst_id -> total cost

        # Stats
        self._predictions_made: int = 0
        self._orders_placed: int = 0
        self._fills: int = 0

    def on_start(self) -> None:
        # Load XGBoost models
        model_dir = Path(self._config.model_dir)
        for seed_idx in range(self._config.n_seeds):
            path = model_dir / f"window_{self._config.model_window}_seed_{seed_idx}.json"
            booster = xgb.Booster()
            booster.load_model(str(path))
            self._models.append(booster)
        self.log.info(f"Loaded {len(self._models)} XGBoost models from {model_dir}")

        # Subscribe to Binance 1-minute bars
        self.subscribe_bars(self._bar_type)
        self.log.info(f"Subscribed to {self._bar_type}")

        if self._config.dry_run:
            self.log.warning("DRY RUN MODE — predictions will be logged but no orders placed")

    def on_bar(self, bar: Bar) -> None:
        # Accumulate candle into buffer
        self._candles.append(_CandleRow(
            timestamp_ns=bar.ts_event,
            open=float(bar.open),
            high=float(bar.high),
            low=float(bar.low),
            close=float(bar.close),
            volume=float(bar.volume),
            trades=0,  # not available from bar aggregation; features handle this gracefully
        ))

        n = len(self._candles)
        if n <= self._config.min_warmup_candles or n % 10 == 0:
            self.log.info(f"Bar #{n}: close={float(bar.close):.2f} (warmup={n}/{self._config.min_warmup_candles})")

        # Pre-subscribe to Polymarket quotes for upcoming windows so quotes
        # are already cached when the boundary bar arrives.
        self._presubscribe_upcoming_instruments()

        # Only predict on bars that close on a 5-minute boundary.
        # In CryptoLake training data, the bar closing at :X0/:X5 is minute 0
        # of that window. This is the moment the Polymarket market opens —
        # the earliest we can enter.
        #
        # NautilusTrader's internal bar timer has sub-second jitter, so round
        # to the nearest minute before checking alignment.
        bar_ts_s = bar.ts_event // 1_000_000_000
        bar_minute_s = (bar_ts_s + 30) // 60 * 60  # round to nearest minute
        if bar_minute_s % 300 == 0:
            self._check_and_trade(window_start_s=bar_minute_s)

    def _presubscribe_upcoming_instruments(self) -> None:
        """Subscribe to quote ticks for all Polymarket instruments in cache.

        Called on every bar so that by the time a 5-min boundary arrives,
        we already have cached quotes for the instruments we'll trade.
        """
        all_pm = self.cache.instruments(venue=POLYMARKET_VENUE)
        for inst in all_pm:
            if not isinstance(inst, BinaryOption):
                continue
            if inst.id not in self._subscribed_pm_instruments:
                self.subscribe_quote_ticks(inst.id)
                self._subscribed_pm_instruments.add(inst.id)
                self.log.info(f"Pre-subscribed quotes: {inst.id}")

    def _check_and_trade(self, window_start_s: int) -> None:
        """Run sniper logic for a specific 5-minute window."""
        if len(self._candles) < self._config.min_warmup_candles:
            self.log.info(f"Boundary bar at {window_start_s} but still warming up ({len(self._candles)}/{self._config.min_warmup_candles})")
            return

        window_start_ms = window_start_s * 1000

        # Reset daily PnL counter at UTC day boundary
        now_ns = self.clock.timestamp_ns()
        utc_day = now_ns // 86_400_000_000_000  # monotonic UTC day number
        if utc_day != self._daily_pnl_reset_day:
            if self._daily_pnl_reset_day >= 0:
                self.log.info(f"Daily PnL reset (was ${self._daily_pnl:.2f})")
            self._daily_pnl = 0.0
            self._daily_pnl_reset_day = utc_day

        # Check daily loss limit
        if self._daily_pnl <= -self._config.daily_loss_limit:
            return

        # Only trade once per window
        if window_start_ms == self._last_traded_window_ms:
            return

        self.log.info(f"5-min boundary: window_start_s={window_start_s}, looking for instruments...")

        # Find active Polymarket instruments for this window
        pm_instruments = self._find_polymarket_instruments(window_start_s)
        if not pm_instruments:
            # Log what we DO have for debugging
            all_pm = self.cache.instruments(venue=POLYMARKET_VENUE)
            self.log.warning(
                f"No instruments found for window {window_start_s}. "
                f"Cache has {len(all_pm)} POLYMARKET instruments."
            )
            if all_pm:
                sample = all_pm[0]
                info = sample.info or {}
                self.log.warning(
                    f"  Sample instrument: raw_symbol={sample.raw_symbol}, "
                    f"outcome={sample.outcome}, "
                    f"market_slug={info.get('market_slug', 'N/A')}, "
                    f"question={info.get('question', 'N/A')[:80]}"
                )
            return

        # Subscribe to any new instruments for quote data
        for inst in pm_instruments:
            if inst.id not in self._subscribed_pm_instruments:
                self.subscribe_quote_ticks(inst.id)
                self._subscribed_pm_instruments.add(inst.id)

        # Run prediction
        prediction = self._predict()
        if prediction is None:
            return

        prob_up, prob_std = prediction
        self._predictions_made += 1

        # Determine direction
        buy_up = prob_up > 0.5
        confidence = prob_up if buy_up else (1.0 - prob_up)

        # Find the Up and Down token instruments
        up_inst = None
        down_inst = None
        for inst in pm_instruments:
            outcome = inst.outcome.lower() if inst.outcome else ""
            if outcome in ("yes", "up"):
                up_inst = inst
            elif outcome in ("no", "down"):
                down_inst = inst

        target_inst = up_inst if buy_up else down_inst
        if target_inst is None:
            self.log.warning(f"No {'Up' if buy_up else 'Down'} token found for window {window_start_s}")
            return

        # Get current ask price from quote cache
        quote = self.cache.quote_tick(target_inst.id)
        if quote is None:
            self.log.debug(f"No quote yet for {target_inst.id}")
            return

        ask_price = float(quote.ask_price)
        bid_price = float(quote.bid_price)

        if ask_price <= 0:
            self.log.warning(f"Invalid ask price {ask_price} for {target_inst.id}")
            return

        spread = abs(ask_price - bid_price)

        direction = "UP" if buy_up else "DOWN"
        self.log.info(
            f"Window {window_start_s} | pred={prob_up:.4f} ± {prob_std:.4f} "
            f"| dir={direction} conf={confidence:.4f} "
            f"| ask={ask_price:.3f} spread={spread:.4f}"
        )

        # Apply filters
        if confidence < self._config.confidence_threshold:
            self.log.info(f"  SKIP: confidence {confidence:.4f} < {self._config.confidence_threshold}")
            self._last_traded_window_ms = window_start_ms
            return

        if ask_price > self._config.max_entry_price:
            self.log.info(f"  SKIP: ask {ask_price:.3f} > max {self._config.max_entry_price}")
            self._last_traded_window_ms = window_start_ms
            return

        if spread > self._config.max_spread:
            self.log.info(f"  SKIP: spread {spread:.4f} > max {self._config.max_spread}")
            self._last_traded_window_ms = window_start_ms
            return

        # Mark window as traded
        self._last_traded_window_ms = window_start_ms

        if self._config.dry_run:
            self.log.info(f"  DRY RUN: would buy {direction} at {ask_price:.3f}")
            return

        # Calculate order size
        tokens = self._config.bet_dollars / ask_price
        quantity = target_inst.make_qty(tokens)

        # Place limit order at ask (effectively a market buy on binary option)
        order = self.order_factory.limit(
            instrument_id=target_inst.id,
            order_side=OrderSide.BUY,
            quantity=quantity,
            price=target_inst.make_price(ask_price),
            time_in_force=TimeInForce.GTC,
        )
        self.submit_order(order)
        self._orders_placed += 1
        self.log.info(
            f"  ORDER: BUY {tokens:.1f} {direction} tokens at {ask_price:.3f} "
            f"(${self._config.bet_dollars:.0f})"
        )

    def _find_polymarket_instruments(self, window_start_s: int) -> list[BinaryOption]:
        """Find Polymarket BinaryOption instruments matching this 5-min window."""
        all_instruments = self.cache.instruments(venue=POLYMARKET_VENUE)
        matching = []
        slug_fragment = f"btc-updown-5m-{window_start_s}"
        for inst in all_instruments:
            if not isinstance(inst, BinaryOption):
                continue
            # Match by slug in the instrument's info dict or raw_symbol
            inst_str = str(inst.raw_symbol).lower()
            info = inst.info or {}
            slug = info.get("slug", "") or info.get("market_slug", "") or ""
            if slug_fragment in inst_str or slug_fragment in slug.lower():
                matching.append(inst)
        return matching

    def _predict(self) -> tuple[float, float] | None:
        """Run XGBoost ensemble on the current candle buffer.

        Returns (probability_up, std) or None if features can't be computed.
        """
        # Build DataFrame from candle buffer
        rows = []
        for c in self._candles:
            rows.append({
                "timestamp": c.timestamp_ns,
                "open": c.open,
                "high": c.high,
                "low": c.low,
                "close": c.close,
                "volume": c.volume,
                "trades": c.trades,
            })
        df = pd.DataFrame(rows)
        df["timestamp_ms"] = (df["timestamp"] // 1_000_000).astype("int64")

        # Assign windows and compute features
        df = assign_windows(df)
        features = compute_features(df)

        if len(features) == 0:
            return None

        # Get feature columns (first time: cache them)
        if self._feature_cols is None:
            self._feature_cols = get_feature_columns(features)

        # Use the last row (most recent candle) for prediction
        last_row = features.iloc[-1:]
        X = last_row[self._feature_cols].values.astype(np.float32)
        X[~np.isfinite(X)] = 0.0

        # Ensemble prediction
        dmat = xgb.DMatrix(X, feature_names=self._feature_cols)
        preds = np.array([m.predict(dmat)[0] for m in self._models])
        return float(preds.mean()), float(preds.std())

    def on_order_filled(self, event: OrderFilled) -> None:
        self._fills += 1
        price = float(event.last_px)
        qty = float(event.last_qty)
        cost = price * qty

        if event.order_side == OrderSide.BUY:
            # Opening position: track cost basis, debit daily PnL
            self._open_positions[event.instrument_id] = (
                self._open_positions.get(event.instrument_id, 0.0) + cost
            )
            self._daily_pnl -= cost
        else:
            # Closing/settlement: credit proceeds
            self._daily_pnl += cost

        self.log.info(
            f"FILLED: {event.order_side} {qty:.1f} @ {price:.4f} "
            f"(instrument: {event.instrument_id}) daily_pnl=${self._daily_pnl:.2f}"
        )

    def on_quote_tick(self, tick: QuoteTick) -> None:
        # Quotes are consumed by the cache; no action needed here.
        pass

    def on_stop(self) -> None:
        self.log.info(
            f"Sniper stats: predictions={self._predictions_made}, "
            f"orders={self._orders_placed}, fills={self._fills}, "
            f"daily_pnl=${self._daily_pnl:.2f}"
        )

    def on_reset(self) -> None:
        self._candles.clear()
        self._last_traded_window_ms = -1
        self._subscribed_pm_instruments.clear()
        self._open_positions.clear()
        self._daily_pnl = 0.0
        self._predictions_made = 0
        self._orders_placed = 0
        self._fills = 0
