"""
Momentum strategy — observes book snapshots via WebSocket and generates signals.
"""

import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class BookTick:
    """A single order book observation."""
    asset_id: str
    timestamp: int  # ms
    bid_price: float
    ask_price: float
    mid_price: float


@dataclass
class Signal:
    """Trading signal output."""
    direction: str       # "buy_up" or "buy_down"
    token_to_buy: str    # token_id to buy
    momentum: float      # raw momentum value
    entry_ask: float     # ask price of the token we'd buy
    mid_at_signal: float # mid-price at signal time


class MomentumStrategy:
    """
    Live momentum strategy for BTC 5-min markets.

    Collects book ticks during the observation window, then
    calculates momentum and generates a signal if threshold is met.
    """

    def __init__(self, config: dict):
        self.observation_window = config.get("observation_window", 0.50)
        self.momentum_threshold = config.get("momentum_threshold", 0.03)
        self.min_entry_price = config.get("min_entry_price", 0.0)
        self.max_entry_price = config.get("max_entry_price", 0.84)
        self.min_book_snapshots = config.get("min_book_snapshots", 20)

        # Per-market tick storage
        self._ticks: dict[str, list[BookTick]] = defaultdict(list)
        self._market_start: dict[str, int] = {}  # slug -> start_ts_ms
        self._market_end: dict[str, int] = {}    # slug -> end_ts_ms
        self._signal_generated: dict[str, bool] = {}

    def reset_market(self, slug: str, start_ts_ms: int, end_ts_ms: int,
                     up_token_id: str, down_token_id: str):
        """Prepare for a new market window."""
        self._ticks[slug] = []
        self._market_start[slug] = start_ts_ms
        self._market_end[slug] = end_ts_ms
        self._signal_generated[slug] = False
        logger.info(f"Strategy reset for {slug}, window "
                    f"{end_ts_ms - start_ts_ms}ms, "
                    f"observe first {self.observation_window:.0%}")

    def on_book_tick(self, slug: str, asset_id: str, timestamp: int,
                     bid_price: float, ask_price: float):
        """Process an incoming book snapshot tick."""
        if slug not in self._market_start:
            return

        mid = (bid_price + ask_price) / 2
        tick = BookTick(
            asset_id=asset_id,
            timestamp=timestamp,
            bid_price=bid_price,
            ask_price=ask_price,
            mid_price=mid,
        )
        self._ticks[slug].append(tick)

    def check_signal(self, slug: str, up_token_id: str,
                     down_token_id: str) -> Optional[Signal]:
        """
        Check if the observation window is complete and a signal should fire.
        Returns Signal or None.

        Called periodically (e.g. after each tick or on a timer).
        """
        if self._signal_generated.get(slug, False):
            return None

        start_ts = self._market_start.get(slug)
        end_ts = self._market_end.get(slug)
        if not start_ts or not end_ts:
            return None

        # Calculate observation cutoff time
        window_duration = end_ts - start_ts
        cutoff_ts = start_ts + int(window_duration * self.observation_window)

        now_ms = int(time.time() * 1000)
        if now_ms < cutoff_ts:
            return None  # still in observation period

        # Mark signal as generated (only fire once per market)
        self._signal_generated[slug] = True

        # Filter ticks for token A (up_token) during observation window
        ticks_a = [
            t for t in self._ticks[slug]
            if t.asset_id == up_token_id and t.timestamp <= cutoff_ts
        ]
        ticks_b = [
            t for t in self._ticks[slug]
            if t.asset_id == down_token_id and t.timestamp <= cutoff_ts
        ]

        if len(ticks_a) < self.min_book_snapshots:
            logger.info(f"[{slug}] Not enough ticks for token A: {len(ticks_a)}")
            return None

        # Sort by timestamp
        ticks_a.sort(key=lambda t: t.timestamp)
        ticks_b.sort(key=lambda t: t.timestamp)

        # Calculate momentum on token A
        start_mid = ticks_a[0].mid_price
        end_mid = ticks_a[-1].mid_price
        momentum = end_mid - start_mid

        # Check threshold
        if abs(momentum) < self.momentum_threshold:
            logger.info(f"[{slug}] No signal: momentum={momentum:.4f} "
                        f"(threshold={self.momentum_threshold})")
            return None

        # Determine direction
        if momentum > 0:
            # Token A pumped -> buy token A
            token_to_buy = up_token_id
            direction = "buy_up"
            entry_ask = ticks_a[-1].ask_price
        else:
            # Token A dumped -> buy token B
            token_to_buy = down_token_id
            direction = "buy_down"
            entry_ask = ticks_b[-1].ask_price if ticks_b else None

        if entry_ask is None or entry_ask <= 0:
            logger.warning(f"[{slug}] No valid ask price for entry")
            return None

        # Check entry price bounds
        if entry_ask < self.min_entry_price or entry_ask > self.max_entry_price:
            logger.info(f"[{slug}] Entry price {entry_ask:.4f} outside bounds "
                        f"[{self.min_entry_price}, {self.max_entry_price}]")
            return None

        # Entry price must be strictly between 0 and 1
        if entry_ask >= 1.0:
            logger.info(f"[{slug}] Entry ask >= 1.0, skipping")
            return None

        signal = Signal(
            direction=direction,
            token_to_buy=token_to_buy,
            momentum=momentum,
            entry_ask=entry_ask,
            mid_at_signal=end_mid,
        )

        logger.info(f"[{slug}] SIGNAL: {direction} | momentum={momentum:+.4f} | "
                     f"entry_ask={entry_ask:.4f}")
        return signal

    def cleanup_market(self, slug: str):
        """Remove stored data for a completed market."""
        self._ticks.pop(slug, None)
        self._market_start.pop(slug, None)
        self._market_end.pop(slug, None)
        self._signal_generated.pop(slug, None)
