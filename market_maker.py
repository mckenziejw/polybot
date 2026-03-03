"""
market_maker.py  (event-driven rewrite)

Architecture
------------
Three threads total, no polling loops:

  main thread     — market rotation via threading.Timer, posts first pair
                    on open, then idles waiting for stop_event
  websocket       — user channel delivers fill/cancel events, triggers
                    all order logic via on_order_update callback
  market channel  — separate websocket for mid-price (read-only)

No monitor thread. Fill logic lives entirely in on_order_update:

  MATCHED/CONFIRMED → mark order filled
  if first fill  → start a threading.Timer for cancel_window_s
  if second fill → cancel the timer, finalize as "both"
  timer fires    → cancel the unfilled leg, finalize as one-sided

threading.Timer is one-shot and self-contained — it holds no locks
and does minimal work. The only shared mutable state is the active
OrderPair, protected by a single RLock that is never held across
a sleep or network call.

Config (config.json)
--------------------
{
  "polymarket": {
    "private_key":    "0x...",
    "api_key":        "...",
    "api_secret":     "...",
    "api_passphrase": "...",
    "proxy_wallet":   "0x..."
  },
  "market_maker": {
    "yes_price":       0.48,
    "no_price":        0.48,
    "order_size":      10.0,
    "cancel_window_s": 6.0,
    "min_remaining_s": 30.0
  }
}
"""

import json
import logging
import signal
import sys
import threading
import time
from dataclasses import dataclass, field

from execution.market_info import MarketInfo, fetch_current_market, fetch_next_market
from execution.market_channel import MarketChannel
from execution.metrics import MetricsWriter
from execution.order_client import Order, OrderClient, OrderPair, OrderStatus, Side
from execution.user_channel import UserChannel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path: str = "config.json") -> dict:
    with open(path) as f:
        return json.load(f)


@dataclass
class MMConfig:
    order_size:      float = 10.0
    cancel_window_s: float = 6.0
    min_remaining_s: float = 30.0
    max_positions:   int   = 1      # max open filled contracts per side
    price_offset:    float = 0.0    # shift applied to best_bid / best_ask

    def __post_init__(self):
        assert self.order_size > 0
        assert self.max_positions >= 1
        assert -0.05 <= self.price_offset <= 0.05


# ---------------------------------------------------------------------------
# Session tracking
# ---------------------------------------------------------------------------

@dataclass
class MarketSession:
    market:          MarketInfo
    pairs:           list = field(default_factory=list)
    both_filled:     int   = 0
    yes_filled_only: int   = 0
    no_filled_only:  int   = 0
    neither_filled:  int   = 0
    total_spent:     float = 0.0

    def log_summary(self):
        total = (self.both_filled + self.yes_filled_only +
                 self.no_filled_only + self.neither_filled)
        logger.info(
            f"Session [{self.market.slug}] pairs={total} "
            f"both={self.both_filled} yes_only={self.yes_filled_only} "
            f"no_only={self.no_filled_only} neither={self.neither_filled} "
            f"spent=${self.total_spent:.2f}"
        )


# ---------------------------------------------------------------------------
# Market maker
# ---------------------------------------------------------------------------

class MarketMaker:
    """
    Event-driven two-sided market maker.

    All order state transitions happen in on_order_update(), which is
    called directly from the user channel websocket thread. No polling.
    """

    def __init__(self, cfg: MMConfig, order_client: OrderClient):
        self.cfg    = cfg
        self.orders = order_client

        # Set externally after construction
        self.user_channel:   UserChannel   | None = None
        self.market_channel: MarketChannel | None = None
        self.metrics:        MetricsWriter | None = None

        self._current_session: MarketSession | None = None
        self._active_pair:     OrderPair    | None = None
        self._cancel_timer:    threading.Timer | None = None

        # Open position counters — filled contracts not yet resolved
        self._yes_positions: int = 0
        self._no_positions:  int = 0

        # order_id → Order for fast lookup in on_order_update
        self._order_index: dict[str, Order] = {}

        # RLock: allows _finalize_pair to be called from within a locked section
        self._lock = threading.RLock()

        self._stop_event = threading.Event()

    # ------------------------------------------------------------------
    # Startup / shutdown
    # ------------------------------------------------------------------

    def start(self):
        self.user_channel.start()

        market = fetch_current_market()
        if self.market_channel:
            self.market_channel.start(market)

        self._open_market(market)

        # Main thread just waits — all work happens in callbacks
        self._stop_event.wait()

    def stop(self):
        logger.info("Stopping market maker")
        self._stop_event.set()
        self._cancel_timer_if_running()

        with self._lock:
            if self._active_pair:
                pair = self._active_pair
                if pair.yes_order and pair.yes_order.status == OrderStatus.OPEN:
                    self.orders.cancel_order(pair.yes_order)
                if pair.no_order and pair.no_order.status == OrderStatus.OPEN:
                    self.orders.cancel_order(pair.no_order)

        self.orders.cancel_all()

        if self.user_channel:
            self.user_channel.stop()
        if self.market_channel:
            self.market_channel.stop()
        if self.metrics:
            self.metrics.stop()
        if self._current_session:
            self._current_session.log_summary()

    # ------------------------------------------------------------------
    # Market lifecycle
    # ------------------------------------------------------------------

    def _open_market(self, market: MarketInfo, initial: bool = False):
        logger.info(f"Market open: {market}")
        with self._lock:
            self._current_session = MarketSession(market=market)
            self._active_pair     = None
            self._order_index.clear()
            self._yes_positions   = 0
            self._no_positions    = 0

        if self.user_channel:
            self.user_channel.subscribe([market.condition_id])
        if self.market_channel and not initial:
            self.market_channel.rotate(market)

        # Schedule pair posting with a short delay to skip open noise
        threading.Timer(2.0, self._post_pair).start()

        # Schedule market close check
        remaining = market.seconds_remaining()
        close_in  = max(0, remaining - self.cfg.min_remaining_s)
        threading.Timer(close_in, self._on_market_expiring).start()

    def _on_market_expiring(self):
        """Called when the market is close to expiry — cancel open orders and rotate."""
        if self._stop_event.is_set():
            return

        logger.info("Market expiring — canceling active pair and rotating")
        self._cancel_timer_if_running()

        with self._lock:
            pair = self._active_pair
            if pair:
                if pair.yes_order and pair.yes_order.status == OrderStatus.OPEN:
                    self.orders.cancel_order(pair.yes_order)
                if pair.no_order and pair.no_order.status == OrderStatus.OPEN:
                    self.orders.cancel_order(pair.no_order)
                self._finalize_pair(pair, self._pair_outcome(pair))

        self.orders.cancel_all()
        if self._current_session:
            self._current_session.log_summary()

        try:
            next_market = fetch_next_market(self._current_session.market)
            self._open_market(next_market)
        except RuntimeError as e:
            logger.error(f"Could not fetch next market: {e}")
            self.stop()

    # ------------------------------------------------------------------
    # Order posting
    # ------------------------------------------------------------------

    def _post_pair(self):
        """Post a two-sided YES+NO buy pair. Called from a Timer thread."""
        if self._stop_event.is_set():
            return

        with self._lock:
            if self._active_pair is not None:
                return  # already have a live pair
            session = self._current_session
            if session is None:
                return
            if session.market.seconds_remaining() < self.cfg.min_remaining_s:
                logger.info("Too close to expiry — skipping pair post")
                return
            # Position limit check: don't post if either side is maxed out
            if (self._yes_positions >= self.cfg.max_positions or
                    self._no_positions >= self.cfg.max_positions):
                logger.info(
                    f"Position limit reached (yes={self._yes_positions} "
                    f"no={self._no_positions} max={self.cfg.max_positions}) "
                    f"-- waiting for resolution before posting"
                )
                return

        market = session.market

        # Derive prices from live book — block until market channel is ready
        yes_price, no_price = self._get_book_prices()
        if yes_price is None:
            logger.error("Could not get book prices after waiting — skipping pair")
            threading.Timer(5.0, self._post_pair).start()
            return

        mid_now = self.market_channel.mid_price if self.market_channel else None
        logger.info(
            f"Posting pair: YES@{yes_price} NO@{no_price} "
            f"size={self.cfg.order_size} mid={mid_now} market={market.slug}"
        )

        yes_order = self.orders.post_limit_order(
            token_id=market.up_token_id,
            side=Side.BUY,
            price=yes_price,
            size=self.cfg.order_size,
        )
        no_order = self.orders.post_limit_order(
            token_id=market.down_token_id,
            side=Side.BUY,
            price=no_price,
            size=self.cfg.order_size,
        )

        if yes_order is None or no_order is None:
            logger.error("Failed to post one or both legs — canceling and retrying")
            if yes_order:
                self.orders.cancel_order(yes_order)
            if no_order:
                self.orders.cancel_order(no_order)
            # Retry after a short delay
            threading.Timer(5.0, self._post_pair).start()
            return

        pair = OrderPair(yes_order=yes_order, no_order=no_order)
        pair.mid_at_placement = mid_now

        with self._lock:
            self._active_pair = pair
            self._order_index[yes_order.order_id] = yes_order
            self._order_index[no_order.order_id]  = no_order

        logger.info(
            f"Pair live: YES={yes_order.order_id[:16]}... "
            f"NO={no_order.order_id[:16]}..."
        )

    # ------------------------------------------------------------------
    # Fill event handler — called from websocket thread
    # ------------------------------------------------------------------

    def on_order_update(self, event: dict):
        event_type = event.get("event_type", "")

        if event_type == "trade":
            self._handle_trade_event(event)
        elif event_type == "order":
            self._handle_order_event(event)

    def _handle_trade_event(self, event: dict):
        status = event.get("status", "")
        if status in ("RETRYING", "FAILED"):
            return

        taker_id  = event.get("taker_order_id")
        maker_ids = [
            m.get("order_id")
            for m in event.get("maker_orders", [])
            if m.get("order_id")
        ]

        filled_orders = []
        with self._lock:
            for oid in ([taker_id] + maker_ids):
                if not oid:
                    continue
                order = self._order_index.get(oid)
                if order and order.status == OrderStatus.OPEN:
                    order.status    = OrderStatus.FILLED
                    order.filled_at = time.time()
                    filled_orders.append(order)
                    logger.info(
                        f"FILL ({status}): {order.side.value} {order.size} "
                        f"@ {order.price} id={oid[:16]}..."
                    )

        for order in filled_orders:
            self._on_fill(order)

    def _handle_order_event(self, event: dict):
        order_id   = event.get("id")
        order_type = event.get("type", "")
        if not order_id:
            return

        with self._lock:
            order = self._order_index.get(order_id)
            if order is None:
                return

            if order_type == "CANCELLATION":
                order.status       = OrderStatus.CANCELLED
                order.cancelled_at = time.time()
                logger.info(f"CANCELLATION confirmed: id={order_id[:16]}...")

            elif order_type == "UPDATE":
                size_matched = float(event.get("size_matched", 0))
                if size_matched >= order.size:
                    order.status    = OrderStatus.FILLED
                    order.filled_at = time.time()
                    logger.info(
                        f"UPDATE fully filled: {order.side.value} "
                        f"{order.size} @ {order.price} id={order_id[:16]}..."
                    )
                    # Trigger fill logic outside lock
                    threading.Timer(0, self._on_fill, args=(order,)).start()
                else:
                    order.fill_size = size_matched
                    logger.debug(f"UPDATE partial: {size_matched}/{order.size}")

    def _on_fill(self, filled_order: Order):
        """
        Called after a fill is recorded. Decides what to do next.
        Runs in websocket thread (or a zero-delay Timer for UPDATE events).
        Must not hold self._lock on entry.
        """
        with self._lock:
            pair = self._active_pair
            if pair is None:
                return

            yes_filled = pair.yes_order.status == OrderStatus.FILLED
            no_filled  = pair.no_order.status  == OrderStatus.FILLED

        if yes_filled and no_filled:
            # Both sides filled — cancel any pending cancel timer
            self._cancel_timer_if_running()
            self._finalize_pair(pair, "both")
            self._schedule_next_pair()
            return

        # First fill — start the cancel window timer
        # Guard: only start if no timer is already running
        with self._lock:
            if self._cancel_timer is not None:
                return  # timer already set from a duplicate event

        which = "YES" if filled_order.token_id == pair.yes_order.token_id else "NO"
        logger.info(
            f"{which} filled first — starting {self.cfg.cancel_window_s}s cancel window"
        )

        timer = threading.Timer(
            self.cfg.cancel_window_s,
            self._on_cancel_window_expired,
            args=(pair,),
        )
        with self._lock:
            self._cancel_timer = timer
        timer.start()

    def _on_cancel_window_expired(self, pair: OrderPair):
        """
        Cancel window elapsed after first fill. Cancel the unfilled leg
        and finalize.
        """
        if self._stop_event.is_set():
            return

        with self._lock:
            self._cancel_timer = None

            # Re-check — second fill may have arrived during the window
            yes_filled = pair.yes_order.status == OrderStatus.FILLED
            no_filled  = pair.no_order.status  == OrderStatus.FILLED

            if yes_filled and no_filled:
                # Both filled while we were waiting — nothing to cancel
                pass
            elif yes_filled:
                logger.info("Cancel window expired — canceling NO")
                self.orders.cancel_order(pair.no_order)
            elif no_filled:
                logger.info("Cancel window expired — canceling YES")
                self.orders.cancel_order(pair.yes_order)
            else:
                # Neither filled (shouldn't happen — we started the timer on first fill)
                logger.warning("Cancel window expired but neither side shows filled")
                self.orders.cancel_order(pair.yes_order)
                self.orders.cancel_order(pair.no_order)

        outcome = self._pair_outcome(pair)
        self._finalize_pair(pair, outcome)
        self._schedule_next_pair()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _pair_outcome(self, pair: OrderPair) -> str:
        yes = pair.yes_order.status == OrderStatus.FILLED
        no  = pair.no_order.status  == OrderStatus.FILLED
        if yes and no:
            return "both"
        if yes:
            return "yes_only"
        if no:
            return "no_only"
        return "neither"

    def _finalize_pair(self, pair: OrderPair, outcome: str):
        """Record outcome, update session tallies, clear active pair."""
        with self._lock:
            if self._active_pair is not pair:
                return  # already finalized (duplicate event guard)
            self._active_pair = None

        session = self._current_session
        if session is None:
            return

        mid_now = self.market_channel.mid_price if self.market_channel else None

        if self.metrics:
            self.metrics.record_pair(
                pair=pair,
                outcome=outcome,
                market_slug=session.market.slug,
                condition_id=session.market.condition_id,
                mid_at_placement=getattr(pair, "mid_at_placement", None),
                mid_at_resolution=mid_now,
            )

        if outcome == "both":
            session.both_filled += 1
            session.total_spent += pair.net_cost
            with self._lock:
                self._yes_positions += 1
                self._no_positions  += 1
            logger.info(
                f"BOTH FILLED — cost=${pair.net_cost:.4f} "
                f"YES@{pair.yes_order.price} NO@{pair.no_order.price} mid={mid_now} "
                f"(open: yes={self._yes_positions} no={self._no_positions})"
            )
        elif outcome == "yes_only":
            session.yes_filled_only += 1
            session.total_spent += pair.yes_order.price * pair.yes_order.size
            with self._lock:
                self._yes_positions += 1
            logger.info(
                f"YES ONLY — YES@{pair.yes_order.price} mid={mid_now} "
                f"(open: yes={self._yes_positions} no={self._no_positions})"
            )
        elif outcome == "no_only":
            session.no_filled_only += 1
            session.total_spent += pair.no_order.price * pair.no_order.size
            with self._lock:
                self._no_positions += 1
            logger.info(
                f"NO ONLY — NO@{pair.no_order.price} mid={mid_now} "
                f"(open: yes={self._yes_positions} no={self._no_positions})"
            )
        else:
            session.neither_filled += 1
            logger.info("NEITHER filled")

    def _get_book_prices(self, timeout: float = 10.0) -> tuple:
        """
        Derive YES and NO buy prices from the live order book.

        YES buy price = best_bid (join the bid queue as a maker)
        NO  buy price = 1 - best_ask (complement of YES ask)

        Blocks up to `timeout` seconds waiting for the market channel
        to have valid prices. Returns (None, None) on timeout.

        Price is rounded to 2 decimal places (Polymarket tick = 0.01).
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._stop_event.is_set():
                return None, None
            bid = self.market_channel.best_bid if self.market_channel else None
            ask = self.market_channel.best_ask if self.market_channel else None
            if bid is not None and ask is not None and bid > 0 and ask > 0:
                yes_price = round(bid + self.cfg.price_offset, 2)
                no_price  = round(1.0 - ask + self.cfg.price_offset, 2)
                # Sanity: both prices must be in (0, 1) and not sum to >= 1
                if (0.01 <= yes_price <= 0.99 and
                        0.01 <= no_price <= 0.99 and
                        yes_price + no_price < 1.0):
                    return yes_price, no_price
                else:
                    logger.warning(
                        f"Derived prices invalid: YES={yes_price} NO={no_price} "
                        f"(bid={bid} ask={ask}) -- retrying"
                    )
            time.sleep(0.2)
        return None, None

    def _schedule_next_pair(self):
        """Post next pair after a brief pause, if market still has time."""
        if self._stop_event.is_set():
            return
        # Respect position limits — only schedule if we have room
        with self._lock:
            if (self._yes_positions >= self.cfg.max_positions or
                    self._no_positions >= self.cfg.max_positions):
                logger.info("Position limit reached -- not scheduling next pair")
                return
        threading.Timer(1.0, self._post_pair).start()

    def _cancel_timer_if_running(self):
        with self._lock:
            timer = self._cancel_timer
            self._cancel_timer = None
        if timer:
            timer.cancel()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    config = load_config()
    pm     = config["polymarket"]
    cfg    = MMConfig(**config.get("market_maker", {}))

    logger.info(
        f"Starting market maker"
        f"size={cfg.order_size} cancel_window={cfg.cancel_window_s}s"
    )

    order_client = OrderClient(
        private_key=pm["private_key"],
        api_key=pm["api_key"],
        api_secret=pm["api_secret"],
        api_passphrase=pm["api_passphrase"],
        proxy_wallet=pm.get("proxy_wallet"),
    )

    mm        = MarketMaker(cfg=cfg, order_client=order_client)
    market_ch = MarketChannel()
    metrics   = MetricsWriter(out_dir="data/metrics")

    user_ch = UserChannel(
        private_key=pm["private_key"],
        api_key=pm["api_key"],
        api_secret=pm["api_secret"],
        api_passphrase=pm["api_passphrase"],
        on_order_update=mm.on_order_update,
        condition_ids=[],
    )

    mm.user_channel   = user_ch
    mm.market_channel = market_ch
    mm.metrics        = metrics

    # Price sampler — every 5s, record mid to prices.csv
    def _price_sampler():
        while not mm._stop_event.is_set():
            mm._stop_event.wait(timeout=5)
            mid = market_ch.mid_price
            if mid and mm._current_session:
                metrics.record_price(
                    mid=mid,
                    best_bid=market_ch.best_bid,
                    best_ask=market_ch.best_ask,
                    market_slug=mm._current_session.market.slug,
                )

    threading.Thread(target=_price_sampler, daemon=True, name="price-sampler").start()

    def _handle_signal(sig, frame):
        logger.info(f"Signal {sig} — shutting down")
        mm.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT,  _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    mm.start()


if __name__ == "__main__":
    main()