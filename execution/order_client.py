"""
order_client.py

Thin wrapper around py-clob-client for posting and canceling orders.
Handles auth setup, logging, and error normalization so the market
maker logic doesn't deal with CLOB client internals directly.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum, auto

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import ApiCreds, OrderArgs, OrderType

logger = logging.getLogger(__name__)


class Side(Enum):
    BUY  = "BUY"
    SELL = "SELL"


class OrderStatus(Enum):
    OPEN      = auto()
    FILLED    = auto()
    CANCELLED = auto()
    FAILED    = auto()


@dataclass
class Order:
    """Tracks a single resting order and its lifecycle."""
    order_id:  str
    token_id:  str
    side:      Side
    price:     float
    size:      float
    status:    OrderStatus = OrderStatus.OPEN
    fill_size: float       = 0.0
    # Timestamps (unix seconds, float)
    placed_at:   float = 0.0
    filled_at:   float = 0.0
    cancelled_at: float = 0.0


@dataclass
class OrderPair:
    """
    Tracks a two-sided (yes + no) order pair placed simultaneously.
    Both orders are at complementary prices — yes_price + no_price ~ 1.0.
    """
    yes_order: Order | None = None
    no_order:  Order | None = None

    @property
    def both_open(self) -> bool:
        return (
            self.yes_order is not None and self.yes_order.status == OrderStatus.OPEN and
            self.no_order  is not None and self.no_order.status  == OrderStatus.OPEN
        )

    @property
    def both_filled(self) -> bool:
        return (
            self.yes_order is not None and self.yes_order.status == OrderStatus.FILLED and
            self.no_order  is not None and self.no_order.status  == OrderStatus.FILLED
        )

    @property
    def yes_filled_only(self) -> bool:
        return (
            self.yes_order is not None and self.yes_order.status == OrderStatus.FILLED and
            self.no_order  is not None and self.no_order.status  == OrderStatus.OPEN
        )

    @property
    def no_filled_only(self) -> bool:
        return (
            self.no_order  is not None and self.no_order.status  == OrderStatus.FILLED and
            self.yes_order is not None and self.yes_order.status == OrderStatus.OPEN
        )

    @property
    def net_cost(self) -> float:
        """Total spent on filled legs."""
        cost = 0.0
        if self.yes_order and self.yes_order.status == OrderStatus.FILLED:
            cost += self.yes_order.price * self.yes_order.size
        if self.no_order and self.no_order.status == OrderStatus.FILLED:
            cost += self.no_order.price * self.no_order.size
        return cost


class OrderClient:
    """
    Wraps py-clob-client with logging and simplified error handling.
    One instance per execution service — stateless, all state lives in OrderPair/Order objects.
    """

    def __init__(
        self,
        private_key: str,
        api_key: str,
        api_secret: str,
        api_passphrase: str,
        proxy_wallet: str | None = None,
    ):
        self._client = ClobClient(
            host="https://clob.polymarket.com",
            chain_id=137,
            key=private_key,
            creds=ApiCreds(
                api_key=api_key,
                api_secret=api_secret,
                api_passphrase=api_passphrase,
            ),
            signature_type=2,    # POLY_GNOSIS_SAFE — required for proxy wallet
            funder=proxy_wallet, # proxy wallet address that holds USDC
        )
        self._client.set_api_creds(self._client.create_or_derive_api_creds())
        logger.info(f"OrderClient initialized (proxy_wallet={proxy_wallet})")

    def post_limit_order(
        self,
        token_id: str,
        side: Side,
        price: float,
        size: float,
    ) -> Order | None:
        """
        Post a GTC limit order. Returns an Order on success, None on failure.

        Args:
            token_id: the YES or NO token ID for this market
            side:     BUY or SELL
            price:    limit price (0.0 - 1.0)
            size:     number of shares (dollars at this price)
        """
        import time
        try:
            args = OrderArgs(
                token_id=token_id,
                price=price,
                size=size,
                side=side.value,
            )
            resp = self._client.create_and_post_order(args)

            # Response is either {"orderID": "...", ...} or {"errorMsg": "..."}
            if not resp or "errorMsg" in resp:
                logger.error(f"Order rejected: {resp}")
                return None

            order_id = resp.get("orderID") or resp.get("order_id")
            if not order_id:
                logger.error(f"No order ID in response: {resp}")
                return None

            order = Order(
                order_id=order_id,
                token_id=token_id,
                side=side,
                price=price,
                size=size,
                placed_at=time.time(),
            )
            logger.info(
                f"Posted {side.value} {size:.2f} shares @ {price:.4f} "
                f"token={token_id[:16]}... id={order_id[:16]}..."
            )
            return order

        except Exception as e:
            logger.error(f"Failed to post order: {e}")
            return None

    def cancel_order(self, order: Order) -> bool:
        """
        Cancel a resting order. Updates order status in place.
        Returns True on success.
        """
        import time
        try:
            resp = self._client.cancel(order.order_id)
            if resp and resp.get("canceled"):
                order.status       = OrderStatus.CANCELLED
                order.cancelled_at = time.time()
                logger.info(f"Cancelled order {order.order_id[:16]}...")
                return True
            else:
                logger.warning(f"Cancel may have failed: {resp}")
                # Mark cancelled anyway — order is likely already gone
                order.status       = OrderStatus.CANCELLED
                order.cancelled_at = time.time()
                return False
        except Exception as e:
            logger.error(f"Failed to cancel order {order.order_id[:16]}...: {e}")
            return False

    def cancel_all(self) -> bool:
        """Cancel all open orders. Used on market close or shutdown."""
        try:
            resp = self._client.cancel_all()
            logger.info(f"Cancel all: {resp}")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel all orders: {e}")
            return False

    def get_order(self, order_id: str) -> dict | None:
        """Fetch current order state from CLOB — used for reconciliation."""
        try:
            return self._client.get_order(order_id)
        except Exception as e:
            logger.error(f"Failed to get order {order_id[:16]}...: {e}")
            return None