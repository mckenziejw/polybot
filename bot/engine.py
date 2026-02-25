"""
Trading engine — wraps py-clob-client for order execution and redemption.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, date
from pathlib import Path
from typing import Optional

import requests
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import (
    ApiCreds,
    MarketOrderArgs,
    OrderArgs,
    OrderType,
    BalanceAllowanceParams,
    BookParams,
)
from py_clob_client.constants import POLYGON

logger = logging.getLogger(__name__)

GAMMA_API = "https://gamma-api.polymarket.com"


@dataclass
class Position:
    """Tracks an open position in a market."""
    slug: str
    condition_id: str
    token_id: str
    signal: str
    entry_price: float
    shares: float
    cost: float
    order_id: str
    timestamp: float
    up_token_id: str
    down_token_id: str


@dataclass
class TradeRecord:
    """Completed trade record for logging."""
    slug: str
    signal: str
    token_id: str
    entry_price: float
    shares: float
    cost: float
    outcome: float
    payout: float
    profit: float
    timestamp: str


@dataclass
class DailyStats:
    """Tracks daily P&L and trade counts."""
    date: str = ""
    trades: int = 0
    wins: int = 0
    losses: int = 0
    pnl: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0

    def reset(self, new_date: str):
        self.date = new_date
        self.trades = 0
        self.wins = 0
        self.losses = 0
        self.pnl = 0.0
        self.gross_profit = 0.0
        self.gross_loss = 0.0


class TradingEngine:
    """
    Manages order execution, position tracking, balance, and redemption.
    Supports dry-run mode for paper trading.
    """

    def __init__(self, polymarket_config: dict, trading_config: dict):
        self.dry_run = trading_config.get("dry_run", True)
        self.position_size = trading_config.get("position_size", 10.0)
        self.autoscale_enabled = trading_config.get("autoscale_enabled", False)
        self.autoscale_fraction = trading_config.get("autoscale_fraction", 0.05)
        self.max_daily_loss = trading_config.get("max_daily_loss", 200.0)

        # State — supports multiple concurrent positions keyed by slug
        self.positions: dict[str, Position] = {}
        self.daily_stats = DailyStats()
        self.total_pnl: float = 0.0
        self.trade_history: list[TradeRecord] = []
        self._capital: float = 0.0  # will be fetched on start
        import threading
        self._positions_lock = threading.Lock()

        # Persistence — separate state files for dry-run vs live
        mode_suffix = "dry" if self.dry_run else "live"
        self.state_path = Path(f"data/bot_state_{mode_suffix}.json")

        # Initialize CLOB client
        host = polymarket_config["host"]
        chain_id = polymarket_config.get("chain_id", POLYGON)
        private_key = polymarket_config["private_key"]

        creds = ApiCreds(
            api_key=polymarket_config["api_key"],
            api_secret=polymarket_config["api_secret"],
            api_passphrase=polymarket_config["api_passphrase"],
        )

        self.client = ClobClient(
            host,
            key=private_key,
            chain_id=chain_id,
            creds=creds,
            signature_type=2,  # POLY_GNOSIS_SAFE for proxy wallet
            funder=polymarket_config.get("proxy_wallet"),
        )

        # In live mode, derive fresh API keys to ensure auth works
        if not self.dry_run:
            try:
                # Suppress noisy httpx logs during key derivation
                logging.getLogger("httpx").setLevel(logging.WARNING)
                fresh_creds = self.client.create_or_derive_api_creds()
                logging.getLogger("httpx").setLevel(logging.INFO)
                self.client.set_api_creds(fresh_creds)
                logger.info("API credentials derived successfully")
            except Exception as e:
                logging.getLogger("httpx").setLevel(logging.INFO)
                logger.error(f"Failed to derive API credentials: {e}")
                logger.error("Live trading may not work — check your private key and proxy wallet")

        logger.info(f"Trading engine initialized (dry_run={self.dry_run})")

    # ------------------------------------------------------------------
    # Balance & Capital
    # ------------------------------------------------------------------

    def fetch_balance(self) -> float:
        """Fetch current USDC balance from Polymarket."""
        if self.dry_run:
            if self._capital <= 0:
                self._capital = 1000.0  # default paper trading capital
            return self._capital
        try:
            result = self.client.get_balance_allowance(
                BalanceAllowanceParams(asset_type="COLLATERAL")
            )
            balance = float(result.get("balance", 0)) / 1e6  # USDC has 6 decimals
            self._capital = balance
            return balance
        except Exception as e:
            logger.error(f"Failed to fetch balance: {e}")
            return self._capital

    def get_position_size(self) -> float:
        """Calculate position size, optionally auto-scaling."""
        if self.autoscale_enabled:
            balance = self.fetch_balance()
            size = balance * self.autoscale_fraction
            return max(1.0, min(size, balance * 0.25))  # cap at 25% of balance
        return self.position_size

    # ------------------------------------------------------------------
    # Daily Loss Management
    # ------------------------------------------------------------------

    def check_daily_reset(self):
        """Reset daily stats if it's a new day."""
        today = date.today().isoformat()
        if self.daily_stats.date != today:
            if self.daily_stats.date:
                logger.info(f"Daily reset: {self.daily_stats.date} -> {today}")
            self.daily_stats.reset(today)

    def is_daily_loss_exceeded(self) -> bool:
        """Check if daily loss limit has been hit."""
        return self.daily_stats.pnl <= -self.max_daily_loss

    # ------------------------------------------------------------------
    # Order Execution
    # ------------------------------------------------------------------

    def get_order_book(self, token_id: str) -> dict:
        """Fetch the current order book for a token."""
        try:
            book = self.client.get_order_book(token_id)
            return book
        except Exception as e:
            logger.error(f"Failed to fetch order book: {e}")
            return None

    def get_best_ask(self, token_id: str) -> Optional[float]:
        """Get the best ask price (lowest ask) for a token."""
        try:
            book = self.client.get_order_book(token_id)
            if book and book.asks:
                # asks may not be sorted — find the minimum (best) ask
                asks_prices = [float(a.price) for a in book.asks]
                best_ask = min(asks_prices)
                return best_ask
        except Exception as e:
            logger.error(f"Failed to get best ask for {token_id[:20]}...: {e}")
        return None

    def place_market_buy(self, token_id: str, amount: float,
                         price: float = 0) -> Optional[str]:
        """
        Place a limit buy order at the given price for the given dollar amount.
        Uses create_order (limit) instead of create_market_order to avoid
        rounding issues in the py-clob-client's market order builder.
        Returns order_id on success, None on failure.
        """
        if self.dry_run:
            order_id = f"dry-{int(time.time()*1000)}"
            logger.info(f"[DRY RUN] Market buy ${amount:.2f} of {token_id[:20]}... -> {order_id}")
            return order_id

        try:
            # If no price given, fetch the best ask
            if price <= 0:
                price = self.get_best_ask(token_id)
                if not price or price <= 0:
                    logger.error("Cannot determine ask price for order")
                    return None

            # Round price to 2 decimal places (tick_size=0.01)
            price = round(price, 2)

            # Calculate shares: floor(amount / price) rounded to 2 decimals
            import math
            size = math.floor((amount / price) * 100) / 100  # floor to 2 dp

            if size < 0.01:
                logger.warning(f"Calculated size {size} too small")
                return None

            logger.info(f"Placing limit buy: {size:.2f} shares @ ${price:.2f} "
                        f"(${size * price:.2f})")

            # Create and post a limit order
            order_args = OrderArgs(
                token_id=token_id,
                price=price,
                size=size,
                side="BUY",
            )
            signed_order = self.client.create_order(order_args)
            if signed_order:
                result = self.client.post_order(signed_order, OrderType.GTC)
                order_id = result.get("orderID", "unknown")
                logger.info(f"Limit buy placed: {size:.2f} @ ${price:.2f} -> "
                            f"order {order_id}")
                return order_id
        except Exception as e:
            logger.error(f"Market buy failed: {e}")
        return None

    # ------------------------------------------------------------------
    # Position Management
    # ------------------------------------------------------------------

    def open_position(self, slug: str, condition_id: str, signal: str,
                      token_id: str, entry_price: float, shares: float,
                      cost: float, order_id: str,
                      up_token_id: str, down_token_id: str):
        """Record an opened position."""
        pos = Position(
            slug=slug,
            condition_id=condition_id,
            token_id=token_id,
            signal=signal,
            entry_price=entry_price,
            shares=shares,
            cost=cost,
            order_id=order_id,
            timestamp=time.time(),
            up_token_id=up_token_id,
            down_token_id=down_token_id,
        )
        with self._positions_lock:
            self.positions[slug] = pos
        if self.dry_run:
            self._capital -= cost
        logger.info(f"Position opened: {slug} ({len(self.positions)} open)")
        self.save_state()

    def get_position(self, slug: str) -> Optional[Position]:
        """Get an open position by slug."""
        return self.positions.get(slug)

    def has_position(self, slug: str) -> bool:
        """Check if there's an open position for a slug."""
        return slug in self.positions

    def close_position(self, slug: str, outcome: float) -> TradeRecord:
        """
        Close a specific position given the resolution outcome.
        Returns the completed TradeRecord.
        """
        with self._positions_lock:
            pos = self.positions.pop(slug, None)
        if not pos:
            raise ValueError(f"No open position for {slug}")

        payout = pos.shares * outcome
        profit = payout - pos.cost

        record = TradeRecord(
            slug=pos.slug,
            signal=pos.signal,
            token_id=pos.token_id,
            entry_price=pos.entry_price,
            shares=pos.shares,
            cost=pos.cost,
            outcome=outcome,
            payout=payout,
            profit=profit,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        # Update stats
        self.daily_stats.trades += 1
        self.daily_stats.pnl += profit
        self.total_pnl += profit

        if profit > 0:
            self.daily_stats.wins += 1
            self.daily_stats.gross_profit += profit
        else:
            self.daily_stats.losses += 1
            self.daily_stats.gross_loss += abs(profit)

        if self.dry_run:
            self._capital += payout

        self.trade_history.append(record)
        logger.info(f"Position closed: {slug} P&L=${profit:+.2f} "
                     f"({len(self.positions)} remaining)")
        self.save_state()
        return record

    # ------------------------------------------------------------------
    # Contract Redemption
    # ------------------------------------------------------------------

    def redeem_positions(self, condition_id: str) -> bool:
        """
        Redeem resolved conditional tokens to free up USDC.
        After a market resolves, winning tokens can be redeemed for $1 each.
        """
        if self.dry_run:
            logger.info(f"[DRY RUN] Redeeming positions for condition {condition_id[:20]}...")
            return True

        try:
            # The py-clob-client doesn't expose a direct redeem method.
            # Redemption goes through the CTF Exchange contract.
            # We use the Polymarket merge/redeem endpoint via the CLOB API.
            #
            # For the CTFExchange, we need to call redeemPositions(conditionId, amounts)
            # This is done through the neg_risk exchange or regular exchange.
            #
            # Approach: Use the CLOB /trade endpoint or call contract directly.
            # For now, attempt via the CLOB API's built-in mechanism.

            host = self.client.host
            headers = self.client.create_or_derive_api_creds()

            # Try the merge-redemption approach
            neg_risk = False
            try:
                # Check if this is a neg_risk market — check any token
                # in an open position or use condition_id
                for pos in self.positions.values():
                    neg_risk = self.client.get_neg_risk(pos.token_id)
                    break
            except Exception:
                pass

            exchange_addr = self.client.get_exchange_address(neg_risk=neg_risk)
            conditional_addr = self.client.get_conditional_address()

            logger.info(f"Redemption: exchange={exchange_addr}, cond={conditional_addr}")
            logger.info(f"Redemption for {condition_id} — manual redemption may be needed")

            # NOTE: Full on-chain redemption requires web3 interaction with the
            # CTFExchange contract's redeemPositions() function.
            # This is a placeholder — the actual implementation depends on
            # whether you use web3.py directly or a higher-level wrapper.
            #
            # For BTC 5-min markets, Polymarket typically auto-redeems after
            # resolution, but manual redemption ensures faster capital turnover.

            return True

        except Exception as e:
            logger.error(f"Redemption failed for {condition_id}: {e}")
            return False

    # ------------------------------------------------------------------
    # State Persistence
    # ------------------------------------------------------------------

    def save_state(self):
        """Save bot state to disk for crash recovery."""
        positions_data = {}
        with self._positions_lock:
            for slug, p in self.positions.items():
                positions_data[slug] = {
                    "slug": p.slug,
                    "condition_id": p.condition_id,
                    "token_id": p.token_id,
                    "signal": p.signal,
                    "entry_price": p.entry_price,
                    "shares": p.shares,
                    "cost": p.cost,
                    "order_id": p.order_id,
                    "timestamp": p.timestamp,
                    "up_token_id": p.up_token_id,
                    "down_token_id": p.down_token_id,
                }

        state = {
            "total_pnl": self.total_pnl,
            "capital": self._capital,
            "daily_stats": {
                "date": self.daily_stats.date,
                "trades": self.daily_stats.trades,
                "wins": self.daily_stats.wins,
                "losses": self.daily_stats.losses,
                "pnl": self.daily_stats.pnl,
            },
            "positions": positions_data,
            "trade_count": len(self.trade_history),
        }
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.state_path, "w") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    def load_state(self):
        """Load bot state from disk."""
        if not self.state_path.exists():
            return
        try:
            with open(self.state_path) as f:
                state = json.load(f)
            self.total_pnl = state.get("total_pnl", 0.0)
            if self.dry_run:
                self._capital = state.get("capital", 1000.0)

            ds = state.get("daily_stats", {})
            self.daily_stats.date = ds.get("date", "")
            self.daily_stats.trades = ds.get("trades", 0)
            self.daily_stats.wins = ds.get("wins", 0)
            self.daily_stats.losses = ds.get("losses", 0)
            self.daily_stats.pnl = ds.get("pnl", 0.0)

            # Load multiple positions
            positions_data = state.get("positions", {})
            for slug, pos_data in positions_data.items():
                self.positions[slug] = Position(**pos_data)
                logger.info(f"Restored position: {slug}")

            # Backward compat: load old single-position format
            if not positions_data:
                pos_data = state.get("current_position")
                if pos_data:
                    self.positions[pos_data["slug"]] = Position(**pos_data)
                    logger.info(f"Restored position (legacy): {pos_data['slug']}")

            logger.info(f"State loaded: total_pnl=${self.total_pnl:.2f}, "
                        f"capital=${self._capital:.2f}, "
                        f"open_positions={len(self.positions)}")
        except Exception as e:
            logger.error(f"Failed to load state: {e}")

    # ------------------------------------------------------------------
    # Resolution Check
    # ------------------------------------------------------------------

    def check_resolution(self, slug: str) -> Optional[dict]:
        """
        Check if a market has resolved via Gamma API.
        Returns resolution dict or None if unresolved.

        BTC 5-min markets don't use UMA resolution — they resolve via
        Chainlink price feeds. We detect resolution by checking if
        outcomePrices has reached [0, 1] or [1, 0], or if the market
        is marked as closed.
        """
        try:
            resp = requests.get(
                f"{GAMMA_API}/markets/slug/{slug}",
                timeout=10,
            )
            resp.raise_for_status()
            result = resp.json()

            clob_token_ids = json.loads(result["clobTokenIds"])
            outcome_prices = json.loads(result["outcomePrices"])
            prices = [float(p) for p in outcome_prices]

            # Check if resolved: one price is 1.0 and the other is 0.0
            # Also accept near-resolution (>0.95) for markets that haven't
            # fully settled in the Gamma API yet
            is_resolved = (
                result.get("umaResolutionStatus") == "resolved"
                or result.get("closed") is True
                or ("1" in outcome_prices and "0" in outcome_prices)
                or max(prices) >= 0.99
            )

            if not is_resolved:
                return None

            # If prices are extreme but not exactly 0/1, snap to 0/1
            if max(prices) >= 0.99:
                winning_idx = prices.index(max(prices))
                prices = [0.0, 0.0]
                prices[winning_idx] = 1.0

            return {
                "token_outcomes": {
                    clob_token_ids[0]: prices[0],
                    clob_token_ids[1]: prices[1],
                },
                "winning_token": clob_token_ids[prices.index(1.0)],
            }
        except Exception as e:
            logger.warning(f"Resolution check failed for {slug}: {e}")
            return None
