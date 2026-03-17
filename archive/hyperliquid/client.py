"""
Hyperliquid REST API client.

Supports both unauthenticated info queries and authenticated exchange
operations (order placement, cancellation) via EIP-712 signing.

Usage:
    from hyperliquid.client import HyperliquidClient

    # Read-only (no private key needed)
    client = HyperliquidClient()
    meta = client.get_meta()

    # With trading capability
    client = HyperliquidClient(private_key="0x...")
    client.place_order("BTC", is_buy=True, price=60000.0, size=0.01)
"""

import json
import time
from typing import Any

import requests
from eth_account import Account
from eth_account.messages import encode_typed_data

from .config import Config


# ---------------------------------------------------------------------------
# EIP-712 domain and types for Hyperliquid
# ---------------------------------------------------------------------------

def _hl_domain(is_mainnet: bool) -> dict:
    """EIP-712 domain separator for Hyperliquid."""
    return {
        "name": "HyperliquidSignTransaction",
        "version": "1",
        "chainId": 42161 if is_mainnet else 421614,
        "verifyingContract": "0x0000000000000000000000000000000000000000",
    }


# Agent type used for signing actions
AGENT_TYPES = {
    "Agent": [
        {"name": "source", "type": "string"},
        {"name": "connectionId", "type": "bytes32"},
    ]
}


def _action_hash(action: dict, nonce: int, vault_address: str | None) -> bytes:
    """
    Compute the connection ID (action hash) used in the Agent message.

    Hyperliquid hashes the action + nonce + vaultAddress to produce
    a bytes32 connectionId that gets signed.
    """
    import hashlib

    # Serialize action deterministically
    action_str = json.dumps(action, separators=(",", ":"), sort_keys=True)
    payload = f"{action_str}{nonce}"
    if vault_address:
        payload += vault_address
    return hashlib.sha256(payload.encode()).digest()


def _sign_action(
    action: dict,
    nonce: int,
    account: Account,
    is_mainnet: bool,
    vault_address: str | None = None,
) -> dict:
    """
    Sign a Hyperliquid exchange action using EIP-712.

    Returns the full payload ready for POST to /exchange.
    """
    connection_id = _action_hash(action, nonce, vault_address)
    domain = _hl_domain(is_mainnet)

    # The Agent message that actually gets signed
    agent_message = {
        "source": "a" if is_mainnet else "b",
        "connectionId": connection_id,
    }

    signable = encode_typed_data(
        domain_data=domain,
        types=AGENT_TYPES,
        primary_type="Agent",
        data=agent_message,
    )

    signed = account.sign_message(signable)

    return {
        "action": action,
        "nonce": nonce,
        "signature": {
            "r": hex(signed.r),
            "s": hex(signed.s),
            "v": signed.v,
        },
        "vaultAddress": vault_address,
    }


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

REQUEST_TIMEOUT = 15  # seconds
HEADERS = {"Content-Type": "application/json"}


class HyperliquidClient:
    """
    Hyperliquid REST API client.

    Supports info queries (no auth) and exchange operations (requires private key).
    Defaults to testnet.
    """

    def __init__(
        self,
        private_key: str | None = None,
        address: str | None = None,
        testnet: bool = True,
        config: Config | None = None,
    ):
        if config:
            self.config = config
        else:
            self.config = Config(
                private_key=private_key,
                address=address,
                testnet=testnet,
            )

        self._account: Account | None = None
        if self.config.private_key:
            self._account = Account.from_key(self.config.private_key)
            # Derive address from key if not explicitly provided
            if not self.config.address:
                self.config.address = self._account.address

        # Cache asset name -> index mapping
        self._asset_index: dict[str, int] | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _post_info(self, payload: dict) -> Any:
        """POST to /info (no auth required)."""
        resp = requests.post(
            self.config.info_url,
            json=payload,
            headers=HEADERS,
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()

    def _post_exchange(self, action: dict) -> Any:
        """POST to /exchange (auth required, signs with EIP-712)."""
        if not self._account:
            raise RuntimeError(
                "Private key required for exchange operations. "
                "Pass private_key to the constructor."
            )

        nonce = int(time.time() * 1000)
        is_mainnet = not self.config.testnet

        payload = _sign_action(
            action=action,
            nonce=nonce,
            account=self._account,
            is_mainnet=is_mainnet,
        )

        resp = requests.post(
            self.config.exchange_url,
            json=payload,
            headers=HEADERS,
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()

    def _ensure_asset_index(self) -> None:
        """Fetch and cache the asset name -> index mapping from meta."""
        if self._asset_index is not None:
            return
        meta = self.get_meta()
        self._asset_index = {}
        for i, asset in enumerate(meta["universe"]):
            self._asset_index[asset["name"]] = i

    def _get_asset_index(self, coin: str) -> int:
        """Get the integer asset index for a coin name."""
        self._ensure_asset_index()
        assert self._asset_index is not None
        if coin not in self._asset_index:
            raise ValueError(
                f"Unknown coin '{coin}'. "
                f"Available: {sorted(self._asset_index.keys())[:20]}..."
            )
        return self._asset_index[coin]

    @property
    def address(self) -> str:
        """Return the configured wallet address, or raise if not set."""
        if not self.config.address:
            raise RuntimeError(
                "Wallet address required. Pass address= or private_key= "
                "to the constructor."
            )
        return self.config.address

    # ------------------------------------------------------------------
    # Info endpoints (no auth)
    # ------------------------------------------------------------------

    def get_meta(self) -> dict:
        """Get all perpetual asset metadata (universe, fees, etc.)."""
        return self._post_info({"type": "meta"})

    def get_mids(self) -> dict[str, float]:
        """Get all mid prices. Returns {coin: mid_price}."""
        data = self._post_info({"type": "allMids"})
        return {k: float(v) for k, v in data.items()}

    def get_l2(self, coin: str) -> dict:
        """Get L2 orderbook snapshot for a coin."""
        return self._post_info({"type": "l2Book", "coin": coin})

    def get_funding_rates(self) -> tuple[dict, list[dict]]:
        """
        Get current funding rates and asset contexts.

        Returns (meta, asset_ctxs) where each asset_ctx contains
        'funding', 'openInterest', 'markPx', etc.
        """
        data = self._post_info({"type": "metaAndAssetCtxs"})
        return data[0], data[1]

    def get_funding_history(self, coin: str, start_time_ms: int) -> list[dict]:
        """Get historical funding rates for a coin since start_time_ms."""
        return self._post_info({
            "type": "fundingHistory",
            "coin": coin,
            "startTime": start_time_ms,
        })

    def get_positions(self, address: str | None = None) -> dict:
        """Get perpetual clearinghouse state (positions, margin, etc.)."""
        addr = address or self.address
        return self._post_info({"type": "clearinghouseState", "user": addr})

    def get_open_orders(self, address: str | None = None) -> list[dict]:
        """Get open orders for an address."""
        addr = address or self.address
        return self._post_info({"type": "openOrders", "user": addr})

    def get_spot_balances(self, address: str | None = None) -> dict:
        """Get spot clearinghouse state (token balances)."""
        addr = address or self.address
        return self._post_info({"type": "spotClearinghouseState", "user": addr})

    # ------------------------------------------------------------------
    # Exchange endpoints (auth required)
    # ------------------------------------------------------------------

    def place_order(
        self,
        coin: str,
        is_buy: bool,
        price: float,
        size: float,
        order_type: str = "limit",
        tif: str = "Gtc",
        reduce_only: bool = False,
    ) -> dict:
        """
        Place an order.

        Args:
            coin: Asset name (e.g. "BTC", "ETH")
            is_buy: True for buy, False for sell
            price: Limit price
            size: Order size in base asset
            order_type: "limit" (default) or "market"
            tif: Time-in-force — "Gtc", "Ioc", or "Alo" (add liquidity only)
            reduce_only: If True, only reduces existing position

        Returns:
            Exchange API response dict.
        """
        asset_idx = self._get_asset_index(coin)

        if order_type == "market":
            order_type_obj = {"market": {"tif": "Ioc"}}
        else:
            order_type_obj = {"limit": {"tif": tif}}

        action = {
            "type": "order",
            "orders": [
                {
                    "a": asset_idx,
                    "b": is_buy,
                    "p": str(price),
                    "s": str(size),
                    "r": reduce_only,
                    "t": order_type_obj,
                }
            ],
            "grouping": "na",
        }

        return self._post_exchange(action)

    def cancel_order(self, coin: str, order_id: int) -> dict:
        """Cancel a single order by coin and order ID."""
        asset_idx = self._get_asset_index(coin)

        action = {
            "type": "cancel",
            "cancels": [{"a": asset_idx, "o": order_id}],
        }

        return self._post_exchange(action)

    def cancel_all(self, coin: str) -> list[dict]:
        """
        Cancel all open orders for a coin.

        Fetches open orders first, then cancels them in a single batch.
        Returns list of cancel responses (empty if no orders).
        """
        orders = self.get_open_orders()
        to_cancel = [o for o in orders if o.get("coin") == coin]

        if not to_cancel:
            return []

        asset_idx = self._get_asset_index(coin)
        action = {
            "type": "cancel",
            "cancels": [
                {"a": asset_idx, "o": int(o["oid"])} for o in to_cancel
            ],
        }

        return self._post_exchange(action)
