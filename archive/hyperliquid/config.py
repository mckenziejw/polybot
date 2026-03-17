"""
Configuration for the Hyperliquid trading framework.

Loads settings from environment variables or a config dict.
Defaults to TESTNET for safety.
"""

import os
from dataclasses import dataclass, field


MAINNET_REST = "https://api.hyperliquid.xyz"
TESTNET_REST = "https://api.hyperliquid-testnet.xyz"

MAINNET_WS = "wss://api.hyperliquid.xyz/ws"
TESTNET_WS = "wss://api.hyperliquid-testnet.xyz/ws"


@dataclass
class Config:
    """Hyperliquid connection and trading configuration."""

    # Auth — private key is only needed for trading
    private_key: str | None = None
    address: str | None = None

    # Network — default to testnet for safety
    testnet: bool = True

    # Funding monitor thresholds
    funding_alert_threshold_annual_pct: float = 20.0  # alert if annualized > this
    funding_poll_interval_s: int = 300                 # REST poll every 5 min
    funding_csv_path: str = "funding_history.csv"

    # Arb executor
    max_position_usd: float = 1000.0
    rebalance_threshold_pct: float = 1.0   # rebalance if delta exceeds this %
    dry_run: bool = True                    # paper mode by default

    # Fees (base tier)
    taker_fee_bps: float = 3.5   # 0.035%
    maker_fee_bps: float = 1.0   # 0.01% on perps

    @property
    def rest_url(self) -> str:
        return TESTNET_REST if self.testnet else MAINNET_REST

    @property
    def ws_url(self) -> str:
        return TESTNET_WS if self.testnet else MAINNET_WS

    @property
    def info_url(self) -> str:
        return f"{self.rest_url}/info"

    @property
    def exchange_url(self) -> str:
        return f"{self.rest_url}/exchange"

    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables."""
        testnet_str = os.environ.get("HYPERLIQUID_TESTNET", "1").lower()
        testnet = testnet_str not in ("0", "false", "no")

        return cls(
            private_key=os.environ.get("HYPERLIQUID_PRIVATE_KEY"),
            address=os.environ.get("HYPERLIQUID_ADDRESS"),
            testnet=testnet,
            funding_alert_threshold_annual_pct=float(
                os.environ.get("HL_FUNDING_ALERT_PCT", "20.0")
            ),
            max_position_usd=float(
                os.environ.get("HL_MAX_POSITION_USD", "1000.0")
            ),
            dry_run=os.environ.get("HL_DRY_RUN", "1").lower()
            not in ("0", "false", "no"),
        )

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        """Load configuration from a dictionary (e.g., parsed JSON)."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
