"""
market_info.py

Shared market metadata and Gamma API fetching logic.
Used by both the ingest service and execution service so the
slug construction, token ID mapping, and retry logic stay in one place.
"""

import json
import logging
import time
from datetime import datetime, timezone

import requests

logger = logging.getLogger(__name__)

GAMMA_API    = "https://gamma-api.polymarket.com"
MARKET_CADENCE_S = 300  # 5-minute markets


class MarketInfo:
    """Metadata for one active BTC up/down market window."""

    def __init__(
        self,
        slug: str,
        condition_id: str,
        up_token_id: str,
        down_token_id: str,
        end_time: str,
    ):
        self.slug         = slug
        self.condition_id = condition_id
        self.up_token_id  = up_token_id
        self.down_token_id = down_token_id
        self.end_time     = end_time

    @property
    def token_ids(self) -> list[str]:
        return [self.up_token_id, self.down_token_id]

    @property
    def end_ts(self) -> int:
        return int(datetime.fromisoformat(
            self.end_time.replace("Z", "+00:00")
        ).timestamp())

    def is_expired(self) -> bool:
        return int(datetime.now(timezone.utc).timestamp()) >= self.end_ts

    def seconds_remaining(self) -> float:
        return max(0.0, self.end_ts - datetime.now(timezone.utc).timestamp())

    def __str__(self):
        return f"MarketInfo(slug={self.slug}, end_time={self.end_time})"


def _parse_market_result(result: dict) -> MarketInfo:
    """Parse a Gamma API market result dict into a MarketInfo."""
    slug           = result["slug"]
    clob_token_ids = json.loads(result["clobTokenIds"])
    assert len(clob_token_ids) == 2, f"Expected 2 token IDs, got {len(clob_token_ids)}"

    outcomes = json.loads(result.get("outcomes", '["Up", "Down"]'))
    if outcomes[0].lower() == "up":
        up_token_id, down_token_id = clob_token_ids[0], clob_token_ids[1]
    else:
        up_token_id, down_token_id = clob_token_ids[1], clob_token_ids[0]

    return MarketInfo(
        slug=slug,
        condition_id=result["conditionId"],
        up_token_id=up_token_id,
        down_token_id=down_token_id,
        end_time=result["endDate"],
    )


def fetch_current_market() -> MarketInfo:
    """Fetch the currently active 5-minute BTC market."""
    now_utc    = int(datetime.now(timezone.utc).timestamp())
    slug_time  = now_utc - (now_utc % MARKET_CADENCE_S)
    slug       = f"btc-updown-5m-{slug_time}"

    logger.info(f"Fetching current market: {slug}")
    resp = requests.get(f"{GAMMA_API}/markets/slug/{slug}", timeout=10)
    resp.raise_for_status()
    result = resp.json()

    assert result.get("active"),  f"Market {slug} is not active"
    assert not result.get("closed"), f"Market {slug} is already closed"

    market = _parse_market_result(result)
    logger.info(f"UP={market.up_token_id[:20]}... DN={market.down_token_id[:20]}...")
    return market


def fetch_next_market(current: MarketInfo, retries: int = 10, delay_s: float = 3.0) -> MarketInfo:
    """
    Fetch the next 5-minute market window after `current`.
    Retries until the market is active — it may not exist yet at window boundary.
    """
    slug = f"btc-updown-5m-{current.end_ts}"
    logger.info(f"Fetching next market: {slug}")

    for attempt in range(retries):
        try:
            resp = requests.get(f"{GAMMA_API}/markets/slug/{slug}", timeout=10)
            resp.raise_for_status()
            result = resp.json()
            if result.get("active") and not result.get("closed"):
                return _parse_market_result(result)
            logger.warning(f"Attempt {attempt + 1}: {slug} not active yet")
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed: {e}")
        time.sleep(delay_s)

    raise RuntimeError(f"Could not fetch next market after {retries} attempts: {slug}")