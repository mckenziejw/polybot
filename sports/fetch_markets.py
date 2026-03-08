"""
Fetch all current Polymarket sports markets and save to JSON.

Usage:
    python sports/fetch_markets.py

Output: data/sports/markets.json
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

BASE_URL = "https://sports-api.polymarket.com"
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "sports"

# Rate-limit: pause between paginated requests
REQUEST_DELAY_S = 0.25


def fetch_json(endpoint: str, params: dict | None = None) -> dict | list:
    """GET a JSON endpoint with error handling."""
    url = f"{BASE_URL}{endpoint}"
    log.info("GET %s  params=%s", url, params)
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data


def fetch_sports() -> list[dict]:
    """Return list of sport categories."""
    data = fetch_json("/sports")
    if isinstance(data, dict):
        # Might be wrapped: {"sports": [...]} or {"data": [...]}
        for key in ("sports", "data", "results"):
            if key in data:
                return data[key]
        log.warning("Unexpected /sports shape, keys=%s. Returning raw.", list(data.keys()))
        return [data]
    return data


def fetch_markets_page(params: dict | None = None) -> list[dict]:
    """Fetch one page of markets. Returns list of market dicts."""
    data = fetch_json("/markets", params=params)
    if isinstance(data, dict):
        for key in ("markets", "data", "results"):
            if key in data:
                return data[key]
        log.warning("Unexpected /markets shape, keys=%s. Returning raw.", list(data.keys()))
        return [data]
    return data


def fetch_all_markets() -> list[dict]:
    """Paginate through all sports markets."""
    all_markets: list[dict] = []
    offset = 0
    limit = 100  # guess at page size

    while True:
        params = {"limit": limit, "offset": offset}
        page = fetch_markets_page(params)
        if not page:
            break
        all_markets.extend(page)
        log.info("Fetched %d markets so far (page had %d)", len(all_markets), len(page))
        if len(page) < limit:
            break
        offset += limit
        time.sleep(REQUEST_DELAY_S)

    return all_markets


def extract_market_info(raw: dict) -> dict:
    """
    Pull the fields we care about from a raw market dict.

    The exact schema is uncertain — we extract what we can and pass
    through the rest so nothing is silently lost.
    """
    # Try common field names; fall back gracefully
    def g(*keys, default=None):
        for k in keys:
            if k in raw:
                return raw[k]
        return default

    # Outcomes / tokens — format varies
    outcomes = g("outcomes", "tokens", "options", default=[])
    token_ids = []
    outcome_prices = {}
    for o in outcomes:
        if isinstance(o, dict):
            tid = o.get("token_id") or o.get("tokenId") or o.get("asset_id")
            label = o.get("outcome") or o.get("label") or o.get("name", "?")
            price = o.get("price") or o.get("lastPrice")
            if tid:
                token_ids.append(tid)
            if price is not None:
                outcome_prices[label] = float(price)
        elif isinstance(o, str):
            # Just a label, no nested info
            pass

    info = {
        "market_id": g("id", "market_id", "marketId"),
        "condition_id": g("condition_id", "conditionId"),
        "sport": g("sport", "sportKey", "category"),
        "title": g("title", "question", "description", default=""),
        "matchup": g("matchup", "teams", "teamNames"),
        "event_time": g("event_time", "eventTime", "startTime", "start_time"),
        "status": g("status", "state"),
        "token_ids": token_ids,
        "outcome_prices": outcome_prices,
    }

    # Keep full raw blob for debugging
    info["_raw"] = raw
    return info


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # ── Sports categories ──
    try:
        sports = fetch_sports()
        log.info("Found %d sports categories", len(sports))
    except Exception as e:
        log.warning("Failed to fetch /sports: %s — continuing without sport list", e)
        sports = []

    # ── Markets ──
    raw_markets = fetch_all_markets()
    log.info("Fetched %d total markets", len(raw_markets))

    if not raw_markets:
        log.warning("No markets returned. Check if the API is live / schema changed.")
        # Save an empty file so downstream doesn't crash on missing file
        out_path = DATA_DIR / "markets.json"
        out_path.write_text(json.dumps({"sports": sports, "markets": []}, indent=2))
        log.info("Wrote empty markets file to %s", out_path)
        return

    markets = [extract_market_info(m) for m in raw_markets]

    out = {
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sports": sports,
        "markets": markets,
    }

    out_path = DATA_DIR / "markets.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    log.info("Wrote %d markets to %s", len(markets), out_path)

    # Quick summary
    by_sport: dict[str, int] = {}
    for m in markets:
        s = m.get("sport") or "unknown"
        by_sport[s] = by_sport.get(s, 0) + 1
    for sport, count in sorted(by_sport.items(), key=lambda x: -x[1]):
        log.info("  %s: %d markets", sport, count)


if __name__ == "__main__":
    main()
