import json
import time
import logging
import requests
from pathlib import Path

logger = logging.getLogger(__name__)

GAMMA_API = "https://gamma-api.polymarket.com"
CACHE_PATH = Path("data/resolution_cache.json")
REQUEST_DELAY = 0.2  # seconds between requests to avoid rate limiting


class ResolutionFetcher:
    """
    Fetches and caches market resolution outcomes from the Gamma API.
    Maps token_id -> outcome (1.0 = paid out, 0.0 = worthless)
    """

    def __init__(self, cache_path: Path = CACHE_PATH):
        self.cache_path = cache_path
        self._cache: dict = self._load_cache()

    def _load_cache(self) -> dict:
        if self.cache_path.exists():
            with open(self.cache_path) as f:
                data = json.load(f)
                logger.info(f"Loaded {len(data)} cached resolutions")
                return data
        return {}

    def _save_cache(self):
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "w") as f:
            json.dump(self._cache, f, indent=2)

    def fetch_resolution(self, slug: str) -> dict | None:
        """
        Returns {token_id: outcome} for a resolved market, or None if
        the market is unresolved or the fetch fails.
        Caches results locally — safe to call repeatedly.
        """
        # Return cached result if available
        if slug in self._cache:
            return self._cache[slug]

        try:
            resp = requests.get(
                f"{GAMMA_API}/markets/slug/{slug}",
                timeout=10
            )
            resp.raise_for_status()
            result = resp.json()

            # Only cache fully resolved markets
            if result.get("umaResolutionStatus") != "resolved":
                logger.debug(f"Market {slug} not yet resolved")
                return None

            clob_token_ids = json.loads(result["clobTokenIds"])
            outcome_prices = json.loads(result["outcomePrices"])

            resolution = {
                "slug": slug,
                "condition_id": result["conditionId"],
                "closed_time": result.get("closedTime"),
                "token_outcomes": {
                    clob_token_ids[0]: float(outcome_prices[0]),
                    clob_token_ids[1]: float(outcome_prices[1]),
                },
                "winning_token": clob_token_ids[
                    outcome_prices.index("1")
                ],
            }

            self._cache[slug] = resolution
            self._save_cache()
            time.sleep(REQUEST_DELAY)
            return resolution

        except Exception as e:
            logger.error(f"Failed to fetch resolution for {slug}: {e}")
            return None

    def fetch_batch(self, slugs: list[str]) -> dict:
        """
        Fetch resolutions for multiple markets.
        Returns {slug: resolution} for all successfully resolved markets.
        Skips already-cached entries without API calls.
        """
        results = {}
        uncached = [s for s in slugs if s not in self._cache]

        if uncached:
            logger.info(f"Fetching {len(uncached)} resolutions ({len(slugs) - len(uncached)} cached)")

        for slug in slugs:
            resolution = self.fetch_resolution(slug)
            if resolution:
                results[slug] = resolution

        return results