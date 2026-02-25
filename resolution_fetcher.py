import os
from pathlib import Path
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

fetcher = ResolutionFetcher()

slugs = [
    f.replace(".parquet", "")
    for f in os.listdir("data/trade_events")
    if f.endswith(".parquet")
]

# print(f"Found {len(slugs)} market windows")
resolutions = fetcher.fetch_batch(slugs)
# print(f"Resolved: {len(resolutions)}")

# for slug, r in resolutions.items():
#     print(f"\n{slug}")
#     print(f"  closed_time:   {r['closed_time']}")
#     print(f"  winning_token: {r['winning_token'][:20]}...")
#     print(f"  outcomes:      {r['token_outcomes']}")

import pandas as pd
import pyarrow.parquet as pq

# Pick a resolved window
slug = "btc-updown-5m-1771347300"
resolution = resolutions[slug]

# Load trade data
trades = pq.read_table(f"data/trade_events/{slug}.parquet").to_pandas()

# Convert timestamps to datetime for readability
trades["exchange_dt"] = pd.to_datetime(trades["exchange_timestamp"], unit="ms", utc=True)
trades["received_dt"] = pd.to_datetime(trades["received_timestamp"], unit="ms", utc=True)

# Tag each row with its outcome
trades["outcome"] = trades["asset_id"].map(resolution["token_outcomes"])
trades["won"] = trades["outcome"] == 1.0

# Look at the final 60 seconds
end_ts = trades["exchange_timestamp"].max()
final_minute = trades[trades["exchange_timestamp"] >= end_ts - 60_000]

print(f"Window: {slug}")
print(f"Total trades: {len(trades)}")
print(f"Trades in final 60s: {len(final_minute)}")
print(f"Winning token: {resolution['winning_token'][:20]}...")
print(f"\nFinal 60s price range by token:")
print(
    final_minute.groupby("asset_id")["mid_price"].agg(["min", "max", "last"])
)
print(f"\nFinal 60s — last few rows:")
print(
    final_minute[["exchange_dt", "asset_id", "mid_price", "best_bid", "best_ask", "won"]]
    .tail(10)
    .to_string()
)
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from pathlib import Path
from dataclasses import dataclass, asdict

DATA_DIR = Path("data")


@dataclass
class WindowSummary:
    # Identification
    slug: str
    condition_id: str
    closed_time: str
    winning_token: str

    # Window-level price metrics
    up_token_id: str
    down_token_id: str
    up_token_outcome: float      # 1.0 = won, 0.0 = lost
    down_token_outcome: float

    # Peak probability metrics (full window)
    max_mid_price: float          # highest mid_price seen for either token
    max_mid_token: str            # which token hit the max
    max_mid_correct: bool         # did the highest probability token win?
    time_of_max_mid: pd.Timestamp # when did peak probability occur?
    seconds_before_close: float   # how long before close did peak occur?

    # Threshold crossing times (for the high-probability token)
    first_cross_90: float | None  # seconds before close when mid first crossed 0.90
    first_cross_95: float | None
    first_cross_97: float | None
    first_cross_99: float | None

    # Final window metrics (last 60s)
    final_60s_max_mid: float
    final_60s_min_mid: float
    final_60s_trade_count: int
    final_60s_volatility: float   # std dev of mid_price in final 60s

    # Final window metrics (last 30s)
    final_30s_max_mid: float
    final_30s_min_mid: float
    final_30s_trade_count: int
    final_30s_volatility: float

    # Liquidity at close (from book snapshots)
    final_book_bid_depth: float   # total size in top 10 bids at close
    final_book_ask_depth: float

    # Volume
    total_volume: float
    final_60s_volume: float


class WindowSummariser:
    """
    Produces a WindowSummary for a single market window by joining
    trade event data with resolution outcomes.
    """

    def __init__(self, data_dir: Path = DATA_DIR):
        self.data_dir = data_dir

    def summarise(self, slug: str, resolution: dict) -> WindowSummary | None:
        """
        Build a WindowSummary for the given slug and resolution.
        Returns None if data files are missing, corrupt, or insufficient.
        """
        trade_path = self.data_dir / "trade_events" / f"{slug}.parquet"
        book_path = self.data_dir / "book_snapshots" / f"{slug}.parquet"

        if not trade_path.exists():
            return None

        try:
            trades = pq.read_table(trade_path).to_pandas()
        except Exception as e:
            logger.warning(f"Failed to read {trade_path}: {e}")
            return None

        if len(trades) < 100:
            # Too few trades — likely incomplete capture
            return None

        token_outcomes = resolution["token_outcomes"]
        token_ids = list(token_outcomes.keys())
        up_token_id = token_ids[0]
        down_token_id = token_ids[1]

        # Tag rows with outcome
        trades["outcome"] = trades["asset_id"].map(token_outcomes)

        # Work only with the winning token for threshold analysis
        # (losing token is just mirror image)
        winning_token = resolution["winning_token"]
        winning_trades = trades[trades["asset_id"] == winning_token].copy()
        winning_trades = winning_trades.sort_values("exchange_timestamp")

        close_ts = trades["exchange_timestamp"].max()
        close_dt = pd.to_datetime(close_ts, unit="ms", utc=True)

        # Seconds before close for each trade
        winning_trades["secs_before_close"] = (
            close_ts - winning_trades["exchange_timestamp"]
        ) / 1000

        # Peak probability
        max_idx = winning_trades["mid_price"].idxmax()
        max_row = winning_trades.loc[max_idx]

        # Threshold crossing times
        def first_cross(threshold: float) -> float | None:
            crossed = winning_trades[winning_trades["mid_price"] >= threshold]
            if crossed.empty:
                return None
            return crossed["secs_before_close"].max()  # earliest = most seconds before close

        # Final window slices
        final_60s = winning_trades[winning_trades["secs_before_close"] <= 60]
        final_30s = winning_trades[winning_trades["secs_before_close"] <= 30]

        # Book depth at close
        final_bid_depth = 0.0
        final_ask_depth = 0.0
        if book_path.exists():
            try:
                books = pq.read_table(book_path).to_pandas()
                books = books[books["asset_id"] == winning_token]
                if not books.empty:
                    last_book = books.sort_values("exchange_timestamp").iloc[-1]
                    final_bid_depth = sum(
                        last_book.get(f"bid_size_{i}", 0) for i in range(1, 11)
                    )
                    final_ask_depth = sum(
                        last_book.get(f"ask_size_{i}", 0) for i in range(1, 11)
                    )
            except Exception as e:
                logger.warning(f"Failed to read {book_path}: {e}")

        return WindowSummary(
            slug=slug,
            condition_id=resolution["condition_id"],
            closed_time=resolution["closed_time"],
            winning_token=winning_token,
            up_token_id=up_token_id,
            down_token_id=down_token_id,
            up_token_outcome=token_outcomes[up_token_id],
            down_token_outcome=token_outcomes[down_token_id],
            max_mid_price=max_row["mid_price"],
            max_mid_token=winning_token,
            max_mid_correct=True,  # we're already filtering to winning token
            time_of_max_mid=pd.to_datetime(max_row["exchange_timestamp"], unit="ms", utc=True),
            seconds_before_close=max_row["secs_before_close"],
            first_cross_90=first_cross(0.90),
            first_cross_95=first_cross(0.95),
            first_cross_97=first_cross(0.97),
            first_cross_99=first_cross(0.99),
            final_60s_max_mid=final_60s["mid_price"].max() if not final_60s.empty else 0.0,
            final_60s_min_mid=final_60s["mid_price"].min() if not final_60s.empty else 0.0,
            final_60s_trade_count=len(final_60s),
            final_60s_volatility=final_60s["mid_price"].std() if len(final_60s) > 1 else 0.0,
            final_30s_max_mid=final_30s["mid_price"].max() if not final_30s.empty else 0.0,
            final_30s_min_mid=final_30s["mid_price"].min() if not final_30s.empty else 0.0,
            final_30s_trade_count=len(final_30s),
            final_30s_volatility=final_30s["mid_price"].std() if len(final_30s) > 1 else 0.0,
            final_book_bid_depth=final_bid_depth,
            final_book_ask_depth=final_ask_depth,
            total_volume=trades["size"].sum(),
            final_60s_volume=final_60s["size"].sum() if not final_60s.empty else 0.0,
        )

    def summarise_all(self, resolutions: dict) -> pd.DataFrame:
        """
        Build summaries for all resolved windows and return as a DataFrame.
        """
        summaries = []
        for slug, resolution in resolutions.items():
            summary = self.summarise(slug, resolution)
            if summary:
                summaries.append(asdict(summary))
            else:
                print(f"Skipped {slug} — insufficient data")

        if not summaries:
            return pd.DataFrame()

        df = pd.DataFrame(summaries)
        df["closed_time"] = pd.to_datetime(df["closed_time"], utc=True)
        df["time_of_max_mid"] = pd.to_datetime(df["time_of_max_mid"], utc=True)
        return df.sort_values("closed_time").reset_index(drop=True)

summariser = WindowSummariser()
df = summariser.summarise_all(resolutions)

print(f"Windows summarised: {len(df)}")
print(f"\nColumns: {list(df.columns)}")
print(f"\nThreshold crossing rates (% of windows where winning token crossed):")
print(f"  >= 0.90: {df['first_cross_90'].notna().mean():.1%}")
print(f"  >= 0.95: {df['first_cross_95'].notna().mean():.1%}")
print(f"  >= 0.97: {df['first_cross_97'].notna().mean():.1%}")
print(f"  >= 0.99: {df['first_cross_99'].notna().mean():.1%}")
print(f"\nFinal 60s stats:")
print(df[["final_60s_max_mid", "final_60s_min_mid", "final_60s_volatility"]].describe())

# Windows where winning token was >= 0.95 in final 60s
high_prob_windows = df[df["final_60s_min_mid"] >= 0.95]

print(f"\nWindows with >= 0.95 probability throughout final 60s: {len(high_prob_windows)}/{len(df)}")
print(f"Failure rate: {0}/{len(high_prob_windows)} = 0.00%")
print(f"\nSample:")
print(high_prob_windows[["slug", "final_60s_min_mid", "final_60s_max_mid", "final_60s_volatility"]])

# More aggressive threshold
high_prob_30s = df[df["final_30s_min_mid"] >= 0.97]
print(f"\nWindows with >= 0.97 probability throughout final 30s: {len(high_prob_30s)}/{len(df)}")

print(f"\nQuestion 3: Opportunity frequency")
print(f"Windows where >= 0.95 for full final 60s: {len(high_prob_windows)} ({len(high_prob_windows)/len(df):.1%})")
print(f"Windows where >= 0.97 for full final 30s: {len(high_prob_30s)} ({len(high_prob_30s)/len(df):.1%})")