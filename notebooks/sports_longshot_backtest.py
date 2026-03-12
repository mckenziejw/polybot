"""
Sports Favorite-Longshot Bias Backtest

Strategy: Mint YES/NO pairs on fee-free sports markets, sell the overpriced
longshot side, keep the favorite to resolution.

Steps:
  1. Load Telonex markets index, filter for resolved sports markets with book data
  2. Download sample of historical orderbook snapshots
  3. Analyze: do longshots trade above fair value? What's the PnL of selling them?
  4. Compare Polymarket prices to sharp lines if available

Usage:
  .venv/bin/python notebooks/sports_longshot_backtest.py analyze   # index analysis only
  .venv/bin/python notebooks/sports_longshot_backtest.py download  # download book data
  .venv/bin/python notebooks/sports_longshot_backtest.py backtest  # run backtest
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MARKETS_URL = "https://api.telonex.io/v1/datasets/polymarket/markets"
DATA_DIR = Path("./datasets/sports_longshot")
BOOK_DIR = DATA_DIR / "books"

def _load_api_key() -> str:
    config_path = Path(__file__).resolve().parent.parent / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
        key = cfg.get("telonex", {}).get("api_key", "")
        if key:
            return key
    return os.environ.get("TELONEX_API_KEY", "")

API_KEY = _load_api_key()

# ---------------------------------------------------------------------------
# Sports categories and identification
# ---------------------------------------------------------------------------

# Slug prefixes that indicate sports markets
SPORTS_PREFIXES = [
    "nba-", "nfl-", "nhl-", "mlb-", "cbb-",  # US sports
    "epl-", "ere-", "spl-", "lig-", "bun-", "ser-",  # soccer leagues
    "wta-", "atp-",  # tennis
    "f1-", "nascar-",  # motorsport
    "ufc-", "boxing-",  # combat
    "lol-", "cs2-", "val-", "dota-",  # esports
    "aus-", "egy1-", "elc-",  # other soccer
]

# Categories that are clearly sports
SPORTS_CATEGORIES = ["sports", "nba", "nfl", "soccer", "tennis", "esports", "mma"]

# Moneyline-like patterns (binary outcome: team wins or doesn't)
MONEYLINE_PATTERNS = [
    "-moneyline", "-win", "-1h-moneyline",
]


def is_sports_market(row: pd.Series) -> bool:
    """Check if a market is a sports market based on slug and category."""
    slug = str(row.get("slug", "")).lower()
    cat = str(row.get("category", "")).lower()

    if any(slug.startswith(p) for p in SPORTS_PREFIXES):
        return True
    if cat in SPORTS_CATEGORIES:
        return True
    return False


def is_binary_outcome(row: pd.Series) -> bool:
    """Check if market is a simple binary outcome (moneyline/win)."""
    slug = str(row.get("slug", "")).lower()
    # Moneyline markets, match winners, game winners
    if any(p in slug for p in MONEYLINE_PATTERNS):
        return True
    # Simple team matchups (e.g., nba-lal-den-2026-01-20)
    # These tend to have just date + teams, no special suffix
    parts = slug.split("-")
    if len(parts) >= 4 and parts[0] in ["nba", "nfl", "nhl", "mlb"]:
        # Check if it's NOT a prop (no "total", "points", "rebounds", etc.)
        prop_keywords = ["total", "points", "rebounds", "assists", "team-total",
                         "1h", "1q", "spread", "over", "under"]
        if not any(kw in slug for kw in prop_keywords):
            return True
    return False


# ---------------------------------------------------------------------------
# Step 1: Analyze the Telonex markets index
# ---------------------------------------------------------------------------

def analyze_index():
    """Load and analyze the Telonex markets index for sports markets."""
    log.info("Loading Telonex markets index...")
    df = pd.read_parquet(MARKETS_URL)
    log.info(f"Total markets: {len(df)}")

    # Filter to sports
    sports_mask = df.apply(is_sports_market, axis=1)
    sports = df[sports_mask].copy()
    log.info(f"Sports markets: {len(sports)}")

    # Filter to resolved (has result_id and settled_at)
    resolved = sports[
        sports["result_id"].notna() &
        (sports["result_id"] != "") &
        sports["settled_at_us"].notna() &
        (sports["settled_at_us"] != "") &
        (sports["settled_at_us"] != "0")
    ].copy()
    log.info(f"Resolved sports markets: {len(resolved)}")

    # Filter to those with book snapshot data
    has_books = resolved[
        resolved["book_snapshot_5_from"].notna() &
        (resolved["book_snapshot_5_from"] != "")
    ].copy()
    log.info(f"Resolved sports with book data: {len(has_books)}")

    # Show category distribution
    if "category" in has_books.columns:
        cat_counts = has_books["category"].value_counts().head(20)
        log.info(f"\nCategory distribution:\n{cat_counts.to_string()}")

    # Show slug prefix distribution
    has_books["prefix"] = has_books["slug"].str.split("-").str[0]
    prefix_counts = has_books["prefix"].value_counts().head(20)
    log.info(f"\nSlug prefix distribution:\n{prefix_counts.to_string()}")

    # Analyze outcome distribution (result_id tells us which outcome won)
    # result_id matches one of outcome_0 or outcome_1
    log.info(f"\nSample resolved markets:")
    sample = has_books.head(10)
    for _, row in sample.iterrows():
        log.info(f"  {row['slug']}: result={row['result_id']}, "
                 f"outcomes=({row['outcome_0']}, {row['outcome_1']})")

    # Check for moneyline/binary markets specifically
    binary_mask = has_books.apply(is_binary_outcome, axis=1)
    binary = has_books[binary_mask]
    log.info(f"\nBinary outcome (moneyline) markets: {len(binary)}")

    # Also check markets with quotes data (gives us price history)
    has_quotes = has_books[
        has_books["quotes_from"].notna() &
        (has_books["quotes_from"] != "")
    ]
    log.info(f"Markets with quotes data: {len(has_quotes)}")

    has_trades = has_books[
        has_books["trades_from"].notna() &
        (has_books["trades_from"] != "")
    ]
    log.info(f"Markets with trades data: {len(has_trades)}")

    # Save filtered index for later use
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_DIR / "sports_resolved_index.parquet"
    has_books.to_parquet(out_path)
    log.info(f"\nSaved {len(has_books)} markets to {out_path}")

    return has_books


# ---------------------------------------------------------------------------
# Step 2: Download book snapshots for a sample
# ---------------------------------------------------------------------------

async def download_books(max_markets: int = 500, sport: str = None):
    """Download book snapshot data for resolved sports markets."""
    from telonex import download_async

    index_path = DATA_DIR / "sports_resolved_index.parquet"
    if not index_path.exists():
        log.info("Index not found, running analyze first...")
        analyze_index()

    df = pd.read_parquet(index_path)
    log.info(f"Loaded {len(df)} resolved sports markets")

    # Filter by sport if specified
    if sport:
        df = df[df["slug"].str.startswith(f"{sport}-")]
        log.info(f"Filtered to {sport}: {len(df)} markets")

    # Prioritize moneyline/binary markets
    df["is_binary"] = df.apply(is_binary_outcome, axis=1)
    df = df.sort_values("is_binary", ascending=False)

    # Take a sample (or all if max_markets is large enough)
    sample = df.head(max_markets)
    log.info(f"Downloading books for {len(sample)} markets...")

    BOOK_DIR.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(10)
    tasks = []

    for _, row in sample.iterrows():
        for asset_col in ("asset_id_0", "asset_id_1"):
            asset_id = row[asset_col]
            if not asset_id or pd.isna(asset_id):
                continue
            from_date = row["book_snapshot_5_from"]
            to_date = row["book_snapshot_5_to"]
            if not from_date or not to_date:
                continue
            tasks.append((row["slug"], asset_id, from_date, to_date))

    log.info(f"Total download tasks: {len(tasks)}")

    async def dl_one(slug, asset_id, from_date, to_date):
        to_excl = (pd.Timestamp(to_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        async with semaphore:
            try:
                return await download_async(
                    api_key=API_KEY,
                    exchange="polymarket",
                    channel="book_snapshot_5",
                    from_date=from_date,
                    to_date=to_excl,
                    asset_id=asset_id,
                    download_dir=str(BOOK_DIR),
                    force_download=False,
                )
            except Exception as e:
                log.error(f"Failed {slug}/{asset_id[:12]}...: {e}")
                return []

    done = 0
    total_files = 0
    for coro in asyncio.as_completed([dl_one(*t) for t in tasks]):
        files = await coro
        total_files += len(files)
        done += 1
        if done % 50 == 0 or done == len(tasks):
            log.info(f"  {done}/{len(tasks)} done, {total_files} files")

    log.info(f"Download complete: {total_files} files")


# ---------------------------------------------------------------------------
# Step 3: Backtest the mint-and-sell-longshot strategy
# ---------------------------------------------------------------------------

def backtest():
    """
    Backtest the favorite-longshot bias strategy:
    - For each resolved market, get the pre-market orderbook prices
    - Identify favorite (lower price) and longshot (higher price for YES)
    - Simulate: mint 1 pair ($1), sell longshot YES at best bid
    - Hold favorite YES to resolution
    - PnL = (longshot_sell_price + favorite_payout) - 1.00

    Actually for YES/NO pairs:
    - outcome_0 is YES, outcome_1 is NO (typically)
    - If YES is the longshot (price < 0.50), sell YES at bid, keep NO
    - If YES is the favorite (price > 0.50), sell NO at bid, keep YES

    Wait — we need to think about this more carefully:
    - Mint: $1 → 1 YES + 1 NO
    - Strategy: sell the LONGSHOT side (the one that's less likely to win)
    - Longshot has price < 0.50 (implied prob < 50%)
    - If favorite-longshot bias holds, longshot is overpriced relative to true prob
    - So selling it at market price is profitable in expectation

    PnL per pair:
    - Cost: $1.00 (minting)
    - Revenue: longshot_bid (from selling longshot) + favorite_payout (0 or 1)
    - If favorite wins: PnL = longshot_bid + 1.00 - 1.00 = longshot_bid
    - If favorite loses: PnL = longshot_bid + 0.00 - 1.00 = longshot_bid - 1.00
    """
    index_path = DATA_DIR / "sports_resolved_index.parquet"
    if not index_path.exists():
        log.error("Run 'analyze' first")
        return

    idx = pd.read_parquet(index_path)
    log.info(f"Loaded {len(idx)} resolved sports markets")

    # Check what book files we have
    book_files = list(BOOK_DIR.glob("*.parquet")) if BOOK_DIR.exists() else []
    if not book_files:
        log.error(f"No book files in {BOOK_DIR}. Run 'download' first.")
        return

    log.info(f"Found {len(book_files)} book files")

    # Build asset_id -> slug mapping from index
    asset_to_slug = {}
    asset_to_outcome = {}  # asset_id -> "outcome_0" or "outcome_1"
    slug_to_result = {}
    slug_to_info = {}

    for _, row in idx.iterrows():
        slug = row["slug"]
        slug_to_result[slug] = row["result_id"]
        slug_to_info[slug] = {
            "outcome_0": row["outcome_0"],
            "outcome_1": row["outcome_1"],
            "asset_id_0": row["asset_id_0"],
            "asset_id_1": row["asset_id_1"],
        }
        asset_to_slug[row["asset_id_0"]] = slug
        asset_to_slug[row["asset_id_1"]] = slug
        asset_to_outcome[row["asset_id_0"]] = "outcome_0"
        asset_to_outcome[row["asset_id_1"]] = "outcome_1"

    # Parse book files, extract prices per slug
    # Group files by asset_id first to aggregate multi-day data per token
    asset_files: dict[str, list[Path]] = {}
    unmapped_count = 0
    for f in book_files:
        parts = f.stem.split("_")
        try:
            date_idx = next(i for i, p in enumerate(parts) if p.startswith("202"))
            asset_id = parts[date_idx + 1]
        except (StopIteration, IndexError):
            continue

        if asset_id not in asset_to_slug:
            unmapped_count += 1
            continue
        asset_files.setdefault(asset_id, []).append(f)

    log.info(f"Grouped {len(asset_files)} asset_ids from {len(book_files)} files "
             f"({unmapped_count} unmapped)")

    # Read only bid_price_0 and ask_price_0 columns for speed
    COLS = ["bid_price_0", "ask_price_0"]
    slug_prices = {}  # slug -> {asset_id: stats}

    for i, (asset_id, files) in enumerate(asset_files.items()):
        slug = asset_to_slug[asset_id]

        all_bids = []
        all_asks = []
        for f in files:
            try:
                df = pd.read_parquet(f, columns=COLS)
            except Exception:
                continue
            if df.empty:
                continue
            bid_0 = pd.to_numeric(df["bid_price_0"], errors="coerce").fillna(0)
            ask_0 = pd.to_numeric(df["ask_price_0"], errors="coerce").fillna(0)
            valid = (bid_0 > 0) & (ask_0 > 0)
            all_bids.append(bid_0[valid])
            all_asks.append(ask_0[valid])

        if not all_bids:
            continue

        bids = pd.concat(all_bids)
        asks = pd.concat(all_asks)
        if len(bids) == 0:
            continue

        mid = (bids + asks) / 2
        slug_prices.setdefault(slug, {})[asset_id] = {
            "mid": mid.median(),
            "best_bid_median": bids.median(),
            "best_bid_p25": bids.quantile(0.25),
            "n_snapshots": len(mid),
            "outcome_key": asset_to_outcome.get(asset_id, "unknown"),
        }

        if (i + 1) % 5000 == 0:
            log.info(f"  Parsed {i+1}/{len(asset_files)} asset_ids...")

    log.info(f"Got prices for {len(slug_prices)} slugs")

    # Now run the backtest
    results = []
    for slug, prices in slug_prices.items():
        if len(prices) != 2:
            continue  # Need both sides

        info = slug_to_info.get(slug)
        result_id = slug_to_result.get(slug)
        if not info or not result_id:
            continue

        # Figure out which side is favorite and which is longshot
        # result_id is "0" or "1" — index into outcomes
        sides = []
        for aid, p in prices.items():
            outcome_key = p["outcome_key"]  # "outcome_0" or "outcome_1"
            outcome_idx = outcome_key[-1]   # "0" or "1"
            won = (str(result_id) == outcome_idx)
            sides.append({
                "asset_id": aid,
                "outcome_key": outcome_key,
                "outcome_label": info.get(outcome_key, "unknown"),
                "mid": p["mid"],
                "best_bid": p["best_bid_median"],
                "best_bid_p25": p["best_bid_p25"],
                "n_snapshots": p["n_snapshots"],
                "won": won,
            })

        # Sort by mid price — higher mid = favorite, lower mid = longshot
        sides.sort(key=lambda x: x["mid"], reverse=True)
        favorite, longshot = sides[0], sides[1]

        # Sanity: prices should roughly sum to 1
        price_sum = favorite["mid"] + longshot["mid"]
        if price_sum < 0.80 or price_sum > 1.20:
            continue  # Bad data

        # Strategy: sell the longshot at its best bid, keep favorite
        sell_price = longshot["best_bid"]
        if sell_price <= 0:
            continue

        fav_payout = 1.0 if favorite["won"] else 0.0
        pnl = sell_price + fav_payout - 1.0

        results.append({
            "slug": slug,
            "favorite_mid": favorite["mid"],
            "longshot_mid": longshot["mid"],
            "longshot_bid": sell_price,
            "longshot_bid_p25": longshot["best_bid_p25"],
            "price_sum": price_sum,
            "favorite_won": favorite["won"],
            "longshot_won": longshot["won"],
            "pnl": pnl,
            "n_snapshots": min(favorite["n_snapshots"], longshot["n_snapshots"]),
        })

    if not results:
        log.error("No valid results")
        return

    res = pd.DataFrame(results)
    log.info(f"\n{'='*60}")
    log.info(f"BACKTEST RESULTS: Mint-and-Sell-Longshot Strategy")
    log.info(f"{'='*60}")
    log.info(f"Total markets: {len(res)}")
    log.info(f"Favorite win rate: {res['favorite_won'].mean():.1%}")
    log.info(f"Mean PnL per trade: ${res['pnl'].mean():.4f}")
    log.info(f"Median PnL per trade: ${res['pnl'].median():.4f}")
    log.info(f"Total PnL: ${res['pnl'].sum():.2f}")
    log.info(f"Std PnL: ${res['pnl'].std():.4f}")
    log.info(f"Sharpe (per trade): {res['pnl'].mean() / res['pnl'].std():.3f}")
    log.info(f"Win rate (PnL > 0): {(res['pnl'] > 0).mean():.1%}")

    # Break down by favorite price bucket
    bins = [0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
    res["fav_bucket"] = pd.cut(res["favorite_mid"], bins=bins)
    bucket_stats = res.groupby("fav_bucket", observed=True).agg(
        count=("pnl", "size"),
        mean_pnl=("pnl", "mean"),
        fav_wr=("favorite_won", "mean"),
        longshot_bid=("longshot_bid", "mean"),
    )
    log.info(f"\nBy favorite price bucket:\n{bucket_stats.to_string()}")

    # Distribution of longshot prices
    log.info(f"\nLongshot price distribution:")
    log.info(f"  Mean:   {res['longshot_mid'].mean():.3f}")
    log.info(f"  Median: {res['longshot_mid'].median():.3f}")
    log.info(f"  P10:    {res['longshot_mid'].quantile(0.10):.3f}")
    log.info(f"  P25:    {res['longshot_mid'].quantile(0.25):.3f}")

    # What about using the pessimistic bid (25th percentile)?
    res["pnl_pessimistic"] = res["longshot_bid_p25"] + \
        res["favorite_won"].astype(float) - 1.0
    log.info(f"\nPessimistic (P25 bid) PnL: ${res['pnl_pessimistic'].mean():.4f}/trade")

    # Spread analysis: how much are we losing to the spread?
    res["longshot_spread"] = res["longshot_mid"] - res["longshot_bid"]
    log.info(f"\nSpread analysis:")
    log.info(f"  Mean longshot spread: {res['longshot_spread'].mean():.3f}")
    log.info(f"  Median longshot spread: {res['longshot_spread'].median():.3f}")

    # What if we sold at mid instead of bid? (limit order simulation)
    res["pnl_at_mid"] = res["longshot_mid"] + \
        res["favorite_won"].astype(float) - 1.0
    log.info(f"\nLimit order (sell at mid) PnL: ${res['pnl_at_mid'].mean():.4f}/trade")

    # Filtered strategies: only trade strong favorites
    for threshold in [0.60, 0.65, 0.70, 0.75, 0.80]:
        filt = res[res["favorite_mid"] >= threshold]
        if len(filt) == 0:
            continue
        log.info(f"\n--- Fav >= {threshold:.0%} ({len(filt)} markets) ---")
        log.info(f"  Fav WR: {filt['favorite_won'].mean():.1%}")
        log.info(f"  PnL/trade (at bid):  ${filt['pnl'].mean():.4f}")
        log.info(f"  PnL/trade (at mid):  ${filt['pnl_at_mid'].mean():.4f}")
        log.info(f"  Total PnL (at bid):  ${filt['pnl'].sum():.2f}")
        log.info(f"  Total PnL (at mid):  ${filt['pnl_at_mid'].sum():.2f}")
        log.info(f"  Mean longshot bid:   {filt['longshot_bid'].mean():.3f}")
        log.info(f"  Mean longshot mid:   {filt['longshot_mid'].mean():.3f}")
        log.info(f"  Win rate (PnL > 0):  {(filt['pnl'] > 0).mean():.1%}")

    # Additional: what about the OPPOSITE strategy? Sell the favorite, keep the longshot?
    # (Anti-bias: if favorites are fairly priced, selling them should be neutral)
    res["pnl_sell_fav"] = (1 - res["longshot_mid"]) + \
        res["longshot_won"].astype(float) - 1.0
    log.info(f"\n--- REVERSE: sell favorite, keep longshot ---")
    log.info(f"  PnL/trade: ${res['pnl_sell_fav'].mean():.4f}")
    log.info(f"  This should be negative if favorite-longshot bias exists")

    # Filter by spread: tight spread markets only
    log.info(f"\n--- BY SPREAD TIGHTNESS ---")
    for max_spread in [0.02, 0.04, 0.06, 0.10, 0.15]:
        tight = res[res["longshot_spread"] <= max_spread]
        if len(tight) < 5:
            continue
        log.info(f"\n  Spread <= {max_spread:.0%} ({len(tight)} markets):")
        log.info(f"    Fav WR: {tight['favorite_won'].mean():.1%}")
        log.info(f"    PnL/trade (at bid): ${tight['pnl'].mean():.4f}")
        log.info(f"    PnL/trade (at mid): ${tight['pnl_at_mid'].mean():.4f}")
        log.info(f"    Mean fav mid: {tight['favorite_mid'].mean():.3f}")

    # Combined filter: strong favorite + tight spread
    log.info(f"\n--- COMBINED: Fav >= 60% AND spread <= 0.06 ---")
    combo = res[(res["favorite_mid"] >= 0.60) & (res["longshot_spread"] <= 0.06)]
    if len(combo) >= 5:
        log.info(f"  Markets: {len(combo)}")
        log.info(f"  Fav WR: {combo['favorite_won'].mean():.1%}")
        log.info(f"  PnL/trade (at bid): ${combo['pnl'].mean():.4f}")
        log.info(f"  PnL/trade (at mid): ${combo['pnl_at_mid'].mean():.4f}")
        log.info(f"  Total PnL (at bid): ${combo['pnl'].sum():.2f}")

    log.info(f"\n--- COMBINED: Fav >= 60% AND spread <= 0.10 ---")
    combo2 = res[(res["favorite_mid"] >= 0.60) & (res["longshot_spread"] <= 0.10)]
    if len(combo2) >= 5:
        log.info(f"  Markets: {len(combo2)}")
        log.info(f"  Fav WR: {combo2['favorite_won'].mean():.1%}")
        log.info(f"  PnL/trade (at bid): ${combo2['pnl'].mean():.4f}")
        log.info(f"  Total PnL (at bid): ${combo2['pnl'].sum():.2f}")

    # Sport-by-sport breakdown
    res["sport"] = res["slug"].str.split("-").str[0]
    sport_stats = res.groupby("sport").agg(
        count=("pnl", "size"),
        mean_pnl=("pnl", "mean"),
        pnl_at_mid=("pnl_at_mid", "mean"),
        fav_wr=("favorite_won", "mean"),
        spread=("longshot_spread", "mean"),
    ).sort_values("count", ascending=False)
    log.info(f"\n--- BY SPORT ---\n{sport_stats.to_string()}")

    # Save results (drop interval column that causes parquet issues)
    out_path = DATA_DIR / "backtest_results.csv"
    res.drop(columns=["fav_bucket"], errors="ignore").to_csv(out_path, index=False)
    log.info(f"\nResults saved to {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["analyze", "download", "backtest", "all"],
                        default="analyze", nargs="?")
    parser.add_argument("--max-markets", type=int, default=500)
    parser.add_argument("--sport", type=str, default=None, help="Filter by sport prefix (nba, nfl, etc.)")
    args = parser.parse_args()

    if args.command in ("analyze", "all"):
        analyze_index()

    if args.command in ("download", "all"):
        asyncio.run(download_books(max_markets=args.max_markets, sport=args.sport))

    if args.command in ("backtest", "all"):
        backtest()
