#!/usr/bin/env python3
"""
Hyperliquid Funding Rate Arbitrage Monitor

Fetches current and predicted funding rates from Hyperliquid perps,
calculates annualized yields, and displays ranked arbitrage opportunities.

Read-only -- no trading functionality. No API key required.

Usage:
    python funding_arb.py
    python funding_arb.py --top 30
    python funding_arb.py --min-yield 10
    python funding_arb.py --history-days 14
"""

import argparse
import sys
import time
from datetime import datetime, timezone
from typing import Any

import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_URL = "https://api.hyperliquid.xyz/info"
HEADERS = {"Content-Type": "application/json"}

FUNDING_PAYMENTS_PER_DAY = 3          # every 8 hours
DAYS_PER_YEAR = 365
ANNUALIZE_MULT = FUNDING_PAYMENTS_PER_DAY * DAYS_PER_YEAR  # 1095

# Fee assumptions (base tier, maker both legs)
MAKER_FEE_BPS = 1.5          # 0.015%
ENTRY_EXIT_COST_BPS = 2 * MAKER_FEE_BPS  # open + close, both legs (perp + spot/hedge)

REQUEST_TIMEOUT = 15  # seconds

# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------


def _post(payload: dict, timeout: int = REQUEST_TIMEOUT) -> Any:
    """POST to the Hyperliquid info endpoint and return parsed JSON."""
    resp = requests.post(BASE_URL, json=payload, headers=HEADERS, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def fetch_meta_and_contexts() -> tuple[list[dict], list[dict]]:
    """Return (universe, assetCtxs) from metaAndAssetCtxs."""
    data = _post({"type": "metaAndAssetCtxs"})
    # Response is a list: [meta_obj, asset_ctxs_list]
    meta = data[0]
    asset_ctxs = data[1]
    return meta["universe"], asset_ctxs


def fetch_predicted_fundings() -> dict[str, dict]:
    """Return {coin: {fundingRate, nextFundingTime}} for predicted rates."""
    data = _post({"type": "predictedFundings"})
    result: dict[str, dict] = {}
    # Response is a nested list: [[venue_data, ...], ...]
    # Each element contains coin name and predicted funding info
    for entry in data:
        # Each entry is a list of [coin, {venues: [...]}] or similar structure
        # The exact shape can vary; handle both known formats
        if isinstance(entry, list):
            for item in entry:
                if isinstance(item, list) and len(item) == 2:
                    coin = item[0]
                    info = item[1]
                    if isinstance(info, dict):
                        result[coin] = info
                elif isinstance(item, dict) and "coin" in item:
                    result[item["coin"]] = item
        elif isinstance(entry, dict) and "coin" in entry:
            result[entry["coin"]] = entry
    return result


def fetch_all_mids() -> dict[str, float]:
    """Return {coin: mid_price}."""
    data = _post({"type": "allMids"})
    return {k: float(v) for k, v in data.items()}


def fetch_funding_history(coin: str, days: int = 7) -> list[dict]:
    """Fetch historical funding rates for a coin over the last N days."""
    start_ms = int((time.time() - days * 86400) * 1000)
    data = _post({"type": "fundingHistory", "coin": coin, "startTime": start_ms})
    return data


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def annualized_yield(funding_rate: float) -> float:
    """Convert a single 8-hour funding rate to annualized percent."""
    return funding_rate * ANNUALIZE_MULT * 100


def compute_history_stats(history: list[dict]) -> dict:
    """Compute risk metrics from historical funding data."""
    if not history:
        return {
            "avg_rate": 0.0,
            "avg_annual": 0.0,
            "volatility": 0.0,
            "pct_positive": 0.0,
            "pct_negative": 0.0,
            "min_rate": 0.0,
            "max_rate": 0.0,
            "n_samples": 0,
        }

    rates = [float(h["fundingRate"]) for h in history]
    n = len(rates)
    avg = sum(rates) / n
    variance = sum((r - avg) ** 2 for r in rates) / n if n > 1 else 0.0
    std = variance ** 0.5

    positive_count = sum(1 for r in rates if r > 0)
    negative_count = sum(1 for r in rates if r < 0)

    return {
        "avg_rate": avg,
        "avg_annual": annualized_yield(avg),
        "volatility": std,
        "vol_annual": annualized_yield(std),
        "pct_positive": positive_count / n * 100,
        "pct_negative": negative_count / n * 100,
        "min_rate": min(rates),
        "max_rate": max(rates),
        "n_samples": n,
    }


def net_yield_after_fees(annual_yield_pct: float, holding_days: int = 30) -> float:
    """
    Estimate net annualized yield after entry/exit maker fees.

    Entry + exit costs are fixed one-time costs spread over the holding period.
    Both legs (perp + hedge) incur maker fees on open and close.
    """
    # Total one-time cost in percent
    total_cost_pct = ENTRY_EXIT_COST_BPS / 100 * 2  # x2 for both legs (perp + spot)
    # Annualize the one-time cost based on holding period
    annualized_cost = total_cost_pct * (DAYS_PER_YEAR / holding_days)
    return annual_yield_pct - annualized_cost


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------


def format_table(rows: list[list[str]], headers: list[str]) -> str:
    """Simple table formatter (no tabulate dependency)."""
    if not rows:
        return "  (no data)\n"

    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))

    # Build format string
    fmt = "  ".join(f"{{:<{w}}}" for w in col_widths)
    sep = "  ".join("-" * w for w in col_widths)

    lines = [fmt.format(*headers), sep]
    for row in rows:
        lines.append(fmt.format(*row))
    return "\n".join(lines)


def display_overview(assets: list[dict]) -> None:
    """Print the ranked overview table of funding opportunities."""
    headers = [
        "Rank",
        "Coin",
        "Price",
        "Funding (8h)",
        "Annual %",
        "Predicted",
        "Pred Ann %",
        "OI ($M)",
        "Direction",
    ]
    rows = []
    for i, a in enumerate(assets, 1):
        direction = "LONG hedge" if a["funding_rate"] > 0 else "SHORT hedge"
        pred_str = f"{a['predicted_rate']:.6f}" if a.get("predicted_rate") is not None else "N/A"
        pred_ann = f"{a['predicted_annual']:.1f}%" if a.get("predicted_annual") is not None else "N/A"
        rows.append([
            str(i),
            a["coin"],
            f"${a['price']:,.2f}",
            f"{a['funding_rate']:.6f}",
            f"{a['annual_yield']:.1f}%",
            pred_str,
            pred_ann,
            f"{a['oi_usd']:.1f}",
            direction,
        ])
    print("\n=== Hyperliquid Funding Rate Arbitrage Opportunities ===")
    print(f"    Sorted by absolute annualized yield (descending)\n")
    print(format_table(rows, headers))
    print()


def display_detailed(asset: dict, stats: dict, holding_days: int = 30) -> None:
    """Print detailed analysis for a single asset."""
    coin = asset["coin"]
    annual = asset["annual_yield"]
    net_30 = net_yield_after_fees(annual, holding_days=30)
    net_90 = net_yield_after_fees(annual, holding_days=90)
    net_180 = net_yield_after_fees(annual, holding_days=180)

    direction = "SHORT perp + LONG spot" if asset["funding_rate"] > 0 else "LONG perp + SHORT spot"

    print(f"  --- {coin} ---")
    print(f"  Price:              ${asset['price']:,.2f}")
    print(f"  Current 8h rate:    {asset['funding_rate']:.6f} ({annual:+.1f}% ann.)")
    if asset.get("predicted_rate") is not None:
        print(f"  Predicted next:     {asset['predicted_rate']:.6f} ({asset['predicted_annual']:+.1f}% ann.)")
    print(f"  Open interest:      ${asset['oi_usd']:.1f}M")
    print(f"  Strategy:           {direction}")
    print()
    print(f"  Net yield (after {ENTRY_EXIT_COST_BPS:.1f}bp maker fees x2 legs, open+close):")
    print(f"    Hold 30 days:     {net_30:+.1f}%  ann.")
    print(f"    Hold 90 days:     {net_90:+.1f}%  ann.")
    print(f"    Hold 180 days:    {net_180:+.1f}%  ann.")
    print()

    if stats["n_samples"] > 0:
        print(f"  Historical stats ({stats['n_samples']} samples):")
        print(f"    Avg 8h rate:      {stats['avg_rate']:.6f} ({stats['avg_annual']:+.1f}% ann.)")
        print(f"    Rate volatility:  {stats['volatility']:.6f} ({stats['vol_annual']:.1f}% ann.)")
        print(f"    % positive:       {stats['pct_positive']:.0f}%")
        print(f"    % negative:       {stats['pct_negative']:.0f}%")
        print(f"    Min / Max rate:   {stats['min_rate']:.6f} / {stats['max_rate']:.6f}")

        # Sharpe-like ratio: avg / std
        if stats["volatility"] > 0:
            consistency = stats["avg_rate"] / stats["volatility"]
            print(f"    Consistency:      {consistency:.2f} (avg/std, higher = more stable)")
    print()


def display_capital_estimates(annual_yield_pct: float, coin: str) -> None:
    """Show expected returns at various capital levels."""
    print(f"  Estimated returns for {coin} at current rates:")
    headers = ["Capital", "Hold 30d", "Hold 90d", "Hold 180d", "Hold 365d"]
    rows = []
    for capital in [1_000, 5_000, 10_000, 50_000, 100_000]:
        returns = []
        for days in [30, 90, 180, 365]:
            net = net_yield_after_fees(annual_yield_pct, holding_days=days)
            dollar_return = capital * (net / 100) * (days / 365)
            returns.append(f"${dollar_return:,.0f}")
        rows.append([f"${capital:>7,}", *returns])
    print(format_table(rows, headers))
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hyperliquid funding rate arbitrage monitor (read-only)"
    )
    parser.add_argument(
        "--top", type=int, default=20,
        help="Number of top opportunities to display (default: 20)"
    )
    parser.add_argument(
        "--detailed", type=int, default=5,
        help="Number of top assets to show detailed analysis for (default: 5)"
    )
    parser.add_argument(
        "--min-yield", type=float, default=0.0,
        help="Minimum absolute annualized yield %% to display (default: 0)"
    )
    parser.add_argument(
        "--history-days", type=int, default=7,
        help="Days of historical funding to fetch for detailed view (default: 7)"
    )
    parser.add_argument(
        "--min-oi", type=float, default=0.0,
        help="Minimum open interest in $M to include (default: 0)"
    )
    args = parser.parse_args()

    print("Fetching Hyperliquid funding data...")

    # 1. Fetch current funding rates and asset metadata
    try:
        universe, asset_ctxs = fetch_meta_and_contexts()
    except requests.RequestException as e:
        print(f"ERROR: Failed to fetch meta/asset data: {e}", file=sys.stderr)
        sys.exit(1)

    # 2. Fetch predicted funding rates
    predicted = {}
    try:
        predicted = fetch_predicted_fundings()
    except requests.RequestException as e:
        print(f"WARNING: Could not fetch predicted fundings: {e}", file=sys.stderr)

    # 3. Fetch mid prices
    mids = {}
    try:
        mids = fetch_all_mids()
    except requests.RequestException as e:
        print(f"WARNING: Could not fetch mid prices: {e}", file=sys.stderr)

    # 4. Build asset list
    assets = []
    for meta, ctx in zip(universe, asset_ctxs):
        coin = meta["name"]
        try:
            funding_rate = float(ctx.get("funding", 0))
            mark_px = float(ctx.get("markPx", 0))
            oi_raw = float(ctx.get("openInterest", 0))
        except (ValueError, TypeError):
            continue

        if mark_px == 0:
            continue

        # Use mid price if available, else mark price
        price = mids.get(coin, mark_px)
        oi_usd = oi_raw * price / 1e6  # convert to $M

        annual = annualized_yield(funding_rate)

        # Predicted rate
        pred_info = predicted.get(coin)
        pred_rate = None
        pred_annual = None
        if pred_info and isinstance(pred_info, dict):
            pr = pred_info.get("fundingRate")
            if pr is not None:
                try:
                    pred_rate = float(pr)
                    pred_annual = annualized_yield(pred_rate)
                except (ValueError, TypeError):
                    pass

        assets.append({
            "coin": coin,
            "price": price,
            "funding_rate": funding_rate,
            "annual_yield": annual,
            "predicted_rate": pred_rate,
            "predicted_annual": pred_annual,
            "oi_usd": oi_usd,
            "mark_px": mark_px,
        })

    # Filter by minimum OI
    if args.min_oi > 0:
        assets = [a for a in assets if a["oi_usd"] >= args.min_oi]

    # Filter by minimum yield
    if args.min_yield > 0:
        assets = [a for a in assets if abs(a["annual_yield"]) >= args.min_yield]

    # Sort by absolute annualized yield (descending)
    assets.sort(key=lambda a: abs(a["annual_yield"]), reverse=True)

    if not assets:
        print("No assets match the given filters.")
        sys.exit(0)

    # 5. Display overview table
    display_overview(assets[:args.top])

    # 6. Summary stats
    pos_count = sum(1 for a in assets if a["funding_rate"] > 0)
    neg_count = sum(1 for a in assets if a["funding_rate"] < 0)
    avg_abs = sum(abs(a["annual_yield"]) for a in assets) / len(assets)
    print(f"  Summary: {len(assets)} assets | {pos_count} positive / {neg_count} negative funding")
    print(f"  Average absolute annualized yield: {avg_abs:.1f}%")
    print(f"  Fee assumption: {MAKER_FEE_BPS:.1f}bp maker per leg (open + close = {ENTRY_EXIT_COST_BPS:.1f}bp total per side)")
    print()

    # 7. Detailed analysis for top N
    detailed_n = min(args.detailed, len(assets))
    if detailed_n > 0:
        print(f"=== Detailed Analysis (Top {detailed_n}) ===\n")

    for asset in assets[:detailed_n]:
        coin = asset["coin"]
        # Fetch historical funding
        stats = {"n_samples": 0}
        try:
            history = fetch_funding_history(coin, days=args.history_days)
            stats = compute_history_stats(history)
        except requests.RequestException as e:
            print(f"  WARNING: Could not fetch history for {coin}: {e}")

        display_detailed(asset, stats)
        display_capital_estimates(asset["annual_yield"], coin)

    # Timestamp
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"  Data as of: {now}")
    print(f"  All rates are indicative. Past funding does not guarantee future rates.")
    print()


if __name__ == "__main__":
    main()
