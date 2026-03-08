#!/usr/bin/env python3
"""
Analyze live trading PnL by:
1. Reading orders.csv for our MATCHED orders
2. Querying CLOB API for actual trade history (authenticated)
3. Querying Gamma API for market resolutions
4. Computing per-trade and aggregate PnL
"""

import csv
import json
import time
import hmac
import hashlib
import base64
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone

import requests
from eth_account import Account

# ── Config ──────────────────────────────────────────────────────────────────

ROOT = Path(__file__).parent
CONFIG = json.loads((ROOT / "config.json").read_text())
PM = CONFIG["polymarket"]

CLOB_URL = PM["host"]
API_KEY = PM["api_key"]
API_SECRET = PM["api_secret"]
API_PASSPHRASE = PM["api_passphrase"]
PRIVATE_KEY = PM["private_key"]
PROXY_WALLET = PM["proxy_wallet"]

EOA_ADDRESS = Account.from_key(PRIVATE_KEY).address

GAMMA_API = "https://gamma-api.polymarket.com"

# ── Auth helpers ────────────────────────────────────────────────────────────

def build_hmac_signature(timestamp: str, method: str, path: str) -> str:
    """Build POLY_SIGNATURE header value."""
    message = timestamp + method + path
    key = base64.urlsafe_b64decode(API_SECRET)
    sig = hmac.new(key, message.encode("utf-8"), hashlib.sha256).digest()
    return base64.urlsafe_b64encode(sig).decode("utf-8")


def auth_headers(method: str, path: str) -> dict:
    """Return dict of CLOB auth headers."""
    ts = str(int(time.time()))
    sig = build_hmac_signature(ts, method, path)
    return {
        "POLY_ADDRESS": EOA_ADDRESS,
        "POLY_SIGNATURE": sig,
        "POLY_TIMESTAMP": ts,
        "POLY_API_KEY": API_KEY,
        "POLY_PASSPHRASE": API_PASSPHRASE,
    }


def clob_get(path: str) -> dict:
    """Authenticated GET to CLOB API."""
    headers = auth_headers("GET", path)
    url = CLOB_URL + path
    r = requests.get(url, headers=headers, timeout=15)
    r.raise_for_status()
    return r.json()


# ── Fetch trade history from CLOB ───────────────────────────────────────────

def fetch_all_trades():
    """Fetch all trades using py-clob-client."""
    from py_clob_client.client import ClobClient
    from py_clob_client.clob_types import ApiCreds

    creds = ApiCreds(
        api_key=API_KEY,
        api_secret=API_SECRET,
        api_passphrase=API_PASSPHRASE,
    )
    client = ClobClient(
        CLOB_URL,
        key=PRIVATE_KEY,
        chain_id=137,
        creds=creds,
        funder=PROXY_WALLET,
    )
    return client.get_trades()


# ── Fetch market resolution from Gamma ──────────────────────────────────────

def fetch_market_resolution(condition_id: str) -> dict | None:
    """Query Gamma API for market details by condition ID."""
    url = f"{GAMMA_API}/markets"
    r = requests.get(url, params={"id": condition_id}, timeout=10)
    if r.status_code != 200:
        return None
    data = r.json()
    if isinstance(data, list) and data:
        return data[0]
    return data if isinstance(data, dict) else None


# ── Parse local orders CSV ──────────────────────────────────────────────────

def load_orders_csv():
    """Load orders from local CSV file."""
    csv_path = ROOT / "execution" / "src" / "data" / "metrics" / "orders.csv"
    if not csv_path.exists():
        return []
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


# ── Main analysis ───────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  Polymarket Live Trading PnL Analysis")
    print("=" * 70)
    print(f"\nEOA Address: {EOA_ADDRESS}")
    print(f"Proxy Wallet: {PROXY_WALLET}")
    print(f"CLOB URL: {CLOB_URL}")

    # ── Step 1: Load local orders ───────────────────────────────────────
    print("\n── Local Orders (orders.csv) ──")
    orders = load_orders_csv()
    matched_orders = [o for o in orders if o["status"] == "MATCHED"]
    canceled_orders = [o for o in orders if o["status"] == "CANCELED"]
    live_orders = [o for o in orders if o["status"] == "LIVE"]

    print(f"Total order records: {len(orders)}")
    print(f"  MATCHED: {len(matched_orders)}")
    print(f"  CANCELED: {len(canceled_orders)}")
    print(f"  LIVE (intermediate): {len(live_orders)}")

    # Deduplicate matched orders by order_id (same order may appear multiple times)
    unique_matched = {}
    for o in matched_orders:
        oid = o["order_id"]
        if oid not in unique_matched:
            unique_matched[oid] = o

    print(f"  Unique MATCHED orders: {len(unique_matched)}")

    # ── Step 2: Try CLOB API for trade history ──────────────────────────
    print("\n── CLOB API Trade History ──")
    clob_trades = []
    try:
        clob_trades = fetch_all_trades()
        print(f"Total CLOB trades returned: {len(clob_trades)}")
    except Exception as e:
        print(f"CLOB API error: {e}")
        print("Falling back to local CSV analysis only.")

    # ── Step 3: Determine unique markets to look up ─────────────────────
    # Collect market slugs from orders
    market_slugs = set()
    market_orders = defaultdict(list)
    for oid, o in unique_matched.items():
        slug = o.get("market_slug", "")
        market_slugs.add(slug)
        market_orders[slug].append(o)

    # Also collect from CLOB trades
    clob_by_market = defaultdict(list)
    for t in clob_trades:
        market = t.get("market", t.get("condition_id", ""))
        clob_by_market[market].append(t)

    print(f"\nUnique markets from orders CSV: {len(market_slugs)}")
    print(f"Unique markets from CLOB trades: {len(clob_by_market)}")

    # ── Step 4: Resolve markets via Gamma API ───────────────────────────
    print("\n── Resolving Markets via Gamma API ──")

    # Extract condition IDs from CLOB trades and order IDs
    # From orders CSV, we need to map slug -> conditionId
    # The slug format is btc-updown-5m-{timestamp}
    # We'll query Gamma by slug

    resolution_cache = {}  # conditionId -> resolution info
    token_to_winner = {}   # asset_id -> bool (True if winning token)

    # Strategy: query Gamma for each unique slug
    for slug in sorted(market_slugs):
        try:
            url = f"{GAMMA_API}/events"
            r = requests.get(url, params={"slug": slug}, timeout=10)
            if r.status_code == 200:
                data = r.json()
                events = data if isinstance(data, list) else [data]
                for event in events:
                    if not isinstance(event, dict):
                        continue
                    markets = event.get("markets", [])
                    for mkt in markets:
                        cid = mkt.get("conditionId", mkt.get("condition_id", ""))
                        closed = mkt.get("closed", False)
                        resolved = mkt.get("resolved", False)
                        winner = mkt.get("winner", "")  # token ID of winner
                        outcome = mkt.get("outcome", "")
                        outcomes = mkt.get("outcomes", "")
                        outcome_prices = mkt.get("outcomePrices", "")
                        tokens = mkt.get("clobTokenIds", "")

                        resolution_cache[cid] = {
                            "slug": slug,
                            "closed": closed,
                            "resolved": resolved,
                            "winner": winner,
                            "outcome": outcome,
                            "outcomes": outcomes,
                            "outcome_prices": outcome_prices,
                            "tokens": tokens,
                        }

                        # Map token IDs to win/loss
                        if tokens and outcome_prices:
                            try:
                                tok_list = json.loads(tokens) if isinstance(tokens, str) else tokens
                                price_list = json.loads(outcome_prices) if isinstance(outcome_prices, str) else outcome_prices
                                for tid, prc in zip(tok_list, price_list):
                                    token_to_winner[str(tid)] = float(prc) > 0.5
                            except (json.JSONDecodeError, ValueError):
                                pass
            time.sleep(0.2)
        except Exception as e:
            print(f"  Error fetching {slug}: {e}")

    print(f"Resolved {len(resolution_cache)} markets")

    # ── Step 5: Compute PnL ─────────────────────────────────────────────
    print("\n── Per-Trade PnL ──")
    print(f"{'Slug':<35} {'Side':<5} {'Price':>6} {'Size':>5} {'Result':>7} {'PnL':>8}")
    print("-" * 75)

    total_pnl = 0.0
    total_cost = 0.0
    wins = 0
    losses = 0
    unknown = 0
    trades_detail = []

    for slug in sorted(market_orders.keys()):
        ords = market_orders[slug]
        for o in ords:
            price = float(o["price"])
            size = int(o["size"])
            asset_id = o["asset_id"]
            cost = price * size

            # Determine if this token won
            won = token_to_winner.get(asset_id)

            if won is True:
                pnl = (1.0 - price) * size  # Payout is $1 per token, cost was price
                result = "WIN"
                wins += 1
            elif won is False:
                pnl = -price * size  # Token worthless
                result = "LOSS"
                losses += 1
            else:
                pnl = 0.0
                result = "???"
                unknown += 1

            total_pnl += pnl
            total_cost += cost

            short_slug = slug[-20:] if len(slug) > 20 else slug
            print(f"  ...{short_slug:<31} {o['side']:<5} {price:>6.2f} {size:>5} {result:>7} ${pnl:>+7.2f}")

            trades_detail.append({
                "slug": slug,
                "price": price,
                "size": size,
                "asset_id": asset_id,
                "won": won,
                "pnl": pnl,
                "result": result,
                "session": o.get("session_id", ""),
            })

    # ── Step 6: Summary ─────────────────────────────────────────────────
    total_trades = wins + losses + unknown
    resolved_trades = wins + losses

    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  Total matched trades:   {total_trades}")
    print(f"  Resolved:               {resolved_trades}")
    print(f"    Wins:                  {wins}")
    print(f"    Losses:                {losses}")
    if resolved_trades > 0:
        print(f"    Win Rate:              {wins/resolved_trades*100:.1f}%")
    print(f"  Unresolved/Unknown:     {unknown}")
    print(f"  Total Cost (invested):  ${total_cost:.2f}")
    print(f"  Total PnL:              ${total_pnl:+.2f}")
    if total_cost > 0:
        print(f"  ROI:                    {total_pnl/total_cost*100:+.1f}%")

    # ── Session breakdown ───────────────────────────────────────────────
    print("\n── By Session ──")
    sessions = defaultdict(lambda: {"wins": 0, "losses": 0, "unknown": 0, "pnl": 0.0, "cost": 0.0})
    for t in trades_detail:
        s = t["session"]
        sessions[s]["pnl"] += t["pnl"]
        sessions[s]["cost"] += t["price"] * t["size"]
        if t["result"] == "WIN":
            sessions[s]["wins"] += 1
        elif t["result"] == "LOSS":
            sessions[s]["losses"] += 1
        else:
            sessions[s]["unknown"] += 1

    for sid, stats in sorted(sessions.items()):
        r = stats["wins"] + stats["losses"]
        wr = f"{stats['wins']/r*100:.0f}%" if r > 0 else "N/A"
        print(f"  {sid}: {stats['wins']}W/{stats['losses']}L/{stats['unknown']}? "
              f"WR={wr} PnL=${stats['pnl']:+.2f} Cost=${stats['cost']:.2f}")

    # ── CLOB API trade details (if available) ───────────────────────────
    if clob_trades:
        print(f"\n── CLOB API Trade Details (first 10) ──")
        for t in clob_trades[:10]:
            print(f"  {json.dumps(t, indent=2)[:200]}...")

    # ── Excluding outlier first trade ─────────────────────────────────
    # First session had 158-token order (likely before bet_dollars was set to $5)
    normal_trades = [t for t in trades_detail if t["size"] <= 20]
    if len(normal_trades) < len(trades_detail):
        n_wins = sum(1 for t in normal_trades if t["result"] == "WIN")
        n_losses = sum(1 for t in normal_trades if t["result"] == "LOSS")
        n_resolved = n_wins + n_losses
        n_pnl = sum(t["pnl"] for t in normal_trades)
        n_cost = sum(t["price"] * t["size"] for t in normal_trades)
        print(f"\n── Excluding Outlier Trades (size > 20 tokens) ──")
        print(f"  Trades:    {len(normal_trades)} (excluded {len(trades_detail) - len(normal_trades)})")
        print(f"  Wins:      {n_wins}")
        print(f"  Losses:    {n_losses}")
        if n_resolved > 0:
            print(f"  Win Rate:  {n_wins/n_resolved*100:.1f}%")
        print(f"  Cost:      ${n_cost:.2f}")
        print(f"  PnL:       ${n_pnl:+.2f}")
        if n_cost > 0:
            print(f"  ROI:       {n_pnl/n_cost*100:+.1f}%")

        # Avg win/loss
        win_pnls = [t["pnl"] for t in normal_trades if t["result"] == "WIN"]
        loss_pnls = [t["pnl"] for t in normal_trades if t["result"] == "LOSS"]
        if win_pnls:
            print(f"  Avg Win:   ${sum(win_pnls)/len(win_pnls):+.2f}")
        if loss_pnls:
            print(f"  Avg Loss:  ${sum(loss_pnls)/len(loss_pnls):+.2f}")

        # Fee estimate (March 2026 crypto fee: C * p * 0.25 * (p*(1-p))^2, C=1)
        print(f"\n── Fee Estimate (new crypto fee formula) ──")
        total_fees = 0.0
        for t in normal_trades:
            p = t["price"]
            fee_rate = p * 0.25 * (p * (1 - p)) ** 2
            fee = fee_rate * t["price"] * t["size"]
            total_fees += fee
        print(f"  Estimated taker fees: ${total_fees:.2f}")
        print(f"  PnL after fees:       ${n_pnl - total_fees:+.2f}")

        # Price distribution
        prices = [t["price"] for t in normal_trades]
        print(f"\n── Entry Price Distribution ──")
        print(f"  Min:    {min(prices):.2f}")
        print(f"  Max:    {max(prices):.2f}")
        print(f"  Mean:   {sum(prices)/len(prices):.2f}")
        print(f"  Median: {sorted(prices)[len(prices)//2]:.2f}")

        # Win rate by price bucket
        print(f"\n── Win Rate by Entry Price ──")
        buckets = {"0.55-0.65": [], "0.65-0.75": [], "0.75-0.85": [], "0.85-0.95": []}
        for t in normal_trades:
            p = t["price"]
            if p < 0.65:
                buckets["0.55-0.65"].append(t)
            elif p < 0.75:
                buckets["0.65-0.75"].append(t)
            elif p < 0.85:
                buckets["0.75-0.85"].append(t)
            else:
                buckets["0.85-0.95"].append(t)
        for bucket, trades in buckets.items():
            if not trades:
                continue
            bw = sum(1 for t in trades if t["result"] == "WIN")
            bl = sum(1 for t in trades if t["result"] == "LOSS")
            br = bw + bl
            bpnl = sum(t["pnl"] for t in trades)
            wr_str = f"{bw/br*100:.0f}%" if br > 0 else "N/A"
            print(f"  {bucket}: {bw}W/{bl}L WR={wr_str} PnL=${bpnl:+.2f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
