#!/usr/bin/env python3
"""Analyze live trading performance from fills and orders CSVs.

Uses the orders CSV as ground truth for our trades (MATCHED status = filled).
Cross-references with Gamma API for win/loss resolution.
Filters fills CSV to only our order_ids, ignoring complement fills and other traders.
"""

import csv
import json
import time
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone

FILLS_CSV = "execution/src/data/metrics/fills.csv"
ORDERS_CSV = "execution/src/data/metrics/orders.csv"


def load_orders():
    """Load orders, dedup by order_id taking MATCHED status if available."""
    raw = []
    with open(ORDERS_CSV) as f:
        for row in csv.DictReader(f):
            raw.append(row)

    # Group by order_id, prefer MATCHED status
    by_oid = defaultdict(list)
    for o in raw:
        by_oid[o["order_id"]].append(o)

    orders = {}
    for oid, rows in by_oid.items():
        matched = [r for r in rows if r["status"] == "MATCHED"]
        if matched:
            orders[oid] = matched[0]
        else:
            orders[oid] = rows[0]

    return orders


def load_our_fills(our_order_ids):
    """Load fills, keep only those matching our order_ids, dedup by (order_id, asset_id, price, size)."""
    fills = []
    seen = set()
    with open(FILLS_CSV) as f:
        for row in csv.DictReader(f):
            if row["order_id"] not in our_order_ids:
                continue
            key = (row["order_id"], row["asset_id"], row["price"], row["size"])
            if key not in seen:
                seen.add(key)
                fills.append(row)
    return fills


def query_gamma(slug):
    """Query Gamma API for market resolution. Returns dict of asset_id -> {outcome, price}."""
    parts = slug.split("-")
    ts = parts[-1]
    event_slug = f"btc-updown-5m-{ts}"
    url = f"https://gamma-api.polymarket.com/events?slug={event_slug}"

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())

        if not data:
            return None

        event = data[0]
        markets = event.get("markets", [])

        result = {}
        for market in markets:
            clob_token_ids = json.loads(market.get("clobTokenIds", "[]"))
            outcomes = json.loads(market.get("outcomes", "[]"))
            outcome_prices = json.loads(market.get("outcomePrices", "[]"))

            for i, token_id in enumerate(clob_token_ids):
                if i < len(outcome_prices):
                    result[token_id] = {
                        "outcome": outcomes[i] if i < len(outcomes) else "?",
                        "price": outcome_prices[i],
                    }

        return result
    except Exception as e:
        print(f"  Error querying {slug}: {e}")
        return None


def main():
    orders = load_orders()
    print(f"Unique orders: {len(orders)}")

    # Only consider MATCHED orders (filled)
    matched_orders = {oid: o for oid, o in orders.items() if o["status"] == "MATCHED"}
    print(f"MATCHED orders (filled trades): {len(matched_orders)}")

    # Load our fills (filtered to our order_ids)
    our_fills = load_our_fills(set(orders.keys()))
    print(f"Our fills (deduped): {len(our_fills)}")

    # For each matched order, find the fill that matches the order's asset_id
    # (not the complement fill)
    # Group fills by order_id
    fills_by_oid = defaultdict(list)
    for f in our_fills:
        fills_by_oid[f["order_id"]].append(f)

    # Get unique market slugs from matched orders
    market_slugs = sorted(set(o["market_slug"] for o in matched_orders.values()))
    print(f"Unique markets traded: {len(market_slugs)}")

    # Query Gamma for each market
    print("\nQuerying Gamma API for market resolutions...")
    resolutions = {}
    for i, slug in enumerate(market_slugs):
        resolutions[slug] = query_gamma(slug)
        if (i + 1) % 10 == 0:
            print(f"  Queried {i + 1}/{len(market_slugs)} markets...")
        time.sleep(0.1)
    resolved_count = sum(1 for v in resolutions.values() if v is not None)
    print(f"  Done. Got data for {resolved_count}/{len(market_slugs)} markets")

    # Build trade records from matched orders
    trades = []
    for oid, order in matched_orders.items():
        slug = order["market_slug"]
        asset_id = order["asset_id"]
        price = float(order["price"])
        size = float(order["size"])
        ts = int(order["timestamp"])
        session = order["session_id"]

        # Check if we have a fill for this order on the intended asset
        order_fills = fills_by_oid.get(oid, [])
        intended_fills = [f for f in order_fills if f["asset_id"] == asset_id]

        # Use fill size if available (actual filled qty may differ from order qty)
        if intended_fills:
            actual_size = float(intended_fills[0]["size"])
            actual_price = float(intended_fills[0]["price"])
        else:
            # Some fills may be on complement asset only (minting)
            # In that case, the order size/price is the intended trade
            actual_size = size
            actual_price = price

        # Resolution
        resolution = resolutions.get(slug)
        if resolution is None:
            outcome = "NO_DATA"
            pnl = None
            token_label = "?"
        elif asset_id not in resolution:
            # The order's asset_id might not match Gamma's token IDs
            # Try to find by checking complement
            outcome = "ASSET_MISMATCH"
            pnl = None
            token_label = "?"
        else:
            res = resolution[asset_id]
            token_label = res["outcome"]
            try:
                op = float(res["price"])
            except (ValueError, TypeError):
                outcome = "PENDING"
                pnl = None
                trades.append({
                    "timestamp": ts, "session": session, "slug": slug,
                    "asset_id": asset_id, "price": actual_price,
                    "size": actual_size, "outcome": "PENDING", "pnl": None,
                    "token_label": token_label, "cost": actual_price * actual_size,
                })
                continue

            if op == 1.0:
                outcome = "WIN"
                pnl = (1.0 - actual_price) * actual_size
            elif op == 0.0:
                outcome = "LOSS"
                pnl = -actual_price * actual_size
            else:
                outcome = "PENDING"
                pnl = None

        trades.append({
            "timestamp": ts, "session": session, "slug": slug,
            "asset_id": asset_id, "price": actual_price,
            "size": actual_size, "outcome": outcome, "pnl": pnl,
            "token_label": token_label, "cost": actual_price * actual_size,
        })

    # Separate resolved vs pending
    resolved = [t for t in trades if t["outcome"] in ("WIN", "LOSS")]
    pending = [t for t in trades if t["outcome"] == "PENDING"]
    errors = [t for t in trades if t["outcome"] in ("NO_DATA", "ASSET_MISMATCH")]

    wins = [t for t in resolved if t["outcome"] == "WIN"]
    losses = [t for t in resolved if t["outcome"] == "LOSS"]

    # ========== SUMMARY ==========
    print("\n" + "=" * 70)
    print("TRADING PERFORMANCE SUMMARY")
    print("=" * 70)

    sessions = sorted(set(t["session"] for t in trades))
    print(f"\nSessions: {len(sessions)}")
    for s in sessions:
        st = [t for t in trades if t["session"] == s]
        print(f"  {s}: {len(st)} trades")

    print(f"\nTotal trades: {len(trades)}")
    print(f"  Resolved: {len(resolved)}")
    print(f"  Pending:  {len(pending)}")
    if errors:
        print(f"  Errors:   {len(errors)}")
        for e in errors:
            print(f"    {e['outcome']}: {e['slug']} asset={e['asset_id'][:20]}...")

    n = len(resolved)
    if n > 0:
        wr = len(wins) / n * 100
        total_pnl = sum(t["pnl"] for t in resolved)
        total_cost = sum(t["cost"] for t in resolved)

        print(f"\n--- Resolved Trades ---")
        print(f"Wins:     {len(wins)}")
        print(f"Losses:   {len(losses)}")
        print(f"Win rate: {wr:.1f}%")
        print(f"\nTotal PnL:        ${total_pnl:+.2f}")
        print(f"Capital deployed: ${total_cost:.2f}")
        if total_cost > 0:
            print(f"ROI:              {total_pnl / total_cost * 100:+.2f}%")

        avg_entry = sum(t["price"] for t in resolved) / n
        avg_win_price = sum(t["price"] for t in wins) / len(wins) if wins else 0
        avg_loss_price = sum(t["price"] for t in losses) / len(losses) if losses else 0
        avg_size = sum(t["size"] for t in resolved) / n

        print(f"\nAvg entry price:        {avg_entry:.3f}")
        print(f"Avg entry price (wins): {avg_win_price:.3f}")
        print(f"Avg entry price (loss): {avg_loss_price:.3f}")
        print(f"Avg position size:      {avg_size:.1f} tokens")

        avg_win_pnl = sum(t["pnl"] for t in wins) / len(wins) if wins else 0
        avg_loss_pnl = sum(t["pnl"] for t in losses) / len(losses) if losses else 0
        print(f"\nAvg PnL per win:  ${avg_win_pnl:+.4f}")
        print(f"Avg PnL per loss: ${avg_loss_pnl:+.4f}")
        if avg_loss_pnl != 0:
            print(f"Win/Loss ratio:   {abs(avg_win_pnl / avg_loss_pnl):.2f}")

    # ========== PnL BY SESSION ==========
    print("\n" + "=" * 70)
    print("PnL BY SESSION")
    print("=" * 70)
    for s in sessions:
        sr = [t for t in resolved if t["session"] == s]
        sw = [t for t in sr if t["outcome"] == "WIN"]
        sl = [t for t in sr if t["outcome"] == "LOSS"]
        sp = [t for t in trades if t["session"] == s and t["outcome"] == "PENDING"]
        sn = len(sr)
        spnl = sum(t["pnl"] for t in sr) if sr else 0
        scost = sum(t["cost"] for t in sr) if sr else 0
        swr = len(sw) / sn * 100 if sn > 0 else 0

        print(f"\n  {s}")
        if sn > 0:
            print(f"    Trades: {sn}  |  W: {len(sw)}  L: {len(sl)}  |  WR: {swr:.1f}%")
            print(f"    PnL: ${spnl:+.2f}  |  Capital: ${scost:.2f}  |  ROI: {spnl / scost * 100:+.1f}%" if scost > 0 else f"    PnL: ${spnl:+.2f}")
        else:
            print(f"    No resolved trades")
        if sp:
            print(f"    Pending: {len(sp)}")

    # ========== PnL BY ENTRY PRICE BUCKET ==========
    print("\n" + "=" * 70)
    print("PnL BY ENTRY PRICE BUCKET")
    print("=" * 70)

    buckets = [
        (0.50, 0.55, "0.50-0.55"),
        (0.55, 0.60, "0.55-0.60"),
        (0.60, 0.65, "0.60-0.65"),
        (0.65, 0.70, "0.65-0.70"),
        (0.70, 0.75, "0.70-0.75"),
        (0.75, 0.80, "0.75-0.80"),
        (0.80, 0.85, "0.80-0.85"),
        (0.85, 0.90, "0.85-0.90"),
        (0.90, 1.01, "0.90-1.00"),
    ]

    print(f"\n  {'Bucket':<12} {'N':>4} {'W':>4} {'L':>4} {'WR':>7} {'PnL':>10} {'Capital':>10} {'ROI':>8}")
    print(f"  {'-' * 12} {'-' * 4} {'-' * 4} {'-' * 4} {'-' * 7} {'-' * 10} {'-' * 10} {'-' * 8}")

    for lo, hi, label in buckets:
        bt = [t for t in resolved if lo <= t["price"] < hi]
        if not bt:
            continue
        bw = len([t for t in bt if t["outcome"] == "WIN"])
        bl = len([t for t in bt if t["outcome"] == "LOSS"])
        bn = len(bt)
        bpnl = sum(t["pnl"] for t in bt)
        bcost = sum(t["cost"] for t in bt)
        bwr = bw / bn * 100
        broi = bpnl / bcost * 100 if bcost > 0 else 0
        print(f"  {label:<12} {bn:>4} {bw:>4} {bl:>4} {bwr:>6.1f}% {bpnl:>+10.2f} {bcost:>10.2f} {broi:>+7.1f}%")

    # ========== PnL BY DAY ==========
    print("\n" + "=" * 70)
    print("PnL BY DAY")
    print("=" * 70)

    by_day = defaultdict(list)
    for t in resolved:
        dt = datetime.fromtimestamp(t["timestamp"] / 1000, tz=timezone.utc)
        by_day[dt.strftime("%Y-%m-%d")].append(t)

    print(f"\n  {'Day':<12} {'N':>4} {'W':>4} {'L':>4} {'WR':>7} {'PnL':>10} {'Capital':>10} {'ROI':>8}")
    print(f"  {'-' * 12} {'-' * 4} {'-' * 4} {'-' * 4} {'-' * 7} {'-' * 10} {'-' * 10} {'-' * 8}")

    for day in sorted(by_day.keys()):
        dt = by_day[day]
        dw = len([t for t in dt if t["outcome"] == "WIN"])
        dl = len([t for t in dt if t["outcome"] == "LOSS"])
        dn = len(dt)
        dpnl = sum(t["pnl"] for t in dt)
        dcost = sum(t["cost"] for t in dt)
        dwr = dw / dn * 100
        droi = dpnl / dcost * 100 if dcost > 0 else 0
        print(f"  {day:<12} {dn:>4} {dw:>4} {dl:>4} {dwr:>6.1f}% {dpnl:>+10.2f} {dcost:>10.2f} {droi:>+7.1f}%")

    # ========== RECENT 20 TRADES ==========
    print("\n" + "=" * 70)
    print("RECENT 20 TRADES (most recent first)")
    print("=" * 70)

    sorted_trades = sorted(trades, key=lambda t: t["timestamp"], reverse=True)

    print(f"\n  {'Time (UTC)':<18} {'Token':<6} {'Price':>6} {'Size':>6} {'Cost':>7} {'Result':>8} {'PnL':>10}")
    print(f"  {'-' * 18} {'-' * 6} {'-' * 6} {'-' * 6} {'-' * 7} {'-' * 8} {'-' * 10}")

    for t in sorted_trades[:20]:
        dt = datetime.fromtimestamp(t["timestamp"] / 1000, tz=timezone.utc)
        time_str = dt.strftime("%m-%d %H:%M:%S")
        pnl_str = f"${t['pnl']:+.2f}" if t["pnl"] is not None else "---"
        print(f"  {time_str:<18} {t['token_label']:<6} {t['price']:>6.2f} {t['size']:>6.0f} ${t['cost']:>5.2f} {t['outcome']:>8} {pnl_str:>10}")

    # ========== STREAK ANALYSIS ==========
    print("\n" + "=" * 70)
    print("STREAK ANALYSIS")
    print("=" * 70)

    sorted_resolved = sorted(resolved, key=lambda t: t["timestamp"])
    max_win_streak = 0
    max_loss_streak = 0
    cur_streak = 0
    cur_type = None
    for t in sorted_resolved:
        if t["outcome"] == cur_type:
            cur_streak += 1
        else:
            cur_type = t["outcome"]
            cur_streak = 1
        if cur_type == "WIN":
            max_win_streak = max(max_win_streak, cur_streak)
        else:
            max_loss_streak = max(max_loss_streak, cur_streak)

    print(f"\n  Max win streak:  {max_win_streak}")
    print(f"  Max loss streak: {max_loss_streak}")

    # Cumulative PnL
    cum_pnl = 0
    max_pnl = 0
    max_dd = 0
    for t in sorted_resolved:
        cum_pnl += t["pnl"]
        max_pnl = max(max_pnl, cum_pnl)
        dd = max_pnl - cum_pnl
        max_dd = max(max_dd, dd)

    print(f"  Max drawdown:    ${max_dd:.2f}")
    print(f"  Final PnL:       ${cum_pnl:+.2f}")


if __name__ == "__main__":
    main()
