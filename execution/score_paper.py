#!/usr/bin/env python3
"""Score paper trading results from momentum v2 log."""

import re
import json
import urllib.request

LOG = "data/paper/momentum_v2_20260306_110459.log"

# Parse trades from log
trades = []
resolutions = {}

with open(LOG) as f:
    lines = f.readlines()

current_slug = None
for line in lines:
    # Market open
    m = re.search(r'=== Market Open: (btc-updown-5m-\d+) ===', line)
    if m:
        current_slug = m.group(1)

    # Trade fill
    m = re.search(r'\[MOM\] BUY (Up|Down) (\d+)@([\d.]+)', line)
    if m and current_slug:
        trades.append({
            'slug': current_slug,
            'direction': m.group(1),
            'tokens': int(m.group(2)),
            'price': float(m.group(3)),
        })

    # Resolution from log
    m = re.search(r'(btc-updown-5m-\d+): winner=(Up|Down)', line)
    if m:
        resolutions[m.group(1)] = m.group(2)

    # Pending resolution
    m = re.search(r'(btc-updown-5m-\d+): resolution pending', line)
    if m:
        resolutions[m.group(1)] = 'pending'

# For any pending, try fetching from API
for slug, winner in list(resolutions.items()):
    if winner == 'pending':
        try:
            url = f"https://gamma-api.polymarket.com/markets/slug/{slug}"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
                outcomes = data.get("outcomes", [])
                prices = data.get("outcomePrices", [])
                # API returns these as JSON strings, not lists
                if isinstance(outcomes, str):
                    outcomes = json.loads(outcomes)
                if isinstance(prices, str):
                    prices = json.loads(prices)
                for i, p in enumerate(prices):
                    if p.strip('"') == "1":
                        resolutions[slug] = outcomes[i]
                        break
        except Exception as e:
            print(f"  API error for {slug}: {e}")

# Score
print(f"{'#':>2}  {'Slug':<32} {'Dir':<5} {'Tokens':>6} {'Price':>6} {'Winner':<7} {'Result':<5} {'PnL':>8}")
print("-" * 90)

total_pnl = 0
wins = 0
losses = 0
pending = 0

for i, t in enumerate(trades, 1):
    slug = t['slug']
    winner = resolutions.get(slug, '???')

    if winner in ('pending', '???'):
        result = '???'
        pnl = 0
        pending += 1
    elif t['direction'] == winner:
        result = 'WIN'
        pnl = t['tokens'] * (1 - t['price'])
        wins += 1
    else:
        result = 'LOSS'
        pnl = -(t['tokens'] * t['price'])
        losses += 1

    total_pnl += pnl
    print(f"{i:>2}  {slug:<32} {t['direction']:<5} {t['tokens']:>6} {t['price']:>6.3f} {winner:<7} {result:<5} {pnl:>+8.2f}")

print("-" * 90)
resolved = wins + losses
print(f"\nResolved: {resolved}/{len(trades)}  |  Wins: {wins}  Losses: {losses}  Pending: {pending}")
if resolved > 0:
    print(f"Win rate: {wins/resolved*100:.1f}%")
print(f"Total PnL: {total_pnl:+.2f} USDC")

# Cumulative equity curve
print(f"\n{'#':>2}  {'Cumulative PnL':>14}  {'Drawdown':>10}")
print("-" * 35)
cum = 0
peak = 0
max_dd = 0
for i, t in enumerate(trades, 1):
    slug = t['slug']
    winner = resolutions.get(slug, '???')
    if winner in ('pending', '???'):
        continue
    if t['direction'] == winner:
        cum += t['tokens'] * (1 - t['price'])
    else:
        cum -= t['tokens'] * t['price']
    peak = max(peak, cum)
    dd = peak - cum
    max_dd = max(max_dd, dd)
    print(f"{i:>2}  {cum:>+14.2f}  {dd:>10.2f}")

print(f"\nMax drawdown: {max_dd:.2f} USDC")
