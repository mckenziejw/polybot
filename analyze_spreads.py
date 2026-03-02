"""
analyze_spreads.py — Spread and book health distribution analysis.

Run from project root:
    python analyze_spreads.py
    python analyze_spreads.py --data-dir data/telonex_100ms --sample 200
"""

import argparse
import random
import numpy as np
import pandas as pd
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', default='data/telonex_100ms')
parser.add_argument('--sample',   type=int, default=100)
parser.add_argument('--seed',     type=int, default=42)
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)

files = sorted(Path(args.data_dir).glob('*.parquet'))
print(f"Total files: {len(files)}")
if not files:
    raise SystemExit("No parquet files found. Check --data-dir.")

sample = random.sample(files, min(args.sample, len(files)))
print(f"Sampling {len(sample)} markets...\n")

# Accumulate raw observations
all_spreads   = []
all_bid_sizes = []
all_ask_sizes = []
n_both = n_bid_only = n_ask_only = n_neither = 0

# Per-market: track fraction of bars that are one-sided
market_onesided_fracs = []

for f in sample:
    df = pd.read_parquet(f)
    for label in ['Up', 'Down']:
        tok = df[df['token_label'] == label]
        if tok.empty:
            continue

        bid_p = tok['bid_price_1'].fillna(0)
        ask_p = tok['ask_price_1'].fillna(0)
        bid_s = tok['bid_size_1'].fillna(0)
        ask_s = tok['ask_size_1'].fillna(0)

        has_bid = (bid_p > 0) & (bid_s > 0)
        has_ask = (ask_p > 0) & (ask_s > 0)

        both    = (has_bid & has_ask)
        bid_only = (has_bid & ~has_ask)
        ask_only = (~has_bid & has_ask)
        neither = (~has_bid & ~has_ask)

        n_both     += int(both.sum())
        n_bid_only += int(bid_only.sum())
        n_ask_only += int(ask_only.sum())
        n_neither  += int(neither.sum())

        # Spreads on two-sided bars
        spread = (ask_p[both] - bid_p[both])
        all_spreads.extend(spread.tolist())

        # Sizes on non-zero levels
        all_bid_sizes.extend(bid_s[has_bid].tolist())
        all_ask_sizes.extend(ask_s[has_ask].tolist())

        # Per-token one-sided fraction
        n_total = len(tok)
        if n_total > 0:
            market_onesided_fracs.append((bid_only.sum() + ask_only.sum()) / n_total)

all_spreads   = np.array(all_spreads)
all_bid_sizes = np.array(all_bid_sizes)
all_ask_sizes = np.array(all_ask_sizes)
market_onesided_fracs = np.array(market_onesided_fracs)

total_bars = n_both + n_bid_only + n_ask_only + n_neither

print("=" * 60)
print("BOOK SIDEDNESS (per 100ms bar)")
print("=" * 60)
print(f"  Both sided:  {n_both:9,}  ({100*n_both/total_bars:5.1f}%)")
print(f"  Bid only:    {n_bid_only:9,}  ({100*n_bid_only/total_bars:5.1f}%)")
print(f"  Ask only:    {n_ask_only:9,}  ({100*n_ask_only/total_bars:5.1f}%)")
print(f"  Neither:     {n_neither:9,}  ({100*n_neither/total_bars:5.1f}%)")
print(f"  Total bars:  {total_bars:9,}")

print()
print("=" * 60)
print("ONE-SIDED FRACTION PER TOKEN (distribution across tokens)")
print("=" * 60)
for p in [0, 10, 25, 50, 75, 90, 95, 99, 100]:
    print(f"  p{p:3d}: {np.percentile(market_onesided_fracs, p)*100:5.1f}% of bars one-sided")

print()
print("=" * 60)
print("SPREAD DISTRIBUTION (two-sided bars only)")
print("=" * 60)
print(f"  Observations: {len(all_spreads):,}")
# Remove crossed/negative spreads
valid = all_spreads[all_spreads >= 0]
invalid = len(all_spreads) - len(valid)
if invalid:
    print(f"  Crossed books (spread < 0): {invalid:,} ({100*invalid/len(all_spreads):.2f}%)")
print(f"  Valid spreads: {len(valid):,}")
for p in [5, 10, 25, 50, 75, 90, 95, 99, 99.9]:
    print(f"  p{p:5.1f}: {np.percentile(valid, p):.4f}  ({np.percentile(valid, p)*100:.2f} cents)")
print(f"  mean:  {valid.mean():.4f}  ({valid.mean()*100:.2f} cents)")
print(f"  max:   {valid.max():.4f}  ({valid.max()*100:.2f} cents)")

# Spread buckets
print()
print("  Spread buckets:")
buckets = [0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.01]
labels  = ['<1c', '1-2c', '2-5c', '5-10c', '10-20c', '20-50c', '>50c']
counts  = np.histogram(valid, bins=[0]+buckets)[0]
for label, count in zip(labels, counts):
    print(f"    {label:8s}: {count:8,}  ({100*count/len(valid):5.1f}%)")

print()
print("=" * 60)
print("BID SIZE DISTRIBUTION (non-zero bid_size_1, in $)")
print("=" * 60)
for p in [5, 10, 25, 50, 75, 90, 95, 99]:
    print(f"  p{p:3d}: ${np.percentile(all_bid_sizes, p):8.2f}")
print(f"  mean: ${all_bid_sizes.mean():8.2f}")

print()
print("=" * 60)
print("ASK SIZE DISTRIBUTION (non-zero ask_size_1, in $)")
print("=" * 60)
for p in [5, 10, 25, 50, 75, 90, 95, 99]:
    print(f"  p{p:3d}: ${np.percentile(all_ask_sizes, p):8.2f}")
print(f"  mean: ${all_ask_sizes.mean():8.2f}")

# Suggested thresholds
print()
print("=" * 60)
print("SUGGESTED THRESHOLDS")
print("=" * 60)
spread_90 = np.percentile(valid, 90)
spread_95 = np.percentile(valid, 95)
size_25   = np.percentile(all_bid_sizes, 25)
size_10   = np.percentile(all_bid_sizes, 10)
print(f"  MAX_SPREAD for 'healthy' book:  {spread_90:.3f} (p90) or {spread_95:.3f} (p95)")
print(f"  MIN_SIZE for 'healthy' book:    ${size_25:.2f} (p25 bid size) or ${size_10:.2f} (p10)")
print()
print("  Interpretation:")
print(f"  - A spread > {spread_90:.2f} suggests a thin/manipulable book")
print(f"  - A top-of-book size < ${size_25:.2f} suggests a token with no real liquidity")