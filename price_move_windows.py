"""
price_move_windows.py

Characterizes how much the mid-price typically moves within 3-6 second
windows across BTC up/down markets, using 100ms orderbook snapshots.

This is the relevant window for the nonce race exploit — a trader has
roughly 3-6 seconds between CLOB match and on-chain settlement to observe
price movement and decide whether to call incrementNonce().

Usage:
    python price_move_windows.py
    python price_move_windows.py --book-dir data/telonex_15m_100ms --n-markets 200
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--book-dir",  default="data/telonex_15m_100ms")
parser.add_argument("--n-markets", type=int, default="200")
parser.add_argument("--out",       default="data/price_move_windows.parquet")
args = parser.parse_args()

# Windows to measure (in seconds)
WINDOWS_S = [3, 4, 5, 6]
# 100ms snapshots, so 1s = 10 steps
SNAP_MS   = 100

# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------

# Exclude "decided" markets — books thin out, moves are noise not signal
PRICE_MIN = 0.10
PRICE_MAX = 0.90

# Exclude the first N seconds of each market (open noise / resting liq pickup)
OPEN_EXCLUDE_S = 30

# Exclude snapshots with unreliable mid — one-sided or wide spread
MAX_SPREAD    = 0.06   # matches env MAX_SPREAD
MIN_BOOK_SIZE = 20.0   # dollars, at least one side must have this

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def mid_price(row: pd.Series) -> float | None:
    """
    Best-effort mid from bid_price_1 / ask_price_1.
    Falls back to mid_price column if present.
    """
    if "mid_price" in row.index and pd.notna(row["mid_price"]) and row["mid_price"] > 0:
        return float(row["mid_price"])
    bid = row.get("bid_price_1", 0) or 0
    ask = row.get("ask_price_1", 0) or 0
    if bid > 0 and ask > 0 and ask > bid:
        return (bid + ask) / 2.0
    if bid > 0:
        return float(bid)
    if ask > 0:
        return float(ask)
    return None


def extract_mids(df: pd.DataFrame) -> np.ndarray:
    """
    Extract mid-price series with quality filters applied per snapshot:
      - Skip first OPEN_EXCLUDE_S seconds (market open noise)
      - Skip decided markets (price outside PRICE_MIN / PRICE_MAX)
      - Skip one-sided or wide-spread snapshots (unreliable mid)
    """
    if "mid_price" in df.columns:
        mids = df["mid_price"].values.astype(float)
    else:
        mids = df.apply(mid_price, axis=1).values.astype(float)

    n = len(mids)

    # --- Filter 1: market open ---
    open_steps = int(OPEN_EXCLUDE_S * 1000 / SNAP_MS)
    if n > open_steps:
        mids[:open_steps] = np.nan

    # --- Filter 2: decided markets (price at extremes) ---
    decided = (mids < PRICE_MIN) | (mids > PRICE_MAX)
    mids[decided] = np.nan

    # --- Filter 3: unreliable book (one-sided or wide spread) ---
    if "bid_price_1" in df.columns and "ask_price_1" in df.columns:
        bid = df["bid_price_1"].values.astype(float)
        ask = df["ask_price_1"].values.astype(float)
        spread = ask - bid

        # One-sided: bid or ask is zero
        one_sided = (bid <= 0) | (ask <= 0)
        # Spread too wide
        wide = spread > MAX_SPREAD

        mids[one_sided | wide] = np.nan

    # --- Filter 4: minimum book size ---
    if "bid_size_1" in df.columns and "ask_size_1" in df.columns:
        bid_sz = df["bid_size_1"].values.astype(float)
        ask_sz = df["ask_size_1"].values.astype(float)
        thin = (bid_sz < MIN_BOOK_SIZE) & (ask_sz < MIN_BOOK_SIZE)
        mids[thin] = np.nan

    # Final validity mask
    mask = (mids > 0) & (mids < 1) & np.isfinite(mids)
    mids[~mask] = np.nan
    return mids


def compute_moves(mids: np.ndarray, window_steps: int) -> np.ndarray:
    """
    For each snapshot i, compute abs(mid[i + window] - mid[i]).
    Both endpoints must pass all filters (non-NaN) to be included.
    Returns array of absolute price moves.
    """
    n      = len(mids)
    moves  = np.full(n - window_steps, np.nan)
    for i in range(n - window_steps):
        p0 = mids[i]
        p1 = mids[i + window_steps]
        if np.isfinite(p0) and np.isfinite(p1):
            moves[i] = abs(p1 - p0)
    return moves[np.isfinite(moves)]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(Path(args.book_dir).glob("btc-updown-*.parquet"))
    if not files:
        print(f"No files found in {args.book_dir}")
        return

    # Sample evenly across time range
    step   = max(1, len(files) // args.n_markets)
    sample = files[::step][: args.n_markets]
    print(f"Found {len(files)} market files, analyzing {len(sample)}")

    # Accumulate moves per window size
    all_moves = {w: [] for w in WINDOWS_S}
    skipped   = 0

    for i, f in enumerate(sample):
        try:
            df   = pd.read_parquet(f)
            mids = extract_mids(df)
        except Exception as e:
            skipped += 1
            continue

        if np.isfinite(mids).sum() < 60:  # need at least 6s of valid data post-filters
            skipped += 1
            continue

        for w_s in WINDOWS_S:
            steps = int(w_s * 1000 / SNAP_MS)  # seconds → snapshot steps
            moves = compute_moves(mids, steps)
            if len(moves):
                all_moves[w_s].append(moves)

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(sample)} processed...")

    print(f"  Done. Skipped {skipped} files (insufficient data).")

    # ---------------------------------------------------------------------------
    # Results
    # ---------------------------------------------------------------------------

    print(f"\n{'='*60}")
    print(f"Absolute mid-price move distribution by window size")
    print(f"(across {len(sample) - skipped} BTC up/down markets, 100ms snapshots)")
    print(f"{'='*60}")

    summary_rows = []

    for w_s in WINDOWS_S:
        moves = np.concatenate(all_moves[w_s]) if all_moves[w_s] else np.array([])
        if not len(moves):
            print(f"\n{w_s}s window: no data")
            continue

        p = np.percentile(moves, [25, 50, 75, 90, 95, 99])
        mean = moves.mean()

        print(f"\n{w_s}s window  (n={len(moves):,} observations):")
        print(f"  mean:  {mean:.4f}  ({mean*100:.2f} cents)")
        print(f"  p25:   {p[0]:.4f}  ({p[0]*100:.2f} cents)")
        print(f"  p50:   {p[1]:.4f}  ({p[1]*100:.2f} cents)")
        print(f"  p75:   {p[2]:.4f}  ({p[2]*100:.2f} cents)")
        print(f"  p90:   {p[3]:.4f}  ({p[3]*100:.2f} cents)")
        print(f"  p95:   {p[4]:.4f}  ({p[4]*100:.2f} cents)")
        print(f"  p99:   {p[5]:.4f}  ({p[5]*100:.2f} cents)")

        # Profitability thresholds — move must exceed this to be worth exploiting
        print(f"  % moves > 1 cent:  {(moves > 0.01).mean()*100:.1f}%")
        print(f"  % moves > 2 cents: {(moves > 0.02).mean()*100:.1f}%")
        print(f"  % moves > 5 cents: {(moves > 0.05).mean()*100:.1f}%")

        for pct in [25, 50, 75, 90, 95, 99]:
            summary_rows.append({
                "window_s":   w_s,
                "percentile": pct,
                "move":       np.percentile(moves, pct),
            })

    # Save summary
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_parquet(out_path)
    print(f"\nSaved summary → {out_path}")


if __name__ == "__main__":
    main()