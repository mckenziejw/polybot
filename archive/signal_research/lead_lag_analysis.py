#!/usr/bin/env python3
"""
Lead-lag analysis: Does Polymarket BTC Up/Down price move before or after BTC spot?

Uses Telonex orderbook snapshots + BTC 1s quotes from the same time period.
Computes cross-correlation at various lags to determine information flow direction.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MARKET_DURATION_MS = 300_000


def load_btc_quotes(btc_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(btc_path)
    df = df.sort_values("timestamp_ms").reset_index(drop=True)
    return df


def compute_lead_lag_for_market(market_df: pd.DataFrame, btc_df: pd.DataFrame, open_ms: int) -> dict | None:
    """Compute cross-correlation between Polymarket mid changes and BTC price changes."""
    close_ms = open_ms + MARKET_DURATION_MS

    # Get the "Up" token (its mid directly tracks P(BTC goes up))
    up_df = market_df[market_df["token_label"] == "Up"].sort_values("exchange_timestamp")
    if len(up_df) < 20:
        return None

    # Filter to market window
    up_df = up_df[(up_df["exchange_timestamp"] >= open_ms) & (up_df["exchange_timestamp"] < close_ms)]
    if len(up_df) < 20:
        return None

    # Resample both to 1-second bars aligned to market open
    # Polymarket: take last mid per second
    up_df = up_df.copy()
    up_df["sec"] = ((up_df["exchange_timestamp"] - open_ms) / 1000).astype(int)
    poly_1s = up_df.groupby("sec")["mid_price"].last()

    # BTC: already 1s bars, align to same second grid
    btc_window = btc_df[
        (btc_df["timestamp_ms"] >= open_ms) &
        (btc_df["timestamp_ms"] < close_ms)
    ].copy()
    if len(btc_window) < 20:
        return None

    btc_window["sec"] = ((btc_window["timestamp_ms"] - open_ms) / 1000).astype(int)
    btc_1s = btc_window.groupby("sec")["mid_price"].last()

    # Align on common seconds
    common_secs = sorted(set(poly_1s.index) & set(btc_1s.index))
    if len(common_secs) < 30:
        return None

    poly_aligned = poly_1s.reindex(common_secs).ffill().values
    btc_aligned = btc_1s.reindex(common_secs).ffill().values

    # Compute returns (first differences)
    poly_ret = np.diff(poly_aligned)
    btc_ret = np.diff(btc_aligned)

    if len(poly_ret) < 20 or np.std(poly_ret) == 0 or np.std(btc_ret) == 0:
        return None

    # Normalize
    poly_ret = (poly_ret - poly_ret.mean()) / poly_ret.std()
    btc_ret = (btc_ret - btc_ret.mean()) / btc_ret.std()

    # Cross-correlation at lags -10s to +10s
    # Positive lag = BTC leads (BTC moves first, Poly follows)
    # Negative lag = Poly leads (Poly moves first, BTC follows)
    max_lag = 10
    n = len(poly_ret)
    correlations = {}

    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            # BTC at time t vs Poly at time t+lag (BTC leads by `lag` seconds)
            btc_slice = btc_ret[:n - lag] if lag > 0 else btc_ret
            poly_slice = poly_ret[lag:] if lag > 0 else poly_ret
        else:
            # Poly at time t vs BTC at time t+|lag| (Poly leads by |lag| seconds)
            abs_lag = abs(lag)
            poly_slice = poly_ret[:n - abs_lag]
            btc_slice = btc_ret[abs_lag:]

        if len(btc_slice) < 10:
            continue
        correlations[lag] = np.corrcoef(btc_slice, poly_slice)[0, 1]

    return correlations


def main():
    btc_path = Path("data/btc_quotes/btcusdt_quotes.parquet")
    data_dir = Path("data/telonex_book_snapshots")
    resolution_path = Path("data/resolution_cache.json")

    print("Lead-Lag Analysis: Polymarket vs BTC Spot")
    print("=" * 60)

    btc_df = load_btc_quotes(btc_path)
    btc_min_ms = btc_df["timestamp_ms"].min()
    btc_max_ms = btc_df["timestamp_ms"].max()
    print(f"BTC quotes: {len(btc_df)} rows")

    parquet_files = sorted(data_dir.glob("btc-updown-5m-*.parquet"))
    print(f"Market files: {len(parquet_files)}")

    # Collect cross-correlations across all markets
    all_corrs = {lag: [] for lag in range(-10, 11)}
    n_processed = 0
    n_skipped = 0

    for i, pf in enumerate(parquet_files):
        slug = pf.stem
        try:
            open_ts = int(slug.split("-")[-1])
        except ValueError:
            continue

        open_ms = open_ts * 1000
        if open_ms < btc_min_ms or open_ms > btc_max_ms:
            continue

        try:
            df = pd.read_parquet(pf)
        except Exception:
            continue

        if "token_label" not in df.columns or "mid_price" not in df.columns:
            continue

        corrs = compute_lead_lag_for_market(df, btc_df, open_ms)
        if corrs is None:
            n_skipped += 1
            continue

        for lag, c in corrs.items():
            if np.isfinite(c):
                all_corrs[lag].append(c)

        n_processed += 1
        if (i + 1) % 500 == 0:
            print(f"  Processed {i+1}, analyzed {n_processed} markets...")

    print(f"\nAnalyzed {n_processed} markets (skipped {n_skipped})")

    # Aggregate results
    print(f"\n{'=' * 60}")
    print("CROSS-CORRELATION: BTC returns vs Polymarket returns")
    print(f"{'=' * 60}")
    print(f"  Positive lag = BTC moves first, Poly follows (BTC leads)")
    print(f"  Negative lag = Poly moves first, BTC follows (Poly leads)")
    print(f"  Lag 0 = simultaneous movement")
    print()
    print(f"  {'Lag':>5s}  {'Mean Corr':>10s}  {'Median':>8s}  {'Std':>8s}  {'N':>6s}  {'Bar'}")
    print("-" * 70)

    peak_lag = 0
    peak_corr = -1

    for lag in range(-10, 11):
        vals = all_corrs[lag]
        if not vals:
            continue
        mean_c = np.mean(vals)
        median_c = np.median(vals)
        std_c = np.std(vals)
        n = len(vals)

        # Track peak
        if abs(mean_c) > abs(peak_corr):
            peak_corr = mean_c
            peak_lag = lag

        bar_len = int(abs(mean_c) * 200)
        if mean_c >= 0:
            bar = " " * 20 + "#" * bar_len
        else:
            bar = " " * max(0, 20 - bar_len) + "#" * bar_len

        marker = " <<<" if lag == 0 else ""
        print(f"  {lag:>+4d}s  {mean_c:>+9.4f}  {median_c:>+7.4f}  {std_c:>7.4f}  {n:>6d}  {bar}{marker}")

    print()
    if peak_lag > 0:
        print(f"  Peak correlation at lag {peak_lag:+d}s → BTC LEADS Polymarket by ~{peak_lag}s")
    elif peak_lag < 0:
        print(f"  Peak correlation at lag {peak_lag:+d}s → Polymarket LEADS BTC by ~{abs(peak_lag)}s")
    else:
        print(f"  Peak correlation at lag 0 → movements are SIMULTANEOUS")

    # Granger-style test: does adding lagged BTC improve Poly prediction?
    print(f"\n{'=' * 60}")
    print("PREDICTIVE POWER: Can lagged BTC returns predict Poly returns?")
    print(f"{'=' * 60}")

    # Collect all aligned 1s return pairs across markets for regression
    print("\n  Collecting aligned return series for regression...")

    all_poly_ret = []
    all_btc_ret_lag1 = []
    all_btc_ret_lag2 = []
    all_btc_ret_lag3 = []
    all_btc_ret_lag5 = []
    all_poly_ret_lag1 = []

    for pf in parquet_files[:2000]:  # subset for speed
        slug = pf.stem
        try:
            open_ts = int(slug.split("-")[-1])
        except ValueError:
            continue
        open_ms = open_ts * 1000
        if open_ms < btc_min_ms or open_ms > btc_max_ms:
            continue

        try:
            df = pd.read_parquet(pf)
        except Exception:
            continue

        if "token_label" not in df.columns:
            continue

        close_ms = open_ms + MARKET_DURATION_MS
        up_df = df[(df["token_label"] == "Up") &
                    (df["exchange_timestamp"] >= open_ms) &
                    (df["exchange_timestamp"] < close_ms)].sort_values("exchange_timestamp")

        if len(up_df) < 30:
            continue

        up_df = up_df.copy()
        up_df["sec"] = ((up_df["exchange_timestamp"] - open_ms) / 1000).astype(int)
        poly_1s = up_df.groupby("sec")["mid_price"].last()

        btc_window = btc_df[
            (btc_df["timestamp_ms"] >= open_ms) &
            (btc_df["timestamp_ms"] < close_ms)
        ].copy()
        if len(btc_window) < 30:
            continue

        btc_window["sec"] = ((btc_window["timestamp_ms"] - open_ms) / 1000).astype(int)
        btc_1s = btc_window.groupby("sec")["mid_price"].last()

        common = sorted(set(poly_1s.index) & set(btc_1s.index))
        if len(common) < 30:
            continue

        poly_vals = poly_1s.reindex(range(max(common) + 1)).ffill().values
        btc_vals = btc_1s.reindex(range(max(common) + 1)).ffill().values

        if len(poly_vals) < 10 or len(btc_vals) < 10:
            continue

        poly_r = np.diff(poly_vals)
        btc_r = np.diff(btc_vals)

        n = min(len(poly_r), len(btc_r))
        if n < 8:
            continue

        poly_r = poly_r[:n]
        btc_r = btc_r[:n]

        # Collect lagged pairs (need at least 5 lags)
        for t in range(5, n):
            all_poly_ret.append(poly_r[t])
            all_btc_ret_lag1.append(btc_r[t - 1])
            all_btc_ret_lag2.append(btc_r[t - 2])
            all_btc_ret_lag3.append(btc_r[t - 3])
            all_btc_ret_lag5.append(btc_r[t - 5])
            all_poly_ret_lag1.append(poly_r[t - 1])

    all_poly_ret = np.array(all_poly_ret)
    all_btc_ret_lag1 = np.array(all_btc_ret_lag1)
    all_btc_ret_lag2 = np.array(all_btc_ret_lag2)
    all_btc_ret_lag3 = np.array(all_btc_ret_lag3)
    all_btc_ret_lag5 = np.array(all_btc_ret_lag5)
    all_poly_ret_lag1 = np.array(all_poly_ret_lag1)

    print(f"  Collected {len(all_poly_ret):,} aligned return observations")

    if len(all_poly_ret) > 100:
        # Simple correlation: does BTC(t-k) predict Poly(t)?
        print(f"\n  Correlation: BTC_return(t-k) → Poly_return(t)")
        for name, lagged in [("BTC(t-1)", all_btc_ret_lag1),
                              ("BTC(t-2)", all_btc_ret_lag2),
                              ("BTC(t-3)", all_btc_ret_lag3),
                              ("BTC(t-5)", all_btc_ret_lag5),
                              ("Poly(t-1)", all_poly_ret_lag1)]:
            mask = np.isfinite(lagged) & np.isfinite(all_poly_ret)
            if mask.sum() > 100:
                c = np.corrcoef(lagged[mask], all_poly_ret[mask])[0, 1]
                print(f"    {name:>10s} → Poly(t):  r = {c:+.4f}")

        # And reverse: does Poly(t-k) predict BTC(t)?
        print(f"\n  Correlation: Poly_return(t-k) → BTC_return(t)")
        # Need to rebuild for reverse direction
        all_btc_ret_t = []
        all_poly_lag1_for_btc = []
        all_poly_lag2_for_btc = []
        all_poly_lag3_for_btc = []

        for t in range(5, len(all_poly_ret)):
            # These arrays are already aligned by time within each market
            pass

        # Simpler: just use the correlation of Poly(t-1) with BTC(t)
        # which is the same data rearranged
        # BTC(t) vs Poly(t-1): does Poly lead BTC?
        n = min(len(all_poly_ret), len(all_btc_ret_lag1))
        # all_btc_ret_lag1[i] = btc_r[t-1], all_poly_ret[i] = poly_r[t]
        # We want: poly_r[t-1] vs btc_r[t]
        # That's: all_poly_ret_lag1[i] = poly_r[t-1] vs btc_r at same t
        # btc_r at t = we need the concurrent btc return, not lagged
        # Actually all_btc_ret_lag1 is btc(t-1). We want btc(t).
        # btc(t) for observation i corresponds to the same t as poly(t)
        # We don't have btc(t) directly stored, but we can get it from
        # btc_lag1 shifted: btc(t) at observation i = btc_lag1 at observation i+1
        if len(all_btc_ret_lag1) > 100:
            btc_t = all_btc_ret_lag1[1:]  # btc(t) = btc_lag1 shifted by 1
            poly_lag1 = all_poly_ret_lag1[1:]  # poly(t-1) for same observations
            poly_lag1_clean = poly_lag1[:-1] if len(poly_lag1) > len(btc_t) else poly_lag1
            btc_t_clean = btc_t[:len(poly_lag1_clean)]

            mask = np.isfinite(poly_lag1_clean) & np.isfinite(btc_t_clean)
            if mask.sum() > 100:
                c = np.corrcoef(poly_lag1_clean[mask], btc_t_clean[mask])[0, 1]
                print(f"    Poly(t-1) → BTC(t):   r = {c:+.4f}")
                if abs(c) > 0.01:
                    print(f"    → Polymarket {'LEADS' if c > 0 else 'INVERSELY LEADS'} BTC by ~1s")
                else:
                    print(f"    → No significant lead-lag relationship")

    print(f"\n{'=' * 60}")
    print("CONCLUSION")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
