#!/usr/bin/env python3
"""
Exploratory analysis of orderbook/orderflow signals for BTC Up/Down 5-min markets.

Computes features from book snapshots and trade events, checks which predict resolution.
"""

import os
import json
import time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import urllib.request
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

BOOK_DIR = Path("/home/josh/Documents/projects/wss_test/data/book_snapshots")
TRADE_DIR = Path("/home/josh/Documents/projects/wss_test/data/trade_events")
GAMMA_API = "https://gamma-api.polymarket.com"
RESOLUTION_CACHE = Path("/home/josh/Documents/projects/wss_test/data/resolution_cache.json")

def parse_slug_timestamp(filename: str) -> int:
    """Extract the unix timestamp from filename like btc-updown-5m-1771346400.parquet"""
    return int(filename.replace("btc-updown-5m-", "").replace(".parquet", ""))


def fetch_resolution(slug: str) -> dict | None:
    """Fetch resolution from Gamma API. Returns {winnerLabel, winningTokenId} or None."""
    try:
        url = f"{GAMMA_API}/markets/slug/{slug}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
            token_ids = json.loads(data["clobTokenIds"]) if isinstance(data.get("clobTokenIds"), str) else data.get("clobTokenIds", [])
            outcomes = json.loads(data["outcomes"]) if isinstance(data.get("outcomes"), str) else data.get("outcomes", [])
            prices = json.loads(data["outcomePrices"]) if isinstance(data.get("outcomePrices"), str) else data.get("outcomePrices", [])

            winning_idx = None
            for i, p in enumerate(prices):
                if float(p) == 1.0:
                    winning_idx = i
                    break
            if winning_idx is None:
                return None
            return {
                "winnerLabel": outcomes[winning_idx],
                "winningTokenId": str(token_ids[winning_idx]),
            }
    except Exception:
        return None


def load_or_fetch_resolutions(slugs: list[str]) -> dict[str, dict]:
    """Load cached resolutions, fetch missing ones, save cache."""
    cache = {}
    if RESOLUTION_CACHE.exists():
        cache = json.loads(RESOLUTION_CACHE.read_text())

    missing = [s for s in slugs if s not in cache]
    if missing:
        print(f"  Fetching {len(missing)} resolutions from Gamma API...")
        batch_size = 20
        for batch_start in range(0, len(missing), batch_size):
            batch = missing[batch_start:batch_start + batch_size]
            with ThreadPoolExecutor(max_workers=10) as pool:
                futures = {pool.submit(fetch_resolution, slug): slug for slug in batch}
                for fut in as_completed(futures):
                    slug = futures[fut]
                    result = fut.result()
                    if result:
                        cache[slug] = result
                    else:
                        cache[slug] = None  # Mark as unfetchable
            if batch_start % 200 == 0 and batch_start > 0:
                print(f"    {batch_start}/{len(missing)} fetched...")
            time.sleep(0.1)  # Light rate limiting between batches

        # Save cache
        RESOLUTION_CACHE.write_text(json.dumps(cache, indent=2))
        resolved = sum(1 for v in cache.values() if v is not None)
        print(f"  Cache updated: {resolved} resolved, {len(cache) - resolved} unresolvable")

    return cache


def identify_tokens(book_df: pd.DataFrame, market_ts: int, resolution: dict | None) -> dict | None:
    """
    Identify winner/loser tokens from book snapshots + API resolution.
    Uses the winning token ID from the API to label tokens.
    """
    if resolution is None:
        return None

    if len(book_df) < 20:
        return None

    market_open_ms = market_ts * 1000
    market_close_ms = market_open_ms + 300_000  # 5 minutes

    # Filter to within market window
    df = book_df[
        (book_df.exchange_timestamp >= market_open_ms - 5000) &
        (book_df.exchange_timestamp <= market_close_ms + 5000)
    ].copy()

    if len(df) < 10:
        return None

    # Filter to real tradeable tokens — those with actual orderbook depth
    real_assets = []
    for aid in df.asset_id.unique():
        subset = df[df.asset_id == aid]
        has_orders = (subset.bid_size_1 > 0).any() or (subset.ask_size_1 > 0).any()
        if has_orders:
            real_assets.append((aid, len(subset)))

    if len(real_assets) < 2:
        return None

    # Pick the two with most rows
    real_assets.sort(key=lambda x: x[1], reverse=True)
    a1, a2 = real_assets[0][0], real_assets[1][0]

    # Match against API resolution (handle both cache formats)
    winning_token_id = resolution.get("winningTokenId") or resolution.get("winning_token")
    if not winning_token_id:
        return None
    winner_label = resolution.get("winnerLabel", "Unknown")
    if a1 == winning_token_id:
        return {"winner_asset": a1, "loser_asset": a2, "winner_label": winner_label}
    elif a2 == winning_token_id:
        return {"winner_asset": a2, "loser_asset": a1, "winner_label": winner_label}
    else:
        # Token IDs don't match — could be a different market overlap
        return None


def compute_features(book_df: pd.DataFrame, trade_df: pd.DataFrame,
                     winner_asset: str, loser_asset: str, market_ts: int) -> dict | None:
    """
    Compute orderbook and orderflow features at various time points.
    All features are computed from the winner token's perspective.
    """
    market_open_ms = market_ts * 1000
    market_close_ms = market_open_ms + 300_000

    # Filter to market window
    books = book_df[
        (book_df.exchange_timestamp >= market_open_ms) &
        (book_df.exchange_timestamp <= market_close_ms)
    ].copy()
    books["elapsed_s"] = (books.exchange_timestamp - market_open_ms) / 1000

    trades = trade_df[
        (trade_df.exchange_timestamp >= market_open_ms) &
        (trade_df.exchange_timestamp <= market_close_ms)
    ].copy()
    trades["elapsed_s"] = (trades.exchange_timestamp - market_open_ms) / 1000

    # Winner token books and trades
    w_books = books[books.asset_id == winner_asset].sort_values("exchange_timestamp")
    l_books = books[books.asset_id == loser_asset].sort_values("exchange_timestamp")
    w_trades = trades[trades.asset_id == winner_asset]
    l_trades = trades[trades.asset_id == loser_asset]

    if len(w_books) < 5 or len(l_books) < 5:
        return None

    features = {}

    # === Mid-price features at various time points ===
    for t_sec in [10, 30, 60, 120, 180, 240]:
        w_at_t = w_books[w_books.elapsed_s <= t_sec]
        l_at_t = l_books[l_books.elapsed_s <= t_sec]
        if len(w_at_t) > 0 and len(l_at_t) > 0:
            features[f"winner_mid_{t_sec}s"] = w_at_t.iloc[-1].mid_price
            features[f"loser_mid_{t_sec}s"] = l_at_t.iloc[-1].mid_price
            features[f"mid_sum_{t_sec}s"] = w_at_t.iloc[-1].mid_price + l_at_t.iloc[-1].mid_price

    # === Mid-price drift (change from 0.5) ===
    for t_sec in [10, 30, 60, 120]:
        key = f"winner_mid_{t_sec}s"
        if key in features:
            features[f"winner_drift_{t_sec}s"] = features[key] - 0.5

    # === Book imbalance (bid_depth / (bid_depth + ask_depth)) on winner token ===
    for t_sec in [10, 30, 60, 120]:
        w_at_t = w_books[w_books.elapsed_s <= t_sec]
        if len(w_at_t) > 0:
            row = w_at_t.iloc[-1]
            bid_depth = sum(row[f"bid_size_{i}"] for i in range(1, 11))
            ask_depth = sum(row[f"ask_size_{i}"] for i in range(1, 11))
            total = bid_depth + ask_depth
            features[f"winner_book_imbal_{t_sec}s"] = bid_depth / total if total > 0 else 0.5

    # === Spread on winner token ===
    for t_sec in [10, 30, 60]:
        w_at_t = w_books[w_books.elapsed_s <= t_sec]
        if len(w_at_t) > 0:
            features[f"winner_spread_{t_sec}s"] = w_at_t.iloc[-1].spread

    # === Trade flow imbalance: aggressive buy volume vs sell volume ===
    for t_sec in [30, 60, 120, 180]:
        wt = w_trades[w_trades.elapsed_s <= t_sec]
        if len(wt) > 0:
            buy_vol = wt[wt.side == "BUY"]["size"].sum()
            sell_vol = wt[wt.side == "SELL"]["size"].sum()
            total_vol = buy_vol + sell_vol
            features[f"winner_flow_imbal_{t_sec}s"] = (buy_vol - sell_vol) / total_vol if total_vol > 0 else 0
            features[f"winner_buy_vol_{t_sec}s"] = buy_vol
            features[f"winner_sell_vol_{t_sec}s"] = sell_vol
            features[f"winner_trade_count_{t_sec}s"] = len(wt)

    # === Cross-token flow: net buying on winner minus net buying on loser ===
    for t_sec in [30, 60, 120]:
        wt = w_trades[w_trades.elapsed_s <= t_sec]
        lt = l_trades[l_trades.elapsed_s <= t_sec]
        w_net = (wt[wt.side == "BUY"]["size"].sum() - wt[wt.side == "SELL"]["size"].sum()) if len(wt) > 0 else 0
        l_net = (lt[lt.side == "BUY"]["size"].sum() - lt[lt.side == "SELL"]["size"].sum()) if len(lt) > 0 else 0
        features[f"cross_flow_diff_{t_sec}s"] = w_net - l_net

    # === Depth ratio: winner bid depth / loser bid depth ===
    for t_sec in [30, 60, 120]:
        w_at_t = w_books[w_books.elapsed_s <= t_sec]
        l_at_t = l_books[l_books.elapsed_s <= t_sec]
        if len(w_at_t) > 0 and len(l_at_t) > 0:
            w_bid = sum(w_at_t.iloc[-1][f"bid_size_{i}"] for i in range(1, 11))
            l_bid = sum(l_at_t.iloc[-1][f"bid_size_{i}"] for i in range(1, 11))
            features[f"depth_ratio_{t_sec}s"] = w_bid / l_bid if l_bid > 0 else 0

    # === Mid-price velocity (slope of mid over window) ===
    for t_sec in [30, 60]:
        w_at_t = w_books[w_books.elapsed_s <= t_sec]
        if len(w_at_t) >= 3:
            first_mid = w_at_t.iloc[0].mid_price
            last_mid = w_at_t.iloc[-1].mid_price
            dt = w_at_t.iloc[-1].elapsed_s - w_at_t.iloc[0].elapsed_s
            if dt > 0:
                features[f"winner_mid_velocity_{t_sec}s"] = (last_mid - first_mid) / dt

    return features


def main():
    book_files = sorted(BOOK_DIR.glob("btc-updown-5m-*.parquet"))
    trade_files = {f.stem: f for f in TRADE_DIR.glob("btc-updown-5m-*.parquet")}

    print(f"Found {len(book_files)} book snapshot files, {len(trade_files)} trade event files")

    # Fetch/load all resolutions
    slugs = [bf.stem for bf in book_files]
    resolutions = load_or_fetch_resolutions(slugs)

    results = []
    skipped = {"no_trades": 0, "no_tokens": 0, "no_features": 0, "sparse": 0, "no_resolution": 0}

    for i, bf in enumerate(book_files):
        if i % 500 == 0:
            print(f"  Processing {i}/{len(book_files)}...")

        stem = bf.stem
        market_ts = parse_slug_timestamp(stem + ".parquet")

        # Check resolution
        resolution = resolutions.get(stem)
        if resolution is None:
            skipped["no_resolution"] += 1
            continue

        # Load book snapshots
        try:
            book_df = pq.read_table(bf).to_pandas()
        except Exception:
            continue

        if len(book_df) < 20:
            skipped["sparse"] += 1
            continue

        # Identify winner/loser using API resolution
        tokens = identify_tokens(book_df, market_ts, resolution)
        if tokens is None:
            skipped["no_tokens"] += 1
            continue

        # Load trade events
        trade_file = trade_files.get(stem)
        if trade_file is None:
            trade_df = pd.DataFrame(columns=["exchange_timestamp", "asset_id", "side", "size", "trade_price"])
            skipped["no_trades"] += 1
        else:
            trade_df = pq.read_table(trade_file).to_pandas()

        # Compute features
        feats = compute_features(book_df, trade_df, tokens["winner_asset"], tokens["loser_asset"], market_ts)
        if feats is None:
            skipped["no_features"] += 1
            continue

        feats["slug"] = stem
        feats["market_ts"] = market_ts
        feats["winner_label"] = tokens["winner_label"]
        results.append(feats)

    print(f"\nProcessed: {len(results)} markets with features")
    print(f"Skipped: {skipped}")

    if len(results) == 0:
        print("No results to analyze!")
        return

    df = pd.DataFrame(results)

    # === Analysis ===
    # For each feature at each time point, compute how well it predicts the winner
    # Since all features are computed from the winner's perspective, we check if
    # the feature value is distinguishable from 0.5 (random) at early time points

    print("\n" + "=" * 80)
    print("FEATURE ANALYSIS — Can we predict the winner from early orderbook signals?")
    print("=" * 80)
    print("\nAll features computed from WINNER token perspective (label = 1).")
    print("A useful signal means the feature is consistently > 0.5 or trending in one direction.")
    print("We need to check: if we see this signal live, can we identify the winner?\n")

    # Key question: at time T, does the winner's token already show distinctive patterns?
    # If winner_mid_30s is consistently > 0.5, then mid-price drift predicts outcome.

    feature_groups = {
        "Mid-price drift": [f"winner_drift_{t}s" for t in [10, 30, 60, 120]],
        "Book imbalance": [f"winner_book_imbal_{t}s" for t in [10, 30, 60, 120]],
        "Trade flow imbalance": [f"winner_flow_imbal_{t}s" for t in [30, 60, 120, 180]],
        "Cross-token flow diff": [f"cross_flow_diff_{t}s" for t in [30, 60, 120]],
        "Depth ratio": [f"depth_ratio_{t}s" for t in [30, 60, 120]],
        "Mid-price velocity": [f"winner_mid_velocity_{t}s" for t in [30, 60]],
        "Winner spread": [f"winner_spread_{t}s" for t in [10, 30, 60]],
    }

    for group_name, feature_names in feature_groups.items():
        print(f"\n--- {group_name} ---")
        for feat in feature_names:
            if feat not in df.columns:
                continue
            vals = df[feat].dropna()
            if len(vals) < 100:
                continue
            mean = vals.mean()
            std = vals.std()
            median = vals.median()
            # What fraction of time is the feature "positive" (> neutral)?
            if "imbal" in feat or "drift" in feat or "velocity" in feat or "flow" in feat:
                neutral = 0.0 if "drift" in feat or "velocity" in feat or "flow" in feat else 0.5
                frac_positive = (vals > neutral).mean()
                print(f"  {feat:35s}  mean={mean:+.4f}  median={median:+.4f}  std={std:.4f}  "
                      f"frac>{neutral}={frac_positive:.3f}  n={len(vals)}")
            elif "depth_ratio" in feat:
                frac_gt1 = (vals > 1.0).mean()
                print(f"  {feat:35s}  mean={mean:.4f}  median={median:.4f}  std={std:.4f}  "
                      f"frac>1={frac_gt1:.3f}  n={len(vals)}")
            elif "spread" in feat:
                print(f"  {feat:35s}  mean={mean:.4f}  median={median:.4f}  std={std:.4f}  n={len(vals)}")

    # === Binned analysis: does stronger signal = higher accuracy? ===
    print("\n" + "=" * 80)
    print("SIGNAL STRENGTH vs ACCURACY")
    print("=" * 80)
    print("\nFor mid-price drift: if we observe drift > X at time T, how often is that the winner?")
    print("(Since we computed features from the winner's perspective, accuracy = frac with drift > 0)\n")

    for t_sec in [10, 30, 60, 120]:
        feat = f"winner_drift_{t_sec}s"
        if feat not in df.columns:
            continue
        vals = df[feat].dropna()
        print(f"  At {t_sec}s — drift distribution:")
        for threshold in [0.0, 0.005, 0.01, 0.02, 0.05, 0.10]:
            n_above = (vals > threshold).sum()
            n_below = (vals < -threshold).sum()
            # n_above = markets where winner drifted above threshold (correct signal)
            # n_below = markets where winner drifted below -threshold (wrong signal — loser was drifting up)
            total = n_above + n_below
            if total > 0:
                accuracy = n_above / total
                print(f"    |drift| > {threshold:.3f}: {n_above} correct, {n_below} wrong, "
                      f"accuracy={accuracy:.1%}  (n={total}, {total/len(vals):.0%} of markets)")

    # === Trade flow analysis ===
    print("\n" + "=" * 80)
    print("TRADE FLOW ANALYSIS")
    print("=" * 80)

    for t_sec in [30, 60, 120]:
        feat = f"winner_flow_imbal_{t_sec}s"
        if feat not in df.columns:
            continue
        vals = df[feat].dropna()
        print(f"\n  Flow imbalance at {t_sec}s:")
        for threshold in [0.0, 0.1, 0.2, 0.3, 0.5]:
            n_pos = (vals > threshold).sum()
            n_neg = (vals < -threshold).sum()
            total = n_pos + n_neg
            if total > 0:
                accuracy = n_pos / total
                print(f"    |flow_imbal| > {threshold:.1f}: {n_pos} correct, {n_neg} wrong, "
                      f"accuracy={accuracy:.1%}  (n={total}, {total/len(vals):.0%} of markets)")


    # === Tradeability analysis ===
    # If we observe drift > X at time T, we'd buy at the ask price at that moment.
    # What's the expected PnL?
    print("\n" + "=" * 80)
    print("TRADEABILITY — Entry price vs expected payoff")
    print("=" * 80)
    print("\nIf we observe winner_mid > 0.5 + threshold at time T, we buy at the ask.")
    print("Win pays $1/token, lose pays $0. Cost = ask price. Net = (accuracy * 1) - ask.\n")

    for t_sec in [30, 60, 120]:
        mid_feat = f"winner_mid_{t_sec}s"
        drift_feat = f"winner_drift_{t_sec}s"
        if mid_feat not in df.columns:
            continue

        print(f"\n  === Entry at {t_sec}s ===")
        for threshold in [0.005, 0.01, 0.02, 0.05, 0.10]:
            # Markets where winner drifted above threshold
            winners_above = df[df[drift_feat] > threshold]
            # Markets where loser drifted above threshold (these look like "winner" to us live)
            losers_above = df[df[drift_feat] < -threshold]

            n_correct = len(winners_above)
            n_wrong = len(losers_above)
            total = n_correct + n_wrong
            if total < 50:
                continue

            accuracy = n_correct / total

            # Average entry price (mid + half spread as proxy for ask)
            avg_winner_mid = winners_above[mid_feat].mean()
            avg_loser_mid = losers_above[f"loser_mid_{t_sec}s"].mean() if f"loser_mid_{t_sec}s" in df.columns else 0.5

            # When we're right, we paid ~winner_mid + spread/2 for a token worth $1
            # When we're wrong, we paid ~(1 - loser_mid) + spread/2 for a token worth $0
            # (because we thought the loser was the winner, so we bought at its ask)
            avg_entry_correct = avg_winner_mid + 0.005  # half-spread estimate
            avg_entry_wrong = (1 - avg_loser_mid) + 0.005

            expected_pnl_per_token = accuracy * (1 - avg_entry_correct) - (1 - accuracy) * avg_entry_wrong

            print(f"    drift>{threshold:.3f}: acc={accuracy:.1%} n={total} "
                  f"avg_entry_win={avg_entry_correct:.3f} avg_entry_lose={avg_entry_wrong:.3f} "
                  f"E[PnL/token]={expected_pnl_per_token:+.4f}")

    # === Time-of-entry analysis: what if we enter at the ask right when we see the signal? ===
    print("\n" + "=" * 80)
    print("REALISTIC PnL — Using actual ask prices from book data")
    print("=" * 80)
    print("\nFor each market, if the leading token at time T has drift > threshold,")
    print("we buy at that token's ask_price_1. Payout = $1 if winner, $0 if loser.\n")

    for t_sec in [30, 60, 120]:
        ask_feat = f"winner_ask1_{t_sec}s"
        drift_feat = f"winner_drift_{t_sec}s"
        # We need to add ask price to features — for now estimate from mid + spread/2
        if drift_feat not in df.columns:
            continue

    # === Consecutive loss analysis ===
    print("\n" + "=" * 80)
    print("STREAK ANALYSIS — Worst-case consecutive losses")
    print("=" * 80)

    # Sort by market timestamp, apply strategy, track streaks
    df_sorted = df.sort_values("market_ts")
    for t_sec in [30, 60]:
        drift_feat = f"winner_drift_{t_sec}s"
        if drift_feat not in df_sorted.columns:
            continue

        for threshold in [0.02, 0.05]:
            results_seq = []
            for _, row in df_sorted.iterrows():
                drift = row.get(drift_feat, 0)
                if pd.isna(drift):
                    continue
                if abs(drift) > threshold:
                    results_seq.append(drift > 0)  # True = correct

            if len(results_seq) < 20:
                continue

            # Compute max consecutive losses
            max_loss_streak = 0
            current_streak = 0
            for r in results_seq:
                if not r:
                    current_streak += 1
                    max_loss_streak = max(max_loss_streak, current_streak)
                else:
                    current_streak = 0

            wins = sum(results_seq)
            print(f"  {drift_feat} > {threshold}: {wins}/{len(results_seq)} wins ({wins/len(results_seq):.1%}), "
                  f"max_loss_streak={max_loss_streak}")


if __name__ == "__main__":
    main()
