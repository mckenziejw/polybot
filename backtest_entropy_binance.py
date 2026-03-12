"""
Combined Conditional Entropy + Binance CEX Price Signal Backtest
================================================================
CORRECTED: Entry price is the alt ask at t=signal_delay (not median over 0-30s).

The key insight: if we wait 30s to observe BTC direction on Binance,
the alt books have ALREADY repriced by then. We must enter at the
ask price that exists at t=30s, not at market open.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

# ==============================================================================
# Configuration
# ==============================================================================
DATA_DIR = Path("data")
ASSETS = ["btc", "eth", "sol", "xrp"]
ALT_ASSETS = ["eth", "sol", "xrp"]
WINDOW_SIZE = 50
ENTROPY_THRESHOLD = 0.5
ENTRY_PRICE_CAP = 0.85
TRADE_SIZE = 10  # tokens per trade

# Sweep parameters
SIGNAL_DELAYS_S = [5, 10, 15, 30, 60]
MIN_RET_THRESHOLDS = [0.0, 0.0001, 0.0003, 0.0005, 0.001]

# Polymarket fee function
def polymarket_fee(p):
    return 0.0222 * p * 0.25 * (p * (1 - p)) ** 2


# ==============================================================================
# Step 1: Resolution tables
# ==============================================================================
def load_btc_resolutions():
    with open(DATA_DIR / "resolution_cache.json") as f:
        rc = json.load(f)
    resolutions = {}
    for slug, info in rc.items():
        if not slug.startswith("btc-"):
            continue
        epoch = int(slug.split("-")[-1])
        winner = None
        if "winnerLabel" in info:
            winner = info["winnerLabel"]
        elif "clobTokenIds" in info and "outcomes" in info and "winningTokenId" in info:
            try:
                idx = info["clobTokenIds"].index(info["winningTokenId"])
                winner = info["outcomes"][idx]
            except (ValueError, IndexError):
                pass
        if winner:
            resolutions[epoch] = winner
    return resolutions


def infer_resolution_from_file(filepath):
    try:
        df = pd.read_parquet(filepath, columns=["slug", "token_label", "exchange_timestamp", "bid_price_1"])
    except Exception:
        return None, None
    if df.empty:
        return None, None
    slug = df["slug"].iloc[0]
    max_ts = df["exchange_timestamp"].max()
    final = df[df["exchange_timestamp"] >= max_ts - 5000]
    up_final = final[final["token_label"] == "Up"]["bid_price_1"]
    down_final = final[final["token_label"] == "Down"]["bid_price_1"]
    up_bid = up_final.mean() if len(up_final) > 0 else 0.5
    down_bid = down_final.mean() if len(down_final) > 0 else 0.5
    if up_bid > 0.8:
        return slug, "Up"
    elif down_bid > 0.8:
        return slug, "Down"
    return slug, None


def build_resolution_table(asset):
    data_dir = DATA_DIR / f"telonex_{asset}_5m_book_snapshots"
    if not data_dir.exists():
        return {}
    files = sorted(data_dir.iterdir())
    resolutions = {}
    for i, f in enumerate(files):
        if f.suffix != ".parquet":
            continue
        slug, winner = infer_resolution_from_file(f)
        if slug and winner:
            epoch = int(slug.split("-")[-1])
            resolutions[epoch] = winner
        if (i + 1) % 1000 == 0:
            print(f"  {asset}: processed {i+1}/{len(files)} files")
    print(f"  {asset}: {len(resolutions)} resolved out of {len(files)} files")
    return resolutions


# ==============================================================================
# Step 2: Conditional entropy
# ==============================================================================
def compute_conditional_entropy(joint_outcomes, target_asset, conditioning_assets, window):
    n = len(joint_outcomes)
    entropies = np.full(n, np.nan)
    target_vals = joint_outcomes[target_asset].values
    cond_state = np.zeros(n, dtype=int)
    for i, c in enumerate(conditioning_assets):
        cond_state += joint_outcomes[c].values.astype(int) * (2 ** i)

    for idx in range(window, n):
        start = idx - window
        t_window = target_vals[start:idx]
        c_window = cond_state[start:idx]
        total = len(t_window)
        joint_keys = t_window * 8 + c_window
        _, joint_counts = np.unique(joint_keys, return_counts=True)
        joint_probs = joint_counts / total
        h_joint = -np.sum(joint_probs * np.log2(joint_probs + 1e-12))
        _, cond_counts = np.unique(c_window, return_counts=True)
        cond_probs = cond_counts / total
        h_cond = -np.sum(cond_probs * np.log2(cond_probs + 1e-12))
        entropies[idx] = h_joint - h_cond

    return entropies


# ==============================================================================
# Step 3: Extract alt ask price AT a specific time offset
# ==============================================================================
def extract_entry_prices_at_delay(asset, epochs_needed, delay_ms):
    """
    Get the ask_price_1 for each token label at t = epoch_ms + delay_ms.
    Uses the closest snapshot AT OR AFTER the delay time.
    This is the price you'd actually face when entering.
    """
    data_dir = DATA_DIR / f"telonex_{asset}_5m_book_snapshots"
    if not data_dir.exists():
        return {}

    files = sorted(data_dir.iterdir())
    entries = {}  # epoch -> {Up: ask, Down: ask}

    for f in files:
        if f.suffix != ".parquet":
            continue
        try:
            df = pd.read_parquet(f, columns=[
                "slug", "token_label", "exchange_timestamp",
                "ask_price_1", "bid_price_1", "mid_price"
            ])
        except Exception:
            continue
        if df.empty:
            continue

        slug = df["slug"].iloc[0]
        epoch = int(slug.split("-")[-1])
        if epoch not in epochs_needed:
            continue

        epoch_ms = epoch * 1000
        target_ts = epoch_ms + delay_ms

        for label in ["Up", "Down"]:
            sub = df[df["token_label"] == label].sort_values("exchange_timestamp")
            if sub.empty:
                continue

            # Get snapshot closest to (at or after) the signal time
            after_signal = sub[sub["exchange_timestamp"] >= target_ts]
            if len(after_signal) > 0:
                row = after_signal.iloc[0]
            else:
                # Market data ends before delay — use last available
                row = sub.iloc[-1]

            ask = float(row["ask_price_1"])
            bid = float(row["bid_price_1"])
            mid = float(row["mid_price"])
            snap_ts = int(row["exchange_timestamp"])

            if epoch not in entries:
                entries[epoch] = {}
            entries[epoch][label] = {
                "ask": ask,
                "bid": bid,
                "mid": mid,
                "snap_ts": snap_ts,
                "delay_actual_ms": snap_ts - epoch_ms,
            }

    return entries


# ==============================================================================
# Step 4: Load Binance BTC price data
# ==============================================================================
def load_binance_btc():
    """Load Binance BTC/USDT quotes and compute returns at various delays."""
    btc_path = DATA_DIR / "btc_quotes" / "btcusdt_quotes.parquet"
    df = pd.read_parquet(btc_path)
    # Ensure timestamp is in ms
    if "timestamp_ms" in df.columns:
        ts_col = "timestamp_ms"
    elif "timestamp" in df.columns:
        ts_col = "timestamp"
    elif "exchange_timestamp" in df.columns:
        ts_col = "exchange_timestamp"
    else:
        raise ValueError(f"No timestamp column found. Columns: {df.columns.tolist()}")

    df = df.sort_values(ts_col).reset_index(drop=True)
    print(f"  Binance BTC data: {len(df)} rows, "
          f"ts range: {df[ts_col].min()} - {df[ts_col].max()}")

    # Check if timestamps are in seconds or ms
    sample_ts = df[ts_col].iloc[0]
    if sample_ts < 1e12:
        # Seconds — convert to ms
        df["ts_ms"] = (df[ts_col] * 1000).astype(int)
        print(f"  Converted timestamps from seconds to ms")
    else:
        df["ts_ms"] = df[ts_col].astype(int)

    return df


def get_btc_return_at_delay(binance_df, epoch, delay_s):
    """Get BTC return from market open to open+delay_s."""
    epoch_ms = epoch * 1000
    target_ms = epoch_ms + delay_s * 1000

    # Find price at market open
    open_mask = (binance_df["ts_ms"] >= epoch_ms - 1000) & (binance_df["ts_ms"] <= epoch_ms + 1000)
    open_rows = binance_df.loc[open_mask]
    if len(open_rows) == 0:
        return None

    # Find price at delay
    delay_mask = (binance_df["ts_ms"] >= target_ms - 1000) & (binance_df["ts_ms"] <= target_ms + 1000)
    delay_rows = binance_df.loc[delay_mask]
    if len(delay_rows) == 0:
        return None

    open_price = open_rows["mid_price"].iloc[len(open_rows) // 2]
    delay_price = delay_rows["mid_price"].iloc[len(delay_rows) // 2]

    if open_price <= 0:
        return None
    return (delay_price - open_price) / open_price


# ==============================================================================
# Main backtest
# ==============================================================================
def run():
    print("=" * 70)
    print("Entropy + Binance Signal — CORRECTED Entry Prices")
    print("=" * 70)

    # 1. Resolution tables
    print("\n1. Building resolution tables...")
    res_tables = {"btc": load_btc_resolutions()}
    for asset in ALT_ASSETS:
        res_tables[asset] = build_resolution_table(asset)

    # 2. Concurrent epochs
    all_sets = [set(res_tables[a].keys()) for a in ASSETS]
    concurrent = sorted(set.intersection(*all_sets))
    print(f"\n  Concurrent epochs: {len(concurrent)}")

    # 3. Joint outcomes + entropy
    print("\n2. Computing conditional entropies...")
    joint = pd.DataFrame({"epoch": concurrent})
    for asset in ASSETS:
        joint[asset] = joint["epoch"].map(
            lambda e, a=asset: 1 if res_tables[a].get(e) == "Up" else 0
        )

    cond_map = {
        "eth": ["btc", "sol", "xrp"],
        "sol": ["btc", "eth", "xrp"],
        "xrp": ["btc", "eth", "sol"],
    }
    for alt in ALT_ASSETS:
        joint[f"h_{alt}"] = compute_conditional_entropy(
            joint, alt, cond_map[alt], WINDOW_SIZE
        )

    # 4. Trade signals (entropy regime only — direction comes from Binance)
    print("\n3. Finding low-entropy trade opportunities...")
    signals = []
    for idx in range(WINDOW_SIZE, len(joint)):
        row = joint.iloc[idx]
        epoch = int(row["epoch"])
        best_alt = None
        best_h = ENTROPY_THRESHOLD
        for alt in ALT_ASSETS:
            h = row[f"h_{alt}"]
            if not np.isnan(h) and h < best_h:
                best_h = h
                best_alt = alt
        if best_alt:
            signals.append({
                "epoch": epoch,
                "target_asset": best_alt,
                "cond_entropy": best_h,
                "btc_outcome": int(row["btc"]),
            })

    signals_df = pd.DataFrame(signals)
    print(f"  Low-entropy signals: {len(signals_df)}")

    # 5. Load Binance data
    print("\n4. Loading Binance BTC data...")
    binance = load_binance_btc()

    # 6. Precompute BTC returns for all signal epochs at all delays
    print("\n5. Computing BTC returns at each delay...")
    signal_epochs = set(signals_df["epoch"].values)
    btc_returns = {}  # (epoch, delay_s) -> return
    for delay_s in SIGNAL_DELAYS_S:
        count = 0
        for epoch in signal_epochs:
            ret = get_btc_return_at_delay(binance, epoch, delay_s)
            if ret is not None:
                btc_returns[(epoch, delay_s)] = ret
                count += 1
        print(f"  {delay_s}s: {count}/{len(signal_epochs)} epochs with data")

    # 7. Preload alt entry prices at each delay
    print("\n6. Loading alt entry prices at each delay...")
    alt_entries = {}  # (asset, delay_ms) -> {epoch: {Up: {ask, bid, mid, ...}, Down: ...}}
    for alt in ALT_ASSETS:
        alt_signal_epochs = set(
            signals_df[signals_df["target_asset"] == alt]["epoch"].values
        )
        if not alt_signal_epochs:
            continue
        for delay_s in SIGNAL_DELAYS_S:
            delay_ms = delay_s * 1000
            print(f"  {alt} at {delay_s}s ({len(alt_signal_epochs)} epochs)...")
            alt_entries[(alt, delay_ms)] = extract_entry_prices_at_delay(
                alt, alt_signal_epochs, delay_ms
            )

    # 8. Sweep all configurations
    print("\n7. Running parameter sweep...")
    print(f"\n{'Delay':>5s}  {'MinRet':>8s}  {'Trades':>6s}  {'WR%':>6s}  "
          f"{'PnL':>10s}  {'PnL/t':>8s}  {'DD':>8s}  {'Sharpe':>7s}  "
          f"{'AvgEntry':>8s}  {'AvgDelay':>8s}")
    print("-" * 95)

    all_results = []

    for delay_s in SIGNAL_DELAYS_S:
        delay_ms = delay_s * 1000

        for min_ret in MIN_RET_THRESHOLDS:
            trades = []

            for _, sig in signals_df.iterrows():
                epoch = int(sig["epoch"])
                alt = sig["target_asset"]

                # Get BTC return at this delay
                btc_ret = btc_returns.get((epoch, delay_s))
                if btc_ret is None:
                    continue

                # Apply magnitude filter
                if abs(btc_ret) < min_ret:
                    continue

                # Direction from Binance price
                direction = "Up" if btc_ret > 0 else "Down"

                # Get alt entry price AT the signal time
                entry_data = alt_entries.get((alt, delay_ms), {}).get(epoch, {}).get(direction)
                if entry_data is None:
                    continue

                ask = entry_data["ask"]
                if ask <= 0 or ask > ENTRY_PRICE_CAP:
                    continue

                entry_price = ask
                actual_delay_ms = entry_data["delay_actual_ms"]

                # Check outcome
                actual_winner = res_tables[alt].get(epoch)
                if actual_winner is None:
                    continue

                won = (actual_winner == direction)
                fee = polymarket_fee(entry_price)
                cost = entry_price + fee
                pnl_per_token = (1.0 - cost) if won else (-cost)
                total_pnl = pnl_per_token * TRADE_SIZE

                trades.append({
                    "epoch": epoch,
                    "asset": alt,
                    "direction": direction,
                    "entry_price": entry_price,
                    "fee": fee,
                    "won": won,
                    "pnl_per_token": pnl_per_token,
                    "total_pnl": total_pnl,
                    "cond_entropy": float(sig["cond_entropy"]),
                    "btc_ret": btc_ret,
                    "actual_delay_ms": actual_delay_ms,
                })

            if len(trades) == 0:
                print(f"{delay_s:>5d}s  {min_ret:>8.4f}  {'---':>6s}")
                continue

            tdf = pd.DataFrame(trades)
            n = len(tdf)
            wr = tdf["won"].mean()
            pnl = tdf["total_pnl"].sum()
            pnl_t = pnl / n
            cum = tdf["total_pnl"].cumsum()
            dd = (cum - cum.cummax()).min()
            std = tdf["total_pnl"].std()
            sharpe = (tdf["total_pnl"].mean() / std * np.sqrt(365.25 * 24 * 12)) if std > 0 else 0
            avg_entry = tdf["entry_price"].mean()
            avg_delay = tdf["actual_delay_ms"].mean() / 1000

            print(f"{delay_s:>5d}s  {min_ret:>8.4f}  {n:>6d}  {wr:>5.1%}  "
                  f"${pnl:>9.2f}  ${pnl_t:>7.3f}  ${dd:>7.0f}  {sharpe:>7.2f}  "
                  f"{avg_entry:>8.4f}  {avg_delay:>7.1f}s")

            all_results.append({
                "delay_s": delay_s,
                "min_ret": min_ret,
                "n_trades": n,
                "win_rate": round(wr, 4),
                "pnl_total": round(pnl, 2),
                "pnl_per_trade": round(pnl_t, 4),
                "max_drawdown": round(dd, 2),
                "sharpe": round(sharpe, 2),
                "avg_entry_price": round(avg_entry, 4),
                "avg_actual_delay_s": round(avg_delay, 1),
            })

    # 9. Detailed breakdown of interesting configs
    DETAIL_CONFIGS = [
        (15, 0.0003),
        (30, 0.0003),
        (30, 0.0005),
        (60, 0.0005),
    ]

    print(f"\n\n{'='*70}")
    print(f"DETAILED BREAKDOWNS — Per-token economics")
    print(f"(TRADE_SIZE = {TRADE_SIZE} tokens per trade)")
    print(f"{'='*70}")

    for cfg_delay_s, cfg_min_ret in DETAIL_CONFIGS:
        cfg_delay_ms = cfg_delay_s * 1000

        trades = []
        for _, sig in signals_df.iterrows():
            epoch = int(sig["epoch"])
            alt = sig["target_asset"]
            btc_ret = btc_returns.get((epoch, cfg_delay_s))
            if btc_ret is None or abs(btc_ret) < cfg_min_ret:
                continue
            direction = "Up" if btc_ret > 0 else "Down"
            entry_data = alt_entries.get((alt, cfg_delay_ms), {}).get(epoch, {}).get(direction)
            if entry_data is None:
                continue
            ask = entry_data["ask"]
            if ask <= 0 or ask > ENTRY_PRICE_CAP:
                continue
            actual_winner = res_tables[alt].get(epoch)
            if actual_winner is None:
                continue
            won = (actual_winner == direction)
            fee = polymarket_fee(ask)
            cost = ask + fee
            pnl_per_token = (1.0 - cost) if won else (-cost)
            trades.append({
                "epoch": epoch,
                "asset": alt,
                "direction": direction,
                "entry_price": ask,
                "fee": fee,
                "won": won,
                "pnl_per_token": pnl_per_token,
                "total_pnl": pnl_per_token * TRADE_SIZE,
                "btc_ret": btc_ret,
                "actual_delay_ms": entry_data["delay_actual_ms"],
            })

        if not trades:
            continue

        tdf = pd.DataFrame(trades)
        n = len(tdf)
        wr = tdf["won"].mean()
        pnl = tdf["total_pnl"].sum()
        avg_entry = tdf["entry_price"].mean()

        print(f"\n--- {cfg_delay_s}s delay, |ret| > {cfg_min_ret} ---")
        print(f"  Trades: {n}, WR: {wr:.1%}, Total PnL: ${pnl:.2f} ({TRADE_SIZE} tokens/trade)")
        print(f"  Avg entry: {avg_entry:.4f}, Avg PnL/token: ${tdf['pnl_per_token'].mean():.4f}")
        print(f"  Per-token math: {wr:.1%} × ${1-avg_entry:.3f} win - {1-wr:.1%} × ${avg_entry:.3f} loss "
              f"= ${wr*(1-avg_entry) - (1-wr)*avg_entry:.4f}/token expected")

        # Entry price buckets
        print(f"\n  Entry price buckets (per-token PnL):")
        print(f"  {'Bucket':>12s}  {'Trades':>6s}  {'WR%':>6s}  {'TotalPnL':>10s}  {'PnL/tok':>8s}  {'FairP':>6s}  {'Edge':>6s}")
        for lo in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            hi = lo + 0.1
            sub = tdf[(tdf["entry_price"] >= lo) & (tdf["entry_price"] < hi)]
            if len(sub) == 0:
                continue
            sub_wr = sub["won"].mean()
            sub_pnl = sub["total_pnl"].sum()
            sub_pnl_tok = sub["pnl_per_token"].mean()
            fair_price = sub_wr  # fair price = probability of winning
            avg_e = sub["entry_price"].mean()
            edge = fair_price - avg_e  # positive = we're buying below fair value
            print(f"  [{lo:.1f}-{hi:.1f})  {len(sub):>6d}  {sub_wr:>5.1%}  ${sub_pnl:>9.2f}  ${sub_pnl_tok:>7.4f}  {fair_price:>5.3f}  {edge:>+5.3f}")

        # By asset
        print(f"\n  By asset:")
        for alt in ALT_ASSETS:
            sub = tdf[tdf["asset"] == alt]
            if len(sub) > 0:
                print(f"    {alt}: {len(sub):>4d} trades, WR={sub['won'].mean():.1%}, "
                      f"PnL=${sub['total_pnl'].sum():.2f}, "
                      f"avg_entry={sub['entry_price'].mean():.4f}, "
                      f"PnL/tok=${sub['pnl_per_token'].mean():.4f}")

        # Temporal stability
        print(f"\n  By week:")
        tdf["week"] = pd.to_datetime(tdf["epoch"], unit="s").dt.isocalendar().week
        for week, sub in tdf.groupby("week"):
            print(f"    Week {week}: {len(sub):>4d} trades, WR={sub['won'].mean():.1%}, "
                  f"PnL=${sub['total_pnl'].sum():.2f}")

    # Save results
    results_path = Path("backtest_entropy_binance_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n\nResults saved to {results_path}")


if __name__ == "__main__":
    run()
