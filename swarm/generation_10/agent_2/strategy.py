"""
Concurrent Resolution Conditional Entropy Strategy
====================================================
Hypothesis: Build a rolling lookup table of joint resolution outcomes
(BTC×ETH×SOL×XRP = 16 states) and measure conditional entropy
H(XRP_outcome | BTC_outcome, ETH_outcome, SOL_outcome) over trailing windows.
When conditional entropy drops below threshold (tight coupling regime),
use real-time BTC book imbalance to trade the alt with lowest conditional entropy.
"""

import os
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
ALT_ASSETS = ["eth", "sol", "xrp"]  # tradeable alts
WINDOW_SIZE = 50  # trailing window for conditional entropy
ENTROPY_THRESHOLD = 0.5  # bits - trade when H < this (strong coupling)
ENTRY_PRICE_CAP = 0.85
MIN_ORDER_SIZE = 5
TRADE_SIZE = 10  # tokens per trade

# Polymarket fee function
def polymarket_fee(p):
    """Fee = 0.0222 * p * 0.25 * (p*(1-p))^2"""
    return 0.0222 * p * 0.25 * (p * (1 - p)) ** 2


# ==============================================================================
# Step 1: Build resolution tables from all assets
# ==============================================================================
def infer_resolution_from_file(filepath):
    """Read a parquet file and infer the winner from final prices."""
    try:
        df = pd.read_parquet(filepath, columns=["slug", "token_label", "exchange_timestamp", "bid_price_1"])
    except Exception:
        return None, None
    if df.empty:
        return None, None

    slug = df["slug"].iloc[0]

    # Get the last timestamp
    max_ts = df["exchange_timestamp"].max()
    # Use last 5 seconds of data
    final = df[df["exchange_timestamp"] >= max_ts - 5000]

    # Check final bid_price_1 for each token
    up_final = final[final["token_label"] == "Up"]["bid_price_1"]
    down_final = final[final["token_label"] == "Down"]["bid_price_1"]

    up_bid = up_final.mean() if len(up_final) > 0 else 0.5
    down_bid = down_final.mean() if len(down_final) > 0 else 0.5

    if up_bid > 0.8:
        return slug, "Up"
    elif down_bid > 0.8:
        return slug, "Down"
    else:
        return slug, None  # ambiguous


def build_resolution_table(asset):
    """Build resolution table for an asset from its book snapshot files."""
    data_dir = DATA_DIR / f"telonex_{asset}_5m_book_snapshots"
    if not data_dir.exists():
        print(f"  Directory not found: {data_dir}")
        return {}

    files = sorted(data_dir.iterdir())
    resolutions = {}

    # Process in batches for efficiency
    for i, f in enumerate(files):
        if f.suffix != ".parquet":
            continue
        slug, winner = infer_resolution_from_file(f)
        if slug and winner:
            epoch = int(slug.split("-")[-1])
            resolutions[epoch] = winner

        if (i + 1) % 1000 == 0:
            print(f"  {asset}: processed {i+1}/{len(files)} files")

    print(f"  {asset}: {len(resolutions)} resolved markets out of {len(files)} files")
    return resolutions


def load_btc_resolutions_from_cache():
    """Load BTC resolutions from the resolution cache."""
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
            # Format: outcomes[i] corresponds to clobTokenIds[i]
            try:
                idx = info["clobTokenIds"].index(info["winningTokenId"])
                winner = info["outcomes"][idx]
            except (ValueError, IndexError):
                pass
        elif "winning_token" in info and "token_outcomes" in info:
            # token_outcomes is a dict {token_id: 0.0 or 1.0}
            # We need to map token_id to Up/Down - skip these for now
            # (they'll be filled by book data inference if needed)
            pass

        if winner:
            resolutions[epoch] = winner

    print(f"  btc: {len(resolutions)} resolved from cache")
    return resolutions


# ==============================================================================
# Step 2: Extract book snapshots for trading decisions
# ==============================================================================
def extract_early_book_data(asset, epochs_needed):
    """Extract early book data (first 30s) for given epochs."""
    data_dir = DATA_DIR / f"telonex_{asset}_5m_book_snapshots"
    if not data_dir.exists():
        return pd.DataFrame()

    records = []
    files = sorted(data_dir.iterdir())

    for f in files:
        if f.suffix != ".parquet":
            continue
        # Quick slug check from filename or read minimal data
        df = pd.read_parquet(f, columns=[
            "slug", "token_label", "exchange_timestamp",
            "mid_price", "book_imbalance", "bid_price_1", "ask_price_1"
        ])
        if df.empty:
            continue

        slug = df["slug"].iloc[0]
        epoch = int(slug.split("-")[-1])

        if epoch not in epochs_needed:
            continue

        # Get early data (first 30 seconds) for trading signals
        epoch_ms = epoch * 1000
        early = df[df["exchange_timestamp"] <= epoch_ms + 30000]
        if early.empty:
            early = df.head(100)

        for label in ["Up", "Down"]:
            sub = early[early["token_label"] == label]
            if sub.empty:
                continue
            records.append({
                "epoch": epoch,
                "asset": asset,
                "token_label": label,
                "mid_price": sub["mid_price"].median(),
                "book_imbalance": sub["book_imbalance"].median(),
                "ask_price_1": sub["ask_price_1"].median(),
                "bid_price_1": sub["bid_price_1"].median(),
                "n_snapshots": len(sub),
            })

    return pd.DataFrame(records)


# ==============================================================================
# Step 3: Conditional entropy computation
# ==============================================================================
def compute_conditional_entropy(joint_outcomes, target_asset, conditioning_assets, window):
    """
    Compute H(target | conditioning) over a trailing window.
    joint_outcomes: DataFrame with columns for each asset's outcome (0=Down, 1=Up)
    Returns Series of conditional entropy values.
    """
    n = len(joint_outcomes)
    entropies = np.full(n, np.nan)

    target_col = target_asset
    cond_cols = conditioning_assets

    target_vals = joint_outcomes[target_col].values
    # Encode conditioning state as single int
    cond_state = np.zeros(n, dtype=int)
    for i, c in enumerate(cond_cols):
        cond_state += joint_outcomes[c].values.astype(int) * (2 ** i)

    for idx in range(window, n):
        start = idx - window
        t_window = target_vals[start:idx]
        c_window = cond_state[start:idx]

        # Compute H(target | cond) = H(target, cond) - H(cond)
        # Using empirical frequencies
        total = len(t_window)

        # Joint distribution
        joint_keys = t_window * 8 + c_window  # target has 2 values, cond has 8
        _, joint_counts = np.unique(joint_keys, return_counts=True)
        joint_probs = joint_counts / total
        h_joint = -np.sum(joint_probs * np.log2(joint_probs + 1e-12))

        # Marginal conditioning distribution
        _, cond_counts = np.unique(c_window, return_counts=True)
        cond_probs = cond_counts / total
        h_cond = -np.sum(cond_probs * np.log2(cond_probs + 1e-12))

        entropies[idx] = h_joint - h_cond

    return entropies


# ==============================================================================
# Step 4: Main backtest
# ==============================================================================
def run_backtest():
    print("=" * 60)
    print("Concurrent Resolution Conditional Entropy Strategy")
    print("=" * 60)

    # Step 1: Build resolution tables
    print("\nStep 1: Building resolution tables...")
    resolution_tables = {}

    # BTC from cache (faster)
    resolution_tables["btc"] = load_btc_resolutions_from_cache()

    # Alts from book data
    for asset in ALT_ASSETS:
        print(f"  Processing {asset}...")
        resolution_tables[asset] = build_resolution_table(asset)

    # Step 2: Find concurrent epochs (all 4 assets resolved)
    print("\nStep 2: Finding concurrent epochs...")
    all_epoch_sets = [set(resolution_tables[a].keys()) for a in ASSETS]
    concurrent_epochs = sorted(set.intersection(*all_epoch_sets))
    print(f"  Concurrent epochs: {len(concurrent_epochs)}")

    if len(concurrent_epochs) < WINDOW_SIZE + 10:
        print(f"  INSUFFICIENT DATA: Need at least {WINDOW_SIZE + 10} concurrent epochs, got {len(concurrent_epochs)}")
        return generate_failed_results(
            f"Only {len(concurrent_epochs)} concurrent epochs found, need {WINDOW_SIZE + 10}+",
            len(concurrent_epochs)
        )

    # Step 3: Build joint outcome DataFrame
    print("\nStep 3: Building joint outcome table...")
    joint = pd.DataFrame({"epoch": concurrent_epochs})
    for asset in ASSETS:
        joint[asset] = joint["epoch"].map(
            lambda e, a=asset: 1 if resolution_tables[a].get(e) == "Up" else 0
        )

    # Encode joint state (4 bits = 16 states)
    joint["joint_state"] = (
        joint["btc"] * 8 + joint["eth"] * 4 + joint["sol"] * 2 + joint["xrp"]
    )

    print(f"  Joint state distribution:")
    state_counts = joint["joint_state"].value_counts().sort_index()
    for state, count in state_counts.items():
        binary = format(state, "04b")
        pct = count / len(joint) * 100
        print(f"    State {state:2d} ({binary}): {count:4d} ({pct:.1f}%)")

    # Step 4: Compute conditional entropy for each alt
    print("\nStep 4: Computing conditional entropies...")
    cond_assets_map = {
        "eth": ["btc", "sol", "xrp"],
        "sol": ["btc", "eth", "xrp"],
        "xrp": ["btc", "eth", "sol"],
    }

    for alt in ALT_ASSETS:
        joint[f"h_{alt}"] = compute_conditional_entropy(
            joint, alt, cond_assets_map[alt], WINDOW_SIZE
        )

    # Show entropy statistics
    for alt in ALT_ASSETS:
        col = f"h_{alt}"
        valid = joint[col].dropna()
        if len(valid) > 0:
            print(f"  H({alt}|others): mean={valid.mean():.4f}, "
                  f"min={valid.min():.4f}, max={valid.max():.4f}, "
                  f"<threshold({ENTROPY_THRESHOLD}): {(valid < ENTROPY_THRESHOLD).sum()}")

    # Step 5: Generate trade signals
    print("\nStep 5: Generating trade signals...")
    # For each epoch after warm-up, find if any alt has H < threshold
    trade_signals = []
    for idx in range(WINDOW_SIZE, len(joint)):
        row = joint.iloc[idx]
        epoch = int(row["epoch"])

        # Find alt with lowest conditional entropy below threshold
        best_alt = None
        best_h = ENTROPY_THRESHOLD

        for alt in ALT_ASSETS:
            h = row[f"h_{alt}"]
            if not np.isnan(h) and h < best_h:
                best_h = h
                best_alt = alt

        if best_alt is None:
            continue

        # BTC book imbalance determines direction
        # We need the BTC imbalance from the current epoch's early data
        # For now, use the BTC resolution as proxy for imbalance sign
        # (In live trading, we'd use real-time imbalance)
        btc_outcome = int(row["btc"])  # 1=Up, 0=Down
        trade_direction = "Up" if btc_outcome == 1 else "Down"

        trade_signals.append({
            "epoch": epoch,
            "target_asset": best_alt,
            "direction": trade_direction,
            "cond_entropy": best_h,
            "btc_outcome": btc_outcome,
        })

    print(f"  Raw trade signals: {len(trade_signals)}")

    if len(trade_signals) == 0:
        print("  NO TRADE SIGNALS GENERATED")
        return generate_failed_results("No epochs with conditional entropy below threshold", len(concurrent_epochs))

    signals_df = pd.DataFrame(trade_signals)

    # Step 6: Get entry prices from book data
    print("\nStep 6: Extracting entry prices...")
    needed_epochs = set(signals_df["epoch"].values)

    # We need book data for each alt at signal epochs
    book_data = {}
    for alt in ALT_ASSETS:
        alt_epochs = set(signals_df[signals_df["target_asset"] == alt]["epoch"].values)
        if alt_epochs:
            print(f"  Loading {alt} book data for {len(alt_epochs)} epochs...")
            book_data[alt] = extract_early_book_data(alt, alt_epochs)

    # Step 7: Execute trades and compute PnL
    print("\nStep 7: Computing PnL...")
    trades = []

    for _, signal in signals_df.iterrows():
        epoch = int(signal["epoch"])
        alt = signal["target_asset"]
        direction = signal["direction"]

        if alt not in book_data or book_data[alt].empty:
            continue

        alt_book = book_data[alt]
        entry_row = alt_book[
            (alt_book["epoch"] == epoch) & (alt_book["token_label"] == direction)
        ]

        if entry_row.empty:
            continue

        ask_price = float(entry_row["ask_price_1"].iloc[0])
        mid_price = float(entry_row["mid_price"].iloc[0])

        # Use ask price for entry (buying)
        entry_price = ask_price if ask_price > 0 else mid_price
        if entry_price <= 0 or entry_price > ENTRY_PRICE_CAP:
            continue

        # Check actual outcome
        actual_winner = resolution_tables[alt].get(epoch)
        if actual_winner is None:
            continue

        won = (actual_winner == direction)

        # PnL calculation
        fee = polymarket_fee(entry_price)
        cost_per_token = entry_price + fee

        if won:
            pnl_per_token = 1.0 - cost_per_token
        else:
            pnl_per_token = -cost_per_token

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
            "cond_entropy": float(signal["cond_entropy"]),
        })

    if len(trades) == 0:
        print("  NO TRADES EXECUTED")
        return generate_failed_results("Trade signals generated but no valid entries found", len(concurrent_epochs))

    trades_df = pd.DataFrame(trades)

    # Step 8: Compute metrics
    print("\nStep 8: Computing metrics...")
    total_trades = len(trades_df)
    wins = trades_df["won"].sum()
    win_rate = wins / total_trades
    pnl_total = trades_df["total_pnl"].sum()
    pnl_per_trade = pnl_total / total_trades

    # Drawdown
    cumulative_pnl = trades_df["total_pnl"].cumsum()
    running_max = cumulative_pnl.cummax()
    drawdowns = cumulative_pnl - running_max
    max_drawdown = drawdowns.min()

    # Sharpe (annualized, assuming 5-min intervals)
    trade_returns = trades_df["total_pnl"]
    if trade_returns.std() > 0:
        intervals_per_year = 365.25 * 24 * 12  # 5-min intervals
        sharpe = (trade_returns.mean() / trade_returns.std()) * np.sqrt(intervals_per_year)
    else:
        sharpe = 0.0

    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"  Total trades:     {total_trades}")
    print(f"  Win rate:         {win_rate:.4f} ({wins}/{total_trades})")
    print(f"  Total PnL:        {pnl_total:.4f}")
    print(f"  PnL per trade:    {pnl_per_trade:.4f}")
    print(f"  Max drawdown:     {max_drawdown:.4f}")
    print(f"  Sharpe ratio:     {sharpe:.4f}")
    print(f"  Avg entry price:  {trades_df['entry_price'].mean():.4f}")
    print(f"  Avg cond entropy: {trades_df['cond_entropy'].mean():.4f}")

    # Breakdown by asset
    print(f"\n  By asset:")
    for alt in ALT_ASSETS:
        sub = trades_df[trades_df["asset"] == alt]
        if len(sub) > 0:
            print(f"    {alt}: {len(sub)} trades, "
                  f"WR={sub['won'].mean():.3f}, "
                  f"PnL={sub['total_pnl'].sum():.4f}")

    # Breakdown by entropy bucket
    print(f"\n  By entropy bucket:")
    trades_df["h_bucket"] = pd.cut(trades_df["cond_entropy"], bins=5)
    for bucket, sub in trades_df.groupby("h_bucket", observed=True):
        if len(sub) > 0:
            print(f"    H={bucket}: {len(sub)} trades, "
                  f"WR={sub['won'].mean():.3f}, "
                  f"PnL={sub['total_pnl'].sum():.4f}")

    # Determine verdict
    if pnl_total > 0 and win_rate > 0.52 and sharpe > 0.5:
        verdict = "promising"
    elif pnl_total > 0 or win_rate > 0.50:
        verdict = "marginal"
    else:
        verdict = "failed"

    # Step 9: Write results
    results = {
        "strategy_name": "concurrent_resolution_conditional_entropy",
        "description": (
            "Measures conditional entropy of alt coin resolution outcomes "
            "given other assets over trailing windows. Trades the alt with "
            "lowest conditional entropy using BTC outcome as directional signal "
            "when cross-asset coupling is temporarily strong."
        ),
        "hypothesis": (
            "Correlation regimes are non-stationary and detectable from joint "
            "outcome history. When conditional entropy drops below threshold, "
            "alts are tightly coupled to BTC, making BTC imbalance predictive "
            "of alt outcomes."
        ),
        "metrics": {
            "total_trades": int(total_trades),
            "win_rate": round(float(win_rate), 4),
            "pnl_total": round(float(pnl_total), 4),
            "pnl_per_trade": round(float(pnl_per_trade), 4),
            "max_drawdown": round(float(max_drawdown), 4),
            "sharpe": round(float(sharpe), 4),
        },
        "parameters": {
            "window_size": WINDOW_SIZE,
            "entropy_threshold": ENTROPY_THRESHOLD,
            "entry_price_cap": ENTRY_PRICE_CAP,
            "trade_size": TRADE_SIZE,
            "assets": ASSETS,
        },
        "verdict": verdict,
        "next_steps": (
            "If promising: test with real-time BTC book imbalance instead of "
            "resolved BTC outcome; sweep entropy thresholds; add time-of-day filter. "
            "If failed: the BTC-alt coupling may not be strong enough in 5-min binary "
            "markets to overcome fees, or the sample may lack sufficient concurrent data."
        ),
    }

    return results


def generate_failed_results(reason, n_concurrent):
    """Generate results JSON for failed strategy."""
    return {
        "strategy_name": "concurrent_resolution_conditional_entropy",
        "description": (
            "Measures conditional entropy of alt coin resolution outcomes "
            "given other assets over trailing windows."
        ),
        "hypothesis": (
            "Correlation regimes are non-stationary and detectable from joint "
            "outcome history."
        ),
        "metrics": {
            "total_trades": 0,
            "win_rate": 0.0,
            "pnl_total": 0.0,
            "pnl_per_trade": 0.0,
            "max_drawdown": 0.0,
            "sharpe": 0.0,
        },
        "parameters": {
            "window_size": WINDOW_SIZE,
            "entropy_threshold": ENTROPY_THRESHOLD,
            "n_concurrent_epochs": n_concurrent,
        },
        "verdict": "failed",
        "next_steps": f"Failed: {reason}. Need more concurrent multi-asset data.",
    }


# ==============================================================================
# IMPORTANT: Also run with real BTC book imbalance (not just resolved outcome)
# ==============================================================================
def run_backtest_with_book_imbalance():
    """
    Enhanced version: uses actual BTC book imbalance sign from early snapshots
    rather than the resolved BTC outcome (which is look-ahead bias).
    """
    print("\n" + "=" * 60)
    print("ENHANCED: Using BTC Book Imbalance (no look-ahead)")
    print("=" * 60)

    # Reuse resolution tables from main backtest
    print("\nStep 1: Building resolution tables...")
    resolution_tables = {}
    resolution_tables["btc"] = load_btc_resolutions_from_cache()
    for asset in ALT_ASSETS:
        print(f"  Processing {asset}...")
        resolution_tables[asset] = build_resolution_table(asset)

    all_epoch_sets = [set(resolution_tables[a].keys()) for a in ASSETS]
    concurrent_epochs = sorted(set.intersection(*all_epoch_sets))
    print(f"  Concurrent epochs: {len(concurrent_epochs)}")

    if len(concurrent_epochs) < WINDOW_SIZE + 10:
        return generate_failed_results(
            f"Only {len(concurrent_epochs)} concurrent epochs",
            len(concurrent_epochs)
        )

    # Build joint outcomes
    joint = pd.DataFrame({"epoch": concurrent_epochs})
    for asset in ASSETS:
        joint[asset] = joint["epoch"].map(
            lambda e, a=asset: 1 if resolution_tables[a].get(e) == "Up" else 0
        )

    # Compute conditional entropies
    cond_assets_map = {
        "eth": ["btc", "sol", "xrp"],
        "sol": ["btc", "eth", "xrp"],
        "xrp": ["btc", "eth", "sol"],
    }
    for alt in ALT_ASSETS:
        joint[f"h_{alt}"] = compute_conditional_entropy(
            joint, alt, cond_assets_map[alt], WINDOW_SIZE
        )

    # Get BTC book imbalance for ALL concurrent epochs
    print("\nLoading BTC book imbalance data...")
    btc_book = extract_early_book_data("btc", set(concurrent_epochs))

    if btc_book.empty:
        print("  No BTC book data loaded")
        return generate_failed_results("No BTC book data", len(concurrent_epochs))

    # Create BTC imbalance lookup (use Up token imbalance)
    btc_imbalance = btc_book[btc_book["token_label"] == "Up"].set_index("epoch")["book_imbalance"]
    print(f"  BTC imbalance data for {len(btc_imbalance)} epochs")

    # Generate signals using ACTUAL book imbalance (not resolved outcome)
    trade_signals = []
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

        if best_alt is None:
            continue

        # Use ACTUAL BTC book imbalance sign
        if epoch not in btc_imbalance.index:
            continue
        imb = btc_imbalance.loc[epoch]
        if np.isnan(imb) or imb == 0:
            continue

        trade_direction = "Up" if imb > 0 else "Down"

        trade_signals.append({
            "epoch": epoch,
            "target_asset": best_alt,
            "direction": trade_direction,
            "cond_entropy": best_h,
            "btc_imbalance": float(imb),
        })

    print(f"  Trade signals with real imbalance: {len(trade_signals)}")

    if len(trade_signals) == 0:
        return generate_failed_results("No signals with book imbalance data", len(concurrent_epochs))

    signals_df = pd.DataFrame(trade_signals)

    # Get alt entry prices
    book_data = {}
    for alt in ALT_ASSETS:
        alt_epochs = set(signals_df[signals_df["target_asset"] == alt]["epoch"].values)
        if alt_epochs:
            book_data[alt] = extract_early_book_data(alt, alt_epochs)

    # Execute trades
    trades = []
    for _, signal in signals_df.iterrows():
        epoch = int(signal["epoch"])
        alt = signal["target_asset"]
        direction = signal["direction"]

        if alt not in book_data or book_data[alt].empty:
            continue

        entry_row = book_data[alt][
            (book_data[alt]["epoch"] == epoch) & (book_data[alt]["token_label"] == direction)
        ]
        if entry_row.empty:
            continue

        ask_price = float(entry_row["ask_price_1"].iloc[0])
        mid_price = float(entry_row["mid_price"].iloc[0])
        entry_price = ask_price if ask_price > 0 else mid_price
        if entry_price <= 0 or entry_price > ENTRY_PRICE_CAP:
            continue

        actual_winner = resolution_tables[alt].get(epoch)
        if actual_winner is None:
            continue

        won = (actual_winner == direction)
        fee = polymarket_fee(entry_price)
        cost_per_token = entry_price + fee
        pnl_per_token = (1.0 - cost_per_token) if won else (-cost_per_token)
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
            "cond_entropy": float(signal["cond_entropy"]),
            "btc_imbalance": float(signal["btc_imbalance"]),
        })

    if len(trades) == 0:
        return generate_failed_results("No valid entries with book imbalance", len(concurrent_epochs))

    trades_df = pd.DataFrame(trades)
    total_trades = len(trades_df)
    wins = trades_df["won"].sum()
    win_rate = wins / total_trades
    pnl_total = trades_df["total_pnl"].sum()
    pnl_per_trade = pnl_total / total_trades

    cumulative_pnl = trades_df["total_pnl"].cumsum()
    running_max = cumulative_pnl.cummax()
    max_drawdown = (cumulative_pnl - running_max).min()

    trade_returns = trades_df["total_pnl"]
    if trade_returns.std() > 0:
        sharpe = (trade_returns.mean() / trade_returns.std()) * np.sqrt(365.25 * 24 * 12)
    else:
        sharpe = 0.0

    print(f"\n{'='*60}")
    print(f"RESULTS (Book Imbalance - No Look-Ahead)")
    print(f"{'='*60}")
    print(f"  Total trades:     {total_trades}")
    print(f"  Win rate:         {win_rate:.4f} ({wins}/{total_trades})")
    print(f"  Total PnL:        {pnl_total:.4f}")
    print(f"  PnL per trade:    {pnl_per_trade:.4f}")
    print(f"  Max drawdown:     {max_drawdown:.4f}")
    print(f"  Sharpe ratio:     {sharpe:.4f}")

    for alt in ALT_ASSETS:
        sub = trades_df[trades_df["asset"] == alt]
        if len(sub) > 0:
            print(f"    {alt}: {len(sub)} trades, WR={sub['won'].mean():.3f}, PnL={sub['total_pnl'].sum():.4f}")

    if pnl_total > 0 and win_rate > 0.52 and sharpe > 0.5:
        verdict = "promising"
    elif pnl_total > 0 or win_rate > 0.50:
        verdict = "marginal"
    else:
        verdict = "failed"

    return {
        "strategy_name": "concurrent_resolution_conditional_entropy",
        "description": (
            "Measures conditional entropy of alt coin resolution outcomes "
            "given other assets over trailing windows. Trades the alt with "
            "lowest conditional entropy using real-time BTC book imbalance sign "
            "(no look-ahead bias)."
        ),
        "hypothesis": (
            "Correlation regimes are non-stationary and detectable from joint "
            "outcome history. When conditional entropy drops below threshold, "
            "alts are tightly coupled to BTC, making BTC book imbalance predictive."
        ),
        "metrics": {
            "total_trades": int(total_trades),
            "win_rate": round(float(win_rate), 4),
            "pnl_total": round(float(pnl_total), 4),
            "pnl_per_trade": round(float(pnl_per_trade), 4),
            "max_drawdown": round(float(max_drawdown), 4),
            "sharpe": round(float(sharpe), 4),
        },
        "parameters": {
            "window_size": WINDOW_SIZE,
            "entropy_threshold": ENTROPY_THRESHOLD,
            "entry_price_cap": ENTRY_PRICE_CAP,
            "trade_size": TRADE_SIZE,
            "assets": ASSETS,
        },
        "verdict": verdict,
        "next_steps": (
            "If promising: optimize entropy threshold via grid search; add "
            "imbalance magnitude filter; consider time-of-day effects. "
            "If failed: cross-asset coupling in 5-min binary markets may be "
            "too weak or too noisy for entropy-based regime detection."
        ),
    }


if __name__ == "__main__":
    # Run oracle version first (sanity check with BTC resolved outcome)
    print("ORACLE BACKTEST (BTC outcome known - sanity check):")
    oracle_results = run_backtest()

    # Run realistic version with book imbalance
    real_results = run_backtest_with_book_imbalance()

    # Use the realistic (no look-ahead) results as final
    output_path = Path("swarm/generation_10/agent_2/results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(real_results, f, indent=2)

    print(f"\nResults written to {output_path}")
    print(f"Verdict: {real_results['verdict']}")
