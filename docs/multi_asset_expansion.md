# Multi-Asset Expansion Plan: ETH, SOL, XRP 5-Minute Markets

## Status: IN PROGRESS — code parameterized, data downloading

## Overview

Expand the current BTC-only XGBoost confidence strategy to trade ETH, SOL, and XRP Up/Down 5-minute markets on Polymarket. The BTC strategy (v3.3) runs live with limit-at-prediction entry, 101 features (including time-of-day encoding), and $20/trade. These additional assets share the same market structure (binary 5-minute up/down) but may have different microstructure characteristics (liquidity, volatility patterns, feature importance).

## Architecture Decision: Separate Instances Per Asset

**Chosen approach**: Run independent execution instances per asset, each with its own model server, config, and data pipeline. This is strongly preferred over a single multi-asset orchestrator.

### Why separate instances wins:

1. **Nonce isolation**: `incrementNonce()` invalidates ALL open orders across the wallet. A single instance trading multiple assets could nuke orders on asset B when trying to cancel on asset A. Separate instances with separate wallets (or careful nonce partitioning) eliminate this.

2. **Failure blast radius**: If ETH model server crashes, BTC keeps trading. A monolithic orchestrator means one bad market takes down everything.

3. **Independent scaling**: Each asset can run different betDollars, confidence thresholds, entry caps based on its own liquidity/performance. No shared state to coordinate.

4. **Simplicity**: Current codebase already works as a single-asset instance. Minimal code changes needed — mostly config and data pipeline parameterization.

5. **Market timing**: All assets use 5-minute windows on the same 300s cadence. A single process handling 4 simultaneous market opens/closes creates timing contention. Separate processes avoid this naturally.

### Operational cost:
- 4 model servers (~200MB RAM each) + 4 execution instances (~50MB each) = ~1GB total
- 4 Binance WS connections (BTC, ETH, SOL, XRP book tickers)
- Need separate proxy wallets per instance OR accept shared-nonce risk with careful cancellation

## Feature Space (v3.3 — 101 features)

All models use the same 101-feature extraction from `xgboost_flexible.py`:

### Orderbook features (per observation window: w10, w20, w30, w45, w60, w90, w120)
- `w{t}_mid` — leader token mid price at window t
- `w{t}_spread` — bid-ask spread
- `w{t}_imbalance` — book imbalance (bid depth - ask depth) / total
- `w{t}_bid_depth_3`, `w{t}_ask_depth_3` — depth within 3 cents
- `w{t}_depth_imb_3` — depth imbalance within 3 cents
- `w{t}_mid_delta`, `w{t}_spread_delta` — change from previous window

### Trajectory features (computed over observation period)
- `drift_from_50`, `drift_speed`, `drift_accel` — price drift dynamics
- `r_squared` — trend linearity
- `realized_vol`, `range`, `choppiness` — volatility measures
- `direction_changes`, `consistency`, `max_drift` — path characteristics
- `spread_min`, `spread_quantile_75`, `spread_diff` — spread dynamics
- `imb_std_10s`, `imb_max_10s` — imbalance volatility
- `ask_depth_2c`, `other_spread`, `cross_mid_sum_trend` — cross-token features

### Spot price features (from Binance feed, 1-hour rolling buffer)
- `btc_mid_zscore` — z-scored spot price (using training-time mean/std)
- `btc_spread` — spot bid-ask spread
- `btc_log_return` — log return over observation period
- `btc_rvol_30s`, `btc_rvol_60s`, `btc_rvol_300s` — realized vol at different scales
- `btc_ret_60s`, `btc_ret_300s`, `btc_ret_900s`, `btc_ret_3600s` — spot returns
- `btc_ret_since_open` — spot return since market open
- `btc_window_vol`, `btc_window_range` — vol/range during observation window
- `btc_vol_30s`, `btc_vol_5m`, `btc_vol_15m` (raw and normalized) — multi-scale vol
- `btc_range_30s`, `btc_range_5m`, `btc_range_15m` — price range at scales

### Time features (cyclical encoding, added v3.3)
- `hour_sin`, `hour_cos` — time of day (UTC, cyclical)
- `dow_sin`, `dow_cos` — day of week (cyclical)

### Observation metadata
- `observe_time_s`, `observe_time_pct` — when features were extracted
- `n_windows_observed` — how many snapshot windows are populated

**Note**: For non-BTC assets, all `btc_*` features are computed from that asset's own spot feed (e.g., ETH model uses ETH/USDT). The `btc_` prefix is kept to avoid refactoring — the model only cares about values, not names.

## Per-Asset Analysis Process

Each asset must go through this full analysis pipeline before going live. This is the process validated on BTC that produced v3.3.

### Step 1: Data Collection & Feature Extraction
```bash
# Download orderbook data from Telonex
python telonex_pipeline.py download --asset eth
python telonex_pipeline.py normalize --asset eth

# Download spot quotes from Binance
python download_spot_quotes.py --asset eth

# Build feature cache (all 7 observation windows per market)
python xgboost_flexible.py --rebuild-cache \
  --data-dir data/telonex_book_snapshots_eth \
  --btc-path data/eth_quotes/ethusdt_quotes.parquet \
  --cache-path data/features_eth.parquet
```

**Gate**: Need 2,000+ markets minimum. Below 500, don't attempt.

### Step 2: Initial Walk-Forward Evaluation

Run 5-window expanding-train walk-forward using BTC's current hyperparameters as a starting point. Evaluate OOS predictions at 120s observation window only.

**Gate**: OOS WR must be >= 70% at conf >= 0.72 to proceed. If below 65%, the asset likely isn't predictable with this feature set.

### Step 3: Breakdown Analysis

With OOS predictions, compute breakdowns by:

| Dimension | What to look for |
|-----------|-----------------|
| **Entry price** | Find the profitable range (BTC sweet spot: 0.70-0.85). Entry prices above this range have negative PnL due to payoff asymmetry. |
| **Confidence level** | Check if lower confidence bins are still profitable. On BTC, 0.72-0.75 was the most profitable per trade due to limit discount. |
| **Time of day (3h UTC blocks)** | Identify strong/weak periods. BTC: 06-09 UTC best, 12-15 UTC worst. |
| **Day of week** | Check for weekly patterns. BTC: Wed/Thu/Sat strongest, Mon/Tue/Fri flat. |
| **Order style (instant vs limit)** | With limit-at-prediction, check fill rates and adverse selection using raw tick data. |

### Step 4: Strategy Parameter Tuning

Based on breakdown analysis, determine per-asset:

| Parameter | How to set |
|-----------|-----------|
| **Entry price cap** | Set to upper bound of profitable entry range. BTC: 0.85 (was 0.90, lowered after 0.85-0.90 showed -$0.78/trade). |
| **Confidence threshold** | Keep at 0.72 unless data shows a clear cutoff. May need to be higher for assets with lower WR. |
| **Order method** | Use limit-at-prediction (validated on BTC: 2x PnL vs buy-at-ask, 100% fill rate on ask-based check). Verify fill rates hold on the new asset's book data. |
| **Bet size** | Start at $10/trade for new assets. Scale up after 50+ live trades validate performance. |
| **Vol filter** | Test but likely disabled (was filtering winners on BTC v3.2+). |

### Step 5: Hyperparameter Optimization

Run Optuna sweep (200-300 trials) on the asset's feature set. Don't assume BTC params transfer.

Search space:
```python
max_depth:         2-7
min_child_weight:  5-50
learning_rate:     0.005-0.1 (log scale)
n_estimators:      100-3000
subsample:         0.5-1.0
colsample_bytree:  0.3-1.0
lambda:            0.1-10.0 (log scale)
alpha:             0.01-5.0 (log scale)
gamma:             0.0-2.0
```

Optimize on AUC with early stopping on validation set.

### Step 6: Retrain & Walk-Forward Confirmation

1. Retrain with sweep-best hyperparameters
2. Re-run walk-forward to confirm OOS performance holds (not just val set improvement)
3. Re-check all breakdowns with new model to verify no regressions
4. Train production model on full dataset with best params
5. Save model + metadata with asset-specific paths

### Step 7: Live Validation

Deploy with $10/trade and monitor:
- First 20 trades: sanity check (fills working, predictions reasonable)
- First 50 trades: compare live WR to walk-forward WR
- If live WR is >10pp below walk-forward, stop and investigate (this is the mid-drift failure mode)
- After 100+ trades with acceptable WR, consider scaling to $20/trade

## Implementation Plan

### Phase 1: Data Collection (~1-2 days)

#### 1a. Parameterize Telonex Pipeline — DONE

`telonex_pipeline.py` accepts `--asset` parameter. Output directories: `data/telonex_book_snapshots_{asset}/`.

#### 1b. Download Historical Data — IN PROGRESS

```bash
python telonex_pipeline.py download --asset eth
python telonex_pipeline.py download --asset sol
python telonex_pipeline.py download --asset xrp
python telonex_pipeline.py normalize --asset eth
python telonex_pipeline.py normalize --asset sol
python telonex_pipeline.py normalize --asset xrp
```

#### 1c. Download Spot Quotes — DONE

Binance historical klines downloaded for ETH, SOL, XRP to `data/{asset}_quotes/`.

#### 1d. External Price Feeds — DONE

`BinanceFeed` parameterized. Config `externalFeeds: ["binance-ethusdt"]` auto-routes to correct WS stream.

### Phase 2: Per-Asset Model Training (~2-3 days)

Follow the full analysis process (Steps 1-6 above) for each asset independently.

#### 2a. Feature Extraction — DONE (parameterized)

`xgboost_flexible.py` accepts `--data-dir`, `--btc-path`, `--cache-path` params. Feature names keep `btc_` prefix for spot features regardless of asset.

#### 2b. Per-Asset Analysis

Run Steps 2-6 for each asset. Key outputs per asset:
- Walk-forward WR and PnL at different thresholds
- Optimal entry price cap
- Sweep-best hyperparameters
- Production model + metadata files

#### 2c. Gate Decisions

| Asset | Markets Available | Gate Status |
|-------|-------------------|-------------|
| BTC   | 6,300+            | LIVE (v3.3) |
| ETH   | TBD               | Pending data download |
| SOL   | TBD               | Pending data download |
| XRP   | TBD               | Pending data download |

### Phase 3: Execution Infrastructure — DONE

All code changes completed:

| File | Change | Status |
|------|--------|--------|
| `telonex_pipeline.py` | `--asset` param | Done |
| `xgboost_flexible.py` | `--data-dir`, `--btc-path`, `--cache-path` params | Done |
| `model_server.py` | `--model-path`, `--meta-path`, `--port` params | Done |
| `execution/src/config.ts` | `asset`, `modelServerUrl` fields | Done |
| `execution/src/rotation/market-rotation.ts` | Parameterized slug prefix | Done |
| `execution/src/strategy/xgb-confidence.ts` | Read `modelServerUrl`, `betDollars` from config | Done |
| `execution/src/data/binance-feed.ts` | Parameterized symbol | Done |
| Per-asset config files | `config.{eth,sol,xrp}.json` | Done |

### Phase 4: Deployment & Monitoring (~1 day)

#### 4a. Process Management

Launcher script:
```bash
#!/bin/bash
python model_server.py --port 8000 &
python model_server.py --port 8001 --model-path data/xgb_model_eth.json --meta-path data/xgb_model_meta_eth.json &
python model_server.py --port 8002 --model-path data/xgb_model_sol.json --meta-path data/xgb_model_meta_sol.json &
python model_server.py --port 8003 --model-path data/xgb_model_xrp.json --meta-path data/xgb_model_meta_xrp.json &

sleep 5

CONFIG=config.json     bun run execution/src/index.ts &
CONFIG=config.eth.json bun run execution/src/index.ts &
CONFIG=config.sol.json bun run execution/src/index.ts &
CONFIG=config.xrp.json bun run execution/src/index.ts &
```

#### 4b. Monitoring

Separate metrics directories per asset:
```
data/metrics/btc/  # orders.csv, fills.csv, trades.csv
data/metrics/eth/
data/metrics/sol/
data/metrics/xrp/
```

#### 4c. Capital Allocation

With $1,600 bankroll:
- BTC: $20/trade (proven)
- New assets: $10/trade until 50+ trades validate OOS performance
- Max concurrent exposure: 4 × $20 = $80 per 5-min window (5% of bankroll)
- Scale up individually as each asset proves profitable

## Risk Factors & Gates

### Must-verify before proceeding:

1. **Data availability**: Need 2,000+ historical markets on Telonex per asset. Below ~500, walk-forward validation is unreliable.

2. **Slug format verification**: Check Gamma API for exact slug pattern. `telonex_download.py` uses `f"{asset}-updown-{timeframe}"` — verified for BTC.

3. **Liquidity adequacy**: If median L1 depth < $10 at t=120s, the asset isn't tradeable at meaningful size.

4. **Walk-forward WR**: Each asset must show >= 70% OOS WR at conf >= 0.72 before going live.

5. **Limit fill validation**: Verify limit-at-prediction fill rates hold on the new asset's orderbook data. Thinner books may have different dynamics.

6. **Nonce conflict**: If using a shared wallet, test that simultaneous orders across instances don't cause nonce conflicts.

### Known risks:

- **Correlated losses**: BTC crash → all crypto assets drop → all 4 positions lose simultaneously. Position sizing must account for this.
- **Thinner books**: ETH/SOL/XRP 5-min markets likely have less liquidity than BTC. Entry prices may be worse.
- **Feature transferability**: BTC model features (depth patterns, spread dynamics) may not predict well on altcoins.
- **Live execution gap**: The mid-drift strategy lost 17pp live vs backtest. This failure mode hits harder on thin markets. Be prepared for an asset to fail live validation.

## BTC v3.3 Reference Results (baseline for comparison)

These are the numbers each new asset should aim to match or exceed.

- **Walk-forward OOS**: 73.9% WR, $0.54/trade avg, $732 total on 1,351 trades ($20 sizing)
- **Entry sweet spot**: 0.70-0.85 (85%+ WR, positive PnL)
- **Best confidence bin**: 0.72-0.75 ($1.52/trade avg — largest limit discount)
- **Limit-at-prediction**: 100% fill rate on ask-based check, 2x PnL vs buy-at-ask
- **Optimal hyperparams**: max_depth=2, lr=0.0148 (pending re-sweep with time features)
- **Time patterns**: 06-09 UTC strongest, 12-15 UTC weakest; Wed/Thu/Sat best days

## Recommended Execution Order

1. ~~Verify slug formats on Gamma API~~ — DONE
2. ~~Parameterize execution code~~ — DONE
3. ~~Download spot quotes~~ — DONE
4. Download + normalize orderbook data for ETH/SOL/XRP — IN PROGRESS
5. **GATE**: Count markets per asset. If <2,000, defer. If <500, don't attempt.
6. Run full analysis process (Steps 1-7) on ETH first
7. **GATE**: ETH walk-forward WR >= 70% at conf >= 0.72
8. Deploy ETH with $10/trade, monitor 24h
9. If ETH validates, repeat for SOL then XRP
10. Scale up proven assets to $20/trade

---

*Generated 2026-03-08. Updated 2026-03-09 with v3.3 feature space, analysis process, and BTC reference results.*
