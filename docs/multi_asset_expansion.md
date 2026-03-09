# Multi-Asset Expansion Plan: ETH, SOL, XRP 5-Minute Markets

## Status: IN PROGRESS — code parameterized, data downloading

## Overview

Expand the current BTC-only XGBoost confidence strategy to trade ETH, SOL, and XRP Up/Down 5-minute markets on Polymarket. The BTC strategy (v3.2) runs at 85%+ WR live with $20/trade. These additional assets share the same market structure (binary 5-minute up/down) but may have different microstructure characteristics (liquidity, volatility patterns, feature importance).

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

## Implementation Plan

### Phase 1: Data Collection (~1-2 days)

#### 1a. Parameterize Telonex Pipeline

Modify `telonex_pipeline.py` to accept an `--asset` parameter:

```python
# Current: hardcoded to BTC
SLUG_PATTERN = r"btc-updown-5m-\d+"

# New: parameterized
ASSET_SLUG_PATTERNS = {
    "btc": r"btc-updown-5m-\d+",
    "eth": r"eth-updown-5m-\d+",
    "sol": r"sol-updown-5m-\d+",
    "xrp": r"xrp-updown-5m-\d+",
}
```

Output directories: `data/telonex_book_snapshots_{asset}/` (keep BTC in current location for backward compatibility).

**Critical**: Verify the exact slug format for ETH/SOL/XRP on Gamma API before downloading. The pattern may differ (e.g., `ethereum-updown-5m-*` vs `eth-updown-5m-*`).

#### 1b. Download Historical Data

```bash
python telonex_pipeline.py download --asset eth
python telonex_pipeline.py download --asset sol
python telonex_pipeline.py download --asset xrp
python telonex_pipeline.py normalize --asset eth
python telonex_pipeline.py normalize --asset sol
python telonex_pipeline.py normalize --asset xrp
```

#### 1c. Add External Price Feeds

Existing: `BinanceFeed` for BTC/USDT.
Add: ETH/USDT, SOL/USDT, XRP/USDT feeds. The `ExternalFeed` interface already supports this — just need new feed instances with different symbols.

For each asset's model, the "spot" feed should be its own asset (ETH model uses ETH/USDT prices), not BTC. The BTC features in the current model (btc_vol_5m, btc_ret_*, etc.) need to become `spot_vol_5m`, `spot_ret_*` etc.

### Phase 2: Per-Asset Model Training (~2-3 days)

#### 2a. Parameterize Feature Extraction

`xgboost_flexible.py` currently hardcodes BTC paths:
- `btc_path=Path("data/btc_quotes/btcusdt_quotes.parquet")`
- Feature names: `btc_vol_5m`, `btc_ret_3600s`, `btc_rvol_*`, etc.

**Two options** (choose one):

**Option A (recommended): Keep `btc_` prefix for all spot features.**
The XGBoost model doesn't care about feature names — it only cares about the values. Rename nothing. When training the ETH model, just point `btc_path` to the ETH/USDT quotes file. The feature called `btc_vol_5m` will actually contain ETH vol. This avoids touching 97 feature columns and all downstream code.

**Option B: Rename to `spot_` prefix.**
Cleaner semantically but requires updating feature extraction, model server, feature cache, walk-forward scripts, and all analysis scripts. Not worth the refactoring cost for zero functional benefit.

#### 2b. Build Feature Caches Per Asset

```bash
python xgboost_flexible.py --rebuild-cache --data-dir data/telonex_book_snapshots_eth --btc-path data/eth_quotes/ethusdt_quotes.parquet --cache-path data/features_eth.parquet
python xgboost_flexible.py --rebuild-cache --data-dir data/telonex_book_snapshots_sol --btc-path data/sol_quotes/solusdt_quotes.parquet --cache-path data/features_sol.parquet
python xgboost_flexible.py --rebuild-cache --data-dir data/telonex_book_snapshots_xrp --btc-path data/xrp_quotes/xrpusdt_quotes.parquet --cache-path data/features_xrp.parquet
```

#### 2c. Train & Validate Per-Asset Models

For each asset:
1. Run walk-forward backtest (`xgb_walkforward.py` parameterized per asset)
2. Check OOS WR >= 75% at conf >= 0.72 (minimum viable)
3. Analyze entry price distribution — may need different caps per asset
4. Check liquidity (depth at t=120s) — position sizing may differ
5. Run position sizing analysis per asset

**Key question**: Do ETH/SOL/XRP markets have enough historical data? BTC has 6,300+ markets. If ETH only has 500, the model may not generalize. **Gate on data availability before proceeding.**

#### 2d. Hyperparameter Tuning

Don't assume BTC's optimal params (max_depth=2, lr=0.0148, etc.) transfer. Run a quick Optuna sweep per asset (100-200 trials, not 500). Different assets may need different regularization.

### Phase 3: Execution Infrastructure (~1-2 days)

#### 3a. Parameterize Market Rotation

`MarketRotation.generateSlug()` currently hardcodes `btc-updown-5m-`:

```typescript
// Current
static generateSlug(timestampS: number): string {
  const aligned = Math.floor(timestampS / FIVE_MINUTES_S) * FIVE_MINUTES_S;
  return `btc-updown-5m-${aligned}`;
}

// New: accept asset parameter
static generateSlug(timestampS: number, asset: string = "btc"): string {
  const aligned = Math.floor(timestampS / FIVE_MINUTES_S) * FIVE_MINUTES_S;
  return `${asset}-updown-5m-${aligned}`;
}
```

Add `execution.asset` to config:
```json
{
  "execution": {
    "asset": "eth",
    "strategyId": "xgb-confidence",
    "externalFeeds": ["binance-ethusdt"],
    "betDollars": 10
  }
}
```

#### 3b. Parameterize External Feeds

`BinanceFeed` currently hardcodes `btcusdt@bookTicker`. Make symbol configurable:

```typescript
// Config-driven: "binance-ethusdt" → connects to ethusdt@bookTicker
const feed = createExternalFeed("binance-ethusdt");
```

The `createExternalFeed` factory already parses the feed name — just needs to extract the symbol suffix and use it for the WS subscription.

#### 3c. Per-Asset Config Files

```
config.json        # BTC (existing, unchanged)
config.eth.json    # ETH instance
config.sol.json    # SOL instance
config.xrp.json    # XRP instance
```

Each config has:
- Same `polymarket` credentials (shared wallet) OR separate proxy wallets
- Asset-specific `execution.asset`, `execution.betDollars`, `execution.externalFeeds`
- Separate `execution.metricsDir` for isolated logging

#### 3d. Per-Asset Model Servers

Each model server loads its own asset's model:
```bash
python model_server.py --port 8000  # BTC (existing)
python model_server.py --port 8001 --model-path data/xgb_model_eth.json --meta-path data/xgb_model_meta_eth.json
python model_server.py --port 8002 --model-path data/xgb_model_sol.json --meta-path data/xgb_model_meta_sol.json
python model_server.py --port 8003 --model-path data/xgb_model_xrp.json --meta-path data/xgb_model_meta_xrp.json
```

Add `--model-path` and `--meta-path` CLI args to model_server.py (partially exists via `--model-path`, need `--meta-path`).

Add `execution.modelServerUrl` to config (currently hardcoded to `http://127.0.0.1:8000` in xgb-confidence.ts).

### Phase 4: Deployment & Monitoring (~1 day)

#### 4a. Process Management

Use systemd units or a simple bash launcher:
```bash
#!/bin/bash
# start_all.sh
python model_server.py --port 8000 &
python model_server.py --port 8001 --model-path data/xgb_model_eth.json &
python model_server.py --port 8002 --model-path data/xgb_model_sol.json &
python model_server.py --port 8003 --model-path data/xgb_model_xrp.json &

sleep 5  # let model servers start

CONFIG=config.json     bun run execution/src/index.ts &
CONFIG=config.eth.json bun run execution/src/index.ts &
CONFIG=config.sol.json bun run execution/src/index.ts &
CONFIG=config.xrp.json bun run execution/src/index.ts &
```

#### 4b. Monitoring

Separate metrics directories per asset. Unified monitoring script that reads all:
```bash
data/metrics/btc/  # orders.csv, fills.csv, trades.csv
data/metrics/eth/
data/metrics/sol/
data/metrics/xrp/
```

#### 4c. Capital Allocation

With $1,600 bankroll and Kelly analysis showing safe sizing up to $100/trade:
- Start conservative: $10/trade per new asset until 50+ trades validate OOS performance
- BTC: keep at $20/trade (proven)
- Max concurrent exposure: 4 × $20 = $80 per 5-min window (5% of bankroll)
- Scale up individually as each asset proves profitable

## Risk Factors & Gates

### Must-verify before proceeding:

1. **Data availability**: Do ETH/SOL/XRP have 2,000+ historical markets on Telonex? Below ~500 markets, walk-forward validation is unreliable. Below ~2,000, model quality is suspect. Run `telonex_download.py` for each asset and count before writing any code.

2. **Slug format verification**: Check Gamma API for exact slug pattern per asset. Don't assume `{ticker}-updown-5m-{ts}`. The `telonex_download.py` already uses `f"{asset}-updown-{timeframe}"` (line 72) but verify this matches Gamma.

3. **Liquidity adequacy**: Run `analyze_liquidity.py` per asset. If median L1 depth < $10, the asset isn't tradeable at meaningful size.

4. **Walk-forward WR**: Each asset must show >= 75% OOS WR in walk-forward before going live. The BTC model's features may not transfer — orderbook dynamics differ per asset. Confidence threshold (0.72) was calibrated on BTC distribution — each asset needs its own threshold tuning.

5. **Nonce conflict**: If using a shared wallet, test that simultaneous orders across instances don't cause nonce conflicts. If they do, need separate proxy wallets (requires funding each).

### Known risks:

- **Correlated losses**: BTC crash → all crypto assets drop → all 4 positions lose simultaneously. Position sizing must account for this (total exposure across all assets, not per-asset).
- **Thinner books**: ETH/SOL/XRP 5-min markets likely have less liquidity than BTC. Entry prices may be worse, reducing edge.
- **Feature transferability**: BTC model features (depth patterns, spread dynamics) may not predict well on altcoins with different market microstructure.

## Code Changes Summary

| File | Change | Scope |
|------|--------|-------|
| `telonex_pipeline.py` | Add `--asset` param, asset-specific slug patterns | Small |
| `xgboost_flexible.py` | Add `--data-dir`, `--btc-path`, `--cache-path` params | Small |
| `model_server.py` | Add `--meta-path` param | Tiny |
| `xgboost_flexible.py` (line 648) | Parameterize `btc-updown-5m-*.parquet` glob | Tiny |
| `xgb_walkforward.py` | Parameterize data/model paths | Small |
| `execution/src/config.ts` | Add `asset`, `modelServerUrl` fields | Tiny |
| `execution/src/rotation/market-rotation.ts` | Parameterize slug prefix | Tiny |
| `execution/src/strategy/xgb-confidence.ts` | Read `modelServerUrl` from config | Tiny |
| `execution/src/data/binance-feed.ts` | Parameterize symbol | Small |
| `execution/src/data/external-feed.ts` | Add factory entries for ethusdt, solusdt, xrpusdt | Tiny |
| New: per-asset config files | `config.{eth,sol,xrp}.json` | New files |

**Total estimated diff**: ~200 lines changed, 3 new config files. The architecture is already modular enough that multi-asset support is mostly a configuration exercise.

## Recommended Execution Order

1. Verify slug formats on Gamma API (5 min)
2. Check Telonex data availability per asset — run `telonex_download.py` for eth/sol/xrp (10 min)
3. **GATE**: If <2,000 markets for an asset, defer it. If <500, don't attempt.
4. Download + normalize data for viable assets
5. Download spot price quotes (Binance historical klines)
6. Build feature caches, run walk-forward per asset
7. **GATE**: If OOS WR < 75%, don't deploy that asset
8. **Start with ONE new asset (ETH)** — validate end-to-end before adding SOL/XRP
9. Parameterize execution code (config, rotation, feeds)
10. Deploy with $10/trade, monitor for 24h
11. Add SOL/XRP only after ETH is validated live
12. Scale up proven assets to $20/trade

**Important**: The same "signals that look good on historical data don't survive live" failure mode (mid-price drift: 71% backtest → 54% live) will likely hit altcoins harder due to thinner liquidity. Be prepared for an asset to fail live validation even with good backtest results.

---

*Generated 2026-03-08. Synthesized from original plan + code-focused evaluation (gaps in nonce isolation, index.ts rewrite scope, feature naming) + architecture evaluation (concurrency, failure modes, capital management).*
