# polybot

Trading framework for Polymarket's crypto Up/Down binary prediction markets. Covers the full pipeline: live data collection, historical data pipelines, strategy research, and live execution.

## Market Structure

Polymarket runs continuous BTC/ETH/SOL/XRP Up/Down binary markets on 5-minute and 4-hour timeframes. Each market has two tokens:

- **Up** — pays $1 if the asset is higher at close than at open
- **Down** — pays $1 if the asset is lower or unchanged

Tokens trade on Polymarket's CLOB between $0.00 and $1.00. Resolution uses Chainlink price feeds on Polygon.

## Current Strategy Focus

**Conditional entropy + Binance CEX price signal** (`entropy-binance`)

Two independent sub-problems:
1. **When to trade** (solved): Conditional entropy H(alt|others) over a rolling window of joint BTC/ETH/SOL/XRP outcomes. When entropy drops below threshold, cross-asset coupling is tight enough to predict alt outcomes from BTC direction.
2. **Which direction**: Binance BTC/USDT spot price return over the first 15 seconds of the market window. BTC price movement on CEX predicts the BTC binary outcome, which propagates to alts via 87-89% cross-asset coupling.

Backtest results (corrected for realistic entry prices at signal time):
- 15s delay, |ret| > 0.03%: 1,095 trades, 66.4% WR, +$477, ~$0.044/token edge
- See `strategies/backtest_entropy_binance.py` for the full sweep

## Repository Structure

```
polybot/
├── execution/          TypeScript/Bun live trading engine
│   └── src/
│       ├── strategy/   Pluggable strategy implementations
│       ├── data/       Market data sources, orderbook, Binance feed
│       ├── orders/     CLOB order management
│       ├── positions/  Position tracking, minting, redemption
│       ├── rotation/   Market window scheduling
│       └── dashboard/  HTTP/SSE monitoring dashboard
│
├── ingest/             Live WebSocket -> Redis Streams data collection
├── persistence/        Redis Streams -> Parquet file writer
│
├── pipelines/          Data download & normalization scripts
│   ├── telonex_download.py    Telonex historical data downloader
│   ├── telonex_pipeline.py    Raw -> normalized parquet
│   ├── btc_pipeline.py        Binance BTC/USDT quotes
│   ├── spot_pipeline.py       Binance multi-asset quotes
│   └── resolution_fetcher.py  Gamma API resolution cacher
│
├── strategies/         Active strategy backtests & models
│   ├── backtest_entropy_binance.py   Current focus strategy
│   ├── xgboost_flexible.py          XGBoost v3 training pipeline
│   ├── xgb_sweep.py                 Hyperparameter optimization
│   ├── xgb_walkforward.py           Walk-forward validation
│   ├── model_server.py              Python model server (live inference)
│   ├── backtest_mm_realistic.py     Market maker simulator
│   └── funding_arb.py               Hyperliquid funding rate monitor
│
├── analysis/           Live performance monitoring
│   ├── analyze_fills.py        Fill analysis from CSV logs
│   ├── analyze_live_pnl.py     Live PnL from CLOB + Gamma API
│   └── market_analysis.py      MarketLoader + ResolutionStore
│
├── notebooks/          Active research notebooks
├── docs/               Reference documentation
│   └── DATA_CATALOG.md         Complete data inventory
├── shared/             Shared protobuf definitions
├── swarm/              Multi-agent strategy research coordinator
├── archive/            Historical research (RL, XGB analysis, etc.)
│
├── docker-compose.yml  Live services (ingest + persistence)
├── config.example.json Config template
├── start_all.sh        Multi-asset execution launcher
└── polymarket.proto     Protobuf schema for Redis events
```

## Quick Start

### Live Data Collection
```bash
docker-compose up -d    # starts ingest + persistence services
```

### Historical Data Download
```bash
python pipelines/telonex_download.py --asset btc --timeframe 5m
python pipelines/telonex_pipeline.py --asset btc  # normalize raw -> canonical
python pipelines/btc_pipeline.py                   # Binance BTC quotes
```

### Run Execution Engine
```bash
cp config.example.json config.json  # edit with API keys
cd execution && bun run src/index.ts
```

### Run Strategy Backtest
```bash
python strategies/backtest_entropy_binance.py
```

## Data

See `docs/DATA_CATALOG.md` for a complete inventory of all datasets with schemas, date ranges, and storage sizes. Key datasets:

- **5m orderbook snapshots** (BTC/ETH/SOL/XRP): ~26 GB, Feb 12 - Mar 8, 2026
- **Binance CEX quotes** (4 assets): ~400 MB, 1-second resolution
- **Live ingest** (book + trade events): ~80 GB, Feb 17 - Mar 13, 2026

## Strategy History

| Strategy | Status | Notes |
|----------|--------|-------|
| Entropy + Binance CEX | **Active** | ~$0.04/token edge, needs multi-asset orchestration |
| XGBoost v3.3 | Completed | 85.8% WR backtest, not profitable live (spread ate edge) |
| Market maker (mint-and-sell) | Tested | Slightly profitable until BTC whipsaw |
| Mid-price drift | Failed live | 71% backtest, 54% live (17pp gap) |
| Deep-ratio orderbook | Failed | 56% on sample, 51% on full dataset |
| RL (PPO+LSTM) | Abandoned | Superseded by XGBoost approach |
