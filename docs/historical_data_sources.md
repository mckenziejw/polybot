# Historical Data Sources for Backtesting

Reference for available crypto market data sources, organized by cost and granularity.

## What We Have

| Source | Asset | Granularity | Rows | Location |
|--------|-------|-------------|------|----------|
| Binance CEX quotes | BTC, ETH, SOL, XRP | 1-second bid/ask | 2.6M (BTC, 720h) | `data/{btc,eth,sol,xrp}_quotes/` |
| Polymarket orderbook | BTC 5m/4h markets | 5-minute L5 snapshots | 7,186 files (BTC 5m) | `data/telonex_*_book_snapshots/` |
| Polymarket 100ms | BTC markets | 100ms orderbook | 3,731 files | `data/telonex_100ms/` |
| Polymarket trades | BTC markets | Tick-level | 5,339 files | `data/trade_events/` |

Currently using the 1s Binance quotes for the evolution system POC.

## Free Exchange Data

### Binance (data.binance.vision)

Portal: https://data.binance.vision/
GitHub: https://github.com/binance/binance-public-data

| Data Type | Granularity | Format | Notes |
|-----------|-------------|--------|-------|
| Individual trades | Tick (μs since Jan 2025) | CSV (zipped) | Every fill, massive volume |
| AggTrades | Tick (aggregated) | CSV (zipped) | Grouped by price level |
| Klines (OHLCV) | 1s to 1mo | CSV (zipped) | 1-second is finest candle |
| Orderbook snapshots | 1-minute, ~20 levels | CSV (zipped) | Very coarse for microstructure |

- Covers both spot and USDT-M perpetual futures
- No auth required, no rate limits on downloads
- Sub-second orderbook NOT available historically (only via live 100ms WebSocket)
- URL pattern: `https://data.binance.vision/data/spot/monthly/trades/BTCUSDT/BTCUSDT-trades-2025-03.zip`

**Best free upgrade path**: Download tick trades for higher frequency than our current 1s quotes.

### Bybit (bybit.com/derivatives/en/history-data)

Portal: https://www.bybit.com/derivatives/en/history-data

| Data Type | Granularity | Format | Notes |
|-----------|-------------|--------|-------|
| Individual trades | Tick (ms precision) | CSV | Since 2020 |
| Klines (OHLCV) | 1-minute minimum | CSV | Coarser than Binance |
| Orderbook L2 | Snapshots | CSV | **Only since May 2025** |

- Free, no auth required
- Download limited to 7 calendar days per request
- Orderbook depth up to 1,000 levels (vs Binance 5,000)
- Less historical depth than Binance overall
- We have an existing Bybit account (useful for live trading)

## Paid Third-Party Sources

### Tardis.dev — Gold Standard

Site: https://tardis.dev/
Docs: https://docs.tardis.dev/

- **Granularity**: Tick-level, 100ns synchronized clock
- **Orderbook**: Full L2/L3 reconstruction from incremental updates
- **History**: 7 years, 20+ exchanges, 200K+ instruments
- **Format**: NDJSON via HTTP/WebSocket, normalized or exchange-native
- **Free access**: First day of each month (no API key needed), free trial available
- **Cost**: Contact sales (subscription tiers: Academic, Solo, Pro, Business)
- **Client libraries**: Node.js, Python

Best for: Serious HFT research, full orderbook reconstruction.

### Crypto Lake — Best Value for Orderbook Data

Site: https://crypto-lake.com/

- **Granularity**: 100ms L2 orderbook snapshots (20 levels each side), tick trades, 1m OHLCV
- **Exchanges**: 10 major exchanges
- **Extras**: Open interest, funding rates
- **Access**: Python API with parallel download, caching, incremental updates
- **Free tier**: Yes — sample datasets available
- **Cost**: Paid tier for full coverage (pricing not disclosed)

Best for: Cost-effective orderbook data, Python-native workflow. **Primary candidate for when we need historical orderbook depth.**

### Kaiko — Enterprise

Site: https://www.kaiko.com/

- **Granularity**: Tick-level L2 updates, full depth
- **Exchanges**: 100+
- **Cost**: $9,500–$55,000/year (enterprise sales model)
- **Free trial**: Yes (request via website)

Overkill for our needs. Enterprise-grade, institutional pricing.

### CoinAPI / CryptoTick

Site: https://www.coinapi.io/ / https://www.cryptotick.com/

- **Granularity**: Tick-level trades, L2x20 orderbook (REST), full depth (flat files)
- **Exchanges**: 400+
- **Format**: CSV flat files via S3, or REST API (JSON)
- **Cost**: Credit-based (REST) or usage-based (flat files)

Broad exchange coverage but less specialized than Tardis for orderbook work.

### Databento — CME Only

Site: https://databento.com/

- **Coverage**: CME Bitcoin Futures, GBTC ETF
- **History**: Starting Sept 2025
- **Free credits**: $125 for new users
- Not useful for spot/perp crypto on Binance/Bybit.

## Free / Open Source

- **Kaggle**: Various BTC datasets (1m OHLCV, sparse LOB). Too coarse for serious work.
  - Bitcoin LOB data: https://www.kaggle.com/datasets/siavashraz/bitcoin-perpetualbtcusdtp-limit-order-book-data
- **GitHub tools**: `binance-historical-data` (PyPI), `bybit-bulk-downloader`, etc.

## Upgrade Path

1. **Now**: 1s Binance quotes (what we have) — sufficient for testing evolution system mechanics
2. **Next**: Binance tick trades from data.binance.vision (free, microsecond precision) — includes orderbook snapshots
3. **When ready for deeper orderbook**: Crypto Lake free tier (100ms L2, 20 levels)
4. **If we need more**: Tardis.dev for full orderbook reconstruction

### Binance Tick Data + Orderbook (Free, Next Priority)
Binance data.binance.vision provides free historical L2 orderbook snapshots alongside tick trade data. When we're ready to add orderbook-based indicators (imbalance, depth ratios, etc.) to the gene pool, this is the first data source to integrate — no cost, just download and parse. Gene pool expansion should include: bid/ask imbalance at N levels, depth-weighted mid price, orderbook slope, volume-at-price clustering.
