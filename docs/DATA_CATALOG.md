# Data Catalog

Complete inventory of all data in `data/` and `datasets/` directories.

Last updated: 2026-03-12

---

## data/ Directory (Active Datasets)

These are the processed, normalized, and actively-used datasets.

### Canonical 5m Orderbook Snapshots (Normalized)

| Directory | Size | Files | Date Range | Asset |
|-----------|------|-------|------------|-------|
| `telonex_btc_5m_book_snapshots/` | 14 GB | 7,186 | Feb 12 -- Mar 8, 2026 | BTC |
| `telonex_eth_5m_book_snapshots/` | 5.5 GB | 5,231 | Feb 18 -- Mar 8, 2026 | ETH |
| `telonex_sol_5m_book_snapshots/` | 3.6 GB | 5,231 | Feb 18 -- Mar 8, 2026 | SOL |
| `telonex_xrp_5m_book_snapshots/` | 3.3 GB | 5,230 | Feb 18 -- Mar 8, 2026 | XRP |

**Schema (29 columns):**
`exchange_timestamp`, `slug`, `token_label` (Up/Down), `mid_price`, `spread`, `book_imbalance`,
`bid_price_1`..`bid_price_5`, `bid_size_1`..`bid_size_5`,
`ask_price_1`..`ask_price_5`, `ask_size_1`..`ask_size_5`

Notes:
- These are the **primary datasets** for backtesting and feature engineering.
- `token_label` uses "Up"/"Down" (not "Yes"/"No" as in live ingest).
- 5 levels of book depth per side.
- Produced by `telonex_pipeline.py` from raw downloads in `datasets/`.

### 4h Orderbook Snapshots

| Directory | Size | Files | Date Range | Asset |
|-----------|------|-------|------------|-------|
| `telonex_btc_4h_book_snapshots/` | ~500 MB | 810 | Oct 2025 -- Mar 2026 | BTC |

Same schema as the 5m snapshots. Used for whale signal analysis and market-maker strategy research on longer-duration markets.

### Binance CEX Quotes

| Directory | Size | Files | Resolution | Asset |
|-----------|------|-------|------------|-------|
| `btc_quotes/` | single parquet | 1 | 1s | BTC |
| `eth_quotes/` | single parquet | 1 | 1s | ETH |
| `sol_quotes/` | single parquet | 1 | 1s | SOL |
| `xrp_quotes/` | single parquet | 1 | 1s | XRP |

**Schema (12 columns):**
`timestamp_ms`, `mid_price`, `spread`, `bid_price`, `ask_price`, `log_return`,
`rvol_30s`, `rvol_60s`, `rvol_300s`, `ret_5s`, `ret_15s`, `ret_60s`

Notes:
- 1-second resolution Binance mid-quotes with pre-computed returns and realized volatility.
- Used as features for XGBoost and other models (CEX price leads Polymarket).

### Live Ingest Data (Raw, 10-Level)

| Directory | Size | Files | Date Range | Description |
|-----------|------|-------|------------|-------------|
| `book_snapshots/` | 31 GB | 5,947 | Feb 17 -- Mar 13, 2026 | 10-level orderbook snapshots |
| `trade_events/` | 50 GB | 5,947 | Feb 17 -- Mar 13, 2026 | Trade-level data |

**book_snapshots schema (50 columns):**
`exchange_timestamp`, `market`, `asset_id`, plus 10 levels each of `bid_price`/`bid_size`/`ask_price`/`ask_size`.
**No `token_label` column** -- only raw `asset_id`. This is the key difference from the normalized Telonex data.

**trade_events schema (12 columns):**
`exchange_timestamp`, `market`, `asset_id`, `side`, `trade_price`, `size`, and additional trade metadata.

Notes:
- Produced by the live WebSocket ingest pipeline (Redis Streams -> Parquet).
- Higher resolution than Telonex data but lacks token labels.
- `book_snapshots/` has 10 bid/ask levels vs 5 in the Telonex normalized data.

### 100ms Resampled Orderbook

| Directory | Size | Files | Date Range |
|-----------|------|-------|------------|
| `telonex_100ms/` | 945 MB | 3,731 | Feb 12 -- Feb 24, 2026 |

Sub-second resampled orderbook data. Used for high-frequency signal research (e.g., whale pull duration analysis).

### 15m Orderbook (Under Review -- May Be Deleted)

| Directory | Size | Files | Description |
|-----------|------|-------|-------------|
| `telonex_15m_book_snapshots/` | 25 GB | -- | 15-minute normalized snapshots |
| `telonex_btc_15m_book_snapshots/` | 44 GB | -- | BTC-only 15-minute snapshots |
| `telonex_15m_100ms/` | 3.2 GB | -- | 15-minute at 100ms resolution |

Notes:
- These are large and may be redundant with the 5m data.
- Candidates for cleanup/deletion pending review.

### Sample Dataset

| Directory | Size | Markets | Description |
|-----------|------|---------|-------------|
| `sample_1k/` | 361 MB | 917 | Subset for fast experimentation |

Used by swarm agents for quick strategy iteration before running on full datasets.

### Metadata Files

| File | Size | Description |
|------|------|-------------|
| `resolution_cache.json` | -- | 6,571 market resolutions (latest) |
| `resolutions.json` | -- | 4,301 market resolutions (older snapshot) |
| `15m_resolutions.json` | -- | 4,519 resolutions for 15m markets |
| `observations.parquet` | 156 MB | Old live ingest aggregate (9.8M rows) |
| `xgb_features_*.parquet` | varies | Cached ML feature matrices (regenerable from raw data) |

---

## datasets/ Directory (Raw Telonex Downloads)

These are the **raw downloads** before normalization. The `data/` versions are the processed/canonical copies. Do not use these directly for analysis -- use the normalized versions in `data/`.

### Raw Orderbook Downloads

| Directory | Files | Size | Timeframe | Date Range |
|-----------|-------|------|-----------|------------|
| `telonex_btc_5m_raw/` | 28,108 | 13 GB | 5m | Feb 12 -- Mar 8, 2026 |
| `telonex_btc_4h_raw/` | 2,686 | 562 MB | 4h | Oct 2025 -- Mar 2026 |
| `telonex_btc_15m_raw/` | 36,960 | 40 GB | 15m | -- |
| `telonex_eth_5m_raw/` | 20,828 | 5.1 GB | 5m | Feb 18 -- Mar 8, 2026 |
| `telonex_sol_5m_raw/` | 20,822 | 3.5 GB | 5m | Feb 18 -- Mar 8, 2026 |
| `telonex_xrp_5m_raw/` | 20,846 | 3.3 GB | 5m | Feb 18 -- Mar 8, 2026 |
| `telonex_15m_raw/` | 18,034 | 21 GB | 15m | -- |
| `telonex_raw/` | -- | 7.4 GB | mixed | -- |

**Raw schema (27 columns):**
`timestamp_us`, `local_timestamp_us`, `exchange`, `market_id`, `slug`, `asset_id`, `outcome`,
`bid_price_0`..`bid_price_4`, `bid_size_0`..`bid_size_4`,
`ask_price_0`..`ask_price_4`, `ask_size_0`..`ask_size_4`

Notes:
- Raw files use 0-indexed columns (`bid_price_0`) vs 1-indexed in normalized (`bid_price_1`).
- Raw files have `outcome` instead of `token_label`.
- Multiple raw files per market (one per asset/token).

### Sports Data

| Directory | Size | Description |
|-----------|------|-------------|
| `sports_longshot/` | 20 GB | NBA/NFL orderbook snapshots for longshot bias research |

Used by `notebooks/sports_longshot_backtest.py`. Analysis showed the favorite-longshot bias is real (~$0.07/trade) but spread absorbs it, making it not scalable.

### Market Index

| File | Size | Rows | Description |
|------|------|------|-------------|
| `telonex_markets_index.parquet` | 309 MB | 622K | Master catalog of all available Telonex markets |

---

## Data Flow

```
Raw Telonex download (datasets/*_raw/)
        |
        v
  telonex_pipeline.py (normalize, add token_label, reindex columns)
        |
        v
Normalized snapshots (data/telonex_*_5m_book_snapshots/)
        |
        v
  Feature engineering (xgboost_confidence.py, etc.)
        |
        v
  xgb_features_*.parquet (cached, regenerable)
```

```
Live WebSocket ingest (Redis Streams, protobuf)
        |
        v
  data/book_snapshots/  (10-level, no token_label)
  data/trade_events/    (trade-level)
```

## Key Gotchas

- **Token label mismatch**: Live ingest uses "Yes"/"No"; Telonex data uses "Up"/"Down". Always check which convention a dataset uses before joining.
- **Book depth**: Telonex normalized = 5 levels; live ingest = 10 levels. Column counts differ accordingly.
- **Column indexing**: Raw Telonex uses 0-indexed (`bid_price_0`); normalized uses 1-indexed (`bid_price_1`).
- **`book_snapshots/` has no `token_label`**: Only raw `asset_id`. Must join against market metadata to determine Up vs Down.
- **15m data is under review**: Large (~72 GB total) and potentially redundant. Check before building pipelines against it.
