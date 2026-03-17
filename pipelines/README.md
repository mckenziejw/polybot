# Pipelines

Python scripts for downloading and normalizing market data from Telonex and the Gamma API. All scripts require `TELONEX_API_KEY` set in the environment (or in `config.json` for `telonex_pipeline.py`).

---

## telonex_download.py

**Purpose:** Generic Telonex orderbook snapshot downloader and normalizer for any asset (BTC, ETH, SOL, etc.) and any timeframe (5m, 15m, 1h, 4h).

**Usage:**
```bash
python telonex_download.py <asset> <timeframe> <step>
# Examples:
python telonex_download.py btc 15m download
python telonex_download.py btc 15m normalize
python telonex_download.py btc 15m all
python telonex_download.py sol 5m all
```

**Input/Output:**
- Raw download: `./datasets/telonex_<asset>_<timeframe>_raw/*.parquet`
- Normalized output: `./data/telonex_book_snapshots/` (BTC 5m) or `./data/telonex_<asset>_<timeframe>_book_snapshots/` (others)
- One parquet file per market slug.

**Output schema:**
| Column | Type | Description |
|---|---|---|
| `exchange_timestamp` | int64 | Exchange-side timestamp (ms) |
| `received_timestamp` | int64 | Telonex receive timestamp (ms) |
| `slug` | string | Market slug (e.g. `btc-updown-15m-2026-03-12-1200`) |
| `market` | string | Market identifier |
| `asset_id` | string | CLOB token ID |
| `token_label` | string | `Up` or `Down` |
| `mid_price` | float64 | (best_bid + best_ask) / 2 |
| `spread` | float64 | best_ask - best_bid |
| `book_imbalance` | float64 | Bid/ask size imbalance at top of book |
| `bid_price_1..5` | float64 | Bid prices, 5 levels |
| `bid_size_1..5` | float64 | Bid sizes, 5 levels |
| `ask_price_1..5` | float64 | Ask prices, 5 levels |
| `ask_size_1..5` | float64 | Ask sizes, 5 levels |

---

## telonex_pipeline.py

**Purpose:** Telonex 5m orderbook snapshot downloader and normalizer. Older, BTC-focused variant of `telonex_download.py` with multi-asset support via `--asset` flag.

**Usage:**
```bash
python telonex_pipeline.py download                  # BTC (default)
python telonex_pipeline.py normalize --asset eth     # ETH normalize only
python telonex_pipeline.py all --asset sol            # SOL download + normalize
```

**Input/Output:**
- Raw download: `./datasets/telonex_5m_raw/` (BTC) or `./datasets/telonex_<asset>_5m_raw/`
- Normalized output: `./data/telonex_book_snapshots/` (BTC) or `./data/telonex_book_snapshots_<asset>/`
- One parquet file per market slug.

**Output schema:** Same as `telonex_download.py` (see above).

---

## btc_pipeline.py

**Purpose:** Downloads Binance BTC/USDT quote data from Telonex and normalizes to 1-second bars with derived volatility and return features.

**Usage:**
```bash
python btc_pipeline.py download
python btc_pipeline.py normalize
python btc_pipeline.py all
```

**Input/Output:**
- Raw download: `./datasets/telonex_raw/binance/*.parquet` (daily files)
- Normalized output: `./data/btc_quotes/btcusdt_quotes.parquet` (single file)
- Date range: 2025-10-01 to 2026-03-09

**Output schema:**
| Column | Type | Description |
|---|---|---|
| `timestamp_ms` | int64 | 1s bar open timestamp (ms) |
| `mid_price` | float64 | (bid + ask) / 2 |
| `spread` | float64 | ask - bid |
| `bid_price` | float64 | Last bid in bar |
| `ask_price` | float64 | Last ask in bar |
| `log_return` | float64 | log(mid_t / mid_{t-1}) |
| `rvol_30s` | float64 | Rolling realized vol, 30s window |
| `rvol_60s` | float64 | Rolling realized vol, 60s window |
| `rvol_300s` | float64 | Rolling realized vol, 300s window |
| `ret_5s` | float64 | Trailing 5s log return |
| `ret_15s` | float64 | Trailing 15s log return |
| `ret_60s` | float64 | Trailing 60s log return |

---

## spot_pipeline.py

**Purpose:** Multi-asset Binance spot quote downloader and normalizer. Generalized version of `btc_pipeline.py` supporting ETH, SOL, XRP, and any other Binance pair.

**Usage:**
```bash
python spot_pipeline.py btcusdt download
python spot_pipeline.py ethusdt normalize
python spot_pipeline.py solusdt all
python spot_pipeline.py xrpusdt all
```

**Input/Output:**
- Raw download: `./datasets/telonex_raw/binance_<symbol>/*.parquet` (daily files)
- Normalized output: `./data/<asset>_quotes/<symbol>_quotes.parquet` (single file, e.g. `./data/eth_quotes/ethusdt_quotes.parquet`)
- Date range: 2025-10-01 to 2026-03-09

**Output schema:** Same as `btc_pipeline.py` (see above).

---

## resolution_fetcher.py

**Purpose:** Fetches and caches Polymarket market resolution outcomes from the Gamma API. Maps each token ID to its payout (1.0 = won, 0.0 = lost).

**Usage:**
```python
from resolution_fetcher import ResolutionFetcher

fetcher = ResolutionFetcher()
result = fetcher.fetch_resolution("btc-updown-5m-2026-03-10-1200")
results = fetcher.fetch_batch(list_of_slugs)
```

**Input/Output:**
- Cache file: `./data/resolution_cache.json`
- API source: `https://gamma-api.polymarket.com/markets/slug/<slug>`
- Rate-limited to 5 req/s (0.2s delay between requests)

**Output schema (per slug):**
| Field | Type | Description |
|---|---|---|
| `slug` | string | Market slug |
| `condition_id` | string | On-chain condition ID |
| `closed_time` | string | ISO timestamp when market resolved |
| `token_outcomes` | dict | `{token_id: 1.0 or 0.0}` for each token |
| `winning_token` | string | Token ID that paid out |
