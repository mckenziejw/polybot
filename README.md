# polybot

A data collection, analysis, and (eventually) RL trading framework for Polymarket's BTC Up/Down binary prediction markets.

## Project Status

**Current phase: Data collection + edge analysis**

The project has moved through several phases:

1. ~~Momentum strategy~~ — falsified. Buying momentum in these markets loses money.
2. **Calibration analysis** — complete. The market is broadly efficient in aggregate across price buckets, time windows, and BTC volatility regimes. No exploitable edge found in aggregate statistics.
3. **Oracle lag analysis** — in progress. Awaiting Polymarket's Chainlink Data Streams API key to measure resolution price accuracy precisely.
4. **RL agent** — planned. Target architecture is PPO with invalid action masking (MaskablePPO), trained on accumulated episode history once sufficient data is collected.

---

## Market Structure

Polymarket runs 5-minute BTC Up/Down binary markets continuously. Each market has two tokens:

- **Yes (Up)** — pays $1 if BTC price is higher at close than at open
- **No (Down)** — pays $1 if BTC price is lower or unchanged

Tokens trade on Polymarket's CLOB between $0.00 and $1.00. Resolution is determined by Chainlink's BTC/USD price feed on Polygon (0.1% deviation threshold, ~32 second average update interval).

These markets have structural properties that make them distinct from standard continuous financial markets:
- Hard episode boundaries — each 5-minute market is fully isolated
- Known termination time — time-to-close is always known precisely
- Prices are probability estimates, not asset values
- Dual-token sum constraint (Yes + No ≈ 1 minus platform take)
- Liquidity collapses in the final ~10 seconds before close

---

## Repository Structure

```
polybot/
├── ingest/
│   └── src/
│       └── market_client.py     # WebSocket client for live orderbook data
├── bot/
│   ├── engine.py                # Trading engine (order execution, position mgmt)
│   └── notifications.py        # Notification service (console, extensible to webhooks)
├── shared/
│   ├── polymarket_pb2.py        # Protobuf definitions for WebSocket messages
│   └── polymarket.proto         # Protobuf schema
├── persistence/                 # Storage utilities
├── notebooks/                   # Exploratory analysis notebooks
├── market_analysis.py           # Core analysis pipeline (MarketLoader, ResolutionStore, CalibrationAnalyser)
├── btc_klines.py                # Binance 1-minute OHLCV fetcher and feature engineering
├── mispricing.py                # Calibration and edge analysis scripts
├── chainlink_oracle.py          # Chainlink BTC/USD round fetcher (Polygon, public RPC)
├── docker-compose.yml           # Container config for live ingest
├── config.example.json          # Credentials template (copy to config.json)
├── requirements.txt
└── .gitignore
```

---

## Data Infrastructure

### Live Ingest

`ingest/src/market_client.py` connects to Polymarket's WebSocket CLOB feed and writes orderbook snapshots to Parquet. The ingest has been running continuously since February 17, 2026.

Data is stored per-market as Parquet files in `data/book_snapshots/`, named by market slug (e.g. `btc-updown-5m-1771346400.parquet`). The slug timestamp is market **open** time in Unix seconds.

**Schema** (book snapshots):

| Field | Type | Description |
|---|---|---|
| `exchange_timestamp` | int64 | Exchange timestamp (milliseconds) |
| `slug` | string | Market slug |
| `token_id` | string | Yes or No token ID |
| `token_label` | string | "Yes" or "No" |
| `bid_price_1..10` | float | Best bid → 10th best bid |
| `ask_price_1..10` | float | Best ask → 10th best ask |
| `bid_size_1..10` | float | Size at each bid level |
| `ask_size_1..10` | float | Size at each ask level |
| `mid_price` | float | (best_bid + best_ask) / 2 |
| `spread` | float | best_ask - best_bid |
| `book_imbalance` | float | (bid_size_1 - ask_size_1) / (bid_size_1 + ask_size_1) |

### Historical Backfill

Historical data (February 12–17, 2026 — the gap before live ingest began) is being sourced from [Telonex](https://telonex.io), which provides daily Parquet files for Polymarket orderbook snapshots, trades, and on-chain fills going back to the market's launch date.

Telonex's `book_snapshot_25` schema is nearly identical to the live ingest schema with two differences:
- Levels are zero-indexed (`bid_price_0` = best bid) vs. one-indexed in live data
- Timestamps are in microseconds vs. milliseconds

The backfill pipeline normalises these differences on ingest.

### Chainlink Oracle Data

`chainlink_oracle.py` fetches historical Chainlink BTC/USD oracle rounds from Polygon via public RPC. Covers ~18,334 rounds across the market history window (~32 second average update interval). Used to measure oracle lag at market close — the time elapsed between the last oracle update and resolution.

The Polygon feed address is `0xc907E116054Ad103354f2D350FD2514433D57F6F` (BTC/USD, 0.1% deviation threshold, phase 3 as of February 2026).

---

## Analysis Pipeline

### `market_analysis.py`

Three-component pipeline:

**`MarketLoader`** — loads book snapshot Parquets with per-slug caching.

**`ResolutionStore`** — fetches and caches market outcomes from the Gamma API. Maps token_id → 1.0 (win) or 0.0 (loss).

**`CalibrationAnalyser`** — builds calibration observations and computes calibration curves. Streams observations to Parquet in batches to avoid OOM on the full 9.5M+ row dataset.

Core method: `build_observations()` joins book snapshots with resolution outcomes and computes `seconds_before_close`, `won`, and `mid_price` for each snapshot. Data cleaning filters: `spread > 0` and `seconds_before_close > 10` (removes negative-spread artifacts concentrated in the final 10 seconds and at 0.35–0.50 price range).

### `btc_klines.py`

Fetches 1-minute OHLCV data from Binance public REST API (no auth required) and caches to Parquet. Computes derived features: returns at 1/5/15/30 minute horizons, realised volatility at 3 timeframes, volume ratio, high-low range.

Two join modes:
- `at='market_open'` — computes BTC features once per market, broadcasts to all rows (fast)
- `at='snapshot'` — per-row feature lookup (slower, more granular)

---

## Key Findings

### Calibration Analysis (9.78M observations, 1,394 markets)

- **Tail contracts (< 0.35) are systematically overpriced** across all time windows and volatility regimes. Negative edge of -0.004 to -0.017 depending on bucket. Buying cheap contracts loses money.
- **High-probability contracts (> 0.65) show mild positive edge in aggregate**, but this disaggregates into noise when split into 30-second sub-windows.
- **The market is broadly efficient** — no exploitable aggregate edge found in price buckets, time windows, or BTC volatility quartiles.
- **Negative spread artifact**: 0.82% of snapshots had negative spreads (bid > ask), concentrated in the final 10 seconds and 0.35–0.50 price bucket. These are data quality artifacts, not real signals. Filtered in all analysis.

### Volatility Regime Analysis

Markets were segmented into quartiles by 5-minute realised BTC volatility at market open (vol_5m). All four regimes show the same qualitative pattern: tails overpriced, high-probability contracts underpriced. No regime-conditional edge found.

### Implication

The sophisticated traders observed in the orderbook are not exploiting aggregate mispricings visible to a calibration curve. Their edge is either:
1. Conditional on information not yet available (precise oracle resolution prices)
2. Complex conditional patterns requiring a richer model than aggregate statistics

---

## Planned: RL Trading Agent

### Architecture

- **Framework**: Stable-Baselines3 + sb3-contrib `MaskablePPO`
- **Environment**: Custom `gymnasium.Env` with invalid action masking
- **Action space**: Discrete — Hold, Buy Yes/No (small/large), Sell Yes/No (partial/full). Invalid actions masked based on current capital and positions.
- **State space**: BTC features (returns, volatility, volume), contract features (Yes/No prices, spread, book imbalance, current position, unrealized P&L), time-to-close
- **Reward**: Hybrid — small step penalty for non-hold actions (models spread cost), light unrealized P&L shaping, terminal realized P&L

### Training Strategy

Episodes are fully self-contained (no state carries across market boundaries), enabling episode-level shuffling during training without breaking temporal dependencies. This is a structural advantage over continuous-market RL approaches.

Train/validation split is temporal: train on earliest markets, validate on most recent. Episode order within the training pool is shuffled each epoch.

Validation uses early stopping on held-out episode performance, with checkpointing.

---

## Configuration

Copy `config.example.json` to `config.json` and fill in your Polymarket credentials:

```json
{
  "polymarket": {
    "host": "https://clob.polymarket.com",
    "private_key": "0x...",
    "proxy_wallet": "0x...",
    "api_key": "...",
    "api_secret": "...",
    "api_passphrase": "...",
    "chain_id": 137
  }
}
```

---

## Dependencies

```
pip install -r requirements.txt
```

Key dependencies:
- `pyarrow` — Parquet I/O
- `pandas` — data manipulation
- `web3` — Chainlink oracle RPC calls
- `protobuf`, `websocket-client` — live WebSocket ingest
- `py-clob-client` — Polymarket CLOB order execution
- `stable-baselines3`, `sb3-contrib` — RL training (planned)
- `gymnasium` — RL environment interface (planned)
- `telonex` — historical data backfill

---

## Running the Ingest

```bash
# Start live orderbook ingest (Docker)
docker-compose up -d

# Or directly
python ingest/src/market_client.py
```

---

## Notes

- `data/` is gitignored — Parquet files are not committed to the repo
- `config.json` is gitignored — never commit credentials
- The market launched February 12, 2026. Complete history is available via Telonex backfill + live ingest from February 17 onward.