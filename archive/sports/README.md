# Polymarket Sports Market Tools

Exploration scripts for Polymarket sports market data — groundwork for a sports betting arbitrage strategy.

## Scripts

- **`fetch_markets.py`** — Fetches all current sports markets from the Polymarket Sports API and saves to `data/sports/markets.json`. Run periodically (cron) to keep the market list fresh.

- **`monitor_prices.py`** — Connects to the Polymarket CLOB WebSocket and records orderbook snapshots (mid, spread, depth) for active sports markets. Saves to parquet files in `data/sports/snapshots/`, one file per sport per day. Runs continuously with auto-reconnect.

- **`analyze_movements.py`** — Reads collected snapshot data and produces a summary report: volatility rankings, spread analysis, liquidity depth, and large price move detection.

## Workflow

```bash
# 1. Fetch current markets
python sports/fetch_markets.py

# 2. Start monitoring prices (runs continuously)
python sports/monitor_prices.py --interval 10

# 3. After collecting data, analyze
python sports/analyze_movements.py
```

## Data Layout

```
data/sports/
  markets.json                    # Current market list
  snapshots/
    basketball_2026-03-07.parquet # Daily snapshot files by sport
    football_2026-03-07.parquet
```
