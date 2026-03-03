# TODO — Polymarket RL Bot

Issues ranked by severity. Fix blockers before moving to enhancements.

---

## 🔴 Blockers — Fix Before Training

---

### 2. Yes/No vs Up/Down Token Label Mismatch
**Problem:** Live ingest labels tokens `"Yes"` / `"No"` (from Polymarket Gamma API). Telonex historical data labels them `"Up"` / `"Down"`. The environment currently filters on `token_label == "Up"` and `token_label == "Down"`.

**Impact:** When the trained model runs against live data, the token filter returns empty DataFrames. The environment silently produces garbage observations or crashes.

**Fix:** Add a normalization step — either at ingest time (relabel live data to `Up`/`Down`) or in the environment at load time (map `Yes`→`Up`, `No`→`Down`). Pick one and be consistent.

---

---

## 🟢 Pending Features

### 8. Chainlink Oracle Data
**Status:** Not started.

The Chainlink oracle price (not the CEX mid) is what determines Polymarket resolution. The CEX-oracle divergence in the final seconds of a market is the primary hypothesized edge for smart money. Without oracle price as a feature, the agent has no direct signal for this.

**Plan:** Fetch Chainlink BTC/USD price feed history, align to market timestamps, add as observation features: `oracle_price`, `cex_oracle_divergence`, `seconds_since_last_oracle_update`.

---

### 9. Slippage / Fill Model Improvement
**Status:** Currently mid-price, immediate fill.

Real fills will cross the spread. In thin Polymarket books the spread can be 5-10 cents wide. The current fill model is systematically optimistic. Consider: fill at ask for buys, bid for sells, or use a probabilistic model based on order size vs available depth.

---

### 10. Live Inference Integration
**Status:** Not started.

The trained model must eventually execute against live data from the ingest pipeline. Requirements:
- Observation construction from live `BookSnapshot` events (matching training schema)
- Position and bankroll state management across a live 5-minute window
- Order execution via Polymarket CLOB API
- Token label normalization (Yes/No → Up/Down, or vice versa — see Blocker #2)

---

### Integrate with cloud GPU (VastAI)
- Training script should offload compute to Vast
- Need cloud storage for market data (Google Drive or S3)
- Startup script for VastAI instace
    - Clone git
    - Pull market data
    - Start analysis
    - Write logs to S3?

### Update live ingest with buffer to resample data to 100ms timeframes required by model

### 