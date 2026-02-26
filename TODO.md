# TODO — Polymarket RL Bot

Issues ranked by severity. Fix blockers before moving to enhancements.

---

## 🔴 Blockers — Fix Before Training

### 1. Terminal Resolution Bug (`polymarket_env.py`)
**Problem:** Positions held to episode end are settled at `final mid_price`. Binary contracts resolve to exactly `1.0` (winner) or `0.0` (loser). Mid-price near expiry is unreliable — the book thins dramatically and the `0.5` default for missing data makes the error worst-case systematic.

**Impact:** The terminal reward (the only training signal) is wrong for every episode where a position is held to expiry. Training is on a corrupted objective.

**Fix:**
1. Confirm Telonex slugs resolve correctly against Gamma API (`data/resolutions.json`)
2. Load `ResolutionStore` in `PolymarketEnv.__init__()`
3. In `_load_market()`, look up resolution by slug, map to `yes_resolved` / `no_resolved` via `asset_id` from the parquet files
4. In `_close_episode()`, settle at resolution value (1.0 or 0.0) instead of mid-price, falling back to mid only if resolution is unavailable

---

### 2. Yes/No vs Up/Down Token Label Mismatch
**Problem:** Live ingest labels tokens `"Yes"` / `"No"` (from Polymarket Gamma API). Telonex historical data labels them `"Up"` / `"Down"`. The environment currently filters on `token_label == "Up"` and `token_label == "Down"`.

**Impact:** When the trained model runs against live data, the token filter returns empty DataFrames. The environment silently produces garbage observations or crashes.

**Fix:** Add a normalization step — either at ingest time (relabel live data to `Up`/`Down`) or in the environment at load time (map `Yes`→`Up`, `No`→`Down`). Pick one and be consistent.

---

### 3. Telonex Schema vs Live Schema Not Validated
**Problem:** Two separate pipelines write parquet files that are supposed to be structurally identical. They were written independently and have diverged in iteration. Key known differences: live schema includes `hash` and `full_bids`/`full_asks` columns; Telonex schema dropped these in later refactors.

**Impact:** If the environment or live inference code reads both file types, column mismatches will cause silent errors or crashes at deployment.

**Fix:** Run an explicit column-by-column comparison on a real file from each source. Document the canonical schema. Add an assertion or schema check at environment load time.

```python
# Quick check
import pandas as pd
live = pd.read_parquet("data/book_snapshots/some-slug.parquet")
hist = pd.read_parquet("data/telonex_book_snapshots/some-slug.parquet")
print(set(live.columns) - set(hist.columns))   # in live, not in hist
print(set(hist.columns) - set(live.columns))   # in hist, not in live
```

---

## 🟡 Important — Address Before Evaluating Training Results

### 4. Reward Shaping
**Problem:** Terminal-only reward over 3,000 steps is extremely sparse. The agent receives no feedback until the end of the episode. This makes early learning very slow and risks the policy collapsing to "always hold" (which is safe but useless).

**Options to explore:**
- Intermediate reward on realized PnL at each trade (not just terminal)
- Shaped reward penalizing holding at expiry on the losing side
- Unrealized PnL as a step reward component (with care — this can encourage overtrading)
- Entropy bonus to maintain exploration

**Status:** TBD. Do not tune until the terminal reward is correct (Blocker #1).

---

### 5. Bankroll Exhaustion Not Masked
**Problem:** Action masking checks max position and flat position, but does not check whether `bankroll < MIN_ORDER`. If bankroll drops below $5, buy actions remain unmasked but will silently do nothing (order capped to 0 by `size = min(size, max_pos - capital_at_risk)`).

**Fix:** Add bankroll check to `action_masks()`:
```python
if self._bankroll < MIN_ORDER:
    masks[ACTION_BUY_YES_SMALL] = False
    masks[ACTION_BUY_YES_LARGE] = False
    masks[ACTION_BUY_NO_SMALL]  = False
    masks[ACTION_BUY_NO_LARGE]  = False
```

---

### 6. Staleness Not Represented in Observations
**Problem:** The 100ms resampling forward-fills stale book data silently. The agent cannot distinguish a fresh book update from a forward-filled stale one. During quiet periods this could be many seconds of stale data.

**Fix:** Add a `book_staleness_ms` feature to the observation (milliseconds since last real update for each token). This gives the agent information about data reliability.

---

### 7. Small Training Window (13 Days)
**Problem:** All training data spans Feb 12–25, 2026. This is a narrow BTC market regime sample. If those 13 days were trending, the agent may learn momentum strategies that fail in ranging markets.

**Action:** Document what BTC did over this window (trend vs range, volatility regime). Plan to expand training data as more Telonex data becomes available.

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

### 11. Evaluation Framework
**Status:** Not started.

Need out-of-sample evaluation before any live deployment:
- Hold out a date range from training (e.g. last 2 days of Telonex data)
- Evaluate policy on held-out markets deterministically
- Metrics: mean episode PnL, Sharpe ratio across episodes, win rate, average position duration, trade frequency