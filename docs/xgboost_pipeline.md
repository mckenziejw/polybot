# XGBoost Confidence Filter for Polymarket BTC Up/Down Markets

## 1. How Gradient Boosted Trees Work

### The Core Idea

XGBoost (eXtreme Gradient Boosting) builds an ensemble of decision trees sequentially, where each new tree corrects the errors of all previous trees combined. Unlike a random forest where trees are built independently, gradient boosting is *additive* -- each tree is fitted to the residual error of the current ensemble.

**Formal setup:**

Given training data `{(x_i, y_i)}`, the model predicts:

```
F(x) = F_0 + eta * h_1(x) + eta * h_2(x) + ... + eta * h_T(x)
```

where:
- `F_0` is a constant (e.g., log-odds of the base rate)
- `h_t(x)` are individual decision trees (weak learners)
- `eta` is the learning rate (shrinkage), typically 0.01-0.3
- `T` is the number of boosting rounds

Each tree `h_t` is fitted to the *negative gradient* of the loss function evaluated at the current ensemble's predictions. For binary classification with log-loss, this gradient is simply the residual: `y_i - p_i` where `p_i` is the current predicted probability. So each tree literally learns "where is the current model wrong, and by how much?"

### Why Trees Split the Way They Do

At each node, the tree searches every feature and every possible split point to find the partition that maximizes information gain (reduction in impurity). For a split that sends samples left vs. right:

```
Gain = 0.5 * [G_L^2 / (H_L + lambda) + G_R^2 / (H_R + lambda) - (G_L + G_R)^2 / (H_L + H_R + lambda)] - gamma
```

where `G` and `H` are sums of first and second derivatives of the loss, `lambda` is L2 regularization, and `gamma` is the minimum gain to make a split (complexity penalty). This is what makes XGBoost "extreme" -- it uses second-order Taylor expansion of the loss, not just first-order gradients.

### Built-in Regularization

XGBoost has several knobs to prevent overfitting:

- **max_depth**: limits tree depth (typically 3-8 for tabular data)
- **min_child_weight**: minimum sum of instance weights in a leaf
- **lambda (L2)** and **alpha (L1)**: regularize leaf weights
- **subsample**: fraction of training data per tree (like bagging)
- **colsample_bytree**: fraction of features per tree (like random forest)
- **early_stopping_rounds**: stop training when validation metric plateaus

### Why XGBoost is Ideal for This Problem

**vs. Neural Networks:**

| Aspect | XGBoost | Neural Networks |
|--------|---------|-----------------|
| Data size needed | Works well with 1k-100k samples | Typically needs 100k+ to shine |
| Feature engineering | Handles raw features, finds interactions automatically | Requires careful normalization, architecture design |
| Tabular data | State-of-the-art on most tabular benchmarks | Generally worse; attention/transformers closing gap |
| Training time | Minutes on CPU | Hours on GPU |
| Interpretability | Feature importance, SHAP values, tree visualization | Black box without extensive analysis |
| Overfitting risk | Well-controlled with built-in regularization | Requires dropout, weight decay, careful tuning |
| Missing data | Handles natively (learns optimal default direction) | Requires imputation |

**vs. RL (PPO/LSTM):**

RL is designed for *sequential decision-making* where actions affect future states. Our problem -- "given market state at time T, will this token win?" -- is a *static classification* problem. There is no sequential dependency: each 5-minute market is independent. RL adds massive complexity (reward shaping, exploration/exploitation, temporal credit assignment, training instability) for zero benefit here.

The PPO+LSTM approach tried earlier failed not because RL is bad, but because it was the wrong tool. A classifier that maps features to P(win) is exactly what we need, and XGBoost is the best classifier for structured/tabular data with moderate sample sizes.

**Key advantage for our use case:** XGBoost is robust to the exact kind of problems that killed our backtest-to-live performance. It can learn nonlinear interactions (e.g., "drift > 0.05 is only predictive when spread < 0.04 AND book_imbalance < -0.2") without us having to enumerate every possible interaction manually. The model finds the conditions under which a signal works and when it doesn't.


## 2. XGBoost as a General Confidence Filter (Meta-Model)

### The Meta-Model Concept

Instead of building one model per strategy, build a *meta-model* that answers a general question:

> "Given the current orderbook state and market microstructure, how likely is a directional bet to succeed?"

This model does not generate signals. It *scores* them. Any upstream signal (mid-drift, trade flow, momentum, BTC correlation) produces a candidate trade. The confidence filter then says: "yes, conditions are favorable for directional bets right now" or "no, the market is too noisy/symmetric/illiquid."

### Architecture

```
Upstream Signal Generator          Confidence Filter              Execution
(mid-drift, momentum, etc.)  -->   XGBoost P(correct)   -->   Trade if P > threshold
                                        |
                                   Market features at
                                   signal time
```

### Training Target

The label is simple: **did the directional bet win?** Specifically:

- For each market, at a chosen observation time T (e.g., 60s after open), look at the "Up" token
- If mid_price > 0.50, the signal says "Up will win" (or equivalently, buy the token above 0.50)
- If mid_price < 0.50, the signal says "Down will win"
- Label = 1 if the prediction was correct, 0 otherwise

This is symmetric and does not require a specific signal -- it covers ANY directional bet. A mid-drift strategy fires when drift > threshold; a momentum strategy fires on price acceleration. Both ultimately reduce to "I think token X will win." The confidence filter asks: "under these market conditions, how reliable is that kind of prediction?"

### Why a Threshold Sweep Matters

The model outputs `P(correct)`, not a binary decision. By sweeping thresholds:

- **P > 0.50**: trade everything (baseline)
- **P > 0.60**: filter out the ~40% of markets where the model is least confident
- **P > 0.70**: highly selective, maybe 20-30% of markets, but much higher win rate
- **P > 0.80**: very few trades, but if calibrated, these are near-certain

The threshold sweep reveals whether the model has learned *anything useful*. If win rate increases monotonically with threshold, the features contain real signal. If win rate is flat across thresholds, the features are noise.

### Closing the Backtest-to-Live Gap

The 17pp backtest-to-live gap for mid-drift likely comes from:

1. **Data quality differences** -- historical snapshots are clean, evenly-spaced; live data is noisy with gaps
2. **Adverse selection** -- in live trading, our orders get filled precisely when the market moves against us
3. **Regime dependence** -- the signal works in trending markets but fails in choppy ones

XGBoost can learn to detect these conditions. Features like spread width, book depth, staleness, and price oscillation frequency can discriminate "this is a clean trending market where drift is predictive" from "this is a choppy mess where drift is noise."


## 3. Feature Engineering

### 3.1 Orderbook Features (available in our data)

**Basic book structure:**
- `spread`: best ask - best bid (tighter = more liquid, but also more competitive)
- `book_imbalance`: (bid_size_1 - ask_size_1) / (bid_size_1 + ask_size_1)
- `mid_price`: (bid_price_1 + ask_price_1) / 2

**Multi-level depth:**
- `total_bid_depth_3`: sum of bid_size_1..3
- `total_ask_depth_3`: sum of ask_size_1..3
- `depth_ratio_3`: total_bid_depth_3 / total_ask_depth_3
- `bid_slope`: (bid_price_1 - bid_price_3) / 3 -- how fast bids decay in price
- `ask_slope`: (ask_price_3 - ask_price_1) / 3

**Imbalance at multiple levels:**
- `imbalance_L1`: (bid_size_1 - ask_size_1) / (bid_size_1 + ask_size_1)
- `imbalance_L3`: (sum bid 1-3 - sum ask 1-3) / (sum bid 1-3 + sum ask 1-3)
- `imbalance_L5`: full book imbalance

**Important finding from prior work:** Book imbalance is *inverted* -- winners have LESS bid depth. This is counterintuitive but makes sense: if the market thinks "Up" will win, participants buy aggressively (consuming bids) and post asks (building ask depth). The imbalance signal is real but inverted from equity markets.

### 3.2 Price Dynamics (computed from the time series within each market)

**Drift features:**
- `drift_from_50`: mid_price - 0.50 at observation time
- `abs_drift`: absolute drift magnitude
- `drift_speed`: drift per second over the last N seconds (first derivative)
- `drift_acceleration`: change in drift speed (second derivative)
- `max_drift_seen`: maximum |mid - 0.50| observed so far in this market

**Volatility:**
- `realized_vol`: std(mid_price changes) over lookback window
- `high_low_range`: max(mid) - min(mid) over lookback window
- `n_direction_changes`: number of times mid_price crossed 0.50 so far
- `price_oscillation`: high_low_range / abs_drift -- measures choppiness vs. trend

**Trend quality:**
- `r_squared`: R-squared of linear fit to mid_price over lookback -- high R2 = clean trend
- `drift_consistency`: fraction of 1-second intervals where price moved in the signal direction

### 3.3 Cross-Token Features

Since each market has both an "Up" and "Down" token, we can compare them:

- `up_down_mid_sum`: should be ~1.0, deviation indicates inefficiency
- `up_down_spread_diff`: spread_up - spread_down
- `up_down_depth_ratio`: total_depth_up / total_depth_down

### 3.4 Time Features

- `seconds_elapsed`: time since market open (e.g., 60 in our t=60 setup)
- `hour_of_day`: UTC hour (market behavior varies by session)
- `day_of_week`: weekend vs weekday effects
- `minutes_since_midnight_utc`: cyclical time feature

### 3.5 Staleness / Data Quality Features

- `staleness_ms`: milliseconds since last real tick (forward-filled bars)
- `ticks_in_window`: number of actual orderbook updates in the lookback window
- `update_frequency`: ticks_in_window / lookback_seconds

### 3.6 Adjacent Market Features (if available)

- `prev_market_winner`: did the immediately preceding 5-min market resolve Up or Down?
- `prev_market_drift_at_close`: final drift of the previous market
- `streak_length`: consecutive Up or Down wins

These capture short-term BTC momentum that transcends individual market boundaries.


## 4. Example Pipeline

See `xgboost_confidence.py` in the project root. The pipeline:

1. Reads Telonex parquet files from `data/telonex_book_snapshots/`
2. Loads resolutions from `data/resolution_cache.json`
3. For each market, extracts the "Up" token's state at t=60s
4. Computes ~30 features from the orderbook snapshot and price history
5. Labels each market: did the token above 0.50 win?
6. Trains XGBoost with time-based train/test split (no future leakage)
7. Reports classification metrics, feature importances, and threshold sweep results

### Running the Pipeline

```bash
# Install dependencies (if not already present)
pip install xgboost scikit-learn

# Run the full pipeline
python xgboost_confidence.py

# Customize observation time or drift threshold
python xgboost_confidence.py --signal-time 30 --min-markets 500
```

### Interpreting Results

**Feature importance** tells you which orderbook features actually predict whether directional bets succeed. If `spread` and `realized_vol` dominate, the model is learning "trade only in liquid, trending markets." If `book_imbalance` ranks high, the inverted-imbalance signal is real and usable.

**Threshold sweep** shows whether the model's confidence scores are calibrated. The key output is a table like:

```
Threshold  Trades  Win%  EV/trade
  0.50      4000   58%    +0.008
  0.55      3200   61%    +0.014
  0.60      2400   65%    +0.022
  0.65      1600   69%    +0.031
  0.70       800   74%    +0.042
```

If win rate increases monotonically with threshold, the model has learned real structure. The operating point depends on how many trades you want per day vs. edge per trade.

### Next Steps After This Pipeline

1. **Add more features** -- BTC price data, adjacent market outcomes, more complex microstructure
2. **Time-series cross-validation** -- use expanding window, not just a single split
3. **Calibration analysis** -- are predicted probabilities well-calibrated? (Platt scaling, isotonic regression)
4. **Feature selection** -- prune correlated or uninformative features
5. **Hyperparameter tuning** -- Optuna or similar for max_depth, learning_rate, etc.
6. **Live integration** -- wrap the model as a confidence filter in the execution component
