# ML Directional Classifier Research Report

**Date**: March 14, 2026
**Objective**: Design an ML system that predicts BTC price direction, magnitude, and confidence over 1-60 minute horizons for trading perpetual contracts on Binance/Bybit/Hyperliquid.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Context: What We've Tried](#2-context-what-weve-tried)
3. [Architecture Comparison](#3-architecture-comparison)
4. [Feature Engineering](#4-feature-engineering)
5. [Output Formulation & Loss Functions](#5-output-formulation--loss-functions)
6. [Training Paradigms](#6-training-paradigms)
7. [Uncertainty & Confidence](#7-uncertainty--confidence)
8. [Regime Detection & Adaptation](#8-regime-detection--adaptation)
9. [RL & Hybrid Approaches](#9-rl--hybrid-approaches)
10. [Novel & Unconventional Approaches](#10-novel--unconventional-approaches)
11. [Frameworks & Tools](#11-frameworks--tools)
12. [Literature & Industry Context](#12-literature--industry-context)
13. [Realistic Performance Expectations](#13-realistic-performance-expectations)
14. [Recommended Implementation Plan](#14-recommended-implementation-plan)

---

## 1. Executive Summary

### The Core Problem

We need a system that answers: **"Is BTC likely to go up or down, and by how much, in the next X minutes?"** with a calibrated confidence score. This is for trading perpetual contracts where payoff is linear (not binary), so both direction and magnitude matter.

### Top-Level Findings

1. **Feature engineering matters more than architecture.** The literature consistently shows that carefully constructed orderbook/microstructure features outperform raw data thrown at complex neural nets. The paper "Better Inputs Matter More Than Stacking Another Hidden Layer" captures this perfectly.

2. **Gradient-boosted trees (XGBoost/LightGBM/CatBoost) match or beat deep learning** for tabular financial features with 100x faster iteration. Deep learning wins only when processing raw sequential data (tick-level orderbook snapshots).

3. **The bottleneck is signal, not architecture.** Our experience confirms this: XGBoost v3.3 hit 85.8% WR in walk-forward but wasn't profitable live. The directional edge ceiling from orderbook alone is ~53%. Architecture improvements add 1-3pp; better features or data sources add 5-15pp.

4. **Uncertainty estimation is the highest-leverage improvement.** The backtest-to-live gap (17pp for mid-drift, 30pp for momentum) is the primary value destroyer. Deep ensembles and calibrated confidence filtering directly address this.

5. **Online adaptation is essential.** Market depth tripled in 3 weeks. Signals decay on timescales of days to weeks. Any fixed model has a short shelf life.

### Recommended Architecture

```
                    ┌─────────────────────┐
                    │  Regime Detector     │
                    │  (HMM / BOCPD)       │
                    └──────────┬──────────┘
                               │ regime features
    ┌──────────────────────────┼──────────────────────────┐
    │                          │                          │
┌───▼───┐              ┌──────▼──────┐             ┌─────▼─────┐
│ Model A│              │  Model B    │             │  Model C  │
│XGBoost │              │ LightGBM    │             │LSTM+Attn  │
│ensemble│              │ diff feats  │             │ raw book  │
└───┬────┘              └──────┬──────┘             └─────┬─────┘
    │                          │                          │
    └──────────────────────────┼──────────────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │   Meta-Labeler      │
                    │ (should I trade?)   │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Kelly Sizing +     │
                    │  Confidence Filter  │
                    └─────────────────────┘
```

**Phase 1**: Deep ensemble of 5-10 XGBoost models + Kelly sizing (1-2 weeks)
**Phase 2**: Add regime detection + bandit model selection (2-4 weeks)
**Phase 3**: LSTM with attention on raw orderbook + meta-labeling (1-2 months)
**Phase 4**: SAC for execution optimization (only after proven signal)

---

## 2. Context: What We've Tried

### Failed Approaches and Lessons

| Strategy | Backtest | Live | Gap | Lesson |
|----------|----------|------|-----|--------|
| Mid-price drift | 71.4% WR | 54.2% WR | -17pp | Data quality + adverse selection |
| BTC momentum (3s) | 76% WR | 46% WR | -30pp | Resampled data ≠ live data |
| XGBoost v3.3 | 85.8% WR | ~75-89% WR | Edge too thin | Spread/fees ate 3-4 cent edge |
| Whale signal | 51-54% acc | N/A | N/A | Too weak for binary outcomes |
| Market maker (mint) | Slightly + | Wiped by whipsaw | N/A | Adverse selection dominates |
| Sports longshot | +$0.036/trade | N/A | N/A | Pennies in front of steamroller |
| Evolution GA (prod_v5) | $154 OOS | N/A | N/A | All converged to "short BTC in downtrend" |

### Key Pattern
Signals that look good on resampled/deduplicated historical data do not survive live execution. The backtest-to-live gap is caused by: (1) fill assumptions, (2) latency, (3) adverse selection, (4) regime dependence, (5) data quality differences.

### What's Partially Working
- **Cross-asset conditional entropy**: 53% WR, +$499, Sharpe 0.67, temporally stable
- **Binance CEX price signal**: 66.4% WR corrected backtest, $0.044/token edge at 15s delay
- **XGBoost directional**: Predictions hold up, but edge too thin after costs

### Implications for New Work
- We're moving from binary Polymarket contracts to perpetual futures (linear payoff, no 0.85 cap)
- Perpetual futures have lower transaction costs (~6-14 bps round-trip vs Polymarket's spread + fees)
- We need to find signals that survive the ~5-15pp backtest-to-live gap
- Target: **55%+ live accuracy** to be meaningfully profitable after costs

---

## 3. Architecture Comparison

### Summary Table

| Architecture | Regime Handling | Inference Speed | Cross-Asset | Interpretability | Data Needs | Recommendation |
|---|---|---|---|---|---|---|
| **XGBoost/LightGBM** | Via features | Very Fast (<1ms) | Via features | High (SHAP) | Moderate | **Primary model** |
| **Mamba/SSM** | Good (selective) | Very Fast | Via extensions | Low | Moderate | Best DL single arch |
| **iTransformer** | Moderate | Medium | Native (core strength) | Medium | High | Best for cross-asset |
| **TFT** | Moderate (VSN) | Medium | Via features | High (VSN, attention) | High | Best for understanding |
| **TCN** | Poor (fixed) | Fast | Via features | Low | Moderate | Fast DL baseline |
| **LSTM + Attention** | Moderate | Medium | Via features | Medium | High | Good DL workhorse |
| **MoE** | Excellent | Medium | Depends on base | Medium | High | Best meta-architecture |
| **PatchTST** | Poor | Medium | Channel-indep | Low | High | Avoid for minute crypto |
| **Neural ODE/CDE** | Moderate | Slow | Via features | Medium | High | Only for irregular data |
| **GNN** | Moderate | Medium | Native | Medium | High | Overkill for 4 assets |

### Detailed Analysis

#### Tier 1: Start Here

**XGBoost / LightGBM / CatBoost** — The literature consistently shows gradient-boosted trees match or beat deep learning on tabular financial features. They are 100x faster to train, have built-in feature importance (SHAP), handle mixed feature types natively, and are robust to feature scale. CatBoost handles categorical features best; LightGBM is fastest; XGBoost is most battle-tested. Use as the primary model.

**Mamba (State Space Models)** — The most exciting single DL architecture for this use case:
- O(n) complexity allows very long lookback windows (24h of 1-min data = 1440 steps, trivial for Mamba)
- Selective state space mechanism is input-dependent, acting like learned regime-switching
- O(1) per-step inference in recurrent mode (fastest DL option)
- CryptoMamba and MambaStock show promising initial results
- Immature ecosystem is the main risk; requires custom CUDA kernels (`mamba-ssm` package)

**Deep Ensemble** — Not an architecture per se, but the single highest-value addition: train 5-10 models with different seeds, average predictions, use variance for uncertainty. Consistently outperforms single models and provides calibrated confidence for free.

#### Tier 2: Add for Specific Advantages

**iTransformer** — Flips the transformer axis: treats each variable as a token (not each timestep). Attention captures cross-asset correlations natively. Given our validated entropy signal across BTC/ETH/SOL/XRP, iTransformer could learn these relationships end-to-end. 5-8% RMSE improvement on multivariate benchmarks.

**MoE (Mixture of Experts)** — Best meta-architecture for the regime problem. 3-4 lightweight experts (e.g., momentum expert, mean-reversion expert, volatile-regime expert) with a gating network that routes inputs. Time-MoE (ICLR 2025 Spotlight) validates the approach at scale. Directly addresses our core failure mode (strategies that work in one regime fail in another). Risk: expert collapse without load-balancing loss.

**LSTM + Attention** — Adding attention over LSTM hidden states lets the model selectively focus on relevant past events (e.g., a support test 45 min ago). DA-RNN (Dual-Stage Attention) adds input-level attention for dynamic feature selection. More data-hungry than trees but captures sequential patterns trees cannot.

#### Tier 3: Research / Special Cases

**TFT** — Interpretable multi-horizon forecasting with variable selection networks. Good for understanding which features matter when. Heavy architecture; better suited for 15-60 min horizons than 1-5 min.

**TCN** — Fast, simple DL baseline using dilated causal convolutions. Good for establishing whether DL adds anything over trees. Worse at capturing large deviations than RNNs.

**Echo State Networks** — Underappreciated. Random reservoir + linear readout = seconds to train. Online adaptation via recursive least squares. Surprisingly competitive for short-term prediction. Best for rapid iteration during feature discovery.

#### Not Recommended

**PatchTST** — "Fails in many cases of extremely fluctuating series." Minute-level crypto is about as rapidly fluctuating as data gets. Avoid.

**Neural ODEs** — Theoretically elegant but inference requires ODE solver (slow, unpredictable runtime). Overkill for regularly-sampled 1-min bars. Only consider for tick-level irregular data.

**Full GNN** — With only 4 assets, a lightweight attention layer captures cross-asset dynamics without the overhead of a graph framework. Only worth it if expanding to 20+ assets.

### Staged Architecture Plan

**Stage 1 (Baseline)**: XGBoost/LightGBM ensemble on engineered features. Compare against simple moving-average baselines to confirm signal exists.

**Stage 2 (Cross-asset)**: iTransformer treating BTC/ETH/SOL/XRP as 4 variables. Tests whether end-to-end learning of cross-asset relationships beats hand-crafted entropy features.

**Stage 3 (Regime adaptation)**: Wrap Stage 1/2 in 3-4 expert MoE with volatility-aware gating. Each expert specializes in a regime.

**Stage 4 (If warranted)**: Replace backbone with Mamba for faster inference and longer lookback windows while keeping MoE structure.

---

## 4. Feature Engineering

### Priority Ranking

#### Tier 1 — High expected value, implement first

| # | Feature | Why | Source | Latency |
|---|---------|-----|--------|---------|
| 1 | **Orderbook imbalance** (multi-level, volume-weighted) | Most validated short-term predictor (Stoikov 2014+). Compute at levels 1, 3, 5 + slope across levels. | Exchange WebSocket | <10ms |
| 2 | **Trade flow imbalance** + derivative | Persistent buy/sell pressure is predictive. The *change* in flow is often more predictive than the level. | aggTrade stream | <10ms |
| 3 | **Returns at multiple horizons** (z-scored) | 1s, 5s, 30s, 1m, 5m, 15m, 1h, 4h, 24h. Short captures microstructure, long captures trend. Use log returns. | OHLCV | <1ms |
| 4 | **Volatility regime** | Garman-Klass on 1m bars, rolling 15/60/240. Short/long vol ratio > 1 = breakout regime. | OHLCV | <1ms |
| 5 | **Temporal features** | Hour-of-day, day-of-week (sine/cosine encoded). Session indicators. Minutes to funding snapshot. | Clock | 0ms |
| 6 | **Funding rate** | Level, predicted rate, change. Hyperliquid hourly is most granular. Time-to-funding creates structural edge. | REST API | 1-15s |

#### Tier 2 — Strong theoretical basis, implement second

| # | Feature | Why | Source |
|---|---------|-----|--------|
| 7 | **Hurst exponent** (rolling DFA) | Most theoretically grounded regime classifier. H>0.55 = momentum, H<0.45 = mean-reversion. Gate strategy on this. | Computed |
| 8 | **Cross-exchange basis** | Perp price Binance vs Bybit vs Hyperliquid. Persistent premium = flow imbalance. | Multi-exchange |
| 9 | **Volume momentum** (time-of-day adjusted) | `volume / sma(volume, 20)`, normalized by same-hour-of-day historical. Unusual volume precedes moves. | OHLCV |
| 10 | **VPIN** | Probability of informed trading. Rising VPIN precedes volatility. Use Binance aggressor-side flags directly. | aggTrade |
| 11 | **IV skew** (Deribit 25-delta) | Put demand = bearish positioning. Strongest medium-term directional signal from options. | Deribit WS |
| 12 | **Liquidation proximity** | Estimated liquidation levels from OI distribution. Price approaching clusters triggers cascades. | Binance forceOrder |

#### Tier 3 — Useful but lower priority

| # | Feature | Why |
|---|---------|-----|
| 13 | **Cross-asset returns** (ETH, SOL lead-lag) | BTC leads alts, but alts sometimes lead BTC by seconds. Rolling Granger causality or lagged correlation. |
| 14 | **Open interest changes** | Rising OI + rising price = new longs (continuation). Falling OI = exhaustion. |
| 15 | **Autocorrelation / entropy** | ACF at lags 1-60. Sample entropy for predictability. Transfer entropy for cross-asset information flow. |
| 16 | **Buy/sell volume ratio** | Cumulative delta divergence from price = higher-order signal. |
| 17 | **Traditional markets** (DXY, ES, VIX) | Correlated during US hours only. Use as regime indicator, not direct predictor. |

#### Tier 4 — Include only if infrastructure exists

| # | Feature | Why |
|---|---------|-----|
| 18 | **On-chain exchange flows** | 10-30 min lead for large deposits. Too slow for <5min horizons. |
| 19 | **Sentiment scores** | Low SNR, high latency. Use only as regime gate (Fear & Greed index). |
| 20 | **Wavelet decomposition** | Denoise price series. Marginal improvement over simpler smoothing. |

### Normalization (Critical)

Financial features are non-stationary. **Never use global standardization** — it leaks future information.

| Method | When to Use |
|--------|-------------|
| **Adaptive z-score** (EWM with halflife) | Default. `z = (x - ewm_mean) / ewm_std`. Halflife = 2-4 hours for intraday. |
| **Quantile transform** | Heavy-tailed features (trade volume, liquidation size). Robust to outliers. |
| **First-differencing** | Level features (price, OI, cumulative volume). Changes are more stationary. |
| **Ratio normalization** | `x / sma(x, 100)`. Good for features with secular trends. |
| **Time-of-day adjusted z-score** | Volume features. Crypto has strong intraday seasonality. |

### Feature Selection Pipeline

1. Start with Tier 1 features (~20 with multiple windows)
2. Train initial XGBoost, compute SHAP values
3. Prune features with consistently near-zero SHAP
4. Add Tier 2 features selectively, re-evaluate
5. Target: **20-30 well-engineered features** (not 200 that overfit)

### Key Pitfalls

- **Spoofing**: Large resting orders pulled before execution create false imbalance. Weight by time-at-price.
- **Feature staleness**: Orderbook imbalance is stale by 30s. Trade flow useful for 1-5 min. Implement staleness tracking.
- **Cross-venue fragmentation**: Binance book doesn't capture Bybit/OKX flow. Aggregating improves signal but adds complexity.
- **Inverted book imbalance**: We already know book imbalance is *inverted* on Polymarket (winners have LESS bid depth). Verify direction on perps.

---

## 5. Output Formulation & Loss Functions

### Recommended: Dual-Head Model

```
Shared Backbone (LSTM/Transformer/Tree features)
    │
    ├── Direction Head: P(up) via sigmoid
    │   Loss: BCE with label smoothing (0.05-0.10)
    │
    └── Magnitude Head: predicted |return| in bps
        Loss: Huber (delta = 1 sigma of returns)
```

**Why dual-head**: Direction accuracy is what makes/loses money. Magnitude informs position sizing. Optimizing both jointly with a shared backbone lets each task regularize the other.

**Confidence**: Derived from direction head probability after temperature scaling calibration. P(up) = 0.65 → confidence = 0.30 (= 2 * |0.65 - 0.5|).

### Alternative: Quantile Regression

Predict 5th, 25th, 50th, 75th, 95th percentile of returns. Direction from median sign. Confidence from interval width. Risk bounds from 5th percentile.

**Pros**: Distribution-free, built-in uncertainty, directly answers "what's my downside?"
**Cons**: Quantile crossing problem, N outputs instead of 2.

### Comparison of All Formulations

| Formulation | Direction | Magnitude | Confidence | Complexity | Best For |
|---|---|---|---|---|---|
| Binary classification | Yes | No | From P(up) | Low | Simple trend-following |
| Multi-class (5 bins) | Yes | Coarse | From probabilities | Medium | Discrete action sets |
| Regression | From sign | Yes | From |prediction|/vol | Low | Combined with vol model |
| **Quantile regression** | From median | From quantiles | From interval width | Medium | **Risk-aware systems** |
| **Dual-head** | Yes | Yes | From P(up) | Medium | **Perp trading** |
| Distributional (GMM) | From mean | Full distribution | From variance | High | Options-like payoffs |
| Ordinal regression | Yes | Ordered | From cumulative probs | Medium | If using discrete levels |

### Loss Function Configuration

For dual-head:
```
Total = 0.6 * BCE_smoothed(direction)
      + 0.3 * Huber(magnitude)
      + 0.1 * Asymmetric_penalty(magnitude, direction)
```

The asymmetric penalty specifically penalizes the magnitude head when its sign disagrees with true direction. Getting direction wrong is categorically worse than being off on magnitude.

For quantile regression:
```
Total = sum_q [ pinball_loss(q, prediction_q, actual) ]
      + lambda * sum_pairs [ max(0, q_low_hat - q_high_hat) ]  # crossing penalty
```

### Labeling: Defining "Up" vs "Down"

**Recommended: Triple Barrier Method** (de Prado)

Three barriers: upper (take-profit at k * daily_vol), lower (stop-loss at k * daily_vol), vertical (time expiry at T minutes). Label = whichever barrier is hit first. Most realistic for trading with a time limit and risk limit.

**Alternative: Volatility-Adjusted Threshold**

`Up if return > k * rolling_std(returns)`, `Down if < -k * rolling_std(returns)`, `Flat otherwise`. k = 0.25-0.5 sigma. Produces balanced classes across regimes.

**Avoid**: Fixed thresholds (everything becomes "flat" in calm periods, "up/down" in volatile periods).

### Meta-Labeling (de Prado)

Separate "which direction?" from "should I trade?":
1. Primary model predicts direction (high recall, take many signals)
2. Secondary model predicts if the primary's trade will be profitable (high precision, filter trades)
3. Output = direction * meta-label confidence = position size

This is a proper generalization of our "only trade when pred >= ask" filter. The meta-labeler uses features the primary doesn't see (regime, volatility, time-of-day, recent model accuracy).

---

## 6. Training Paradigms

### Walk-Forward Validation (Non-Negotiable)

Standard k-fold cross-validation is **invalid** for time series — it leaks future information.

**Sliding Window** (recommended for deployment):
- Train on [T-W, T], validate on [T, T+V], slide by V
- W = 90-180 days, V = 7-14 days for minute-level BTC
- Constant training size, naturally adapts to recent data

**Purged Cross-Validation** (for hyperparameter selection):
- Standard CV with purge gap >= prediction horizon between train/validation
- Also embargo: exclude training samples whose labels depend on validation prices
- Use combinatorial purged CV (CPCV) for rigorous hyperparameter tuning

**Expanding Window** (for initial research):
- Train on [0, T], validate on [T, T+V], expand T
- Ensures results aren't window-dependent
- Uses all history — model sees all past regimes

### Retraining Schedule

| Model Type | Retrain Frequency | Training Time | Notes |
|---|---|---|---|
| XGBoost/LightGBM | Every 1-7 days | 5-30 min (CPU) | Sliding window of 60-180 days |
| LSTM/Transformer | Every 3-7 days | 2-8 hours (GPU) | Fine-tune, not full retrain |
| Online update (last layer) | Every 1-4 hours | Seconds | Supplement full retraining |

**Staleness detection**: Track rolling Brier score / log-loss. If degradation > 10% from baseline, trigger retrain. Track feature importance stability — if top features change dramatically, regime has shifted.

### Multi-Task / Multi-Timeframe Learning

Predict multiple horizons simultaneously (1m, 5m, 15m, 60m) from the same backbone:
- Auxiliary horizons serve as regularizers
- Longer horizons force learning of robust features (not just 1m noise)
- If 60m is strongly bullish but 1m is uncertain, you know the trend supports the trade
- One of the most effective regularization techniques for financial models

### Ensemble Strategies

**Architecture Ensemble** (recommended for production):
- Combine XGBoost + LightGBM + LSTM (different error patterns)
- Diversity gain from architecture ensembles >> seed ensembles
- Meta-model (logistic regression) learns which to trust per regime

**Temporal Ensemble**:
- Keep snapshots at different training checkpoints, average predictions
- Free diversity (no extra training cost)

**Seed Ensemble**:
- 5-10 models with different random seeds
- Average for prediction, variance for uncertainty
- Trivial to implement, always recommended

### Anti-Overfitting Checklist

| Technique | Setting | Notes |
|---|---|---|
| Dropout | 0.1-0.3 (transformers), 0.2-0.5 (FC) | Variational dropout for LSTMs |
| Weight decay (AdamW) | 1e-4 to 1e-2 | Don't apply to biases or LayerNorm |
| Early stopping | Patience 5-20 epochs | Restore best weights, not last |
| Noise injection | sigma = 0.01-0.05 * std(feature) | Safe, consistently helpful |
| Window slicing | Random 50-58 of 60 bar windows | Simulates different entry points |
| Max tree depth | 3-5 for XGBoost | Deeper = overfit |
| Feature count | 20-30 max | More = overfit on financial data |

**Data augmentation caution**: Gaussian noise and window slicing are safe. Time/magnitude warping are moderate risk. Mixup creates unrealistic patterns for financial data. Synthetic GARCH data risks teaching wrong patterns. Always validate augmentation improves walk-forward metrics.

---

## 7. Uncertainty & Confidence

### Why This Matters Most

The backtest-to-live gap is our primary value destroyer. Uncertainty estimation directly addresses this by identifying when the model is operating outside its competence.

### Method Comparison

| Method | Quality | Cost | Complexity | Recommendation |
|---|---|---|---|---|
| **Deep Ensembles** | Best | N× training/inference | Low | **Gold standard** |
| **MC Dropout** | Good | N× inference only | Trivial | **Easy add to any model** |
| Temperature Scaling | Good calibration | Minimal | Trivial | **Minimum viable calibration** |
| Conformal Prediction | Distribution-free guarantees | Moderate | Moderate | Complementary safety net |
| Bayesian NNs | Theoretically best | Very high | Very high | Use MC Dropout instead |
| Gaussian Processes | Excellent for small data | O(n^3) | High | Only for sub-problems |

### Recommended Stack

1. **Deep Ensemble** (5-10 XGBoost or 5-10 LSTMs): Mean = prediction, variance = uncertainty. Consistently outperforms all other practical methods.

2. **Temperature Scaling**: Fit single parameter T on validation set. T > 1 softens overconfident predictions. Recalibrate every retraining cycle. Takes 1 line of code.

3. **Conformal Prediction**: Distribution-free coverage guarantees. Only trade when conformal prediction set is a singleton. Use interval width for position sizing. Valid regardless of model misspecification.

### Using Uncertainty for Trading

**Trade filtering**: Only trade when uncertainty < threshold (e.g., 75th percentile of historical uncertainty). Typically filters 30-60% of trades, significantly improving win rate.

**Position sizing**: `position = (predicted_edge / uncertainty) * scaling_factor` — essentially Sharpe-weighted per-trade sizing.

**Ensemble disagreement**: If ensemble members disagree on direction, don't trade. More robust than threshold on average probability.

**Adaptive thresholds**: Use rolling percentile of recent uncertainty, not fixed threshold (avoids filtering all trades during persistently uncertain periods).

---

## 8. Regime Detection & Adaptation

### Why Regimes Matter

Our evolution system discovered this empirically: all top strategies converged to "short BTC in Oct-Mar downtrend." The optimal trading strategy depends on the current regime. No single static model works across all conditions.

### Regime Detection Methods

#### Hidden Markov Models (HMMs)
- 3-state model on 5-minute returns + volatility typically discovers: (1) low-vol consolidation, (2) trending up, (3) trending down/crash
- Online inference via forward algorithm: O(1) per step, real-time regime probabilities
- Use `hmmlearn`, fit on rolling 30-90 day windows
- Limitation: Markov property too restrictive for long-duration regimes; Gaussian emissions too simple (use GMM-HMM)

#### Bayesian Online Change-Point Detection (BOCPD)
- Outputs P(change point at this step) at every timestep
- No retraining needed — fully online
- When change-point probability is high, reduce position size
- Implementation: ~50 lines of Python for basic version

#### Feature-Based Regime Indicators (Simplest, Often Best)
- Rolling volatility bucket (low/medium/high)
- Hurst exponent (trending vs mean-reverting)
- ADX/trend strength
- Volume regime (time-of-day adjusted)
- Use as input features to the model — let the model learn regime-conditional behavior

**Recommendation**: Use regime indicators as input features (not separate models). More robust, simpler to maintain, and doesn't split already-limited training data across regimes. Add BOCPD for online risk management (reduce size when change-point probability is high).

### Handling Non-Stationarity

| Approach | Complexity | Effectiveness | Notes |
|---|---|---|---|
| Sliding window retraining | Low | High | Every 1-7 days |
| Regime features as inputs | Low | Medium | Let model learn conditional behavior |
| Exponential decay sample weights | Low | Medium | Recent data weighted more heavily |
| BOCPD risk reduction | Low-Medium | Medium | Reduce size at change points |
| Temporal batch normalization | Low | Medium | Per-period statistics, not global |
| ADWIN drift detection | Medium | Medium | Auto-shrinks window on drift |
| Domain adaptation | High | Low-Medium | Marginal gains over simple retraining |

---

## 9. RL & Hybrid Approaches

### Hybrid Classifier + RL (Most Promising Architecture)

Decompose the problem into two sub-problems solved with different tools:

1. **Direction/confidence model** (supervised learning): Predict P(price up) using engineered features. This is where the alpha lives. XGBoost/ensemble is the right tool.

2. **Position sizing / execution model** (RL or analytical): Given prediction + confidence, decide how much to trade. This is a smaller, more structured problem.

**Why this decomposition works**:
- Supervised learning is far more sample-efficient for prediction
- The execution problem is smaller — RL doesn't need to learn market dynamics
- You can swap classifiers without retraining execution, and vice versa
- The classifier trains on historical data; the RL agent trains in simulation with classifier signals

### Position Sizing: Kelly vs RL

**Kelly Criterion**: `f* = mu / sigma^2` where mu = expected return, sigma^2 = variance.

For a classifier with confidence c: `position = f_Kelly(c) * max_position`

**Half-Kelly is standard** — provides ~75% of growth rate with dramatically less variance.

**When RL beats Kelly**: When optimal action depends on complex state (current position, recent trade history, execution constraints). For small positions on crypto perps, Kelly is usually sufficient. The edge is in prediction, not execution.

**Recommendation**: Start with half-Kelly. Only add RL for execution if you have a proven, persistent signal and evidence that execution optimization matters.

### Pure RL Assessment

| Algorithm | Suitability | Pros | Cons |
|---|---|---|---|
| PPO | Low-Medium | Well-understood, stable | On-policy = sample hungry. 1yr = ~525K bars, not enough |
| SAC | Medium | Off-policy (replay buffer), continuous actions | Sensitive to reward scale, complex tuning |
| DQN (Rainbow) | Medium | Stable, distributional variant gives uncertainty | Discrete actions only |
| Multi-Agent | Low | Theoretically appealing | Market too complex to model as adversary |

**Bottom line**: Pure RL for directional prediction is the wrong tool. RL excels at sequential decision-making (when should I enter/exit, how much to size) not at regression/classification (will price go up). Use supervised learning for prediction, RL only for execution.

### Bandit Algorithms for Model Selection

When you have multiple models, use a bandit algorithm to dynamically select the best one:
- **EXP3**: Adversarial bandit, no distribution assumptions. Best for financial markets.
- **Discounted UCB**: Exponential decay of old performance. Recent accuracy weighted more.
- Track rolling realized accuracy per model; bandit allocates opportunities to best performer.
- Trivial to implement. Natural regime adaptation.

---

## 10. Novel & Unconventional Approaches

### Worth a Research Spike

| Approach | Why Interesting | Risk | Data Needs |
|---|---|---|---|
| **Echo State Networks** | Seconds to train, online adaptation via RLS. Surprisingly competitive. | Lower ceiling than DL | Low |
| **Liquid Neural Networks** | Input-dependent dynamics adapt without retraining. 5-10x fewer neurons. | No financial track record, ODE training | Moderate |
| **KAN (Kolmogorov-Arnold)** | Learnable activation functions, can discover symbolic trading rules. Interpretable. | Very new (2024), unstable training | Moderate |

### Probably Not Worth It

| Approach | Why Not |
|---|---|
| **Foundation models** (TimeGPT, Chronos) | Trained on non-adversarial domains. Comparable to ARIMA on financial data. |
| **Neural CDEs** | Only advantage is irregular timestamps. Overkill for regular 1-min bars. |
| **Hyena operators** | Efficiency gain over attention only matters for 1000+ step sequences. |
| **Full Bayesian NNs** | MC Dropout or deep ensembles provide 80% of benefit at 20% complexity. |
| **Differentiable trading systems** | Elegant but fragile. Non-convex Sharpe optimization on single time series = overfitting. |

### Self-Supervised Pre-Training

**Potentially valuable for deep learning models**:
- Masked time series prediction (PatchTST-style): mask segments, reconstruct
- Contrastive learning (TS2Vec): similar market states close in embedding space
- Next-step prediction (GPT-style): predict next bar's features

**Value**: 2-5% accuracy improvement when labeled data is scarce relative to model capacity. For BTC with years of minute data, benefit is real but not dramatic. Most valuable for pre-training transformers.

**Cross-asset transfer**: Pre-train on ETH+SOL+XRP, fine-tune on BTC. Crypto assets share microstructure patterns. 2-5x more effective training data.

---

## 11. Frameworks & Tools

### Recommended Stack

| Tool | Use For | Why |
|---|---|---|
| **tsai** | Model experimentation | Native time series classification. InceptionTime, ROCKET, PatchTST built-in. Walk-forward CV. Best fit for directional classifiers. |
| **XGBoost/LightGBM** | Primary model | Fast, interpretable, proven. Already in our stack. |
| **Qlib** (Microsoft) | Factor research | Full ML pipeline: data → features → model → backtest. 17k GitHub stars, institutional-grade. |
| **PyTorch** | Custom models | Production deep learning. We already use it (CleanRL). |
| **hmmlearn** | Regime detection | Simple HMM fitting. |
| **ruptures** | Change-point detection | Offline analysis. Custom BOCPD for online. |

### Framework Comparison

| Framework | Classification Support | Ease of Use | Production Ready | Stars |
|---|---|---|---|---|
| **tsai** | Native | High | Moderate | 5k |
| **Darts** | Via wrapper | Highest | Moderate | 8k |
| **NeuralForecast** | Via wrapper | High | High | 3k |
| **PyTorch Forecasting** | Via wrapper | High | Moderate | 4k |
| **GluonTS** | Via wrapper | Moderate | High | 4k |
| **Qlib** | Yes | Moderate | High | 17k |
| **FinRL** | RL only | Moderate | Low | 10k |

---

## 12. Literature & Industry Context

### Key Papers

| Paper | Key Finding | Relevance |
|---|---|---|
| "Better Inputs Matter More Than Stacking Another Hidden Layer" (2025) | Feature engineering > model architecture for LOB prediction | Validates our focus on features |
| DeepLOB (Zhang & Zohren, 2019) | CNN+LSTM benchmark for LOB mid-price prediction | Architecture template for orderbook models |
| TLOB (2025) | Dual attention (spatial + temporal) outperforms DeepLOB | Modern LOB architecture |
| "Explainable Patterns in Cryptocurrency Microstructure" (2026) | CatBoost+SHAP on LOB features explains 10-37% of 500ms variance | Feature rankings stable across assets |
| "Cross-Cryptocurrency Return Predictability" (2024) | BTC leads alts; lagged BTC returns predict altcoin returns | Validates cross-asset signal direction |
| "Algorithmic Crypto Trading with Triple Barrier" (2025) | 82.68% accuracy, 151bps/trade with information-driven bars | Information bars > time bars |
| Time-MoE (ICLR 2025 Spotlight) | 2.4B parameter MoE for time series forecasting | Validates MoE at scale |

### Industry Reality

1. **Most profitable crypto firms are market-makers**, not directional traders. The edge is the spread, not prediction.
2. **Execution speed > prediction accuracy**. A 55% predictor at 10ms beats 65% at 1s.
3. **Simple strategies work in inefficient markets**; complexity increases as markets mature.
4. **Risk management contributes more to Sharpe than signal quality** in most live systems.
5. **28-to-3 survival rate**: One firm developed 28 strategies; only 3 survived live testing.
6. **44% replication failure rate**: Backtested Sharpe has R² < 0.025 with real-world performance.

### What the Literature Says Works

- **Information-driven bars** (volume/dollar bars instead of time bars): 82.68% accuracy in one study
- **Triple barrier labeling**: More realistic than simple up/down classification
- **Meta-labeling**: Separating direction from sizing significantly improves live performance
- **Cross-asset signals**: BTC leads alts; lagged correlation is exploitable
- **Ensemble methods**: Most robust against crypto noise in comparative studies
- **Regime-conditional models**: Essential for non-stationary markets

---

## 13. Realistic Performance Expectations

### Accuracy by Horizon

| Horizon | Backtest Accuracy | Realistic Live Accuracy | Notes |
|---------|------------------|------------------------|-------|
| 1 min | 55-65% | 51-54% | Latency-dependent; HFT firms dominate |
| 5 min | 55-62% | 52-56% | **Sweet spot**: enough signal, less latency pressure |
| 15 min | 53-60% | 52-55% | More regime-dependent |
| 60 min | 52-58% | 51-54% | Macro/regime factors dominate |

The consistent **5-15pp gap** between backtest and live is the central challenge. Our own experience (17pp for mid-drift, 30pp for momentum) is on the high end but not atypical.

### Profitability After Costs

For BTC perpetual futures on Binance (~6-14 bps round-trip):

| Live Accuracy | Edge/Trade | Annual PnL/$10K | Viable? |
|---|---|---|---|
| 51% | ~0 after costs | Negative | No |
| 53% | 2-4 bps | $500-2,000 | Marginal |
| **55%** | **6-10 bps** | **$3,000-8,000** | **Yes, with volume** |
| 58% | 12-18 bps | $8,000-20,000 | Good |
| 60%+ | 20+ bps | $20,000+ | Excellent (rare to sustain) |

**Target**: 55%+ live accuracy = meaningfully profitable. This requires 60-70% backtest accuracy to survive the gap.

### Achievable Sharpe Ratios

| Strategy Type | Realistic Sharpe |
|---|---|
| Passive BTC hold | 0.95-2.42 (regime-dependent) |
| Simple trend-following | 1.5-2.0 |
| **ML directional (after costs)** | **0.5-1.5** |
| Market-making | 2.0-5.0 (infrastructure moat) |
| Statistical arbitrage | 1.5-3.0 (decaying) |

**Aspirational target**: Sharpe 1.0-1.5 after costs. Anything claiming Sharpe > 3.0 for directional prediction should be viewed with extreme skepticism.

### Alpha Decay Rates

| Signal Type | Half-Life | Implication |
|---|---|---|
| Microstructure (orderbook, trade flow) | Days to weeks | Retrain weekly |
| Technical (momentum, mean reversion) | Weeks to months | Retrain monthly |
| Cross-asset (lead-lag, entropy) | Weeks to months | Slower decay, harder to discover |
| Fundamental/on-chain | Months | Weakest short-term power |

**Implication**: Need a **signal generation pipeline** that continuously discovers, tests, and deploys new features. Not just a model — a system.

---

## 14. Recommended Implementation Plan

### Phase 1: Foundation (Weeks 1-2)

**Goal**: Establish baseline with proper infrastructure on Binance perp data.

1. **Data pipeline**: Binance BTC-USDT perpetual klines + aggTrades + orderbook depth (WebSocket)
2. **Feature engineering**: Tier 1 features (orderbook imbalance, trade flow, returns, volatility, temporal, funding) — ~20 features with multiple windows
3. **Labeling**: Triple barrier method with volatility-scaled barriers
4. **Model**: Deep ensemble of 5-10 XGBoost models with different seeds
5. **Validation**: Walk-forward with sliding window (90-day train, 7-day test), purged gaps
6. **Uncertainty**: Ensemble variance for trade filtering, temperature-scaled confidence
7. **Position sizing**: Half-Kelly based on calibrated confidence
8. **Baseline comparison**: Verify ML beats simple moving-average crossover and buy-and-hold

**Success criteria**: Walk-forward accuracy > 55% with reasonable trade frequency (>5 trades/day). If not, the signal doesn't exist and we need better features before better models.

### Phase 2: Regime Awareness (Weeks 3-4)

**Goal**: Address the regime-dependence problem.

1. **HMM regime detector**: 3-state on 5m returns + Garman-Klass vol. Add regime probabilities as features.
2. **Hurst exponent**: Rolling DFA, use to gate momentum vs mean-reversion feature weights.
3. **BOCPD**: Online change-point detection. Reduce position when P(change) > 0.3.
4. **Bandit model selection**: If Phase 1 produced multiple viable configurations, use EXP3 to select dynamically.
5. **Sliding window retraining**: Automate retraining every 3-7 days on most recent 90 days.
6. **Feature importance tracking**: Monitor SHAP stability across retraining cycles. Flag when top features shift.

**Success criteria**: Walk-forward accuracy remains > 55% across different 30-day windows (regime robustness).

### Phase 3: Deep Learning + Meta-Labeling (Weeks 5-8)

**Goal**: Extract additional signal from raw sequential data.

1. **LSTM + attention**: Train on raw orderbook snapshots (not just engineered features). DA-RNN for dynamic feature selection.
2. **iTransformer**: 4-asset (BTC/ETH/SOL/XRP) model with attention learning cross-asset relationships.
3. **Meta-labeling**: Primary model (Phase 1 ensemble) predicts direction. Secondary model predicts if trade will be profitable. Secondary uses regime, volatility, time-of-day, recent model accuracy.
4. **Architecture ensemble**: Combine XGBoost ensemble + LSTM + iTransformer via stacking meta-model.
5. **MC Dropout**: Add to LSTM for additional uncertainty estimation.
6. **Multi-timeframe**: Auxiliary heads for 1m, 15m, 60m predictions as regularization.

**Success criteria**: Architecture ensemble outperforms Phase 1 XGBoost by >2pp accuracy on walk-forward. Meta-labeling improves Sharpe by >20%.

### Phase 4: Live Testing & Execution (Weeks 9-12)

**Goal**: Validate in live market with real execution.

1. **Paper trading**: Run on Binance testnet or with real-time data but simulated fills.
2. **Execution analysis**: Measure actual fill quality vs assumptions. Quantify slippage, latency, adverse selection.
3. **Live with small size**: $100-500 positions. Track accuracy and PnL for 2+ weeks.
4. **Compare live vs backtest**: If gap > 10pp, diagnose and fix before scaling.
5. **RL for execution** (optional): If signal is proven and execution is the bottleneck, add SAC for optimal limit order placement.

**Success criteria**: Live accuracy within 5pp of walk-forward backtest. Positive PnL after costs over 2+ week period.

### Phase 5: Scale & Iterate (Ongoing)

1. **Multi-asset**: Extend to ETH, SOL perpetuals.
2. **Signal pipeline**: Automated feature discovery, testing, and deployment.
3. **Monitoring**: Real-time accuracy tracking, drift detection, automatic retraining triggers.
4. **Novel architectures**: Budget 10-20% of research time for Mamba, Echo State Networks, or other speculative experiments.

---

## Appendix A: Data Requirements

### Minimum Data for Training

| Source | What | Resolution | Coverage Needed |
|---|---|---|---|
| Binance | BTC-USDT perp klines | 1m | 6-12 months |
| Binance | aggTrades (buy/sell flagged) | Tick | 3-6 months |
| Binance | Orderbook depth | 100ms-1s | 3-6 months |
| Binance | Funding rate | 8-hourly | 6-12 months |
| Binance | ETH, SOL, XRP klines | 1m | 6-12 months (for cross-asset) |
| Deribit | BTC options IV, skew | 1m | 3-6 months (Tier 2 feature) |
| Hyperliquid | Funding rate | 1-hourly | 3-6 months (Tier 2 feature) |

### What We Already Have
- BTC/ETH/SOL/XRP Binance klines: Jun 2025 → Mar 2026 (286 days)
- Polymarket orderbook snapshots: Feb 12 → Mar 8, 2026
- Live ingest (raw): Feb 17 → Mar 13, 2026

### What We Need
- Binance perpetual aggTrades (historical) — available via Tardis.dev or Binance data portal
- Binance perpetual orderbook snapshots (historical) — Tardis.dev
- Higher-resolution Binance klines (we may already have 1m from kline download pipeline)

## Appendix B: Computational Requirements

| Component | Hardware | Cost |
|---|---|---|
| XGBoost training (10 models) | CPU, 8 cores | $0 (local) |
| LSTM/Transformer training | 1x RTX 3060+ (12GB) | $0 (local) or $0.50/hr cloud |
| Live inference (all models) | CPU | Negligible |
| Data storage (1 year, all sources) | ~50-100 GB | Negligible |
| Binance WebSocket feeds | Always-on connection | $0 |

Total infrastructure cost: **Near-zero** for initial development. Cloud GPU only needed for Phase 3 deep learning experiments.

## Appendix C: Key References

### Must-Read Papers
1. Lopez de Prado, "Advances in Financial Machine Learning" (2018) — triple barrier, meta-labeling, purged CV
2. Zhang & Zohren, "DeepLOB" (2019) — LOB prediction benchmark
3. Lim et al., "Temporal Fusion Transformers" (2019) — interpretable multi-horizon forecasting
4. Gu & Dao, "Mamba: Linear-Time Sequence Modeling" (2023) — selective state spaces
5. Nie et al., "PatchTST" (ICLR 2023) — patched time series transformer
6. Liu et al., "iTransformer" (ICLR 2024) — inverted transformer for multivariate TS
7. Shi et al., "Time-MoE" (ICLR 2025) — billion-scale MoE for time series

### Crypto-Specific
8. "Better Inputs Matter More Than Stacking Another Hidden Layer" (2025)
9. "Explainable Patterns in Cryptocurrency Microstructure" (2026)
10. "Algorithmic Crypto Trading with Triple Barrier Labeling" (2025)

### Frameworks
11. tsai: https://github.com/timeseriesAI/tsai
12. Qlib: https://github.com/microsoft/qlib
13. NeuralForecast: https://github.com/Nixtla/neuralforecast
14. mamba-ssm: https://github.com/state-spaces/mamba

---

## 15. V2 Pipeline Results & Next Steps (March 16, 2026)

### V2 Results Summary

The V2 pipeline (triple barrier labels + regime/Hurst features, SG features removed) completed a full 32-window walk-forward evaluation:

| Metric | Value |
|---|---|
| Walk-forward accuracy | 53.1% |
| AUC | 0.545 |
| Confidence @ 0.65 threshold | 64.8% accuracy (108K samples) |
| Backtest win rate | **0%** — all trades lose after fees |

**Root cause**: 1σ triple barriers capture ~5 bps moves, but round-trip fees are 6.75-8.5 bps. The model is directionally correct (65% of trades have positive gross PnL) but the moves are 16× too small to overcome trading costs.

### Barrier Sigma Analysis

| σ | Approx Threshold (bps) | Avg Move (bps) | % > Fees | Accuracy |
|---|---|---|---|---|
| 1.0 | ~5 | 5.0 | ~20% | 53.1% |
| 2.0 | ~10 | 9.3 | 43.9% | 50.6% |
| Higher σ | sweep in progress | — | — | — |

Breakeven requires ~1.7σ minimum. Higher sigma = larger moves but longer hold times and potentially lower accuracy as microstructure signals decay.

### Fee Structure (Bybit VIP0 Perps)

| Component | Cost |
|---|---|
| Maker fee | 0.018% (1.8 bps) |
| Taker fee | 0.0495% (4.95 bps) |
| Round-trip (maker both sides) | 3.6 bps |
| Round-trip (taker both sides) | 9.9 bps |
| Round-trip (taker entry, maker exit) | 6.75 bps |
| Slippage | ~0 at small size (deep liquidity, 1-tick spread) |

Fees are on notional (not margin), so leverage doesn't help with fee-to-move ratio. Bybit scales fees down dramatically with volume — VIP3+ gets 0% maker, which is how HFT firms operate profitably.

### Key Insight: Microstructure Alpha vs Fee Tiers

The fundamental challenge: **orderbook microstructure features predict ~5 bps moves at 200ms-5s horizons, but retail fee tiers require 7-10 bps moves to break even.** This is a structural disadvantage — HFT firms at maker-0% tiers profitably trade these same signals.

Options:
1. **Widen barriers** (2-5σ) — captures larger moves but signals may not persist at longer horizons
2. **Alternative data sources** — find signals that predict larger moves (OI, liquidations, funding, options)
3. **Longer horizons** (1-min bars, 15-60 min prediction) — different market dynamics, different signals
4. **Fee tier improvement** — volume-based fee reduction (requires significant capital commitment)

---

## 16. Alternative Data Sources for BTC Direction

### Ranked by Expected Value

#### Tier 1 — High expected value, implement first

| Source | Signal | API | Latency | Why |
|---|---|---|---|---|
| **Delta Open Interest** | OI change + price direction = new positioning vs liquidation | Binance `GET /fapi/v1/openInterest` (3s) or WS `@openInterest` | 1-3s | Rising OI + rising price = new longs (continuation). OI drop = forced liquidation (reversal). Most validated derivatives signal after price itself. |
| **Liquidation flow** | Forced selling cascades, predictable price impact | Binance `forceOrder` WebSocket stream | Real-time | Liquidation clusters trigger cascading stops. Proximity to liquidation levels (from OI distribution) predicts where cascades start. Academic validation: Saggu & Ante (2024). |
| **Funding rate dynamics** | Rate level + change + time-to-snapshot | Binance `GET /fapi/v1/fundingRate`, Hyperliquid (hourly) | 1-15s | Pre-funding directional pressure is well-documented. Rate extremes predict reversals. Hourly Hyperliquid rate captures dynamics 8x faster than Binance. |

#### Tier 2 — Strong theoretical basis, implement second

| Source | Signal | API | Why |
|---|---|---|---|
| **Deribit options flow** | IV skew, put/call ratio, term structure slope | Deribit WS `ticker.{instrument}` | 25-delta risk reversal is the strongest medium-term directional signal from options. Term structure inversion predicts volatility events. |
| **NQ/ES futures** | US equity risk sentiment, leads crypto during US hours | CME via CQG/Rithmic or delayed via Yahoo | During US hours (13:30-20:00 UTC), NQ leads BTC by seconds to minutes. Acts as regime filter: risk-on/risk-off context. |
| **Stablecoin supply** | USDT/USDC mint/burn events | Etherscan API or Whale Alert | Saggu (2025): USDT mints predict BTC buying pressure. 10-60 min lead time. Lower frequency but strong signal when it fires. |

#### Tier 3 — Supplementary

| Source | Signal | Why |
|---|---|---|
| **Cross-exchange basis** | Binance vs Bybit vs Hyperliquid perp premium | Persistent premium = flow imbalance, exchange-specific demand |
| **ETH/SOL/XRP lead-lag** | Cross-asset momentum spillover | Our conditional entropy signal validates this; more granular features possible |
| **Long/short ratio** | Retail positioning | Binance `GET /futures/data/globalLongShortAccountRatio` — contrarian indicator |
| **Whale wallet flow** | Large exchange deposits/withdrawals | On-chain, 10-30 min lead — too slow for <5 min but useful for 15-60 min |

### Implementation Priority

**Phase 1** (immediate): Delta OI + liquidation flow + funding rate. All available from Binance with minimal infrastructure. Combined with existing orderbook features, this adds 10-15 new features covering the derivatives microstructure that LOB alone misses.

**Phase 2** (after Phase 1 validated): Deribit options + NQ/ES. Requires additional exchange connections. Options data is particularly valuable for predicting volatility regime shifts.

**Phase 3** (if signal exists): On-chain flow, stablecoin supply. Lower frequency, harder to backtest accurately, but potentially less crowded signals.

---

## 17. Timescale and Market Structure

### Is Market Microstructure Fractal?

**No.** Market microstructure is NOT fractal — different participants and dynamics operate at each timescale:

| Timescale | Dominant Dynamics | Participants | Signal Type |
|---|---|---|---|
| 200ms - 1s | Queue position, latency arbitrage, spoofing | HFT, co-located bots | Microstructure (imbalance, microprice) |
| 1s - 30s | Order flow momentum, informed trading | Algorithmic traders | Flow signals (aggressor imbalance, VPIN) |
| 1 min - 5 min | Aggregate supply/demand, liquidation cascades | Algo + manual traders | Derivatives signals (OI, funding, liquidations) |
| 15 min - 1 hr | Regime shifts, macro events, positioning | Institutional, macro | Macro + sentiment + options flow |

**Key implication**: The information content in 200ms snapshots does NOT tell us about 5 minutes in the future the same way 1-minute bars tell us about 1 hour. The participants and mechanics are fundamentally different at each scale.

This means:
1. Our current 200ms orderbook features are optimal for <5s prediction but **insufficient for >30s prediction**
2. Moving to 1-min bars with derivatives features (OI, funding, liquidations) is not just "looking further" — it's accessing a **different signal regime** with different dynamics
3. The optimal feature set changes dramatically by horizon

### Updated Strategy: Transition to 1-Minute Bars

Given the fee constraint at retail tiers, the most promising path forward:

1. **Keep XGBoost** — architecture is not the bottleneck (Wang 2025 confirms this)
2. **Change inputs** — add Tier 1 alternative data (OI, liquidations, funding) alongside orderbook features
3. **Move to 1-min bars** — capture dynamics at the 1-60 minute timescale where moves can exceed fees
4. **Triple barrier at 15-60 min max horizon** — let barriers find natural exit points at fee-viable move sizes

### Key References Added

- Saggu & Ante (2024): Liquidation cascades as predictive signals in perpetual futures
- Saggu (2025): USDT mint events predict BTC buying pressure (10-60 min lead)
- Wang (2025): "Better Inputs Matter More Than Stacking Another Hidden Layer" — confirmed: better features >> better architecture for crypto LOB prediction
