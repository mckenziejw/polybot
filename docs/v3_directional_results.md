# V3 Directional Classifier Results

Date: 2026-03-17
Data: Bybit BTCUSDT perpetual, 1-minute bars, Jun 2025 → Mar 2026 (~288 days)
Walk-forward: 90-day train / 7-day test, rolling weekly

## Summary

**Neither feature variant nor any barrier configuration produces meaningful predictive power.**
All AUCs fall in 0.50-0.53 — barely distinguishable from random.

---

## Phase 1: Initial Symmetric Run (tp=1.0σ, sl=1.0σ)

Two feature variants trained in parallel:

### Orderbook (OB) Variant
- Features: LOB imbalance, microprice delta, depth ratio, weighted mid delta, rolling windows
- 32 walk-forward windows, 299,201 total test samples

| Metric | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| Accuracy | 0.5064 | 0.024 | 0.4526 | 0.5554 |
| AUC | 0.5078 | 0.028 | 0.4489 | 0.5590 |
| Hold period | 20.7 bars (1242s / ~21 min) | | | |

Windows below 50% accuracy: 10/32
Windows below 50% AUC: 9/32

### OHLCV+Derivatives Variant
- Features: OHLCV bars (RSI, MACD, Bollinger, ATR, vol), funding rate, OI, liquidations, time encoding
- 28 walk-forward windows, 260,454 total test samples

| Metric | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| Accuracy | 0.5109 | 0.026 | 0.4481 | 0.5877 |
| AUC | 0.5105 | 0.029 | 0.4343 | 0.5590 |
| Hold period | 20.7 bars (1239s / ~21 min) | | | |

Windows below 50% accuracy: 11/28
Windows below 50% AUC: 9/28

### Per-Window Results (OB)

| Window | Accuracy | AUC | N | Test Period |
|--------|----------|-----|---|-------------|
| W0 | 0.5108 | 0.5051 | 9,526 | 2025-07-30 → 2025-08-06 |
| W1 | 0.5127 | 0.5211 | 9,448 | 2025-08-06 → 2025-08-13 |
| W2 | 0.5182 | 0.5190 | 9,411 | 2025-08-13 → 2025-08-20 |
| W3 | 0.5254 | 0.5169 | 9,289 | 2025-08-20 → 2025-08-27 |
| W4 | 0.5281 | 0.5402 | 9,328 | 2025-08-27 → 2025-09-03 |
| W5 | 0.5250 | 0.5308 | 9,325 | 2025-09-03 → 2025-09-10 |
| W6 | 0.5050 | 0.5590 | 9,323 | 2025-09-10 → 2025-09-17 |
| W7 | 0.5323 | 0.5507 | 9,450 | 2025-09-17 → 2025-09-24 |
| W8 | 0.4918 | 0.5032 | 9,390 | 2025-09-24 → 2025-10-01 |
| W9 | 0.5084 | 0.5492 | 9,359 | 2025-10-01 → 2025-10-08 |
| W10 | 0.5117 | 0.4594 | 9,419 | 2025-10-08 → 2025-10-15 |
| W11 | 0.5109 | 0.5328 | 9,293 | 2025-10-15 → 2025-10-22 |
| W12 | 0.5016 | 0.5132 | 9,347 | 2025-10-22 → 2025-10-29 |
| W13 | 0.4526 | 0.4489 | 9,383 | 2025-10-29 → 2025-11-05 |
| W14 | 0.5082 | 0.5184 | 9,497 | 2025-11-05 → 2025-11-12 |
| W15 | 0.5432 | 0.5168 | 9,490 | 2025-11-12 → 2025-11-19 |
| W16 | 0.5134 | 0.5046 | 9,497 | 2025-11-19 → 2025-11-26 |
| W17 | 0.4681 | 0.4667 | 9,240 | 2025-11-26 → 2025-12-03 |
| W18 | 0.4892 | 0.4887 | 9,369 | 2025-12-03 → 2025-12-10 |
| W19 | 0.5302 | 0.4958 | 9,368 | 2025-12-10 → 2025-12-17 |
| W20 | 0.4731 | 0.5010 | 9,230 | 2025-12-17 → 2025-12-24 |
| W21 | 0.4696 | 0.4723 | 9,254 | 2025-12-24 → 2025-12-31 |
| W22 | 0.4818 | 0.5374 | 9,473 | 2025-12-31 → 2026-01-07 |
| W23 | 0.5189 | 0.5406 | 9,344 | 2026-01-07 → 2026-01-14 |
| W24 | 0.4930 | 0.5010 | 9,370 | 2026-01-14 → 2026-01-21 |
| W25 | 0.4565 | 0.4630 | 9,351 | 2026-01-21 → 2026-01-28 |
| W26 | 0.5554 | 0.5005 | 9,274 | 2026-01-28 → 2026-02-04 |
| W27 | 0.5072 | 0.5048 | 9,318 | 2026-02-04 → 2026-02-11 |
| W28 | 0.4979 | 0.4919 | 9,217 | 2026-02-11 → 2026-02-18 |
| W29 | 0.5190 | 0.4641 | 9,073 | 2026-02-18 → 2026-02-25 |
| W30 | 0.5096 | 0.5201 | 9,237 | 2026-02-25 → 2026-03-04 |
| W31 | 0.5358 | 0.5126 | 9,308 | 2026-03-04 → 2026-03-11 |

### Per-Window Results (OHLCV)

| Window | Accuracy | AUC | N | Test Period |
|--------|----------|-----|---|-------------|
| W0 | 0.5116 | 0.5522 | 9,334 | 2025-08-30 → 2025-09-06 |
| W1 | 0.4915 | 0.5292 | 9,420 | 2025-09-06 → 2025-09-13 |
| W2 | 0.5311 | 0.5391 | 9,441 | 2025-09-13 → 2025-09-20 |
| W3 | 0.5357 | 0.5545 | 9,375 | 2025-09-20 → 2025-09-27 |
| W4 | 0.4481 | 0.4343 | 9,388 | 2025-09-27 → 2025-10-04 |
| W5 | 0.5301 | 0.5062 | 9,326 | 2025-10-04 → 2025-10-11 |
| W6 | 0.5223 | 0.5113 | 9,479 | 2025-10-11 → 2025-10-18 |
| W7 | 0.4894 | 0.5224 | 9,343 | 2025-10-18 → 2025-10-25 |
| W8 | 0.4984 | 0.4819 | 9,343 | 2025-10-25 → 2025-11-01 |
| W9 | 0.5506 | 0.5052 | 9,494 | 2025-11-01 → 2025-11-08 |
| W10 | 0.5187 | 0.4884 | 9,478 | 2025-11-08 → 2025-11-15 |
| W11 | 0.5285 | 0.4997 | 9,510 | 2025-11-15 → 2025-11-22 |
| W12 | 0.4713 | 0.4943 | 9,357 | 2025-11-22 → 2025-11-29 |
| W13 | 0.5070 | 0.4575 | 9,389 | 2025-11-29 → 2025-12-06 |
| W14 | 0.5056 | 0.5314 | 9,243 | 2025-12-06 → 2025-12-13 |
| W15 | 0.5131 | 0.5590 | 9,324 | 2025-12-13 → 2025-12-20 |
| W16 | 0.4995 | 0.5286 | 9,171 | 2025-12-20 → 2025-12-27 |
| W17 | 0.4970 | 0.4886 | 9,373 | 2025-12-27 → 2026-01-03 |
| W18 | 0.5161 | 0.5580 | 9,451 | 2026-01-03 → 2026-01-10 |
| W19 | 0.4960 | 0.5165 | 9,407 | 2026-01-10 → 2026-01-17 |
| W20 | 0.4988 | 0.4941 | 9,358 | 2026-01-17 → 2026-01-24 |
| W21 | 0.4799 | 0.4721 | 9,287 | 2026-01-24 → 2026-01-31 |
| W22 | 0.5877 | 0.5057 | 8,081 | 2026-01-31 → 2026-02-07 |
| W23 | 0.4927 | 0.5031 | 9,399 | 2026-02-07 → 2026-02-14 |
| W24 | 0.5027 | 0.5077 | 8,923 | 2026-02-14 → 2026-02-21 |
| W25 | 0.5444 | 0.5153 | 9,254 | 2026-02-21 → 2026-02-28 |
| W26 | 0.5227 | 0.5249 | 9,282 | 2026-02-28 → 2026-03-07 |
| W27 | 0.5153 | 0.5118 | 9,224 | 2026-03-07 → 2026-03-14 |

---

## Phase 2: Barrier Sweep (OHLCV variant with higher-order derivatives)

Added higher-order derivative features before sweep:
- `funding_rate_acceleration`, `funding_rate_momentum`, `funding_rate_vol_1h`
- `basis_acceleration`, `basis_momentum`
- `oi_momentum`
- Cross-features: `oi_price_divergence`, `oi_price_divergence_mag`, `liq_funding_interaction`

9 barrier configurations tested:

| Config | TP σ | SL σ | Type | Accuracy | AUC | N |
|--------|------|------|------|----------|-----|---|
| symmetric_1.0 | 1.0 | 1.0 | Balanced | 0.5096 | 0.5068 | 259,147 |
| symmetric_1.5 | 1.5 | 1.5 | Balanced | 0.5114 | 0.5100 | 242,037 |
| symmetric_2.0 | 2.0 | 2.0 | Balanced | 0.5163 | 0.5158 | 230,982 |
| trend_2.0_0.5 | 2.0 | 0.5 | Trend-following | 0.7362 | 0.5048 | 264,312 |
| trend_2.0_1.0 | 2.0 | 1.0 | Trend-following | 0.6177 | 0.5073 | 247,434 |
| trend_3.0_1.0 | 3.0 | 1.0 | Trend-following | 0.6474 | 0.5140 | 244,497 |
| revert_0.5_2.0 | 0.5 | 2.0 | Mean-reversion | 0.7229 | 0.5138 | 265,171 |
| revert_1.0_2.0 | 1.0 | 2.0 | Mean-reversion | 0.5955 | 0.5173 | 247,563 |
| revert_1.0_3.0 | 1.0 | 3.0 | Mean-reversion | 0.6274 | **0.5255** | 243,864 |

### Best config per-window: revert_1.0_3.0 (TP=1.0σ, SL=3.0σ)

Mean accuracy: 0.6272, Mean AUC: 0.5275

| Window | Accuracy | AUC | N | Test Period |
|--------|----------|-----|---|-------------|
| W0 | 0.6454 | 0.5504 | 8,807 | 2025-08-30 → 2025-09-06 |
| W1 | 0.6739 | 0.5598 | 8,874 | 2025-09-06 → 2025-09-13 |
| W2 | 0.6250 | 0.5678 | 8,839 | 2025-09-13 → 2025-09-20 |
| W3 | 0.6080 | 0.6043 | 8,603 | 2025-09-20 → 2025-09-27 |
| W4 | 0.6622 | 0.4784 | 8,829 | 2025-09-27 → 2025-10-04 |
| W5 | 0.5977 | 0.5156 | 8,841 | 2025-10-04 → 2025-10-11 |
| W6 | 0.5663 | 0.5208 | 8,984 | 2025-10-11 → 2025-10-18 |
| W7 | 0.6384 | 0.5376 | 8,796 | 2025-10-18 → 2025-10-25 |
| W8 | 0.6161 | 0.4926 | 8,845 | 2025-10-25 → 2025-11-01 |
| W9 | 0.5997 | 0.5380 | 8,786 | 2025-11-01 → 2025-11-08 |
| W10 | 0.6136 | 0.5222 | 8,821 | 2025-11-08 → 2025-11-15 |
| W11 | 0.5844 | 0.5151 | 8,884 | 2025-11-15 → 2025-11-22 |
| W12 | 0.6673 | 0.5422 | 8,790 | 2025-11-22 → 2025-11-29 |
| W13 | 0.6288 | 0.5217 | 8,755 | 2025-11-29 → 2025-12-06 |
| W14 | 0.6404 | 0.5350 | 8,666 | 2025-12-06 → 2025-12-13 |
| W15 | 0.6429 | 0.5873 | 8,695 | 2025-12-13 → 2025-12-20 |
| W16 | 0.6766 | 0.5445 | 8,540 | 2025-12-20 → 2025-12-27 |
| W17 | 0.6662 | 0.5133 | 8,712 | 2025-12-27 → 2026-01-03 |
| W18 | 0.6624 | 0.5599 | 8,810 | 2026-01-03 → 2026-01-10 |
| W19 | 0.6540 | 0.5336 | 8,791 | 2026-01-10 → 2026-01-17 |
| W20 | 0.6302 | 0.5219 | 8,777 | 2026-01-17 → 2026-01-24 |
| W21 | 0.5974 | 0.4828 | 8,657 | 2026-01-24 → 2026-01-31 |
| W22 | 0.5652 | 0.4930 | 7,542 | 2026-01-31 → 2026-02-07 |
| W23 | 0.6419 | 0.4775 | 8,844 | 2026-02-07 → 2026-02-14 |
| W24 | 0.6352 | 0.4742 | 8,379 | 2026-02-14 → 2026-02-21 |
| W25 | 0.6017 | 0.5307 | 8,619 | 2026-02-21 → 2026-02-28 |
| W26 | 0.5917 | 0.4901 | 8,715 | 2026-02-28 → 2026-03-07 |
| W27 | 0.6282 | 0.5592 | 8,663 | 2026-03-07 → 2026-03-14 |

Note: 62.7% accuracy is misleading — the asymmetric barriers (TP=1σ, SL=3σ) create ~63% base rate
for the "up" class (TP triggers more easily than SL), so the model is mostly predicting the majority class.
AUC of 0.5275 confirms marginal discriminative power.

---

## Feature Importance (symmetric_1.0, OHLCV variant)

Consistently top features across 28 windows (by XGBoost gain):

**Tier 1 — Top 5 in most windows:**
- `volatility_1h` — single most important feature overall
- `hour_cos` / `hour_sin` — time-of-day encoding
- `funding_rate` / `funding_rate_ma_1h` / `funding_rate_ma_15m` — funding rate cluster
- `basis_std_1h` — mark-index basis volatility

**Tier 2 — Reliably top 15:**
- `dow_sin` / `dow_cos` — day-of-week encoding
- `price_vs_ma_1h` / `price_vs_ma_4h` — mean-reversion signals
- `minutes_to_funding` — funding countdown
- `oi_relative` / `oi_volatility_1h` — open interest
- `liq_intensity_1h` — liquidation flow
- `gk_volatility_1h` / `gk_volatility_15m` — Garman-Klass vol

**Tier 3 — Appear in later windows:**
- `return_autocorr` — serial correlation
- `funding_rate_vol_1h` — higher-order feature (from ~window 12)
- `liq_funding_interaction` — cross-feature (windows 26-27)
- `basis_momentum` — one appearance at #2

**Red flag**: Feature gain is extremely flat (top=0.04-0.07, #15=0.028-0.034, ratio ~2x).
A model with real signal would show top features dominating at 5-10x.
Temporal features (`hour_cos`, `dow_sin`) consistently rank #1-3, indicating the model is
memorizing "which hours/days were profitable" in training — classic regime overfitting.

---

## Label Statistics (symmetric 1.0σ)

- Up/Down split: ~49.3% / 50.7% (near-balanced by construction)
- Barrier types: TP 43%, SL 44%, Vertical 6%, Flat/NaN 7%
- Mean holding period: ~21 bars (21 minutes)
- Median holding period: ~16 bars (16 minutes)

---

## Infrastructure

- EC2: c8g.12xlarge (48 vCPU Graviton3, 96 GB RAM), ~$0.51/hr
- Preprocessing: 14 seconds for 288 days of OHLCV features
- Training: ~60 seconds per 28-window walk-forward (sequential seeds, `parallel=False` to avoid numba fork deadlock)
- Full barrier sweep (9 configs): ~10 minutes total
- XGBoost: max_depth=3, n_estimators=300, learning_rate=0.05, 5-seed ensemble

---

## Conclusions

1. **Feature set exhaustion within single-asset perps data**: OB, OHLCV, and derivatives (funding, OI, liquidations) all produce AUC 0.50-0.53. Adding higher-order features (acceleration, momentum, cross-feature interactions) didn't help.

2. **Barrier geometry doesn't matter when there's no signal**: All 9 configs cluster in the same AUC range. The classification problem (predicting which barrier gets hit first) is fundamentally hard at 1-minute resolution with these features.

3. **Temporal overfitting**: The model's top features are time-of-day and day-of-week encoding, meaning it learns "BTC goes up at 3pm on Tuesdays" patterns from the training window that don't persist.

4. **Consistent with Phase 1**: The 200ms LOB classifier also topped out at ~53% with orderbook features. Moving to 1m bars + derivatives didn't unlock new signal.

## Not Yet Tried
- Cross-asset features (ETH/SOL/XRP lead-lag, relative strength)
- Trade flow features (VPIN, buy/sell ratio from Bybit tick trades — data available)
- Inter-exchange basis (Bybit vs Binance spread, funding differentials)
- XGBoost hyperparameter optimization (max_depth=3 is conservative)
- Different model architectures (LSTM, transformer)
- Regime-conditional models
- Meta-labeling (separate direction from trade/no-trade decision)
- On-chain data (exchange flows, whale movements)
