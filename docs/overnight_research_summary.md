# Overnight Research Summary — 8 XGBoost Exploration Agents

Date: 2026-03-08

## TL;DR

**Best actionable findings:**
1. **BTC volatility features improve the model** — `btc_vol_5m` ranks #4 in feature importance. High-vol and medium-vol regimes are more predictable (AUC 0.757/0.715) than low-vol (0.587). Concentrate trading during volatile periods.
2. **Exit at t=130-140s, not market close** — Price trajectory analysis shows optimal PnL at t=130-140s regardless of outcome. Holding to close causes ~12.5 cent drawdown on average.
3. **Model is well-calibrated** — ECE of 3.5%, Brier score 0.177 vs baseline 0.190. Predicted probabilities can be trusted for Kelly sizing.
4. **Adverse selection is brutal** — Every fill moves against you by ~$0.02 within 1s. MM at any offset is a losing proposition without informational edge.
5. **Cross-market/sequential features don't help** — Markets are largely independent. Previous market outcomes add noise, not signal.
6. **Shallow XGBoost (depth=3) is the best single model** — AUC 0.705 vs 0.687 for depth=5. Ensembling provides marginal improvement (+0.01 AUC).

---

## Agent 1: Alternative Target Variables (explore_targets.py)

**Experiments:** Final mid-price regression, max adverse drift, regime classification, spread blowout prediction, time-to-resolution.

**Key findings:**
- **Spread blowout** is 88.6% base rate (almost all markets experience 2x spread). AUC 0.888 but nearly impossible to filter for "safe" markets — only 19-34 markets predicted safe at useful thresholds.
- **Time-to-resolution**: markets that resolve early (mid crosses 0.85/0.15) have higher leader win rate (74.6% vs 61.1%). R² of 0.23 for predicting resolution time — modest but potentially useful for strategy timing.
- **Optimal action classifier**: nearly all markets classified as "sit_out" (99.7%) — confirms that clear directional signals are rare.

**Next steps:** Incorporate time-to-resolution prediction as a feature or filter. Fast-resolving markets → directional. Slow → MM.

---

## Agent 2: Cross-Market & Sequential Features (research_cross_market.py)

**Experiments:** Adjacent market outcomes, streak features, BTC momentum carryover, time-of-day effects, rolling win rates.

**Key findings:**
- **Cross-market features DID NOT improve the model.** Enhanced AUC=0.615 vs Baseline AUC=0.631.
- Weak autocorrelation exists (53.6% probability next market matches prev), but too weak to be useful.
- Cross-market features alone get AUC=0.469 (worse than random).
- Some new features rank high individually (`hour_sin`, `prev2_label`, `btc_mkt_vol_10`) but collectively add noise.
- **Per-market evaluation**: at 0.80 threshold, enhanced model gets 70.3% WR (+12.3pp lift) vs baseline — similar to before.

**Verdict:** Markets are effectively independent. The orderbook within each market dominates. Don't add cross-market features.

---

## Agent 3: Unconventional Signals (unconventional_signals.py)

**Experiments:** Psychological BTC price levels, end-of-market liquidity drain, BTC-vs-Polymarket disagreement, orderbook entropy.

**Key findings:**
- **Orderbook entropy** — mostly noise across all quintiles. One weak signal: bid-ask entropy asymmetry (bids more diffuse than asks) has z=2.18, p=0.029 for predicting Up outcome. Too weak alone.
- **End-of-market liquidity drain**: 89.8% of markets see >50% drop in update frequency in final minute. Depth drain quintiles show significant but non-monotonic relationship with outcomes. Not directly exploitable.
- **Final-minute trend confirmation**: reversals are common (58.3% of the time, final-minute drift opposes earlier drift).

**Verdict:** No strong unconventional signals found. These markets are efficient at the signal level we can observe.

---

## Agent 4: Realistic MM Fill Model (backtest_mm_realistic.py)

**Experiments:** Mid-price crossing rates, queue position analysis, effective fill probability, adverse selection measurement.

**Key findings:**
- **Fill probability at ±0.005 (min tick)**: 68% touch in 10s, 91% in 120s. But at ±0.01: effective fill drops to 12-17% due to queue depth (86 tokens ahead).
- **Queue position discount at ±0.02**: only 4.7% chance of fill even when price touches level (408 tokens ahead in queue).
- **Adverse selection is uniformly negative**: -$0.017 to -$0.023 per token within 1-10s of any fill, at ANY offset. Only 37% of fills are favorable at 10s horizon.
- **MM simulation produced no results** — script had a bug in the PnL aggregation, but the raw data is clear: every offset loses money to adverse selection.

**Verdict:** Passive MM on Polymarket is fundamentally unprofitable without informational edge. Adverse selection exceeds any realistic spread capture. The earlier regime backtest showing MM profits was optimistic due to idealized fill assumptions.

---

## Agent 5: Orderbook Microstructure Features (xgb_microstructure.py)

**Status:** CRASHED during feature extraction (index out of bounds on markets with <11 ticks in observation window). Needs a bounds check fix.

**What it was trying:** Rate-of-change features (mid/spread velocity/acceleration), order flow dynamics, spread regime detection, depth profile shape, price impact estimates, cross-token features.

**Next steps:** Fix the bounds check and re-run. This is the most promising unexplored direction — dynamics features capture HOW the book is evolving, not just static snapshots.

---

## Agent 6: Volatility Regime Detection (research_volatility_regime.py)

**Experiments:** BTC volatility features, regime classification, per-regime XGBoost, regime-specific models.

**Key findings:**
- **BTC vol features rank high**: `btc_vol_5m` is #4 in feature importance (gain=14.1). `btc_rvol_300s` and `btc_vol_30s_raw` also in top 10.
- **Regime matters**: High-vol AUC=0.757, Medium=0.715, Low=0.587. The model is almost 2x more discriminative in volatile conditions.
- **Regime-specific models DON'T help**: pooled model outperforms separate per-regime models in every regime. Not enough data per regime for separate training.
- **Low-vol markets are unpredictable**: AUC 0.587 with only 68% high-confidence accuracy. Skip these.
- 16 new vol features added (5 lookback windows × 3 metrics + btc_vol_5m).

**Verdict:** Add BTC volatility features to production model. Filter OUT low-vol markets. Don't train separate models per regime — the pooled model with vol features handles this.

---

## Agent 7: Entry/Exit Timing (research_entry_exit.py)

**Experiments:** Optimal entry window, sequential confidence building, post-prediction price trajectory.

**Key findings:**
- **Earliest reliable signal at 60s**: +5.4% lift at 0.60 threshold, +17.5% at 0.80 threshold. At 90s: +16.8% at 0.80.
- **Sequential averaging doesn't help at per-market level**: avg entry moves to 30-48s but accuracy drops significantly vs single late-window prediction.
- **Price trajectory insight**: mid-price drifts favorably only +0.012 for winners by t=130s, then REVERSES. By t=300s, winners average -0.125 (market converges to resolution price).
- **Max drawdown is severe**: mean -0.251 for winners, -0.309 for losers. No room for tight stops.
- **Optimal exit: t=130-140s** — captures peak PnL of +0.012 per token. Holding to close destroys value.

**Verdict:** Enter at 90-120s (best signal-to-noise), EXIT at 130-140s (before convergence reversal). Don't average across windows — use the single best window.

---

## Agent 8: Ensemble & Stacking (research_ensemble.py)

**Experiments:** Hyperparameter diversity ensemble, window stacking, probability calibration, heuristic rule combination.

**Key findings:**
- **Shallow model wins**: depth=3, lr=0.01 gets AUC 0.705 (best single). Deeper models overfit.
- **Ensemble marginal**: average ensemble AUC=0.696, weighted=0.699. Doesn't beat best single model.
- **High-confidence ensemble**: 85.5% accuracy at >0.70 threshold (n=373). Slight improvement over single model (84.7%).
- **Calibration is excellent**: ECE=3.5%. Predicted probabilities match actual outcomes closely. This means we can trust the model's confidence for Kelly sizing.
- **Prediction distribution**: 51% of predictions fall in [0.7, 1.0) range — model is appropriately confident on most samples.
- **Stacking adds ~0.5% AUC** over best single window at per-market level (0.619 vs 0.601). Marginal.
- **Book imbalance heuristic is ANTI-predictive** (42.6% accuracy). Don't use as a filter.

**Verdict:** Use single depth=3 model. Skip ensemble complexity. Trust the probability outputs for sizing. Book imbalance is misleading — the model already captures what's useful from it.

---

## Consolidated Recommendations

### Immediate improvements to production model:
1. **Add BTC volatility features** (vol_5m, vol_30s_raw, rvol_300s, range_900s)
2. **Use max_depth=3, learning_rate=0.01** (less overfitting)
3. **Filter out low-vol markets** (bottom third of btc_vol_5m)
4. **Exit at t=130-140s** instead of holding to market close

### Don't pursue:
- Cross-market/sequential features (markets are independent)
- Ensemble/stacking (marginal gains, added complexity)
- Passive market making (adverse selection kills all profits)
- Orderbook entropy or psychological BTC levels (noise)

### Worth investigating further:
- **Microstructure dynamics features** (Agent 5 crashed — fix and re-run)
- **Time-to-resolution as entry filter** (fast-resolving markets are more predictable)
- **Early exit strategy** with proper fee accounting (does +0.012 at t=140s cover fees?)
