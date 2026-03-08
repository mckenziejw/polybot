# Trading Strategy Brainstorms

## What We've Learned (Hard Truths)

Before exploring new ideas, let's be honest about what hasn't worked and why:

1. **Binary prediction markets are directional.** Prices trend from ~0.50 to extremes. 50% of markets end below 0.10 or above 0.90. This kills all symmetric strategies (market making, spread capture, buy-both-sides).

2. **Mid-drift is our best signal but the edge is razor-thin.** Backtest: 71% WR, +$0.0064 EV/dollar. Live: 54% WR, -$65 on 72 trades. The 17pp backtest-to-live gap is the core problem.

3. **Trade flow is predictive, not anti-predictive.** We can't fade it. Following it is just a worse version of mid-drift.

4. **All symmetric strategies fail.** Market making, mint+sell, buy-both-sides — structurally broken in binary markets because adverse selection always wins.

5. **Timeframe doesn't help.** 5m, 15m, 4h all show ~17% time near 0.50 and ~52% extreme resolution. Market making isn't viable at any timeframe.

6. **Backtest-to-live gap is structural.** Historical snapshots are clean and deduped. Live orderbooks are noisy, stale, and gamed. Every signal that looks good on historical data degrades live.

---

## TIER 1: Highest Potential Ideas

### 1A. Selective Trading with ML (XGBoost, not RL)

**Core insight:** Instead of trading every market, train a model to predict WHEN the edge is largest. Only trade high-confidence setups.

**Why XGBoost over PPO+LSTM:**
- Trains on CPU (no GPU costs)
- Interpretable (feature importance tells you what matters)
- Handles tabular features naturally
- No reward shaping / environment design headaches
- Can train on 18,000+ markets in minutes, not hours
- Well-proven for binary classification tasks

**Features to engineer:**
- Mid-price drift at various windows (10s, 30s, 60s)
- Drift velocity (acceleration of mid-price movement)
- Spread at signal time
- Book depth asymmetry
- Trade flow magnitude and direction
- BTC volatility regime (30s, 60s, 300s realized vol)
- BTC trend (5m, 15m, 1h returns from Binance)
- Time of day / day of week
- Recent market outcomes (autocorrelation between adjacent windows)
- Order count / trade count in first N seconds

**Target variable:** Binary — did the drifting token win? (or: did the trade make money after costs?)

**Implementation plan:**
1. Pre-compute all features from Telonex parquet files (CPU, one-time cost)
2. Save as a single feature matrix (market_id, features, label)
3. Train XGBoost with cross-validation (time-split, not random)
4. Analyze feature importance — which features actually matter?
5. Set prediction threshold to only trade top-quintile confidence
6. Backtest the selective strategy: fewer trades, higher win rate

**Why this might work:** Our backtest shows 71% WR across ALL markets. But the WR varies by market conditions. If we can identify the 30% of markets where WR is 85%+, we trade fewer times but with much higher EV.

**Why it might not work:** If the features that predict live performance aren't in the training data (latency, order queue position, live spread dynamics), the model won't help with the backtest-to-live gap.

**Cost: ~$0 in GPU. Just CPU time and our existing data.**

**Self-critique:** This is essentially trying to filter which mid-drift signals to trust. If the backtest-to-live gap is caused by data quality differences (not signal quality differences), no amount of filtering will help. We need to test this quickly before over-investing.

---

### 1B. Cross-Platform Prediction Market Arbitrage (Polymarket vs Kalshi)

**Core insight:** The same events are priced differently across platforms 15-20% of the time, with spreads of 5+ percentage points.

**How it works:** If Polymarket prices "BTC above X" at YES=0.60 and Kalshi prices the equivalent at YES=0.53, buy YES on Kalshi + buy NO on Polymarket. Combined cost < $1.00, guaranteed $1.00 payout regardless of outcome.

**Kalshi details (from research):**
- CFTC-regulated US exchange
- REST API + WebSocket + FIX protocol
- Maker/taker fee structure, max $0.02/contract
- Markets: sports, politics, economics, weather, crypto
- Fee scales with contract price (low fees at extremes)

**Why this is attractive:**
- Risk-free profit (true arbitrage, not statistical)
- No directional view needed
- Both platforms have APIs
- Windows reportedly last seconds → bot advantage
- Capital efficiency is good — positions resolve at known times

**Challenges:**
- Speed: these windows close fast, need sub-second execution on both platforms
- Capital split: need funds deposited on both platforms
- Resolution risk: different platforms may define "same event" differently (the BTC reserve example — Polymarket resolved YES, Kalshi would have resolved NO)
- Slippage: orderbook depth may not support meaningful size
- Regulatory: Kalshi is US-regulated, Polymarket is offshore — may have jurisdictional issues

**Implementation requirements:**
- Kalshi API integration (REST + WebSocket)
- Event matching engine: map Polymarket markets to Kalshi equivalents
- Real-time price monitoring on both platforms
- Atomic-ish execution (place both legs within milliseconds)
- Resolution risk assessment per market pair

**Critical new finding:** Kalshi now offers 15-minute BTC Up/Down markets — essentially the SAME product as Polymarket. They also have 1-hour BTC contracts. An open-source arb bot already exists (polymarket-kalshi-btc-arbitrage-bot) targeting 1-hour markets, reporting 2-5% spreads. However:

- Polymarket charges dynamic fee surcharges on 15m crypto contracts (peak ~50% probability), reducing net arb profitability
- Arb windows last seconds, not minutes — need sub-second execution on both platforms
- Someone has already reported 12-20% monthly returns from cross-platform arb with liquidity-aware execution

**Kalshi API:** Python SDK available (`pip install kalshi-python`), supports REST + WebSocket + FIX. Uses API key + private key PEM auth.

**Cost: Dev time. Kalshi account + deposit.**

**Fee impact on arb:** As of March 6, 2026, Polymarket charges taker fees on ALL crypto markets: max 1.56% at p=0.50, formula `C*p*0.25*(p*(1-p))^2`. Combined with Kalshi's max $0.02/contract fee, the typical 2-5% gross arb spread shrinks to 0-3% net. Arb is only viable on spreads > ~3.5% after both platforms' fees.

**Self-critique:** The fact that open-source arb bots already exist means competition is real. Polymarket explicitly introduced crypto fees to "curb latency arbitrage" — they know this is happening and are taxing it. The 15-minute BTC market overlap is newer and may be less arbitraged than 1-hour markets, but the fee headwind is significant. Resolution risk remains a concern for non-crypto markets. For BTC price above/below X, both platforms likely use similar reference prices, reducing this risk.

---

### 1C. Sports Market Mispricing on Polymarket

**Core insight:** Polymarket sports prices reportedly lag behind sharp line movements from professional sportsbooks. This is a classic "fast market vs slow market" information edge.

**How it works:**
1. Monitor professional sportsbook APIs (odds feeds) for sharp line movements
2. When a sportsbook line moves sharply (e.g., NBA game spread shifts 2+ points), immediately check corresponding Polymarket market
3. If Polymarket hasn't adjusted, buy the underpriced side

**Why this might work:**
- Professional sportsbooks are priced by sharps (professional bettors with real edges)
- Polymarket is priced by retail prediction market participants
- Information flows from sharp books to retail books with delay
- This delay is the edge

**Advantages over crypto prediction markets:**
- Sports outcomes are less correlated with each other (portfolio diversification)
- Discrete events (games) with clear resolution
- Sportsbook odds data is widely available
- Polymarket expanded into sports — liquidity may still be thin (= wider spreads = more opportunity)

**Challenges:**
- Need access to real-time sportsbook odds feeds
- Must identify which Polymarket markets correspond to which sportsbook lines
- Sports market liquidity on Polymarket may be thin
- Odds data latency — if our feed is slow too, the edge is gone
- Capital lockup — sports events take hours/days to resolve (worse capital efficiency than 5m crypto)

**Self-critique:** This is essentially the same edge (fast data source vs slow market) that we tried with BTC momentum and failed. The difference is that sportsbook lines are shaped by professional money, whereas our BTC signal was just raw price data. Whether that makes the signal more robust is an empirical question. Also, if Polymarket sports liquidity is thin, we can't put meaningful size on.

---

## TIER 2: Worth Investigating

### 2A. Funding Rate Arbitrage (Traditional Crypto)

**Core insight:** When perpetual futures trade at a premium to spot, longs pay shorts a "funding rate" every 8 hours. Go long spot + short perp = market neutral, collect funding.

**Current landscape:**
- BTC funding rates fluctuate but can be 0.01-0.1% per 8 hours
- At 0.03%: ~$30 per 8h on $100k position = ~$1,100/month
- Binance offers a built-in funding rate arbitrage bot
- Hyperliquid: 0.045% taker / 0.015% maker (perps), volume discounts available

**Why this is different from what we've been doing:**
- Delta-neutral (no directional risk)
- Well-understood strategy with decades of history in TradFi (basis trade)
- Works with larger capital (scales linearly)
- No signal needed — just monitor rates and execute

**Challenges:**
- Edge has compressed as more participants enter
- Negative funding rate periods can erode profits
- Requires capital on two venues simultaneously
- Liquidation risk if leverage used on the short side
- Opportunity cost of locked capital
- Not available through Polymarket

**Venue comparison:**
| Venue | Maker Fee | Taker Fee | API | Notes |
|-------|-----------|-----------|-----|-------|
| Hyperliquid | 0.015% | 0.045% | Yes, WebSocket + REST | On-chain, no KYC |
| Binance | 0.02% | 0.05% | Full API suite | Built-in arb bot |
| Bybit | varies | varies | Full API | |

**Self-critique:** This is "solved" — everyone knows about it, and the carry has compressed accordingly. The edge is probably 5-15% APY after fees, which is decent but not exciting. It's also a completely different codebase and infrastructure from what we've built.

---

### 2B. Revisit RL with Efficient Training Pipeline

**If we do want to revisit the PPO+LSTM approach, here's how to make training efficient:**

**The problem:** Huge data volumes (18k+ markets, ~100ms resolution), GPU time on VastAI is expensive.

**The solution: Pre-compute everything, minimize GPU time.**

Step 1: **Feature pre-computation (CPU, free)**
- Process all parquet files into fixed-length tensors
- Compute all features: book levels, spreads, BTC returns, vol, etc.
- Normalize and pad sequences to uniform length
- Save as memory-mapped tensors (np.memmap or torch .pt files)
- This is the expensive part computationally but runs on your local machine

Step 2: **Efficient data loading (GPU)**
- Use `torch.utils.data.DataLoader` with `num_workers=4`, `pin_memory=True`
- Pre-fetch batches while GPU trains on current batch
- Use `persistent_workers=True` to avoid DataLoader startup overhead

Step 3: **Mixed precision training (free 2x speedup)**
```python
scaler = torch.cuda.amp.GradScaler()
with torch.cuda.amp.autocast():
    loss = compute_loss(...)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```
- BF16 on A100/H100, FP16 on consumer GPUs
- Cuts memory usage ~50%, training time ~40-50%

Step 4: **Checkpointing for preemptible instances**
- VastAI spot instances are 50-70% cheaper but can be interrupted
- Save checkpoint every N updates (already in train_cleanrl.py at 100k steps)
- Include optimizer state, LSTM hidden state, global step, RNG state
- Upload checkpoints to cloud storage (rsync/rclone to S3/R2)
- Resume seamlessly on new instance

Step 5: **Docker image optimization**
- Pre-build Docker image with all dependencies
- Include pre-computed feature tensors in the image (or mount via network storage)
- Startup time: <30 seconds instead of 10+ minutes installing packages

**VastAI pricing (current):**
| GPU | Price/hr | Best for |
|-----|----------|----------|
| RTX 4090 | $0.20-0.40 | Best value for single-GPU training |
| A100 | ~$0.90 | Larger batch sizes, BF16 |
| H100 | $1.49+ | Overkill for our model size |

**For our model (256-dim LSTM + MLP), an RTX 4090 at ~$0.30/hr is plenty.** A full training run of 10M timesteps should take 2-4 hours = ~$1.

**Self-critique:** The model architecture might not be the bottleneck. The PPO+LSTM might be overfitting to historical market dynamics that don't persist live. Before spending time on efficient training, we should validate that RL adds value over simpler approaches (like the XGBoost selective filter in 1A). If XGBoost on pre-computed features gives us a good confidence filter, the RL approach may be unnecessary.

---

### 2C. Multi-Market Correlation Exploitation

**Core insight:** Adjacent 5m BTC markets are not independent. If the last three windows resolved "Up", BTC is in a trend. This should affect the next window's pricing.

**Possible approaches:**
- Track running BTC trend over multiple windows
- Buy the trend-continuation side early in new windows
- Weight confidence by trend strength and duration
- Combine with mid-drift signal for confirmation

**Also:** 4h and 15m market prices contain information about 5m markets. If the 4h "Up" token is at 0.70, the next several 5m markets should lean "Up".

**Self-critique:** The market probably already prices in the trend. If BTC is trending up strongly, the 5m "Up" token will open at 0.65, not 0.50. The edge only exists if the market under-prices the trend continuation — which is an empirical question. Also, our BTC momentum strategy (3s return) already failed live, suggesting the market is efficient at pricing BTC trends into prediction markets.

---

### 2D. Polymarket Non-Crypto Event Trading

**Available categories:** Politics, sports, weather, entertainment, economics, culture

**Where inefficiencies might live:**
- **Weather markets**: Participants may over/underweight recent weather patterns. Could use weather model output (NWS, ECMWF) as a superior signal
- **Economic data**: Fed decisions, CPI prints. Could use economic models or consensus tracking
- **Politics**: Long duration but behavioral biases (overconfidence, recency bias, partisanship)

**The capital efficiency problem:** These markets take days/weeks/months to resolve. At $5/trade, we'd need hundreds of open positions to match the throughput of 5m crypto markets. Capital is locked for the full duration.

**Math:** 5m markets at $5/trade, 20 trades/hour = $100/hour cycling. Even at 1% edge per trade, that's $1/hour = $24/day. A political market at $500 with 2% edge = $10 total, but capital locked for months. The 5m market generates more absolute return per dollar of capital per unit time.

**Self-critique:** The capital efficiency argument strongly favors short-duration markets. Non-crypto events are worth exploring ONLY if the edge is dramatically larger (10%+ per trade) to compensate for the capital lockup. Sports might be the sweet spot — short duration (hours) with potentially larger mispricings.

---

## TIER 3: Longer Shots

### 2E. Maker Rebate Farming (NEW — enabled by March 2026 fee change)

**Core insight:** Polymarket now charges 1.56% max taker fees on crypto and redistributes 20% to makers. This creates a new income stream for liquidity providers that didn't exist before.

**How it works:**
- Post resting limit orders (maker) on both sides of crypto markets
- Earn 20% of all taker fees collected on your fills
- At 50% probability: taker pays 1.56%, maker earns 0.31% rebate per fill
- Combined with the existing spread capture, this adds a meaningful revenue stream

**Back-of-envelope:**
- $1000 posted across 10 markets ($50/side/market)
- Average fill rate: ~30% per market (from our backtest data)
- Fills: ~$300 in volume
- Rebate at 0.31%: ~$0.93 per rotation
- With 5-minute markets rotating continuously: ~$0.93 * 12/hour * 24 = ~$267/day on $1000 capital

**Why this is better than before:** Previously, maker rebates were negligible. Now with 1.56% taker fees, the 20% rebate is 0.31% — significant when compounded across many small fills.

**Challenges:**
- Still face adverse selection (get filled when wrong)
- Must manage directional inventory actively
- Rebates are paid daily, not per-trade
- Need to be net positive after directional losses

**Self-critique:** This is essentially market making again, which we know loses money due to directional binary market dynamics. The rebate adds ~0.31% per fill but our MM backtest showed -$0.30/pair average loss. The rebate doesn't cover the directional losses. However, if combined with the mid-drift signal (only make markets when signal is absent, use directional when signal fires), there might be a hybrid approach worth testing.

---

### 3A. Liquidity Provision with Directional Hedging

Market make on Polymarket (earn spread + maker rebates) while hedging directional risk using Binance BTC futures.

- When you're net long "Up" tokens, go short BTC futures proportionally
- When you're net long "Down" tokens, go long BTC futures proportionally
- Continuously rebalance as inventory changes

**Why it might work:** Solves the adverse selection problem — when the market moves against your Polymarket position, your hedge profits.

**Why it probably won't:** The hedge ratio is unclear (how many BTC futures per $1 of "Up" tokens?), basis risk between Polymarket pricing and BTC spot, and the spreads in 5m markets (2%) may not cover hedging costs.

---

### 3B. Hyperliquid Market Making / DEX Strategies

Hyperliquid is an on-chain perp DEX with:
- Low fees (0.015% maker, 0.045% taker)
- Good API (WebSocket + REST)
- Growing liquidity
- No KYC

**Possible strategies:**
- Market making on mid-cap perps (wider spreads, less competition)
- Cross-venue arb (Hyperliquid vs Binance for same assets)
- Funding rate harvesting on Hyperliquid specifically

**Self-critique:** Market making on perps is a well-established, competitive field. Without co-location or sub-millisecond infrastructure, we'd be adverse-selected by faster participants. The execution infrastructure (TypeScript/Bun) we've built is Polymarket-specific and would need substantial rework.

---

### 3C. Sentiment/News-Driven Trading

Monitor real-time news feeds, Twitter/X, and other social signals for BTC-moving events. Trade prediction markets based on news before the market reacts.

**Why it might work:** Information propagation has latency. If we detect a major BTC event (exchange hack, ETF approval, regulatory action) before it's priced into prediction markets, we can trade the direction.

**Why it probably won't:** NLP-based trading is extremely competitive. The time window between "event detectable in text" and "event priced into markets" is milliseconds, not seconds. We'd be competing against firms with dedicated infrastructure.

---

### 3D. Whale Watching / Copy Trading

Monitor large Polymarket wallets for their trades. If a whale consistently predicts outcomes correctly, follow their trades with a delay.

**Feasibility:** Polymarket is on Polygon — all transactions are public. Could monitor mempool or track known profitable addresses.

**Self-critique:** By the time we see the whale's trade on-chain and react, the price has already moved. The alpha is in the signal, not the transaction. Also, whales may be market makers, not directional traders.

---

## TIER 4: Speculative / Creative

### 4A. Ensemble of Weak Signals

Instead of finding one strong signal, combine many weak ones:
- Mid-drift (our strongest)
- BTC momentum (weak alone)
- Trade flow direction
- Book depth asymmetry
- BTC volatility regime
- Time-of-day patterns
- Adjacent market outcomes

Each signal might be 52-55% accurate alone, but combined (XGBoost/logistic regression) might reach 65%+ in favorable conditions.

**This is basically 1A (XGBoost selective trading) but framing the model as a signal combiner rather than a confidence filter.**

### 4B. Market Microstructure Features (VPIN, Kyle's Lambda)

Academic market microstructure metrics applied to Polymarket orderbooks:
- **VPIN** (Volume-Synchronized Probability of Informed Trading): measures toxicity of order flow
- **Kyle's Lambda**: measures price impact per unit of volume
- **Amihud illiquidity**: measures price impact per dollar traded

These might identify markets where informed traders are active (and where to follow them, not fade them).

**Self-critique:** These metrics are designed for continuous markets with persistent order flow. Binary prediction markets with 5-minute lifespans and discrete outcomes may not provide enough data for meaningful estimates.

### 4C. Calendar Spread Between Timeframes

If a 4h "Up" market is at 0.70, the three 5m "Up" markets within that window should average out to 0.70. If one of them is at 0.60, it might be underpriced relative to the 4h market.

**Self-critique:** This assumes the 4h market is more efficiently priced than the 5m market, which may not be true. Also, the 5m markets are sequential, not simultaneous — each new one prices in updated BTC information.

---

## Recommended Next Steps (Priority Order)

1. **Build XGBoost feature pipeline** (1A) — cheapest experiment, uses existing data, no GPU costs. If it can identify high-confidence subsets of mid-drift trades, it directly improves our existing strategy.

2. **Investigate Kalshi-Polymarket arb** (1B) — explore the Kalshi API, identify overlapping markets, measure historical price divergences. True arb is the holy grail if it exists.

3. **Sports market signal** (1C) — get sportsbook odds data, check if Polymarket sports prices lag. Quick feasibility check before investing.

4. **If XGBoost works, consider RL** (2B) — only if the feature pipeline reveals that there's signal in the data that a simple model can't capture. Use the efficient training pipeline to keep costs under $5/run.

5. **Funding rate arb** (2A) — steady, boring income. Good for capital that's not deployed in prediction markets. Different infrastructure entirely.

---

## Efficient ML Training Pipeline (Detailed)

For when we're ready to train models (XGBoost or RL):

### Data Preparation (runs locally, one-time)
```
Raw parquet files (Telonex)
    → Load + align timestamps
    → Compute features (mid, spread, depth, returns, vol, flow)
    → Normalize (z-score per feature, computed on training set)
    → Save as feature_matrix.parquet (one row per market-timestep)
    → Or: tensors.pt for PyTorch (pre-batched sequences)
```

### XGBoost Path (CPU only, minutes)
```
feature_matrix.parquet
    → One row per market (aggregate features at signal time)
    → Target: did_win (binary)
    → time-based train/val/test split (no leakage)
    → xgb.XGBClassifier(tree_method='hist', n_estimators=500)
    → Feature importance analysis
    → Calibrated probability output
    → Threshold sweep for precision-recall tradeoff
```

### RL Path (GPU, hours)
```
tensors.pt → Upload to VastAI (or network-mount)
    → Pre-built Docker image (torch, gymnasium, tensorboard)
    → train_cleanrl.py with --resume
    → Mixed precision (AMP)
    → Checkpoint every 100k steps → sync to cloud storage
    → Use RTX 4090 spot instance (~$0.25/hr)
    → Total cost: ~$1-2 per full training run
```

### Key Optimization: Don't Upload Raw Data
The raw parquet files are ~100GB+. Feature tensors are ~1-5GB. Pre-compute locally, upload only what the model needs. This saves both transfer time and storage costs on VastAI.

---

## Empirical Finding: Adjacent Market Autocorrelation

Tested on 2,285 sequential BTC 5m markets from resolution cache:

| Metric | Value |
|--------|-------|
| Base rate (Up wins) | 51.1% |
| P(Up \| prev Up) | 48.6% |
| P(Up \| prev Down) | 53.6% |
| Lag-1 correlation | -0.05 |
| After 5+ same streak | 57.1% reversal rate |

**Conclusion:** Slight mean-reversion tendency, but far too weak to trade alone (corr = -0.05). Worth including as a feature in XGBoost ensemble, not as a standalone signal.

---

## Open Questions

1. Is the backtest-to-live gap caused by **data quality** (noisy live orderbooks) or **market dynamics** (adaptive competitors)? If data quality, we might close the gap with better live data processing. If adaptive competitors, the edge will keep shrinking.

2. Can we get **historical Kalshi data** for overlap analysis? Without historical prices on both platforms, we can't quantify the arb opportunity.

3. How much **Polymarket sports liquidity** actually exists? If it's thin, the theoretical edge is meaningless.

4. Does **BTC volatility regime** predict mid-drift signal quality? Quick analysis was inconclusive due to column name mismatches between ingest/Telonex data formats. Needs proper feature pipeline to answer correctly. This is exactly why the XGBoost approach (1A) should be built carefully with proper data alignment.

5. **ANSWERED: Polymarket fee structure update (March 6, 2026).**
   - ALL crypto markets now have taker fees (activated March 6, 2026!)
   - Fee formula: `fee = C * p * 0.25 * (p * (1-p))^2`
   - Max effective rate at p=0.50: **1.56%**
   - Maker rebate: 20% of collected fees
   - Sports (NCAAB, Serie A): lower fees (rate=0.0175, exponent=1, max 0.44%)
   - **Only applies to markets deployed on/after activation date**
   - This is HUGE — our live mid-drift strategy was running March 6-7, meaning we were paying 1.56% taker fees at ~0.50 entries that we weren't accounting for! This likely explains a chunk of the backtest-to-live gap.
   - At entry price ~0.59 (our average), fee = 0.25 * 0.59 * (0.59 * 0.41)^2 = ~0.86% per trade
   - On a $5 trade, that's ~$0.043 per trade. Over 72 trades = ~$3.10 in fees.
   - Not the whole story (we lost $65) but it adds up, especially on marginal edges.
