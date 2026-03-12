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

## TIER 0: Novel Alpha Ideas (March 2026 Brainstorm)

These ideas emerged from a systematic brainstorm after exhausting standard quant playbook approaches. They focus on structural edges unique to on-chain prediction markets.

### 0A. On-Chain Trade Attribution (HIGHEST PRIORITY)

**Core insight:** Every Polymarket trade settles as an ERC-1155 `TransferSingle` event on Polygon with wallet addresses attached. This solves the exact problem that killed our trade flow strategies — we couldn't distinguish informed from noise flow. With on-chain attribution, we can.

**The system:**
1. Backfill historical `TransferSingle` events from CTF Core (`0x4D97DCd97eC945f40cF65F87097ACe5EA0476045`)
2. Cross-reference with market resolutions to score each wallet's historical PnL and win rate
3. Identify the top 20-50 "sharp" wallets with >60% WR across 100+ resolved markets
4. Monitor their positions in real-time — when sharp wallets cluster on one side, that's the signal
5. Real-time: use `watchContractEvent` on CTF Core for `TransferSingle`, decode `id` field to map to market via `getPositionId`, join with CLOB WebSocket feed by timestamp proximity

**Latency reality:** On-chain events arrive 5-30s after CLOB fills. Not HFT. For 5-minute markets, gives ~10 data points per market. For 4-hour markets, builds a slow-updating directional bias traded via limit orders.

**Why this is different from the killed whale signal (3D below):** The whale signal was about 2 MMs pulling/repositioning orders on BTC books. This tracks *directional bets* by sharp wallets across ALL market types. The alpha is in wallet identification, not market microstructure.

**Infrastructure leverage:** Already have viem, RPC, WebSocket feeds. The backfill is a one-time data pipeline project. Uses existing execution engine for live trading.

**Key risks:**
- Settlement latency means price may have moved by the time the on-chain transfer is visible
- Sharp wallets may use multiple addresses (Sybil)
- Copy-trading alpha decays if others do it too
- Need sufficient resolved markets per wallet to establish statistical significance

**Status:** Not started. First step: pull historical ERC-1155 events and score wallets.

---

### 0B. Multi-Outcome Dutch Book Scanner

**Core insight:** Markets with 3+ outcomes (election fields, awards, sports divisions) frequently have prices summing to >$1.00 or <$1.00. This is pure arbitrage — no prediction skill needed.

**How it works:**
- When `sum(best_bids) > $1.00 + fees`: sell all outcomes → guaranteed profit
- When `sum(best_asks) < $1.00 - fees`: buy all outcomes → guaranteed $1.00 at resolution
- Can also mint all outcomes for $1.00 and sell each at market when sum of bids > $1.00

**Why it might be under-exploited:** The more outcomes in a market, the more likely mispricing exists. Wide-field election markets, sports division winners, "who will win the Grammy" etc. Individual outcomes may have thin liquidity, meaning arbs can't fully exploit the mispricing.

**Fee considerations:** Sports markets are mostly fee-free. Crypto multi-outcome markets have taker fees (max 1.56% at p=0.50). Need `sum > $1.02` to clear spread + fees.

**Key risks:**
- Thin liquidity on individual outcomes may prevent execution of all legs
- Partial fills leave directional exposure
- Need fast multi-leg execution (sequential limit orders vs. market orders)
- Competition from other arb bots

**Status:** TESTED — NOT VIABLE. Scanned all 2,654 active NegRisk events with orderbooks (out of 8,158 total active events). Only 2 had bid_sum > $1.02, both from stale/phantom bids on illiquid long-tail outcomes with $0.34-0.97 spreads. The NegRisk conversion mechanism + existing arb bots keep prices extremely tight. 352 events had bid_sum in $0.98-1.00 range. Market is well-arbitraged.

---

### 0C. Resolution Timing Arbitrage

**Core insight:** There's a window between when an outcome is *knowable* (game ends, vote counted, data released) and when UMA oracle *resolves* it. During this window, losing contracts may still trade above zero.

**Mechanism:**
- UMA Optimistic Oracle resolution process: (1) `proposePrice` with outcome, (2) dispute window, (3) `reportPayouts` on CTF Core
- For BTC 5m/4h markets: automated resolver shortcuts the dispute window, but there's still latency
- For sports/events: game ends → result is public → but resolution takes minutes to hours
- During this window, can buy winning side at discount or sell losing side before it zeroes

**On-chain variant (Resolution Race):**
- Monitor Polygon mempool for `reportPayouts` calls targeting your market's conditionId
- When you see one in pending transactions, you know the resolution outcome before the CLOB freezes
- Submit a market order buying the winning side at whatever price is available
- Window: ~2 seconds (Polygon block time)

**Key risks:**
- Polymarket's operator may freeze the CLOB server-side before the chain tx confirms (likely for automated markets)
- Narrow mempool window (2s block times on Polygon)
- Need mempool access (Alchemy/QuickNode/Blocknative)
- Stale liquidity may not exist on the book at resolution time

**Investigation plan:** Monitor a handful of sports markets through resolution, measure latency and residual prices. Check if CLOB freezes before or after on-chain resolution.

**Status:** INVESTIGATED — MARGINAL. Analyzed 1,000 raw Telonex files (Feb-Mar 2026 BTC 5m markets):
- CLOB stays open ~1-2 seconds after market close before book is cleared
- 3.4% of markets had actionable stale liquidity post-close (both sides)
- Average profit: $6.73/opportunity, available at ~0.3s after close
- Extrapolated: ~$67/day gross (~$2k/month) from ~10 opportunities/day
- BUT: sub-second execution window, someone already exploiting (0.99 bid spikes visible), taker fees apply
- Would require dedicated low-latency infrastructure watching every 5m market
- Not worth building as standalone strategy; could be a "bonus" on top of existing market-watching infra
- **4x multiplier**: BTC+ETH+SOL+XRP all have 5m markets → ~$270/day, ~$8k/month gross
- **Pre-open sell straddle idea**: orderbook opens 12+ hours before market start. Place limit sells at ~$0.51 on both YES+NO → first in queue. BACKTESTED: both-fill rate 25%, single-fill 45%, neither 30%. Mean PnL = -$0.038/market. Adverse selection kills it: when one side fills, the buyer was informed (55-66% of the time they bought the winner). Same adverse selection problem as market making. Would need a way to avoid single fills or exit the losing position quickly.

---

### 0D. Split/Merge Parity Arbitrage (CTF Mint/Merge)

**Core insight:** YES + NO must equal $1.00 via CTF split/merge, but the CLOB only approximates this through arbitrageurs.

**How it works:**
- When `YES_ask + NO_ask < $0.995`: buy both on CLOB, merge on-chain for $1.00
- When `YES_bid + NO_bid > $1.005`: split $1.00 on-chain, sell both on CLOB
- Gas costs on Polygon are negligible (~0.01 MATIC)

**Where it might be under-exploited:** Well-arbitraged on high-liquidity markets, but on thin BTC 5m/4h markets during transitions (new market opens, old one resolving), parity can break briefly. BTC volatility events can also knock books out of parity.

**Key constraint:** On-chain split/merge via Gnosis Safe takes ~5-10s. Someone with a direct EOA could beat you. The window needs to persist long enough for execution.

**Implementation:** Add a parity check to existing WebSocket market data handler. Log frequency and duration of parity breaks before committing to execution.

**Status:** SKIPPED — likely well-arbed. If 2,654 multi-outcome NegRisk markets maintain tight parity, simple binary YES+NO parity (even easier to arb) is almost certainly locked down.

---

### 0E. New Market Sniping

**Core insight:** Fresh markets often have inefficient initial pricing before sophisticated participants arrive. First-mover advantage in thinly-traded new markets.

**Two approaches:**

**1. CLOB monitoring:** Watch Polymarket's market creation feed, evaluate mispricing against external data, take positions early in the first hour when spreads are wide (often 5-10 cents).

**2. On-chain `registerToken` sniping:** Before a market goes live on the CLOB, the admin calls `registerToken(tokenId, complementId, conditionId)` on the CTF Exchange contract. This on-chain tx is visible before the market appears in the Polymarket UI/API. Pre-split USDC into YES+NO, post initial quotes as soon as CLOB opens.

- CTF Exchange: `0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E`
- NegRisk CTF Exchange: `0xC5d563A36AE78145C45a50134d48A1215220f80a`

**Key risks:**
- Need to correctly price the market to avoid adverse selection from informed traders
- `questionId` in `prepareCondition` may not be human-readable
- BTC recurring markets are programmatic (less edge), event markets have larger spreads but require domain knowledge

**Status:** INVESTIGATED — LIMITED. Key findings:
- On-chain `registerToken` happens AFTER market appears in Gamma API (3-7s later). No on-chain detection advantage for event markets.
- Exception: recurring crypto markets are pre-registered on-chain ~5min before API appearance, but these are predictable anyway.
- Sports markets: created minutes before games with tight spreads (2-5c) and immediate MM liquidity. Already efficiently priced.
- Esports/niche: wide spreads (8-95c) but near-zero liquidity ($50-800). Not enough capital at stake to matter.
- High-value event markets (politics, elections): created weeks/months ahead, efficiently priced within hours.
- Conclusion: no practical sniping edge via on-chain detection. API polling could catch new markets but they're either efficiently priced (sports, politics) or too thin to bother (esports, niche).

---

### 0F. PACER / Primary Source Monitoring

**Core insight:** Court filings (PACER), SEC EDGAR filings, government data releases are public but not widely monitored in real-time. Markets on legal/regulatory outcomes could be front-run by monitoring primary sources.

**Examples:**
- SEC enforcement actions → crypto regulatory markets
- Court rulings → legal outcome markets
- Government economic data → inflation/employment markets
- FDA approvals → pharma prediction markets

**Key barriers:**
- Requires domain-specific parsing and market mapping
- PACER charges per-document fees ($0.10/page)
- Need to identify which Polymarket markets correspond to which filings
- Latency between filing and market price reaction is unknown

**Status:** REJECTED — requires deep domain expertise per market category that we don't have, and monitoring infrastructure would be complex for uncertain payoff.

---

### 0G. Operator Settlement Batching Pattern Exploitation

**Core insight:** The Polymarket operator batches trades into `fillOrders` transactions with a predictable timing pattern (already measured via `settlement_latency.py`).

**Use cases:**
- Time own on-chain operations (splits, merges, nonce increments) to avoid colliding with operator batches — reduces Gnosis Safe nonce collision issues
- Observe operator's batch tx in mempool → infer CLOB book state
- Infrastructure optimization more than standalone alpha

**Status:** Partially explored via settlement_latency.py. Low priority as standalone strategy.

---

## Recommended Next Steps (Priority Order)

**Novel strategies (Tier 0):**
1. **On-chain trade attribution** (0A) — highest EV, solves the core problem that killed earlier strategies. Build wallet scorecards from historical on-chain data first.
2. **Dutch book scanner** (0B) — low-effort scanner, either finds opportunities or doesn't. Run alongside #1.
3. **Resolution timing feasibility check** (0C) — 1-hour investigation: monitor a few markets through resolution, check if CLOB freezes before/after chain tx.
4. **Parity break monitoring** (0D) — add monitoring to existing WebSocket handler, measure frequency/duration of breaks.

**Previously explored strategies (Tiers 1-4):**
5. **XGBoost confidence filter** (1A) — currently live as v3.3, walk-forward confirmed 85.8% WR. Ongoing refinement.
6. **Cross-platform arb** (1B) — viable but competitive, fees eat margins. Kalshi API integration needed.
7. **Sports market mispricing** (1C) — favorite-longshot bias confirmed but not tradeable (spread absorbs edge).

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
