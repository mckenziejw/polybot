# Microcap Pump Detection: Research & System Design

*Research date: 2026-03-17*

This document covers the design of a screening system for accumulation patterns preceding microcap cryptocurrency pumps. The problem is fundamentally different from directional prediction on liquid markets — it is anomaly detection for rare, extreme-payoff events on assets with thin liquidity.

**Key asymmetry**: If you screen 1,000 coins and take small positions in 50, losing 80% of them to rug pulls or fizzled momentum, you still profit massively if 3-5 hit a 10-50x move. The math only works if (a) you detect accumulation *before* the parabolic move, (b) you size positions small enough to survive the loss rate, and (c) you can actually exit during the pump.

---

## 1. Signal Taxonomy

### 1.1 Volume Anomalies

**What to look for:**
- **Volume-to-market-cap ratio spikes**: A coin with $50K daily volume suddenly doing $500K in 4 hours, while price is still flat or barely up. This is the single most reliable precursor — someone is accumulating.
- **Volume preceding price**: Volume leads price by 1-24 hours in most pump setups. Measure the volume z-score relative to a 7-day or 30-day rolling baseline.
- **Buy-side volume dominance**: Taker buy volume > 70% of total volume, sustained over multiple hours. On DEXes, this is directly observable from swap direction.
- **Volume clustering by time-of-day**: Unusual volume during typically dead hours (e.g., 3-5 AM UTC for a coin with primarily Asian trading) suggests coordinated activity.
- **Volume without social catalyst**: Volume spikes accompanied by nothing on Twitter/Telegram are more interesting than those accompanying a trending hashtag. The latter is often retail FOMO at the top.

**Strength**: High. Volume is the hardest thing to fake at scale because it costs real money.

**Weakness**: Wash trading on low-fee DEXes and exchanges like MEXC can produce fake volume. Need to filter for genuine net buying pressure, not just raw volume.

### 1.2 Order Book Shape Changes (CEX only)

**What to look for:**
- **Bid stacking below current price**: Large limit buy orders appearing at 3-10% below market, creating a floor. Often placed by market makers or accumulators who want to buy dips.
- **Ask thinning**: Sell-side depth gradually decreasing as sellers are absorbed. A coin with $50K in asks at +5% that drops to $5K in asks is primed to move.
- **Bid/ask depth ratio shift**: Track the ratio of bid depth (within 5%) to ask depth (within 5%) over time. A shift from 1:1 to 3:1 over 24 hours signals accumulation.
- **Spoofing detection**: Large orders appearing and disappearing (lifetime < 30s) on the bid side — suggests someone is trying to create the *appearance* of demand. Paradoxically, this is still a signal because someone is spending effort to manipulate this specific coin.
- **Spread compression**: Spread tightening on a usually wide-spread coin suggests a market maker has arrived, which often precedes a listing or promotional event.

**Strength**: Medium-high on CEXes where you can get L2 data. Provides early signal before volume materializes.

**Weakness**: Not available on DEXes (AMM pools don't have order books). On CEXes, only useful for the subset of microcaps listed on orderbook exchanges.

### 1.3 On-Chain Wallet/Address Patterns

**What to look for:**
- **Whale accumulation**: New wallets accumulating >1% of supply, especially through many small transactions rather than one large buy (deliberate accumulation vs. OTC).
- **Holder count acceleration**: Number of unique holders increasing faster than price — distribution is happening, which is bullish before the pump (but bearish during/after).
- **DEX LP additions**: Someone adding significant liquidity to a Uniswap/Raydium pool. This deepens the book and reduces slippage, often in preparation for a promotion.
- **Token concentration declining from dev wallets**: Team wallets distributing to multiple new addresses (could be vesting, could be market-making preparation).
- **Smart money wallets buying**: Track wallets that have historically bought before 5x+ moves. If 3+ "smart money" wallets buy the same token in 48 hours, that's a strong signal.
- **Bridge activity**: Tokens being bridged from one chain to another (e.g., Ethereum to Solana) can precede a new DEX listing and pump.
- **Contract interactions**: Unusual smart contract calls — token approvals, staking contract deployments, or liquidity lock transactions often precede promotional campaigns.

**Strength**: Very high for DEX tokens. On-chain data is ground truth and cannot be faked (the transactions are real, even if the *intent* is manipulation).

**Weakness**: High latency for analysis (block confirmation times + indexing delay). Many false positives from normal DeFi activity. Requires chain-specific infrastructure.

### 1.4 Social Signals

**What to look for:**
- **Telegram group member count spikes**: A project's Telegram going from 500 to 5,000 members in a week.
- **Twitter mention velocity**: Not absolute count, but rate of change. A coin going from 10 to 200 mentions/day.
- **Influencer posts**: Specific crypto Twitter accounts with track records of calling pumps. Some are paid promoters (which is itself a signal — someone is spending money to promote).
- **Reddit/4chan /biz/ threads**: New threads appearing about obscure tokens, especially with "accumulate" language.
- **Discord alpha group leaks**: Some paid groups share calls that leak to public channels with a delay.
- **Sentiment shift**: Moving from negative/neutral to positive without fundamental catalyst.

**Strength**: Medium. Social signals are noisy but provide context that other signals lack. Best used as confirmation, not primary trigger.

**Weakness**: Very noisy. Bot activity inflates metrics. By the time social is loud, you're often late. Hard to get historical data for backtesting. LunarCrush and similar APIs help but are expensive and incomplete.

### 1.5 Cross-Exchange Price Divergence

**What to look for:**
- **DEX price leading CEX**: The token pumps 5-10% on Uniswap while the Binance/KuCoin price lags. This happens when informed buyers hit the more accessible venue first.
- **CEX-to-CEX divergence**: Price on Gate.io moving before MEXC, suggesting the pump originates on one exchange.
- **Arbitrage gap persistence**: Normally, arb bots close cross-exchange gaps in seconds. If a gap persists for minutes, it suggests the move is real and bots can't keep up with demand.

**Strength**: Medium-high. Very actionable — if you see a divergence, you can buy the lagging venue.

**Weakness**: Requires monitoring many exchange pairs simultaneously. Latency-sensitive. The most obvious divergences are already exploited by arb bots.

### 1.6 Listing & Structural Events

**What to look for:**
- **New exchange listings**: A coin being listed on Binance from only being on KuCoin. The listing announcement itself causes a 20-100% pump. But pre-listing accumulation is the real signal — insiders often buy before announcement.
- **New trading pairs**: A USDT pair being added (previously only BTC pair) increases accessibility.
- **Binance/Coinbase "exploring" announcements**: Even a mention of "under consideration" can cause a spike.
- **Token unlock schedules**: Counter-intuitively, the period *after* a large unlock (once selling pressure is absorbed) can precede a recovery pump.
- **Contract migrations/upgrades**: A token migrating to a new contract (v2 token) often resets the chart and creates pump conditions.
- **Airdrop announcements**: Generate attention and new holders.

**Strength**: High for listing events (well-documented alpha). Lower for other structural events.

**Weakness**: Listing alpha is well-known and increasingly front-run. Insider trading on listings is rampant — hard to compete without the same information.

### 1.7 Developer & Fundamental Activity

**What to look for:**
- **GitHub commit frequency**: A dead project suddenly seeing daily commits.
- **Smart contract deployments**: New contracts deployed by the project's deployer address.
- **Partnership announcements**: Real integrations (not just "MOU signed" press releases).
- **TVL growth**: For DeFi tokens, total value locked increasing before price moves.

**Strength**: Low-medium as a standalone signal (too slow, too noisy). Useful for filtering out dead projects from your screening universe.

**Weakness**: Easy to fake (bot commits, meaningless contract deploys). Fundamental value is largely irrelevant for pump-and-dump dynamics.

### 1.8 Market Structure / Cycle Signals

**What to look for:**
- **BTC dominance declining**: Altcoin season is when microcap pumps cluster. When BTC dominance drops below key levels, capital rotates to smaller coins.
- **Sector rotation patterns**: Pumps often cluster by narrative (AI tokens, then RWA tokens, then memecoins). Detecting which sector is "hot" helps narrow the universe.
- **Recent pump-and-dump on similar coins**: If 3 dog-themed memecoins pumped this week, the 4th is more likely to pump too.
- **Overall market volatility**: Low-vol consolidation periods in BTC precede altcoin pumps as traders seek returns.

**Strength**: Medium. Sets the macro context. During "altcoin season," your hit rate goes up 3-5x.

**Weakness**: Not useful for timing specific coins. Only useful for universe filtering and position sizing.

---

## 2. Data Sources

### 2.1 CEX Market Data

| Source | Data Type | Cost | Latency | Historical | Notes |
|--------|-----------|------|---------|------------|-------|
| **Binance API** | OHLCV, trades, L2 orderbook | Free (rate-limited) | Real-time WS | 2017+ (klines) | Best for coins that make it to Binance. Limited microcap coverage. |
| **KuCoin API** | OHLCV, trades, L2 orderbook | Free | Real-time WS | 2018+ | Better microcap coverage. Less reliable API. |
| **Gate.io API** | OHLCV, trades, L2 | Free | Real-time WS | 2018+ | Excellent microcap coverage (1,700+ pairs). |
| **MEXC API** | OHLCV, trades | Free | Real-time WS | 2020+ | Best microcap coverage of any CEX. Lots of wash trading — filter carefully. |
| **Bybit API** | OHLCV, trades, L2 | Free | Real-time WS | 2020+ | Growing microcap listings. Reliable API. |
| **CCXT** | Unified wrapper for all above | Free (OSS) | Adds ~10ms | Via exchange | Essential library. Use async version for parallel polling. |
| **Tardis.dev** | Historical L2, trades for major CEXes | $99-499/mo | N/A (historical) | 2019+ | Best source for historical orderbook data. Worth the cost for backtesting. |
| **CoinGecko API** | OHLCV, market cap, volume for 13K+ coins | Free (limited) / $129/mo (Pro) | 1-5 min delay | 2014+ | Essential for universe screening. Free tier: 10-30 calls/min. Pro: 500/min. |
| **CoinMarketCap API** | Similar to CoinGecko | Free (limited) / $79/mo | 1-5 min delay | 2013+ | Alternative to CoinGecko. Historical snapshots useful for survivorship bias analysis. |

**Recommendation**: Start with CoinGecko Pro ($129/mo) for universe screening + Gate.io and MEXC free APIs for real-time monitoring of microcaps. Add Tardis.dev ($99/mo) when ready to backtest.

### 2.2 DEX / On-Chain Data

| Source | Data Type | Cost | Latency | Historical | Notes |
|--------|-----------|------|---------|------------|-------|
| **DEXTools API** | DEX trades, pair info, token scores | Free (limited) / $50/mo | ~5s | 6 months | Good UI for manual research. API is rate-limited. |
| **Birdeye API** | Solana DEX trades, token analytics | Free (limited) / $99/mo | ~2s | 2022+ | Best Solana DEX data source. Essential for Raydium/Jupiter. |
| **DexScreener** | Multi-chain DEX pair data, trending | Free (scraping) / unofficial | ~10s | Limited | Great for discovery. No official API — scrape or use community wrappers. |
| **GeckoTerminal API** | Multi-chain DEX data (CoinGecko-owned) | Free | ~10s | 2023+ | Free and multi-chain. Good starting point for DEX data. |
| **Etherscan/BSCScan/Solscan APIs** | Raw transactions, token transfers, contract events | Free (limited) | ~15s | Full chain history | You have Etherscan access. 5 calls/sec free. Need BSCScan + Solscan too. |
| **Alchemy / QuickNode / Infura** | Full node RPC (getLogs, traces) | Free tier available / $50-200/mo | ~1s | Full chain history | Needed for custom on-chain analysis. Alchemy free tier: 300M compute units/mo. |
| **The Graph (subgraphs)** | Indexed on-chain events (Uniswap, etc.) | Free (hosted) | ~30s | Varies by subgraph | Query Uniswap swap events, LP adds, etc. GraphQL API. |
| **Dune Analytics** | SQL on indexed blockchain data | Free (limited) / $349/mo (Pro) | Minutes to hours | Full chain history | Excellent for ad-hoc research. Too slow for real-time screening. |
| **Flipside Crypto** | Similar to Dune (SQL on-chain) | Free | Minutes | Full chain history | Free alternative to Dune. Smaller community but solid data. |
| **Transpose API** | Real-time indexed on-chain data | $50-500/mo | ~3s | Full chain history | Faster than Dune for programmatic access. SQL + real-time streams. |
| **Covalent API** | Multi-chain token balances, transfers | Free (limited) / $100/mo | ~10s | Full chain history | Good for wallet tracking (token balance changes). |

**Recommendation**: Start with GeckoTerminal (free) + Etherscan (free, you have it) + Alchemy free tier. Add Birdeye ($99/mo) for Solana coverage when needed.

### 2.3 Wallet / Smart Money Tracking

| Source | Data Type | Cost | Latency | Notes |
|--------|-----------|------|---------|-------|
| **Nansen** | Labeled wallets, smart money flows, token god candle | $150-2,500/mo | Minutes | Gold standard for wallet labels. Expensive but invaluable. Their "smart money" label set is the best available. |
| **Arkham Intelligence** | Wallet labels, entity tracking, alerts | Free (limited) / $50/mo | Minutes | Cheaper Nansen alternative. Good entity identification. Alert system useful for real-time monitoring. |
| **Lookonchain** | Curated smart money moves (Twitter + dashboard) | Free (Twitter) | Hours | Follow their Twitter. They manually identify whale movements. Good for learning patterns, not for automation. |
| **Zerion API** | Wallet portfolio tracking | Free (limited) | ~10s | Track specific wallets' holdings over time. |
| **Debank API** | Multi-chain wallet tracking | Free (limited) / $200/mo | ~10s | Similar to Zerion. Better multi-chain coverage. |
| **Custom wallet tracking** | Roll your own via RPC + token transfer events | Infra cost only | ~15s | Most flexible. Track specific wallets of interest with Transfer event filters. |

**Recommendation**: Start with Arkham free tier + custom tracking via Etherscan/Alchemy. Build a "watchlist" of 50-100 wallets identified from Lookonchain posts and past pump analysis.

### 2.4 Social / Sentiment Data

| Source | Data Type | Cost | Latency | Notes |
|--------|-----------|------|---------|-------|
| **LunarCrush** | Social volume, sentiment, influencer tracking | $200-1,000/mo | Minutes | Most comprehensive crypto social data. Expensive. |
| **Twitter/X API** | Raw tweets, mention counts | $100/mo (Basic) / $5,000/mo (Pro) | Real-time (streaming) | Basic tier too limited for serious use. Pro is expensive. Consider scraping alternatives. |
| **Telegram API** | Group messages, member counts | Free (user API) | Real-time | Need to join groups. Use Telethon (Python) to monitor group activity. Gray area legally. |
| **Santiment** | Social volume, dev activity, on-chain metrics | $49-250/mo | Hours | Good combined social + on-chain. Historical data for backtesting. |
| **Reddit API** | Subreddit posts, comments | Free (rate-limited) | Minutes | Monitor r/CryptoMoonShots, r/altcoin, etc. |
| **Custom scraping** | 4chan /biz/, Discord, Telegram | Infra cost | Varies | Fragile, but 4chan /biz/ is historically a leading indicator for microcap pumps. |

**Recommendation**: Start with Santiment ($49/mo) for historical social data. Custom Telegram monitoring via Telethon for real-time. Skip Twitter API (too expensive for the value).

---

## 3. Modeling Approach

### 3.1 Problem Framing

This is **not** a standard binary classification problem. It is more accurately described as:

> **Rare event detection with extreme class imbalance and asymmetric payoff, where false negatives are cheap (missed opportunity) and false positives are expensive only if position sizing is wrong.**

The base rate of a "pump" (defined below) for any given microcap coin in any given day is roughly 0.1-1%. But you're screening hundreds or thousands of coins, so at the portfolio level, you expect to see several pumps per week.

### 3.2 Defining "Pump" for Labeling

This is the single most important modeling decision. Options:

**Option A — Magnitude-based (recommended for v1):**
- A coin's price increases by >= 100% (2x) within a rolling 72-hour window, starting from any point where it was below the 30-day moving average.
- The "accumulation" label window is the 24-72 hours preceding the start of the pump.
- Exclude the first 7 days after listing (listing pumps are a different signal).

**Option B — Percentile-based:**
- A coin's 24h return is in the top 0.5% of all coin-days for that market-cap tier.
- More adaptive to market conditions but harder to interpret.

**Option C — Multi-stage:**
- Stage 1: "Accumulation" — volume > 3x baseline, price flat or slightly up (< 20%).
- Stage 2: "Breakout" — price up > 50% in < 12 hours.
- Stage 3: "Pump" — price up > 100% from pre-accumulation level.
- This is the most informative labeling but requires the most complex pipeline.

**Recommendation**: Start with Option A for simplicity. Use a 100% threshold (2x) over 72 hours. Label the 24-48h window before the move starts as "accumulation positive." Everything else is negative.

### 3.3 Handling Class Imbalance

The ratio of positive (pre-pump) to negative (normal) windows will be roughly 1:500 to 1:2000. Standard approaches:

1. **Don't try to classify directly.** Instead, build an anomaly scoring system. Rank all coins by "pump likelihood score" and take the top N. You don't need to predict whether a specific coin will pump — you need the coins that *do* pump to be ranked near the top.

2. **Two-stage architecture:**
   - **Stage 1 (Filter)**: Cheap, high-recall filter that flags 5-10% of coin-days as "interesting." Uses simple thresholds: volume z-score > 2, or holder count increasing > 10%/day, or new smart money wallet detected.
   - **Stage 2 (Scorer)**: More expensive model that scores the filtered set. This is where XGBoost or a neural net operates on a much more balanced dataset (1:20 to 1:50 ratio after filtering).

3. **SMOTE/oversampling**: Generally **not recommended** for this problem. The positive class has high variance (pumps look very different from each other), so synthetic examples add noise. Better to use class weights in the loss function.

4. **Focal loss / class-weighted loss**: If using a neural net, focal loss helps. For XGBoost, use `scale_pos_weight = negative_count / positive_count`.

### 3.4 Feature Engineering

**Volume features (most important):**
- Volume z-score (current 4h volume vs 30-day rolling mean/std)
- Volume trend (slope of log volume over 7 days)
- Buy/sell volume ratio (taker buy / taker sell, if available)
- Volume / market cap ratio
- Volume relative to total DEX pool liquidity
- Number of distinct trading addresses (on-chain)

**Price features:**
- Distance from 30-day low (% above recent bottom)
- ATR (average true range) relative to price — volatility baseline
- Price vs short-term MAs (5d, 10d) — trend detection
- Drawdown from ATH — "how beaten down is this coin?"
- Higher lows pattern (3+ consecutive higher daily lows)

**Order book features (CEX only):**
- Bid/ask depth ratio (within 2%, 5%, 10%)
- Depth change rate (how fast is depth growing?)
- Spread relative to historical spread
- Large order detection (orders > 5x median order size)

**On-chain features:**
- Holder count and rate of change
- Gini coefficient of holdings (concentration)
- Number of "new" wallets (first interaction with this token in 30 days)
- Smart money wallet presence (binary: any known smart money wallet holds this?)
- Transfer count vs unique addresses (many transfers between few addresses = wash)
- DEX liquidity pool depth and change rate

**Social features:**
- Social mention z-score (mentions relative to 30-day baseline)
- Sentiment polarity shift (negative-to-positive transition)
- Influencer mention binary (any tracked influencer mentioned this?)
- Telegram member count change rate

**Market context features:**
- BTC dominance (current + 7-day change)
- BTC 7-day return (positive = risk-on)
- Sector momentum (average 7-day return of same-sector coins)
- Days since last pump for this coin
- Market cap rank (ordinal, not absolute)

### 3.5 Model Architecture

**Recommended: Two-stage anomaly scorer**

```
Stage 1: Rule-based filter (Python)
  IF volume_zscore > 2.0 OR
     holder_count_change_7d > 20% OR
     smart_money_wallet_detected OR
     social_mention_zscore > 3.0
  THEN pass to Stage 2

Stage 2: XGBoost ranker
  Input: ~40-60 features for filtered coins
  Output: pump_probability score
  Training: All positive examples + 20x random negative examples
  Objective: binary:logistic with scale_pos_weight
  Evaluation: Precision@K (K=10, 20, 50)
```

**Why XGBoost over deep learning:**
- Small positive class (maybe 500-2,000 labeled pump events in historical data) means a neural net will overfit.
- XGBoost handles mixed feature types (continuous + binary) natively.
- Feature importance is interpretable — you need to understand *why* the model flags a coin.
- Fast inference (scoring 1,000 coins should take < 1 second).
- You already have XGBoost experience.

**Later iteration: Temporal model**
- Once you have a working scorer, add an LSTM or Transformer that processes the *sequence* of daily feature vectors for each coin over the past 14 days.
- This captures accumulation *patterns* (gradual volume increase over days) that point-in-time features miss.
- Use the XGBoost score as one input feature to the temporal model.

### 3.6 Changepoint Detection (Alternative/Complementary)

Instead of supervised learning, use unsupervised changepoint detection on volume and holder count time series:

- **BOCPD (Bayesian Online Changepoint Detection)**: Detects regime changes in streaming data. Flag any coin where BOCPD detects a volume regime change in the last 24 hours.
- **CUSUM**: Cumulative sum control chart. Simple, fast, well-understood. Set a threshold on the cumulative deviation from expected volume.
- **PELT (Pruned Exact Linear Time)**: Offline changepoint detection. Use in backtesting to identify optimal changepoint thresholds.

These can serve as features in the XGBoost model or as standalone Stage 1 filters.

### 3.7 Online Learning / Adaptation

Microcap pump patterns evolve. What worked in 2024 (Telegram pump groups, simple volume spikes) may not work in 2026 (more sophisticated accumulation, DEX-first launches).

- Retrain the XGBoost model weekly on a rolling 90-day window.
- Track feature importance drift — if a feature that was important becomes irrelevant, investigate why.
- Maintain a "false positive log" — every time the model flags a coin that doesn't pump, record why (rug pull? fakeout? wash trading?). This is your most valuable training data.
- Consider an ensemble where recent data gets higher weight (exponential decay on training samples).

### 3.8 False Positive Cost Analysis

What happens when you're wrong? The model flags a coin, you buy, and it doesn't pump:

| Outcome | Probability | Cost per $100 position |
|---------|-------------|----------------------|
| Coin stays flat, you exit after 72h | ~40% | -$2 to -$5 (spread + fees) |
| Coin dumps 20-50% (normal microcap volatility) | ~30% | -$20 to -$50 |
| Rug pull / exploit / delist (total loss) | ~10-15% | -$100 |
| Coin pumps modestly (20-50%) | ~10% | +$20 to +$50 |
| Coin pumps big (100%+) | ~5-10% | +$100 to +$1,000+ |

**Key insight**: The expected value is positive even with a 10-15% hit rate on "real" pumps, **as long as position sizes are small enough to absorb the rug pull / total loss scenarios.** This is a Kelly Criterion problem (see Section 4).

---

## 4. Execution Strategy

### 4.1 Position Sizing

This is a **barbell strategy**: many small bets with occasionally huge payoffs.

**Kelly Criterion adaptation:**
- Estimate: p = 0.08 (8% of flagged coins actually pump 2x+)
- Estimate: b = 5 (average winning return is 5x, after accounting for partial exits)
- Kelly fraction: f* = (bp - q) / b = (5*0.08 - 0.92) / 5 = -0.104

Wait — that's negative, meaning the raw numbers don't support betting. This is because the 10-15% total loss probability is brutal. Let's refine:
- p_big_win = 0.05, payoff = 10x
- p_small_win = 0.10, payoff = 1.5x
- p_flat = 0.40, payoff = 0.97x (small loss to fees)
- p_dump = 0.30, payoff = 0.65x
- p_rug = 0.15, payoff = 0.0x

Expected value per dollar: 0.05*10 + 0.10*1.5 + 0.40*0.97 + 0.30*0.65 + 0.15*0 = 0.5 + 0.15 + 0.388 + 0.195 + 0 = **1.233**

That's a 23.3% expected return per position, which is strongly positive. The Kelly fraction for this distribution is approximately 10-15% of bankroll across *all* active positions combined — meaning if you have 10 active positions, each should be ~1-1.5% of bankroll.

**Practical sizing:**
- Total allocation to microcap screening: 5-10% of total portfolio
- Per-coin position: $50-200 (assuming $10K allocation = $500-1,000 total across active positions)
- Maximum concurrent positions: 5-10
- Never risk more than 2% of total portfolio on any single microcap

### 4.2 Entry Execution

**The core challenge**: Microcap order books are thin. A $500 market buy can move the price 5-10%.

**Strategies:**
1. **Limit orders at/near ask**: Place a limit buy at the current ask price. If there's genuine accumulation happening, you'll get filled as sellers come in. Patience matters more than speed here.
2. **TWAP over 1-4 hours**: Split your $200 position into 4-8 orders spread over time. This is feasible because your signal typically gives you 12-48 hours of lead time before the parabolic move.
3. **DEX execution**: For Solana memecoins, use Jupiter aggregator with slippage tolerance of 1-2%. For EVM chains, use 1inch or Uniswap with similar settings. DEX execution is simpler (no order management) but slippage is worse.
4. **Multi-exchange arbitrage**: If the coin is on multiple exchanges, buy on the one with the deepest book. MEXC and Gate.io typically have the worst spread but the most microcap listings.

**What NOT to do:**
- Don't market buy. Ever. On a microcap with $10K daily volume, a $500 market buy can move the price 10%.
- Don't chase. If the coin has already moved 30%+ from your signal price, the risk/reward is degraded. Set a maximum entry price threshold.

### 4.3 Exit Strategy

This is where most people fail. The standard failure mode: hold through the pump and give it all back on the dump.

**Tiered exit plan:**
- **Exit 25% at 2x**: Lock in your initial investment. Everything remaining is "house money."
- **Exit 25% at 5x**: Take substantial profit.
- **Trail-stop the remaining 50%**: Use a 30-40% trailing stop from the peak. For a coin that hits 10x, a 30% trailing stop exits at 7x.
- **Time stop**: If a coin hasn't moved >50% within 72 hours of entry, exit the full position. The "accumulation" thesis has failed.
- **Hard stop at -40%**: If the coin dumps 40% from entry, exit everything. Don't let a "pump bet" become a bag-hold.

**Practical considerations:**
- Set exit orders immediately after entry. Don't rely on watching charts.
- On DEXes, there's no native stop-loss. You need a bot monitoring price and auto-selling. Use a service like GoodEntry or build your own with a cron job + Jupiter/Uniswap SDK.
- Sell-side slippage during a pump is *low* (everyone wants to buy). Sell-side slippage during a dump is *high* (everyone wants to sell). This asymmetry means your trailing stop needs to be wide enough to avoid getting stopped out on normal volatility.

### 4.4 Risk Management

**Rug pull protection:**
- Check token contract for: renounced ownership, locked liquidity, no mint function, verified source code. Use tools like TokenSniffer, GoPlus Security API (free), or RugCheck (Solana).
- Never buy tokens with liquidity < $10K in the DEX pool.
- Never buy tokens where top 10 holders control > 80% of supply (excluding exchange wallets and LP contracts).
- Set a maximum loss per day ($200?) and stop trading if hit.

**Exchange risk:**
- Don't keep funds on small CEXes (MEXC, Gate.io) longer than necessary. Deposit, trade, withdraw.
- For DEX execution, use a dedicated hot wallet with limited funds. Never connect your main wallet.

**Correlation risk:**
- Microcap pumps are correlated (they cluster during altcoin season). Don't assume independence between positions.
- If BTC drops 10% in a day, *all* your microcap positions will dump. Consider hedging with a small BTC short during high-exposure periods.

### 4.5 Execution Venue Selection

| Venue | Pros | Cons | Best For |
|-------|------|------|----------|
| **MEXC** | Widest microcap selection, low fees | Wash trading, withdrawal issues, regulatory risk | First-listed CEX coins |
| **Gate.io** | Wide selection, decent API | High withdrawal fees, some fake volume | CEX microcaps |
| **KuCoin** | Good API, reasonable selection | Declining listings, past hack history | Mid-cap alts (not ultra-micro) |
| **Raydium/Jupiter (Solana)** | Fastest DEX, lowest fees, memecoin epicenter | Solana-only, need SOL for gas, frequent network congestion | Solana memecoins (biggest pump category in 2025-2026) |
| **Uniswap (Ethereum)** | Deepest DEX liquidity, most tokens | High gas ($5-50 per swap), slow (12s blocks) | ERC-20 tokens with ETH liquidity |
| **PancakeSwap (BSC)** | Low fees, fast | Declining relevance, more scams | BSC tokens (less common for pumps now) |

**Recommendation**: Focus on Solana (Jupiter) and Gate.io/MEXC as primary venues. Solana memecoins are the largest category of microcap pumps in 2025-2026. Gate.io/MEXC for anything listed on CEX.

### 4.6 Latency Requirements

This is **not** a latency-sensitive strategy. You're looking for accumulation patterns over hours/days, not seconds.

- **Signal generation**: Hourly batches are fine. Run the screening pipeline every 1-4 hours.
- **Entry execution**: Minutes to hours. No need for sub-second execution.
- **Exit execution**: During the pump, you want exits within minutes. The trailing stop bot should check price every 10-30 seconds.
- **On-chain data**: 1-5 minute delay is acceptable for most signals.
- **Social data**: Hourly scans are sufficient.

This means you can run everything on a single server with cron jobs. No need for colocated servers or WebSocket streaming infrastructure (except for the exit bot).

### 4.7 Portfolio Monitoring

**Universe management:**
- Start with a universe of ~500-1,000 coins: all coins with market cap $1M-$50M that have been listed for >7 days and have at least $10K daily volume.
- Refresh the universe daily (new listings, delistings, market cap changes).
- The screening pipeline scores all coins in the universe. Top 20-50 go on a "watchlist." Top 5-10 get positions.

**Monitoring dashboard:**
- Active positions: entry price, current price, P&L, time held, exit levels
- Watchlist: top 50 scored coins with their signal breakdown
- Recent signals: what coins were flagged in the last 24h and what happened
- Performance: hit rate, average return, Sharpe of the microcap allocation

---

## 5. Backtesting Challenges

### 5.1 Survivorship Bias

**The problem**: Most data sources only have currently-listed coins. Coins that were rugged, delisted, or abandoned (which is the *majority* of microcaps) are missing. This inflates backtest returns because you're only testing on coins that survived.

**Mitigations:**
- **CoinGecko/CMC historical snapshots**: Both provide weekly/daily market cap snapshots that include coins that later disappeared. CoinGecko API has `coins/{id}/market_chart/range` which works even for inactive coins if you have the ID.
- **CoinMarketCap historical snapshots**: `historical_snapshot` endpoint returns top N coins by market cap on a specific date. Available back to 2013.
- **Dune/Flipside**: On-chain data never disappears. You can query swap events, token transfers, and LP activity for any token that ever existed on-chain, even if it's now worthless.
- **Build your own survival dataset**: Start recording *all* coins meeting your market cap criteria today. In 6 months, you'll have a clean survivorship-bias-free dataset.

### 5.2 Look-Ahead Bias in Pump Labeling

**The problem**: You define a "pump" as a 2x move over 72 hours. But to label the "accumulation" period before the pump, you need to look forward in time. If any feature in your model can implicitly access this future information, your backtest is worthless.

**Mitigations:**
- **Strict temporal separation**: Label pumps using data from time T+24h to T+96h. Build features using *only* data from time T-168h to T-1h. Leave a 1-hour gap to prevent any information leakage.
- **Walk-forward validation**: Train on months 1-3, validate on month 4, test on month 5. Never use future data for any purpose during training.
- **Feature audit**: For each feature, confirm it can be computed using *only* data available at prediction time. On-chain data has block timestamps — use these, not ingestion timestamps.

### 5.3 Simulating Realistic Fills

**The problem**: You can't just assume you buy at the closing price. On a coin with $10K daily volume, your $200 buy moves the price.

**Mitigations:**
- **Slippage model**: Assume slippage = `position_size / daily_volume * 100%`. So $200 on a $10K volume coin = 2% slippage. This is conservative but reasonable.
- **Volume participation limit**: Never assume you can buy more than 5% of daily volume in one bar. If daily volume is $10K, max position entry per day is $500.
- **DEX AMM simulation**: For DEX tokens, you can simulate the exact slippage using the constant-product formula: `slippage = position_size / (pool_liquidity + position_size)`. Pull historical pool sizes from on-chain data.
- **Exit slippage asymmetry**: During the pump, exit slippage is low (high volume). During a dump, exit slippage is 2-5x entry slippage. Model this explicitly.

### 5.4 Historical Data for Dead Coins

**Practical sources:**
1. **Dune Analytics**: Query `dex.trades` table (or chain-specific equivalents) for any token address. This gives you swap-level data with timestamps and amounts for any token that ever traded on a major DEX. Free tier sufficient for research.
2. **Flipside Crypto**: Similar SQL access, often faster than Dune for historical queries.
3. **Blockchain RPCs + event logs**: For any ERC-20 token, you can get Transfer events from the genesis block. Reconstruct holder counts, volume, and wallet distributions from raw events. This is the most work but the most complete.
4. **The Graph**: Subgraphs for Uniswap v2/v3, Raydium, etc. give you historical swap data. Some subgraphs have been deprecated — check availability.
5. **CoinGecko inactive coins**: CoinGecko keeps historical OHLCV for many delisted coins if you have the coin ID. The `coins/list` endpoint includes inactive coins with `include_platform=true`.

### 5.5 Minimum Viable Backtest Design

**Phase 1: Historical pump identification (1-2 weeks)**
1. Pull daily OHLCV for all coins with market cap $1M-$50M from CoinGecko, going back 12 months.
2. Identify all pump events (2x in 72h).
3. Record: coin ID, pump start date, peak date, peak magnitude, pre-pump 30-day volume, market cap at pump start.
4. This gives you the base rate and magnitude distribution.

**Phase 2: Feature backtesting (2-4 weeks)**
1. For each identified pump, pull 7 days of pre-pump data: volume, OHLCV, holder counts (on-chain), social mentions (Santiment).
2. For a matched set of non-pump coin-weeks (same market cap tier, same date range), pull the same data.
3. Compare distributions: are pre-pump volume z-scores significantly different from baseline?
4. This validates individual signal strength before building a model.

**Phase 3: Model backtesting (2-4 weeks)**
1. Walk-forward: train on months 1-6, predict months 7-8, retrain on months 1-8, predict months 9-10, etc.
2. Primary metric: **Precision@20** — of the top 20 scored coins each week, how many pumped within 7 days?
3. Secondary metric: **Profit factor** — gross profit / gross loss, accounting for slippage and total losses.
4. Target: Precision@20 > 10% with realistic slippage simulation. (10% hit rate on 20 bets/week with 5x average winner = highly profitable.)

---

## 6. Practical First Steps

### 6.1 MVP: Volume Anomaly Screener (2-3 weeks)

The simplest signal that has real alpha is **volume anomaly on coins that haven't moved yet**.

**Week 1: Data pipeline**
```
1. CoinGecko Pro API ($129/mo) → pull daily OHLCV + volume for all coins
   with market cap $1M-$50M (~2,000-3,000 coins)
2. Store in SQLite or Parquet files (one file per day)
3. Compute for each coin each day:
   - volume_zscore: (today_vol - mean_30d) / std_30d
   - price_change_7d: % change over last 7 days
   - market_cap_rank: ordinal position
4. Flag coins where volume_zscore > 3.0 AND price_change_7d < 20%
   (volume spike without corresponding price move = accumulation)
```

**Week 2: Backtest & validation**
```
1. Pull 6-12 months of historical daily data from CoinGecko
2. Run the same volume anomaly filter historically
3. For each flagged coin-day, check if it pumped 2x+ in the next 7 days
4. Calculate: hit rate, average return, false positive rate
5. Optimize thresholds: volume_zscore cutoff, price_change filter, market cap range
```

**Week 3: Live monitoring**
```
1. Cron job every 4 hours: pull latest data, score all coins, output top 20
2. Telegram/Discord bot sends alerts for new flags
3. Paper trade: track what would have happened if you'd bought each flagged coin
4. Run for 2-4 weeks before committing real capital
```

### 6.2 What Data to Acquire First

**Immediate (free):**
1. CoinGecko free API: 10-30 calls/min, sufficient for daily screening of ~2K coins
2. Etherscan API (you have this): token holder counts, transfer events for EVM tokens
3. GeckoTerminal API: DEX pool data (liquidity, volume, new pairs)
4. GoPlus Security API: token contract safety checks (free, essential for rug pull avoidance)

**Month 1 ($180-280/mo):**
1. CoinGecko Pro ($129/mo): higher rate limits, historical data access
2. Santiment Basic ($49/mo): historical social + on-chain metrics, backtesting
3. Alchemy free tier: on-chain RPC access for custom analysis

**Month 2+ (if MVP validates):**
1. Arkham Intelligence ($50/mo): smart money wallet tracking
2. Birdeye ($99/mo): Solana DEX data (if targeting Solana memecoins)
3. Tardis.dev ($99/mo): historical orderbook data for CEX backtesting

### 6.3 Minimum Viable Signal to Test

**Signal**: Volume z-score > 3 with price change < 20% over the volume-spike window.

**Why this first:**
- Single data source (CoinGecko/exchange OHLCV)
- Easy to compute and backtest
- Well-documented in academic literature (volume leads price)
- No on-chain infrastructure needed
- Clear hypothesis to validate or reject

**Expected outcome of the test:**
- If the raw volume signal shows Precision@20 > 5% with realistic assumptions, it's worth building the full system.
- If it's below 3%, volume alone isn't sufficient — need to combine with on-chain or social data for the next test.

### 6.4 Timeline & Compute Requirements

| Phase | Duration | Compute | Cost |
|-------|----------|---------|------|
| MVP data pipeline | 1 week | Local machine | $129 (CoinGecko Pro) |
| Historical backtest | 1-2 weeks | Local machine | $0 (included above) |
| Live paper trading | 2-4 weeks | $5/mo VPS (cron jobs) | $10-20 |
| On-chain feature expansion | 2-3 weeks | Local + Alchemy free tier | $49 (Santiment) |
| XGBoost model development | 2-3 weeks | Local machine | $0 |
| Full system live paper trade | 4 weeks | $5-10/mo VPS | $5-10/mo |
| **Real money pilot** | After 3+ months | Same VPS | $200-500 initial capital |

**Total timeline**: 3-4 months to first real trade.
**Monthly ongoing cost**: ~$200-300/mo for data + infra.
**Compute**: A laptop with 16GB RAM is sufficient for everything. No GPU needed (XGBoost, not deep learning). The bottleneck is data acquisition, not compute.

### 6.5 Technical Stack

```
Data Layer:
  - Python 3.11+
  - pandas / polars for data manipulation
  - SQLite for coin metadata + signal history
  - Parquet for time-series storage (OHLCV, features)
  - requests / aiohttp for API calls

Modeling:
  - XGBoost for scoring model
  - scipy.signal for changepoint detection
  - scikit-learn for preprocessing, evaluation

Execution:
  - ccxt for CEX order management
  - solders + Jupiter SDK for Solana DEX execution (or use Jupiter API)
  - web3.py for EVM DEX execution

Monitoring:
  - python-telegram-bot for alerts
  - Simple Flask dashboard (optional)

Scheduling:
  - cron (Linux) or APScheduler (Python)
  - systemd for process management on VPS
```

---

## Appendix A: Key Risks & Honest Assessment

**This strategy has real edge potential but also real failure modes:**

1. **Regulatory risk**: Many microcap pumps are coordinated manipulation. Participating knowingly in pump-and-dump schemes is illegal in most jurisdictions. The defense is that you're detecting accumulation, not coordinating it — but the line is blurry. Stay on the right side by never participating in pump groups and never promoting coins you hold.

2. **Survivorship bias in your research about this strategy**: The people who talk about making money on microcap pumps are survivors. The silent majority lost money. Be rigorous about your backtest methodology.

3. **Alpha decay**: As more people build pump screeners, the edge will decay. The most accessible signals (CoinGecko volume spikes) will be arbitraged away first. On-chain and social signals have higher barriers and will decay slower.

4. **Rug pull concentration**: The microcap space in 2025-2026 has become increasingly hostile. Solana memecoins have a ~70% rug pull rate. Your rug detection filter is load-bearing — if it fails, you'll lose everything.

5. **Capital efficiency**: Even with a profitable strategy, the absolute dollar amounts may be small. If you can only safely bet $100-200 per coin and get 5 signals per week, your weekly expected profit might be $50-200. This is real money but not life-changing. The strategy scales by improving signal quality, not by increasing position size.

6. **Emotional discipline**: Watching 8 out of 10 positions go to zero while waiting for the 2 that 10x requires discipline that most people lack. The Kelly math works over hundreds of bets, not tens.

## Appendix B: Academic & Practitioner References

- **Xu & Livshits (2019)**: "The Anatomy of a Cryptocurrency Pump-and-Dump Scheme" — empirical analysis of 4,818 pump signals on Telegram, found 1-10 minute advance warning is possible from volume.
- **Hamrick et al. (2021)**: "An Examination of the Cryptocurrency Pump-and-Dump Ecosystem" — 3,400+ pump events catalogued, showed coins with lower market cap and higher pre-pump volume anomalies are most likely targets.
- **Victor & Weintraud (2021)**: "Detecting and Quantifying Wash Trading on Decentralized Exchanges" — methods for filtering fake volume on DEXes.
- **Li et al. (2023)**: "Cryptocurrency Pump-and-Dump Schemes: A Machine Learning Approach" — random forest on price/volume features achieved 75% AUC for pump detection with 6-hour advance warning.
- **ChainalysisReports (ongoing)**: Annual crypto crime reports with data on pump-and-dump prevalence.
