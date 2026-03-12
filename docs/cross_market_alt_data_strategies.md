# Cross-Market & Alternative Data Strategies for BTC Up/Down Markets

> Research date: March 11, 2026
> Context: All directional prediction strategies (XGBoost, momentum, drift, whale signals) failed live despite good backtests. The edge from prediction alone is too thin. This document explores structural and informational edges instead.

---

## 1. Binance-Polymarket Latency Arbitrage

### The Thesis

BTC moves on Binance before Polymarket MMs reprice their orders. If we can detect a BTC move on Binance and buy the correct Polymarket token before MMs cancel/reprice, we capture the stale quote.

### Architecture Analysis

**Polymarket CLOB architecture** (from docs):
- Matching is **centralized off-chain** (operator matches orders)
- Settlement is on-chain (Polygon) but happens AFTER matching
- WebSocket pushes `price_change` events when orders are placed/cancelled
- WebSocket pushes `book` snapshots after trades
- No documented matching engine latency, but the operator is a single centralized service

**Binance feed** (already implemented in `execution/src/data/binance-feed.ts`):
- Uses `bookTicker` stream: real-time best bid/ask updates
- Sub-100ms latency from Binance matching engine to client
- Updates arrive at ~10-50 per second during active periods

### Latency Chain Analysis

The critical path for this arb:

```
BTC moves on Binance
  -> Binance WS bookTicker to our server:     ~50-100ms
  -> Our decision logic:                        ~1-5ms
  -> Place Polymarket order (REST POST /order): ~100-300ms
  -> Polymarket matching engine processes:      ~50-100ms (estimated)
  -> TOTAL: ~200-500ms from BTC move to fill
```

The MM repricing path:

```
BTC moves on Binance
  -> MM's Binance feed (likely colocated):     ~10-30ms
  -> MM's repricing logic:                      ~1-10ms
  -> MM cancels stale orders (REST DELETE):    ~50-200ms
  -> Polymarket processes cancellation:         ~50-100ms
  -> MM posts new orders (REST POST):          ~100-300ms
  -> TOTAL: ~200-640ms from BTC move to MM reprice
```

### Key Insight: The Window

The window is **extremely tight** -- roughly 0-200ms at best. Both paths compete for the same centralized matching engine. The MM has advantages:
- Likely colocated or on fast infrastructure near Polymarket's operator
- Can use batch cancel endpoints
- Has dedicated connectivity

We have one potential advantage:
- We only need to TAKE (one API call), while the MM needs to CANCEL then POST (two API calls)
- Taking is one REST call; repricing is cancel + post = two sequential calls

### Feasibility Assessment

**Edge magnitude**: The 4h market best asks average ~$0.53 (near 50/50). If BTC moves decisively (say 0.1% in 1 second), the fair value of the Up token might shift by 2-5 cents. At $20/trade, that is $0.40-$1.00 per successful latency arb.

**Win rate needed**: With taker fees of up to 1.56% at p=0.50, the fee on a $10 trade is about $0.10. You need the stale quote to be mispriced by at least $0.10 per token to break even, which requires a meaningful BTC move (>0.05%).

**Practical issues**:
- Polymarket rate limit: 3,500 orders/10s burst. Not a bottleneck.
- But REST round-trip to Polymarket from a non-colocated server is 100-300ms. MMs are likely faster.
- BTC moves of 0.05%+ in <500ms happen ~50-200 times per day, but most are noise that reverses immediately.

**Verdict**: MARGINAL. The window exists but is narrow. Worth measuring empirically by logging Binance BTC moves alongside Polymarket book update timestamps. If you consistently see Polymarket books stale for >300ms after Binance moves, there is an opportunity. If MMs reprice within 100ms, this is dead.

**How to test**: Add timestamp logging to BinanceFeed and the Polymarket WebSocket market channel. Compute cross-correlation of BTC returns vs Polymarket mid-price changes with millisecond-resolution lag analysis. This requires NO capital -- just data collection.

**Implementation cost**: LOW -- you already have both feeds connected.

---

## 2. Other Exchanges as Leading Indicators

### Exchange Hierarchy for BTC Price Discovery

The empirical price discovery hierarchy for BTC (as of 2025-2026):

1. **CME Bitcoin Futures** -- institutional flow, often leads during US hours (9:30 AM - 4:00 PM ET). CME is where macro traders and hedge funds express BTC views. During FOMC days, CME leads by 1-5 seconds.

2. **Binance BTCUSDT Perpetual** -- highest volume globally (~$10-20B/day). Leads during Asian hours. The funding rate mechanism creates its own price dynamics.

3. **Binance BTCUSDT Spot** -- you already have this feed. Closely tracks the perp but with a slight lag (50-200ms).

4. **Coinbase BTCUSD Spot** -- important for US-originated flow. Coinbase premium/discount (vs Binance) is itself a signal of US vs international sentiment.

5. **Bybit, OKX perps** -- typically lag Binance by 100-500ms.

### Stacking Multiple Feeds

The idea: if CME futures move, then Binance perp follows, then Binance spot follows, you get a ~200-500ms cascade. By watching the FIRST mover, you might front-run Polymarket MMs who watch Binance spot.

**Feasibility**:
- CME real-time data requires a CME market data license ($100-500/month for non-professional). The CME WebSocket (via Databento, Polygon.io, or TradingView) can deliver updates with ~50ms latency.
- Coinbase WebSocket is free and low-latency.
- Adding 2-3 more feeds is straightforward with your `ExternalFeed` interface.

**Edge magnitude**: The cascade from CME to Binance spot is typically 100-500ms, but it is inconsistent. During high-vol events (CPI releases, FOMC), the cascade is clearest. During quiet periods, all exchanges move near-simultaneously (within 50ms).

**Verdict**: WORTH ADDING but not as primary strategy. Add CME futures and Coinbase feeds. The incremental signal over Binance alone is small (50-200ms earlier) but could tip the balance for latency arb. During macro events, the edge is larger.

**Best API sources for CME BTC futures**:
- **Databento** -- institutional-grade, low latency, ~$50-200/month for BTC futures
- **Polygon.io** -- WebSocket feed, ~$29-99/month tier
- **Binance BTCUSDT Perp** -- free, use `wss://fstream.binance.com/ws/btcusdt@bookTicker` (you just need to change from spot to futures in your BinanceFeed class)

**Implementation cost**: LOW -- add one more `ExternalFeed` implementation. Start with Binance perp (free, trivial change).

---

## 3. Funding Rates & Perpetual Futures as Directional Signal

### The Theory

When BTC perp funding rate is highly positive, longs are aggressively paying shorts. This indicates bullish sentiment/positioning. The question: does this predict short-term BTC direction?

### Empirical Evidence

Academic and practitioner research on funding rates as a predictor:

- **Very short-term (seconds to minutes)**: Funding rate is essentially constant over this window. It is set every 8 hours and changes slowly. It tells you NOTHING about the next 5 minutes of BTC price action. The current funding rate is already priced into the market.

- **Medium-term (hours to days)**: Extreme funding rates (>0.05% per 8h, i.e., >65% annualized) do have mild mean-reversion. Extremely positive funding tends to precede BTC pullbacks (shorts get paid AND price drops). But the signal is noisy and the time horizon is too long for 5m/4h binary markets.

- **As a volatility signal**: This is more promising. When funding rates are volatile (swinging between positive and negative), it indicates uncertainty and regime transitions. High funding volatility correlates with higher BTC realized volatility in the next 1-4 hours.

### Application to Your Markets

For 5-minute markets: **USELESS for direction**. Funding is a slow-moving variable.

For 4-hour markets: **MARGINAL for direction**, potentially useful as a volatility regime filter. High funding rate volatility -> higher BTC vol -> more predictable markets (per your volatility regime research showing high-vol AUC=0.757 vs low-vol 0.587).

**Verdict**: NOT USEFUL for directional prediction. MARGINALLY USEFUL as a volatility regime indicator for the 4h markets. You already have `btc_vol_5m` as a feature -- funding rate adds very little incremental information.

**Implementation cost**: You already have `funding_arb.py`. Extracting Hyperliquid funding rate history is trivial. But the payoff for short-horizon markets is near zero.

---

## 4. Order Flow Analysis from Trade Events Data

### 4a. Trade Flow Imbalance as Volatility Predictor

You have 5,339 trade event files. The idea: don't predict DIRECTION from trade flow, predict VOLATILITY. High trade flow imbalance (lots of buying on one side) means the market is one-sided, which could predict either a trending or mean-reverting outcome.

**How to test**:
```
For each 5m market:
  - Compute buy_volume / total_volume for Up token in first 60s
  - Compute same for Down token
  - Imbalance = abs(buy_up_ratio - 0.5) + abs(buy_down_ratio - 0.5)
  - Correlate with: (a) final outcome binary, (b) max price swing, (c) spread dynamics
```

**Expected finding**: High imbalance likely correlates with LARGER price swings (more volatile markets). This does not directly make money but could be used as a filter: only trade in high-imbalance markets where your model has higher AUC.

**Edge magnitude**: Likely small. Your overnight research already showed that volatility regime (from BTC features) is the dominant filter, and cross-market features add noise.

**Verdict**: TESTABLE with existing data, but likely redundant with BTC volatility features. Test it as a supplement -- if it adds orthogonal information to `btc_vol_5m`, it could improve the vol filter by 1-3pp.

### 4b. Large Trade Detection (Whale Trades)

The idea: a single trade of 100+ tokens in a 5m market is unusual and might indicate informed flow.

**Reality check**: Your whale signal research already showed that whale MMs telegraph via ORDER PLACEMENT, not via TRADES. A whale placing a large resting order is visible in the book; a whale taking liquidity appears as a trade. The trade events data would capture the latter.

**How to test**:
```
For each market:
  - Flag trades > 50 tokens (or > 2 std dev of that market's trade size)
  - Check: does the market resolve in the direction of the large trade?
  - Compute: hit rate of large-trade direction vs market outcome
```

**Expected finding**: Mixed. You already found that trade flow is PREDICTIVE (not anti-predictive), meaning going WITH large trades is slightly better than fading them. But the signal was too weak for profitable directional trading.

**Verdict**: LOW PRIORITY. Whale signals from orderbook positioning are stronger than from trade flow. This is a weaker version of what you already tried.

### 4c. Trade Arrival Rate as Regime Indicator

The idea: when trades arrive faster, the market is more active and potentially more predictable (or more dangerous for MMs).

**How to test**:
```
For each market:
  - Count trades per 10s bucket in first 60s
  - Compare high-activity markets (>50 trades/min) vs low-activity (<10 trades/min)
  - Check prediction model AUC in each regime
```

**Expected finding**: High trade arrival rate likely correlates with high BTC volatility (same underlying cause). Probably redundant with vol features.

**Verdict**: LOW PRIORITY, likely redundant.

---

## 5. On-Chain Signals (Polygon Mempool & Transfers)

### The Architecture Problem

Polymarket matching is **centralized off-chain**. The on-chain settlement on Polygon happens AFTER matching. This means:

1. **Mempool front-running is irrelevant**: Orders go to the centralized CLOB operator, not to the blockchain. There is no mempool for Polymarket orders. You cannot see pending orders before they are matched.

2. **On-chain settlement transactions**: These are visible on Polygon after matching. But by the time the settlement tx appears on-chain, the trade has already been matched and the book has already updated. You are seeing STALE information.

3. **Large USDC.e transfers to Polymarket proxy wallets**: A whale depositing $100K USDC.e to their Polymarket proxy wallet might indicate intent to trade. But the deposit-to-trade pipeline involves: bridge to Polygon -> deposit to proxy -> allowance approval -> order placement. This takes minutes to hours. Not useful for 5-minute or even 4-hour markets.

4. **CTF token transfers**: Watching `splitPosition` and `mergePositions` calls on the CTF contract could show when MMs are minting/redeeming pairs. This is already visible from orderbook behavior (new liquidity appearing on both sides simultaneously = someone just minted pairs).

### One Potentially Useful On-Chain Signal

**Nonce cancellations**: When a trader calls the on-chain nonce increment to invalidate all orders, this is visible on Polygon before the CLOB processes the cancellations. However, the lag is small (Polygon block time ~2s) and the CLOB operator likely processes nonce invalidations near-instantly.

**Verdict**: DEAD END. The centralized CLOB architecture eliminates all on-chain front-running opportunities. The meaningful signals are in the CLOB WebSocket feed, not on-chain.

**Implementation cost**: N/A -- do not pursue.

---

## 6. Sentiment & News Signals

### Available APIs and Data Sources

**Real-time crypto sentiment APIs:**

1. **LunarCrush** -- Social listening across Twitter/X, Reddit, YouTube. API provides real-time "Galaxy Score" and "AltRank" for BTC. Free tier available. Data is aggregated every ~15 minutes (too slow for 5m markets, possibly useful for 4h).

2. **Santiment** -- On-chain + social metrics. "Social Volume" and "Social Dominance" metrics. API at ~$50-200/month. Data granularity: ~5 minutes for social metrics.

3. **The TIE** -- Institutional-grade NLP on crypto Twitter. Expensive ($500+/month). Sub-minute sentiment scores. Used by hedge funds.

4. **CryptoPanic** -- News aggregator with sentiment tags (bullish/bearish/neutral). Free API. Real-time news feed. Could detect breaking news before it hits BTC price.

5. **Twitter/X Firehose** -- Direct access is expensive and complex. Alternative: monitor specific accounts (e.g., @whale_alert, @btc_archive, major analysts) via the free API tier (rate-limited).

### Feasibility for Your Markets

**For 5-minute markets**: Sentiment is essentially noise at this timescale. BTC price moves in 5 minutes are driven by order flow, not by tweets being written and read. The causal chain (event -> tweet -> read -> trade) takes minutes to hours.

**For 4-hour markets**: More plausible but still questionable. The research literature on crypto Twitter sentiment as a predictor shows:
- Sentiment changes CAN lead BTC price moves by 15-60 minutes during major events
- But the signal is noisy: many sentiment spikes have no price impact
- "Breaking news" (exchange hacks, regulatory announcements, ETF decisions) CAN move BTC 2-5% in minutes, but these are rare events (1-5 per month)

**The real question**: Do sentiment shifts predict BTC Up/Down outcomes better than BTC price features alone? Almost certainly not at 5m. Possibly at 4h but only for tail events.

### A More Targeted Approach: Event-Driven Trading

Instead of continuous sentiment monitoring, watch for SPECIFIC events:
- FOMC rate decisions (scheduled, known dates)
- CPI/PPI releases (scheduled)
- Major exchange outages (unscheduled but detectable via status pages)
- Large BTC ETF flow data (published daily with ~1 day lag)

For scheduled events: you KNOW when they happen. You can adjust strategy (increase size during high-vol events, or sit out). No NLP needed -- just a calendar.

For unscheduled events: by the time NLP detects it, the BTC price has already moved. The BTC price IS the signal.

**Verdict**: NOT WORTH THE COMPLEXITY for 5m markets. For 4h markets, a simple economic calendar (FOMC, CPI) is 10x more useful than NLP sentiment analysis and costs nothing to implement.

**Implementation**: Add a calendar check to your strategy: if current time is within 30 minutes of a scheduled macro release, either (a) increase size (higher vol = higher model AUC) or (b) sit out (if you are market making, the spread will blow out).

**Implementation cost**: LOW for calendar approach. HIGH for NLP pipeline with questionable payoff.

---

## 7. Calendar Effects (Time-of-Day, Day-of-Week)

### Known BTC Volatility Patterns

**Intraday patterns (well-documented in academic literature):**
- **US market open (9:30 AM ET)**: Vol spike as equity traders hedge crypto exposure and CME futures open
- **US market close (4:00 PM ET)**: Another vol spike
- **Asian open (~8:00 PM ET / 9:00 AM JST)**: Moderate vol increase
- **London open (~3:00 AM ET)**: Vol increase, particularly for BTC/GBP and BTC/EUR pairs
- **Dead zone (~11:00 PM - 3:00 AM ET)**: Lowest volatility, thinnest liquidity

**Weekly patterns:**
- **Monday**: Higher vol (weekend position unwinds, gap fills)
- **Weekend**: Lower vol, wider spreads, less institutional participation
- **Friday afternoon**: Lower vol (traders flatten before weekend)

**Monthly/Special:**
- **Options expiry (last Friday of month)**: Vol spike around BTC options settlement on Deribit
- **FOMC days**: Vol compressed before announcement, explosive after
- **Quad witching**: Less relevant for crypto but affects correlated equity flows

### Application to Your Markets

Your overnight research already found: `hour_sin` ranked high as a feature but cross-market features collectively added noise. This suggests the time-of-day effect IS captured by your existing BTC volatility features (high vol during US hours -> `btc_vol_5m` is high -> model performs better).

**But there is a simpler application**: Use calendar as an activity FILTER, not a predictor.

```
Strategy: ONLY trade during high-vol windows
  - 4h markets that overlap US market hours (9:30 AM - 4:00 PM ET)
  - Skip 4h markets that span the dead zone (11 PM - 3 AM ET)
  - For 5m markets: only trade during first 2 hours after US/London/Asia open
```

**Expected impact**: Concentrate the same capital on fewer, more predictable markets. If high-vol markets have AUC 0.757 vs low-vol 0.587, trading only during high-vol windows could increase overall hit rate by 5-10pp.

**Verdict**: ACTIONABLE and FREE. This does not require any new data or infrastructure. Simply add a time-of-day filter to your strategy evaluation.

**How to validate**: Take your existing XGBoost backtest results. Split by hour-of-day. Compute per-hour AUC and PnL. Confirm that US-hours markets outperform dead-zone markets. Then hard-code a schedule.

**Implementation cost**: ZERO -- add an `if` statement to your strategy.

---

## 8. Market Microstructure of the CLOB

### Queue Priority

Polymarket docs do not specify queue priority rules, but standard CLOB mechanics apply:
- **Price priority**: Better prices fill first (higher bids, lower asks)
- **Time priority**: At the same price level, earlier orders fill first (FIFO)
- **No pro-rata**: Standard CLOBs use price-time priority, not pro-rata

**Implication for your MM strategy**: Queue position is critical. Your market-maker already avoids unnecessary requotes to preserve queue position (the `shouldRequote` method only triggers on price moves > `requoteThreshold`). This is correct behavior.

### Order Book Pressure Patterns

From your data (798 4h markets, 7,187 5m markets), you can study:

**Pattern 1: Depth asymmetry predicts resolution**
- If ask depth on Up >> ask depth on Down, MMs are more willing to sell Up (bearish signal?)
- Your research found book imbalance is ANTI-predictive (42.6% accuracy). This is because MMs are sophisticated: they post more depth on the side they expect to win (because they want to sell tokens that will be worth $1.00 at the highest price possible).
- This is actually a USEFUL signal if you flip its interpretation: heavy ask depth on Up = MMs are bullish on Up (they want to sell high before resolution).

**Pattern 2: Spread dynamics around fills**
- After a large take, the spread typically widens for 2-10 seconds then mean-reverts
- A MM strategy should avoid posting during spread blowouts and re-enter after stabilization
- Your 500ms `MIN_REQUOTE_INTERVAL_MS` partially addresses this

**Pattern 3: Hidden liquidity (iceberg orders)**
- Some MMs may use iceberg orders (post small visible size, refill after fills)
- Detectable by watching for a level that keeps refilling at the same price after fills
- Your trade events data could confirm: if the same price level gets hit repeatedly with small size and keeps refilling, that is an iceberg
- Icebergs indicate strong conviction at that level -- useful for understanding true support/resistance

**Pattern 4: Last-look behavior**
- The Polymarket CLOB operator has discretion in matching. While docs say matching is fair, the operator could potentially delay or reject matches during extreme events.
- The 3-second matching delay for sports markets suggests the operator CAN impose delays. Whether this happens for crypto markets is unknown.
- **Risk**: If the operator implements any form of speed bump or last-look, latency arb becomes harder.

### Testable Microstructure Hypotheses

1. **Post-fill spread recovery time**: How fast does the spread return to normal after a large take? If recovery is >2 seconds, there is a window to snipe the wide spread.
   - Test with: 5m trade events + book snapshots, compute spread at t+1s, t+2s, t+5s after each fill

2. **Queue depth at BBO vs fill probability**: At the best ask, how many tokens are ahead of a new order? This determines whether posting at BBO is viable or you need to improve by one tick.
   - Test with: book snapshots, measure total depth at BBO across markets

3. **Tick-level price impact**: When someone takes 50 tokens at the ask, how much does the mid move? If price impact is <1 tick for 50 tokens but >2 ticks for 200 tokens, there is a size threshold where stealth trading is possible.
   - Test with: trade events matched against book snapshots at same timestamp

**Verdict**: Microstructure analysis is USEFUL for improving your market-maker execution but unlikely to generate a new standalone edge. The main actionable insight is: avoid posting during spread blowouts, respect queue priority, and detect iceberg orders to understand true levels.

**Implementation cost**: MEDIUM -- requires careful data alignment between trade events and book snapshots.

---

## Ranked Strategy Recommendations

### Tier 1: Do These Now (Free or Trivial Implementation)

1. **Time-of-day filter** -- Only trade during high-vol windows (US hours, macro events). Zero implementation cost, potentially +5-10pp hit rate improvement. Validate with existing backtest data split by hour.

2. **Macro event calendar** -- Hard-code FOMC, CPI, options expiry dates. Increase size or MM aggressiveness during these windows. The vol spike is guaranteed and your model performs better in high-vol regimes.

3. **Binance perp feed** -- Change one line in `BinanceFeed` constructor: use `fstream.binance.com` instead of `stream.binance.com` and subscribe to BTCUSDT perpetual. The perp leads spot by 50-200ms. Free, 5-minute code change.

### Tier 2: Measure First, Then Decide

4. **Latency measurement** -- Log BTC price changes (from Binance) alongside Polymarket book updates (from WebSocket). Compute the empirical lag. If Polymarket books are stale by >300ms after a 0.05%+ BTC move, latency arb is viable. If <100ms, abandon the idea. This requires 1-2 days of data collection and no capital risk.

5. **Trade flow imbalance as vol predictor** -- Analyze your 5,339 trade event files. If trade imbalance in the first 60s adds orthogonal information to `btc_vol_5m` for predicting market difficulty/predictability, use it as an additional regime filter.

### Tier 3: Meaningful Effort, Uncertain Payoff

6. **CME futures feed** -- Add CME BTC futures as a leading indicator via Databento or Polygon.io ($50-200/month). Only useful if latency arb (item 4) shows promise. CME leads Binance by 100-500ms during US hours, which could provide the extra edge needed.

7. **Microstructure execution improvements** -- Post-fill spread recovery analysis, iceberg detection, queue depth optimization. Improves your market maker by reducing adverse selection costs. Worth doing if you are running the MM at scale (>$200 capital per market).

### Tier 4: Do Not Pursue

8. **Funding rates for direction** -- Too slow for 5m/4h horizons. Already captured by vol features.
9. **On-chain signals** -- Dead end due to centralized CLOB architecture.
10. **NLP sentiment** -- Too slow, too noisy, too expensive for the timescale of these markets.
11. **Stacking trade flow for direction** -- Already failed (fading lost, following was marginal).

---

## The Meta-Insight

The pattern from all your research (XGBoost, whale signals, momentum, drift, trade flow, sentiment) is consistent: **directional prediction of 5-minute BTC moves is a solved problem with no remaining edge for a retail participant**. The MMs who dominate these markets have the same Binance feed, faster infrastructure, and more capital.

The strategies that show structural (not predictive) edge are:
1. **Market making** (mint-and-sell captures ask_up + ask_down > $1.00)
2. **Volatility filtering** (only trade when your model is most accurate)
3. **Latency arb** (structural speed advantage, if the window exists)
4. **Cross-platform arb** (Polymarket vs Kalshi price discrepancies)

All of these are about exploiting the STRUCTURE of the market, not predicting direction. This is the right framing going forward.
