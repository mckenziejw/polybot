# Polymarket Sports Markets & Sportsbook Edge Research

*Research date: March 2026*

## Table of Contents
1. [Polymarket Sports API](#1-polymarket-sports-api)
2. [Contract Mechanics for Sports](#2-contract-mechanics-for-sports)
3. [Sharp Sportsbooks & Data Sources](#3-sharp-sportsbooks--data-sources)
4. [The Edge Thesis](#4-the-edge-thesis)
5. [Practical Implementation](#5-practical-implementation)

---

## 1. Polymarket Sports API

### How Sports Markets Differ from Other Markets

Polymarket has first-class sports infrastructure that goes well beyond generic prediction markets:

- **Dedicated metadata endpoints** (`GET /sports`, `GET /teams`, `GET /sports/market-types`)
- **Live WebSocket feed** (`wss://sports-api.polymarket.com/ws`) for real-time scores and game state
- **Team database** with logos, abbreviations, leagues, and records
- **Tag-based categorization** — sports markets are organized via tag IDs that map to specific leagues
- **Auto-cancellation of orders** — outstanding limit orders are automatically cancelled when a game begins (important: if a game starts early, orders may not be cancelled in time)

### API Endpoints

**Gamma API** (`https://gamma-api.polymarket.com`):

| Endpoint | Description |
|----------|-------------|
| `GET /sports` | Sports metadata: sport ID, image, resolution source, ordering (home/away), tag IDs, series |
| `GET /sports/market-types` | Returns valid market type strings (moneyline, spread, totals, etc. — must query to enumerate) |
| `GET /teams?league=NBA&limit=100` | Team data with filtering by league, name, abbreviation |
| `GET /events?tag_id={id}&active=true&closed=false` | List sports events by tag/league |
| `GET /tags` | All available tags including sports leagues |

**CLOB API** (`https://clob.polymarket.com`):

| Endpoint | Description |
|----------|-------------|
| `GET /book?token_id={id}` | Full order book (bids, asks, min_order_size, tick_size) |
| `GET /price?token_id={id}` | Current price |
| `GET /midpoint?token_id={id}` | Midpoint price |
| `GET /spread?token_id={id}` | Bid-ask spread |
| `GET /tick-size?token_id={id}` | Minimum price increment (typically 0.01) |
| `GET /fee-rate?token_id={id}` | Fee rate for the market |

**Sports WebSocket** (`wss://sports-api.polymarket.com/ws`):
- No authentication required
- No subscription message — connect and receive all active sports updates
- Ping/pong heartbeat every 5 seconds (must reply within 10s)
- Message type: `sport_result`

### Available Sports (Known Leagues)

Based on the API documentation and fee structure, confirmed leagues include:

- **Basketball**: NBA, NCAAB (college basketball — fees enabled)
- **Football**: NFL, CFB (college football)
- **Soccer**: Serie A (fees enabled), likely Premier League, Champions League, La Liga, MLS
- **Hockey**: NHL
- **Baseball**: MLB
- **Esports**: Various (uses different status vocabulary)
- **Tennis**: Various tournaments

The `GET /sports` endpoint returns the full list with tag IDs for each.

### Sports WebSocket Message Format

```json
{
  "slug": "nba-lakers-celtics-2026-03-07",
  "live": true,
  "ended": false,
  "score": "98-102",
  "period": "4Q",
  "elapsed": "08:32",
  "last_update": "2026-03-07T02:15:00.000Z",
  "turn": null
}
```

**Sport-specific status values:**
- **NFL/NHL/NBA/CBB/CFB**: Scheduled, InProgress, Final, F/OT, Suspended, Postponed, Delayed, Canceled, Forfeit
- **Soccer**: Adds Break, PenaltyShootout
- **Esports**: not_started, running, finished, postponed, canceled
- **Tennis**: scheduled, inprogress, finished (lowercase)

**Period indicators:**
- Basketball: 1Q, 2Q, 3Q, 4Q, OT
- Football: 1Q, 2Q, 3Q, 4Q, OT
- Soccer: 1H, 2H, HT, FT, PEN
- Baseball: Top/Bot + inning number
- Esports: Map numbers (1/3, 2/3, 3/3 for Bo3)

### Market Structure

Sports markets on Polymarket use the same binary outcome framework as all other markets. Each market resolves to $1.00 (winning outcome) or $0.00 (losing outcome).

**Market types** (exact list via `GET /sports/market-types`, likely includes):
- Moneyline / Winner (Who wins the game?)
- Spread / Handicap (Will team X win by Y+ points?)
- Totals / Over-Under (Will the total score be over/under X?)
- Player props (Will player X score 30+ points?)
- Game props (Will there be overtime?)
- Series/tournament outcomes (Who wins the championship?)

Multi-market events group related outcomes (e.g., "NBA Champion 2026" with one market per team).

---

## 2. Contract Mechanics for Sports

### Resolution

- **Oracle**: UMA Optimistic Oracle
- **Resolution source**: Specified per sport in the `/sports` metadata (e.g., official league website like nba.com, nfl.com)
- **Process**: Anyone can propose an outcome → 2-hour challenge period → resolution (if undisputed)
- **Disputed timeline**: 4-6 days total if challenged (rare for sports with clear outcomes)
- **Bond**: $750 USDC.e to propose
- **Sports advantage**: Game outcomes are objectively verifiable, so disputes are extremely rare. Resolution is typically fast (~2 hours after game ends).

### Trading Windows

- **Pre-game**: Full order book trading until game start
- **Auto-cancellation**: All outstanding limit orders are cancelled when the game begins
- **In-play**: The order book clears at game start. Whether in-play trading is available depends on whether new orders can be placed after this point. The auto-cancel behavior suggests Polymarket may not support robust in-play trading for sports (unlike their crypto markets which trade continuously).
- **Key risk**: If a game starts earlier than its scheduled time, limit orders may not be cancelled promptly.

### Fee Structure (Sports Markets)

Currently only NCAAB and Serie A have fees enabled (as of March 2026). Other sports are fee-free.

**Fee formula**: `fee = C * p * feeRate * (p * (1 - p))^exponent`

| Parameter | Value |
|-----------|-------|
| Fee Rate | 0.0175 |
| Exponent | 1 |
| Maker Rebate | 25% of fees collected |
| Max effective rate | 0.44% (at p=0.50) |
| Min charged | 0.0001 USDC |

Fees are collected in shares on buy orders and USDC on sell orders.

**Holding Rewards**: 4.00% annualized on total position value in eligible markets.

### Order Book Parameters

- **Tick size**: Typically 0.01 (1 cent), queryable per token via `GET /tick-size`
- **Min order size**: Queryable per market via order book endpoint (`min_order_size` field). The CLOB enforces a minimum of 5 tokens for crypto markets; sports likely similar.
- **Typical spread**: Varies by liquidity. Popular games (NBA, NFL) likely 1-3 cents. Niche markets can be 5-10+ cents.

---

## 3. Sharp Sportsbooks & Data Sources

### Tier 1: Sharpest Books (Accept Large Bets, Don't Limit Winners)

| Book | Region | API | Notes |
|------|--------|-----|-------|
| **Pinnacle** | Worldwide (not US) | Yes — free API for affiliates | Gold standard for sharp lines. Lowest margins in industry (2-3% overround). Lines move first. |
| **Circa Sports** | Las Vegas | No public API | Known for accepting high limits. Posts consensus-leading NFL lines. |
| **Bookmaker.eu / CRIS** | Offshore | No public API | One of the oldest sharps-friendly books. Will accept six-figure NFL bets. |
| **BetCRIS** | Offshore | No public API | Sister brand to Bookmaker. |
| **Betfair Exchange** | UK/International | Yes — paid API | Exchange model (peer-to-peer). Most liquid betting exchange globally. True market prices. |

### Tier 2: Market-Moving Books

| Book | Notes |
|------|-------|
| **DraftKings** | US market leader. Moves lines in response to sharp action. Has API for DFS but not sports odds. |
| **FanDuel** | Large US book. Lines move with DraftKings. |
| **BetMGM** | Major US operator. |
| **Caesars** | Large US book, tends to be slower to adjust. |

### Odds Data Aggregators / APIs

#### The Odds API (the-odds-api.com)
- **Best option for this project**
- Covers 70+ sports across 80+ bookmakers worldwide
- Includes Pinnacle, DraftKings, FanDuel, BetMGM, Betfair, and many others
- **Pricing** (as of late 2025):
  - Free tier: 500 requests/month
  - Starter: $20/month — 10,000 requests
  - Standard: $50/month — 50,000 requests
  - Pro: $150/month — 200,000 requests
- **Endpoints**:
  - `GET /v4/sports` — List available sports
  - `GET /v4/sports/{sport}/odds` — Get odds from multiple books
  - `GET /v4/sports/{sport}/scores` — Live scores
  - `GET /v4/sports/{sport}/events/{eventId}/odds` — Single event odds
- **Data**: Returns American, decimal, or fractional odds. Includes moneyline, spreads, totals, and some props.
- **Latency**: Near real-time for pre-game odds. Updates every few seconds.
- **Key limitation**: Free tier is severely limited. For real-time arb detection you need Standard+ tier.

#### Pinnacle API
- Free for registered affiliates (must apply)
- **Endpoints**:
  - `GET /v1/fixtures` — List events
  - `GET /v1/odds` — Current odds (supports `since` parameter for incremental updates)
  - `GET /v1/line` — Get a specific line
- **Rate limit**: ~6 requests per minute per endpoint
- **Latency**: Near real-time. The `since` parameter allows efficient polling for changes.
- **Advantage**: Direct access to the sharpest pre-game and in-play lines in the world.
- **Limitation**: Must be an affiliate; they can revoke access.

#### OddsJam
- Paid service ($99-$399/month)
- Pre-built positive EV alerts and screen
- Compares sharp lines (Pinnacle) to soft books
- Has an API for paid subscribers
- Useful for validation but expensive for automated trading

#### BetQL / Action Network
- Paid tiers ($20-$80/month)
- Line movement tracking, consensus data
- No real-time API suitable for automated trading
- Better for manual analysis

#### Odds API alternatives
- **OddsPortal**: Free historical odds data. Good for backtesting. No real-time API.
- **Betfair Exchange API**: $0 for basic access, paid for streaming. Best for exchange odds.
- **Sportradar**: Enterprise pricing ($$$). Used by sportsbooks themselves. Overkill.
- **Don Best**: Industry standard for real-time odds. Enterprise pricing. Not practical.

### Sharp Money Detection

**Methods to identify sharp line movements:**

1. **Reverse Line Movement (RLM)**: Line moves opposite to public betting percentages. If 70% of tickets are on Team A but the line moves toward Team B, sharp money is likely on Team B.

2. **Steam Moves**: Sudden, simultaneous line movements across multiple books (typically originating from Pinnacle/CRIS). These indicate coordinated sharp action.

3. **Opening Line Value (OLV)**: Pinnacle's opening line is already sharp. When their line moves 1+ points from open, it's almost always due to sharp action.

4. **Consensus data sources**:
   - Pregame.com (free consensus percentages)
   - VegasInsider.com (line movement history)
   - Action Network ($) — ticket counts and handle percentages
   - Scores and Odds (free line movement charts)

---

## 4. The Edge Thesis

### Core Hypothesis

Sportsbook lines (especially Pinnacle) are set by sophisticated algorithms and move within seconds of sharp action. Polymarket sports markets, being newer and less liquid, likely lag behind sportsbook line movements. This creates a window to trade on Polymarket before the market adjusts.

### Speed of Sportsbook Line Movements

- **Pinnacle**: Lines move within 1-5 seconds of sharp action. Pinnacle's model re-prices continuously.
- **US books (DK, FanDuel)**: Follow Pinnacle within 30 seconds to 5 minutes.
- **Market-wide steam moves**: Can propagate across all major books in under 2 minutes.

### Expected Polymarket Lag

Several structural factors suggest Polymarket sports markets will lag sportsbooks:

1. **Participant base**: Polymarket users are primarily prediction market enthusiasts, crypto users, and retail bettors — not professional sports bettors. The books have syndicates with decades of modeling experience.

2. **Liquidity fragmentation**: Each Polymarket market has its own order book. Unlike sportsbooks that cross-subsidize, Polymarket markets are independent.

3. **Market structure**: Polymarket is a limit order book (LOB) while sportsbooks are dealer markets that can instantly adjust posted odds. LOBs need someone to actively cancel stale orders and post new ones.

4. **No market makers with sportsbook feeds**: Traditional sportsbook market makers have direct feeds from Pinnacle/Don Best. Polymarket sports market makers likely do not.

5. **Auto-cancellation at game start**: When the order book clears at game start, there's a gap before new liquidity is posted.

### Where Mispricing Is Most Likely

| Factor | Higher Edge | Lower Edge |
|--------|------------|------------|
| Sport popularity | Niche (soccer leagues, tennis) | NFL, NBA playoff games |
| Bet type | Props, totals, spreads | Simple moneylines |
| Timing | In-play / just before game | Days before game |
| Line movement size | Large steam moves (2+ points) | Small adjustments |
| Polymarket liquidity | Thin books | Deep books |

### Academic/Industry Evidence

- **Prediction markets vs. sportsbooks**: Research consistently shows that prediction markets are efficient for binary political events but less tested for sports. Sportsbooks have more sophisticated pricing models for sports due to decades of data and competition.

- **Betfair vs. sportsbooks**: Studies of Betfair exchange odds show they are comparable to sportsbook closing lines for major events but can lag for niche markets.

- **Closing Line Value (CLV)**: The single best predictor of long-term sports betting profitability. If you consistently bet before lines move to their closing value, you have edge. This is directly applicable — if we can detect where Polymarket's implied probability will end up (using sportsbook lines) and trade before it gets there, we capture CLV.

- **Key paper**: Levitt (2004) "Why are Gambling Markets Organised So Differently from Financial Markets?" — discusses how sportsbooks use information asymmetry differently from exchanges.

### Risk Factors

1. **Polymarket may improve**: As sports markets grow, more sophisticated participants will narrow the gap.
2. **Liquidity limits**: Even if a mispricing exists, available size on Polymarket may be small.
3. **Resolution risk**: UMA oracle disputes (rare but non-zero for edge cases like canceled games, protests).
4. **Fee drag**: 0.44% max fee on NCAAB/Serie A eats into edge.
5. **Capital lockup**: Positions are locked until resolution (game end + ~2 hours for UMA).
6. **Mapping complexity**: Matching sportsbook events to Polymarket markets programmatically is non-trivial.

---

## 5. Practical Implementation

### Recommended Architecture

```
[Pinnacle API / The Odds API]     [Polymarket Gamma API]
         |                                |
         v                                v
   Sharp Line Feed                Sports Market Feed
         |                                |
         +---------+     +----------------+
                   |     |
                   v     v
            Edge Detection Engine
            (compare implied probs)
                   |
                   v
            Signal Generator
            (threshold + confidence)
                   |
                   v
            Polymarket CLOB
            (place orders)
```

### Step 1: Data Pipeline

**Sportsbook odds (pick one or both):**

Option A — The Odds API (recommended to start):
```
GET https://api.the-odds-api.com/v4/sports/basketball_nba/odds
  ?apiKey={key}
  &regions=us,eu
  &markets=h2h,spreads,totals
  &oddsFormat=decimal
  &bookmakers=pinnacle,draftkings,fanduel
```
- Poll every 10-30 seconds
- $50/month for adequate request budget
- Convert decimal odds to implied probability: `prob = 1 / decimal_odds`
- Remove vig using Pinnacle's method: normalize both sides to sum to 1.0

Option B — Pinnacle API (if affiliate access obtained):
```
GET https://guest.api.arcadia.pinnacle.com/0.1/fixtures?sportId=4
GET https://guest.api.arcadia.pinnacle.com/0.1/odds?sportId=4&since={timestamp}
```
- Use `since` parameter for incremental updates (efficient polling)
- Rate limit: ~6 req/min per endpoint
- Free but requires affiliate status

**Polymarket sports markets:**
```
# Discover sports markets
GET https://gamma-api.polymarket.com/sports           # Get sport tag IDs
GET https://gamma-api.polymarket.com/events?tag_id={nba_tag}&active=true&closed=false

# Monitor prices
GET https://clob.polymarket.com/book?token_id={token_id}
GET https://clob.polymarket.com/midpoint?token_id={token_id}

# Live scores
WSS wss://sports-api.polymarket.com/ws
```

### Step 2: Event Mapping

The hardest engineering challenge. Must map between:
- Sportsbook: "Los Angeles Lakers vs Boston Celtics" (various name formats)
- Polymarket: Market slug like "nba-lakers-celtics-2026-03-07"

**Approach:**
1. Fetch Polymarket teams via `GET /teams?league=NBA` to get canonical names and abbreviations
2. Parse Polymarket event titles and slugs
3. Match against sportsbook event names using fuzzy string matching (team names + date)
4. Cache mappings; most team names are stable

**Edge cases:**
- Different date formats / timezone offsets
- Team name variations ("LA Lakers" vs "Los Angeles Lakers" vs "LAL")
- Polymarket may combine markets differently (e.g., "Lakers to win AND score 110+")

### Step 3: Edge Detection

```python
def detect_edge(sharp_prob: float, poly_price: float, threshold: float = 0.03) -> Signal | None:
    """
    Compare sharp implied probability to Polymarket price.

    sharp_prob: Pinnacle's vig-removed implied probability (0-1)
    poly_price: Polymarket YES token price (0-1)
    threshold: minimum edge to trigger (e.g., 0.03 = 3 cents)
    """
    edge = sharp_prob - poly_price

    if abs(edge) >= threshold:
        side = "BUY" if edge > 0 else "SELL"  # sharp says higher → buy YES
        return Signal(
            side=side,
            edge=abs(edge),
            sharp_prob=sharp_prob,
            poly_price=poly_price,
        )
    return None
```

**Key parameters:**
- `threshold`: Start conservatively at 3-5 cents. Lower as you gain confidence.
- Use Pinnacle closing line as the benchmark for "true" probability.
- Apply vig removal before comparing: `true_prob = raw_prob / (prob_yes + prob_no)`

### Step 4: Order Execution

- **Maker orders preferred**: Post limit orders at or near current best bid/ask. Collect maker rebates.
- **Taker orders for urgency**: If edge is large and decaying fast, cross the spread.
- **Size**: Start small (minimum order size, likely 5-10 tokens). Scale up as you validate.
- **Cancel stale orders**: If the sportsbook line moves against you before fill, cancel immediately.

### Latency Requirements

| Scenario | Required Speed | Feasibility |
|----------|---------------|-------------|
| Pre-game arb (line moves) | 10-60 seconds | Very feasible with polling |
| In-play trading | 1-5 seconds | Challenging; requires WebSocket feeds |
| Steam move capture | 5-30 seconds | Feasible if monitoring Pinnacle |

For pre-game trading, polling both data sources every 10-15 seconds is adequate. The Polymarket order book is unlikely to adjust within seconds of a sportsbook line move (unlike crypto where HFT bots operate).

### Capital Efficiency

- **Lock-up period**: From trade to resolution = game duration + UMA resolution (~2 hours minimum)
  - NBA game: ~2.5 hours + 2 hours = ~4.5 hours
  - NFL game: ~3.5 hours + 2 hours = ~5.5 hours
  - Soccer: ~2 hours + 2 hours = ~4 hours
- **Alternative**: Sell position before resolution if the edge has been captured (price moved to fair value). This is the preferred approach — capture the price correction, don't wait for settlement.
- **Holding rewards**: 4% annualized on position value (small but helps)

### Cost Analysis (Per Trade)

| Cost | Amount |
|------|--------|
| Fee (NCAAB/Serie A at p=0.50) | 0.44% max |
| Fee (other sports) | 0% (currently) |
| Spread cost (crossing) | 1-3% typical |
| Gas | Negligible (Polygon) |
| Data cost (The Odds API) | ~$50/month |

**Break-even**: With 0% fees (most sports), you need edge > spread cost (~1-3 cents). With fees (NCAAB), you need edge > ~1.5-3.5 cents.

### Best Free/Cheap Odds Data

| Source | Cost | Best For |
|--------|------|----------|
| The Odds API (free tier) | $0 (500 req/month) | Initial validation / backtesting |
| The Odds API (starter) | $20/month | Light monitoring |
| OddsPortal.com | $0 | Historical odds for backtesting |
| Pinnacle API | $0 (affiliate) | Best real-time sharp lines |
| Pregame.com | $0 | Consensus percentages |

### Recommended Starting Plan

1. **Week 1**: Sign up for The Odds API free tier. Query Polymarket `/sports` and `/teams` endpoints. Build the event mapping pipeline.

2. **Week 2**: Compare Pinnacle implied probabilities to Polymarket prices for all active NBA/NFL markets. Log the data. Measure the actual spread between the two.

3. **Week 3**: Analyze logged data. Identify:
   - Average edge (Pinnacle vs Polymarket)
   - How quickly Polymarket adjusts after sportsbook moves
   - Which sports/bet types show the most persistent edge
   - Typical available liquidity at mispriced levels

4. **Week 4**: If edge exists, implement the signal generator and paper trade. Track CLV (did Polymarket price move toward our entry after we would have traded?).

5. **Month 2**: Go live with minimum size. Scale based on results.

### Key Unknowns (Needs Live Data)

- Actual liquidity on Polymarket sports markets (how many $ can you trade?)
- Whether Polymarket supports in-play trading or just pre-game
- Exact list of available sports and market types (query `GET /sports/market-types`)
- How fast Polymarket prices adjust to sportsbook line moves (this is THE key question)
- Whether Polymarket's auto-cancel at game start creates exploitable gaps

---

## Appendix: Quick Reference

### Polymarket Sports API Cheatsheet

```bash
# Get all sports metadata (tag IDs, resolution sources)
curl https://gamma-api.polymarket.com/sports

# Get valid market types (moneyline, spread, etc.)
curl https://gamma-api.polymarket.com/sports/market-types

# List NBA teams
curl "https://gamma-api.polymarket.com/teams?league=NBA"

# Get active NBA events (replace TAG_ID with NBA tag from /sports)
curl "https://gamma-api.polymarket.com/events?tag_id=TAG_ID&active=true&closed=false&limit=50"

# Get order book for a specific market
curl "https://clob.polymarket.com/book?token_id=TOKEN_ID"

# Get fee rate
curl "https://clob.polymarket.com/fee-rate?token_id=TOKEN_ID"

# Connect to live scores WebSocket
wscat -c wss://sports-api.polymarket.com/ws
```

### The Odds API Cheatsheet

```bash
# List available sports
curl "https://api.the-odds-api.com/v4/sports?apiKey=KEY"

# Get NBA odds from Pinnacle
curl "https://api.the-odds-api.com/v4/sports/basketball_nba/odds?apiKey=KEY&regions=eu&bookmakers=pinnacle&oddsFormat=decimal&markets=h2h,spreads,totals"

# Get live scores
curl "https://api.the-odds-api.com/v4/sports/basketball_nba/scores?apiKey=KEY"
```

### Vig Removal Formula

```python
def remove_vig(prob_yes: float, prob_no: float) -> tuple[float, float]:
    """Remove bookmaker vig to get true implied probabilities."""
    total = prob_yes + prob_no  # Will be > 1.0 due to vig
    return prob_yes / total, prob_no / total

# Example: Pinnacle odds -150 / +130
# Implied: 0.600 / 0.435 (sum = 1.035, vig = 3.5%)
# True: 0.580 / 0.420
```
