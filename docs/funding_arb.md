# Funding Rate Arbitrage on Hyperliquid

## How Delta-Neutral Funding Arbitrage Works

Perpetual futures contracts use **funding rates** to keep the perp price anchored to the spot price. When the perp trades above spot (bullish sentiment), longs pay shorts. When it trades below spot, shorts pay longs. On Hyperliquid, funding is settled every **8 hours** (3 times per day).

A **delta-neutral funding arb** captures this payment without directional exposure:

1. **Positive funding** (longs pay shorts): SHORT the perp + LONG the spot asset. You collect funding from the short position while the long spot position hedges your price risk.

2. **Negative funding** (shorts pay longs): LONG the perp + SHORT the spot asset. You collect funding from the long position while the short spot hedges. (Note: shorting spot requires a lending market or a second venue with inverse exposure.)

Your net delta is zero -- you profit purely from the funding payments, regardless of which direction the asset moves.

### Yield Calculation

```
Annualized Yield = funding_rate_per_8h * 3 * 365 * 100

Example: 0.01% per 8h = 0.0001 * 3 * 365 = 10.95% annualized
```

## Risk Factors

### Funding Rate Risk
- Funding rates are **variable** and can flip sign at any time
- A position entered at +50% annualized can drop to 0% or go negative
- Historical consistency (% of periods positive, volatility) matters more than current snapshot
- The tool's "consistency score" (avg/std) helps gauge stability

### Execution Risk
- **Entry/exit slippage**: market orders incur taker fees (0.045%) vs maker (0.015%)
- **Leg risk**: if one side fills and the other doesn't, you have unhedged exposure
- Mitigate by using limit orders and ensuring sufficient liquidity (check OI)

### Liquidation Risk
- Perp positions require margin; adverse price moves can trigger liquidation
- Use conservative leverage (1-2x) and monitor margin ratio
- Even a "hedged" position can be liquidated on the perp side if margin is insufficient

### Basis Risk
- Spot and perp prices can temporarily diverge beyond the funding rate
- Mark-to-market losses on one leg may exceed funding income short-term
- This typically mean-reverts but can cause margin pressure

### Opportunity Cost
- Capital is locked in collateral for both legs
- Compare net yield against other uses of capital (staking, lending, etc.)

### Smart Contract / Exchange Risk
- Hyperliquid is an L1 with on-chain order matching
- Funds are held on their chain -- standard DeFi custody risks apply
- No insurance fund guarantees in extreme market conditions

### Spot Hedge Availability
- Not all Hyperliquid perp assets have liquid spot markets
- For assets without spot, you need a second venue (CEX, another DEX)
- Cross-venue hedging adds withdrawal/deposit time and transfer risk

## Expected Returns

Assumptions: maker fees on both legs (0.015% per side), open + close.

| Annual Yield | Capital  | 30-day Net | 90-day Net | 365-day Net |
|-------------|----------|-----------|-----------|-------------|
| 10%         | $1,000   | $5        | $20       | $94         |
| 10%         | $10,000  | $54       | $206      | $940        |
| 10%         | $100,000 | $540      | $2,060    | $9,400      |
| 25%         | $1,000   | $18       | $57       | $244        |
| 25%         | $10,000  | $179      | $569      | $2,440      |
| 25%         | $100,000 | $1,790    | $5,690    | $24,400     |
| 50%         | $1,000   | $38       | $119      | $494        |
| 50%         | $10,000  | $380      | $1,194    | $4,940      |
| 50%         | $100,000 | $3,800    | $11,940   | $49,400     |

*Net returns account for maker fee drag. Actual results will vary as funding rates fluctuate.*

**Key insight**: Funding arb favors longer holding periods because the fixed entry/exit cost is amortized over more funding payments. Short-term carries (< 7 days) are rarely profitable after fees unless the rate is exceptionally high.

## Setup Instructions

### Requirements

- Python 3.10+
- `requests` library (included in project requirements.txt)

### Running

```bash
# Basic usage -- shows top 20 opportunities with top 5 detailed
python funding_arb.py

# Show more results
python funding_arb.py --top 30

# Only show opportunities above 15% annualized
python funding_arb.py --min-yield 15

# Only show assets with > $1M open interest
python funding_arb.py --min-oi 1

# Use 14 days of history instead of default 7
python funding_arb.py --history-days 14

# Combine filters
python funding_arb.py --top 10 --min-yield 20 --min-oi 5 --history-days 14
```

### Output

The tool displays:

1. **Overview table**: All assets ranked by absolute annualized funding yield, showing current rate, predicted next rate, open interest, and hedge direction.

2. **Detailed analysis** (top N): For each top opportunity:
   - Current and predicted funding rates
   - Net yield after fees at 30/90/180-day holding periods
   - Historical stats: average rate, volatility, % time positive/negative
   - Consistency score (higher = more stable funding)
   - Dollar return estimates at various capital levels

### No API Key Required

All endpoints used are public read-only info endpoints. No wallet, API key, or authentication is needed.

### Interpreting Results

- **High yield + high consistency + high % positive** = best opportunities
- **High yield + high volatility** = risky, rate may flip
- **Direction "LONG hedge"** means funding is positive (short perp, long spot)
- **Direction "SHORT hedge"** means funding is negative (long perp, short spot)
- Always cross-check that a liquid spot market exists for your hedge
