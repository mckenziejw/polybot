# BTC Up/Down 4h Market Maker Strategy

## Market Structure

Polymarket BTC Up/Down 4-hour binary markets. Each market has two tokens (Up/Down) that pay $1 if correct, $0 if wrong. Markets open every 4 hours, 6 per day.

Key characteristics (from backtesting 804 markets, Oct 2025 - Mar 2026):
- **Spreads**: Median 4c, stable throughout the market lifecycle
- **Tick volume**: ~1000 ticks per 15-min window, active the full 4 hours
- **Price behavior**: Highly mean-reverting, swings of 40-60c common. Leader at 0.90 only wins 51%
- **Market efficiency**: Both momentum and mean-reversion directional strategies are -EV. The ask price is a calibrated probability. The spread is where the money is.

## Strategy Overview

A dual-token market maker that:
1. **Mints** complete token pairs (Up + Down) for $1.00 via the CTF contract
2. **Posts asks** for both tokens at or near the current best ask
3. **Captures the combined spread**: ask_up + ask_down > $1.00 on 99.8% of ticks (mean $1.067)
4. **Manages inventory** via Avellaneda-Stoikov quote skewing
5. **Redeems** unsold pairs at market end for $1.00 (zero loss except gas)

## Why Mint-and-Sell

Traditional MM buys at the bid and sells at the ask, one token at a time. Mint-and-sell is superior because:

- **Guaranteed $0.50 cost basis** per token regardless of where mid is
- **Simultaneous dual-side inventory** — no waiting for bid fills to build position
- **Riskless exit**: unsold pairs redeem for $1.00. Worst case = gas cost (~$0.01 on Polygon)
- **Structural edge**: ask_up + ask_down averages $1.067, so every completed pair earns ~$0.067 theoretical

The edge is structural, not predictive. We don't need to know which direction BTC will move.

## Mechanics

### Entry: Minting

Call `redeemPositions` / CTF contract to mint pairs:
- Input: $1.00 USDC per pair
- Output: 1 Up token + 1 Down token
- Gas: ~$0.01 on Polygon
- Settlement: near-instant on Polygon

Mint in batches. Starting inventory: 10-50 pairs ($10-$50 capital).

### Quoting: Post Asks on Both Sides

For each token (Up and Down):
- Post a SELL limit order at or near the current best ask
- Order size: number of tokens we hold on that side

When a fill occurs (someone buys our token), we collect the ask price. When both sides of a pair have filled, that pair is "completed" with profit = ask_up + ask_down - $1.00.

### Inventory Management: Avellaneda-Stoikov Skewing

Core problem: fills are asymmetric. If Up is trending, our Up asks fill (good) but Down doesn't (bad). We accumulate Down inventory.

Solution: skew ask prices based on inventory imbalance.

```
inventory_imbalance = tokens_held_this_side - tokens_held_other_side
ask_adjustment = k * inventory_imbalance
our_ask = best_ask - ask_adjustment
```

Parameters:
- **k = 0.005 to 0.01** (0.5c to 1c per unit of imbalance)
- When we hold excess Down tokens: lower Down ask (more eager to sell), raise Up ask (less eager)
- This naturally drives inventory toward balance

Backtest results (k=0.005, dual-token):
- PnL: $25.88/market
- Win rate: 95%
- Max drawdown: -$2.88
- Avg max inventory: 13 units per side

### Replenishment: Continuous Minting

After selling tokens on one side, mint new pairs to replenish:
- If we sell 5 Up tokens: we now hold 5 excess Down tokens
- Mint 5 new pairs → now we have 5 Up + 10 Down
- The skewing mechanism will favor selling Down to rebalance
- Repeat throughout the market

### Exit: End-of-Market

At market close (or ~30 min before to reduce binary settlement risk):
1. Cancel all open orders
2. Count remaining inventory: N_up Up tokens, N_down Down tokens
3. Pairs redeemable: min(N_up, N_down) → redeem for $1.00 each (breakeven)
4. Excess single-side tokens: settle at binary outcome ($1 or $0)
   - With proper skewing, excess should be minimal (median 0 units in backtest)

## Sizing and Capital

### Per-Market Capital

| Inventory Size | Capital | Theoretical PnL/Mkt | Realized PnL/Mkt* | Daily (6 mkts) |
|---------------|---------|---------------------|-------------------|----------------|
| 10 pairs | $10 | $0.67 | $0.13 | $0.80 |
| 50 pairs | $50 | $3.35 | $0.67 | $4.00 |
| 200 pairs | $200 | $13.40 | $2.68 | $16.10 |
| 500 pairs | $500 | $33.50 | $6.70 | $40.20 |

*Realized assumes 20% of theoretical due to queue competition with other makers.

### Capital Efficiency

- Capital is recycled: redeemed pairs free USDC for the next market
- Max capital in market = inventory_size * $1.00
- No margin/leverage needed
- Multiple markets can overlap if capital allows

### Minimum Order Size

Polymarket enforces a minimum of **5 tokens** per order. At $0.50 cost basis, that's $2.50 per side minimum. Practical minimum: 10 pairs ($10).

## Risk Analysis

### Risks and Mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| One-sided fills (inventory accumulation) | Medium | A-S skewing keeps inventory balanced |
| Binary settlement of excess inventory | Medium | Skewing drives excess to ~0; early exit option |
| Queue competition (low fill rate) | High | Post at BBO, accept 20% fill rate as baseline |
| Spread compression over time | Medium | Monitor; wider spreads = more profit |
| Gas costs for minting/redeeming | Low | ~$0.01/tx on Polygon, negligible |
| Smart contract risk | Low | CTF is battle-tested (Gnosis ConditionalTokens) |
| API/connectivity failure | Medium | Graceful shutdown: cancel orders, redeem pairs |

### Worst Case

Max observed loss in backtest: -$13.51 on a single market (out of 476). This was an inventory blowout on a strongly trending market. With k=0.01 skewing, max drawdown drops to -$1.79.

### Sharpe Ratio

Annualized Sharpe ~16 at $200 deployment. The strategy is extremely consistent — 95% of markets are profitable.

## Implementation Notes

### Required Infrastructure (already built)

- WebSocket connection to Polymarket CLOB (market data + user channel)
- Order placement/cancellation via @polymarket/clob-client
- Position tracking via PositionStore
- Market rotation detection

### New Components Needed

1. **Minting module**: Call CTF `splitPosition()` to mint pairs, `mergePositions()` to redeem
   - Uses viem, same pattern as nonce-cancel.ts
   - CTF address: `0x4D97DCd97eC945f40cF65F87097ACe5EA0476045`

2. **Market maker strategy**: New Strategy implementation
   - Tracks inventory per side
   - Computes skewed asks via A-S model
   - Manages quote lifecycle (post, cancel, re-quote on book changes)
   - Triggers replenishment minting when inventory depleted

3. **Market rotation for 4h markets**: Adjust timing from 5-min to 4-hour windows
   - Markets open every 4 hours (6/day)
   - Slug format: `btc-updown-4h-{unix_timestamp}`

### Quote Update Frequency

The backtest uses tick-level updates (~1000/15min = ~1/second). In practice:
- Re-quote when the book changes (new BBO)
- Re-quote when inventory changes (fill received)
- Periodic re-quote every 5-10 seconds as a safety net

### Fee Structure

- **Maker**: 20% rebate on collected taker fees (we receive ~0.1-0.2% rebate per fill)
- **Taker**: ~1.5% at mid=0.50. We never take; always post limits
- Net effect: fees are slightly positive for us (maker rebate)

## Validation Plan

1. **Paper trading**: Run strategy with real market data but simulated fills. Log all quote placements, fills, inventory. Compare to backtest expectations.
2. **Minimum size live test**: 10 pairs ($10 capital). Run for 1-2 days (6-12 markets). Measure actual fill rate vs backtest.
3. **Scale up**: If fill rate > 10% of theoretical, increase to 50-200 pairs.
4. **Key metric**: Actual fill rate. If <10% of theoretical, strategy is not viable at this venue.
