# RL Market Maker for Polymarket BTC Up/Down Binary Markets

## Design Document

Date: 2026-03-11

---

## 1. Problem Statement

We need an RL agent that **makes markets** (provides liquidity) on Polymarket BTC Up/Down binary prediction markets. Previous attempts at directional prediction (XGBoost v3.3, momentum, mid-price drift) all failed live despite good backtests. The core lesson: **taking liquidity destroys edge; making liquidity is the only viable path**.

However, a naive Avellaneda-Stoikov market maker with fixed parameters got whipsawed on trending markets. The research (Agent 4, overnight summary) showed that **adverse selection at any fixed offset is uniformly negative** (-$0.017 to -$0.023 per token). The existing `MintAndSellMaker` handles this somewhat via inventory skewing and whale signal detection, but the parameters (k, thresholds, cooldowns) are static and hand-tuned.

**The thesis**: An RL agent can learn to dynamically adjust quoting behavior — widening spreads in danger, tightening in safety, pulling quotes before adverse moves, and managing inventory more aggressively than any fixed-parameter scheme. The whale signal (2 dominant MMs telegraphing BTC moves 2-3s in advance) gives a learnable information advantage that a static strategy cannot fully exploit.

---

## 2. What Makes This Different From Standard RL Market Making

This is **not** a standard Avellaneda-Stoikov / Guéant-Lehalle-Fernandez-Tapia problem. The differences are fundamental and drive every design choice:

| Standard MM Literature | Polymarket BTC Up/Down |
|---|---|
| Continuous price (GBM, mean-reverting) | Binary outcome: token pays $1 or $0 |
| Infinite horizon or arbitrary | Fixed duration: 4 hours (or 15 min) |
| Unknown terminal value | Known resolution time, binary settlement |
| Many participants, anonymous flow | 2 dominant MMs control 95% of book |
| Independent bid/ask management | Two correlated assets: Up + Down = $1.00 |
| Buy low, sell high on same asset | Mint pairs ($1 -> 1 Up + 1 Down), sell both sides |
| Inventory risk is price risk | Inventory risk is binary: unpaired tokens go to 0 or 1 |
| Transaction costs are spread + fees | Maker rebate (positive), taker fee (negative) |
| Deep, liquid orderbooks | Thin books, 5 levels, min 5 token orders |

### Key implications:

1. **The minting mechanic is unique**. The agent gets both tokens at $0.50 each, guaranteed. The "spread" is not bid-ask on one token — it's `ask_up + ask_down - $1.00`. This averages $0.067 per pair and is positive 99.8% of the time. No traditional MM paper accounts for this.

2. **Terminal value is binary, not continuous**. At resolution, unpaired inventory is worth $1 (if you're on the winning side) or $0 (losing side). This creates a unique risk structure: small inventory imbalances are catastrophic if you're wrong, irrelevant if you're right. The reward function must account for this.

3. **Two dominant MMs are identifiable**. Their order sizes (~4800 and ~2500 tokens) are consistent enough to track. Their order pulls/repositioning telegraphs BTC moves 2-3 seconds early. This is an **information advantage** the RL agent can learn to exploit — no academic MM paper has this.

4. **Time is a first-class feature**. With known resolution time, the agent should quote more aggressively early (time to recover from adverse fills) and defensively late (binary settlement risk dominates). This is closer to options market making near expiry than equity MM.

---

## 3. Observation Space

### 3.1 Design Principles

- **Include everything the existing `MintAndSellMaker` uses**, plus everything it should use but doesn't.
- **Normalize aggressively** — all features should be roughly [-1, 1] or [0, 1] for stable training.
- **Include whale features explicitly** — the agent needs to see what the dominant MMs are doing.
- **Keep dimensionality manageable** — the LSTM handles temporal patterns, so we don't need to stuff lagged features into a single observation.

### 3.2 Feature Groups

Total observation dimension: **~95 features** (up from 63 in the directional env).

#### Group 1: Orderbook State (per token, x2) — 30 features

For each token (Up, Down):
```
mid_price                    # midpoint, raw [0, 1]
spread                       # ask_1 - bid_1, raw [0, 0.10]
book_imbalance               # (bid_size - ask_size) / (bid_size + ask_size), [-1, 1]
bid_price_1..5               # 5 levels, raw [0, 1]
bid_size_1..5 / 1000         # 5 levels, normalized
ask_price_1..5               # 5 levels, raw [0, 1]
ask_size_1..5 / 1000         # 5 levels, normalized
```

Per token: 3 + 5*4 = 23 features. Two tokens: **46 features**.

Note: keeping raw prices (not delta-from-mid) because binary market prices carry absolute meaning (probability). A mid of 0.50 is qualitatively different from 0.90.

#### Group 2: Combined Spread Features — 4 features

```
combined_ask_spread          # ask_up + ask_down - 1.00 (the structural edge, ~0.067 mean)
combined_bid_spread          # 1.00 - bid_up - bid_down (complement)
up_down_mid_sum              # mid_up + mid_down (should be ~1.00; deviation = arb signal)
pair_completion_rate         # min(sell_fills_up, sell_fills_down) / pairs_minted
```

These are unique to binary markets and not in any standard MM observation space. The `combined_ask_spread` is the edge we're capturing; the agent should learn to only quote when it's positive enough.

#### Group 3: Inventory State — 8 features

```
inventory_up / max_inventory          # [0, 1]
inventory_down / max_inventory        # [0, 1]
inventory_imbalance / max_inventory   # [-1, 1] (up - down)
abs_imbalance / max_unhedged          # [0, 1+] (how close to risk limit)
pairs_minted / max_inventory          # [0, 1] (total minted)
sell_fills_up / max_inventory         # [0, 1] (cumulative fills)
sell_fills_down / max_inventory       # [0, 1]
pending_mint                          # {0, 1} binary flag
```

Critical for inventory management. The imbalance features give the agent the same information the A-S skew formula uses, but let it learn a nonlinear policy.

#### Group 4: Our Order State — 8 features

```
our_ask_up_price                      # current posted ask, [0, 1] or 0 if none
our_ask_up_size / max_inventory       # [0, 1]
our_ask_down_price                    # [0, 1] or 0
our_ask_down_size / max_inventory     # [0, 1]
our_bid_up_price                      # (traditional mode)
our_bid_up_size / max_inventory
our_bid_down_price
our_bid_down_size / max_inventory
```

The agent must know where its own orders are to decide whether to requote. Without this, it can't learn queue priority management.

#### Group 5: Whale Features — 10 features

```
whale_ask_up_price                    # whale resting ask level, [0, 1] or 0 if absent
whale_ask_up_size / 10000             # normalized by whale scale
whale_ask_down_price
whale_ask_down_size / 10000
whale_bid_up_price
whale_bid_up_size / 10000
whale_bid_down_price
whale_bid_down_size / 10000
whale_signal_active                   # {0, 1} — is a whale signal currently firing?
whale_cooldown_remaining / 3000       # [0, 1] — how much cooldown is left
```

This is the agent's primary information advantage. The LSTM will learn temporal patterns in whale behavior (e.g., "whale pulled their ask 500ms ago, price move imminent").

#### Group 6: BTC Features — 8 features

```
btc_mid_norm                          # BTC price / 100000
btc_log_return                        # 1s return
btc_ret_5s                            # 5-second momentum
btc_ret_15s                           # 15-second momentum
btc_ret_60s                           # 1-minute momentum
btc_rvol_30s                          # 30s realized vol
btc_rvol_60s                          # 1-minute realized vol
btc_rvol_300s                         # 5-minute realized vol
```

Same as existing env. BTC vol is the #4 feature in XGBoost — the agent should widen spreads in high-vol regimes.

#### Group 7: Time & PnL — 5 features

```
time_remaining_pct                    # [1.0, 0.0] (fraction of market remaining)
time_remaining_minutes                # raw minutes, for the agent to learn absolute thresholds
realized_pnl / max_inventory          # normalized running PnL
unrealized_pnl / max_inventory        # mark-to-market on current inventory
combined_pnl / max_inventory          # realized + unrealized
```

#### Group 8: Staleness — 2 features

```
yes_staleness_norm                    # ms since last tick / 5000, clipped [0, 1]
no_staleness_norm
```

Total: 46 + 4 + 8 + 8 + 10 + 8 + 5 + 2 = **91 features**.

### 3.3 What NOT to Include

- **Lagged features** (ret_5s, ret_15s already capture this; the LSTM handles temporal patterns)
- **Cross-market features** (research shows markets are independent)
- **Orderbook entropy** (research showed this is noise)
- **Explicit fee calculations** (fees are tiny for makers; the agent will internalize this from rewards)

---

## 4. Action Space

### 4.1 Design Choice: Parameterized Discrete

The action space should be **discrete but parameterized**, not continuous. Rationale:
- PPO with discrete actions is more sample-efficient and stable than continuous.
- The actual order book has discrete tick sizes ($0.01).
- The agent's decisions are naturally categorical: where to quote relative to the BBO, whether to quote at all, whether to mint.

### 4.2 Action Definition

**10 discrete actions:**

```
Action 0:  HOLD           — change nothing (keep existing orders as-is)
Action 1:  QUOTE_TIGHT    — post asks 1 tick inside whale/BBO (aggressive, high fill rate)
Action 2:  QUOTE_AT_BBO   — post asks at current best ask (moderate)
Action 3:  QUOTE_WIDE     — post asks 2-3 ticks outside BBO (defensive, low fill rate)
Action 4:  QUOTE_SKEWED   — apply A-S skew to push the long side (inventory rebalance)
Action 5:  PULL_QUOTES    — cancel all orders (danger signal, preserve inventory)
Action 6:  PULL_LAGGING   — cancel only the side with lower fills (protect against trend)
Action 7:  MINT_SMALL     — mint 10 additional pairs (replenish inventory)
Action 8:  MINT_LARGE     — mint 50 additional pairs (aggressive replenishment)
Action 9:  EMERGENCY_EXIT — market-sell excess inventory on the lagging side at best bid
```

### 4.3 Why Not Continuous?

A continuous action space (e.g., output raw bid/ask prices) has these problems:
1. The CLOB only accepts prices on a tick grid ($0.01). Rounding introduces a non-differentiable step.
2. The price differences that matter are 1-3 ticks. Continuous output needs to be extremely precise.
3. PPO struggles with high-dimensional continuous actions in low-data regimes.
4. The discrete actions map cleanly to the strategy decisions a human MM would make.

### 4.4 Action Masking

Essential for safety. Mask invalid actions each step:

```python
def action_masks(state):
    masks = [True] * 10  # all valid by default

    # Can't HOLD if no orders posted and have inventory to sell
    # (actually, HOLD should always be valid — sometimes doing nothing is correct)

    # Can't MINT if mint is pending or at max inventory
    if state.pending_mint or min(inv_up, inv_down) + 50 > max_inventory:
        masks[7] = False  # MINT_SMALL
    if state.pending_mint or min(inv_up, inv_down) + 50 > max_inventory:
        masks[8] = False  # MINT_LARGE

    # Can't EMERGENCY_EXIT if no excess inventory
    if abs(inv_up - inv_down) < 5:
        masks[9] = False

    # Can't QUOTE_TIGHT/BBO/WIDE if no inventory to sell on either side
    if inv_up < 5 and inv_down < 5:
        masks[1] = masks[2] = masks[3] = masks[4] = False

    # Can't PULL_QUOTES if no orders outstanding
    if len(open_orders) == 0:
        masks[5] = masks[6] = False

    # Force PULL_QUOTES in final 60 seconds
    if time_remaining_s < 60:
        masks = [False] * 10
        masks[5] = True  # only PULL_QUOTES allowed
        if len(open_orders) == 0:
            masks[0] = True  # HOLD if already pulled

    return masks
```

### 4.5 Action-to-Order Translation

Each action translates to concrete CLOB operations in the execution engine. The translation layer handles:
- Converting relative price targets (e.g., "1 tick inside BBO") to absolute prices.
- Respecting the 5-token minimum order size.
- Splitting inventory between Up and Down sides.
- Clamping prices to valid ranges.

This separation is critical: the RL agent decides **strategy** (tight/wide/pull), the execution engine handles **mechanics** (order placement, cancellation, CLOB API calls).

---

## 5. Reward Function

This is the hardest and most important design decision. The reward must:
1. Credit spread capture (the source of profit).
2. Penalize inventory imbalance (the source of risk).
3. Handle the binary terminal settlement correctly.
4. Provide enough signal per step for learning (not too sparse).

### 5.1 Recommended: Composite Reward

```python
def compute_reward(state, prev_state, is_terminal):
    # Component 1: Mark-to-market PnL delta (continuous signal)
    # Uses mid-prices for unrealized, actual fill prices for realized.
    mtm_delta = state.portfolio_value - prev_state.portfolio_value

    # Component 2: Spread capture bonus (rewards completing pairs)
    # When both sides of a minted pair have been sold, the profit is
    # ask_up + ask_down - $1.00. This is the core edge.
    new_completed_pairs = min(state.sell_fills_up, state.sell_fills_down) - \
                          min(prev_state.sell_fills_up, prev_state.sell_fills_down)
    spread_bonus = new_completed_pairs * 0.05  # ~$0.05 per completed pair

    # Component 3: Inventory imbalance penalty (continuous, quadratic)
    # Penalize holding unhedged inventory. Quadratic because risk grows
    # nonlinearly — 10 excess tokens is much worse than 2x 5 excess.
    imbalance = abs(state.inv_up - state.inv_down)
    imbalance_penalty = -0.001 * imbalance ** 2

    # Component 4: Time-decay inventory urgency
    # As time runs out, unhedged inventory becomes more dangerous.
    # Scale the imbalance penalty by inverse time remaining.
    time_factor = 1.0 + 2.0 * (1.0 - state.time_remaining_pct)
    imbalance_penalty *= time_factor

    if is_terminal:
        # Component 5: Terminal settlement (dominant signal)
        # Settle all inventory at binary resolution values.
        # Completed pairs are already accounted for in realized PnL.
        # Excess single-side inventory settles at 0 or 1.
        terminal_pnl = compute_terminal_settlement(state)
        return terminal_pnl  # unscaled — this is the true objective

    # Step reward: weighted combination
    return mtm_delta + spread_bonus + imbalance_penalty
```

### 5.2 Why This Structure

**Mark-to-market delta** (mtm_delta): Provides a continuous learning signal every step. Without this, the reward is zero for hundreds of steps until a fill occurs, making credit assignment impossible. This is what the existing `polymarket_env.py` already uses.

**Spread capture bonus**: Explicitly rewards the intended behavior — completing pairs. Without this, the agent might learn to accumulate inventory on one side (which looks good mark-to-market if the price moves favorably) rather than maintaining balanced flow.

**Quadratic imbalance penalty**: Linear penalties don't bite hard enough. Going from 5 to 10 excess tokens should be much more painful than 0 to 5, because the binary settlement risk grows combinatorially. The A-S model uses a similar quadratic inventory penalty.

**Time-decay urgency**: In the first hour of a 4-hour market, holding 20 excess Down tokens is manageable — there's time for fills to rebalance. In the last 30 minutes, those same 20 tokens represent a $20 binary bet. The penalty should reflect this.

**Terminal settlement**: Must dominate. The entire point of the strategy is that paired inventory is riskless and excess is catastrophic. The terminal reward is the ground truth.

### 5.3 What NOT to Use

- **Raw PnL only**: Too sparse. The agent won't learn anything for thousands of steps.
- **Sharpe ratio**: Requires episode-level computation, can't be used per-step.
- **Fill count reward**: Rewards filling for filling's sake, regardless of profitability. The agent would learn to undercut the BBO and take adverse fills.
- **Unrealized PnL alone**: Doesn't account for the cost basis ($0.50 per minted token). The agent could think it's profitable when it's just holding tokens that haven't moved.

### 5.4 Reward Normalization

Use the VecNormalize running reward normalization from the existing `train_cleanrl.py`. The composite reward has mixed scales (mtm_delta in dollars, imbalance in penalty units), so variance-based normalization is essential.

---

## 6. Architecture

### 6.1 Why PPO + LSTM

**PPO** (already implemented in `train_cleanrl.py`):
- Stable with discrete actions and action masking.
- Sample-efficient enough for the data regime (~800 4h markets, ~7000 15m markets).
- On-policy: avoids the deadly triad issues of off-policy methods in this non-stationary environment.
- Clipping prevents catastrophic policy updates, important when reward variance is high (binary terminal settlements).

**LSTM** (already implemented):
- Temporal patterns in the orderbook are the primary signal. The whale MM repositions over 2-3 second windows — the LSTM can learn this temporal signature.
- Episode boundaries (market open/close) naturally reset the hidden state — each market is independent, which the LSTM handles correctly.
- Order flow momentum is sequential: a burst of sells followed by whale bid pulls is a pattern that a memoryless policy cannot capture.

### 6.2 Network Architecture

Extend the existing `Agent` class from `train_cleanrl.py`:

```
Observation (91 dims)
    |
    v
[Feature Extractor: Linear(91, 256) -> LayerNorm -> ReLU -> Linear(256, 256) -> LayerNorm -> ReLU]
    |
    v
[LSTM(256, 256)] <-- hidden state carries across steps within episode
    |
    v
[Actor Head: Linear(256, 10)]  -- 10 discrete actions, masked
[Critic Head: Linear(256, 1)]  -- state value estimate
```

Changes from the existing Agent:
1. **LayerNorm instead of Tanh**: Tanh activations saturate and kill gradients. LayerNorm provides normalization without saturation, better for the wider observation space.
2. **Same LSTM size (256)**: The temporal patterns are relatively simple (2-3 second whale windows). No need for larger memory.
3. **10 actions instead of 6**: Matching the new action space.

### 6.3 Episode Structure

**One market = one episode.** This is already the pattern in `polymarket_env.py` and is correct for this problem. Each market is independent (confirmed by research).

For 4-hour markets at 100ms steps: **144,000 steps per episode**. This is very long for RL. Mitigations:
- Use 1-second steps for training (not 100ms). This gives **14,400 steps per episode**, still long but manageable.
- The LSTM hidden state resets at each episode boundary.
- GAE lambda of 0.95 handles long-horizon credit assignment.

For 15-minute markets at 100ms steps: **9,000 steps per episode** (or 900 at 1s). Much more tractable.

**Recommendation**: Start with 15-minute markets (more data, shorter episodes, faster iteration). Transfer to 4-hour markets once the 15-minute agent is working.

### 6.4 Training Data

Available historical data:
- **15m orderbook (normalized)**: `data/telonex_book_snapshots/` — 7,187 files with token labels
- **4h orderbook**: `data/telonex_btc_4h_book_snapshots/` — 798 files
- **BTC quotes**: `data/btc_quotes/btcusdt_quotes.parquet`

Estimated training budget:
- 7,187 15m markets x 900 steps = ~6.5M steps
- 8 parallel envs x 2048 steps/rollout = 16,384 steps/iteration
- ~400 iterations to see every market once
- Target: 10-20M total timesteps (2-3 passes through data)

---

## 7. Simulator Design

### 7.1 The Sim-to-Real Gap

This is the **single biggest risk** in the entire project. Every backtest and simulation showed good results that failed live. The gap comes from:

1. **Fill model**: Historical data shows the orderbook state but not who would have filled our orders. We must simulate fills, and the simulation will be wrong.
2. **Market impact**: Our orders change the orderbook. Historical data doesn't reflect this.
3. **Queue position**: We can't know our position in the queue from historical data. Fill probability depends heavily on this.
4. **Whale reaction**: The dominant MMs might change behavior in response to a new participant.

### 7.2 Fill Model: Conservative Priority Queue

The fill model is the most important simulator component. Design it to be **pessimistic** — if the agent profits in a pessimistic sim, it's more likely to profit live.

```python
class ConservativeFillModel:
    """
    Simulates order fills with pessimistic assumptions.

    Rules:
    1. Our orders ALWAYS go to the back of the queue at their price level.
    2. A fill occurs only when the opposite side crosses our price AND
       enough volume has traded through our level to reach our queue position.
    3. Adverse selection: after filling, the mid-price moves against us
       by adverse_selection_bps (calibrated from historical data: ~1.7 cents).
    4. Partial fills are possible (proportional to volume at our level).
    """

    def __init__(self, adverse_selection_cents=0.017, queue_position_pct=0.9):
        self.adverse_selection = adverse_selection_cents
        self.queue_pct = queue_position_pct  # we're in the back 90% of queue

    def check_fill(self, our_price, our_size, our_side, book_snapshot, trade_flow):
        # For a SELL order: filled when a BUY at our price or higher arrives
        # For a BUY order: filled when a SELL at our price or lower arrives

        if our_side == "SELL":
            # Look at buy-side aggression: did any trades execute at >= our price?
            fills_at_level = sum(t.size for t in trade_flow
                                if t.price >= our_price and t.side == "BUY")
            # Queue depth: how many tokens were ahead of us?
            queue_ahead = sum(l.size for l in book_snapshot.asks
                            if l.price <= our_price) * self.queue_pct
            fillable = max(0, fills_at_level - queue_ahead)
            filled = min(our_size, fillable)
            return filled

        # Similar for BUY side...
```

### 7.3 Key Calibration Parameters

These should be measured from historical data before training:

| Parameter | Source | Expected Value |
|---|---|---|
| Adverse selection (cents) | Agent 4 research | 1.7-2.3 cents |
| Queue position discount | Agent 4: 4.7% at 2 ticks out | 0.05-0.15 at BBO |
| Fill probability at BBO | Agent 4: 68% touch in 10s | 0.68 per 10s window |
| Whale reaction time (ms) | Whale signal research | 2000-3000 ms |
| Spread mean/std | Market maker doc | 4c median |
| Mint latency (ms) | Live testing | 2000-10000 ms (Polygon) |

### 7.4 Domain Randomization

To reduce sim-to-real gap, randomize simulator parameters during training:

```python
# Each episode, sample parameters from a distribution
adverse_selection = np.random.uniform(0.010, 0.030)  # 1-3 cents
queue_discount    = np.random.uniform(0.70, 0.95)     # 70-95% back of queue
fill_delay_ms     = np.random.uniform(100, 2000)       # 0.1-2s fill latency
mint_delay_ms     = np.random.uniform(2000, 15000)     # 2-15s mint confirmation
```

This forces the agent to learn policies that are robust across a range of execution conditions, rather than overfitting to a single fill model. This is a standard technique from robotics RL (sim-to-real transfer).

---

## 8. Environment Implementation

### 8.1 New Environment: `MarketMakerEnv`

This should be a new Gymnasium environment, separate from the existing `PolymarketEnv` (which is designed for directional taker strategies). Key differences:

| | `PolymarketEnv` (existing) | `MarketMakerEnv` (new) |
|---|---|---|
| Role | Taker (buys/sells tokens) | Maker (posts limit orders, mints pairs) |
| Fill model | Mid-price, instant | Queue-based, delayed, adverse selection |
| Inventory | Single-sided (Yes OR No) | Dual-sided (Up AND Down simultaneously) |
| Actions | Buy/Sell/Hold | Quote levels, mint, pull quotes |
| Episode length | 900-9000 steps | 900-14400 steps |
| Observation | 63 dims | 91 dims |
| Terminal | Binary settlement | Binary settlement + pair redemption |

### 8.2 Step Function Pseudocode

```python
def step(self, action):
    # 1. Translate action to order operations
    orders_to_place, orders_to_cancel = self.action_to_orders(action)

    # 2. Execute cancellations immediately
    for order in orders_to_cancel:
        self.cancel_order(order)

    # 3. Place new orders (go to back of queue)
    for order in orders_to_place:
        self.place_order(order)

    # 4. Check for pending mint completion
    if self.pending_mint and self.steps_since_mint >= self.mint_delay:
        self.complete_mint()

    # 5. Advance time by one step
    self.step_idx += 1

    # 6. Check fills against new book state
    new_book = self.get_book_snapshot()
    trade_flow = self.get_trade_flow()  # trades that occurred this step
    for order in self.open_orders:
        filled = self.fill_model.check_fill(order, new_book, trade_flow)
        if filled > 0:
            self.process_fill(order, filled)

    # 7. Compute reward
    if self.is_terminal():
        reward = self.terminal_settlement()
    else:
        reward = self.step_reward()

    # 8. Build observation
    obs = self.get_obs()

    return obs, reward, self.is_terminal(), False, self.get_info()
```

### 8.3 Data Pipeline

The environment needs to load and align multiple data sources per episode:
1. **Orderbook snapshots** (Up and Down tokens, resampled to 1s)
2. **BTC quotes** (1s bars with vol features)
3. **Trade events** (from `data/trade_events/`) — needed for fill model calibration
4. **Resolution outcomes** (from ResolutionStore cache)

The existing `_load_market()` and `_resample_book()` in `polymarket_env.py` handle (1), (2), and (4). The trade events need to be loaded additionally.

---

## 9. Training Pipeline

### 9.1 Phase 1: 15-Minute Markets (Bootstrap)

- Use 7,187 available 15-minute markets.
- 1-second steps: 900 steps/episode.
- 8 parallel environments.
- Target: 10M steps (2 passes through data).
- Focus: learn basic quoting behavior (when to be tight vs wide vs pull).

Hyperparameters (starting point, from existing `train_cleanrl.py`):
```
learning_rate    = 3e-4 (anneal to 0)
gamma            = 0.99
gae_lambda       = 0.95
clip_coef        = 0.2
ent_coef         = 0.05 (high — encourage exploration early)
vf_coef          = 0.5
max_grad_norm    = 0.5
num_steps        = 2048
update_epochs    = 10
num_minibatches  = 8
lstm_hidden      = 256
```

### 9.2 Phase 2: 4-Hour Markets (Fine-tune)

- 798 available 4-hour markets.
- 1-second steps: 14,400 steps/episode.
- Reduce to 4 parallel environments (memory).
- Initialize from Phase 1 weights (transfer learning).
- Target: 5M additional steps.
- Focus: learn longer-horizon inventory management, time-decay behavior.

Key adjustment: reduce `num_steps` to 1024 (otherwise each rollout is a single 14,400-step episode, which is wasteful for PPO).

### 9.3 Phase 3: Self-Play Refinement (Optional, Advanced)

Once the base agent works, train against itself:
1. Deploy trained agent as one of the MMs in the sim.
2. Train a new agent in the same environment.
3. The new agent learns to exploit the trained agent's patterns.
4. Iterate (NFSP-style).

This addresses the whale reaction problem: the sim whales don't react to us, but real ones will. Self-play forces the agent to learn robust strategies against adaptive opponents.

### 9.4 Curriculum Learning

Start with easy markets (wide spreads, balanced flow) and progressively add harder ones:

**Stage 1**: Markets with combined spread > $0.08 (profitable even with poor execution)
**Stage 2**: Markets with combined spread $0.04-$0.08 (typical)
**Stage 3**: Markets with combined spread < $0.04 (tight, competitive)
**Stage 4**: Markets with strong directional trends (inventory management challenge)

Implement via the `market_slugs` parameter in the environment.

---

## 10. Evaluation Metrics

### 10.1 Per-Episode Metrics (logged to TensorBoard)

```
episode_pnl                    # total realized + unrealized PnL ($)
episode_pnl_per_pair_minted    # PnL / pairs minted (edge per unit capital)
pairs_completed                # min(sell_fills_up, sell_fills_down)
completion_rate                # pairs_completed / pairs_minted
max_imbalance                  # max |inv_up - inv_down| during episode
time_at_max_imbalance          # when the worst imbalance occurred
orphaned_tokens                # tokens settling at binary outcome
avg_spread_captured            # mean(ask_up + ask_down) for completed pairs - 1.00
fill_rate                      # fills / quotes posted
adverse_selection_cost         # avg mid-price move after our fills
time_to_first_fill             # how quickly the agent starts generating flow
quotes_pulled                  # number of PULL_QUOTES actions (defensive behavior)
```

### 10.2 Aggregate Metrics (across evaluation episodes)

```
mean_episode_pnl               # should be positive
win_rate                       # % of episodes with positive PnL
max_drawdown                   # worst single episode
sharpe_ratio                   # mean(pnl) / std(pnl) * sqrt(6) for 4h (6/day)
inventory_efficiency           # 1 - mean(orphaned_tokens) / mean(pairs_minted)
```

### 10.3 Baseline Comparisons

The RL agent should outperform these baselines:

1. **Static A-S (k=0.005)**: The current `MintAndSellMaker` with fixed parameters.
2. **Static A-S (k=0.01)**: Higher skew, more defensive.
3. **Whale-front static**: The current whale-front pricing mode.
4. **Random quoting**: Random actions (sanity check — RL should beat this easily).
5. **Hold**: Mint pairs, never sell, redeem at end (PnL = 0 minus gas).

---

## 11. Deployment Path

### 11.1 Paper Trading

Before any real money:
1. Run the trained agent against the live WebSocket feed.
2. Simulate fills using the same conservative fill model.
3. Compare agent's decisions to the static `MintAndSellMaker` running in parallel.
4. Log every decision, fill, and state transition for analysis.
5. Run for at least 48 hours (12 4-hour markets or 288 15-minute markets).

### 11.2 Integration with Execution Engine

The RL agent should be a new `Strategy` implementation in the TypeScript execution engine, following the existing `Strategy` interface. The architecture:

```
Python inference server (FastAPI/Flask)
    - Loads trained model (.pt)
    - Receives state observations
    - Returns action + confidence

TypeScript execution engine
    - RLStrategy implements Strategy interface
    - Formats MarketState -> observation vector
    - Sends observation to Python inference server
    - Translates action -> TradeAction[]
    - Handles fills, mints, cancellations
```

This is the same pattern as the existing `modelServerUrl` in the config — the execution engine already supports calling an external model server.

### 11.3 Live Safeguards

Hard-coded in the execution engine, not the RL policy (the agent cannot override these):

```typescript
// Maximum capital at risk per market
const MAX_CAPITAL_PER_MARKET = 200;  // $200

// Maximum inventory imbalance (absolute, in tokens)
const HARD_MAX_IMBALANCE = 100;

// Kill switch: if unrealized loss exceeds this, cancel all and redeem
const KILL_SWITCH_LOSS = -50;  // -$50

// Stop quoting this many seconds before market close
const STOP_QUOTING_BUFFER_S = 120;  // 2 minutes

// Maximum position across ALL concurrent markets
const MAX_TOTAL_POSITION = 500;

// Rate limit: max order operations per second
const MAX_OPS_PER_SECOND = 5;
```

### 11.4 Gradual Scale-Up

1. **$10 capital** (10 pairs): Validate fills happen at all. Measure actual fill rate vs sim.
2. **$50 capital** (50 pairs): Validate inventory management. Measure actual adverse selection.
3. **$200 capital** (200 pairs): Full scale. Compare to backtest expectations.

At each stage, run for at least 48 hours before scaling up. The key metric is **actual fill rate / simulated fill rate** — if this ratio is below 0.3, the sim is too optimistic and needs recalibration before proceeding.

---

## 12. Literature Reference

### 12.1 Core Papers

**Avellaneda & Stoikov (2008)**: "High-frequency trading in a limit order book." The foundation. Derives optimal bid/ask placement as a function of inventory, volatility, and risk aversion. The key result: reservation price = mid - q * gamma * sigma^2 * T, where q is inventory, gamma is risk aversion, sigma is volatility, T is time remaining. Our environment operationalizes all these variables — the RL agent should learn something similar but nonlinear.

**Gueant, Lehalle & Fernandez-Tapia (2012)**: "Dealing with the Inventory Risk." Extends A-S to realistic orderbook dynamics. Shows optimal spread is wider when inventory is high, which is exactly the behavior we want the RL agent to discover. Their closed-form solution assumes constant volatility and Poisson arrivals — our RL agent can handle the actual non-stationary dynamics.

**Spooner et al. (2018)**: "Market Making via Reinforcement Learning." Applied RL to equity market making in a realistic LOB simulator. Used DQN (we use PPO, which is better for this setting). Key finding: RL agent learned to widen spreads in volatile periods and tighten in calm periods — exactly the behavior we want. They found that reward shaping with inventory penalty was critical for learning, confirming our composite reward design.

**Gueant & Lehalle (2017)**: "General Intensity Shapes in Optimal Liquidation." Relevant for the end-of-market inventory liquidation problem. As time-to-close shrinks, the optimal strategy shifts from passive (limit orders) to aggressive (market orders). Our action space includes EMERGENCY_EXIT for this reason.

### 12.2 Prediction Market Specific

**Hanson (2003, 2007)**: "Combinatorial Information Market Design" and the LMSR (Logarithmic Market Scoring Rule). While Polymarket uses a CLOB (not LMSR), Hanson's work on prediction market microstructure informs how prices aggregate information. Key insight: in a binary market, the sum p(Yes) + p(No) should equal 1, and deviations represent arbitrage or inefficiency.

**Othman et al. (2013)**: "A Practical Liquidity-Sensitive Automated Market Maker." Extends LMSR with liquidity sensitivity. While not directly applicable to CLOB-based Polymarket, the insight that market makers in prediction markets should adjust their exposure based on total market liquidity is relevant to our inventory management.

**Chen & Pennock (2010)**: "Designing Markets for Prediction." Overview of prediction market mechanism design. Relevant insight: binary markets have a natural "conservation" constraint (Up + Down = $1) that creates correlations between the two tokens. Our minting mechanic exploits this directly.

### 12.3 RL-Specific

**Schulman et al. (2017)**: "Proximal Policy Optimization Algorithms." The PPO paper. Our implementation follows CleanRL-style PPO with LSTM, which is well-suited for partially observable environments (we only see 5 levels of the book, not the full queue depth).

**Huang et al. (2022)**: "CleanRL: High-quality Single-file Implementations of Deep Reinforcement Learning Algorithms." The implementation pattern we follow. Their LSTM-PPO implementation handles episode boundaries correctly, which is critical for our market-episodic structure.

### 12.4 Key Departures from Literature

1. **No Poisson arrival assumption**: A-S and GLFT assume order arrivals are Poisson. Polymarket has 2 dominant MMs whose behavior is decidedly non-Poisson. The RL agent can learn the actual arrival dynamics.

2. **No constant volatility assumption**: BTC vol varies by 3x between regimes (research finding). The agent sees vol features and can adapt.

3. **Minting has no analogue in the literature**. No academic paper addresses a market maker who can create both sides of a binary contract simultaneously. This is a structural advantage unique to prediction markets with CTF-style minting.

4. **The whale signal is a unique information advantage**. The 2 dominant MMs telegraph moves 2-3s early. No academic paper addresses market making with identifiable, predictable counterparties. This is closer to poker (reading opponents) than to anonymous-flow market making.

---

## 13. Implementation Priority

### Phase 0: Fill Model Calibration (1-2 days)
- Analyze `data/trade_events/` to calibrate adverse selection, fill probability, queue depth.
- Build and validate the `ConservativeFillModel` class.
- Compare simulated fill rates against known backtest results.

### Phase 1: Environment (2-3 days)
- Implement `MarketMakerEnv` extending Gymnasium.
- 91-dimensional observation space.
- 10-action discrete space with masking.
- Composite reward function.
- Conservative fill model.
- Validate on a handful of markets manually.

### Phase 2: Training (1-2 days)
- Adapt `train_cleanrl.py` for the new environment.
- Train on 15-minute markets first.
- TensorBoard monitoring of all metrics.
- Checkpoint every 100K steps.

### Phase 3: Evaluation (1-2 days)
- Run trained agent against held-out markets.
- Compare to static baselines (A-S k=0.005, k=0.01, whale-front).
- Analyze behavior: when does it pull quotes? How does it manage inventory?

### Phase 4: Integration (2-3 days)
- Python inference server for trained model.
- TypeScript `RLStrategy` implementing the `Strategy` interface.
- Paper trading against live feed.
- Safeguard implementation.

### Phase 5: Live Deployment (ongoing)
- $10 test, $50 test, $200 scale.
- Continuous monitoring and model updates.

**Total estimated timeline: 2-3 weeks to first live test.**

---

## 14. Open Questions

1. **Step frequency**: 100ms gives 3x more data but 3x longer episodes. Is 1s the right tradeoff?

2. **Multi-market concurrency**: Should the agent manage one market at a time or multiple overlapping markets? The current design is one-market-per-episode, but capital efficiency improves with concurrency.

3. **Reward scale tuning**: The imbalance penalty coefficient (0.001) and spread bonus (0.05) need tuning. Consider using reward normalization to handle this automatically.

4. **Whale feature engineering**: Should whale tracking be done in the environment (pre-computed features) or should the agent learn to identify whales from raw book data? Pre-computing is more sample-efficient; raw data is more robust to whale behavior changes.

5. **Off-policy pretraining**: Could we pre-train the agent using behavioral cloning on the existing `MintAndSellMaker`'s actions? This would give a warm start, but might bias the agent toward the static strategy's limitations.

6. **Catastrophic forgetting**: As the market structure evolves (spread compression, new participants), the agent needs periodic retraining. How often? Should we use continual learning techniques?
