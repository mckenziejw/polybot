PPO (what you have)
Pros: stable, well-documented, handles discrete action spaces and action masking natively via sb3-contrib, you already have a working implementation. For a binary outcome market the sparse terminal signal is actually cleaner than continuous markets — there's no ambiguity about what the reward means.
Cons: data inefficient as you've experienced, on-policy constraint wastes most of your historical data, random episode sampling means poor coverage at low step counts, the 3,000-step episode with a terminal binary signal is a hard credit assignment problem even with mark-to-market shaping.
Specific to your context: the 5-minute window is short enough that a single PPO episode is tractable, but long enough that early actions are heavily discounted from the terminal signal. The binary resolution is actually a strength — the reward is unambiguous, unlike a continuous P&L signal that could mean many things.
Verdict: viable but you need far more data and far more steps than you currently have. 2 weeks of markets is simply not enough. This approach becomes interesting at 6-12 months of data.

DQN / Rainbow DQN
Pros: off-policy, meaning it uses a replay buffer and can learn from the same experience multiple times. Far more data efficient than PPO. Handles discrete action spaces naturally. Rainbow DQN adds prioritized experience replay, which would let you oversample the rare terminal reward steps — exactly the credit assignment problem you have.
Cons: more complex to implement correctly than PPO, hyperparameter sensitive, doesn't have native action masking support (requires custom implementation), Q-value overestimation is a known issue in noisy environments which describes yours perfectly.
Specific to your context: the replay buffer is the key advantage. You could store every (observation, action, reward, next_observation) tuple and replay profitable terminal outcomes repeatedly, extracting more signal from each resolved market. With only 2 weeks of data this matters a lot.
Verdict: probably better suited than PPO for your data constraints. The replay buffer effectively multiplies your data. Worth investigating after you've exhausted what you can learn from PPO.

Offline RL (CQL, IQL, TD3+BC)
Pros: trains entirely on a fixed historical dataset, no environment interaction required during training, full coverage of all 3,700 markets with multiple passes, can extract far more signal per market than online RL.
Cons: requires generating a dataset of (s, a, r, s') tuples first using some behavioral policy, conservative by design — explicitly penalizes out-of-distribution actions, which may be too conservative for finding edge. The "distribution shift" problem means the learned policy may fail when deployed if it encounters states not well-represented in the historical data.
Specific to your context: the behavioral policy problem is non-trivial. You'd need to generate actions for your historical data somehow — random actions, or a hand-crafted rule-based strategy. The quality of this behavioral dataset heavily influences what offline RL can learn. With only 2 weeks of data the distribution coverage issue is severe — offline RL needs broad behavioral coverage to learn well, and a random policy on 3,700 markets might not provide that.
Verdict: theoretically appealing but practically difficult with 2 weeks of data. Becomes much more interesting at 6+ months when you have genuine behavioral diversity in the historical record.

Recurrent policies (PPO with LSTM)
This isn't a separate algorithm but a significant architectural change worth calling out. Your current MLP policy sees one observation at a time with no memory of the episode trajectory. An LSTM policy maintains a hidden state across the episode, which is important for trading because:

Entry price and position history matter for exit decisions
Order book dynamics evolve in sequences, not independent snapshots
The agent currently has to infer trajectory from position features, which is a lossy approximation

Pros: much better suited to sequential decision making, can learn time-series patterns within an episode, can be combined with PPO so you keep your existing infrastructure.
Cons: slower to train, harder to tune, longer rollouts needed for the LSTM to develop useful hidden states, gradient flow through time is tricky.
Specific to your context: the 3,000-step episode with a binary outcome is actually a good fit for LSTM — the hidden state can accumulate evidence about market direction over the episode. If momentum patterns exist (which your earlier analysis suggested they might at ~75% win rate), an LSTM is much better positioned to exploit them than an MLP.
Verdict: probably the highest-value architectural change you could make to your current setup. More impactful than switching algorithms.

Contextual bandits
A radical simplification worth considering. Instead of a full sequential RL problem, frame each trading decision as an independent bandit problem: given the current observation, what action maximizes expected immediate reward? This throws away the sequential structure entirely but is far simpler and more data-efficient.
Pros: no credit assignment problem, no discount factor, no value function instability, can be trained on any historical snapshot independently, extremely data efficient.
Cons: ignores position state and episode trajectory, can't learn to time entries and exits as a sequence, likely leaves a lot of edge on the table.
Specific to your context: for a 1-2% edge target this might actually be sufficient. If the edge exists in specific observable market states (e.g. "when spread is tight and momentum is positive, Yes has 52% probability of winning"), a contextual bandit can find and exploit that without needing the full RL apparatus. Simpler model, easier to validate, harder to overfit.
Verdict: underrated for your use case. If the edge is in entry timing rather than position management, this is worth a prototype.

My honest recommendation given your constraints
Short term (2 weeks of data, current codebase): finish the current PPO run, do a proper quantitative train/test evaluation, and treat the result as a proof of concept for the environment and infrastructure rather than a tradeable model. The data is too limited for PPO to find a reliable 1-2% edge.
Medium term (when you have 2-3 months of data): switch to PPO with LSTM. The sequential structure of the problem warrants it, and the additional data will give the recurrent policy enough episodes to develop useful hidden states.
Longer term (6+ months of data): explore offline RL or DQN with a replay buffer. At that point you have enough behavioral diversity in the historical record to make data-efficient algorithms genuinely powerful.
The thing worth keeping in mind: the 75% momentum win rate you identified earlier in your historical analysis is more actionable right now than any RL model. A simple rule-based strategy exploiting that pattern — buy in the direction of a 30-second price move if spread is tight — would be easier to validate, harder to overfit, and potentially live-tradeable in weeks rather than months. RL is the right long-term approach but you may be skipping a step.