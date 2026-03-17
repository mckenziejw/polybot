"""
train_cleanrl.py — PPO + LSTM training for Polymarket BTC Up/Down markets.

CleanRL-style implementation: single file, no framework abstractions.
Everything visible and modifiable.

Usage:
    # Fresh run
    python train_cleanrl.py

    # Resume from checkpoint
    python train_cleanrl.py --resume --model-name ppo_lstm_v1

    # Override hyperparameters
    python train_cleanrl.py --total-timesteps 20_000_000 --num-envs 7

    # Watch training
    tensorboard --logdir runs/
"""

import argparse
import logging
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

import gymnasium as gym

from polymarket_env import PolymarketEnv, N_ACTIONS, OBS_DIM

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

@dataclass
class Args:
    # Paths
    book_dir:        str   = "data/telonex_15m_100ms"
    btc_path:        str   = "data/btc_quotes/btcusdt_quotes.parquet"
    resolution_path: str   = "data/15m_resolutions.json"
    models_dir:      str   = "models"
    runs_dir:        str   = "runs"
    model_name:      str   = "ppo_lstm_v1_15m"

    # Training
    total_timesteps: int   = 10_000_000
    num_envs:        int   = 8
    num_steps:       int   = 2048       # steps per env per rollout
    num_minibatches: int   = 8          # minibatches per update epoch
    update_epochs:   int   = 10         # passes over rollout buffer per update
    checkpoint_freq: int   = 100_000    # save every N total steps

    # PPO
    learning_rate:   float = 3e-4
    gamma:           float = 0.99
    gae_lambda:      float = 0.95
    clip_coef:       float = 0.2
    clip_vloss:      bool  = True
    ent_coef:        float = 0.05
    vf_coef:         float = 0.5
    max_grad_norm:   float = 0.5
    anneal_lr:       bool  = True
    norm_adv:        bool  = True
    target_kl:       float = None       # set e.g. 0.01 to enable early stopping

    # Architecture
    lstm_hidden:     int   = 256        # LSTM hidden state size
    mlp_hidden:      int   = 256        # feature extractor width

    # VecNormalize
    clip_obs:        float = 10.0
    clip_reward:     float = 10.0

    # Misc
    seed:            int   = 42
    resume:          bool  = False

    # Computed at runtime
    batch_size:      int   = 0          # num_envs * num_steps
    minibatch_size:  int   = 0          # batch_size // num_minibatches
    num_iterations:  int   = 0          # total_timesteps // batch_size


def parse_args() -> Args:
    args = Args()
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--book-dir",         default=args.book_dir)
    p.add_argument("--btc-path",         default=args.btc_path)
    p.add_argument("--resolution-path",  default=args.resolution_path)
    p.add_argument("--models-dir",       default=args.models_dir)
    p.add_argument("--runs-dir",         default=args.runs_dir)
    p.add_argument("--model-name",       default=args.model_name)
    p.add_argument("--total-timesteps",  type=int,   default=args.total_timesteps)
    p.add_argument("--num-envs",         type=int,   default=args.num_envs)
    p.add_argument("--num-steps",        type=int,   default=args.num_steps)
    p.add_argument("--num-minibatches",  type=int,   default=args.num_minibatches)
    p.add_argument("--update-epochs",    type=int,   default=args.update_epochs)
    p.add_argument("--checkpoint-freq",  type=int,   default=args.checkpoint_freq)
    p.add_argument("--learning-rate",    type=float, default=args.learning_rate)
    p.add_argument("--gamma",            type=float, default=args.gamma)
    p.add_argument("--gae-lambda",       type=float, default=args.gae_lambda)
    p.add_argument("--clip-coef",        type=float, default=args.clip_coef)
    p.add_argument("--ent-coef",         type=float, default=args.ent_coef)
    p.add_argument("--vf-coef",          type=float, default=args.vf_coef)
    p.add_argument("--max-grad-norm",    type=float, default=args.max_grad_norm)
    p.add_argument("--target-kl",        type=float, default=args.target_kl)
    p.add_argument("--lstm-hidden",      type=int,   default=args.lstm_hidden)
    p.add_argument("--mlp-hidden",       type=int,   default=args.mlp_hidden)
    p.add_argument("--seed",             type=int,   default=args.seed)
    p.add_argument("--resume",           action="store_true")
    parsed = p.parse_args()

    # Copy parsed values onto dataclass
    for field in Args.__dataclass_fields__:
        cli_key = field.replace("_", "-")
        if hasattr(parsed, field):
            setattr(args, field, getattr(parsed, field))

    # Compute derived values
    args.batch_size     = args.num_envs * args.num_steps
    args.minibatch_size = args.batch_size // args.num_minibatches
    args.num_iterations = args.total_timesteps // args.batch_size
    return args


# ---------------------------------------------------------------------------
# VecNormalize
# ---------------------------------------------------------------------------

class RunningMeanStd:
    """Welford's online algorithm for running mean and variance."""
    def __init__(self, shape=(), epsilon=1e-4):
        self.mean  = np.zeros(shape, dtype=np.float64)
        self.var   = np.ones(shape,  dtype=np.float64)
        self.count = epsilon

    def update(self, x: np.ndarray):
        x = np.asarray(x, dtype=np.float64)
        batch_mean  = x.mean(axis=0)
        batch_var   = x.var(axis=0)
        batch_count = x.shape[0]
        delta     = batch_mean - self.mean
        tot_count = self.count + batch_count
        self.mean = self.mean + delta * batch_count / tot_count
        m_a       = self.var * self.count
        m_b       = batch_var * batch_count
        m_2       = m_a + m_b + delta ** 2 * self.count * batch_count / tot_count
        self.var   = m_2 / tot_count
        self.count = tot_count


class VecNormalize:
    """
    Running normalization of observations and rewards across vectorized envs.

    Observations: normalized per-dimension by running mean/variance.
    Rewards: normalized by variance of discounted return estimate (same as SB3).
    """
    def __init__(self, num_envs: int, obs_shape: tuple,
                 clip_obs: float = 10.0, clip_reward: float = 10.0,
                 gamma: float = 0.99, epsilon: float = 1e-8):
        self.obs_rms    = RunningMeanStd(shape=obs_shape)
        self.ret_rms    = RunningMeanStd(shape=())
        self.clip_obs   = clip_obs
        self.clip_rew   = clip_reward
        self.epsilon    = epsilon
        self.gamma      = gamma
        self.returns    = np.zeros(num_envs, dtype=np.float64)
        self.training   = True

    def normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        if self.training:
            self.obs_rms.update(obs)
        norm = (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon)
        return np.clip(norm, -self.clip_obs, self.clip_obs).astype(np.float32)

    def normalize_reward(self, reward: np.ndarray, dones: np.ndarray) -> np.ndarray:
        if self.training:
            self.returns = self.returns * self.gamma + reward
            self.ret_rms.update(self.returns)
            self.returns[dones.astype(bool)] = 0.0
        norm = reward / np.sqrt(self.ret_rms.var + self.epsilon)
        return np.clip(norm, -self.clip_rew, self.clip_rew).astype(np.float32)

    def save(self, path: str):
        np.savez(path,
                 obs_mean=self.obs_rms.mean, obs_var=self.obs_rms.var,
                 obs_count=np.array([self.obs_rms.count]),
                 ret_var=self.ret_rms.var,   ret_count=np.array([self.ret_rms.count]))

    def load(self, path: str):
        d = np.load(path)
        self.obs_rms.mean  = d["obs_mean"]
        self.obs_rms.var   = d["obs_var"]
        self.obs_rms.count = float(d["obs_count"][0])
        self.ret_rms.var   = d["ret_var"]
        self.ret_rms.count = float(d["ret_count"][0])
        return self


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def make_env(book_dir: str, btc_path: str, resolution_path: str, seed: int):
    """Returns a thunk that constructs one PolymarketEnv."""
    def thunk():
        env = PolymarketEnv(
            book_dir=Path(book_dir),
            btc_path=Path(btc_path),
            resolution_path=str(resolution_path),
        )
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env
    return thunk


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    """
    PPO agent with shared LSTM trunk and separate actor/critic heads.

    Architecture:
        obs (63) → MLP feature extractor → LSTM → actor head  (logits over 6 actions)
                                                 → critic head (scalar value)

    The shared trunk means actor and critic develop a common representation
    of market state before specializing. Episode boundaries zero the hidden
    state so each market starts fresh.
    """
    def __init__(self, obs_dim: int = OBS_DIM, action_dim: int = N_ACTIONS,
                 mlp_hidden: int = 256, lstm_hidden: int = 256):
        super().__init__()

        # Shared feature extractor: compress raw obs before LSTM
        self.network = nn.Sequential(
            layer_init(nn.Linear(obs_dim, mlp_hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(mlp_hidden, mlp_hidden)),
            nn.Tanh(),
        )

        # LSTM: learns temporal dependencies within episode
        # hidden_size is the tunable memory capacity
        self.lstm = nn.LSTM(mlp_hidden, lstm_hidden)
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)

        # Actor: near-zero init keeps initial policy close to uniform
        self.actor  = layer_init(nn.Linear(lstm_hidden, action_dim), std=0.01)
        # Critic: std=1.0 allows value estimates to start in reasonable range
        self.critic = layer_init(nn.Linear(lstm_hidden, 1), std=1.0)

    def get_states(self, x: torch.Tensor, lstm_state: tuple,
                   done: torch.Tensor) -> tuple[torch.Tensor, tuple]:
        """
        Run observation through feature extractor and LSTM.

        Handles two cases:
          - Single step (rollout collection): x is (num_envs, obs_dim)
          - Sequence (update phase): x is (seq_len * num_envs, obs_dim)

        Episode boundaries (done=1) zero the hidden state for that env
        so the LSTM doesn't carry memory across market episodes.
        """
        hidden     = self.network(x)
        batch_size = lstm_state[0].shape[1]

        # Reshape to (seq_len, batch, features) for LSTM
        hidden = hidden.reshape((-1, batch_size, self.lstm.input_size))
        done   = done.reshape((-1, batch_size))

        # Step through sequence, resetting hidden state at episode boundaries
        new_hidden = []
        for h, d in zip(hidden, done):
            h, lstm_state = self.lstm(
                h.unsqueeze(0),
                (
                    (1.0 - d).view(1, -1, 1) * lstm_state[0],
                    (1.0 - d).view(1, -1, 1) * lstm_state[1],
                ),
            )
            new_hidden.append(h)

        # Flatten back to (seq_len * batch, lstm_hidden)
        new_hidden = torch.flatten(torch.cat(new_hidden), 0, 1)
        return new_hidden, lstm_state

    def get_value(self, x: torch.Tensor, lstm_state: tuple,
                  done: torch.Tensor) -> torch.Tensor:
        hidden, _ = self.get_states(x, lstm_state, done)
        return self.critic(hidden)

    def get_action_and_value(self, x: torch.Tensor, lstm_state: tuple,
                             done: torch.Tensor, action: torch.Tensor = None,
                             action_masks: torch.Tensor = None):
        """
        Sample or evaluate an action.

        Args:
            x:            observation tensor
            lstm_state:   (h, c) tuple carried between steps
            done:         episode boundary flags (resets LSTM hidden state)
            action:       if provided, evaluate log_prob of this action
                          (used during PPO update). If None, sample.
            action_masks: bool tensor (True = valid). Invalid actions get
                          logit = -inf so they have zero probability.
                          Must be stored during rollout and replayed during
                          update to keep log probs consistent.

        Returns:
            action, log_prob, entropy, value, lstm_state
        """
        hidden, lstm_state = self.get_states(x, lstm_state, done)
        logits = self.actor(hidden)

        if action_masks is not None:
            # Zero out invalid actions: softmax(-inf) = 0, never sampled
            logits = logits.masked_fill(~action_masks.bool(), float("-inf"))

        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden), lstm_state


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(path: Path, agent: Agent, optimizer: optim.Optimizer,
                    vec_norm: VecNormalize, global_step: int, args: Args):
    torch.save({
        "global_step":       global_step,
        "agent_state_dict":  agent.state_dict(),
        "optimizer_state":   optimizer.state_dict(),
        "args":              vars(args),
    }, path / f"ckpt_{global_step:010d}.pt")
    vec_norm.save(str(path / f"vecnorm_{global_step:010d}.npz"))
    log.info(f"Checkpoint saved at step {global_step:,}")


def latest_checkpoint(checkpoints_dir: Path):
    """Return (ckpt_path, vecnorm_path, global_step) for latest checkpoint."""
    pts = sorted(checkpoints_dir.glob("ckpt_*.pt"))
    if not pts:
        return None, None, 0
    ckpt = pts[-1]
    step = int(ckpt.stem.split("_")[1])
    vecnorm = checkpoints_dir / f"vecnorm_{step:010d}.npz"
    return ckpt, vecnorm, step


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    args = parse_args()

    # Derived sizes
    assert args.batch_size % args.num_minibatches == 0, \
        f"batch_size ({args.batch_size}) must be divisible by num_minibatches ({args.num_minibatches})"

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")
    log.info(f"Batch size: {args.batch_size} ({args.num_envs} envs × {args.num_steps} steps)")
    log.info(f"Minibatch size: {args.minibatch_size} ({args.num_minibatches} minibatches)")
    log.info(f"Total iterations: {args.num_iterations:,}")

    # Directories
    models_dir      = Path(args.models_dir)
    checkpoints_dir = models_dir / "checkpoints"
    runs_dir        = Path(args.runs_dir)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    # Verify data paths
    for label, path in [("book_dir",        args.book_dir),
                         ("btc_path",        args.btc_path),
                         ("resolution_path", args.resolution_path)]:
        if not Path(path).exists():
            raise FileNotFoundError(f"--{label} not found: {path}")

    # ------------------------------------------------------------------
    # Environments
    # ------------------------------------------------------------------
    log.info("Initializing environments...")
    envs = gym.vector.SyncVectorEnv([
        make_env(args.book_dir, args.btc_path, args.resolution_path, seed=args.seed + i)
        for i in range(args.num_envs)
    ])

    assert isinstance(envs.single_action_space, gym.spaces.Discrete), \
        "Only discrete action spaces supported"
    assert envs.single_observation_space.shape == (OBS_DIM,), \
        f"Expected obs dim {OBS_DIM}, got {envs.single_observation_space.shape}"

    # ------------------------------------------------------------------
    # VecNormalize
    # ------------------------------------------------------------------
    vec_norm = VecNormalize(
        num_envs=args.num_envs,
        obs_shape=(OBS_DIM,),
        clip_obs=args.clip_obs,
        clip_reward=args.clip_reward,
        gamma=args.gamma,
    )

    # ------------------------------------------------------------------
    # Agent and optimizer
    # ------------------------------------------------------------------
    agent = Agent(
        obs_dim=OBS_DIM,
        action_dim=N_ACTIONS,
        mlp_hidden=args.mlp_hidden,
        lstm_hidden=args.lstm_hidden,
    ).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ------------------------------------------------------------------
    # Resume from checkpoint
    # ------------------------------------------------------------------
    start_step = 0
    if args.resume:
        ckpt_path, vecnorm_path, start_step = latest_checkpoint(checkpoints_dir)
        if ckpt_path is None:
            raise FileNotFoundError(
                f"--resume specified but no checkpoints in {checkpoints_dir}"
            )
        log.info(f"Resuming from {ckpt_path} at step {start_step:,}")
        ckpt = torch.load(ckpt_path, map_location=device)
        agent.load_state_dict(ckpt["agent_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        vec_norm.load(str(vecnorm_path))
        log.info(f"Loaded VecNormalize stats from {vecnorm_path}")

    # ------------------------------------------------------------------
    # TensorBoard
    # ------------------------------------------------------------------
    run_name = f"{args.model_name}_{int(time.time())}"
    writer   = SummaryWriter(f"{args.runs_dir}/{run_name}")
    writer.add_text("hyperparameters",
                    "|param|value|\n|-|-|\n" +
                    "\n".join(f"|{k}|{v}|" for k, v in vars(args).items()))

    # ------------------------------------------------------------------
    # Rollout storage buffers
    # ------------------------------------------------------------------
    obs          = torch.zeros((args.num_steps, args.num_envs, OBS_DIM)).to(device)
    actions      = torch.zeros((args.num_steps, args.num_envs)).to(device)
    logprobs     = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards      = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones        = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values       = torch.zeros((args.num_steps, args.num_envs)).to(device)
    action_masks = torch.zeros((args.num_steps, args.num_envs, N_ACTIONS),
                               dtype=torch.bool).to(device)

    # ------------------------------------------------------------------
    # Initial environment state
    # ------------------------------------------------------------------
    next_obs_raw, _ = envs.reset(seed=args.seed)
    next_obs  = torch.tensor(vec_norm.normalize_obs(next_obs_raw)).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    # LSTM hidden state: (num_layers, num_envs, hidden_size) for h and c
    next_lstm_state = (
        torch.zeros(1, args.num_envs, args.lstm_hidden).to(device),
        torch.zeros(1, args.num_envs, args.lstm_hidden).to(device),
    )

    global_step = start_step
    start_time  = time.time()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    for iteration in range(1, args.num_iterations + 1):

        # Snapshot LSTM state at start of rollout — needed for update phase
        # because we replay the whole rollout sequence through the LSTM
        initial_lstm_state = (
            next_lstm_state[0].clone(),
            next_lstm_state[1].clone(),
        )

        # Learning rate annealing
        if args.anneal_lr:
            frac  = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        # --------------------------------------------------------------
        # Rollout collection
        # --------------------------------------------------------------
        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step]   = next_obs
            dones[step] = next_done

            # Get action masks from each env
            masks = torch.tensor(
                np.array([env.unwrapped.action_masks() for env in envs.envs]),
                dtype=torch.bool,
            ).to(device)
            action_masks[step] = masks

            with torch.no_grad():
                action, logprob, _, value, next_lstm_state = agent.get_action_and_value(
                    next_obs, next_lstm_state, next_done, action_masks=masks
                )
                values[step] = value.flatten()
            actions[step]  = action
            logprobs[step] = logprob

            # Step environments
            next_obs_raw, reward, terminations, truncations, infos = \
                envs.step(action.cpu().numpy())

            next_done_np = np.logical_or(terminations, truncations)
            next_obs  = torch.tensor(
                vec_norm.normalize_obs(next_obs_raw)
            ).to(device)
            next_done = torch.tensor(next_done_np, dtype=torch.float32).to(device)
            rewards[step] = torch.tensor(
                vec_norm.normalize_reward(reward, next_done_np)
            ).to(device)

            # Episode logging via RecordEpisodeStatistics
            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        ep_ret = info["episode"]["r"]
                        ep_len = info["episode"]["l"]
                        log.info(f"step={global_step:,}  ep_return={ep_ret:.2f}"
                                 f"  ep_len={ep_len}")
                        writer.add_scalar("charts/episodic_return", ep_ret, global_step)
                        writer.add_scalar("charts/episodic_length", ep_len, global_step)

        # --------------------------------------------------------------
        # Advantage estimation (GAE) — runs after rollout is complete
        # --------------------------------------------------------------
        with torch.no_grad():
            next_value = agent.get_value(
                next_obs, next_lstm_state, next_done
            ).reshape(1, -1)

            advantages  = torch.zeros_like(rewards).to(device)
            lastgaelam  = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues      = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues      = values[t + 1]
                delta          = (rewards[t]
                                  + args.gamma * nextvalues * nextnonterminal
                                  - values[t])
                advantages[t]  = lastgaelam = (delta
                                               + args.gamma * args.gae_lambda
                                               * nextnonterminal * lastgaelam)
            returns = advantages + values

        # --------------------------------------------------------------
        # Flatten rollout buffers for minibatch sampling
        # --------------------------------------------------------------
        b_obs          = obs.reshape((-1, OBS_DIM))
        b_actions      = actions.reshape(-1)
        b_logprobs     = logprobs.reshape(-1)
        b_dones        = dones.reshape(-1)
        b_advantages   = advantages.reshape(-1)
        b_returns      = returns.reshape(-1)
        b_values       = values.reshape(-1)
        b_action_masks = action_masks.reshape((-1, N_ACTIONS))

        # --------------------------------------------------------------
        # PPO update
        # --------------------------------------------------------------
        # LSTM requires env-sequential minibatches (not random flat indices)
        # because we replay hidden states from the start of each rollout.
        # We shuffle over environments, then take contiguous time slices.
        assert args.num_envs % args.num_minibatches == 0, \
            "num_envs must be divisible by num_minibatches for LSTM minibatching"
        envs_per_batch = args.num_envs // args.num_minibatches
        env_inds  = np.arange(args.num_envs)
        flat_inds = np.arange(args.batch_size).reshape(args.num_steps, args.num_envs)

        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(env_inds)
            for start in range(0, args.num_envs, envs_per_batch):
                end        = start + envs_per_batch
                mb_env_inds = env_inds[start:end]
                # Contiguous time-slice for these envs — preserves LSTM sequence order
                mb_inds    = flat_inds[:, mb_env_inds].ravel()

                _, newlogprob, entropy, newvalue, _ = agent.get_action_and_value(
                    b_obs[mb_inds],
                    (initial_lstm_state[0][:, mb_env_inds],
                     initial_lstm_state[1][:, mb_env_inds]),
                    b_dones[mb_inds],
                    b_actions.long()[mb_inds],
                    action_masks=b_action_masks[mb_inds],
                )

                logratio = newlogprob - b_logprobs[mb_inds]
                ratio    = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl     = ((ratio - 1) - logratio).mean()
                    clipfracs.append(
                        ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                    )

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = ((mb_advantages - mb_advantages.mean())
                                     / (mb_advantages.std() + 1e-8))

                # Policy loss (clipped surrogate objective)
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - args.clip_coef, 1 + args.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss (optionally clipped)
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = (b_values[mb_inds]
                                 + torch.clamp(newvalue - b_values[mb_inds],
                                               -args.clip_coef, args.clip_coef))
                    v_loss_clipped   = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            # Early stopping on KL divergence
            if args.target_kl is not None and approx_kl > args.target_kl:
                log.info(f"Early stopping at epoch {epoch} (KL {approx_kl:.4f} > {args.target_kl})")
                break

        # --------------------------------------------------------------
        # Logging
        # --------------------------------------------------------------
        y_pred = b_values.cpu().numpy()
        y_true = b_returns.cpu().numpy()
        var_y  = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        sps = int(global_step / (time.time() - start_time))
        writer.add_scalar("charts/learning_rate",        optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("charts/SPS",                  sps,                             global_step)
        writer.add_scalar("losses/value_loss",           v_loss.item(),                   global_step)
        writer.add_scalar("losses/policy_loss",          pg_loss.item(),                  global_step)
        writer.add_scalar("losses/entropy",              entropy_loss.item(),             global_step)
        writer.add_scalar("losses/old_approx_kl",        old_approx_kl.item(),            global_step)
        writer.add_scalar("losses/approx_kl",            approx_kl.item(),                global_step)
        writer.add_scalar("losses/clipfrac",             np.mean(clipfracs),              global_step)
        writer.add_scalar("losses/explained_variance",   explained_var,                   global_step)

        if iteration % 10 == 0:
            log.info(f"iter={iteration}  step={global_step:,}  SPS={sps}"
                     f"  v_loss={v_loss.item():.4f}"
                     f"  entropy={entropy_loss.item():.3f}"
                     f"  explained_var={explained_var:.3f}")

        # --------------------------------------------------------------
        # Checkpointing
        # --------------------------------------------------------------
        if global_step % args.checkpoint_freq < args.batch_size:
            save_checkpoint(checkpoints_dir, agent, optimizer, vec_norm,
                            global_step, args)

    # ------------------------------------------------------------------
    # Final save
    # ------------------------------------------------------------------
    final_pt  = models_dir / f"{args.model_name}_final.pt"
    final_vn  = models_dir / f"{args.model_name}_vecnorm_final.npz"
    torch.save({
        "global_step":      global_step,
        "agent_state_dict": agent.state_dict(),
        "args":             vars(args),
    }, final_pt)
    vec_norm.save(str(final_vn))
    log.info(f"Saved final model to {final_pt}")
    log.info(f"Saved VecNormalize stats to {final_vn}")

    envs.close()
    writer.close()


if __name__ == "__main__":
    main()