"""
train.py — Train MaskablePPO on Polymarket 5-minute BTC Up/Down markets.

Usage:
    # Fresh run
    python train.py

    # Resume from latest checkpoint
    python train.py --resume

    # Override key hyperparameters
    python train.py --timesteps 20_000_000 --n-envs 7 --gamma 0.9999

    # Watch training live (separate terminal)
    tensorboard --logdir runs/

Outputs:
    models/checkpoints/   — checkpoint every --checkpoint-freq steps
    models/ppo_v1_final   — final saved model
    runs/                 — TensorBoard logs
"""

import argparse
import logging
import os
from pathlib import Path

import torch
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from polymarket_env import PolymarketEnv

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULTS = dict(
    book_dir       = "data/telonex_100ms",
    btc_path       = "data/btc_quotes/btcusdt_quotes.parquet",
    resolution_path  = "data/resolutions.json",
    n_envs         = 7,
    timesteps      = 10_000_000,
    n_steps        = 4096,      # steps per env per rollout; total batch = n_steps * n_envs
    batch_size     = 256,       # minibatch size for gradient updates
    n_epochs       = 10,        # passes over each rollout buffer
    gamma          = 0.99,      # reduced from 0.999 — prevents value fn exploiting time signal
    learning_rate  = 3e-4,
    ent_coef       = 0.1,      # increased from 0.01 — slows entropy collapse and overtrading
    checkpoint_freq = 100_000,  # save checkpoint every N *total* steps
    model_name     = "ppo_v1",
    runs_dir       = "runs",
    models_dir     = "models",
)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--resume",          action="store_true",
                   help="Resume training from latest checkpoint in models/checkpoints/")
    p.add_argument("--book-dir",        default=DEFAULTS["book_dir"])
    p.add_argument("--btc-path",        default=DEFAULTS["btc_path"])
    p.add_argument("--resolution-path", default=DEFAULTS["resolution_path"])
    p.add_argument("--n-envs",          type=int,   default=DEFAULTS["n_envs"])
    p.add_argument("--timesteps",       type=int,   default=DEFAULTS["timesteps"])
    p.add_argument("--n-steps",         type=int,   default=DEFAULTS["n_steps"])
    p.add_argument("--batch-size",      type=int,   default=DEFAULTS["batch_size"])
    p.add_argument("--n-epochs",        type=int,   default=DEFAULTS["n_epochs"])
    p.add_argument("--gamma",           type=float, default=DEFAULTS["gamma"])
    p.add_argument("--learning-rate",   type=float, default=DEFAULTS["learning_rate"])
    p.add_argument("--ent-coef",        type=float, default=DEFAULTS["ent_coef"])
    p.add_argument("--checkpoint-freq", type=int,   default=DEFAULTS["checkpoint_freq"])
    p.add_argument("--model-name",                  default=DEFAULTS["model_name"])
    p.add_argument("--models-dir",                  default=DEFAULTS["models_dir"])
    p.add_argument("--runs-dir",                    default=DEFAULTS["runs_dir"])
    return p.parse_args()


# ---------------------------------------------------------------------------
# Env factory
# ---------------------------------------------------------------------------

def make_env_fn(book_dir: str, btc_path: str, resolution_path: str, seed: int):
    """Returns a callable that creates one PolymarketEnv instance."""
    def _init():
        env = PolymarketEnv(
            book_dir=Path(book_dir),
            btc_path=Path(btc_path),
            resolution_path=str(resolution_path),
        )
        env = Monitor(env)  # required for ep_rew_mean / ep_len_mean logging
        env.reset(seed=seed)
        return env
    return _init


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def latest_checkpoint(checkpoints_dir: Path) -> Path | None:
    """Return the most recent checkpoint .zip, or None."""
    zips = sorted(checkpoints_dir.glob("*.zip"))
    return zips[-1] if zips else None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger(__name__)
    args = parse_args()

    # Directories
    models_dir      = Path(args.models_dir)
    checkpoints_dir = models_dir / "checkpoints"
    runs_dir        = Path(args.runs_dir)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    # Verify data paths exist before spinning up subprocesses
    for label, path in [("book_dir", args.book_dir),
                         ("btc_path", args.btc_path),
                         ("resolution_path", args.resolution_path)]:
        if not Path(path).exists():
            raise FileNotFoundError(f"--{label} not found: {path}")

    log.info(f"PyTorch {torch.__version__} | device: cpu (env-bottlenecked)")
    log.info(f"n_envs={args.n_envs}  total_timesteps={args.timesteps:,}")
    log.info(f"gamma={args.gamma}  lr={args.learning_rate}  ent_coef={args.ent_coef}  VecNormalize=on")
    log.info(f"book_dir={args.book_dir}")

    # ------------------------------------------------------------------
    # Build vectorised environment
    # ------------------------------------------------------------------
    log.info("Spawning SubprocVecEnv...")
    env_fns = [
        make_env_fn(args.book_dir, args.btc_path, args.resolution_path, seed=i)
        for i in range(args.n_envs)
    ]
    raw_vec_env = SubprocVecEnv(env_fns, start_method="fork")

    # Separate eval env (single, deterministic episode order) for EvalCallback
    raw_eval_env = SubprocVecEnv(
        [make_env_fn(args.book_dir, args.btc_path, args.resolution_path, seed=999)],
        start_method="fork",
    )

    # ------------------------------------------------------------------
    # VecNormalize — normalises observations and rewards online.
    # Fixes two problems:
    #   1. Reward scale: terminal PnL (~$350) causes value function divergence
    #   2. Observation scale: features span very different ranges
    #      (mid_price ~0.5, bid_size ~thousands, staleness ~0-5000)
    # norm_reward=True  — reward normalisation (running mean/var)
    # norm_obs=True     — observation normalisation (running mean/var per dim)
    # clip_obs=10.0     — clips normalised obs to [-10, 10] (SB3 default)
    # clip_reward=10.0  — clips normalised reward to [-10, 10]
    #
    # IMPORTANT: VecNormalize statistics must be saved/loaded alongside the
    # model checkpoint — a model loaded without its normalizer will perform
    # garbage inference because it sees un-normalised observations.
    # ------------------------------------------------------------------
    normalizer_path = checkpoints_dir / "vecnormalize.pkl"

    # ------------------------------------------------------------------
    # Build or load model
    # ------------------------------------------------------------------
    if args.resume:
        ckpt = latest_checkpoint(checkpoints_dir)
        if ckpt is None:
            raise FileNotFoundError(
                f"--resume specified but no checkpoints found in {checkpoints_dir}"
            )
        if not normalizer_path.exists():
            raise FileNotFoundError(
                f"--resume specified but normalizer not found at {normalizer_path}. "
                f"Cannot resume without VecNormalize statistics."
            )
        log.info(f"Resuming from checkpoint: {ckpt}")
        log.info(f"Loading VecNormalize stats from {normalizer_path}")

        vec_env  = VecNormalize.load(str(normalizer_path), raw_vec_env)
        eval_env = VecNormalize.load(str(normalizer_path), raw_eval_env)
        eval_env.training = False   # freeze normalizer stats during eval
        eval_env.norm_reward = False  # don't normalise eval rewards (want real PnL)

        model = MaskablePPO.load(ckpt, env=vec_env, device="cpu")
        steps_done = int(ckpt.stem.split("_")[-2])
        remaining  = max(0, args.timesteps - steps_done)
        log.info(f"Steps done: {steps_done:,}  Remaining: {remaining:,}")
    else:
        remaining = args.timesteps
        vec_env  = VecNormalize(raw_vec_env,  norm_obs=True, norm_reward=True,
                                clip_obs=10.0, clip_reward=10.0)
        eval_env = VecNormalize(raw_eval_env, norm_obs=True, norm_reward=False,
                                clip_obs=10.0, training=False)
        # eval_env shares the same normalizer stats but:
        #   training=False  — don't update running stats during eval
        #   norm_reward=False — report real PnL in eval, not normalised reward

        model = MaskablePPO(
            policy         = "MlpPolicy",
            env            = vec_env,
            n_steps        = args.n_steps,
            batch_size     = args.batch_size,
            n_epochs       = args.n_epochs,
            gamma          = args.gamma,
            learning_rate  = args.learning_rate,
            ent_coef       = args.ent_coef,
            verbose        = 1,
            tensorboard_log= str(runs_dir),
            device         = "cpu",
            seed           = 42,
        )

    log.info(f"Policy network:\n{model.policy}")

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    checkpoint_cb = CheckpointCallback(
        save_freq   = max(args.checkpoint_freq // args.n_envs, 1),  # per-env steps
        save_path   = str(checkpoints_dir),
        name_prefix = args.model_name,
        verbose     = 1,
    )

    eval_cb = MaskableEvalCallback(
        eval_env,
        best_model_save_path = str(models_dir / "best"),
        log_path             = str(runs_dir / "eval"),
        eval_freq            = max(500_000 // args.n_envs, 1),  # every ~500k total steps
        n_eval_episodes      = 50,
        deterministic        = True,
        verbose              = 1,
    )

    # Save VecNormalize stats whenever a model checkpoint is saved,
    # and keep eval_env normalizer in sync with training env.
    from stable_baselines3.common.callbacks import BaseCallback

    class NormalizerSyncCallback(BaseCallback):
        """
        Saves VecNormalize statistics alongside each model checkpoint,
        and syncs eval_env running stats from the training env so eval
        observations are normalised consistently.
        """
        def __init__(self, vec_env, eval_env, save_path, save_freq, verbose=0):
            super().__init__(verbose)
            self.vec_env   = vec_env
            self.eval_env  = eval_env
            self.save_path = save_path
            self.save_freq = save_freq

        def _on_step(self) -> bool:
            if self.n_calls % self.save_freq == 0:
                self.vec_env.save(str(self.save_path))
                if self.verbose:
                    print(f"Saved VecNormalize stats to {self.save_path}")
                # Sync eval normalizer stats from training env
                self.eval_env.obs_rms    = self.vec_env.obs_rms
                self.eval_env.ret_rms    = self.vec_env.ret_rms
            return True

    norm_save_freq = max(args.checkpoint_freq // args.n_envs, 1)
    normalizer_cb  = NormalizerSyncCallback(
        vec_env, eval_env,
        save_path = normalizer_path,
        save_freq = norm_save_freq,
        verbose   = 1,
    )

    callbacks = CallbackList([checkpoint_cb, eval_cb, normalizer_cb])

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    log.info("Training started. Run: tensorboard --logdir runs/")
    try:
        model.learn(
            total_timesteps   = remaining,
            callback          = callbacks,
            reset_num_timesteps = not args.resume,
            tb_log_name       = args.model_name,
            progress_bar      = True,
        )
    except KeyboardInterrupt:
        log.info("Training interrupted by user.")

    # ------------------------------------------------------------------
    # Save final model
    # ------------------------------------------------------------------
    final_path = models_dir / f"{args.model_name}_final"
    model.save(str(final_path))
    vec_env.save(str(models_dir / f"{args.model_name}_vecnormalize.pkl"))
    log.info(f"Saved final model to {final_path}.zip")
    log.info(f"Saved VecNormalize stats to {args.model_name}_vecnormalize.pkl")

    vec_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()