"""
train_agent.py

Training entry point for the Travel & Hospitality Dynamic Pricing project.

Scope of this deliverable (Week 2, Day 1 — "Integrate RL Training
Pipeline", issue #38)
--------------------------------------------------------------------------
This script wires the training *pipeline* together: it loads a
`TrainingConfig`, builds a `PricingEnvironment` from it, and verifies the
environment is compatible with the training loop (Gymnasium API
compliance + a clean `reset()`).

It deliberately does NOT yet implement a trainable agent (tabular
Q-Learning / DQN) — that is scoped as a separate, follow-on task, exactly
the same way `pricing_env.py`'s Day 3 skeleton shipped `reset()`/`render()`
with a placeholder `step()` before Day 4 filled in the real demand/reward
logic (see Week1_report_Abinaya.md, Day 3). Following that precedent here:
rather than silently faking a trained policy, the per-episode action is
supplied by an explicit, swappable `policy_fn` that defaults to a random
policy. This keeps the pipeline runnable end-to-end today, while making it
unmistakable — in the code and in the logs — that no learning is happening
yet. The next task drops a real agent in by replacing `policy_fn`.

Usage
-----
    python -m training.train_agent
    python -m training.train_agent --episodes 200 --seed 7
"""

from __future__ import annotations

import argparse
import logging
from typing import Callable, Optional

import gymnasium as gym
import numpy as np
from gymnasium.utils.env_checker import check_env

from configs.training_config import TrainingConfig, get_default_training_config
from pricing_env import PricingEnvironment

logger = logging.getLogger(__name__)

# A policy is any callable mapping an observation to a valid discrete
# action index. Typed as its own alias so swapping in a real agent later
# (e.g. `q_learning_agent.select_action`) is a one-line change here.
PolicyFn = Callable[[np.ndarray], int]


# --------------------------------------------------------------------------- #
# Environment construction
# --------------------------------------------------------------------------- #
def build_environment(config: TrainingConfig) -> PricingEnvironment:
    """
    Construct the `PricingEnvironment` for this training run.

    Kept as its own function (rather than inlined in `main()`) so the
    evaluation harness and any future agent-training script can build an
    identically-configured environment from the same `TrainingConfig`
    without duplicating this call.
    """
    env = PricingEnvironment(config.env_config)
    logger.info(
        "Environment constructed | inventory=%d | horizon=%d days | "
        "base_price=$%.2f | actions=%d",
        config.env_config.initial_inventory,
        config.env_config.selling_horizon_days,
        config.env_config.base_price,
        env.action_space.n,
    )
    return env


def verify_environment_compatibility(env: PricingEnvironment) -> None:
    """
    Verify the environment is Gymnasium-API-compliant and resets cleanly.

    Two checks, corresponding directly to this issue's acceptance
    criteria:

      1. `check_env` — Gymnasium's own compliance checker. Catches
         malformed observation/action spaces, incorrect `step()`/`reset()`
         return shapes, etc. Running this here (once, at pipeline-startup
         time) means any future accidental regression in `pricing_env.py`
         fails loudly during pipeline setup, rather than surfacing as a
         confusing shape-mismatch deep inside an agent's training loop.
      2. `env.reset()` — confirms a fresh episode can actually be started
         and returns a well-formed observation, independent of whatever
         `check_env` covers internally.

    Raises
    ------
    Exception
        Re-raises whatever `check_env` or `reset()` raise, uncaught. A
        training run must never proceed against an environment that fails
        this check — silently continuing would risk training against
        malformed observations/rewards with no clear symptom until much
        later (mirrors the fail-loud philosophy already used throughout
        `pricing_env.py`, e.g. `step()`'s `RuntimeError`s).
    """
    logger.info("Verifying Gymnasium API compliance (check_env)...")
    check_env(env.unwrapped, skip_render_check=True)
    logger.info("check_env passed — environment is Gymnasium-API-compliant.")

    logger.info("Verifying reset()...")
    observation, info = env.reset(seed=None)
    if observation.shape != env.observation_space.shape:
        raise RuntimeError(
            f"reset() returned observation shape {observation.shape}, "
            f"expected {env.observation_space.shape}"
        )
    logger.info(
        "reset() passed | observation=%s | initial_inventory=%s | "
        "selling_horizon_days=%s",
        observation.tolist(),
        info.get("initial_inventory"),
        info.get("selling_horizon_days"),
    )


# --------------------------------------------------------------------------- #
# Placeholder policy (replaced by the real agent in a follow-on task)
# --------------------------------------------------------------------------- #
def random_policy(observation: np.ndarray, env: PricingEnvironment) -> int:
    """
    Sample a uniformly random valid action.

    This is an explicit placeholder, not an agent. It exists so the
    training loop below can be exercised end-to-end today (env -> action
    -> step -> reward) without waiting on the Q-Learning/DQN agent, the
    same way Day 3's placeholder `step()` let the environment skeleton be
    exercised before Day 4's real demand/reward logic landed. `env` is
    accepted alongside `observation` purely to sample from its
    `action_space`; a trained agent's `policy_fn` would ignore `env`
    entirely and act on `observation` alone.
    """
    return int(env.action_space.sample())


# --------------------------------------------------------------------------- #
# Training loop
# --------------------------------------------------------------------------- #
def run_training(
    env: PricingEnvironment,
    config: TrainingConfig,
    policy_fn: Optional[PolicyFn] = None,
) -> None:
    """
    Run the training loop for `config.num_episodes` episodes.

    Parameters
    ----------
    env : PricingEnvironment
        Already-constructed environment (see `build_environment`).
    config : TrainingConfig
        Supplies episode budget, the seed, and the per-episode step cap.
    policy_fn : PolicyFn, optional
        Maps an observation to an action index. Defaults to
        `random_policy` — see the module docstring for why. Replacing
        this with a real agent's action-selection method (and adding the
        corresponding learning update after each `step()`) is the entire
        scope of the follow-on "implement Q-Learning agent" task; nothing
        else in this loop should need to change.

    Notes
    -----
    No agent weights/checkpoints are saved by this placeholder loop even
    though `config.checkpoint_dir` is already defined — there is nothing
    trainable to checkpoint yet. The config field exists now so the loop
    below only needs a few added lines (not a reshaped config) once a real
    agent lands.
    """
    if policy_fn is None:
        policy_fn = lambda obs: random_policy(obs, env)  # noqa: E731

    for episode in range(1, config.num_episodes + 1):
        observation, _info = env.reset(seed=config.seed + episode)
        terminated = truncated = False
        episode_reward = 0.0
        steps = 0

        while not (terminated or truncated):
            action = policy_fn(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1

            if (
                config.max_steps_per_episode is not None
                and steps >= config.max_steps_per_episode
            ):
                # Defensive cap only — see TrainingConfig.max_steps_per_episode.
                truncated = True

        if episode % config.log_every_n_episodes == 0 or episode == 1:
            logger.info(
                "Episode %d/%d | steps=%d | episode_reward=%.2f | "
                "final_revenue=$%.2f",
                episode,
                config.num_episodes,
                steps,
                episode_reward,
                info.get("episode_revenue", 0.0),
            )

    logger.info("Training loop complete: %d episodes run.", config.num_episodes)


# --------------------------------------------------------------------------- #
# CLI / entry point
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    """
    Minimal CLI for overriding the most commonly-tweaked training
    parameters without editing `configs/training_config.py` directly.
    Anything not exposed here can still be changed by editing
    `TrainingConfig`'s defaults or constructing one programmatically.
    """
    parser = argparse.ArgumentParser(description="Train an agent against PricingEnvironment.")
    parser.add_argument(
        "--episodes", type=int, default=None, help="Override TrainingConfig.num_episodes."
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Override TrainingConfig.seed."
    )
    parser.add_argument(
        "--skip-verification",
        action="store_true",
        help="Skip the check_env/reset() compatibility verification step.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    args = parse_args()

    config = get_default_training_config()
    if args.episodes is not None or args.seed is not None:
        # TrainingConfig is frozen (immutable) by design — see its
        # docstring — so an override is built via `dataclasses.replace`
        # rather than mutating fields in place.
        from dataclasses import replace

        overrides = {}
        if args.episodes is not None:
            overrides["num_episodes"] = args.episodes
        if args.seed is not None:
            overrides["seed"] = args.seed
        config = replace(config, **overrides)

    logger.info(
        "Loaded TrainingConfig | num_episodes=%d | seed=%d | lr=%.3f | gamma=%.3f",
        config.num_episodes,
        config.seed,
        config.learning_rate,
        config.discount_factor,
    )

    env = build_environment(config)

    if not args.skip_verification:
        verify_environment_compatibility(env)

    try:
        run_training(env, config)
    finally:
        env.close()


if __name__ == "__main__":
    main()