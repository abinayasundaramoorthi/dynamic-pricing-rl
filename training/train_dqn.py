"""
train_dqn.py

DQN training entry point for the Travel & Hospitality Dynamic Pricing
project.

Structure (build environment -> verify compatibility -> build agent ->
train -> save -> reload from disk -> evaluate) mirrors
`training/train_agent.py`'s Q-Learning entry point - see that file's
module docstring for the rationale. `training/env_utils.py` holds the
`build_environment` / `verify_environment_compatibility` logic shared by
both.

Per environment step, the training loop:
  1. `agent.select_action(observation)` -- epsilon-greedy action choice
  2. `env.step(action)` -- take the action
  3. `agent.remember(...)` -- push the resulting transition onto the
     replay buffer
  4. `agent.train_step()` -- one gradient update from a sampled
     mini-batch (a no-op returning `None` until the buffer has at least
     `config.min_replay_size_before_training` transitions)

Usage
-----
    python -m training.train_dqn
    python -m training.train_dqn --episodes 500 --seed 7
    python -m training.train_dqn --episodes 50 --skip-verification   # fast smoke test
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import replace
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

from agents.dqn_agent import DQNAgent
from configs.dqn_config import DQNConfig, get_default_dqn_config
from pricing_env import PricingEnvironment
from training.env_utils import build_environment, verify_environment_compatibility

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Agent construction
# --------------------------------------------------------------------------- #
def build_agent(config: DQNConfig, env: PricingEnvironment) -> DQNAgent:
    """Construct a `DQNAgent` sized to `env`'s observation/action spaces."""
    return DQNAgent.from_config(config, env)


# --------------------------------------------------------------------------- #
# Training loop
# --------------------------------------------------------------------------- #
def run_training(
    env: PricingEnvironment,
    config: DQNConfig,
    agent: DQNAgent,
) -> Tuple[List[float], List[float]]:
    """
    Run `config.num_episodes` episodes, training `agent` online.

    Every environment step is stored via `agent.remember(...)` and
    followed by `config.gradient_steps_per_env_step` calls to
    `agent.train_step()` (a no-op until the replay buffer has warmed up
    past `config.min_replay_size_before_training`). `agent.decay_epsilon()`
    runs once per completed episode.

    Returns
    -------
    Tuple[List[float], List[float]]
        `(episode_rewards, episode_revenues)`, one entry per episode.
    """
    episode_rewards: List[float] = []
    episode_revenues: List[float] = []

    for episode in range(1, config.num_episodes + 1):
        observation, _info = env.reset(seed=config.seed + episode)
        terminated = truncated = False
        episode_reward = 0.0
        steps = 0
        info = {}

        while not (terminated or truncated):
            action = agent.select_action(observation)
            next_observation, reward, terminated, truncated, info = env.step(action)

            agent.remember(observation, action, reward, next_observation, terminated or truncated)
            for _ in range(config.gradient_steps_per_env_step):
                agent.train_step()

            observation = next_observation
            episode_reward += reward
            steps += 1

            if (
                config.max_steps_per_episode is not None
                and steps >= config.max_steps_per_episode
            ):
                truncated = True

        agent.decay_epsilon()

        episode_rewards.append(episode_reward)
        episode_revenues.append(info.get("episode_revenue", 0.0))

        if episode % config.log_every_n_episodes == 0 or episode == 1:
            logger.info(
                "Episode %d/%d | steps=%d | reward=%.2f | revenue=₹%.2f | "
                "epsilon=%.3f | buffer=%d",
                episode,
                config.num_episodes,
                steps,
                episode_reward,
                info.get("episode_revenue", 0.0),
                agent.epsilon,
                len(agent.replay_buffer),
            )

    logger.info("Training loop complete: %d episodes run.", config.num_episodes)
    return episode_rewards, episode_revenues


# --------------------------------------------------------------------------- #
# Policy saving and evaluation
# --------------------------------------------------------------------------- #
def save_policy(agent: DQNAgent, config: DQNConfig) -> Path:
    """Save `agent`'s Q-network weights under `config.checkpoint_dir`."""
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "dqn_policy.pt"
    agent.save_model(str(checkpoint_path))
    return checkpoint_path


def evaluate_agent(
    env: PricingEnvironment,
    agent: DQNAgent,
    num_episodes: int,
    seed_offset: int = 1_000_000,
) -> dict:
    """
    Run `num_episodes` episodes with `agent` acting purely greedily (no
    exploration) and return summary statistics.

    `seed_offset` shifts evaluation seeds well clear of the training seed
    range (`config.seed + episode`) - evaluating on the same seeds a
    policy trained against risks measuring memorization rather than
    generalization.
    """
    episode_rewards = []
    episode_revenues = []

    for episode in range(1, num_episodes + 1):
        observation, info = env.reset(seed=seed_offset + episode)
        terminated = truncated = False
        episode_reward = 0.0

        while not (terminated or truncated):
            action = agent.select_action(observation, greedy=True)
            observation, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward

        episode_rewards.append(episode_reward)
        episode_revenues.append(info["episode_revenue"])

    return {
        "num_eval_episodes": num_episodes,
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "mean_revenue": float(np.mean(episode_revenues)),
        "std_revenue": float(np.std(episode_revenues)),
    }


# --------------------------------------------------------------------------- #
# CLI / entry point
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a DQN agent against PricingEnvironment."
    )
    parser.add_argument(
        "--episodes", type=int, default=None, help="Override DQNConfig.num_episodes."
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Override DQNConfig.seed."
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

    config = get_default_dqn_config()
    overrides = {}
    if args.episodes is not None:
        overrides["num_episodes"] = args.episodes
    if args.seed is not None:
        overrides["seed"] = args.seed
    if overrides:
        config = replace(config, **overrides)

    device = "cuda" if (config.device == "cuda" and torch.cuda.is_available()) else "cpu"
    if device != config.device:
        config = replace(config, device=device)

    logger.info(
        "Loaded DQNConfig | num_episodes=%d | seed=%d | lr=%.4f | gamma=%.3f | "
        "epsilon=%.3f->%.3f | batch_size=%d | device=%s",
        config.num_episodes,
        config.seed,
        config.learning_rate,
        config.discount_factor,
        config.exploration_rate,
        config.exploration_min,
        config.batch_size,
        config.device,
    )

    env = build_environment(config.env_config)

    if not args.skip_verification:
        verify_environment_compatibility(env)

    try:
        agent = build_agent(config, env)
        run_training(env, config, agent)

        checkpoint_path = save_policy(agent, config)
        logger.info("Saved trained agent to %s", checkpoint_path)

        # Reload from disk into a FRESH agent instance -- proves the
        # save/load round-trip actually works, not just that a file
        # got written.
        reloaded_agent = build_agent(config, env)
        reloaded_agent.load_model(str(checkpoint_path))

        eval_summary = evaluate_agent(
            env, reloaded_agent, num_episodes=config.num_eval_episodes
        )
        logger.info(
            "Evaluation of reloaded policy (%d episodes, greedy) | "
            "mean_reward=%.2f | mean_revenue=₹%.2f | std_revenue=₹%.2f",
            eval_summary["num_eval_episodes"],
            eval_summary["mean_reward"],
            eval_summary["mean_revenue"],
            eval_summary["std_revenue"],
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()