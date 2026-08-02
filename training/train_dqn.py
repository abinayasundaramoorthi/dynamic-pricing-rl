"""
train_agent.py

Q-Learning training entry point for the Travel & Hospitality Dynamic
Pricing project.

Structure (build environment -> verify compatibility -> build agent ->
train -> save -> reload from disk -> evaluate) is the reference pattern
that training/train_dqn.py and training/run_experiment.py both mirror -
see those files' module docstrings for how they reuse this shape.

Usage
-----
    python -m training.train_agent
    python -m training.train_agent --episodes 500 --seed 7
    python -m training.train_agent --agent random   # baseline, no learning, no save/reload
    python -m training.train_agent --episodes 50 --skip-verification   # fast smoke test
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import replace
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np

from agents.q_learning_agent import QLearningAgent
from configs.training_config import TrainingConfig, get_default_training_config
from pricing_env import PricingEnvironment
from training.env_utils import build_environment, verify_environment_compatibility

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Baseline placeholder policy
# --------------------------------------------------------------------------- #
def random_policy(observation: np.ndarray, env: PricingEnvironment) -> int:
    """
    Baseline placeholder policy: pick a uniformly random valid action,
    ignoring `observation` entirely. Used as the default when no agent is
    given to `run_training()`, and directly by `--agent random` runs and
    by `training/run_experiment.py`'s current pre-agent experiment suite.
    """
    return env.action_space.sample()


# --------------------------------------------------------------------------- #
# Agent construction
# --------------------------------------------------------------------------- #
def build_agent(config: TrainingConfig, env: PricingEnvironment) -> QLearningAgent:
    """Construct a `QLearningAgent` sized to `env`'s action space."""
    return QLearningAgent(
        num_actions=env.action_space.n,
        learning_rate=config.learning_rate,
        discount_factor=config.discount_factor,
        epsilon=config.exploration_rate,
        epsilon_decay=config.exploration_decay,
        epsilon_min=config.exploration_min,
    )


# --------------------------------------------------------------------------- #
# Training loop
# --------------------------------------------------------------------------- #
def run_training(
    env: PricingEnvironment,
    config: TrainingConfig,
    agent: Optional[QLearningAgent] = None,
    policy_fn: Optional[Callable[[np.ndarray, PricingEnvironment], int]] = None,
    collect_metrics: bool = False,
) -> Optional[Tuple[List[float], List[float]]]:
    """
    Run `config.num_episodes` episodes.

    Parameters
    ----------
    env : PricingEnvironment
    config : TrainingConfig
    agent : QLearningAgent, optional
        If given, takes priority over `policy_fn`: the agent's
        `select_action()` chooses each action, `update()` is called after
        every step so it can learn from the transition, and
        `decay_exploration()` is called once per completed episode.
    policy_fn : callable, optional
        Maps an observation to an action index, with NO learning. Used
        only when `agent` is not given. Defaults to `random_policy`.
    collect_metrics : bool, default=False
        If True, record every episode's total reward and final revenue
        and return them as `(episode_rewards, episode_revenues)`. If
        False, returns None - progress is only logged, not returned.
        `training/run_experiment.py` does NOT call this function with
        collect_metrics=True (it has its own parallel loop for that, by
        design - see that file's `_run_episodes()` docstring); this flag
        exists for direct callers of `train_agent.py` that want metrics
        back without re-implementing the loop.

    Returns
    -------
    Optional[Tuple[List[float], List[float]]]
        `(episode_rewards, episode_revenues)` if `collect_metrics` is
        True, otherwise None.
    """
    use_agent = agent is not None
    if not use_agent and policy_fn is None:
        policy_fn = random_policy

    episode_rewards: List[float] = []
    episode_revenues: List[float] = []

    for episode in range(1, config.num_episodes + 1):
        observation, _info = env.reset(seed=config.seed + episode)
        terminated = truncated = False
        episode_reward = 0.0
        steps = 0
        info = {}

        while not (terminated or truncated):
            if use_agent:
                action = agent.select_action(observation)
            else:
                action = policy_fn(observation, env)

            next_observation, reward, terminated, truncated, info = env.step(action)

            if use_agent:
                agent.update(
                    observation, action, reward, next_observation,
                    terminated or truncated,
                )

            observation = next_observation
            episode_reward += reward
            steps += 1

            if (
                config.max_steps_per_episode is not None
                and steps >= config.max_steps_per_episode
            ):
                truncated = True

        if use_agent:
            agent.decay_exploration()

        episode_rewards.append(episode_reward)
        episode_revenues.append(info.get("episode_revenue", 0.0))

        if episode % config.log_every_n_episodes == 0 or episode == 1:
            logger.info(
                "Episode %d/%d | steps=%d | reward=%.2f | revenue=$%.2f",
                episode,
                config.num_episodes,
                steps,
                episode_reward,
                info.get("episode_revenue", 0.0),
            )

    logger.info("Training loop complete: %d episodes run.", config.num_episodes)

    if collect_metrics:
        return episode_rewards, episode_revenues
    return None


# --------------------------------------------------------------------------- #
# Policy saving and evaluation
# --------------------------------------------------------------------------- #
def save_policy(agent: QLearningAgent, config: TrainingConfig) -> Path:
    """Save `agent`'s Q-table under `config.checkpoint_dir`."""
    checkpoint_path = Path(config.checkpoint_dir) / f"{config.agent_type}_policy.pkl"
    agent.save(checkpoint_path)
    return checkpoint_path


def evaluate_agent(
    env: PricingEnvironment,
    agent: QLearningAgent,
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
            action = agent.select_greedy_action(observation)
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
        description="Train an agent against PricingEnvironment."
    )
    parser.add_argument(
        "--episodes", type=int, default=None, help="Override TrainingConfig.num_episodes."
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Override TrainingConfig.seed."
    )
    parser.add_argument(
        "--agent",
        type=str,
        default=None,
        choices=["random", "q_learning"],
        help="Override TrainingConfig.agent_type. 'random' runs the "
        "placeholder policy only (no learning, no save/reload/evaluate).",
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
    overrides = {}
    if args.episodes is not None:
        overrides["num_episodes"] = args.episodes
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.agent is not None:
        overrides["agent_type"] = args.agent
    if overrides:
        config = replace(config, **overrides)

    logger.info(
        "Loaded TrainingConfig | agent_type=%s | num_episodes=%d | seed=%d | "
        "lr=%.3f | gamma=%.3f | epsilon=%.3f->%.3f",
        config.agent_type,
        config.num_episodes,
        config.seed,
        config.learning_rate,
        config.discount_factor,
        config.exploration_rate,
        config.exploration_min,
    )

    env = build_environment(config.env_config)

    if not args.skip_verification:
        verify_environment_compatibility(env)

    try:
        if config.agent_type == "random":
            # Placeholder-policy run: no learning happens, so there is
            # nothing to save, reload, or evaluate - just run and log.
            run_training(env, config, policy_fn=random_policy)
            logger.info("Random-policy run complete (no agent to save/evaluate).")
        else:
            agent = build_agent(config, env)
            run_training(env, config, agent=agent)

            checkpoint_path = save_policy(agent, config)
            logger.info("Saved trained agent to %s", checkpoint_path)

            # Reload from disk into a FRESH agent instance -- proves the
            # save/load round-trip actually works, not just that a file
            # got written.
            reloaded_agent = QLearningAgent.load(
                checkpoint_path, action_space=env.action_space, seed=config.seed
            )

            eval_summary = evaluate_agent(
                env, reloaded_agent, num_episodes=config.num_eval_episodes
            )
            logger.info(
                "Evaluation of reloaded policy (%d episodes, greedy) | "
                "mean_reward=%.2f | mean_revenue=$%.2f | std_revenue=$%.2f",
                eval_summary["num_eval_episodes"],
                eval_summary["mean_reward"],
                eval_summary["mean_revenue"],
                eval_summary["std_revenue"],
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()