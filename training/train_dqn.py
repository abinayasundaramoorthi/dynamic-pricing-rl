"""
train_dqn.py

DQN training entry point for the Travel & Hospitality Dynamic Pricing
project (Week 3, issue #62 — "Design and Integrate the DQN Training
Architecture").

Mirrors `training/train_agent.py`'s structure deliberately (build
environment -> verify compatibility -> build agent -> train -> save ->
reload from disk -> evaluate) so anyone already familiar with that
pipeline can read this one quickly. The training LOOP differs from
`train_agent.run_training()`, though: DQN needs `agent.remember()` to
store each transition and `agent.train_step()` to periodically sample a
minibatch and update the network, neither of which the tabular agent's
single-transition `update()` contract expresses. That difference is why
this is a separate entry point rather than another branch inside
`train_agent.build_agent()` — see `reports/dqn_architecture.md` for the
full rationale.

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
from gymnasium.utils.env_checker import check_env

from agents.dqn_agent import DQNAgent
from configs.dqn_config import DQNConfig, get_default_dqn_config
from pricing_env import PricingEnvironment

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Environment construction (same shape as train_agent.build_environment)
# --------------------------------------------------------------------------- #
def build_environment(config: DQNConfig) -> PricingEnvironment:
    """Construct the `PricingEnvironment` for this DQN run."""
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

    Identical in spirit to `train_agent.verify_environment_compatibility()`
    — same two checks (`check_env`, then a real `reset()`), same fail-loud
    philosophy: a DQN run must never start against an environment that
    fails this check, since malformed observations would silently corrupt
    every downstream gradient step, not just produce one bad episode.
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
# Agent construction
# --------------------------------------------------------------------------- #
def build_agent(config: DQNConfig, env: PricingEnvironment) -> DQNAgent:
    """Construct a `DQNAgent` sized to `env`'s observation/action spaces."""
    return DQNAgent(
        observation_space=env.observation_space,
        action_space=env.action_space,
        hidden_layer_sizes=config.hidden_layer_sizes,
        learning_rate=config.learning_rate,
        discount_factor=config.discount_factor,
        exploration_rate=config.exploration_rate,
        exploration_min=config.exploration_min,
        exploration_decay=config.exploration_decay,
        batch_size=config.batch_size,
        replay_buffer_size=config.replay_buffer_size,
        min_replay_size_before_training=config.min_replay_size_before_training,
        target_update_frequency=config.target_update_frequency,
        grad_clip_norm=config.grad_clip_norm,
        device=config.device,
        seed=config.seed,
    )


# --------------------------------------------------------------------------- #
# Training loop
# --------------------------------------------------------------------------- #
def run_training(
    env: PricingEnvironment, config: DQNConfig, agent: DQNAgent
) -> Tuple[List[float], List[float]]:
    """
    Run DQN training for `config.num_episodes` episodes.

    Per environment step: select an action (epsilon-greedy), step the
    environment, store the transition (`agent.remember()`), then perform
    `config.gradient_steps_per_env_step` minibatch updates
    (`agent.train_step()` — a no-op returning `None` until the replay
    buffer clears its warm-up threshold). Exploration decays once per
    completed episode, matching the tabular agent's cadence.

    Returns
    -------
    Tuple[List[float], List[float]]
        Per-episode (rewards, revenues), for the caller to summarize or
        persist — same "return metrics, don't touch disk here" separation
        of concerns as `train_agent.run_training(collect_metrics=True)`.
    """
    episode_rewards: List[float] = []
    episode_revenues: List[float] = []
    recent_losses: List[float] = []

    for episode in range(1, config.num_episodes + 1):
        observation, _info = env.reset(seed=config.seed + episode)
        terminated = truncated = False
        episode_reward = 0.0
        steps = 0
        info = {}

        while not (terminated or truncated):
            action = agent.select_action(observation)
            next_observation, reward, terminated, truncated, info = env.step(action)
            agent.remember(observation, action, reward, next_observation, terminated)

            for _ in range(config.gradient_steps_per_env_step):
                loss = agent.train_step()
                if loss is not None:
                    recent_losses.append(loss)

            observation = next_observation
            episode_reward += reward
            steps += 1

            if (
                config.max_steps_per_episode is not None
                and steps >= config.max_steps_per_episode
            ):
                truncated = True

        agent.decay_exploration()
        episode_rewards.append(episode_reward)
        episode_revenues.append(info.get("episode_revenue", 0.0))

        if episode % config.log_every_n_episodes == 0 or episode == 1:
            avg_loss = float(np.mean(recent_losses[-200:])) if recent_losses else float("nan")
            logger.info(
                "Episode %d/%d | steps=%d | reward=%.2f | revenue=$%.2f | "
                "epsilon=%.3f | avg_loss=%.4f | buffer=%d/%d",
                episode,
                config.num_episodes,
                steps,
                episode_reward,
                info.get("episode_revenue", 0.0),
                agent.exploration_rate,
                avg_loss,
                len(agent.replay_buffer),
                config.replay_buffer_size,
            )

    logger.info("DQN training loop complete: %d episodes run.", config.num_episodes)
    return episode_rewards, episode_revenues


# --------------------------------------------------------------------------- #
# Policy saving and evaluation
# --------------------------------------------------------------------------- #
def save_policy(agent: DQNAgent, config: DQNConfig) -> Path:
    """Save `agent`'s network weights under `config.checkpoint_dir`."""
    checkpoint_path = Path(config.checkpoint_dir) / "dqn_policy.pt"
    agent.save(checkpoint_path)
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
    range (`config.seed + episode`), matching `train_agent.evaluate_agent`'s
    rationale exactly: evaluating on the same seeds a policy trained
    against risks measuring memorization rather than generalization.
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
        description="Train a DQN agent against PricingEnvironment."
    )
    parser.add_argument(
        "--episodes", type=int, default=None, help="Override DQNConfig.num_episodes."
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Override DQNConfig.seed."
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda"],
        help="Override DQNConfig.device.",
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
    if args.episodes is not None or args.seed is not None or args.device is not None:
        overrides = {}
        if args.episodes is not None:
            overrides["num_episodes"] = args.episodes
        if args.seed is not None:
            overrides["seed"] = args.seed
        if args.device is not None:
            overrides["device"] = args.device
        config = replace(config, **overrides)

    logger.info(
        "Loaded DQNConfig | num_episodes=%d | seed=%d | lr=%.4f | gamma=%.3f | "
        "batch_size=%d | replay_buffer_size=%d | device=%s",
        config.num_episodes,
        config.seed,
        config.learning_rate,
        config.discount_factor,
        config.batch_size,
        config.replay_buffer_size,
        config.device,
    )

    env = build_environment(config)

    if not args.skip_verification:
        verify_environment_compatibility(env)

    agent = build_agent(config, env)

    try:
        run_training(env, config, agent)

        checkpoint_path = save_policy(agent, config)

        # Reload from disk into a FRESH agent instance -- proves the
        # save/load round-trip actually works, same rationale as
        # train_agent.main()'s reload-before-evaluate step.
        reloaded_agent = DQNAgent.load(
            checkpoint_path,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=config.device,
            seed=config.seed,
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