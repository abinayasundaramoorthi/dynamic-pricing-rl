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
Scope of this deliverable
--------------------------------------------------------------------------
Week 2, Day 1 (issue #38 / #42) wired the training *pipeline* together:
loading a `TrainingConfig`, building a `PricingEnvironment` from it, and
verifying compatibility (Gymnasium API compliance + a clean `reset()`).
That version deliberately shipped without a trainable agent — actions came
from an explicit, swappable `policy_fn` defaulting to `random_policy` — so
the pipeline was runnable end-to-end while making it unmistakable, in the
code and the logs, that no learning was happening yet.

Week 2, Day 4 ("Integrate optimized agent configuration into a stable
training pipeline") replaces that placeholder for real: when
`config.agent_type == "q_learning"`, this script now builds a
`QLearningAgent` (agents/q_learning.py), trains it via `run_training()`,
saves the resulting Q-table to disk, reloads it from disk into a fresh
agent instance (proving the save/load round-trip actually works, not just
that the in-memory object still functions), and evaluates that reloaded
policy greedily (no exploration) on held-out episode seeds.

`policy_fn` and `random_policy` are UNCHANGED and still fully supported —
`agent_type == "random"` still takes that exact code path, with no
saving/evaluation performed (there is nothing trainable to persist). This
keeps `training/run_experiment.py`, which calls `build_environment()`,
`random_policy()`, and `verify_environment_compatibility()` directly and
does not pass an `agent`, working exactly as before with zero changes on
its end.

Usage
-----
    python -m training.train_agent
    python -m training.train_agent --episodes 200 --seed 7
    python -m training.train_agent --agent q_learning --episodes 5000
"""

from __future__ import annotations

import argparse
import logging
from typing import Callable, Optional

import gymnasium as gym
import numpy as np
from gymnasium.utils.env_checker import check_env

from configs.training_config import TrainingConfig, get_default_training_config
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from gymnasium.utils.env_checker import check_env

from agents.q_learning import QLearningAgent
from configs.training_config import TrainingConfig, get_final_training_config
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
    agent: Optional[QLearningAgent] = None,
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
        Maps an observation to an action index, with NO learning. Used
        only when `agent` is not given. Defaults to `random_policy`.
    agent : QLearningAgent, optional
        If given, takes priority over `policy_fn`: the agent's
        `select_action()` chooses each action, `update()` is called after
        every step so it can learn from the transition, and
        `decay_exploration()` is called once per completed episode. This
        is the only change from the Day 1 placeholder loop — the loop
        shape itself (reset -> step -> log) is unchanged, exactly as that
        version's docstring anticipated.

    Notes
    -----
    Policy saving is NOT done inside this function — `main()` calls
    `save_policy()` separately after `run_training()` returns. Keeping
    "run episodes" and "persist the result" as separate steps means this
    function stays reusable by callers (like `run_experiment.py`'s own
    episode loop, conceptually) that don't want every call to touch disk.
    """
    use_agent = agent is not None
    if not use_agent and policy_fn is None:
        policy_fn = lambda obs: random_policy(obs, env)  # noqa: E731

    for episode in range(1, config.num_episodes + 1):
        observation, _info = env.reset(seed=config.seed + episode)
        terminated = truncated = False
        episode_reward = 0.0
        steps = 0

        while not (terminated or truncated):
            action = policy_fn(observation)
            observation, reward, terminated, truncated, info = env.step(action)
        info = {}

        while not (terminated or truncated):
            if use_agent:
                action = agent.select_action(observation)
            else:
                action = policy_fn(observation)

            next_observation, reward, terminated, truncated, info = env.step(action)

            if use_agent:
                agent.update(observation, action, reward, next_observation, terminated)

            observation = next_observation
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
        if use_agent:
            agent.decay_exploration()

        if episode % config.log_every_n_episodes == 0 or episode == 1:
            log_msg = (
                "Episode %d/%d | steps=%d | episode_reward=%.2f | "
                "final_revenue=$%.2f"
            )
            log_args = [
                episode,
                config.num_episodes,
                steps,
                episode_reward,
                info.get("episode_revenue", 0.0),
            )
            ]
            if use_agent:
                log_msg += " | exploration_rate=%.3f | states_visited=%d"
                log_args += [agent.exploration_rate, len(agent.q_table)]
            logger.info(log_msg, *log_args)

    logger.info("Training loop complete: %d episodes run.", config.num_episodes)


# --------------------------------------------------------------------------- #
# Agent construction, policy saving, and evaluation (Day 4)
# --------------------------------------------------------------------------- #
def build_agent(config: TrainingConfig, env: PricingEnvironment) -> Optional[QLearningAgent]:
    """
    Construct the agent named by `config.agent_type`, or `None` for
    `"random"` (the Day 1 placeholder path, which has no agent object —
    `run_training()` falls back to `random_policy` when `agent is None`).
    """
    if config.agent_type == "random":
        return None
    if config.agent_type == "q_learning":
        return QLearningAgent(
            action_space=env.action_space,
            learning_rate=config.learning_rate,
            discount_factor=config.discount_factor,
            exploration_rate=config.exploration_rate,
            exploration_min=config.exploration_min,
            exploration_decay=config.exploration_decay,
            seed=config.seed,
        )
    # TrainingConfig.__post_init__ already restricts agent_type to known
    # values, so reaching here would mean this function and the config
    # validation have drifted out of sync — fail loudly rather than
    # silently falling back to the random policy.
    raise ValueError(f"build_agent() has no case for agent_type={config.agent_type!r}")


def save_policy(agent: QLearningAgent, config: TrainingConfig) -> Path:
    """
    Save `agent`'s learned policy under `config.checkpoint_dir`.

    Returns the path written to, so callers (and `main()`'s subsequent
    reload-and-evaluate step) don't need to reconstruct it independently.
    """
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
    exploration, no further learning) and return summary statistics.

    `seed_offset` shifts evaluation episode seeds well clear of the seed
    range training used (`config.seed + episode`, episode in
    [1, num_episodes]) — evaluating on the exact seeds the policy trained
    against would risk measuring memorization of those specific random
    draws rather than a policy that generalizes across the demand
    distribution.
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
        "--agent",
        type=str,
        default=None,
        choices=["random", "q_learning"],
        help="Override TrainingConfig.agent_type.",
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
    config = get_final_training_config()
    if args.episodes is not None or args.seed is not None or args.agent is not None:
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
        if args.agent is not None:
            overrides["agent_type"] = args.agent
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

    env = build_environment(config)

    if not args.skip_verification:
        verify_environment_compatibility(env)

    try:
        run_training(env, config)
    agent = build_agent(config, env)

    try:
        run_training(env, config, agent=agent)

        if agent is None:
            # agent_type == "random": nothing trainable to save or
            # evaluate, matching the Day 1 placeholder's documented
            # behavior exactly.
            logger.info(
                "agent_type='random': no policy to save or evaluate "
                "(this run exercised the environment only)."
            )
            return

        checkpoint_path = save_policy(agent, config)

        # Reload from disk into a FRESH agent instance, rather than just
        # reusing the in-memory `agent` object, deliberately: this is what
        # actually proves the save/load round-trip works — evaluating the
        # in-memory agent would tell us nothing about whether `save()`
        # wrote a usable file, only that training itself ran. This is the
        # concrete thing "evaluation can use the trained policy" means.
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