"""
env_utils.py

Shared environment construction and compatibility verification, used by
BOTH `train_agent.py` (Q-Learning) and `train_dqn.py` (DQN).

Week 3 Day 5 refactor (issue #82): `build_environment()` and
`verify_environment_compatibility()` were previously duplicated verbatim
in both training entry points — identical logic, just type-hinted against
`TrainingConfig` in one file and `DQNConfig` in the other. Since both
config classes carry a `.env_config: PricingEnvConfig` attribute, the
actual logic never depended on which config wrapped it. Extracting it
here means there is now exactly one place this logic is implemented;
a future third training entry point (e.g. a PPO/SAC pipeline named in the
project roadmap) reuses this directly instead of copying it a third time.
"""

from __future__ import annotations

import logging

from gymnasium.utils.env_checker import check_env

from pricing_env import PricingEnvConfig, PricingEnvironment

logger = logging.getLogger(__name__)


def build_environment(env_config: PricingEnvConfig) -> PricingEnvironment:
    """
    Construct the `PricingEnvironment` for a training run.

    Takes a `PricingEnvConfig` directly (rather than a `TrainingConfig` or
    `DQNConfig`) so this function has no dependency on either agent-specific
    config class — callers pass `config.env_config`.
    """
    env = PricingEnvironment(env_config)
    logger.info(
        "Environment constructed | inventory=%d | horizon=%d days | "
        "base_price=₹%.2f | actions=%d",
        env_config.initial_inventory,
        env_config.selling_horizon_days,
        env_config.base_price,
        env.action_space.n,
    )
    return env


def verify_environment_compatibility(env: PricingEnvironment) -> None:
    """
    Verify the environment is Gymnasium-API-compliant and resets cleanly.

    Two checks:
      1. `check_env` — Gymnasium's own compliance checker. Catches
         malformed observation/action spaces, incorrect `step()`/`reset()`
         return shapes, etc. Running this here (once, at pipeline-startup
         time) means any future accidental regression in `pricing_env.py`
         fails loudly during pipeline setup, rather than surfacing as a
         confusing shape-mismatch deep inside an agent's training loop —
         and this applies equally whether the agent behind it is a
         Q-table or a neural network.
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
        later (mirrors the fail-loud philosophy used throughout
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