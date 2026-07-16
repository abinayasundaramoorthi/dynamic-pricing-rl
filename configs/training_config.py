"""
training_config.py

Single source of truth for "what does one training run consist of".

Design intent
-------------
Mirrors the pattern already established by `PricingEnvConfig` in
`pricing_env/pricing_env.py`: one immutable dataclass that fully specifies
an experimental condition, so a run is reproducible from this object (or an
equivalent YAML/JSON dict) alone.

`TrainingConfig` deliberately composes a `PricingEnvConfig` rather than
duplicating environment fields (inventory, horizon, base_price, ...) as
loose top-level attributes. That would create two places that could drift
out of sync (e.g. someone changes `initial_inventory` in the env config for
an experiment but forgets to also change it here). Composing the real
`PricingEnvConfig` object means there is exactly one definition of the
environment's parameters, and `TrainingConfig` only owns what is genuinely
*training*-specific: episode budget, agent hyperparameters, seeding, and
logging/checkpoint cadence.

This file has no dependency on Gymnasium, PyTorch, or any specific agent
implementation — it is a pure config module, importable by
`training/train_agent.py`, by future baseline/Q-Learning agent modules, and
by the Week 4 evaluation harness, without pulling in training machinery.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from pricing_env import PricingEnvConfig


@dataclass(frozen=True)
class TrainingConfig:
    """
    Immutable configuration for one training run.

    Attributes
    ----------
    env_config : PricingEnvConfig
        Full specification of the environment this run trains against.
        Defaults to `PricingEnvConfig()` (Phase-1 defaults: 100 units,
        30-day horizon, $200 base price) so `TrainingConfig()` alone is
        enough to get a runnable configuration with no further setup —
        matching the "Configuration file loads correctly" acceptance
        criterion without requiring the caller to hand-build an env config
        first.
    num_episodes : int
        Total number of training episodes to run. Must be > 0.
    max_steps_per_episode : Optional[int]
        Safety cap on steps per episode, independent of the environment's
        own termination condition (inventory sold out / deadline reached).
        `None` means "no artificial cap — rely entirely on the
        environment's terminated/truncated signals". Present mainly as a
        defensive guard against a misconfigured environment that could
        otherwise never terminate, so a single episode cannot hang the
        entire training run.
    learning_rate : float
        Step size for the agent's value/policy update. Must be > 0.
    discount_factor : float
        Gamma in the Bellman equation — how much the agent values future
        reward relative to immediate reward. Must be in (0, 1]. Kept here
        (not in `PricingEnvConfig`) because discounting is a property of
        the *agent's* objective, not of the environment/MDP transition
        dynamics themselves.
    epsilon_start, epsilon_end, epsilon_decay : float
        Standard epsilon-greedy exploration schedule for the tabular
        Q-Learning / DQN agents landing later in Week 2. `epsilon_start`
        is the initial exploration rate, `epsilon_end` the floor it decays
        towards, `epsilon_decay` the per-episode multiplicative decay
        factor. Included now (even though no agent consumes them yet) so
        the config's shape does not need to change again once the
        Q-Learning agent is integrated in the next task.
    seed : int
        Seed passed to `PricingEnvironment.reset(seed=...)`. Fixing this
        here (rather than leaving it to whichever script happens to call
        reset) is what makes a training run reproducible end to end from
        `TrainingConfig` alone.
    log_every_n_episodes : int
        How often (in episodes) to emit a training-progress log line.
        Must be > 0.
    checkpoint_dir : Optional[str]
        Directory to write agent checkpoints to. `None` disables
        checkpointing — useful for the Day 1 smoke-run, where there is no
        trainable agent yet to checkpoint.
    """

    env_config: PricingEnvConfig = field(default_factory=PricingEnvConfig)

    num_episodes: int = 1000
    max_steps_per_episode: Optional[int] = None

    learning_rate: float = 0.1
    discount_factor: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995

    seed: int = 42
    log_every_n_episodes: int = 50
    checkpoint_dir: Optional[str] = "checkpoints"

    def __post_init__(self) -> None:
        if not isinstance(self.env_config, PricingEnvConfig):
            raise TypeError(
                f"env_config must be a PricingEnvConfig instance, got {type(self.env_config)}"
            )
        if self.num_episodes <= 0:
            raise ValueError(f"num_episodes must be > 0, got {self.num_episodes}")
        if self.max_steps_per_episode is not None and self.max_steps_per_episode <= 0:
            raise ValueError(
                f"max_steps_per_episode must be > 0 or None, got {self.max_steps_per_episode}"
            )
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}")
        if not (0.0 < self.discount_factor <= 1.0):
            raise ValueError(
                f"discount_factor must be in (0, 1], got {self.discount_factor}"
            )
        if not (0.0 <= self.epsilon_end <= self.epsilon_start <= 1.0):
            raise ValueError(
                "Require 0 <= epsilon_end <= epsilon_start <= 1, got "
                f"epsilon_start={self.epsilon_start}, epsilon_end={self.epsilon_end}"
            )
        if not (0.0 < self.epsilon_decay <= 1.0):
            raise ValueError(
                f"epsilon_decay must be in (0, 1], got {self.epsilon_decay}"
            )
        if self.log_every_n_episodes <= 0:
            raise ValueError(
                f"log_every_n_episodes must be > 0, got {self.log_every_n_episodes}"
            )


def get_default_training_config() -> TrainingConfig:
    """
    Return the default training configuration.

    Kept as a function (rather than having callers just do
    `TrainingConfig()` directly) so this is the one place a future
    experiment-tracking / CLI-args layer needs to hook into to build a
    config from a YAML file or command-line flags instead, without every
    call site needing to change.
    """
    return TrainingConfig()