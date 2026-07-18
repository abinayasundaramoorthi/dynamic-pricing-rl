"""
training_config.py

Centralized training configuration for the Week 2+ training pipeline
(issue #42 — "Integrate Training Workflow with Agent and Environment").

Defines a single, immutable `TrainingConfig` that fully specifies one
training run: which environment to build (`env_config`), how long to train
(`num_episodes`, `max_steps_per_episode`), and the RL hyperparameters
(`learning_rate`, `discount_factor`, `exploration_rate`, ...). Centralizing
these here — rather than scattering constants across `train_agent.py` — is
what lets the evaluation harness and any future notebook reconstruct the
exact configuration a given training run used, from one object.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from pricing_env import PricingEnvConfig
from pricing_env.demand_simulator import DemandConfig
from pricing_env.reward import RewardConfig


@dataclass(frozen=True)
class TrainingConfig:
    """
    Immutable configuration for a single training run.

    Frozen for the same reason `PricingEnvConfig` is frozen: a run's
    configuration must not be mutable mid-experiment, or results become
    irreproducible. `train_agent.py`'s CLI overrides (--episodes, --seed)
    are applied via `dataclasses.replace()`, which builds a *new* config
    rather than mutating this one — see `train_agent.main()`.

    Attributes
    ----------
    env_config : PricingEnvConfig
        Full configuration for the `PricingEnvironment` this run trains
        against (inventory, horizon, base price, demand model, reward
        weights). Nested here (rather than duplicating individual env
        fields on TrainingConfig) so there is exactly one place that
        defines "what environment did this experiment use."
    num_episodes : int
        Number of episodes to run. Must be > 0.
    seed : int
        Base seed for reproducibility. `train_agent.py` seeds each episode
        as `seed + episode_number`, so this one value determines the
        entire run's random sequence.
    max_steps_per_episode : Optional[int]
        Defensive step cap per episode, independent of the environment's
        own inventory/deadline termination. `None` (default) means no
        artificial cap is applied — the environment's own termination
        logic is the only thing that ends an episode. Exists purely as a
        safety net against an unforeseen infinite-loop bug in a future
        agent, not as a normal part of the MDP.
    log_every_n_episodes : int
        How often `train_agent.py` logs progress. Must be > 0.
    learning_rate : float
        Step size for value/parameter updates. Consumed starting with the
        Q-Learning agent (tabular Q-table updates) and later DQN (optimizer
        learning rate). Must be in (0, 1].
    discount_factor : float
        Gamma — how much future reward is weighted against immediate
        reward. Kept close to 1 (default 0.99) so an agent reasons about
        the *entire* remaining selling season, not just the next sale,
        consistent with the MDP formulation in the design doc (Section 6).
        Must be in [0, 1).
    exploration_rate : float
        Starting epsilon for epsilon-greedy exploration. Not yet consumed
        by the current random-policy placeholder in `train_agent.py` (see
        that module's docstring) — defined here now so the upcoming
        Q-Learning/DQN agent can read it without a further config change.
        Must be in [exploration_min, 1.0].
    exploration_min : float
        Floor epsilon never decayed below, so the agent always retains
        some exploration. Must be in [0, exploration_rate].
    exploration_decay : float
        Per-episode multiplicative decay applied to the exploration rate
        by the agent that consumes it (not by this config or by
        train_agent.py itself). Must be in (0, 1].
    checkpoint_dir : str
        Where a learning agent should save trained weights/Q-tables.
        Unused by the current placeholder loop (there is nothing trainable
        to checkpoint yet) — defined now so the training loop only needs a
        few added lines, not a reshaped config, once a real agent lands.
    results_dir : str
        Where evaluation results/plots should be written (Week 4).
    """

    env_config: PricingEnvConfig = field(default_factory=PricingEnvConfig)

    num_episodes: int = 1000
    seed: int = 42
    max_steps_per_episode: Optional[int] = None
    log_every_n_episodes: int = 50

    learning_rate: float = 0.1
    discount_factor: float = 0.99
    exploration_rate: float = 1.0
    exploration_min: float = 0.05
    exploration_decay: float = 0.995

    checkpoint_dir: str = "agents/checkpoints"
    results_dir: str = "evaluation/results"

    def __post_init__(self) -> None:
        if not isinstance(self.env_config, PricingEnvConfig):
            raise TypeError(
                f"env_config must be a PricingEnvConfig instance, got {type(self.env_config)}"
            )
        if self.num_episodes <= 0:
            raise ValueError(f"num_episodes must be > 0, got {self.num_episodes}")
        if self.max_steps_per_episode is not None and self.max_steps_per_episode <= 0:
            raise ValueError(
                "max_steps_per_episode must be > 0 if set, got "
                f"{self.max_steps_per_episode}"
            )
        if self.log_every_n_episodes <= 0:
            raise ValueError(
                f"log_every_n_episodes must be > 0, got {self.log_every_n_episodes}"
            )
        if not (0.0 < self.learning_rate <= 1.0):
            raise ValueError(
                f"learning_rate must be in (0, 1], got {self.learning_rate}"
            )
        if not (0.0 <= self.discount_factor < 1.0):
            raise ValueError(
                f"discount_factor must be in [0, 1), got {self.discount_factor}"
            )
        if not (0.0 <= self.exploration_min <= self.exploration_rate <= 1.0):
            raise ValueError(
                "Require 0 <= exploration_min <= exploration_rate <= 1, got "
                f"exploration_rate={self.exploration_rate}, "
                f"exploration_min={self.exploration_min}"
            )
        if not (0.0 < self.exploration_decay <= 1.0):
            raise ValueError(
                f"exploration_decay must be in (0, 1], got {self.exploration_decay}"
            )


def get_default_training_config() -> TrainingConfig:
    """
    Build the default `TrainingConfig` for this project.

    Kept as a function (rather than every call site writing
    `TrainingConfig()` directly) for two reasons:

    1. `train_agent.py`, the evaluation harness, and any future
       notebook all share one canonical definition of "what a default
       run looks like," instead of each hardcoding its own
       `PricingEnvConfig`.
    2. This is the one place a future upgrade — loading overrides from a
       YAML/JSON file or environment variables — gets wired in, without
       touching any call site that already depends on this function.
    """
    return TrainingConfig(
        env_config=PricingEnvConfig(
            initial_inventory=100,
            selling_horizon_days=30,
            base_price=200.0,
            demand=DemandConfig(),
            reward=RewardConfig(),
        )
    )