"""
dqn_config.py

Configuration for the Deep Q-Network (DQN) training pipeline
(Week 3, issue #62 — "Design and Integrate the DQN Training Architecture").

Deliberately mirrors the structure of configs/training_config.py: same
frozen-dataclass style, same "one object fully specifies one run"
philosophy, same nested PricingEnvConfig composition — anyone already
familiar with TrainingConfig can read this in a few seconds.

DQNConfig is a SEPARATE dataclass from TrainingConfig, not a subclass or
shared base, on purpose: Q-Learning and DQN have meaningfully different
parameter sets. DQN needs network architecture, replay buffer size, batch
size, and target-update frequency; tabular Q-Learning needs none of that.
Forcing both into one shared config would mean each agent type carries
fields that are meaningless for it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from pricing_env import PricingEnvConfig
from pricing_env.demand_simulator import DemandConfig
from pricing_env.reward import RewardConfig

_ALLOWED_DEVICES = {"cpu", "cuda"}


@dataclass(frozen=True)
class DQNConfig:
    """
    Immutable configuration for a single DQN training run.

    Attributes
    ----------
    env_config : PricingEnvConfig
        Full environment configuration this run trains against. Nested
        here for the same reason as in TrainingConfig — one object fully
        defines "what environment did this experiment use."
    num_episodes : int
        Number of training episodes. Must be > 0. Higher than the tabular
        agent's default (2000 vs. 5000 for Q-Learning is comparable in
        practice — DQN needs enough environment steps to fill the replay
        buffer past its warm-up threshold before learning even begins).
    seed : int
        Base seed for reproducibility; episodes are seeded as
        `seed + episode_number`, matching `train_agent.py`'s convention.
    max_steps_per_episode : Optional[int]
        Defensive step cap, independent of the environment's own
        termination logic. `None` means no artificial cap.
    log_every_n_episodes : int
        Logging cadence. Must be > 0.
    hidden_layer_sizes : List[int]
        Width of each hidden layer in the Q-network's MLP, in order.
        Default [64, 64] — see agents/dqn_agent.py's QNetwork docstring
        for why a small network is the right choice for this 2-dimensional
        state space.
    learning_rate : float
        Adam optimizer learning rate. Deliberately a different scale
        (1e-3) than the tabular agent's `learning_rate` (0.1) — these are
        not the same kind of parameter: one is a Q-table update step size,
        the other is a neural-network gradient-descent step size. Must be
        in (0, 1].
    discount_factor : float
        Gamma. Same meaning and same default (0.99) as the tabular agent,
        for the same reason (Section 6 of the design doc: reason about the
        entire remaining season). Must be in [0, 1).
    exploration_rate, exploration_min, exploration_decay : float
        Same epsilon-greedy scheme as the tabular agent, same defaults,
        for direct comparability between the two agents' exploration
        behavior.
    batch_size : int
        Minibatch size sampled from replay for each gradient step. Must
        be > 0.
    replay_buffer_size : int
        Maximum transitions retained in the replay buffer before the
        oldest are evicted. Must be > 0.
    min_replay_size_before_training : int
        Minimum transitions required in the buffer before any gradient
        step is taken (a warm-up period) — training on a near-empty,
        low-diversity buffer is a well-documented source of early
        instability. Must be >= batch_size.
    target_update_frequency : int
        How many gradient steps between syncing the target network from
        the online network (Mnih et al., 2015). Counted in gradient
        steps, not episodes or environment steps, since that's the unit
        target-network staleness is actually measured against. Must be > 0.
    gradient_steps_per_env_step : int
        How many replay-buffer minibatch updates to perform per
        environment step taken. 1 is standard; > 1 trades more compute
        for potentially faster learning per environment interaction.
        Must be > 0.
    grad_clip_norm : Optional[float]
        Maximum gradient norm (clipped via `torch.nn.utils.clip_grad_norm_`)
        to guard against occasional large TD-error gradients destabilizing
        the network. `None` disables clipping. Must be > 0 if set.
    checkpoint_dir : str
        Where the trained network's weights are saved.
    results_dir : str
        Where evaluation results are written.
    num_eval_episodes : int
        Episodes run, greedily, when evaluating a trained policy. Must be > 0.
    device : str
        One of {"cpu", "cuda"}. Validated here as a plain string (this
        module intentionally has no torch dependency, mirroring
        demand_simulator.py's "no Gymnasium dependency" pattern for
        independent testability) — actual device availability is resolved
        by `training/train_dqn.py`, which does import torch.
    """

    env_config: PricingEnvConfig = field(default_factory=PricingEnvConfig)

    num_episodes: int = 2000
    seed: int = 42
    max_steps_per_episode: Optional[int] = None
    log_every_n_episodes: int = 50

    hidden_layer_sizes: List[int] = field(default_factory=lambda: [64, 64])

    learning_rate: float = 1e-3
    discount_factor: float = 0.99
    exploration_rate: float = 1.0
    exploration_min: float = 0.05
    exploration_decay: float = 0.995

    batch_size: int = 64
    replay_buffer_size: int = 50_000
    min_replay_size_before_training: int = 1_000
    target_update_frequency: int = 500
    gradient_steps_per_env_step: int = 1
    grad_clip_norm: Optional[float] = 10.0

    checkpoint_dir: str = "agents/checkpoints"
    results_dir: str = "evaluation/results"
    num_eval_episodes: int = 200

    device: str = "cpu"

    def __post_init__(self) -> None:
        if not isinstance(self.env_config, PricingEnvConfig):
            raise TypeError(
                f"env_config must be a PricingEnvConfig instance, got {type(self.env_config)}"
            )
        if self.num_episodes <= 0:
            raise ValueError(f"num_episodes must be > 0, got {self.num_episodes}")
        if self.max_steps_per_episode is not None and self.max_steps_per_episode <= 0:
            raise ValueError(
                f"max_steps_per_episode must be > 0 if set, got {self.max_steps_per_episode}"
            )
        if self.log_every_n_episodes <= 0:
            raise ValueError(
                f"log_every_n_episodes must be > 0, got {self.log_every_n_episodes}"
            )
        if len(self.hidden_layer_sizes) == 0 or any(
            h <= 0 for h in self.hidden_layer_sizes
        ):
            raise ValueError(
                f"hidden_layer_sizes must be a non-empty list of positive "
                f"integers, got {self.hidden_layer_sizes}"
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
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {self.batch_size}")
        if self.replay_buffer_size <= 0:
            raise ValueError(
                f"replay_buffer_size must be > 0, got {self.replay_buffer_size}"
            )
        if self.min_replay_size_before_training < self.batch_size:
            raise ValueError(
                "min_replay_size_before_training must be >= batch_size, got "
                f"min_replay_size_before_training={self.min_replay_size_before_training}, "
                f"batch_size={self.batch_size}"
            )
        if self.target_update_frequency <= 0:
            raise ValueError(
                f"target_update_frequency must be > 0, got {self.target_update_frequency}"
            )
        if self.gradient_steps_per_env_step <= 0:
            raise ValueError(
                "gradient_steps_per_env_step must be > 0, got "
                f"{self.gradient_steps_per_env_step}"
            )
        if self.grad_clip_norm is not None and self.grad_clip_norm <= 0:
            raise ValueError(
                f"grad_clip_norm must be > 0 if set, got {self.grad_clip_norm}"
            )
        if self.num_eval_episodes <= 0:
            raise ValueError(
                f"num_eval_episodes must be > 0, got {self.num_eval_episodes}"
            )
        if self.device not in _ALLOWED_DEVICES:
            raise ValueError(
                f"device must be one of {sorted(_ALLOWED_DEVICES)}, got {self.device!r}"
            )


def get_default_dqn_config() -> DQNConfig:
    """
    Build the default `DQNConfig` for this project.

    Kept as a function (rather than every call site writing `DQNConfig()`
    directly) for the same reason `get_default_training_config()` and
    `get_final_training_config()` exist in `training_config.py`: one
    canonical definition of "what a default DQN run looks like," and one
    place a future upgrade (loading overrides from YAML/CLI) gets wired in.

    Uses the SAME environment configuration (100 inventory, 30-day
    horizon, $200 base price) as the Q-Learning `get_final_training_config()`,
    deliberately — this is what makes a later Q-Learning-vs-DQN comparison
    (Week 3/4 evaluation work) an apples-to-apples comparison of the
    learning algorithms, not a comparison confounded by different problem
    instances.
    """
    return DQNConfig(
        env_config=PricingEnvConfig(
            initial_inventory=100,
            selling_horizon_days=30,
            base_price=200.0,
            demand=DemandConfig(),
            reward=RewardConfig(),
        )
    )