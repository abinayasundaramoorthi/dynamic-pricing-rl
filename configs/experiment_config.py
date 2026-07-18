"""
experiment_config.py

Defines a structured way to specify and enumerate multiple RL experiment
configurations, so different (learning_rate, discount_factor,
exploration_rate, num_episodes) combinations can be run and compared
without hardcoding any of them into the execution script.

Builds directly on configs/training_config.py: where `TrainingConfig` fully
specifies *one* run, `ExperimentConfig` here names that run and lets a
small set of hyperparameters be overridden per-experiment, and
`get_experiment_suite()` enumerates the full set of experiments a
comparison sweep should execute. `training/run_experiment.py` never
hardcodes a hyperparameter value anywhere — it only iterates the list this
file returns.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import List, Optional

from configs.training_config import TrainingConfig, get_default_training_config


@dataclass(frozen=True)
class ExperimentConfig:
    """
    Names and parameterizes one experiment: a specific override of one or
    more hyperparameters, to be run and recorded independently of every
    other experiment in the suite.

    Attributes
    ----------
    name : str
        Unique, filesystem-safe identifier (e.g. "baseline", "high_lr").
        Used directly as the results subfolder name (see
        `run_experiment.py`), so two experiments sharing a name would
        silently overwrite each other's results — `get_experiment_suite()`
        callers must ensure uniqueness; `run_experiment.run_experiment_suite()`
        enforces it explicitly before running anything.
    learning_rate : Optional[float]
        Override for `TrainingConfig.learning_rate`. `None` means "use
        the shared default from `get_default_training_config()`" — this
        is what lets a sweep vary exactly one hyperparameter at a time
        while leaving every other parameter at a common baseline.
    discount_factor : Optional[float]
        Override for `TrainingConfig.discount_factor`. `None` means default.
    exploration_rate : Optional[float]
        Override for `TrainingConfig.exploration_rate`. `None` means default.
    num_episodes : Optional[int]
        Override for `TrainingConfig.num_episodes`. `None` means default.
    seed : Optional[int]
        Override for `TrainingConfig.seed`. `None` means default. Exposed
        so a "does this result replicate under a different seed" check can
        be expressed as an experiment the same way a hyperparameter sweep
        entry is.
    description : str
        Human-readable note on what this experiment tests. Surfaced in
        logs and the persisted summary so a reviewer isn't left reverse
        engineering intent from bare parameter values.
    """

    name: str
    learning_rate: Optional[float] = None
    discount_factor: Optional[float] = None
    exploration_rate: Optional[float] = None
    num_episodes: Optional[int] = None
    seed: Optional[int] = None
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("ExperimentConfig.name must be a non-empty string")
        unsafe_chars = set(r'/\:*?"<>| ')
        if any(c in unsafe_chars for c in self.name):
            raise ValueError(
                f"ExperimentConfig.name={self.name!r} must be filesystem-safe "
                "(no spaces, slashes, or special characters), since it is used "
                "directly as a results folder name."
            )

    def build_training_config(self) -> TrainingConfig:
        """
        Materialize this experiment's overrides into a full `TrainingConfig`.

        Starts from `get_default_training_config()` and applies only the
        fields this experiment actually overrides, via `dataclasses.replace`
        (`TrainingConfig` is frozen). Every field this experiment does not
        name stays at the shared default — this is what keeps a sweep
        isolating exactly the variable(s) under study, rather than
        accidentally varying something else between experiments too.
        """
        base = get_default_training_config()
        overrides = {}
        if self.learning_rate is not None:
            overrides["learning_rate"] = self.learning_rate
        if self.discount_factor is not None:
            overrides["discount_factor"] = self.discount_factor
        if self.exploration_rate is not None:
            overrides["exploration_rate"] = self.exploration_rate
        if self.num_episodes is not None:
            overrides["num_episodes"] = self.num_episodes
        if self.seed is not None:
            overrides["seed"] = self.seed
        return replace(base, **overrides) if overrides else base


def get_experiment_suite() -> List[ExperimentConfig]:
    """
    Enumerate the experiments to run in a comparison sweep.

    This is the single source of truth for experiment parameters —
    `run_experiment.py` contains no hardcoded learning rate, discount
    factor, exploration rate, or episode count anywhere; it only iterates
    this list. Adding, removing, or retuning an experiment is a one-line
    change here, with zero changes required in the execution script.
    """
    return [
        ExperimentConfig(
            name="baseline",
            description=(
                "Default hyperparameters from TrainingConfig, unmodified — "
                "the control run every other experiment is compared against."
            ),
        ),
        ExperimentConfig(
            name="high_learning_rate",
            learning_rate=0.5,
            description=(
                "5x the baseline learning rate — tests whether faster value "
                "updates speed convergence or destabilize training."
            ),
        ),
        ExperimentConfig(
            name="low_learning_rate",
            learning_rate=0.01,
            description=(
                "10x smaller than baseline — tests whether slower updates "
                "improve final policy stability at the cost of convergence speed."
            ),
        ),
        ExperimentConfig(
            name="low_discount_factor",
            discount_factor=0.90,
            description=(
                "More myopic than baseline (0.99) — tests whether heavily "
                "discounting future reward degrades long-horizon pricing "
                "behavior, e.g. holding price too long near the deadline."
            ),
        ),
        ExperimentConfig(
            name="short_run",
            num_episodes=200,
            description=(
                "5x fewer episodes than baseline — establishes how much of "
                "baseline's performance is achievable within a much smaller "
                "training budget."
            ),
        ),
        ExperimentConfig(
            name="alternate_seed",
            seed=1234,
            description=(
                "Same hyperparameters as baseline under a different random "
                "seed — a replication check, not a hyperparameter sweep entry."
            ),
        ),
    ]