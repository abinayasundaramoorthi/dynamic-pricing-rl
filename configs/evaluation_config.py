"""
evaluation_config.py

Centralized configuration for the Week 4 policy evaluation workflow
(issue #90 — "Design and Integrate Policy Evaluation Framework").

Mirrors the style of `configs/training_config.py` and `configs/dqn_config.py`
deliberately: a single frozen dataclass fully specifies one evaluation run
(which environment, how many simulated seasons, which policies, where
checkpoints live, which business KPIs matter), validated in
`__post_init__`, with a `get_default_evaluation_config()` factory as the
one canonical place a caller gets a ready-to-run configuration.

Why evaluation needs its own config (not reuse TrainingConfig/DQNConfig):
those two describe *how to train one agent*; this describes *how to run
one comparison across five differently-sourced policies* (two loaded
checkpoints, three parameterless heuristics) against a shared set of
episodes and a shared set of business-facing metrics. None of
TrainingConfig's or DQNConfig's fields (learning rate, replay buffer size,
...) are meaningful here, and this config's fields (which policies to
include, KPI thresholds) aren't meaningful there — same reasoning
`configs/dqn_config.py`'s module docstring gives for why DQNConfig is not
a subclass of TrainingConfig.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from pricing_env import PricingEnvConfig
from pricing_env.demand_simulator import DemandConfig
from pricing_env.reward import RewardConfig
from utils.currency import USD_TO_INR_RATE

# Every policy name this framework knows how to build. Kept as a module
# level constant (rather than inlined in validation) so
# `evaluate_policies.py` and any test can import the same canonical list
# instead of maintaining a second copy.
ALL_POLICY_NAMES = (
    "dqn",
    "q_learning",
    "random",
    "fixed_price",
    "time_based_discount",
)


@dataclass(frozen=True)
class BusinessKPIConfig:
    """
    Defines the business-facing metrics the evaluation report scores every
    policy against, and the target thresholds pulled directly from the
    project's own success criteria
    (reports/project_planning/problem_statement.md, Section 19).

    These are report/interpretation thresholds, not statistical
    hypothesis-test parameters — `evaluate_policies.py` uses them to flag
    which policies clear the bar the project defined for itself, so a
    reviewer scanning the output table doesn't have to hold the Section 19
    numbers in their head.

    Attributes
    ----------
    target_revenue_uplift_pct : float
        Minimum mean-revenue improvement over the `fixed_price` baseline
        a policy should show to be considered a meaningful improvement.
        Default 10.0 (%), matching "Revenue Improvement over Baselines
        > 10%" in Section 19.
    target_sell_through_pct : float
        Minimum acceptable sell-through rate (percentage of initial
        inventory sold by episode end). Default 95.0, matching "Sell-Through
        Rate > 95%".
    max_spoilage_pct : float
        Maximum acceptable spoilage rate (percentage of initial inventory
        left unsold at episode end — the complement of sell-through, kept
        as its own explicit field since it is the number a revenue
        manager audience reads more naturally). Default 5.0, matching
        "Inventory Spoilage < 5%".
    """

    target_revenue_uplift_pct: float = 10.0
    target_sell_through_pct: float = 95.0
    max_spoilage_pct: float = 5.0

    def __post_init__(self) -> None:
        if self.target_revenue_uplift_pct < 0:
            raise ValueError(
                "target_revenue_uplift_pct must be >= 0, got "
                f"{self.target_revenue_uplift_pct}"
            )
        if not (0.0 <= self.target_sell_through_pct <= 100.0):
            raise ValueError(
                "target_sell_through_pct must be in [0, 100], got "
                f"{self.target_sell_through_pct}"
            )
        if not (0.0 <= self.max_spoilage_pct <= 100.0):
            raise ValueError(
                f"max_spoilage_pct must be in [0, 100], got {self.max_spoilage_pct}"
            )


@dataclass(frozen=True)
class EvaluationConfig:
    """
    Immutable, top-level configuration for one policy-evaluation run.

    Attributes
    ----------
    env_config : PricingEnvConfig
        Environment every policy is evaluated against. All five policies
        MUST be evaluated on the same environment configuration (and, per
        `episode_seeds`, the same demand realizations) for the comparison
        to be valid — this is the single object that guarantees that.
    num_episodes : int
        Number of simulated booking seasons per policy. Fixed at 1,000 by
        default, matching the internship brief's Week 4 evaluation
        requirement and Section 19's "Evaluation Protocol: 1,000 simulated
        booking seasons per agent". Must be > 0.
    base_seed : int
        Base seed used to generate `num_episodes` per-episode seeds (see
        `episode_seeds`). Deliberately offset well clear of any training
        seed range (matching `train_agent.evaluate_agent`'s
        `seed_offset=1_000_000` convention) so evaluation never
        accidentally reuses episodes a checkpointed policy trained on.
    policies_to_evaluate : List[str]
        Which policies to include in this run, by name (see
        `ALL_POLICY_NAMES`). Defaults to all five. Exists so a quick
        smoke-test run can evaluate a subset (e.g. just the baselines)
        without touching this file.
    dqn_checkpoint_path : str
        Path to the trained DQN network weights
        (`agents/dqn_agent.py:DQNAgent.save()` output), consumed only if
        "dqn" is in `policies_to_evaluate`.
    q_learning_checkpoint_path : str
        Path to the trained Q-table (`agents/q_learning.py:QLearningAgent.save()`
        output), consumed only if "q_learning" is in `policies_to_evaluate`.
    dqn_hidden_layer_sizes : List[int]
        Network architecture used to reconstruct the DQN's Q-network
        before loading its weights. Must exactly match the architecture
        the checkpoint was trained with (`DQNAgent.load()` reads the
        architecture out of the checkpoint file itself, so this field is
        only a documentation/consistency aid, not load-bearing).
    random_policy_seed : int
        RNG seed for `baselines.RandomPolicy`, independent of
        `base_seed`, so the random baseline's action sequence is itself
        reproducible across runs.
    time_based_discount_max_discount, time_based_discount_time_urgency_weight,
    time_based_discount_pacing_weight : float
        Tunable parameters forwarded to
        `baselines.TimeBasedDiscountPolicy` — see that class's docstring.
    business_kpis : BusinessKPIConfig
        Thresholds the evaluation report scores each policy against.
    results_dir : str
        Directory the evaluation harness writes its output artifacts to:
        the episode-level results file (`evaluation_results.csv`, named to
        match the project's deliverable spec exactly) and the aggregate
        summary CSV/JSON. Defaults to `"evaluation"` so
        `evaluation_results.csv` lands directly at `evaluation/evaluation_results.csv`
        with no manual copy/rename step required after running the
        pipeline.
    reference_policy_for_uplift : str
        Which policy name in `policies_to_evaluate` is used as the "1.0x"
        baseline when computing each other policy's revenue-uplift
        percentage. Default "fixed_price" — the canonical legacy-system
        stand-in referenced throughout the design doc. Must itself be a
        member of `policies_to_evaluate`.
    """

    env_config: PricingEnvConfig = field(default_factory=PricingEnvConfig)

    num_episodes: int = 1000
    base_seed: int = 5_000_000

    policies_to_evaluate: List[str] = field(
        default_factory=lambda: list(ALL_POLICY_NAMES)
    )

    dqn_checkpoint_path: str = "agents/checkpoints/dqn_policy.pt"
    q_learning_checkpoint_path: str = "agents/checkpoints/q_learning_policy.pkl"
    dqn_hidden_layer_sizes: List[int] = field(default_factory=lambda: [64, 64])

    random_policy_seed: int = 7

    time_based_discount_max_discount: float = 0.5
    time_based_discount_time_urgency_weight: float = 0.15
    time_based_discount_pacing_weight: float = 1.5

    business_kpis: BusinessKPIConfig = field(default_factory=BusinessKPIConfig)

    results_dir: str = "evaluation"
    reference_policy_for_uplift: str = "fixed_price"

    def __post_init__(self) -> None:
        if not isinstance(self.env_config, PricingEnvConfig):
            raise TypeError(
                f"env_config must be a PricingEnvConfig instance, got {type(self.env_config)}"
            )
        if not isinstance(self.business_kpis, BusinessKPIConfig):
            raise TypeError(
                "business_kpis must be a BusinessKPIConfig instance, got "
                f"{type(self.business_kpis)}"
            )
        if self.num_episodes <= 0:
            raise ValueError(f"num_episodes must be > 0, got {self.num_episodes}")
        if len(self.policies_to_evaluate) == 0:
            raise ValueError(
                "policies_to_evaluate must name at least one policy to evaluate"
            )
        unknown = set(self.policies_to_evaluate) - set(ALL_POLICY_NAMES)
        if unknown:
            raise ValueError(
                f"policies_to_evaluate contains unknown policy name(s) {sorted(unknown)}; "
                f"must be a subset of {sorted(ALL_POLICY_NAMES)}"
            )
        if len(set(self.policies_to_evaluate)) != len(self.policies_to_evaluate):
            raise ValueError(
                f"policies_to_evaluate must not contain duplicates, got "
                f"{self.policies_to_evaluate}"
            )
        if self.reference_policy_for_uplift not in self.policies_to_evaluate:
            raise ValueError(
                f"reference_policy_for_uplift={self.reference_policy_for_uplift!r} "
                f"must be one of the evaluated policies {self.policies_to_evaluate}"
            )
        if len(self.dqn_hidden_layer_sizes) == 0:
            raise ValueError(
                "dqn_hidden_layer_sizes must be a non-empty list, got "
                f"{self.dqn_hidden_layer_sizes}"
            )
        if not (0.0 < self.time_based_discount_max_discount <= 1.0):
            raise ValueError(
                "time_based_discount_max_discount must be in (0, 1], got "
                f"{self.time_based_discount_max_discount}"
            )

    def episode_seeds(self) -> List[int]:
        """
        The exact, ordered list of `num_episodes` seeds every policy in
        this run is evaluated against.

        Centralized here (rather than each policy's evaluation loop
        deriving its own `base_seed + i`) so it is structurally impossible
        for two policies in the same run to silently diverge onto
        different seed sequences — the single requirement that makes a
        head-to-head comparison across policies valid (identical demand
        realizations per simulated season, per
        `pricing_env/pricing_env.py`'s `reset()` docstring).
        """
        return [self.base_seed + i for i in range(1, self.num_episodes + 1)]


def get_default_evaluation_config() -> EvaluationConfig:
    """
    Build the default `EvaluationConfig` for this project.

    Uses the SAME environment configuration (100 inventory, 30-day
    horizon, ₹19,078 base price (200 USD × 95.39)) as
    `configs.training_config.get_final_training_config()` and
    `configs.dqn_config.get_default_dqn_config()`, deliberately — this is
    what makes the Week 4 comparison an apples-to-apples evaluation of
    algorithms against a fixed problem instance, not a comparison
    confounded by different environment settings between training and
    evaluation.
    """
    return EvaluationConfig(
        env_config=PricingEnvConfig(
            initial_inventory=100,
            selling_horizon_days=30,
            base_price=round(200.0 * USD_TO_INR_RATE, 2),
            demand=DemandConfig(),
            reward=RewardConfig(),
        )
    )


def get_smoke_test_evaluation_config() -> EvaluationConfig:
    """
    A fast, small-`num_episodes` variant of the default configuration, for
    verifying the evaluation pipeline initializes and runs end to end
    (acceptance criterion: "Evaluation pipeline initializes successfully")
    without waiting on a full 1,000-episode run per policy. Not intended
    for reported results — only for CI / local smoke testing.
    """
    from dataclasses import replace

    return replace(get_default_evaluation_config(), num_episodes=20)