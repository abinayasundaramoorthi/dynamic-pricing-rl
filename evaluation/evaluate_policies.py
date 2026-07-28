"""
evaluate_policies.py

Policy evaluation workflow for the Travel & Hospitality Dynamic Pricing
project (Week 4, issue #90 — "Design and Integrate Policy Evaluation
Framework").

This is the single place every pricing strategy the project has produced —
two learned agents and three heuristic baselines — is run through an
identical large-scale simulation protocol and compared on the same
business-facing metrics. See `reports/policy_evaluation_design.md` for the
full architecture write-up; this module docstring covers only the
practical "what does running this actually do" summary.

Pipeline
--------
  1. Load an `EvaluationConfig` (`configs/evaluation_config.py`).
  2. Build one shared `PricingEnvironment` and verify it (reusing
     `training/env_utils.py`, exactly as both training entry points do).
  3. Build every requested policy (`build_policies()`): load the DQN and
     Q-Learning checkpoints from disk, and construct the three
     parameterless baselines (`baselines/`).
  4. Generate one shared, ordered list of episode seeds
     (`EvaluationConfig.episode_seeds()`) — every policy is evaluated on
     the *exact same* sequence of simulated booking seasons, so
     differences in outcome reflect differences in policy quality, not
     differences in demand luck.
  5. Run `config.num_episodes` (1,000 by default) episodes per policy,
     acting greedily (no exploration), recording business KPIs every
     episode.
  6. Aggregate per-policy summary statistics, score them against
     `EvaluationConfig.business_kpis`, and compute each policy's revenue
     uplift over the configured reference baseline.
  7. Persist episode-level and summary results under
     `config.results_dir` and log a human-readable comparison table.

Usage
-----
    python -m evaluation.evaluate_policies
    python -m evaluation.evaluate_policies --episodes 100          # faster run
    python -m evaluation.evaluate_policies --smoke-test            # 20-episode pipeline check
    python -m evaluation.evaluate_policies --policies random fixed_price
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Sequence

import numpy as np
import pandas as pd

from agents.dqn_agent import DQNAgent
from agents.q_learning import QLearningAgent
from baselines import FixedPricePolicy, RandomPolicy, TimeBasedDiscountPolicy
from configs.evaluation_config import (
    ALL_POLICY_NAMES,
    EvaluationConfig,
    get_default_evaluation_config,
)
from pricing_env import PricingEnvironment
from training.env_utils import build_environment, verify_environment_compatibility

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Uniform policy protocol
# --------------------------------------------------------------------------- #
class Policy(Protocol):
    """
    The interface every evaluated policy — learned or heuristic — must
    satisfy. `DQNAgent`, `QLearningAgent`, and all three classes in
    `baselines/` already implement `select_greedy_action` with this exact
    signature, so no adapter/wrapper class is needed: this Protocol exists
    purely to type-check `run_episode()`/`evaluate_policy()` against a
    single shared contract instead of five different concrete classes.
    """

    def select_greedy_action(self, observation: np.ndarray) -> int: ...


# --------------------------------------------------------------------------- #
# Policy construction
# --------------------------------------------------------------------------- #
def build_policies(config: EvaluationConfig, env: PricingEnvironment) -> Dict[str, Policy]:
    """
    Construct every policy named in `config.policies_to_evaluate`.

    A policy whose checkpoint file is missing (e.g. `dqn` before
    `training/train_dqn.py` has ever been run) is logged as a warning and
    excluded from the returned dict, rather than raising — this is what
    lets "Evaluation pipeline initializes successfully" (acceptance
    criterion) hold true even on a fresh checkout where not every agent
    has been trained yet, while still surfacing loudly (via the warning
    log and the caller re-checking `config.policies_to_evaluate` against
    the returned keys) that a requested policy did not actually run.

    Returns
    -------
    Dict[str, Policy]
        Maps policy name -> constructed policy object, for every
        successfully-built policy. Iteration order matches
        `config.policies_to_evaluate`.
    """
    policies: Dict[str, Policy] = {}

    for name in config.policies_to_evaluate:
        if name == "dqn":
            checkpoint = Path(config.dqn_checkpoint_path)
            if not checkpoint.exists():
                logger.warning(
                    "Skipping 'dqn': no checkpoint found at %s. Run "
                    "`python -m training.train_dqn` first to produce one.",
                    checkpoint,
                )
                continue
            policies["dqn"] = DQNAgent.load(
                checkpoint,
                observation_space=env.observation_space,
                action_space=env.action_space,
                device="cpu",
                seed=config.base_seed,
            )

        elif name == "q_learning":
            checkpoint = Path(config.q_learning_checkpoint_path)
            if not checkpoint.exists():
                logger.warning(
                    "Skipping 'q_learning': no checkpoint found at %s. Run "
                    "`python -m training.train_agent --agent q_learning` "
                    "first to produce one.",
                    checkpoint,
                )
                continue
            policies["q_learning"] = QLearningAgent.load(
                checkpoint, action_space=env.action_space, seed=config.base_seed
            )

        elif name == "random":
            policies["random"] = RandomPolicy(
                action_space=env.action_space, seed=config.random_policy_seed
            )

        elif name == "fixed_price":
            policies["fixed_price"] = FixedPricePolicy(
                action_space=env.action_space,
                price_adjustment_pct=config.env_config.price_adjustment_pct,
            )

        elif name == "time_based_discount":
            policies["time_based_discount"] = TimeBasedDiscountPolicy(
                action_space=env.action_space,
                price_adjustment_pct=config.env_config.price_adjustment_pct,
                initial_inventory=config.env_config.initial_inventory,
                selling_horizon_days=config.env_config.selling_horizon_days,
                max_discount=config.time_based_discount_max_discount,
                time_urgency_weight=config.time_based_discount_time_urgency_weight,
                pacing_weight=config.time_based_discount_pacing_weight,
            )

        else:
            # EvaluationConfig.__post_init__ already restricts
            # policies_to_evaluate to ALL_POLICY_NAMES, so reaching here
            # would mean this function and that validation have drifted
            # out of sync — fail loudly rather than silently skipping.
            raise ValueError(f"build_policies() has no case for policy name {name!r}")

    logger.info(
        "Built %d/%d requested policies: %s",
        len(policies),
        len(config.policies_to_evaluate),
        list(policies.keys()),
    )
    return policies


# --------------------------------------------------------------------------- #
# Episode-level and policy-level evaluation
# --------------------------------------------------------------------------- #
@dataclass
class EpisodeResult:
    """One simulated booking season's outcome for one policy."""

    policy: str
    seed: int
    total_reward: float
    final_revenue: float
    units_sold: int
    initial_inventory: int
    sell_through_pct: float
    spoilage_pct: float
    mean_price: float
    mean_discount_depth_pct: float
    steps: int


def run_episode(env: PricingEnvironment, policy: Policy, seed: int) -> EpisodeResult:
    """
    Run exactly one simulated booking season with `policy` acting greedily
    (no exploration — this is an evaluation run, not a training run) and
    return its business-facing outcome.
    """
    observation, info = env.reset(seed=seed)
    if hasattr(policy, "reset"):
        policy.reset()

    initial_inventory = int(info["initial_inventory"])
    base_price = float(env.config.base_price)

    terminated = truncated = False
    total_reward = 0.0
    prices: List[float] = []
    discount_depths: List[float] = []
    steps = 0

    while not (terminated or truncated):
        action = policy.select_greedy_action(observation)
        observation, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        prices.append(info["price"])
        discount_depths.append(max(0.0, (base_price - info["price"]) / base_price))
        steps += 1

    remaining_inventory = int(observation[0])
    units_sold = initial_inventory - remaining_inventory
    sell_through_pct = 100.0 * units_sold / initial_inventory
    spoilage_pct = 100.0 - sell_through_pct

    return EpisodeResult(
        policy=getattr(policy, "name", policy.__class__.__name__),
        seed=seed,
        total_reward=total_reward,
        final_revenue=float(info["episode_revenue"]),
        units_sold=units_sold,
        initial_inventory=initial_inventory,
        sell_through_pct=sell_through_pct,
        spoilage_pct=spoilage_pct,
        mean_price=float(np.mean(prices)) if prices else 0.0,
        mean_discount_depth_pct=float(np.mean(discount_depths)) * 100.0 if discount_depths else 0.0,
        steps=steps,
    )


def evaluate_policy(
    env: PricingEnvironment,
    policy_name: str,
    policy: Policy,
    seeds: Sequence[int],
) -> List[EpisodeResult]:
    """Run `policy` across every seed in `seeds` (one simulated season per
    seed) and return the full list of per-episode results."""
    results: List[EpisodeResult] = []
    log_every = max(1, len(seeds) // 10)

    for i, seed in enumerate(seeds, start=1):
        result = run_episode(env, policy, seed)
        result.policy = policy_name
        results.append(result)

        if i % log_every == 0 or i == len(seeds):
            logger.info(
                "[%s] episode %d/%d | mean_revenue_so_far=$%.2f",
                policy_name,
                i,
                len(seeds),
                float(np.mean([r.final_revenue for r in results])),
            )

    return results


# --------------------------------------------------------------------------- #
# Aggregation and business-KPI scoring
# --------------------------------------------------------------------------- #
def summarize_policy(policy_name: str, episode_results: List[EpisodeResult]) -> Dict[str, float]:
    """Reduce one policy's per-episode results into the summary row used
    in the final comparison table."""
    df = pd.DataFrame([asdict(r) for r in episode_results])
    return {
        "policy": policy_name,
        "num_episodes": len(df),
        "mean_reward": float(df["total_reward"].mean()),
        "std_reward": float(df["total_reward"].std(ddof=0)),
        "mean_revenue": float(df["final_revenue"].mean()),
        "std_revenue": float(df["final_revenue"].std(ddof=0)),
        "median_revenue": float(df["final_revenue"].median()),
        "mean_sell_through_pct": float(df["sell_through_pct"].mean()),
        "mean_spoilage_pct": float(df["spoilage_pct"].mean()),
        "mean_price": float(df["mean_price"].mean()),
        "mean_discount_depth_pct": float(df["mean_discount_depth_pct"].mean()),
    }


def score_against_business_kpis(
    summary: pd.DataFrame, config: EvaluationConfig
) -> pd.DataFrame:
    """
    Add pass/fail columns against `config.business_kpis`, plus each
    policy's revenue uplift over `config.reference_policy_for_uplift`.

    Kept as a separate post-processing step over the plain summary table
    (rather than folded into `summarize_policy`) so the raw summary always
    stays interpretable on its own — the KPI columns are a derived
    business-facing overlay, not a change to what was actually measured.
    """
    summary = summary.copy()
    kpis = config.business_kpis

    reference_row = summary.loc[summary["policy"] == config.reference_policy_for_uplift]
    if reference_row.empty:
        raise ValueError(
            f"reference_policy_for_uplift={config.reference_policy_for_uplift!r} "
            "not found among evaluated policies; cannot compute revenue uplift."
        )
    reference_revenue = float(reference_row["mean_revenue"].iloc[0])

    summary["revenue_uplift_pct"] = (
        (summary["mean_revenue"] - reference_revenue) / reference_revenue * 100.0
        if reference_revenue != 0
        else np.nan
    )
    summary["meets_revenue_uplift_target"] = (
        summary["revenue_uplift_pct"] >= kpis.target_revenue_uplift_pct
    )
    summary["meets_sell_through_target"] = (
        summary["mean_sell_through_pct"] >= kpis.target_sell_through_pct
    )
    summary["meets_spoilage_target"] = (
        summary["mean_spoilage_pct"] <= kpis.max_spoilage_pct
    )
    return summary


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def save_results(
    all_episode_results: Dict[str, List[EpisodeResult]],
    summary: pd.DataFrame,
    config: EvaluationConfig,
) -> Dict[str, Path]:
    """
    Write episode-level and summary results to `config.results_dir`.

    Returns a dict of the paths written, so `main()` can log/print them
    without the caller having to reconstruct the naming convention.
    """
    results_dir = Path(config.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    written: Dict[str, Path] = {}

    episode_rows = [
        asdict(r) for results in all_episode_results.values() for r in results
    ]
    episode_path = results_dir / "policy_evaluation_episodes.csv"
    pd.DataFrame(episode_rows).to_csv(episode_path, index=False)
    written["episodes_csv"] = episode_path

    summary_csv_path = results_dir / "policy_evaluation_summary.csv"
    summary.to_csv(summary_csv_path, index=False)
    written["summary_csv"] = summary_csv_path

    summary_json_path = results_dir / "policy_evaluation_summary.json"
    with open(summary_json_path, "w") as f:
        json.dump(
            {
                "config": {
                    "num_episodes": config.num_episodes,
                    "base_seed": config.base_seed,
                    "policies_to_evaluate": config.policies_to_evaluate,
                    "reference_policy_for_uplift": config.reference_policy_for_uplift,
                    "business_kpis": asdict(config.business_kpis),
                },
                "results": summary.to_dict(orient="records"),
            },
            f,
            indent=2,
        )
    written["summary_json"] = summary_json_path

    logger.info("Wrote evaluation results to %s", results_dir)
    return written


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run_evaluation(config: EvaluationConfig) -> pd.DataFrame:
    """
    Execute the full evaluation workflow end to end and return the
    business-KPI-scored summary table. This is the function to call
    programmatically (e.g. from a notebook); `main()` is a thin CLI shell
    around it.
    """
    env = build_environment(config.env_config)
    verify_environment_compatibility(env)

    policies = build_policies(config, env)
    if not policies:
        raise RuntimeError(
            "No policies were successfully built — nothing to evaluate. "
            "Check that requested checkpoints exist (see the warnings "
            "logged above)."
        )

    seeds = config.episode_seeds()
    logger.info(
        "Evaluating %d polic(y/ies) across %d simulated booking seasons each "
        "(%d total episodes)...",
        len(policies),
        len(seeds),
        len(policies) * len(seeds),
    )

    all_episode_results: Dict[str, List[EpisodeResult]] = {}
    try:
        for policy_name, policy in policies.items():
            all_episode_results[policy_name] = evaluate_policy(
                env, policy_name, policy, seeds
            )
    finally:
        env.close()

    summary = pd.DataFrame(
        [
            summarize_policy(name, results)
            for name, results in all_episode_results.items()
        ]
    )

    # If the configured reference policy wasn't actually built (e.g. its
    # checkpoint was missing), fall back to whichever policy ran so the
    # KPI scoring step still produces a usable table rather than raising.
    effective_config = config
    if config.reference_policy_for_uplift not in policies:
        fallback = next(iter(policies))
        logger.warning(
            "reference_policy_for_uplift=%r was not built; falling back to "
            "%r for revenue-uplift scoring.",
            config.reference_policy_for_uplift,
            fallback,
        )
        effective_config = replace(config, reference_policy_for_uplift=fallback)

    summary = score_against_business_kpis(summary, effective_config)
    save_results(all_episode_results, summary, config)
    return summary


def log_summary_table(summary: pd.DataFrame) -> None:
    """Log a compact, human-readable comparison table."""
    display_cols = [
        "policy",
        "num_episodes",
        "mean_revenue",
        "std_revenue",
        "revenue_uplift_pct",
        "mean_sell_through_pct",
        "mean_spoilage_pct",
        "meets_revenue_uplift_target",
        "meets_sell_through_target",
        "meets_spoilage_target",
    ]
    display_cols = [c for c in display_cols if c in summary.columns]
    sorted_summary = summary.sort_values("mean_revenue", ascending=False)
    logger.info(
        "\nPolicy evaluation summary (sorted by mean revenue):\n%s",
        sorted_summary[display_cols].to_string(index=False),
    )


# --------------------------------------------------------------------------- #
# CLI / entry point
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate all pricing policies across simulated booking seasons."
    )
    parser.add_argument(
        "--episodes", type=int, default=None, help="Override EvaluationConfig.num_episodes."
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a fast 20-episode pipeline check instead of the full 1,000-episode evaluation.",
    )
    parser.add_argument(
        "--policies",
        nargs="+",
        default=None,
        choices=list(ALL_POLICY_NAMES),
        help="Subset of policy names to evaluate. Defaults to all.",
    )
    parser.add_argument(
        "--results-dir", type=str, default=None, help="Override EvaluationConfig.results_dir."
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    args = parse_args()

    if args.smoke_test:
        from configs.evaluation_config import get_smoke_test_evaluation_config

        config = get_smoke_test_evaluation_config()
    else:
        config = get_default_evaluation_config()

    overrides = {}
    if args.episodes is not None:
        overrides["num_episodes"] = args.episodes
    if args.policies is not None:
        overrides["policies_to_evaluate"] = args.policies
    if args.results_dir is not None:
        overrides["results_dir"] = args.results_dir
    if overrides:
        config = replace(config, **overrides)

    logger.info(
        "Loaded EvaluationConfig | num_episodes=%d | policies=%s | results_dir=%s",
        config.num_episodes,
        config.policies_to_evaluate,
        config.results_dir,
    )

    summary = run_evaluation(config)
    log_summary_table(summary)


if __name__ == "__main__":
    main()