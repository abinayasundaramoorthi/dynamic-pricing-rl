"""
run_experiment.py

Runs a suite of RL experiments (configs/experiment_config.py) and stores
each experiment's results independently, so different hyperparameter
configurations can be compared side-by-side later (Week 4
evaluation/dashboard work) — without any experiment parameter being
hardcoded in this file.

Scope note (same placeholder-policy precedent set by train_agent.py's Day 1
pipeline, issue #42): no trainable agent exists yet, so every experiment in
today's suite runs the same random-action policy — only the *config*
varies between experiments. This means today's numeric results are NOT a
meaningful comparison of hyperparameter effects on learning; there is no
learning happening yet. What this script proves is that the experiment
*execution workflow* itself is correct end to end: multiple configurations
run independently, are recorded separately, and are comparable side by
side. Once the Q-Learning agent lands, swapping the policy used in
`_run_episodes()` for a real learning agent is the only change required
for these hyperparameters to start actually affecting outcomes.

Usage
-----
    python -m training.run_experiment
    python -m training.run_experiment --only baseline high_learning_rate
    python -m training.run_experiment --episodes-override 20   # fast smoke test
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import replace
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from configs.experiment_config import ExperimentConfig, get_experiment_suite
from configs.training_config import TrainingConfig
from pricing_env import PricingEnvironment
from training.train_agent import (
    build_environment,
    run_training,
    verify_environment_compatibility,
)

logger = logging.getLogger(__name__)

DEFAULT_RESULTS_ROOT = Path("evaluation/results")


# --------------------------------------------------------------------------- #
# Per-experiment execution
# --------------------------------------------------------------------------- #
def _run_episodes(
    env: PricingEnvironment, config: TrainingConfig
) -> Tuple[List[float], List[float]]:
    """
    Execute `config.num_episodes` episodes, recording per-episode reward
    and revenue.

    Week 2 Day 5 refactor: this is now a thin wrapper around
    `training.train_agent.run_training(collect_metrics=True)` instead of
    duplicating its loop. Previously this function contained its own
    copy of the episode loop (same shape as `run_training()`, same
    `random_policy` fallback, same `max_steps_per_episode` cap) — that
    duplication was flagged as follow-up work at the time (see the Day 4
    version of this docstring) and is resolved here: `run_training()` now
    accepts `collect_metrics=True` and returns exactly what this function
    needs, so there is exactly one place the episode loop is implemented.
    """
    result = run_training(env, config, collect_metrics=True)
    assert result is not None  # guaranteed by collect_metrics=True
    return result


def _summarize_and_persist(
    experiment: ExperimentConfig,
    config: TrainingConfig,
    episode_rewards: List[float],
    episode_revenues: List[float],
    results_root: Path,
) -> dict:
    """
    Write this experiment's per-episode data and summary statistics to its
    own results subfolder, and return the summary as a dict.

    Each experiment gets its own folder (`results_root / experiment.name`)
    containing:
      - episode_metrics.csv : full per-episode reward/revenue trace, for
        later plotting (Week 4 "Price Trajectory" / training-curve work).
      - summary.json : aggregated stats plus the exact hyperparameters
        used, so a result can be fully attributed without cross-referencing
        this script's source.

    This per-folder separation, combined with the suite-level name
    uniqueness check in `run_experiment_suite()`, is what satisfies
    "results are stored separately for comparison": no experiment can
    silently overwrite another's data, and each folder is self-contained
    enough to inspect independently of the rest of the suite.
    """
    experiment_dir = results_root / experiment.name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    episodes_df = pd.DataFrame(
        {
            "episode": range(1, len(episode_rewards) + 1),
            "reward": episode_rewards,
            "revenue": episode_revenues,
        }
    )
    episodes_df.to_csv(experiment_dir / "episode_metrics.csv", index=False)

    last_n = min(50, len(episode_rewards))
    summary = {
        "experiment_name": experiment.name,
        "description": experiment.description,
        "num_episodes": config.num_episodes,
        "learning_rate": config.learning_rate,
        "discount_factor": config.discount_factor,
        "exploration_rate": config.exploration_rate,
        "seed": config.seed,
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "mean_revenue": float(np.mean(episode_revenues)),
        "std_revenue": float(np.std(episode_revenues)),
        "mean_reward_last_n": float(np.mean(episode_rewards[-last_n:])),
        "mean_revenue_last_n": float(np.mean(episode_revenues[-last_n:])),
    }

    with open(experiment_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def run_single_experiment(
    experiment: ExperimentConfig,
    results_root: Path,
    episodes_override: Optional[int] = None,
) -> dict:
    """
    Run one experiment end to end: build its config, construct and verify
    its environment, execute all episodes, and persist its results.

    Parameters
    ----------
    experiment : ExperimentConfig
        The experiment to run.
    results_root : Path
        Root directory under which this experiment's own subfolder is created.
    episodes_override : int, optional
        If given, overrides `num_episodes` for this run regardless of what
        the experiment or default config specify — used for fast smoke
        testing the whole suite without waiting on full-length runs.

    Returns
    -------
    dict
        The summary dict also written to `<experiment_dir>/summary.json`.
    """
    config = experiment.build_training_config()
    if episodes_override is not None:
        config = replace(config, num_episodes=episodes_override)

    logger.info(
        "Starting experiment %r | %s | episodes=%d lr=%.4f gamma=%.3f epsilon=%.3f seed=%d",
        experiment.name,
        experiment.description,
        config.num_episodes,
        config.learning_rate,
        config.discount_factor,
        config.exploration_rate,
        config.seed,
    )

    env = build_environment(config)
    try:
        verify_environment_compatibility(env)
        episode_rewards, episode_revenues = _run_episodes(env, config)
    finally:
        env.close()

    summary = _summarize_and_persist(
        experiment, config, episode_rewards, episode_revenues, results_root
    )
    logger.info(
        "Finished experiment %r | mean_reward=%.2f | mean_revenue=$%.2f",
        experiment.name,
        summary["mean_reward"],
        summary["mean_revenue"],
    )
    return summary


# --------------------------------------------------------------------------- #
# Suite execution
# --------------------------------------------------------------------------- #
def run_experiment_suite(
    experiments: List[ExperimentConfig],
    results_root: Path = DEFAULT_RESULTS_ROOT,
    episodes_override: Optional[int] = None,
) -> pd.DataFrame:
    """
    Run every experiment in `experiments`, persist each one's results
    independently, and return a single comparison DataFrame (also written
    to `results_root/comparison_summary.csv`) with one row per experiment.

    Raises
    ------
    ValueError
        If any two experiments in `experiments` share a `name`. Enforced
        here, before any experiment runs, rather than letting the second
        one silently overwrite the first's results directory partway
        through a potentially long-running suite.
    """
    names = [e.name for e in experiments]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise ValueError(
            f"Duplicate experiment name(s) in suite: {duplicates} — each "
            "ExperimentConfig.name must be unique, since it is used as the "
            "results folder name and would otherwise silently overwrite "
            "another experiment's results."
        )

    results_root.mkdir(parents=True, exist_ok=True)

    summaries = [
        run_single_experiment(experiment, results_root, episodes_override)
        for experiment in experiments
    ]

    comparison_df = pd.DataFrame(summaries).sort_values(
        "mean_revenue", ascending=False
    )
    comparison_df.to_csv(results_root / "comparison_summary.csv", index=False)

    logger.info(
        "Experiment suite complete (%d experiments). Comparison summary:\n%s",
        len(experiments),
        comparison_df[["experiment_name", "mean_reward", "mean_revenue"]].to_string(
            index=False
        ),
    )

    return comparison_df


# --------------------------------------------------------------------------- #
# CLI / entry point
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run and compare a suite of RL experiments."
    )
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        metavar="NAME",
        help="Run only the named experiment(s) from the suite, "
        "e.g. --only baseline high_learning_rate.",
    )
    parser.add_argument(
        "--episodes-override",
        type=int,
        default=None,
        help="Override num_episodes for every selected experiment "
        "(useful for a fast smoke test of the whole suite).",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=str(DEFAULT_RESULTS_ROOT),
        help="Root directory under which experiment results are stored.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    args = parse_args()
    experiments = get_experiment_suite()

    if args.only:
        selected_names = set(args.only)
        available_names = {e.name for e in experiments}
        unknown = selected_names - available_names
        if unknown:
            raise ValueError(
                f"Unknown experiment name(s) in --only: {sorted(unknown)}. "
                f"Available: {sorted(available_names)}"
            )
        experiments = [e for e in experiments if e.name in selected_names]

    if not experiments:
        raise ValueError("No experiments selected to run.")

    run_experiment_suite(
        experiments,
        results_root=Path(args.results_dir),
        episodes_override=args.episodes_override,
    )


if __name__ == "__main__":
    main()