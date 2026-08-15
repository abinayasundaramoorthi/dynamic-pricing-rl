"""
significance_testing.py

Statistical significance testing for the policy evaluation framework
(closes the "no statistical significance testing" item in Known Issues).

`evaluate_policies.py` already evaluates every policy on the exact same
ordered sequence of episode seeds (`EvaluationConfig.episode_seeds()`),
which makes this a *paired* comparison: for every seed, policy A and
policy B saw the identical simulated booking season. That pairing is what
makes a paired t-test (and its non-parametric counterpart, the Wilcoxon
signed-rank test) the statistically appropriate tool here, rather than an
unpaired/independent-samples test — it removes episode-to-episode demand
variance from the comparison and isolates the effect actually attributable
to the pricing policy.

This module reads the episode-level CSV `evaluate_policies.py` already
writes (`evaluation/evaluation_results.csv`) - it does not run any new
simulations - and reports, for every policy pair, whether the observed
revenue difference is statistically significant.

Usage
-----
    python -m evaluation.significance_testing
    python -m evaluation.significance_testing --reference fixed_price
    python -m evaluation.significance_testing --alpha 0.01
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

DEFAULT_RESULTS_PATH = "evaluation/evaluation_results.csv"
DEFAULT_OUTPUT_PATH = "evaluation/significance_results.csv"


def load_episode_results(path: str = DEFAULT_RESULTS_PATH) -> pd.DataFrame:
    """Load the episode-level evaluation CSV written by evaluate_policies.py."""
    results_path = Path(path)
    if not results_path.exists():
        raise FileNotFoundError(
            f"No episode-level results found at {results_path}. Run "
            "`python -m evaluation.evaluate_policies` first."
        )
    return pd.read_csv(results_path)


def _paired_revenue(
    df: pd.DataFrame, policy_a: str, policy_b: str, metric: str = "final_revenue"
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return two arrays of `metric`, aligned by seed, for `policy_a` and
    `policy_b`. Raises if either policy has no rows, or if the seed sets
    don't match (which would silently break the pairing assumption).
    """
    a = df[df["policy"] == policy_a].set_index("seed")[metric]
    b = df[df["policy"] == policy_b].set_index("seed")[metric]

    if a.empty:
        raise ValueError(f"No episodes found for policy '{policy_a}'.")
    if b.empty:
        raise ValueError(f"No episodes found for policy '{policy_b}'.")

    shared_seeds = a.index.intersection(b.index)
    if len(shared_seeds) < len(a) or len(shared_seeds) < len(b):
        logger.warning(
            "Policies '%s' (%d episodes) and '%s' (%d episodes) do not "
            "share the full seed set (%d shared) - comparison is restricted "
            "to the %d seeds both policies were evaluated on.",
            policy_a, len(a), policy_b, len(b), len(shared_seeds), len(shared_seeds),
        )
    a = a.loc[shared_seeds].sort_index()
    b = b.loc[shared_seeds].sort_index()
    return a.to_numpy(), b.to_numpy()


def compare_policies(
    df: pd.DataFrame,
    policy_a: str,
    policy_b: str,
    metric: str = "final_revenue",
    alpha: float = 0.05,
) -> dict:
    """
    Paired comparison of `policy_a` vs `policy_b` on `metric`.

    Runs both a paired t-test (parametric, assumes approximately normal
    paired differences) and a Wilcoxon signed-rank test (non-parametric,
    no normality assumption) - reporting both rather than picking one lets
    a reader judge robustness instead of trusting a single test's
    assumptions blindly.
    """
    a, b = _paired_revenue(df, policy_a, policy_b, metric)
    diff = a - b  # positive => policy_a higher

    t_stat, t_pvalue = stats.ttest_rel(a, b)

    try:
        w_stat, w_pvalue = stats.wilcoxon(a, b)
    except ValueError:
        # All paired differences are zero - wilcoxon is undefined.
        w_stat, w_pvalue = float("nan"), float("nan")

    mean_diff = float(np.mean(diff))
    pooled_std = float(np.std(diff, ddof=1)) if len(diff) > 1 else float("nan")
    cohens_d = mean_diff / pooled_std if pooled_std not in (0.0, float("nan")) else float("nan")

    return {
        "policy_a": policy_a,
        "policy_b": policy_b,
        "metric": metric,
        "n_paired_episodes": int(len(diff)),
        "mean_a": float(np.mean(a)),
        "mean_b": float(np.mean(b)),
        "mean_diff_a_minus_b": mean_diff,
        "paired_t_statistic": float(t_stat),
        "paired_t_pvalue": float(t_pvalue),
        "wilcoxon_statistic": float(w_stat),
        "wilcoxon_pvalue": float(w_pvalue),
        "cohens_d": cohens_d,
        "significant_at_alpha": bool(t_pvalue < alpha),
        "alpha": alpha,
    }


def compare_all_to_reference(
    df: pd.DataFrame,
    reference: str,
    metric: str = "final_revenue",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Compare every other policy present in `df` against `reference`."""
    other_policies: List[str] = sorted(
        p for p in df["policy"].unique() if p != reference
    )
    rows = [
        compare_policies(df, policy_a=p, policy_b=reference, metric=metric, alpha=alpha)
        for p in other_policies
    ]
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Paired statistical significance testing across the policies "
            "already evaluated by evaluate_policies.py."
        )
    )
    parser.add_argument(
        "--results-path", type=str, default=DEFAULT_RESULTS_PATH,
        help="Path to the episode-level CSV written by evaluate_policies.py.",
    )
    parser.add_argument(
        "--reference", type=str, default="fixed_price",
        help="Policy every other policy is compared against (default: fixed_price).",
    )
    parser.add_argument(
        "--metric", type=str, default="final_revenue",
        help="Column in the episode-level CSV to compare (default: final_revenue).",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.05,
        help="Significance threshold (default: 0.05).",
    )
    parser.add_argument(
        "--output-path", type=str, default=DEFAULT_OUTPUT_PATH,
        help="Where to write the comparison table CSV.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    args = parse_args()

    df = load_episode_results(args.results_path)
    result_table = compare_all_to_reference(
        df, reference=args.reference, metric=args.metric, alpha=args.alpha
    )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_table.to_csv(output_path, index=False)

    logger.info("Wrote significance results to %s", output_path)
    logger.info(
        "\nPaired comparison vs '%s' (alpha=%.2f):\n%s",
        args.reference,
        args.alpha,
        result_table[
            [
                "policy_a", "mean_a", "mean_b", "mean_diff_a_minus_b",
                "paired_t_pvalue", "wilcoxon_pvalue", "cohens_d", "significant_at_alpha",
            ]
        ].to_string(index=False),
    )


if __name__ == "__main__":
    main()
