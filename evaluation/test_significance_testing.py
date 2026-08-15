"""
test_significance_testing.py

Tests for evaluation/significance_testing.py.
"""

import numpy as np
import pandas as pd
import pytest

from evaluation.significance_testing import compare_all_to_reference, compare_policies


def _make_results_df(seeds, policy_a_values, policy_b_values):
    rows = []
    for seed, value in zip(seeds, policy_a_values):
        rows.append({"policy": "policy_a", "seed": seed, "final_revenue": value})
    for seed, value in zip(seeds, policy_b_values):
        rows.append({"policy": "policy_b", "seed": seed, "final_revenue": value})
    return pd.DataFrame(rows)


def test_compare_policies_detects_clear_difference():
    seeds = list(range(100))
    rng = np.random.default_rng(0)
    a_values = 1000 + rng.normal(0, 10, size=100)
    b_values = 500 + rng.normal(0, 10, size=100)  # clearly lower, same noise

    df = _make_results_df(seeds, a_values, b_values)
    result = compare_policies(df, "policy_a", "policy_b")

    assert result["n_paired_episodes"] == 100
    assert result["mean_diff_a_minus_b"] > 0
    assert result["paired_t_pvalue"] < 0.05
    assert result["significant_at_alpha"] is True


def test_compare_policies_no_difference_not_significant():
    seeds = list(range(50))
    rng = np.random.default_rng(1)
    values = 1000 + rng.normal(0, 50, size=50)

    df = _make_results_df(seeds, values, values)  # identical -> zero diff
    result = compare_policies(df, "policy_a", "policy_b")

    assert result["mean_diff_a_minus_b"] == pytest.approx(0.0)
    assert result["significant_at_alpha"] is False


def test_compare_policies_missing_policy_raises():
    df = _make_results_df([1, 2, 3], [1, 2, 3], [1, 2, 3])
    with pytest.raises(ValueError):
        compare_policies(df, "policy_a", "nonexistent_policy")


def test_compare_all_to_reference_returns_one_row_per_other_policy():
    seeds = list(range(20))
    rng = np.random.default_rng(2)
    df_parts = []
    for name, offset in [("fixed_price", 1000), ("dqn", 900), ("random", 700)]:
        values = offset + rng.normal(0, 5, size=20)
        for seed, value in zip(seeds, values):
            df_parts.append({"policy": name, "seed": seed, "final_revenue": value})
    df = pd.DataFrame(df_parts)

    table = compare_all_to_reference(df, reference="fixed_price")

    assert set(table["policy_a"]) == {"dqn", "random"}
    assert all(table["policy_b"] == "fixed_price")
