"""
charts.py

Pure chart-DATA builders for Feature 2 (Advanced Revenue Management
Dashboard Enhancement). Every function returns a plain, JSON-serializable
dict of `{"labels": [...], "datasets": [...]}` — Chart.js's native input
shape — with zero rendering dependency (no matplotlib, no Streamlit).
Rendering happens client-side in `dashboard/web_dashboard/templates/*`
via Chart.js, exactly matching the `revenueChart`/`occupancyChart`/
`pricingDistChart` canvases already defined there.

Two real data sources are used, deliberately matching the existing
project exactly rather than inventing a third:

1. Episode-level results (`evaluation/evaluation_results.csv`, columns
   defined by `dashboard.data_contract.REQUIRED_EPISODE_COLUMNS`) and the
   per-policy summary (`evaluation/policy_evaluation_summary.csv`,
   `REQUIRED_SUMMARY_COLUMNS`) — same contract
   `dashboard/data_contract.py` already loads and validates.

2. The project's real `pricing_env.demand_simulator.DemandSimulator` —
   used (not re-implemented) for the demand-forecast series, so the
   forecast reflects the actual calibrated demand model, not a
   fabricated curve.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from pricing_env.demand_simulator import DemandSimulator

try:
    from dashboard.data_contract import POLICY_DISPLAY_NAMES
except Exception:  # pragma: no cover - data_contract.py always exists in this repo
    POLICY_DISPLAY_NAMES: Dict[str, str] = {}

# Keyed by policy KEY (not display name) so it composes directly with
# `evaluation_results.csv`'s `policy` column without an extra lookup
# step. Hex colors chosen to read well against Chart.js's default
# canvas background and to stay consistent across every chart.
POLICY_COLORS: Dict[str, str] = {
    "dqn": "#2563eb",
    "q_learning": "#7c3aed",
    "fixed_price": "#64748b",
    "time_based_discount": "#f59e0b",
    "random": "#94a3b8",
}


def _display_name(policy: str) -> str:
    return POLICY_DISPLAY_NAMES.get(policy, policy)


def _color_for(policy: str) -> str:
    return POLICY_COLORS.get(policy, "#334155")


def _episode_series(
    episodes: pd.DataFrame,
    column: str,
    policies: Optional[Iterable[str]],
    max_points: int,
) -> Dict[str, Any]:
    """Shared implementation for the per-episode line-chart builders below."""
    selected = list(policies) if policies is not None else sorted(episodes["policy"].unique())
    datasets: List[Dict[str, Any]] = []
    max_len = 0
    for policy in selected:
        subset = episodes.loc[episodes["policy"] == policy].reset_index(drop=True)
        if subset.empty:
            continue
        values = subset[column].head(max_points).tolist()
        max_len = max(max_len, len(values))
        datasets.append(
            {
                "policy": policy,
                "label": _display_name(policy),
                "color": _color_for(policy),
                "data": values,
            }
        )
    return {"labels": [f"Episode {i + 1}" for i in range(max_len)], "datasets": datasets}


def revenue_trend_series(
    episodes: pd.DataFrame, policies: Optional[Iterable[str]] = None, max_points: int = 50
) -> Dict[str, Any]:
    """Per-episode `final_revenue`, one series per policy (variance view across simulated seasons)."""
    return _episode_series(episodes, "final_revenue", policies, max_points)


def occupancy_trend_series(
    episodes: pd.DataFrame, policies: Optional[Iterable[str]] = None, max_points: int = 50
) -> Dict[str, Any]:
    """Per-episode `sell_through_pct` (the project's occupancy proxy), one series per policy."""
    return _episode_series(episodes, "sell_through_pct", policies, max_points)


def spoilage_trend_series(
    episodes: pd.DataFrame, policies: Optional[Iterable[str]] = None, max_points: int = 50
) -> Dict[str, Any]:
    """Per-episode `spoilage_pct` (unsold inventory at episode end), one series per policy."""
    return _episode_series(episodes, "spoilage_pct", policies, max_points)


def reward_trend_series(
    episodes: pd.DataFrame, policies: Optional[Iterable[str]] = None, max_points: int = 50
) -> Dict[str, Any]:
    """Per-episode `total_reward`, one series per policy — the RL training/evaluation signal itself."""
    return _episode_series(episodes, "total_reward", policies, max_points)


def price_action_distribution(episodes: pd.DataFrame, policy: str) -> Dict[str, Any]:
    """
    Distribution of `mean_discount_depth_pct` for one policy's episodes,
    bucketed into Discount / Hold / Premium — the real replacement for
    the dashboard's "Pricing Action Distribution" donut chart. Buckets
    on the actual `mean_discount_depth_pct` column (negative = premium
    pricing, positive = discounting, ~0 = holding at base price) rather
    than a fabricated split.
    """
    subset = episodes.loc[episodes["policy"] == policy]
    if subset.empty:
        return {"labels": ["Discount", "Hold", "Premium"], "data": [0, 0, 0], "total_episodes": 0}

    depth = subset["mean_discount_depth_pct"]
    discount = int((depth > 2.0).sum())
    premium = int((depth < -2.0).sum())
    hold = int(len(depth) - discount - premium)

    return {
        "labels": ["Discount", "Hold", "Premium"],
        "data": [discount, hold, premium],
        "total_episodes": int(len(depth)),
    }


def baseline_comparison_series(summary: pd.DataFrame) -> Dict[str, Any]:
    """
    `mean_revenue` (with `std_revenue` as an error-bar-style companion
    array) for every policy in the aggregate summary — the direct "AI
    pricing vs. baseline strategies" bar-chart data.
    """
    ordered = summary.sort_values("mean_revenue", ascending=False)
    return {
        "labels": [_display_name(p) for p in ordered["policy"]],
        "colors": [_color_for(p) for p in ordered["policy"]],
        "mean_revenue": ordered["mean_revenue"].round(2).tolist(),
        "std_revenue": ordered["std_revenue"].round(2).tolist(),
    }


def revenue_uplift_series(summary: pd.DataFrame) -> Dict[str, Any]:
    """`revenue_uplift_pct` per policy, relative to the configured reference baseline."""
    ordered = summary.sort_values("revenue_uplift_pct", ascending=False)
    return {
        "labels": [_display_name(p) for p in ordered["policy"]],
        "data": ordered["revenue_uplift_pct"].round(2).tolist(),
        "colors": ["#16a34a" if v >= 0 else "#dc2626" for v in ordered["revenue_uplift_pct"]],
    }


def price_trend_series(summary: pd.DataFrame) -> Dict[str, Any]:
    """`mean_price` per policy, from the aggregate summary."""
    ordered = summary.sort_values("mean_price", ascending=False)
    return {
        "labels": [_display_name(p) for p in ordered["policy"]],
        "colors": [_color_for(p) for p in ordered["policy"]],
        "data": ordered["mean_price"].round(2).tolist(),
    }


def demand_forecast_series(
    demand_simulator: DemandSimulator,
    price: float,
    reference_price: float,
    remaining_inventory: int,
    days_remaining: int,
    selling_horizon_days: int,
    seed: int = 0,
    num_simulations: int = 200,
) -> Dict[str, Any]:
    """
    Forward-looking expected-demand outlook for the CURRENT episode's
    remaining days, computed by repeatedly calling the project's real
    `DemandSimulator.sample()` (Monte-Carlo averaged over
    `num_simulations` independent random draws per remaining day) — not
    a fabricated forecast curve.

    This reuses `pricing_env/demand_simulator.py` exactly as
    `pricing_env/pricing_env.py::PricingEnvironment.step()` does,
    holding `price` fixed at its current value to show how expected
    demand intensity evolves purely from time-to-deadline urgency
    (`DemandConfig.urgency_growth`) over the rest of the season.
    """
    rng = np.random.default_rng(seed)
    days_axis: List[int] = list(range(int(days_remaining), 0, -1))
    expected_demand_series: List[float] = []

    for day in days_axis:
        draws = [
            demand_simulator.sample(
                price=price,
                reference_price=reference_price,
                remaining_inventory=remaining_inventory,
                days_remaining=day,
                selling_horizon_days=selling_horizon_days,
                rng=rng,
            ).expected_demand
            for _ in range(num_simulations)
        ]
        expected_demand_series.append(round(float(np.mean(draws)), 3))

    return {
        "labels": [f"Day -{d}" for d in days_axis],
        "data": expected_demand_series,
        "price": price,
        "num_simulations": num_simulations,
    }
