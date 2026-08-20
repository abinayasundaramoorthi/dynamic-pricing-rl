"""
kpi_components.py

Pure-Python KPI computation for Feature 2 (Advanced Revenue Management
Dashboard Enhancement). No UI framework dependency at all — this module
is called by `dashboard/web_dashboard/blueprints/api.py` (Flask) and
returns plain dataclasses/dicts that the blueprint serializes to JSON;
the actual rendering happens client-side in the dashboard's HTML/JS
templates (Chart.js + Bootstrap), not here.

Every function reads the *exact* column contract
`dashboard/data_contract.py` already defines and validates
(`REQUIRED_SUMMARY_COLUMNS`, `REQUIRED_EPISODE_COLUMNS`), so a
`policy_evaluation_summary.csv` / `evaluation_results.csv` DataFrame
that already works elsewhere in the project works here unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd

try:
    # Reuse the shared display-name mapping instead of duplicating it —
    # keeps this module and dashboard/data_contract.py from ever
    # silently drifting apart on how a policy key is labeled.
    from dashboard.data_contract import POLICY_DISPLAY_NAMES as _POLICY_DISPLAY_NAMES
    from dashboard.data_contract import display_name as _existing_display_name
except Exception:  # pragma: no cover - data_contract.py always exists in this repo
    _existing_display_name = None
    _POLICY_DISPLAY_NAMES: Dict[str, str] = {}


def display_name(policy: str) -> str:
    """
    Human-readable label for a policy key.

    Delegates to `dashboard.data_contract.display_name` when importable
    (the normal case in this repo) so this module never maintains a
    second, independently-drifting copy of `POLICY_DISPLAY_NAMES`.
    """
    if _existing_display_name is not None:
        return _existing_display_name(policy)
    return _POLICY_DISPLAY_NAMES.get(policy, policy)  # pragma: no cover - fallback only


@dataclass(frozen=True)
class AiVsBaselineKPIs:
    """
    Head-to-head comparison of one "AI" policy against one baseline
    policy, computed entirely from real `policy_evaluation_summary.csv`
    rows (the same file `dashboard/data_contract.py` already loads via
    `load_summary_results()`).

    Attributes
    ----------
    ai_policy, baseline_policy : str
        Policy keys compared (e.g. "dqn", "fixed_price").
    revenue_uplift_pct : float
        `(ai.mean_revenue - baseline.mean_revenue) / baseline.mean_revenue * 100`.
    sell_through_delta_pct : float
        `ai.mean_sell_through_pct - baseline.mean_sell_through_pct`.
    spoilage_delta_pct : float
        `ai.mean_spoilage_pct - baseline.mean_spoilage_pct` (negative =
        AI spoils less inventory than the baseline).
    ai_meets_all_targets : bool
        True only if the AI policy's row has
        `meets_revenue_uplift_target`, `meets_sell_through_target`, and
        `meets_spoilage_target` all True.
    """

    ai_policy: str
    baseline_policy: str
    revenue_uplift_pct: float
    sell_through_delta_pct: float
    spoilage_delta_pct: float
    ai_meets_all_targets: bool

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serializable representation, for the Flask API layer."""
        return {
            "ai_policy": self.ai_policy,
            "ai_policy_label": display_name(self.ai_policy),
            "baseline_policy": self.baseline_policy,
            "baseline_policy_label": display_name(self.baseline_policy),
            "revenue_uplift_pct": self.revenue_uplift_pct,
            "sell_through_delta_pct": self.sell_through_delta_pct,
            "spoilage_delta_pct": self.spoilage_delta_pct,
            "ai_meets_all_targets": self.ai_meets_all_targets,
        }


def compute_ai_vs_baseline_kpis(
    summary: pd.DataFrame,
    ai_policy: str = "dqn",
    baseline_policy: str = "fixed_price",
) -> Optional[AiVsBaselineKPIs]:
    """
    Compute `AiVsBaselineKPIs` from a `policy_evaluation_summary.csv`-shaped
    DataFrame. Returns `None` (rather than fabricating figures) if either
    requested policy is not present in `summary` — e.g. the DQN
    checkpoint hasn't been evaluated yet.
    """
    ai_rows = summary.loc[summary["policy"] == ai_policy]
    baseline_rows = summary.loc[summary["policy"] == baseline_policy]
    if ai_rows.empty or baseline_rows.empty:
        return None

    ai_row = ai_rows.iloc[0]
    baseline_row = baseline_rows.iloc[0]

    if baseline_row["mean_revenue"] == 0:
        revenue_uplift_pct = 0.0
    else:
        revenue_uplift_pct = (
            (ai_row["mean_revenue"] - baseline_row["mean_revenue"])
            / baseline_row["mean_revenue"]
            * 100.0
        )

    return AiVsBaselineKPIs(
        ai_policy=ai_policy,
        baseline_policy=baseline_policy,
        revenue_uplift_pct=round(float(revenue_uplift_pct), 2),
        sell_through_delta_pct=round(
            float(ai_row["mean_sell_through_pct"] - baseline_row["mean_sell_through_pct"]), 2
        ),
        spoilage_delta_pct=round(
            float(ai_row["mean_spoilage_pct"] - baseline_row["mean_spoilage_pct"]), 2
        ),
        ai_meets_all_targets=bool(
            ai_row["meets_revenue_uplift_target"]
            and ai_row["meets_sell_through_target"]
            and ai_row["meets_spoilage_target"]
        ),
    )


def summary_headline(summary: pd.DataFrame) -> Dict[str, Any]:
    """
    Top-line KPI headline computed from real `policy_evaluation_summary.csv`
    rows: best policy by mean revenue, its revenue/sell-through, and how
    many policies meet all 3 business KPI targets. Returns an explicit
    `"empty": True` marker (never fabricated numbers) if `summary` has no
    rows.
    """
    if summary.empty:
        return {"empty": True}

    best = summary.sort_values("mean_revenue", ascending=False).iloc[0]
    meets_all = (
        summary["meets_revenue_uplift_target"]
        & summary["meets_sell_through_target"]
        & summary["meets_spoilage_target"]
    )

    return {
        "empty": False,
        "best_policy": best["policy"],
        "best_policy_label": display_name(best["policy"]),
        "best_mean_revenue": round(float(best["mean_revenue"]), 2),
        "best_sell_through_pct": round(float(best["mean_sell_through_pct"]), 2),
        "policies_meeting_all_targets": int(meets_all.sum()),
        "policies_evaluated": int(len(summary)),
    }
