"""
dashboard_app.py

Business-facing results dashboard (issue #102 — "Integrate Business
Dashboard and Visualization Pipeline"), the final Week 4 deliverable
referenced in `README.md` ("Week 4: ... business-facing dashboard") and
`reports/project_planning/problem_statement.md` Section 7.1 ("a
lightweight results dashboard").

This module does NOT run any simulations itself. It is purely a
consumption/visualization layer on top of the outputs `evaluation/
evaluate_policies.py` (#90, #98) already produces:

    evaluation/evaluation_results.csv        — episode-level results, all policies
    evaluation/policy_evaluation_summary.csv — per-policy aggregate stats + KPI flags

That separation is deliberate: the evaluation pipeline's job is to run
1,000-episode simulations correctly (#90/#98's concern); the dashboard's
job is to make what it already produced legible to a non-technical
stakeholder — the "Revenue Manager" persona in the problem statement's
stakeholder table, who "reviews the RL agent's pricing strategy on a
monitoring dashboard" but is not expected to read a CSV.

Run with:
    streamlit run dashboard/dashboard_app.py

See `reports/dashboard_design.md` for the full architecture write-up,
including the compatibility contract this module relies on and how it
degrades when evaluation hasn't been run yet.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

# Allow running via `streamlit run dashboard/dashboard_app.py` from the
# repo root without needing dashboard/ on sys.path already — mirrors how
# `evaluation/evaluate_policies.py` is runnable via `python -m` from the
# repo root without extra setup.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from configs.evaluation_config import BusinessKPIConfig  # noqa: E402

# --------------------------------------------------------------------------- #
# Data contract — the exact files/columns this dashboard depends on.
# Kept as module-level constants so `reports/dashboard_design.md`'s
# "compatibility contract" section and this code can never silently drift
# apart from each other.
# --------------------------------------------------------------------------- #
EVALUATION_RESULTS_PATH = REPO_ROOT / "evaluation" / "evaluation_results.csv"
EVALUATION_SUMMARY_PATH = REPO_ROOT / "evaluation" / "policy_evaluation_summary.csv"

REQUIRED_EPISODE_COLUMNS = {
    "policy", "seed", "total_reward", "final_revenue", "units_sold",
    "initial_inventory", "sell_through_pct", "spoilage_pct", "mean_price",
    "mean_discount_depth_pct", "steps",
}
REQUIRED_SUMMARY_COLUMNS = {
    "policy", "num_episodes", "mean_revenue", "std_revenue",
    "mean_sell_through_pct", "mean_spoilage_pct", "mean_price",
    "mean_discount_depth_pct", "revenue_uplift_pct",
    "meets_revenue_uplift_target", "meets_sell_through_target",
    "meets_spoilage_target",
}

POLICY_DISPLAY_NAMES = {
    "dqn": "DQN Agent",
    "q_learning": "Q-Learning Agent",
    "random": "Random Policy",
    "fixed_price": "Fixed Price Policy",
    "time_based_discount": "Time-Based Discount Policy",
}


# --------------------------------------------------------------------------- #
# Data loading + compatibility verification
# --------------------------------------------------------------------------- #
class DataCompatibilityError(Exception):
    """Raised when an evaluation output file exists but doesn't match the
    schema this dashboard expects — e.g. produced by an older/incompatible
    version of `evaluate_policies.py`. Distinguished from a plain
    FileNotFoundError so the UI can give a different, more specific
    message for "wrong shape of data" versus "no data yet"."""


@st.cache_data(show_spinner="Loading evaluation results...")
def load_episode_results(path: Path = EVALUATION_RESULTS_PATH) -> pd.DataFrame:
    """Load the episode-level results CSV, verifying it has the columns
    this dashboard's charts and tables actually read before returning it."""
    if not path.exists():
        raise FileNotFoundError(str(path))
    df = pd.read_csv(path)
    missing = REQUIRED_EPISODE_COLUMNS - set(df.columns)
    if missing:
        raise DataCompatibilityError(
            f"{path} is missing expected column(s) {sorted(missing)}. "
            "This usually means it was produced by an older version of "
            "evaluation/evaluate_policies.py — re-run the evaluation "
            "pipeline to regenerate it."
        )
    return df


@st.cache_data(show_spinner="Loading policy summary...")
def load_summary_results(path: Path = EVALUATION_SUMMARY_PATH) -> pd.DataFrame:
    """Load the per-policy aggregate summary CSV, verifying schema
    compatibility the same way `load_episode_results` does."""
    if not path.exists():
        raise FileNotFoundError(str(path))
    df = pd.read_csv(path)
    missing = REQUIRED_SUMMARY_COLUMNS - set(df.columns)
    if missing:
        raise DataCompatibilityError(
            f"{path} is missing expected column(s) {sorted(missing)}. "
            "This usually means it was produced by an older version of "
            "evaluation/evaluate_policies.py — re-run the evaluation "
            "pipeline to regenerate it."
        )
    return df


def verify_cross_source_compatibility(episodes: pd.DataFrame, summary: pd.DataFrame) -> None:
    """
    Confirm the two evaluation outputs and the business-KPI config
    actually agree with each other before rendering anything from them.

    This is the concrete implementation of issue #102's "Verify
    compatibility between Policy Evaluation, Business Metrics, and
    Visualization Components" task: it checks that the same set of
    policies appears in both files, that per-policy episode counts match
    between the raw episode data and the aggregate summary (catching a
    partially-written or stale file), and that `BusinessKPIConfig`
    (Business Metrics) targets are consistent with the KPI flag columns
    the summary file already computed (Policy Evaluation output) rather
    than silently trusting two independently-computed sources of truth.

    Raises `DataCompatibilityError` on any mismatch, with a message
    specific enough to fix without needing to read this function's code.
    """
    episode_policies = set(episodes["policy"].unique())
    summary_policies = set(summary["policy"].unique())
    if episode_policies != summary_policies:
        raise DataCompatibilityError(
            "Policy set mismatch between evaluation_results.csv "
            f"({sorted(episode_policies)}) and policy_evaluation_summary.csv "
            f"({sorted(summary_policies)}) — these two files must come from "
            "the same evaluation run. Re-run "
            "`python -m evaluation.evaluate_policies` to regenerate both "
            "together."
        )

    episode_counts = episodes.groupby("policy").size()
    for _, row in summary.iterrows():
        policy = row["policy"]
        expected = int(row["num_episodes"])
        actual = int(episode_counts.get(policy, 0))
        if actual != expected:
            raise DataCompatibilityError(
                f"Episode count mismatch for policy {policy!r}: summary "
                f"reports num_episodes={expected} but evaluation_results.csv "
                f"contains {actual} rows for that policy. This indicates a "
                "partially-written or stale results file — re-run the "
                "evaluation pipeline."
            )

    # Business-KPI target consistency: re-derive the pass/fail flags from
    # BusinessKPIConfig's defaults and confirm they agree with what the
    # summary file already computed. A mismatch here would mean the
    # dashboard's business-metrics targets have drifted from the config
    # the evaluation run actually used.
    kpis = BusinessKPIConfig()
    recomputed_sell_through = summary["mean_sell_through_pct"] >= kpis.target_sell_through_pct
    recomputed_spoilage = summary["mean_spoilage_pct"] <= kpis.max_spoilage_pct
    if not (recomputed_sell_through == summary["meets_sell_through_target"]).all():
        raise DataCompatibilityError(
            "Sell-through KPI flags in policy_evaluation_summary.csv do not "
            "match BusinessKPIConfig's current target_sell_through_pct "
            f"({kpis.target_sell_through_pct}%). Re-run the evaluation "
            "pipeline after any BusinessKPIConfig change so the flags stay "
            "in sync."
        )
    if not (recomputed_spoilage == summary["meets_spoilage_target"]).all():
        raise DataCompatibilityError(
            "Spoilage KPI flags in policy_evaluation_summary.csv do not "
            f"match BusinessKPIConfig's current max_spoilage_pct "
            f"({kpis.max_spoilage_pct}%). Re-run the evaluation pipeline "
            "after any BusinessKPIConfig change so the flags stay in sync."
        )


def display_name(policy: str) -> str:
    """Human-readable label for a policy key, falling back to the raw key
    for any policy this dashboard doesn't have a friendly name for yet
    (keeps the dashboard forward-compatible with a new policy being added
    to evaluate_policies.py without a matching dashboard code change)."""
    return POLICY_DISPLAY_NAMES.get(policy, policy)


# --------------------------------------------------------------------------- #
# Section renderers
# --------------------------------------------------------------------------- #
def render_header(summary: pd.DataFrame) -> None:
    st.title("Dynamic Pricing — Policy Performance Dashboard")
    st.caption(
        "Business-facing view of the 1,000-simulated-season evaluation "
        "(`evaluation/evaluate_policies.py`). Every metric below is "
        "computed from that evaluation run — this dashboard renders it, "
        "it does not simulate anything itself."
    )

    best = summary.sort_values("mean_revenue", ascending=False).iloc[0]
    reference = summary.loc[summary["revenue_uplift_pct"] == 0.0]
    reference_name = display_name(reference["policy"].iloc[0]) if not reference.empty else "N/A"

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Top policy (by mean revenue)", display_name(best["policy"]))
    col2.metric("Top policy mean revenue", f"${best['mean_revenue']:,.2f}")
    col3.metric("Reference baseline", reference_name)
    col4.metric("Policies evaluated", len(summary))


def render_policy_performance(summary: pd.DataFrame) -> None:
    st.header("Policy Performance")
    st.write(
        "Mean revenue per simulated booking season, with variability "
        "(±1 std) across all 1,000 episodes."
    )

    chart_df = summary[["policy", "mean_revenue", "std_revenue"]].copy()
    chart_df["policy"] = chart_df["policy"].map(display_name)
    chart_df = chart_df.sort_values("mean_revenue", ascending=False).set_index("policy")
    st.bar_chart(chart_df["mean_revenue"])

    st.subheader("Full comparison table")
    display_cols = [
        "policy", "mean_revenue", "std_revenue", "revenue_uplift_pct",
        "mean_sell_through_pct", "mean_spoilage_pct",
        "meets_revenue_uplift_target", "meets_sell_through_target",
        "meets_spoilage_target",
    ]
    table = summary[display_cols].copy()
    table["policy"] = table["policy"].map(display_name)
    table = table.sort_values("mean_revenue", ascending=False)
    table.columns = [
        "Policy", "Mean Revenue ($)", "Std Revenue ($)", "Revenue Uplift (%)",
        "Sell-Through (%)", "Spoilage (%)",
        "Meets Revenue Target", "Meets Sell-Through Target", "Meets Spoilage Target",
    ]
    st.dataframe(table, width="stretch", hide_index=True)


def render_pricing_trends(summary: pd.DataFrame, episodes: pd.DataFrame) -> None:
    st.header("Pricing Trends")
    st.write(
        "How aggressively each policy discounts off the environment's "
        "base price, and the resulting mean price actually charged, "
        "averaged across all simulated seasons."
    )

    col1, col2 = st.columns(2)
    with col1:
        price_df = summary[["policy", "mean_price"]].copy()
        price_df["policy"] = price_df["policy"].map(display_name)
        price_df = price_df.sort_values("mean_price", ascending=False).set_index("policy")
        st.caption("Mean price charged")
        st.bar_chart(price_df["mean_price"])
    with col2:
        discount_df = summary[["policy", "mean_discount_depth_pct"]].copy()
        discount_df["policy"] = discount_df["policy"].map(display_name)
        discount_df = discount_df.sort_values(
            "mean_discount_depth_pct", ascending=False
        ).set_index("policy")
        st.caption("Mean discount depth (%)")
        st.bar_chart(discount_df["mean_discount_depth_pct"])

    with st.expander("Revenue distribution across all 1,000 simulated seasons per policy"):
        selected = st.multiselect(
            "Policies to show",
            options=sorted(episodes["policy"].unique()),
            default=sorted(episodes["policy"].unique()),
            format_func=display_name,
        )
        if selected:
            pivot = episodes[episodes["policy"].isin(selected)].pivot(
                columns="policy", values="final_revenue"
            )
            pivot.columns = [display_name(c) for c in pivot.columns]
            st.line_chart(pivot.reset_index(drop=True))
            st.caption(
                "Each line is one policy's per-episode revenue across all "
                "simulated seasons (x-axis = episode index, not time) — "
                "useful for eyeballing variance, not a real time series."
            )


def render_business_metrics(summary: pd.DataFrame, kpis: BusinessKPIConfig) -> None:
    st.header("Business Metrics")
    st.write(
        f"Scored against this project's own success criteria: revenue "
        f"uplift > {kpis.target_revenue_uplift_pct:.0f}% over the fixed-price "
        f"baseline, sell-through > {kpis.target_sell_through_pct:.0f}%, "
        f"spoilage < {kpis.max_spoilage_pct:.0f}%."
    )

    for _, row in summary.sort_values("mean_revenue", ascending=False).iterrows():
        with st.container(border=True):
            st.subheader(display_name(row["policy"]))
            c1, c2, c3 = st.columns(3)
            c1.metric(
                "Revenue Uplift",
                f"{row['revenue_uplift_pct']:.1f}%",
                delta="✅ Target met" if row["meets_revenue_uplift_target"] else "❌ Below target",
                delta_color="off",
            )
            c2.metric(
                "Sell-Through",
                f"{row['mean_sell_through_pct']:.1f}%",
                delta="✅ Target met" if row["meets_sell_through_target"] else "❌ Below target",
                delta_color="off",
            )
            c3.metric(
                "Spoilage",
                f"{row['mean_spoilage_pct']:.1f}%",
                delta="✅ Target met" if row["meets_spoilage_target"] else "❌ Above target",
                delta_color="off",
            )


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    st.set_page_config(
        page_title="Dynamic Pricing — Policy Dashboard",
        layout="wide",
    )

    try:
        episodes = load_episode_results()
        summary = load_summary_results()
        verify_cross_source_compatibility(episodes, summary)
    except FileNotFoundError as exc:
        st.title("Dynamic Pricing — Policy Performance Dashboard")
        st.warning(
            f"No evaluation results found at `{exc}`.\n\n"
            "Run the evaluation pipeline first:\n\n"
            "```bash\npython -m evaluation.evaluate_policies\n```\n\n"
            "Then reload this dashboard."
        )
        return
    except DataCompatibilityError as exc:
        st.title("Dynamic Pricing — Policy Performance Dashboard")
        st.error(f"Evaluation output is incompatible with this dashboard:\n\n{exc}")
        return

    with st.sidebar:
        st.header("Filters")
        selected_policies = st.multiselect(
            "Policies to display",
            options=sorted(summary["policy"].unique()),
            default=sorted(summary["policy"].unique()),
            format_func=display_name,
        )
        if st.button("Reload evaluation data"):
            load_episode_results.clear()
            load_summary_results.clear()
            st.rerun()

    if not selected_policies:
        st.info("Select at least one policy in the sidebar to view results.")
        return

    filtered_summary = summary[summary["policy"].isin(selected_policies)]
    filtered_episodes = episodes[episodes["policy"].isin(selected_policies)]

    render_header(filtered_summary)
    render_policy_performance(filtered_summary)
    render_pricing_trends(filtered_summary, filtered_episodes)
    render_business_metrics(filtered_summary, BusinessKPIConfig())


if __name__ == "__main__":
    main()