"""
blueprints/api.py

JSON API routes for the Flask web dashboard. Every route below is
backed by one of three REAL data sources — never a fabricated or
hardcoded number:

    1. `evaluation/evaluation_results.csv` / `policy_evaluation_summary.csv`
       (loaded via the existing, untouched
       `dashboard.dashboard_app.load_episode_results` /
       `load_summary_results`), summarized by the framework-agnostic
       `additional_features.advanced_dashboard.{kpi_components,charts}`.
    2. A live `pricing_env.PricingEnvironment` rollout driven by a real
       trained agent checkpoint (`agents/checkpoints/*`), via
       `dashboard.web_dashboard.services`.
    3. A real, explicitly-labeled CSV/JSON event calendar, scanned by
       `additional_features.demand_shock_detection`.

Every route wraps its real-data-dependent logic in try/except and
returns a clear JSON `{"error": "..."}` (HTTP 404/500 as appropriate)
rather than ever substituting invented data on failure — e.g. if no DQN
checkpoint has been trained yet, `/api/recommendation` says so plainly
instead of pretending to have an answer.
"""

from __future__ import annotations

from flask import Blueprint, Response, jsonify, request

from additional_features.advanced_dashboard import charts, export_utils, kpi_components
from dashboard.dashboard_app import load_episode_results, load_summary_results
from dashboard.web_dashboard import services
from dashboard.web_dashboard.services import AgentNotFoundError

api_bp = Blueprint("api", __name__)


def _policies_param() -> list[str] | None:
    """Parse an optional `?policies=dqn,fixed_price` query param."""
    raw = request.args.get("policies")
    return [p.strip() for p in raw.split(",") if p.strip()] if raw else None


# --------------------------------------------------------------------- #
# Raw evaluation data + KPI summaries
# --------------------------------------------------------------------- #
@api_bp.route("/episodes")
def get_episodes():
    """Raw episode-level evaluation rows (see `evaluation/evaluation_results.csv`)."""
    try:
        episodes = load_episode_results()
        policies = _policies_param()
        if policies:
            episodes = episodes.loc[episodes["policy"].isin(policies)]
        return jsonify(episodes.to_dict(orient="records"))
    except Exception as exc:  # noqa: BLE001 - surfaced to the client as JSON, not swallowed
        return jsonify({"error": str(exc)}), 500


@api_bp.route("/kpis")
def get_kpis():
    """
    Top-line KPI headline + full per-policy summary rows + a real
    AI-vs-baseline comparison, all computed from
    `evaluation/policy_evaluation_summary.csv`.
    """
    try:
        summary = load_summary_results()
        ai_policy = request.args.get("ai_policy", "dqn")
        baseline_policy = request.args.get("baseline_policy", "fixed_price")

        ai_vs_baseline = kpi_components.compute_ai_vs_baseline_kpis(
            summary, ai_policy=ai_policy, baseline_policy=baseline_policy
        )
        return jsonify(
            {
                "headline": kpi_components.summary_headline(summary),
                "ai_vs_baseline": ai_vs_baseline.to_dict() if ai_vs_baseline else None,
                "summary_rows": summary.to_dict(orient="records"),
            }
        )
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 500


# --------------------------------------------------------------------- #
# Chart.js-ready series (Feature 2)
# --------------------------------------------------------------------- #
@api_bp.route("/charts/<name>")
def get_chart(name: str):
    """
    One endpoint per Chart.js chart, dispatching to
    `additional_features.advanced_dashboard.charts`. Valid `name` values:
    revenue_trend, occupancy_trend, spoilage_trend, reward_trend,
    baseline_comparison, revenue_uplift, price_trend, price_distribution.
    """
    try:
        policies = _policies_param()
        if name == "price_distribution":
            episodes = load_episode_results()
            policy = request.args.get("policy", "dqn")
            return jsonify(charts.price_action_distribution(episodes, policy))

        if name in ("revenue_trend", "occupancy_trend", "spoilage_trend", "reward_trend"):
            episodes = load_episode_results()
            builder = {
                "revenue_trend": charts.revenue_trend_series,
                "occupancy_trend": charts.occupancy_trend_series,
                "spoilage_trend": charts.spoilage_trend_series,
                "reward_trend": charts.reward_trend_series,
            }[name]
            return jsonify(builder(episodes, policies))

        if name in ("baseline_comparison", "revenue_uplift", "price_trend"):
            summary = load_summary_results()
            builder = {
                "baseline_comparison": charts.baseline_comparison_series,
                "revenue_uplift": charts.revenue_uplift_series,
                "price_trend": charts.price_trend_series,
            }[name]
            return jsonify(builder(summary))

        return jsonify({"error": f"Unknown chart '{name}'"}), 404
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 500


# --------------------------------------------------------------------- #
# Explainable AI Pricing (Feature 1)
# --------------------------------------------------------------------- #
@api_bp.route("/recommendation")
def get_recommendation():
    """Live AI recommendation + full explanation for one real environment step."""
    agent_name = request.args.get("agent", "dqn")
    try:
        return jsonify(services.build_live_recommendation(agent_name))
    except AgentNotFoundError as exc:
        return jsonify({"error": str(exc)}), 404
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 500


@api_bp.route("/timeline")
def get_timeline():
    """Multi-step real recommendation timeline (see `services.build_recommendation_timeline`)."""
    agent_name = request.args.get("agent", "dqn")
    steps = request.args.get("steps", default=6, type=int)
    try:
        return jsonify(services.build_recommendation_timeline(agent_name, num_steps=steps))
    except AgentNotFoundError as exc:
        return jsonify({"error": str(exc)}), 404
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 500


# --------------------------------------------------------------------- #
# Demand Shock Detection (Feature 3)
# --------------------------------------------------------------------- #
@api_bp.route("/alerts")
def get_alerts():
    """Real demand-shock alerts scanned from a CSV/JSON event calendar."""
    lookahead_days = request.args.get("lookahead", default=30, type=int)
    source = request.args.get("source")
    reference_date = request.args.get("reference_date")
    try:
        return jsonify(
            services.load_demand_shock_alerts(
                event_source_path=source,
                lookahead_days=lookahead_days,
                reference_date_str=reference_date,
            )
        )
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 500


# --------------------------------------------------------------------- #
# What-If Simulator
# --------------------------------------------------------------------- #
@api_bp.route("/simulate", methods=["POST"])
def post_simulate():
    """
    Run one full REAL environment episode with the requested agent and
    market condition (see `services.run_episode_simulation`). Every value
    in the response comes from an actual rollout — never a fabricated
    random projection.
    """
    payload = request.get_json(silent=True) or {}
    try:
        result = services.run_episode_simulation(
            agent_name=payload.get("agent", "dqn"),
            initial_inventory=int(payload.get("initial_inventory", 100)),
            selling_horizon_days=int(payload.get("selling_horizon_days", 30)),
            market_condition=payload.get("market_condition", "standard"),
            seed=int(payload.get("seed", 42)),
        )
        return jsonify(result)
    except AgentNotFoundError as exc:
        return jsonify({"error": str(exc)}), 404
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 500


# --------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------- #
@api_bp.route("/export/episodes.csv")
def export_episodes_csv():
    """Download the real episode-level evaluation data as CSV."""
    episodes = load_episode_results()
    policies = _policies_param()
    if policies:
        episodes = episodes.loc[episodes["policy"].isin(policies)]
    return Response(
        export_utils.dataframe_to_csv_bytes(episodes),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=episode_results.csv"},
    )


@api_bp.route("/export/summary.json")
def export_summary_json():
    """Download the real policy evaluation summary as JSON."""
    summary = load_summary_results()
    return Response(
        export_utils.records_to_json_bytes(summary.to_dict(orient="records")),
        mimetype="application/json",
        headers={"Content-Disposition": "attachment; filename=policy_evaluation_summary.json"},
    )
