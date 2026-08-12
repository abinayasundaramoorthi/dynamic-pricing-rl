"""
test_advanced_dashboard.py

Validation tests for Feature 2 — Advanced Revenue Management Dashboard
Enhancement — now built on Flask + Chart.js instead of Streamlit.

Covers:
  1. `kpi_components` pure computation against the project's REAL
     `evaluation/policy_evaluation_summary.csv`.
  2. Every `charts.py` series builder returns the expected Chart.js-ready
     shape when given the project's REAL episode/summary DataFrames.
  3. `charts.demand_forecast_series` runs against the project's REAL
     `DemandSimulator`.
  4. `export_utils` CSV/JSON round-trip correctly.
  5. `dashboard.web_dashboard.services` — real agent loading, real
     environment rollouts (recommendation / timeline / simulation), and
     real event-file scanning.
  6. `dashboard.web_dashboard.app` — every HTML page route and every
     JSON API route, exercised through Flask's test client.

Run with:
    python -m pytest additional_features/tests/test_advanced_dashboard.py -v
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from additional_features.advanced_dashboard import charts, export_utils, kpi_components
from dashboard.dashboard_app import load_episode_results, load_summary_results
from dashboard.web_dashboard import services
from dashboard.web_dashboard.app import create_app
from pricing_env import PricingEnvConfig
from pricing_env.demand_simulator import DemandSimulator


@pytest.fixture(scope="module")
def episodes() -> pd.DataFrame:
    return load_episode_results()


@pytest.fixture(scope="module")
def summary() -> pd.DataFrame:
    return load_summary_results()


@pytest.fixture(scope="module")
def client():
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


# --------------------------------------------------------------------------- #
# 1. kpi_components — pure computation
# --------------------------------------------------------------------------- #
def test_compute_ai_vs_baseline_kpis_with_real_summary(summary: pd.DataFrame):
    kpis = kpi_components.compute_ai_vs_baseline_kpis(summary, ai_policy="dqn", baseline_policy="fixed_price")
    assert kpis is not None, "'dqn' and 'fixed_price' are expected to both be present in this repo's summary"
    assert isinstance(kpis.revenue_uplift_pct, float)
    assert isinstance(kpis.ai_meets_all_targets, bool)
    assert kpis.to_dict()["ai_policy_label"]  # real display_name reuse, never empty


def test_compute_ai_vs_baseline_kpis_returns_none_for_missing_policy(summary: pd.DataFrame):
    kpis = kpi_components.compute_ai_vs_baseline_kpis(
        summary, ai_policy="a_policy_that_does_not_exist", baseline_policy="fixed_price"
    )
    assert kpis is None, "must return None rather than fabricate a comparison for a missing policy"


def test_summary_headline_with_real_summary(summary: pd.DataFrame):
    headline = kpi_components.summary_headline(summary)
    assert headline["empty"] is False
    assert headline["policies_evaluated"] == len(summary)


def test_summary_headline_empty_dataframe():
    headline = kpi_components.summary_headline(summary=pd.DataFrame(columns=["policy", "mean_revenue"]))
    assert headline == {"empty": True}


# --------------------------------------------------------------------------- #
# 2. charts.py — every series builder against real data
# --------------------------------------------------------------------------- #
def test_revenue_trend_series(episodes: pd.DataFrame):
    series = charts.revenue_trend_series(episodes)
    assert "labels" in series and "datasets" in series
    assert len(series["datasets"]) == episodes["policy"].nunique()


def test_occupancy_trend_series(episodes: pd.DataFrame):
    series = charts.occupancy_trend_series(episodes)
    assert all(0 <= v <= 100 for d in series["datasets"] for v in d["data"])


def test_spoilage_trend_series(episodes: pd.DataFrame):
    series = charts.spoilage_trend_series(episodes)
    assert all(v >= 0 for d in series["datasets"] for v in d["data"])


def test_reward_trend_series(episodes: pd.DataFrame):
    series = charts.reward_trend_series(episodes)
    assert len(series["datasets"]) >= 1


def test_price_action_distribution(episodes: pd.DataFrame):
    dist = charts.price_action_distribution(episodes, "dqn")
    assert dist["labels"] == ["Discount", "Hold", "Premium"]
    assert sum(dist["data"]) == dist["total_episodes"]


def test_price_action_distribution_missing_policy(episodes: pd.DataFrame):
    dist = charts.price_action_distribution(episodes, "a_policy_that_does_not_exist")
    assert dist["total_episodes"] == 0
    assert dist["data"] == [0, 0, 0]


def test_baseline_comparison_series(summary: pd.DataFrame):
    series = charts.baseline_comparison_series(summary)
    assert len(series["labels"]) == len(summary)
    assert len(series["mean_revenue"]) == len(series["std_revenue"]) == len(summary)


def test_revenue_uplift_series(summary: pd.DataFrame):
    series = charts.revenue_uplift_series(summary)
    assert len(series["data"]) == len(summary)


def test_price_trend_series(summary: pd.DataFrame):
    series = charts.price_trend_series(summary)
    assert len(series["data"]) == len(summary)


def test_charts_respect_policy_subset_filter(episodes: pd.DataFrame):
    all_policies = sorted(episodes["policy"].unique())
    assert len(all_policies) >= 2, "test repo fixture is expected to have >=2 evaluated policies"
    series = charts.revenue_trend_series(episodes, policies=[all_policies[0]])
    assert len(series["datasets"]) == 1


# --------------------------------------------------------------------------- #
# 3. charts.py — demand forecast against the real DemandSimulator
# --------------------------------------------------------------------------- #
def test_demand_forecast_series_runs_against_real_simulator():
    env_config = PricingEnvConfig()
    simulator = DemandSimulator(env_config.demand)

    series = charts.demand_forecast_series(
        demand_simulator=simulator,
        price=env_config.base_price,
        reference_price=env_config.base_price,
        remaining_inventory=env_config.initial_inventory,
        days_remaining=5,
        selling_horizon_days=env_config.selling_horizon_days,
        seed=0,
        num_simulations=20,  # small for test speed
    )
    assert len(series["labels"]) == len(series["data"]) == 5
    assert all(v >= 0 for v in series["data"]), "expected demand intensity must be non-negative"


# --------------------------------------------------------------------------- #
# 4. export_utils.py
# --------------------------------------------------------------------------- #
def test_dataframe_to_csv_bytes_round_trips():
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    csv_bytes = export_utils.dataframe_to_csv_bytes(df)
    assert isinstance(csv_bytes, bytes)
    roundtripped = pd.read_csv(pd.io.common.BytesIO(csv_bytes))
    pd.testing.assert_frame_equal(df, roundtripped)


def test_records_to_json_bytes_round_trips():
    records = [{"name": "Event A", "impact": 1.25}]
    json_bytes = export_utils.records_to_json_bytes(records)
    assert json.loads(json_bytes.decode("utf-8")) == records


# --------------------------------------------------------------------------- #
# 5. dashboard.web_dashboard.services — real agents, real rollouts
# --------------------------------------------------------------------------- #
def test_build_live_recommendation_dqn():
    result = services.build_live_recommendation("dqn")
    assert "recommendation_summary" in result
    assert result["agent"] == "dqn"


def test_build_live_recommendation_q_learning():
    result = services.build_live_recommendation("q_learning")
    assert result["agent"] == "q_learning"


def test_build_recommendation_timeline_multi_step():
    timeline = services.build_recommendation_timeline("dqn", num_steps=3)
    assert len(timeline) <= 3
    assert all("recommendation_summary" in step for step in timeline)
    steps_seen = [t["step"] for t in timeline]
    assert steps_seen == sorted(steps_seen)


def test_run_episode_simulation_standard():
    result = services.run_episode_simulation(
        "dqn", initial_inventory=20, selling_horizon_days=7, market_condition="standard"
    )
    assert result["final_revenue"] >= 0
    assert len(result["days"]) <= 7
    assert result["days"][-1]["day"] == len(result["days"])


def test_run_episode_simulation_rejects_unknown_market_condition():
    with pytest.raises(ValueError):
        services.run_episode_simulation(market_condition="competitor_price_war")


def test_load_demand_shock_alerts_uses_sample_by_default():
    result = services.load_demand_shock_alerts(lookahead_days=60)
    assert result["is_sample_data"] is True
    assert "sample_events.csv" in result["source"]


# --------------------------------------------------------------------------- #
# 6. dashboard.web_dashboard.app — Flask routes, exercised end to end
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("route", ["/", "/dashboard", "/recommendations", "/events", "/simulator"])
def test_html_pages_return_200(client, route):
    resp = client.get(route)
    assert resp.status_code == 200


def test_unknown_page_returns_404(client):
    resp = client.get("/this-route-does-not-exist")
    assert resp.status_code == 404


def test_api_kpis(client):
    resp = client.get("/api/kpis")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "headline" in data and "summary_rows" in data


@pytest.mark.parametrize(
    "chart_name",
    ["revenue_trend", "occupancy_trend", "spoilage_trend", "reward_trend", "baseline_comparison", "revenue_uplift", "price_trend"],
)
def test_api_charts(client, chart_name):
    resp = client.get(f"/api/charts/{chart_name}")
    assert resp.status_code == 200


def test_api_chart_unknown_returns_404(client):
    resp = client.get("/api/charts/does_not_exist")
    assert resp.status_code == 404


def test_api_recommendation(client):
    resp = client.get("/api/recommendation?agent=dqn")
    assert resp.status_code == 200
    assert "recommendation_summary" in resp.get_json()


def test_api_timeline(client):
    resp = client.get("/api/timeline?agent=dqn&steps=2")
    assert resp.status_code == 200
    assert isinstance(resp.get_json(), list)


def test_api_alerts(client):
    resp = client.get("/api/alerts?lookahead=60")
    assert resp.status_code == 200
    assert "alerts" in resp.get_json()


def test_api_simulate_success(client):
    resp = client.post(
        "/api/simulate",
        json={"agent": "dqn", "initial_inventory": 20, "selling_horizon_days": 7, "market_condition": "low_demand"},
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert "final_revenue" in data and "days" in data


def test_api_simulate_bad_market_condition(client):
    resp = client.post("/api/simulate", json={"market_condition": "not_a_real_condition"})
    assert resp.status_code == 400


def test_api_export_episodes_csv(client):
    resp = client.get("/api/export/episodes.csv")
    assert resp.status_code == 200
    assert resp.mimetype == "text/csv"


def test_api_export_summary_json(client):
    resp = client.get("/api/export/summary.json")
    assert resp.status_code == 200
    assert resp.mimetype == "application/json"
