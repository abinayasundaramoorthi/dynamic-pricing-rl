"""
deployment_smoke_test.py

Standalone, dependency-light smoke test suitable for a CI/CD deployment
gate (e.g. a pre-deploy step in a GitHub Actions / GitLab CI pipeline,
or a Docker `HEALTHCHECK`/init-container check) — separate from the
`pytest` suite in `additional_features/tests/`, which is for
development-time correctness testing.

This script answers one question: "is `additional_features` safe to
ship in this build?" It:

  1. Imports every public module of all three features (catches broken
     imports / missing dependencies before they reach production).
  2. Runs one real `PricingEnvironment` episode end-to-end and builds a
     `PricingExplanation` for each step (catches runtime breakage in the
     explainability pipeline against the actual environment contract).
  3. Loads whichever trained checkpoint(s) are present in
     `agents/checkpoints/` and verifies `dqn_state_sensitivity` /
     confidence scoring run against them (skips, rather than fails, if
     no checkpoint has been trained yet in this build — matching
     `evaluation/evaluate_policies.py`'s existing "missing checkpoint is
     a warning, not a hard failure" policy).
  4. Loads the real `evaluation/*.csv` files (if present) and builds
     every `advanced_dashboard.charts` series + KPI computation against
     them.
  5. Boots the real Flask app (`dashboard.web_dashboard.app.create_app`)
     with a test client and hits every page route and the core API
     routes, including a real `/api/simulate` POST.
  6. Runs `DemandShockDetector.scan()` against a minimal in-memory event
     list (no file I/O needed for this check).

Exit code 0 = safe to deploy. Exit code 1 = do not deploy; see stderr
for exactly which check failed.

Run with:
    python -m additional_features.deployment_smoke_test
"""

from __future__ import annotations

import sys
import traceback
from datetime import date, timedelta
from pathlib import Path
from typing import Callable, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]

CheckResult = Tuple[str, bool, str]


def _run_check(name: str, fn: Callable[[], str]) -> CheckResult:
    try:
        detail = fn()
        return name, True, detail
    except Exception as exc:  # noqa: BLE001 - deliberately broad: this is a smoke-test gate
        return name, False, f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"


def check_imports() -> str:
    import additional_features  # noqa: F401
    import additional_features.advanced_dashboard  # noqa: F401
    import additional_features.demand_shock_detection  # noqa: F401
    import additional_features.explainable_ai_pricing  # noqa: F401

    return "all additional_features submodules imported successfully"


def check_explainer_against_live_environment() -> str:
    from additional_features.explainable_ai_pricing import PricingExplainer
    from pricing_env import PricingEnvConfig, PricingEnvironment

    env = PricingEnvironment(PricingEnvConfig())
    obs, info = env.reset(seed=2026)
    previous_price = info["current_price"]
    explainer = PricingExplainer()

    steps_explained = 0
    for _ in range(5):
        next_obs, reward, terminated, truncated, next_info = env.step(3)
        explanation = explainer.explain(observation=next_obs, info=next_info, previous_price=previous_price)
        assert explanation.recommendation_summary, "empty recommendation_summary"
        assert len(explanation.factors) > 0, "no factors extracted"
        json_payload = explanation.to_json()
        assert json_payload.startswith("{"), "to_json() did not produce a JSON object"
        previous_price = next_info["price"]
        steps_explained += 1
        obs = next_obs
        if terminated or truncated:
            break

    return f"explained {steps_explained} real environment step(s) without error"


def check_dqn_checkpoint() -> str:
    from configs.evaluation_config import get_default_evaluation_config
    from pricing_env import PricingEnvConfig, PricingEnvironment

    eval_config = get_default_evaluation_config()
    checkpoint = REPO_ROOT / eval_config.dqn_checkpoint_path
    if not checkpoint.exists():
        return f"SKIPPED — no DQN checkpoint at {checkpoint} (train one with `python -m training.train_dqn`)"

    import torch

    from additional_features.explainable_ai_pricing import dqn_state_sensitivity
    from agents.dqn_agent import DQNAgent

    env = PricingEnvironment(PricingEnvConfig())
    agent = DQNAgent.load(
        checkpoint,
        observation_space=env.observation_space,
        action_space=env.action_space,
        hidden_layer_sizes=eval_config.dqn_hidden_layer_sizes,
        device="cpu",
    )
    obs, _ = env.reset(seed=2026)
    action = agent.select_greedy_action(obs)
    sensitivities = dqn_state_sensitivity(agent.q_network, obs, action)
    assert set(sensitivities.keys()) == {"remaining_inventory", "days_remaining"}
    return f"real DQN checkpoint loaded and sensitivity-analyzed ({checkpoint.name})"


def check_q_learning_checkpoint() -> str:
    from configs.evaluation_config import get_default_evaluation_config
    from pricing_env import PricingEnvConfig, PricingEnvironment

    eval_config = get_default_evaluation_config()
    checkpoint = REPO_ROOT / eval_config.q_learning_checkpoint_path
    if not checkpoint.exists():
        return (
            f"SKIPPED — no Q-Learning checkpoint at {checkpoint} "
            "(train one with `python -m training.train_agent --agent q_learning`)"
        )

    from additional_features.explainable_ai_pricing import confidence_from_action_values
    from agents.q_learning_agent import QLearningAgent

    env = PricingEnvironment(PricingEnvConfig())
    agent = QLearningAgent.load(checkpoint, action_space=env.action_space)
    obs, _ = env.reset(seed=2026)
    action = agent.select_greedy_action(obs)
    state_key = tuple(round(float(x)) for x in obs)
    confidence = confidence_from_action_values(agent.q_table[state_key], action)
    assert 0.0 <= confidence.score <= 1.0
    return f"real Q-Learning checkpoint loaded and confidence-scored ({checkpoint.name})"


def check_dashboard_against_real_csvs() -> str:
    from dashboard.dashboard_app import load_episode_results, load_summary_results

    try:
        episodes = load_episode_results()
        summary = load_summary_results()
    except FileNotFoundError as exc:
        return f"SKIPPED — evaluation CSVs not present yet ({exc})"

    from additional_features.advanced_dashboard import charts, kpi_components

    headline = kpi_components.summary_headline(summary)
    assert headline["empty"] is False
    kpi_components.compute_ai_vs_baseline_kpis(summary, ai_policy="dqn", baseline_policy="fixed_price")
    series = [
        charts.revenue_trend_series(episodes),
        charts.occupancy_trend_series(episodes),
        charts.spoilage_trend_series(episodes),
        charts.price_trend_series(summary),
        charts.reward_trend_series(episodes),
        charts.baseline_comparison_series(summary),
        charts.revenue_uplift_series(summary),
        charts.price_action_distribution(episodes, "dqn"),
    ]
    assert all(s is not None for s in series)
    return f"built {len(series)} dashboard chart series + KPI summary from real evaluation CSVs"


def check_flask_web_dashboard() -> str:
    from dashboard.web_dashboard.app import create_app

    app = create_app()
    app.config["TESTING"] = True
    client = app.test_client()

    page_routes = ["/", "/dashboard", "/recommendations", "/events", "/simulator"]
    for route in page_routes:
        resp = client.get(route)
        assert resp.status_code == 200, f"GET {route} returned {resp.status_code}"

    resp = client.get("/api/kpis")
    assert resp.status_code == 200, f"GET /api/kpis returned {resp.status_code}"

    resp = client.get("/api/charts/revenue_trend")
    assert resp.status_code == 200, f"GET /api/charts/revenue_trend returned {resp.status_code}"

    resp = client.get("/api/alerts?lookahead=30")
    assert resp.status_code == 200, f"GET /api/alerts returned {resp.status_code}"

    resp = client.post(
        "/api/simulate",
        json={"agent": "dqn", "initial_inventory": 20, "selling_horizon_days": 7, "market_condition": "standard"},
    )
    assert resp.status_code in (200, 404), f"POST /api/simulate returned {resp.status_code}"

    resp = client.get("/this-route-does-not-exist")
    assert resp.status_code == 404, "custom 404 handler did not fire"

    return f"Flask app served {len(page_routes)} page route(s) + core API routes via test client"


def check_demand_shock_detection() -> str:
    from additional_features.demand_shock_detection import DemandShockConfig, DemandShockDetector, Event
    from configs.evaluation_config import BusinessKPIConfig
    from pricing_env import PricingEnvConfig

    today = date.today()
    events = [
        Event(
            name="Smoke Test Event",
            event_date=today + timedelta(days=3),
            category="festival",
            expected_impact=1.2,
            confidence=0.7,
        )
    ]
    detector = DemandShockDetector(DemandShockConfig(lookahead_days=14))
    alerts = detector.scan(events, PricingEnvConfig(), BusinessKPIConfig().target_sell_through_pct)
    assert len(alerts) == 1
    assert alerts[0].risk_level in {"Low", "Medium", "High"}
    return "DemandShockDetector.scan() produced a valid alert from an in-memory event"


CHECKS: List[Tuple[str, Callable[[], str]]] = [
    ("imports", check_imports),
    ("explainer_live_environment", check_explainer_against_live_environment),
    ("dqn_checkpoint", check_dqn_checkpoint),
    ("q_learning_checkpoint", check_q_learning_checkpoint),
    ("dashboard_real_csvs", check_dashboard_against_real_csvs),
    ("flask_web_dashboard", check_flask_web_dashboard),
    ("demand_shock_detection", check_demand_shock_detection),
]


def main() -> int:
    print("=" * 78)
    print("additional_features deployment smoke test")
    print("=" * 78)

    results = [_run_check(name, fn) for name, fn in CHECKS]
    all_passed = True

    for name, passed, detail in results:
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {name}: {detail.splitlines()[0]}")
        if not passed:
            all_passed = False
            print(detail, file=sys.stderr)

    print("=" * 78)
    if all_passed:
        print("RESULT: all checks passed — safe to deploy.")
        return 0
    print("RESULT: one or more checks FAILED — do not deploy. See stderr above.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
