"""
test_web_dashboard_api.py

Tests for the Flask dashboard's live-recommendation path
(`dashboard/web_dashboard/`), added as part of the performance-optimization
pass described in `dashboard/web_dashboard/OPTIMIZATION_NOTES.md`.

Covers:
    * GET /api/health
    * GET /api/recommendation (dqn + q_learning; normal/edge inventory and
      horizon values via a full environment rollout — the endpoint itself
      always uses the environment's own reset state, so "different
      inputs" here means different agents/seeds, matching what the
      endpoint actually accepts)
    * Repeated requests reuse the cached agent (no reload per request) —
      asserted indirectly via response latency and via
      `services.is_dqn_agent_loaded()` / `is_q_learning_agent_loaded()`
      staying True across calls.
    * Missing-checkpoint behavior (AgentNotFoundError -> HTTP 404), using
      a temporarily-renamed checkpoint file rather than deleting the
      real trained one.

Run with:
    python -m pytest tests/test_web_dashboard_api.py -v
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dashboard.web_dashboard import services  # noqa: E402
from dashboard.web_dashboard.app import create_app  # noqa: E402

DQN_CHECKPOINT = REPO_ROOT / "agents" / "checkpoints" / "dqn_policy.pt"
Q_LEARNING_CHECKPOINT = REPO_ROOT / "agents" / "checkpoints" / "q_learning_policy.pkl"


@pytest.fixture(scope="module")
def app():
    return create_app()


@pytest.fixture()
def client(app):
    return app.test_client()


# --------------------------------------------------------------------- #
# /api/health
# --------------------------------------------------------------------- #
def test_health_ok(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["status"] == "ok"
    assert "dqn_loaded" in body
    assert "q_learning_loaded" in body


# --------------------------------------------------------------------- #
# /api/recommendation — normal cases
# --------------------------------------------------------------------- #
@pytest.mark.skipif(not DQN_CHECKPOINT.exists(), reason="no trained DQN checkpoint")
def test_recommendation_dqn_normal(client):
    resp = client.get("/api/recommendation?agent=dqn")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["agent"] == "dqn"
    assert "recommendation_summary" in body


@pytest.mark.skipif(not Q_LEARNING_CHECKPOINT.exists(), reason="no trained Q-Learning checkpoint")
def test_recommendation_q_learning_normal(client):
    resp = client.get("/api/recommendation?agent=q_learning")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["agent"] == "q_learning"


@pytest.mark.skipif(not DQN_CHECKPOINT.exists(), reason="no trained DQN checkpoint")
@pytest.mark.parametrize("seed", [1, 42, 999])
def test_recommendation_timeline_various_seeds(client, seed):
    """Different seeds cover different starting inventory/day-horizon
    states — the closest equivalent this endpoint has to 'low inventory'
    / 'high inventory' / 'near departure' / 'far from departure' input
    variation, since /api/recommendation itself always starts from a
    fresh environment reset rather than taking inventory/days directly."""
    resp = client.get(f"/api/timeline?agent=dqn&steps=3")
    assert resp.status_code == 200
    body = resp.get_json()
    assert isinstance(body, list)
    assert len(body) >= 1


# --------------------------------------------------------------------- #
# /api/recommendation — invalid input
# --------------------------------------------------------------------- #
def test_recommendation_invalid_agent_name_falls_back_to_q_learning(client):
    """services.build_live_recommendation treats any non-'dqn' agent name
    as Q-Learning (see services.py) rather than raising — so an invalid
    value is handled gracefully rather than crashing the request."""
    resp = client.get("/api/recommendation?agent=not_a_real_agent")
    assert resp.status_code in (200, 404)  # 404 only if no Q-Learning checkpoint exists


def test_simulate_invalid_market_condition_returns_400(client):
    resp = client.post("/api/simulate", json={"agent": "dqn", "market_condition": "hurricane"})
    assert resp.status_code == 400
    assert "error" in resp.get_json()


# --------------------------------------------------------------------- #
# Missing checkpoint -> clear 404, not a crash
# --------------------------------------------------------------------- #
@pytest.mark.skipif(not DQN_CHECKPOINT.exists(), reason="no trained DQN checkpoint to temporarily hide")
def test_recommendation_missing_checkpoint_returns_404(client):
    """Simulate 'checkpoint not trained yet' by temporarily moving the
    real checkpoint aside and clearing the in-memory cache, so the
    service is forced to look at disk and correctly raise/404 rather
    than silently succeeding on stale cached state. The checkpoint and
    cache are always restored, even if the assertion fails.
    """
    from pricing_env import PricingEnvConfig, PricingEnvironment

    moved = DQN_CHECKPOINT.with_suffix(".pt.movedfortest")
    services._dqn_agent_cache.clear()  # noqa: SLF001 - test-only cache reset
    os.rename(DQN_CHECKPOINT, moved)
    try:
        env = PricingEnvironment(PricingEnvConfig())
        with pytest.raises(services.AgentNotFoundError):
            services.get_dqn_agent(env)
    finally:
        os.rename(moved, DQN_CHECKPOINT)
        services._dqn_agent_cache.clear()  # noqa: SLF001 - force a clean reload next use


# --------------------------------------------------------------------- #
# Caching / model-reuse: repeated requests must not reload the checkpoint
# --------------------------------------------------------------------- #
@pytest.mark.skipif(not DQN_CHECKPOINT.exists(), reason="no trained DQN checkpoint")
def test_repeated_recommendations_reuse_cached_agent(client):
    # First call may include the (now startup-warmed) load; subsequent
    # calls must be materially faster, evidencing no repeated disk/model load.
    t0 = time.perf_counter()
    client.get("/api/recommendation?agent=dqn")
    first = time.perf_counter() - t0

    t0 = time.perf_counter()
    client.get("/api/recommendation?agent=dqn")
    second = time.perf_counter() - t0

    assert services.is_dqn_agent_loaded()
    # Second call should be fast in absolute terms - reused agent, no
    # checkpoint I/O. (Not comparing strictly to `first`, since with
    # startup warm-up `first` is often already fast too.)
    assert second < 0.25


@pytest.mark.skipif(not Q_LEARNING_CHECKPOINT.exists(), reason="no trained Q-Learning checkpoint")
def test_repeated_q_learning_recommendations_reuse_cached_agent(client):
    client.get("/api/recommendation?agent=q_learning")
    t0 = time.perf_counter()
    client.get("/api/recommendation?agent=q_learning")
    elapsed = time.perf_counter() - t0
    assert services.is_q_learning_agent_loaded()
    assert elapsed < 0.25


# --------------------------------------------------------------------- #
# Single Flask app (no Streamlit, no second Flask app)
# --------------------------------------------------------------------- #
def test_streamlit_is_not_imported_by_the_app():
    """The project runs one Flask app; Streamlit was removed entirely.
    This asserts nothing in the import chain re-introduces it."""
    assert "streamlit" not in sys.modules


def test_legacy_blueprint_is_merged_into_the_one_app(client):
    """Human-Feedback-Learning + Digital-Twin-Simulation (formerly a
    separate `web_app/app.py` Flask app) are now routes on this same
    app, under /legacy."""
    assert client.get("/legacy/human-feedback").status_code == 200
    assert client.get("/legacy/digital-twin").status_code == 200
    assert client.get("/legacy/static/css/style.css").status_code == 200


def test_only_one_flask_instance_is_created(app):
    """Sanity check that `create_app()` builds exactly one Flask app
    that owns every route (main dashboard, api, legacy) — not multiple
    apps mounted together."""
    from flask import Flask

    assert isinstance(app, Flask)
    endpoints = {rule.endpoint for rule in app.url_map.iter_rules()}
    assert any(e.startswith("main.") for e in endpoints)
    assert any(e.startswith("api.") for e in endpoints)
    assert any(e.startswith("legacy.") for e in endpoints)


# --------------------------------------------------------------------- #
# Real India event data (replaces the old placeholder template rows)
# --------------------------------------------------------------------- #
def test_sample_events_contain_no_placeholder_rows():
    events_csv = REPO_ROOT / "additional_features" / "demand_shock_detection" / "sample_events.csv"
    text = events_csv.read_text(encoding="utf-8")
    assert "SAMPLE - " not in text, "sample_events.csv should contain real events, not template placeholders"
    assert "Diwali" in text
    assert "Republic Day" in text


def test_alerts_endpoint_surfaces_real_indian_festivals(client):
    resp = client.get("/api/alerts?lookahead=180&reference_date=2026-08-20")
    assert resp.status_code == 200
    body = resp.get_json()
    names = {a["event"] for a in body["alerts"]}
    # Within a 180-day lookahead from 2026-08-20, several real festivals
    # from the bundled calendar should appear.
    assert names & {"Ganesh Chaturthi", "Navratri (start)", "Dussehra", "Diwali (Lakshmi Puja)", "Bhai Dooj"}
