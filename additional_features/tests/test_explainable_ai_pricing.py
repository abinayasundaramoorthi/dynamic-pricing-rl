"""
test_explainable_ai_pricing.py

Validation tests for Feature 1 — Explainable AI Pricing Insights.

Covers:
  1. `extract_pricing_factors` derives correct, non-fabricated factors
     from a real `PricingEnvironment` step (occupancy, pacing, demand,
     urgency, reward composition, price-change).
  2. `extract_pricing_factors` fails loudly (KeyError) on incomplete
     `info`, rather than silently fabricating a default.
  3. `confidence_from_action_values` returns a valid, bounded score for
     both a decisive and a near-tied action-value vector.
  4. `dqn_state_sensitivity` runs against the project's REAL trained DQN
     checkpoint (`agents/checkpoints/dqn_policy.pt`) if present, and is
     skipped (not failed) if that checkpoint hasn't been produced yet —
     matching this project's existing pattern for optional-checkpoint
     tests (see `evaluation/evaluate_policies.py::build_policies`).
  5. `PricingExplainer.explain()` produces a consistent, fully-populated
     `PricingExplanation` end-to-end for both a DQN agent and a
     Q-Learning agent, using their REAL trained checkpoints.

Run with:
    python -m pytest additional_features/tests/test_explainable_ai_pricing.py -v
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from additional_features.explainable_ai_pricing import (
    PricingExplainer,
    confidence_from_action_values,
    dqn_state_sensitivity,
    extract_pricing_factors,
)
from agents.dqn_agent import DQNAgent
from agents.q_learning_agent import QLearningAgent
from configs.evaluation_config import get_default_evaluation_config
from pricing_env import PricingEnvironment, PricingEnvConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_CONFIG = get_default_evaluation_config()
DQN_CHECKPOINT = REPO_ROOT / EVAL_CONFIG.dqn_checkpoint_path
Q_LEARNING_CHECKPOINT = REPO_ROOT / EVAL_CONFIG.q_learning_checkpoint_path


@pytest.fixture
def env():
    return PricingEnvironment(PricingEnvConfig())


@pytest.fixture
def one_real_step(env):
    """(observation, info, previous_price, next_observation, next_info) from one real env step."""
    obs, info = env.reset(seed=7)
    previous_price = info["current_price"]
    next_obs, reward, terminated, truncated, next_info = env.step(3)
    return obs, info, previous_price, next_obs, next_info


# --------------------------------------------------------------------------- #
# 1-2. factors.py
# --------------------------------------------------------------------------- #
def test_extract_pricing_factors_returns_ranked_normalized_factors(one_real_step):
    _, _, previous_price, next_obs, next_info = one_real_step
    factors = extract_pricing_factors(next_obs, next_info, previous_price=previous_price)

    assert len(factors) >= 4, "expected at least occupancy/pacing/demand/urgency factors"

    weights = [f.weight for f in factors]
    assert weights == sorted(weights, reverse=True), "factors must be ranked by descending weight"
    assert all(w >= 0 for w in weights), "weights must be non-negative"
    assert abs(sum(weights) - 1.0) < 1e-6, "weights must be normalized to sum to 1.0"

    keys = {f.key for f in factors}
    assert "occupancy_pressure" in keys
    assert "booking_pace" in keys
    assert "demand_intensity" in keys
    assert "time_urgency" in keys

    for factor in factors:
        assert factor.direction in {"raises_price", "lowers_price", "neutral"}
        assert isinstance(factor.narrative, str) and len(factor.narrative) > 0


def test_extract_pricing_factors_never_references_unavailable_signals(one_real_step):
    """
    Explicit regression guard for the project's core explainability
    requirement: this repo has no competitor-pricing, special-event, or
    customer-behaviour data, so no factor key/narrative may claim to.
    """
    _, _, previous_price, next_obs, next_info = one_real_step
    factors = extract_pricing_factors(next_obs, next_info, previous_price=previous_price)

    forbidden_terms = ["competitor", "special event", "customer segment"]
    for factor in factors:
        lowered = factor.narrative.lower()
        for term in forbidden_terms:
            assert term not in lowered, f"factor narrative fabricated unavailable signal: {term!r}"


def test_extract_pricing_factors_raises_on_incomplete_info(one_real_step):
    _, _, previous_price, next_obs, next_info = one_real_step
    incomplete_info = dict(next_info)
    del incomplete_info["reward_breakdown"]

    with pytest.raises(KeyError):
        extract_pricing_factors(next_obs, incomplete_info, previous_price=previous_price)


# --------------------------------------------------------------------------- #
# 3. confidence.py — confidence_from_action_values
# --------------------------------------------------------------------------- #
def test_confidence_from_action_values_decisive_case():
    action_values = [1.0, 5.0, 1.2, 0.9]  # action 1 clearly dominant
    confidence = confidence_from_action_values(action_values, selected_action=1)
    assert 0.0 <= confidence.score <= 1.0
    assert confidence.label in {"Low", "Medium", "High", "Very High"}
    assert confidence.score > 0.5, "a clearly dominant action should score above 0.5"


def test_confidence_from_action_values_near_tied_case():
    action_values = [3.01, 3.0, 3.02, 2.99]  # near-uniform
    confidence = confidence_from_action_values(action_values, selected_action=2)
    assert confidence.label in {"Low", "Medium"}, "near-tied actions should not report high confidence"


def test_confidence_from_action_values_rejects_out_of_range_action():
    with pytest.raises(ValueError):
        confidence_from_action_values([1.0, 2.0, 3.0], selected_action=5)


# --------------------------------------------------------------------------- #
# 4. confidence.py — dqn_state_sensitivity (real trained checkpoint)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(
    not DQN_CHECKPOINT.exists(),
    reason=f"No DQN checkpoint at {DQN_CHECKPOINT}; run `python -m training.train_dqn` first.",
)
def test_dqn_state_sensitivity_against_real_checkpoint(env):
    agent = DQNAgent.load(
        DQN_CHECKPOINT,
        observation_space=env.observation_space,
        action_space=env.action_space,
        hidden_layer_sizes=EVAL_CONFIG.dqn_hidden_layer_sizes,
        device="cpu",
        seed=42,
    )
    obs, _ = env.reset(seed=11)
    action = agent.select_greedy_action(obs)

    sensitivities = dqn_state_sensitivity(agent.q_network, obs, action)

    assert set(sensitivities.keys()) == {"remaining_inventory", "days_remaining"}
    assert all(v >= 0.0 for v in sensitivities.values()), "sensitivities must be non-negative magnitudes"
    assert all(np.isfinite(v) for v in sensitivities.values())


# --------------------------------------------------------------------------- #
# 5. explainer.py — end-to-end against real trained agents
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(
    not DQN_CHECKPOINT.exists(),
    reason=f"No DQN checkpoint at {DQN_CHECKPOINT}; run `python -m training.train_dqn` first.",
)
def test_pricing_explainer_end_to_end_with_real_dqn_agent(env):
    agent = DQNAgent.load(
        DQN_CHECKPOINT,
        observation_space=env.observation_space,
        action_space=env.action_space,
        hidden_layer_sizes=EVAL_CONFIG.dqn_hidden_layer_sizes,
        device="cpu",
        seed=42,
    )
    obs, info = env.reset(seed=99)
    previous_price = info["current_price"]
    action = agent.select_greedy_action(obs)
    next_obs, reward, terminated, truncated, next_info = env.step(action)

    import torch

    q_values = agent.q_network(torch.tensor(obs, dtype=torch.float32)).detach().numpy()

    explainer = PricingExplainer()
    explanation = explainer.explain(
        observation=next_obs,
        info=next_info,
        previous_price=previous_price,
        action_values=q_values,
        selected_action=action,
        q_network=agent.q_network,
    )

    assert explanation.recommendation_summary.startswith("Price")
    assert "Reasons:" in explanation.human_readable
    assert explanation.confidence is not None
    assert explanation.sensitivity is not None
    assert len(explanation.factors) > 0

    # to_json() must be valid, parseable JSON
    parsed = json.loads(explanation.to_json())
    assert parsed["recommendation_summary"] == explanation.recommendation_summary
    assert "feature_importance" in parsed
    assert "developer" in parsed


@pytest.mark.skipif(
    not Q_LEARNING_CHECKPOINT.exists(),
    reason=(
        f"No Q-Learning checkpoint at {Q_LEARNING_CHECKPOINT}; run "
        "`python -m training.train_agent --agent q_learning` first."
    ),
)
def test_pricing_explainer_end_to_end_with_real_q_learning_agent(env):
    agent = QLearningAgent.load(Q_LEARNING_CHECKPOINT, action_space=env.action_space, seed=42)
    obs, info = env.reset(seed=99)
    previous_price = info["current_price"]
    action = agent.select_greedy_action(obs)
    next_obs, reward, terminated, truncated, next_info = env.step(action)

    state_key = tuple(round(float(x)) for x in obs)
    q_values = agent.q_table[state_key]

    explainer = PricingExplainer()
    explanation = explainer.explain(
        observation=next_obs,
        info=next_info,
        previous_price=previous_price,
        action_values=q_values,
        selected_action=action,
    )

    assert explanation.confidence is not None
    assert explanation.sensitivity is None, "sensitivity requires a DQN q_network, not a Q-table"
    developer_view = explanation.developer_view()
    assert developer_view["confidence"]["score"] == pytest.approx(explanation.confidence.score)


def test_pricing_explainer_without_action_values_omits_confidence(one_real_step):
    """Confidence/sensitivity must be None (not fabricated) when not supplied."""
    _, _, previous_price, next_obs, next_info = one_real_step
    explanation = PricingExplainer().explain(
        observation=next_obs, info=next_info, previous_price=previous_price
    )
    assert explanation.confidence is None
    assert explanation.sensitivity is None
    assert "Confidence:" not in explanation.human_readable
