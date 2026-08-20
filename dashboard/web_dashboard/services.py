"""
services.py

Shared business logic for the Flask web dashboard's API blueprint
(`blueprints/api.py`). Kept in one place — imported by every route that
needs it — rather than duplicated per-route, per the project's
"no multiple duplicate files" requirement.

Every function here operates on the project's REAL objects:
`pricing_env.PricingEnvironment`, real trained `DQNAgent`/`QLearningAgent`
checkpoints (`agents/checkpoints/*`), and the real
`additional_features.*` explainability/shock-detection modules. Nothing
in this file fabricates data — where a real trained checkpoint or a real
event file is missing, functions raise a clear, caught-by-the-blueprint
error rather than silently returning invented numbers.
"""

from __future__ import annotations

import dataclasses
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from additional_features.demand_shock_detection import (
    CSVEventSource,
    DemandShockConfig,
    DemandShockDetector,
    Event,
    JSONEventSource,
)
from additional_features.explainable_ai_pricing import PricingExplainer
from agents.dqn_agent import DQNAgent
from agents.q_learning_agent import QLearningAgent
from configs.evaluation_config import BusinessKPIConfig, get_default_evaluation_config
from pricing_env import PricingEnvConfig, PricingEnvironment
from pricing_env.demand_simulator import DemandConfig

REPO_ROOT = Path(__file__).resolve().parents[2]

# Real, project-defined market-condition levers: each maps to a
# multiplier applied to the REAL `DemandConfig.base_daily_arrival_rate`
# field (pricing_env/demand_simulator.py). No invented mechanism (e.g. a
# "competitor price war") is introduced — every option here changes an
# actual, already-existing config parameter.
MARKET_CONDITIONS: Dict[str, float] = {
    "standard": 1.0,
    "high_demand": 1.30,
    "low_demand": 0.70,
}

_explainer = PricingExplainer()

# Process-local caches so a trained checkpoint is loaded from disk once,
# not on every request. Flask's dev server is single-process by default;
# for a multi-worker production deployment each worker gets its own
# cache, which is safe (checkpoints are read-only).
_dqn_agent_cache: Dict[str, DQNAgent] = {}
_q_learning_agent_cache: Dict[str, QLearningAgent] = {}


class AgentNotFoundError(RuntimeError):
    """Raised when a requested agent's checkpoint file does not exist on disk."""


def is_dqn_agent_loaded() -> bool:
    """Cheap, no-I/O check of whether the DQN agent is currently cached
    in memory (used by `/api/health`; never touches disk or the model)."""
    return bool(_dqn_agent_cache)


def is_q_learning_agent_loaded() -> bool:
    """Cheap, no-I/O check of whether the Q-Learning agent is currently
    cached in memory (used by `/api/health`)."""
    return bool(_q_learning_agent_cache)


def warm_agents() -> Dict[str, bool]:
    """
    Eagerly load (and cache) every agent whose checkpoint exists, once, at
    process startup — call this from `create_app()`.

    Without this, `get_dqn_agent` / `get_q_learning_agent` load lazily on
    whichever request happens to be first. That's fine for later requests
    (the cache makes them effectively free), but it means the FIRST real
    `/api/recommendation` click a person makes pays the full checkpoint
    load + (for DQN) PyTorch's one-time kernel warmup cost — the exact
    "clicking a button takes several seconds" symptom, just moved to
    whichever click happens to be first instead of eliminated. Doing this
    once at startup instead means every user-facing request is fast,
    including the first one.

    Missing checkpoints are not an error here — same as any other
    request, they're just skipped (the agent will still correctly raise
    `AgentNotFoundError` with a clear message if requested later, e.g.
    before the user has run training).

    Returns a dict like `{"dqn": True, "q_learning": False}` recording
    which agents were successfully warmed, for startup logging.
    """
    env = PricingEnvironment(PricingEnvConfig())
    warmed: Dict[str, bool] = {}
    for name, loader in (("dqn", get_dqn_agent), ("q_learning", get_q_learning_agent)):
        try:
            loader(env)
            warmed[name] = True
        except AgentNotFoundError:
            warmed[name] = False
    return warmed


def get_dqn_agent(env: PricingEnvironment) -> DQNAgent:
    """
    Load (and cache) the real trained DQN checkpoint at the path
    configured in `configs.evaluation_config.get_default_evaluation_config()`.

    Raises
    ------
    AgentNotFoundError
        If the checkpoint file does not exist — e.g. before the first
        `python training/train_dqn.py` run.
    """
    eval_config = get_default_evaluation_config()
    checkpoint_path = REPO_ROOT / eval_config.dqn_checkpoint_path
    cache_key = str(checkpoint_path)

    if cache_key in _dqn_agent_cache:
        return _dqn_agent_cache[cache_key]

    if not checkpoint_path.exists():
        raise AgentNotFoundError(
            f"No trained DQN checkpoint found at '{checkpoint_path}'. "
            f"Train one first (see training/train_dqn.py)."
        )

    agent = DQNAgent.load(
        checkpoint_path,
        observation_space=env.observation_space,
        action_space=env.action_space,
        hidden_layer_sizes=eval_config.dqn_hidden_layer_sizes,
        device="cpu",
    )
    _dqn_agent_cache[cache_key] = agent
    return agent


def get_q_learning_agent(env: PricingEnvironment) -> QLearningAgent:
    """
    Load (and cache) the real trained Q-Learning checkpoint at the path
    configured in `configs.evaluation_config.get_default_evaluation_config()`.

    Raises
    ------
    AgentNotFoundError
        If the checkpoint file does not exist.
    """
    eval_config = get_default_evaluation_config()
    checkpoint_path = REPO_ROOT / eval_config.q_learning_checkpoint_path
    cache_key = str(checkpoint_path)

    if cache_key in _q_learning_agent_cache:
        return _q_learning_agent_cache[cache_key]

    if not checkpoint_path.exists():
        raise AgentNotFoundError(
            f"No trained Q-Learning checkpoint found at '{checkpoint_path}'. "
            f"Train one first (see training/train_agent.py)."
        )

    agent = QLearningAgent.load(checkpoint_path, action_space=env.action_space)
    _q_learning_agent_cache[cache_key] = agent
    return agent


def _q_values_for(agent, observation) -> List[float]:
    """Per-action values for either agent type, as a plain float list."""
    if isinstance(agent, DQNAgent):
        with torch.no_grad():
            values = agent.q_network(torch.tensor(observation, dtype=torch.float32))
        return values.detach().cpu().numpy().tolist()
    # QLearningAgent: the Q-table row for this (rounded) state.
    state_key = tuple(round(float(x)) for x in observation)
    return agent.q_table[state_key].tolist()


def _select_action(agent, observation) -> int:
    return int(agent.select_greedy_action(observation))


def build_live_recommendation(agent_name: str = "dqn", seed: int = 42) -> Dict[str, Any]:
    """
    Roll one real environment step forward with the requested trained
    agent and return a full `PricingExplanation` (see
    `additional_features.explainable_ai_pricing.explainer.PricingExplanation`)
    as a JSON-serializable dict.

    Parameters
    ----------
    agent_name : {"dqn", "q_learning"}
    seed : int
        Environment reset seed, for a reproducible "current" state.
    """
    env = PricingEnvironment(PricingEnvConfig())
    obs, info = env.reset(seed=seed)
    previous_price = info["current_price"]

    agent = get_dqn_agent(env) if agent_name == "dqn" else get_q_learning_agent(env)
    action = _select_action(agent, obs)
    q_values = _q_values_for(agent, obs)

    next_obs, _reward, _terminated, _truncated, next_info = env.step(action)

    explanation = _explainer.explain(
        observation=next_obs,
        info=next_info,
        previous_price=previous_price,
        action_values=q_values,
        selected_action=action,
        q_network=agent.q_network if isinstance(agent, DQNAgent) else None,
    )
    result = explanation.to_dict()
    result["agent"] = agent_name
    return result


def build_recommendation_timeline(agent_name: str = "dqn", seed: int = 42, num_steps: int = 6) -> List[Dict[str, Any]]:
    """
    Roll `num_steps` real environment steps forward with the requested
    trained agent, returning one `PricingExplanation` dict per step, in
    order — the real, non-fabricated data behind the dashboard's
    "Pricing Timeline & Projections" panel.

    Each step's `previous_price` is the ACTUAL price from the previous
    real step (not a guess), so the sequence of `recommendation_summary`
    strings is a genuine step-by-step account of one simulated episode.
    """
    env = PricingEnvironment(PricingEnvConfig())
    obs, info = env.reset(seed=seed)
    agent = get_dqn_agent(env) if agent_name == "dqn" else get_q_learning_agent(env)

    timeline: List[Dict[str, Any]] = []
    previous_price = info["current_price"]

    for step_index in range(num_steps):
        action = _select_action(agent, obs)
        q_values = _q_values_for(agent, obs)
        next_obs, _reward, terminated, truncated, next_info = env.step(action)

        explanation = _explainer.explain(
            observation=next_obs,
            info=next_info,
            previous_price=previous_price,
            action_values=q_values,
            selected_action=action,
            q_network=agent.q_network if isinstance(agent, DQNAgent) else None,
        )
        entry = explanation.to_dict()
        entry["step"] = step_index + 1
        entry["days_remaining"] = float(next_obs[1])
        timeline.append(entry)

        previous_price = next_info["price"]
        obs, info = next_obs, next_info
        if terminated or truncated:
            break

    return timeline


def run_episode_simulation(
    agent_name: str = "dqn",
    initial_inventory: int = 100,
    selling_horizon_days: int = 30,
    market_condition: str = "standard",
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Run one FULL real episode of `PricingEnvironment` with the requested
    trained agent, under the requested (real, config-driven) market
    condition, and return the actual day-by-day trajectory. This is the
    real backend for the "What-If Scenario Simulator" page — every value
    returned comes from an actual environment rollout, never
    `Math.random()` or another fabricated placeholder.

    Parameters
    ----------
    agent_name : {"dqn", "q_learning"}
    initial_inventory : int
        Overrides `PricingEnvConfig.initial_inventory`.
    selling_horizon_days : int
        Overrides `PricingEnvConfig.selling_horizon_days`.
    market_condition : {"standard", "high_demand", "low_demand"}
        Scales the real `DemandConfig.base_daily_arrival_rate` field by
        `MARKET_CONDITIONS[market_condition]` (see module docstring).
    seed : int
        Environment reset seed, for a reproducible run.

    Raises
    ------
    ValueError
        If `market_condition` is not a recognized key.
    AgentNotFoundError
        If the requested agent's checkpoint is missing.
    """
    if market_condition not in MARKET_CONDITIONS:
        raise ValueError(
            f"Unknown market_condition '{market_condition}'. "
            f"Valid options: {sorted(MARKET_CONDITIONS)}"
        )

    base_demand = DemandConfig()
    scaled_demand = dataclasses.replace(
        base_demand,
        base_daily_arrival_rate=base_demand.base_daily_arrival_rate * MARKET_CONDITIONS[market_condition],
    )
    env_config = PricingEnvConfig(
        initial_inventory=initial_inventory,
        selling_horizon_days=selling_horizon_days,
        demand=scaled_demand,
    )
    env = PricingEnvironment(env_config)
    agent = get_dqn_agent(env) if agent_name == "dqn" else get_q_learning_agent(env)

    obs, info = env.reset(seed=seed)
    cumulative_revenue = 0.0
    days: List[Dict[str, Any]] = []
    day_index = 0
    terminated = truncated = False

    while not (terminated or truncated):
        day_index += 1
        action = _select_action(agent, obs)
        obs, reward, terminated, truncated, info = env.step(action)
        cumulative_revenue = float(info["episode_revenue"])
        sell_through_pct = round(
            (1.0 - float(obs[0]) / float(info["initial_inventory"])) * 100.0, 2
        )
        days.append(
            {
                "day": day_index,
                "price": round(float(info["price"]), 2),
                "units_sold": int(info["units_sold"]),
                "cumulative_revenue": round(cumulative_revenue, 2),
                "occupancy_pct": sell_through_pct,
                "remaining_inventory": int(obs[0]),
            }
        )

    return {
        "agent": agent_name,
        "market_condition": market_condition,
        "initial_inventory": initial_inventory,
        "selling_horizon_days": selling_horizon_days,
        "final_revenue": round(cumulative_revenue, 2),
        "final_occupancy_pct": days[-1]["occupancy_pct"] if days else 0.0,
        "days": days,
    }


def load_demand_shock_alerts(
    event_source_path: Optional[str] = None,
    lookahead_days: int = 30,
    reference_date_str: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Load real events (CSV or JSON) and scan them for demand shocks with
    `additional_features.demand_shock_detection.DemandShockDetector`.

    If `event_source_path` is not given, falls back to the bundled,
    explicitly-labeled example calendar at
    `additional_features/demand_shock_detection/sample_events.csv` — the
    response's `"source"` field always says exactly which file was used,
    so the dashboard can clearly flag when it's showing sample data
    versus the caller's own real event calendar (never a silent
    fabrication).
    """
    if event_source_path:
        source_path = Path(event_source_path)
        if not source_path.is_absolute():
            source_path = REPO_ROOT / source_path
        is_sample = False
    else:
        source_path = (
            REPO_ROOT
            / "additional_features"
            / "demand_shock_detection"
            / "sample_events.csv"
        )
        is_sample = True

    if not source_path.exists():
        return {
            "source": str(source_path),
            "is_sample_data": is_sample,
            "alerts": [],
            "error": f"Event source file not found: {source_path}",
        }

    events: List[Event]
    if source_path.suffix.lower() == ".json":
        events = JSONEventSource(source_path).load()
    else:
        events = CSVEventSource(source_path).load()

    detector = DemandShockDetector(DemandShockConfig(lookahead_days=lookahead_days))
    env_config = PricingEnvConfig()
    kpi_config = BusinessKPIConfig()
    reference = (
        date.fromisoformat(reference_date_str) if reference_date_str else date.today()
    )

    alerts = detector.scan(
        events, env_config, kpi_config.target_sell_through_pct, reference_date=reference
    )

    return {
        "source": str(source_path.relative_to(REPO_ROOT)) if source_path.is_relative_to(REPO_ROOT) else str(source_path),
        "is_sample_data": is_sample,
        "reference_date": reference.isoformat(),
        "alerts": [
            {
                "event": a.event.name,
                "category": a.event.category,
                "date": a.event.event_date.isoformat(),
                "days_until": a.days_until_event,
                "risk": a.risk_level,
                "expected_impact_multiplier": a.event.expected_impact,
                "expected_demand_lift_pct": round((a.event.expected_impact - 1.0) * 100.0, 1),
                "suggested_occupancy_target_pct": a.suggested_occupancy_target_pct,
                "suggested_exploration_adjustment": a.suggested_exploration_adjustment,
                "suggested_pricing_ceiling": a.suggested_pricing_ceiling,
                "confidence": a.confidence,
                "explanation": a.explanation,
            }
            for a in alerts
        ],
    }
