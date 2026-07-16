"""
state.py

Owns the MDP state representation for the pricing environment: the mutable
per-episode state object, and the (single, authoritative) definition of the
Gymnasium observation space built from that state.

Design intent
-------------
Every other module (reward, demand_simulator, pricing_env) reads or writes
state through this module's `EnvState` class rather than passing around loose
tuples of numbers. This keeps "what is the state" defined in exactly one
place — if the Phase 2 team extends the observation to include
`current_price` or `seasonality`, this is the only file that needs to change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
from gymnasium import spaces


@dataclass
class EnvState:
    """
    Mutable per-episode state for the pricing environment.

    This is intentionally a *mutable* dataclass (unlike PricingEnvConfig,
    which is frozen) because the environment updates it in place every
    `step()` call. Keeping state mutation confined to this one object
    (rather than scattering loose instance attributes across
    PricingEnvironment) makes it trivial to snapshot, log, or reset the
    entire episode state in one line.

    Attributes
    ----------
    remaining_inventory : int
        Unsold units left this episode.
    days_remaining : int
        Days left until the deadline (departure / check-in).
    current_price : float
        The price currently in effect (result of the most recent action).
    current_step : int
        Number of steps (days) elapsed in the current episode.
    episode_revenue : float
        Cumulative revenue earned so far this episode.
    last_units_sold : int
        Units sold on the most recent step. Tracked for rendering/info,
        not part of the formal observation.
    last_demand_level : str
        Qualitative demand label ("Low"/"Medium"/"High") from the most
        recent step, surfaced via `info` for interpretability — revenue
        managers reviewing agent behaviour want a human-readable signal,
        not just raw counts.
    """

    remaining_inventory: int
    days_remaining: int
    current_price: float
    current_step: int = 0
    episode_revenue: float = 0.0
    last_units_sold: int = 0
    last_demand_level: str = "Unknown"

    def to_observation(self) -> np.ndarray:
        """
        Return the Phase-1 core observation vector.

        Fixed as [remaining_inventory, days_remaining] per the MDP design
        document. Deliberately does NOT include current_price or demand
        signals yet — those are exposed via `to_info()` instead, so they
        can be folded into the observation in a later phase without
        breaking the action-index / policy contracts of agents already
        trained against this 2-dimensional space.
        """
        return np.array(
            [float(self.remaining_inventory), float(self.days_remaining)],
            dtype=np.float32,
        )

    def to_info(self, config: "Any") -> Dict[str, Any]:
        """
        Return the auxiliary info dict for this state.

        Parameters
        ----------
        config : PricingEnvConfig
            Passed in (rather than stored on EnvState) to avoid a circular
            import between state.py and pricing_env.py, and because static
            config values (initial_inventory, horizon) don't belong on a
            per-episode mutable state object.
        """
        return {
            "current_price": self.current_price,
            "current_step": self.current_step,
            "episode_revenue": self.episode_revenue,
            "last_units_sold": self.last_units_sold,
            "last_demand_level": self.last_demand_level,
            "initial_inventory": config.initial_inventory,
            "selling_horizon_days": config.selling_horizon_days,
        }


def build_observation_space(config: "Any") -> spaces.Box:
    """
    Build the Gymnasium observation space matching `EnvState.to_observation()`.

    Kept as a standalone function (not a method) so `pricing_env.py` can
    call it during `__init__`, before any `EnvState` instance exists (state
    is only created in `reset()`).

    Uses float32, matching the dtype every standard DQN implementation
    (Stable-Baselines3, CleanRL) expects natively — using int32 would force
    every neural-network agent to cast on every step.
    """
    low = np.array([0.0, 0.0], dtype=np.float32)
    high = np.array(
        [float(config.initial_inventory), float(config.selling_horizon_days)],
        dtype=np.float32,
    )
    return spaces.Box(low=low, high=high, dtype=np.float32)