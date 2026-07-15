"""
pricing_env.py

Custom Reinforcement Learning environment for the Dynamic Pricing project.

This environment simulates selling a fixed inventory of items (e.g. tickets)
over a limited selling season. At each step, the agent sets a price, and the
environment simulates how many units get sold, updates inventory, and returns
the new state, reward, and whether the episode is done.

State  = [inventory_left, days_left]
Action = index into a list of possible price levels
Reward = revenue earned this step (see reward.py)

The actual state transition math lives in transition.py — this class is a
thin wrapper providing the standard reset() / step() / render() interface.
"""

from transition import apply_transition


class PricingEnv:
    """A simple custom environment for dynamic pricing RL."""

    def __init__(self, initial_inventory=100, total_days=30,
                 price_levels=None, base_demand=10):
        """
        Parameters
        ----------
        initial_inventory : int
            Number of units available at the start of each episode.
        total_days : int
            Number of days in one selling season (one episode).
        price_levels : list of float
            The discrete list of prices the agent can choose from.
            Example: [200, 300, 400, 500, 600]
        base_demand : float
            Baseline daily demand used in the demand simulation formula.
        """
        self.initial_inventory = initial_inventory
        self.total_days = total_days
        self.price_levels = price_levels if price_levels is not None else [200, 300, 400, 500, 600]
        self.base_demand = base_demand

        self.inventory = None
        self.days_left = None

    def reset(self):
        """Start a new episode (new selling season)."""
        self.inventory = self.initial_inventory
        self.days_left = self.total_days
        return self._get_state()

    def step(self, action):
        """
        Apply one pricing decision and move the environment forward by one day.
Custom Gymnasium environment for the Travel & Hospitality Dynamic Pricing
Reinforcement Learning project.

Day 4 deliverable: integrates the independently-developed environment
modules — state.py, action_space.py, reward.py, demand_simulator.py — into
a single, fully-functional `step()` implementation. Day 3 shipped the
skeleton (init/reset/render with a placeholder step); this file completes
the MDP loop end to end: an agent can now call reset() -> step() -> step()
-> ... through a full episode and receive real observations, rewards,
and termination signals.

Author: ML Engineering Team
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .action_space import action_to_price, build_action_space, describe_action
from .demand_simulator import DemandConfig, DemandSimulator, SaleOutcome
from .reward import RewardBreakdown, RewardConfig, compute_reward
from .state import EnvState, build_observation_space

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PricingEnvConfig:
    """
    Immutable, top-level configuration for PricingEnvironment.

    Composes the sub-configs owned by each integrated module
    (DemandConfig, RewardConfig) alongside the core environment
    parameters, so a single object fully specifies one experimental
    condition — this is what makes an experiment reproducible from a
    config file/dict alone (e.g. `PricingEnvConfig(**yaml_dict)`).

    Attributes
    ----------
    initial_inventory : int
        Total sellable units at the start of each episode. Must be > 0.
    selling_horizon_days : int
        Days in the selling season before the deadline. Must be > 0.
    base_price : float
        Reference price that price actions adjust relative to, and the
        anchor for the demand model's acceptance curve and the reward
        function's discount/unsold penalties. Must be > 0.
    price_adjustment_pct : List[float]
        Ordered fractional price adjustments defining the discrete action
        space. Order must stay stable across training/evaluation runs —
        changing it invalidates any already-trained policy.
    min_price, max_price : Optional[float]
        Hard price bounds. Default to 1% / 300% of base_price if omitted.
    render_mode : Optional[str]
        One of {None, "human", "ansi"}.
    demand : DemandConfig
        Configuration for the stochastic demand model (demand_simulator.py).
    reward : RewardConfig
        Configuration for the reward function (reward.py).
    """

    initial_inventory: int = 100
    selling_horizon_days: int = 30
    base_price: float = 200.0
    price_adjustment_pct: List[float] = field(
        default_factory=lambda: [-0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20]
    )
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    render_mode: Optional[str] = None
    demand: DemandConfig = field(default_factory=DemandConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)

    def __post_init__(self) -> None:
        if self.initial_inventory <= 0:
            raise ValueError(
                f"initial_inventory must be a positive integer, got {self.initial_inventory}"
            )
        if self.selling_horizon_days <= 0:
            raise ValueError(
                f"selling_horizon_days must be a positive integer, got {self.selling_horizon_days}"
            )
        if self.base_price <= 0:
            raise ValueError(f"base_price must be > 0, got {self.base_price}")
        if len(self.price_adjustment_pct) == 0:
            raise ValueError("price_adjustment_pct must contain at least one action")
        if self.render_mode not in (None, "human", "ansi"):
            raise ValueError(
                f"render_mode must be one of (None, 'human', 'ansi'), got {self.render_mode!r}"
            )
        if not isinstance(self.demand, DemandConfig):
            raise TypeError(
                f"demand must be a DemandConfig instance, got {type(self.demand)}"
            )
        if not isinstance(self.reward, RewardConfig):
            raise TypeError(
                f"reward must be a RewardConfig instance, got {type(self.reward)}"
            )

        if self.min_price is None:
            object.__setattr__(self, "min_price", round(0.01 * self.base_price, 2))
        if self.max_price is None:
            object.__setattr__(self, "max_price", round(3.0 * self.base_price, 2))
        if self.min_price >= self.max_price:
            raise ValueError(
                f"min_price ({self.min_price}) must be < max_price ({self.max_price})"
            )


# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #
class PricingEnvironment(gym.Env):
    """
    Gymnasium-compatible environment simulating dynamic pricing of finite,
    perishable inventory (e.g. airline seats, hotel rooms) over a fixed
    selling horizon.

    One environment "step" = one day. Each day, the agent chooses a price
    action; a stochastic number of customers arrive and independently
    decide to purchase based on that price (demand_simulator.py); the
    resulting sale updates inventory and revenue; reward is computed
    (reward.py) from the sale, the discount depth, terminal spoilage risk,
    and inventory pacing.

    State (Phase 1 core observation): [remaining_inventory, days_remaining].
    Action space: Discrete(N) — N fixed percentage price adjustments.
    Episode ends when inventory reaches 0 or days_remaining reaches 0,
    whichever comes first.
    """

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 1}

    def __init__(self, config: Optional[PricingEnvConfig] = None) -> None:
        super().__init__()

        self.config: PricingEnvConfig = config or PricingEnvConfig()
        self.render_mode: Optional[str] = self.config.render_mode

        # Action space + fixed adjustment mapping (action_space.py owns the
        # conversion logic; we keep a local tuple copy so it cannot be
        # mutated after construction).
        self._price_adjustment_pct: Tuple[float, ...] = tuple(
            self.config.price_adjustment_pct
        )
        self.action_space: spaces.Discrete = build_action_space(self.config)

        # Observation space (state.py owns the single definition).
        self.observation_space: spaces.Box = build_observation_space(self.config)

        # Demand simulator — stateless aside from config; all randomness
        # is routed through self.np_random (set by gym.Env.reset(seed=...))
        # so episodes remain reproducible from a seed alone.
        self._demand_simulator: DemandSimulator = DemandSimulator(self.config.demand)

        # Per-episode state. `None` until reset() is called — see the
        # `_is_reset` guard in step()/render() for why we don't fabricate
        # a fake initial state here.
        self._state: Optional[EnvState] = None
        self._is_reset: bool = False
        self._terminated: bool = False

        logger.info(
            "PricingEnvironment initialized | inventory=%d | horizon=%d days | "
            "actions=%d | render_mode=%s",
            self.config.initial_inventory,
            self.config.selling_horizon_days,
            self.action_space.n,
            self.render_mode,
        )

    # ------------------------------------------------------------------- #
    # reset
    # ------------------------------------------------------------------- #
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to the start of a new selling season.

        Parameters
        ----------
        seed : int, optional
            Seeds `self.np_random` via `super().reset(seed=seed)`. Routed
            straight into the demand simulator on every `step()` call, so
            passing a seed here makes the *entire episode* — every day's
            random arrivals and purchase decisions — exactly reproducible.
            This is what makes the Week 4 "1,000 simulated booking
            seasons" evaluation protocol scientifically valid: baselines
            and RL agents can be compared on identical demand realizations.
        options : dict, optional
            Optional scenario overrides (e.g. `{"initial_inventory": 50}`
            for a unit test or stress test). Accepted now so this method's
            signature never needs to change when that need arises.

        Returns
        -------
        observation : np.ndarray, shape (2,), dtype float32
        info : dict
        """
        super().reset(seed=seed)
        options = options or {}

        self._state = EnvState(
            remaining_inventory=int(
                options.get("initial_inventory", self.config.initial_inventory)
            ),
            days_remaining=int(
                options.get("selling_horizon_days", self.config.selling_horizon_days)
            ),
            current_price=float(self.config.base_price),
            current_step=0,
            episode_revenue=0.0,
        )
        self._is_reset = True
        self._terminated = False

        observation = self._state.to_observation()
        info = self._state.to_info(self.config)

        logger.debug(
            "Environment reset | seed=%s | initial_obs=%s", seed, observation.tolist()
        )

        if self.render_mode == "human":
            self.render()

        return observation, info

    # ------------------------------------------------------------------- #
    # step
    # ------------------------------------------------------------------- #
    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Apply a pricing action, simulate one day of demand, and advance
        the environment.

        Pipeline (each stage owned by its respective module):
          1. Validate action                          -> this method
          2. action index -> concrete price            -> action_space.py
          3. simulate today's demand -> units sold      -> demand_simulator.py
          4. update inventory / revenue / time          -> this method
          5. compute reward from the outcome            -> reward.py
          6. determine terminated/truncated              -> this method
          7. assemble observation + info                -> state.py

        Parameters
        ----------
        action : int
            Index into self.price_levels, chosen by the agent.

        Returns
        -------
        next_state, reward, done, info  (see transition.apply_transition)
        """
        price = self.price_levels[action]

        next_state, reward, done, info = apply_transition(
            price=price,
            inventory=self.inventory,
            days_left=self.days_left,
            base_demand=self.base_demand
        )

        # Update the environment's internal state
        self.inventory, self.days_left = next_state

        return next_state, reward, done, info

    def render(self):
        """Print the current state in a human-readable way."""
        print(f"Inventory left: {self.inventory} | Days left: {self.days_left}")

    def _get_state(self):
        """Return the current state as [inventory_left, days_left]."""
        return [self.inventory, self.days_left]
            Index into `self.action_space`.

        Returns
        -------
        observation : np.ndarray, shape (2,), dtype float32
        reward : float
        terminated : bool
            True if the episode ended naturally: inventory sold out or the
            deadline (days_remaining == 0) was reached.
        truncated : bool
            Always False in Phase 1 — there is no artificial step-count
            cutoff independent of the deadline. Returned explicitly
            (rather than omitted) because the 5-tuple return signature is
            required by the current Gymnasium API and downstream code
            (agents, evaluation harness) is written against it.
        info : dict
            Diagnostic detail: price offered, units sold, demand level,
            reward breakdown, etc. Never required for an agent to act.

        Raises
        ------
        RuntimeError
            If called before `reset()`, or after the episode has already
            terminated. We raise rather than silently continuing into an
            already-terminal state, because stepping past termination
            (e.g. selling from zero remaining inventory, or going to
            negative days_remaining) is undefined behaviour that would
            silently corrupt training data if allowed to proceed. Some
            reference Gymnasium environments only warn in this situation;
            we intentionally hold this environment to a stricter standard
            given it feeds directly into revenue-relevant business metrics.
        ValueError
            If `action` is not a valid member of `self.action_space`.
        """
        if not self._is_reset or self._state is None:
            raise RuntimeError(
                "step() called before reset(). Call reset() at the start "
                "of every episode before taking any action."
            )
        if self._terminated:
            raise RuntimeError(
                "step() called after the episode already terminated. "
                "Call reset() to start a new episode."
            )
        if not self.action_space.contains(action):
            raise ValueError(
                f"Invalid action {action!r}. Must be an integer in "
                f"[0, {self.action_space.n - 1}]."
            )

        # --- 2. Action -> price ------------------------------------------------ #
        price = action_to_price(
            action=action,
            current_price=self._state.current_price,
            price_adjustment_pct=self._price_adjustment_pct,
            min_price=self.config.min_price,
            max_price=self.config.max_price,
        )

        # --- 3. Simulate demand -------------------------------------------------#
        sale_outcome: SaleOutcome = self._demand_simulator.sample(
            price=price,
            reference_price=self.config.base_price,
            remaining_inventory=self._state.remaining_inventory,
            days_remaining=self._state.days_remaining,
            selling_horizon_days=self.config.selling_horizon_days,
            rng=self.np_random,
        )
        units_sold = sale_outcome.units_sold

        # --- 4. Update state ------------------------------------------------- #
        # Defensive max(..., 0): the demand simulator already caps
        # units_sold at remaining_inventory and step() forbids calling
        # past termination, so these should never go negative — the
        # clamp exists purely as a last line of defense against silent
        # state corruption, per the fail-loud philosophy of this codebase.
        self._state.remaining_inventory = max(
            0, self._state.remaining_inventory - units_sold
        )
        self._state.days_remaining = max(0, self._state.days_remaining - 1)
        self._state.current_price = price
        self._state.current_step += 1
        self._state.episode_revenue += price * units_sold
        self._state.last_units_sold = units_sold
        self._state.last_demand_level = sale_outcome.demand_level

        # --- 6. Termination ------------------------------------------------- #
        terminated = (
            self._state.remaining_inventory <= 0 or self._state.days_remaining <= 0
        )
        truncated = False
        self._terminated = terminated

        # --- 5. Reward -------------------------------------------------------- #
        reward_breakdown: RewardBreakdown = compute_reward(
            price=price,
            base_price=self.config.base_price,
            units_sold=units_sold,
            remaining_inventory_after=self._state.remaining_inventory,
            initial_inventory=self.config.initial_inventory,
            days_remaining_after=self._state.days_remaining,
            selling_horizon_days=self.config.selling_horizon_days,
            terminated=terminated,
            config=self.config.reward,
        )

        # --- 7. Observation + info -------------------------------------------- #
        observation = self._state.to_observation()
        info = self._state.to_info(self.config)
        info.update(
            {
                "action": action,
                "action_label": describe_action(action, self._price_adjustment_pct),
                "price": price,
                "units_sold": units_sold,
                "arrivals": sale_outcome.arrivals,
                "acceptance_probability": sale_outcome.acceptance_probability,
                "demand_level": sale_outcome.demand_level,
                "reward_breakdown": {
                    "revenue": reward_breakdown.revenue,
                    "discount_penalty": reward_breakdown.discount_penalty,
                    "unsold_penalty": reward_breakdown.unsold_penalty,
                    "balance_bonus": reward_breakdown.balance_bonus,
                },
            }
        )

        logger.debug(
            "step %d | action=%s (%s) | price=$%.2f | sold=%d | "
            "inventory=%d | days_left=%d | reward=%.2f | terminated=%s",
            self._state.current_step,
            action,
            info["action_label"],
            price,
            units_sold,
            self._state.remaining_inventory,
            self._state.days_remaining,
            reward_breakdown.total,
            terminated,
        )

        if self.render_mode == "human":
            self.render()

        return observation, reward_breakdown.total, terminated, truncated, info

    # ------------------------------------------------------------------- #
    # render
    # ------------------------------------------------------------------- #
    def render(self) -> Optional[str]:
        """
        Render the current state of the environment.

        Render mode is fixed at construction time (`config.render_mode`),
        matching the standard Gymnasium contract so this environment
        remains compatible with standard wrappers without modification.

        Returns
        -------
        Optional[str]
            None in "human" mode (prints to stdout); the rendered string
            in "ansi" mode; None (with a logged warning) if render_mode
            is unset.
        """
        if self.render_mode is None:
            logger.warning(
                "render() called but render_mode is None. "
                "Set render_mode='human' or 'ansi' in PricingEnvConfig to enable rendering."
            )
            return None
        if self._state is None:
            logger.warning("render() called before reset(); nothing to show yet.")
            return None

        sell_through_pct = (
            100.0
            * (self.config.initial_inventory - self._state.remaining_inventory)
            / self.config.initial_inventory
        )

        lines = [
            "=" * 56,
            f"  Step:                {self._state.current_step}",
            f"  Days remaining:      {self._state.days_remaining} / {self.config.selling_horizon_days}",
            f"  Inventory remaining: {self._state.remaining_inventory:>5d} / {self.config.initial_inventory}",
            f"  Sell-through:        {sell_through_pct:5.1f}%",
            f"  Current price:       ${self._state.current_price:,.2f}",
            f"  Last units sold:     {self._state.last_units_sold}",
            f"  Last demand level:   {self._state.last_demand_level}",
            f"  Episode revenue:     ${self._state.episode_revenue:,.2f}",
            "=" * 56,
        ]
        rendered = "\n".join(lines)

        if self.render_mode == "human":
            print(rendered)
            return None
        return rendered  # "ansi"

    # ------------------------------------------------------------------- #
    # close
    # ------------------------------------------------------------------- #
    def close(self) -> None:
        """No external resources held in Phase 1; implemented explicitly
        so this environment is a drop-in replacement anywhere code calls
        `env.close()` as part of the standard Gymnasium lifecycle."""
        logger.debug("PricingEnvironment closed.")

    def __repr__(self) -> str:
        if self._state is None:
            return "PricingEnvironment(not yet reset)"
        return (
            f"PricingEnvironment(inventory={self._state.remaining_inventory}/"
            f"{self.config.initial_inventory}, "
            f"days_remaining={self._state.days_remaining}/{self.config.selling_horizon_days}, "
            f"price=${self._state.current_price:.2f})"
        )


# --------------------------------------------------------------------------- #
# Optional Gymnasium registration
# --------------------------------------------------------------------------- #
try:
    gym.register(
        id="PricingEnv-v0",
        entry_point="pricing_env.pricing_env:PricingEnvironment",
        max_episode_steps=None,  # episode length governed by days_remaining, not a step wrapper
    )
except gym.error.Error:
    logger.debug("PricingEnv-v0 already registered; skipping re-registration.")


if __name__ == "__main__":
    # Manual smoke test: run one full episode with a fixed "hold price"
    # policy and confirm the environment behaves sensibly end to end.
    # NOT a substitute for tests/test_pricing_env.py (Day 5).
    logging.basicConfig(level=logging.INFO)

    env = PricingEnvironment(PricingEnvConfig(render_mode="ansi"))
    obs, info = env.reset(seed=42)
    print(env.render())

    HOLD_ACTION = env._price_adjustment_pct.index(0.0)
    terminated = truncated = False
    total_reward = 0.0

    while not (terminated or truncated):
        obs, reward, terminated, truncated, info = env.step(HOLD_ACTION)
        total_reward += reward

    print(env.render())
    print(f"\nEpisode finished | total_reward={total_reward:.2f} | "
          f"final_revenue=${info['episode_revenue']:.2f}")
