"""
pricing_env.py

Custom Gymnasium environment for the Travel & Hospitality Dynamic Pricing
Reinforcement Learning project.

This module defines the FOUNDATIONAL environment skeleton (Day 3 deliverable).
It establishes the observation space, action space, episode lifecycle, and
rendering behaviour that every downstream module (demand model, baselines,
Q-Learning, DQN, evaluation harness) will plug into.

Design intent
-------------
This file intentionally does NOT implement market/demand dynamics yet.
`step()` is a contract-only placeholder: it validates inputs and defines the
exact return signature the rest of the team must code against, but raises
`NotImplementedError` so that no agent can silently train against a fake,
zero-reward environment. The demand model is owned by Day 4, and the full
`step()` logic (wiring the demand model into this environment) is owned by
Day 5. Anyone importing this file before then will fail loudly, not silently.

Author: ML Engineering Team
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PricingEnvConfig:
    """
    Immutable configuration for PricingEnvironment.

    Using a dataclass (rather than a long list of constructor kwargs) gives
    us three things that matter for a multi-person project:

    1. A single, typed, validated source of truth for environment
       parameters, instead of every module guessing constructor arg order.
    2. Easy serialization to/from YAML/JSON later (configs/ folder) without
       touching this file — Hydra/OmegaConf or a simple `PricingEnvConfig(**yaml_dict)`
       will work as-is.
    3. Immutability (`frozen=True`) prevents an agent or training loop from
       accidentally mutating shared config mid-experiment, which is a classic
       source of irreproducible RL results.

    Attributes
    ----------
    initial_inventory : int
        Total number of sellable units (seats/rooms) available at the start
        of each episode. Must be a positive integer.
    selling_horizon_days : int
        Number of days in the selling season before the deadline
        (departure / check-in). Must be a positive integer.
    base_price : float
        Reference price (in currency units) that price actions adjust
        relative to. Must be strictly positive.
    price_adjustment_pct : List[float]
        Ordered list of fractional price adjustments defining the discrete
        action space, e.g. -0.20 means "20% discount off base_price".
        The order defines the action index -> price mapping and MUST stay
        stable across training/evaluation runs (changing it invalidates any
        already-trained Q-table or DQN policy).
    min_price : Optional[float]
        Hard floor on price after adjustment, used to prevent economically
        nonsensical (e.g. negative or near-zero) prices. Defaults to 1% of
        base_price if not provided.
    max_price : Optional[float]
        Hard ceiling on price after adjustment. Defaults to 3x base_price
        if not provided.
    render_mode : Optional[str]
        One of {None, "human", "ansi"}. Follows the standard Gymnasium
        render-mode contract so this env is compatible with any Gymnasium
        wrapper (e.g. RecordVideo) out of the box.
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

    def __post_init__(self) -> None:
        """Validate configuration eagerly, at construction time.

        Why: failing fast here means a bad config raises a clear, immediate
        ValueError before any training run starts, instead of surfacing as
        a confusing NaN reward or index-out-of-bounds error 400 episodes
        into a DQN training job.
        """
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

        # min_price / max_price have defaults derived from base_price.
        # Dataclass is frozen, so we use object.__setattr__ to set derived
        # defaults during __post_init__ — this is the standard pattern for
        # frozen dataclasses with computed fields.
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

    This class defines the MDP interface — observation space, action space,
    episode reset/termination, and rendering — that all agents (baselines,
    Q-Learning, DQN) and the evaluation harness are built against.

    State (Phase 1 core observation)
    ---------------------------------
    A 2-element vector: [remaining_inventory, days_remaining].
    This matches the MDP design document exactly. Additional state
    variables (current_price, demand_level, seasonality) are tracked
    internally and exposed via `info`, so they can be folded into the
    observation later (Phase 1 extension) without breaking this contract.

    Action space
    ------------
    Discrete(N), where N = len(config.price_adjustment_pct). Each action
    index maps to a fixed percentage price adjustment relative to
    `config.base_price` (see PricingEnvConfig). Discrete actions are used
    deliberately in Phase 1 — see the Action Space section of the project
    design document for the full justification (tabular Q-Learning
    compatibility, auditability, simpler debugging before moving to
    continuous control in Phase 2).

    Reward
    ------
    Not computed in this file. `step()` is a placeholder — see module
    docstring. Reward logic is owned by the reward-function design
    (Section 13 of the project design doc) and wired in during Day 5
    environment integration.

    Notes for implementers plugging into this class
    -------------------------------------------------
    - Always call `reset()` before the first `step()` call, per the
      standard Gymnasium contract. `step()` will raise if called before
      `reset()`.
    - Do not mutate `self.config` — it is frozen intentionally.
    - `render_mode` is fixed at construction time (per Gymnasium spec) and
      cannot be changed mid-episode.
    """

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 1}

    def __init__(self, config: Optional[PricingEnvConfig] = None) -> None:
        """
        Initialize the pricing environment.

        Parameters
        ----------
        config : PricingEnvConfig, optional
            Environment configuration. If not provided, sensible Phase-1
            defaults are used (see PricingEnvConfig). Accepting a config
            object (rather than **kwargs) keeps this constructor stable
            even as the number of tunable parameters grows.

        Why __init__ only *declares* spaces and does not set episode state:
        Gymnasium's contract is that `__init__` configures the environment
        (spaces, rendering, static parameters), while `reset()` is
        responsible for (re)initializing per-episode state. Mixing the two
        causes subtle bugs where `gym.make(...)` produces an env that is
        already "mid-episode" before `reset()` is ever called.
        """
        super().__init__()

        self.config: PricingEnvConfig = config or PricingEnvConfig()
        self.render_mode: Optional[str] = self.config.render_mode

        # ------------------------------------------------------------- #
        # Action space
        # ------------------------------------------------------------- #
        # Discrete action per the project design doc. Each index maps to
        # a percentage price adjustment defined in config.price_adjustment_pct.
        # We keep the mapping (index -> pct -> price) as a fixed tuple so it
        # is immutable for the lifetime of the environment instance — this
        # matters because a trained Q-table / DQN policy's action indices
        # are only meaningful relative to a fixed mapping.
        self._price_adjustment_pct: Tuple[float, ...] = tuple(
            self.config.price_adjustment_pct
        )
        self.action_space: spaces.Discrete = spaces.Discrete(
            len(self._price_adjustment_pct)
        )

        # ------------------------------------------------------------- #
        # Observation space
        # ------------------------------------------------------------- #
        # Phase 1 core state: [remaining_inventory, days_remaining].
        # We use a Box with dtype=float32 (rather than int32) because:
        #   1. This is the standard dtype expected by most Gymnasium
        #      wrappers and all common DQN implementations (Stable-Baselines3,
        #      CleanRL, etc.) — using int32 here would force every downstream
        #      neural-network agent to cast on every step.
        #   2. Tabular Q-Learning can still trivially discretize a float32
        #      observation (round/bin it); the reverse (recovering float
        #      precision from an int32 space) is not possible, so float32
        #      is the more general, future-proof choice.
        low = np.array([0.0, 0.0], dtype=np.float32)
        high = np.array(
            [
                float(self.config.initial_inventory),
                float(self.config.selling_horizon_days),
            ],
            dtype=np.float32,
        )
        self.observation_space: spaces.Box = spaces.Box(
            low=low, high=high, dtype=np.float32
        )

        # ------------------------------------------------------------- #
        # Per-episode state — declared here for clarity/type-hinting only.
        # Actual values are assigned in reset(). We deliberately initialize
        # them to None/0 rather than leaving them undeclared, so that any
        # IDE, type checker, or new team member reading this class can see
        # the full internal state surface in one place.
        # ------------------------------------------------------------- #
        self.remaining_inventory: int = self.config.initial_inventory
        self.days_remaining: int = self.config.selling_horizon_days
        self.current_price: float = self.config.base_price
        self.current_step: int = 0
        self.episode_revenue: float = 0.0
        self.sales_log: List[Dict[str, Any]] = []

        # Tracks whether reset() has been called at least once. step()
        # checks this and raises rather than silently operating on
        # uninitialized state — a common source of hard-to-debug RL bugs.
        self._is_reset: bool = False

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
        Reset the environment to the start of a new selling season (episode).

        Parameters
        ----------
        seed : int, optional
            Seed for this episode's random number generator. Passed to
            `super().reset(seed=seed)`, which is the standard Gymnasium
            mechanism for seeding `self.np_random`. Passing a seed here
            (rather than seeding a global RNG) is what makes it possible to
            run fully reproducible, parallel evaluation episodes — critical
            for the Week 4 "1,000 simulated booking seasons" evaluation
            protocol, where each season must be independently seedable.
        options : dict, optional
            Optional dict for scenario overrides (e.g. forcing a specific
            starting inventory for a unit test or a stress-test scenario).
            Not used by the base implementation yet; accepted now so the
            method signature never has to change when that need arises.

        Returns
        -------
        observation : np.ndarray
            Initial observation, shape (2,), dtype float32:
            [remaining_inventory, days_remaining].
        info : dict
            Auxiliary diagnostic info not part of the formal observation
            (current_price, episode number, etc.). Per Gymnasium convention,
            nothing in `info` should be required for an agent to act —
            it exists for logging/debugging only.
        """
        # IMPORTANT: this seeds self.np_random and stores self._np_random_seed.
        # Always call this before touching any randomness in reset/step.
        super().reset(seed=seed)

        options = options or {}

        self.remaining_inventory = int(
            options.get("initial_inventory", self.config.initial_inventory)
        )
        self.days_remaining = int(
            options.get("selling_horizon_days", self.config.selling_horizon_days)
        )
        self.current_price = float(self.config.base_price)
        self.current_step = 0
        self.episode_revenue = 0.0
        self.sales_log = []
        self._is_reset = True

        observation = self._get_observation()
        info = self._get_info()

        logger.debug(
            "Environment reset | seed=%s | initial_obs=%s", seed, observation.tolist()
        )

        if self.render_mode == "human":
            self.render()

        return observation, info

    # ------------------------------------------------------------------- #
    # step (PLACEHOLDER — Day 4/5 deliverable)
    # ------------------------------------------------------------------- #
    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Apply a pricing action and advance the environment by one time step.

        THIS IS A PLACEHOLDER. It intentionally does not implement demand
        simulation or reward computation yet.

        Why raise NotImplementedError instead of returning dummy/zero values:
        A `step()` that silently returns e.g. `(obs, 0.0, False, False, {})`
        would let a teammate accidentally start training an agent against a
        fake, reward-free environment — the training would "run" without
        errors and produce a policy that looks trained but has learned
        nothing meaningful. Raising here converts that silent failure mode
        into a loud, immediate one, which is the correct default for
        unfinished core infrastructure in a multi-person codebase.

        Once implemented (Day 4 demand model + Day 5 integration), this
        method MUST:
          1. Validate `action` is contained in `self.action_space`.
          2. Convert `action` -> price using `self._action_to_price(action)`.
          3. Query the demand model for a stochastic sale outcome given
             (price, remaining_inventory, days_remaining).
          4. Update `self.remaining_inventory`, `self.days_remaining`,
             `self.current_price`, `self.current_step`, `self.episode_revenue`.
          5. Compute reward per the reward function design (Section 13).
          6. Determine `terminated` (episode-ending condition reached:
             inventory == 0 or days_remaining == 0) vs. `truncated` (external
             cutoff, e.g. a max-step wrapper — not used in Phase 1 but
             required by the Gymnasium 5-tuple API for forward compatibility).
          7. Return `(observation, reward, terminated, truncated, info)`.

        Parameters
        ----------
        action : int
            Index into the discrete action space; must satisfy
            `self.action_space.contains(action)`.

        Returns
        -------
        Never returns in the current placeholder implementation.

        Raises
        ------
        RuntimeError
            If called before `reset()`.
        ValueError
            If `action` is not a valid member of `self.action_space`.
        NotImplementedError
            Always, once the above checks pass — this is the Day 3
            placeholder contract.
        """
        if not self._is_reset:
            raise RuntimeError(
                "step() called before reset(). Call reset() at the start "
                "of every episode before taking any action."
            )
        if not self.action_space.contains(action):
            raise ValueError(
                f"Invalid action {action!r}. Must be an integer in "
                f"[0, {self.action_space.n - 1}]."
            )

        # TODO(Day 4 owner): implement stochastic demand model in
        #   environment/demand_model.py, exposing something like
        #   demand_model.sample_sale(price, remaining_inventory, days_remaining, rng)
        # TODO(Day 5 owner): wire the demand model into this method, compute
        #   reward per Section 13 of the design doc, and update episode state.
        raise NotImplementedError(
            "PricingEnvironment.step() is a Day 3 placeholder. "
            "Demand-model integration is scheduled for Day 4-5. "
            "See TODO comments in pricing_env.py."
        )

    # ------------------------------------------------------------------- #
    # render
    # ------------------------------------------------------------------- #
    def render(self) -> Optional[str]:
        """
        Render the current state of the environment.

        Follows the Gymnasium render contract: the render mode is fixed at
        construction time via `config.render_mode`, and `render()` takes no
        arguments. This keeps the environment compatible with standard
        Gymnasium wrappers (e.g. `RecordVideo`, `HumanRendering`) without
        modification.

        Returns
        -------
        Optional[str]
            - In "human" mode: prints to stdout and returns None (matches
              Gymnasium convention for human-mode rendering).
            - In "ansi" mode: returns the rendered string instead of
              printing, so callers (e.g. a Jupyter notebook or logging
              pipeline) can capture and store it.
            - If render_mode is None: logs a warning and returns None,
              rather than raising — a missing render mode should never
              crash a training run.
        """
        if self.render_mode is None:
            logger.warning(
                "render() called but render_mode is None. "
                "Set render_mode='human' or 'ansi' in PricingEnvConfig to enable rendering."
            )
            return None

        sell_through_pct = (
            100.0
            * (self.config.initial_inventory - self.remaining_inventory)
            / self.config.initial_inventory
        )

        lines = [
            "=" * 52,
            f"  Step:               {self.current_step}",
            f"  Days remaining:     {self.days_remaining} / {self.config.selling_horizon_days}",
            f"  Inventory remaining:{self.remaining_inventory:>5d} / {self.config.initial_inventory}",
            f"  Sell-through:       {sell_through_pct:5.1f}%",
            f"  Current price:      ${self.current_price:,.2f}",
            f"  Episode revenue:    ${self.episode_revenue:,.2f}",
            "=" * 52,
        ]
        rendered = "\n".join(lines)

        if self.render_mode == "human":
            print(rendered)
            return None
        # render_mode == "ansi"
        return rendered

    # ------------------------------------------------------------------- #
    # close
    # ------------------------------------------------------------------- #
    def close(self) -> None:
        """
        Perform any necessary cleanup (open file handles, plotting windows,
        etc.). No external resources are held in Phase 1, so this is a
        no-op — but it is implemented explicitly (rather than omitted) so
        that this environment is a drop-in replacement anywhere code calls
        `env.close()` as part of the standard Gymnasium lifecycle.
        """
        logger.debug("PricingEnvironment closed.")

    # ------------------------------------------------------------------- #
    # Internal helpers
    # ------------------------------------------------------------------- #
    def _get_observation(self) -> np.ndarray:
        """
        Assemble the current observation vector.

        Kept as a private helper (rather than inlined in reset()/step()) so
        there is exactly ONE place that defines what "the observation" is.
        When the observation is extended in a later phase (e.g. adding
        current_price or seasonality), only this method needs to change.
        """
        return np.array(
            [float(self.remaining_inventory), float(self.days_remaining)],
            dtype=np.float32,
        )

    def _get_info(self) -> Dict[str, Any]:
        """
        Assemble the auxiliary info dict returned alongside observations.

        Exposes internal state that is useful for debugging/logging but is
        deliberately NOT part of the formal observation space, per
        Gymnasium convention (agents must not depend on `info` to act).
        """
        return {
            "current_price": self.current_price,
            "current_step": self.current_step,
            "episode_revenue": self.episode_revenue,
            "initial_inventory": self.config.initial_inventory,
            "selling_horizon_days": self.config.selling_horizon_days,
        }

    def _action_to_price(self, action: int) -> float:
        """
        Convert a discrete action index into a concrete price.

        This is implemented now (not just a placeholder) because it has no
        dependency on the demand model — it is a pure function of
        `action` and `self.current_price` / `self.config`, so there is no
        reason to block it on Day 4/5 work. Downstream owners of `step()`
        should call this method rather than reimplementing the price
        calculation, to guarantee every module agrees on the action ->
        price mapping.

        Parameters
        ----------
        action : int
            Index into the discrete action space.

        Returns
        -------
        float
            The resulting price, clipped to [config.min_price, config.max_price].
        """
        if not self.action_space.contains(action):
            raise ValueError(
                f"Invalid action {action!r}. Must be an integer in "
                f"[0, {self.action_space.n - 1}]."
            )
        pct_adjustment = self._price_adjustment_pct[action]
        raw_price = self.current_price * (1.0 + pct_adjustment)
        return float(
            np.clip(raw_price, self.config.min_price, self.config.max_price)
        )

    def __repr__(self) -> str:
        return (
            f"PricingEnvironment(inventory={self.remaining_inventory}/"
            f"{self.config.initial_inventory}, "
            f"days_remaining={self.days_remaining}/{self.config.selling_horizon_days}, "
            f"price=${self.current_price:.2f})"
        )


# --------------------------------------------------------------------------- #
# Optional Gymnasium registration
# --------------------------------------------------------------------------- #
# Registering the environment lets any teammate do:
#     import gymnasium as gym
#     env = gym.make("PricingEnv-v0")
# instead of importing PricingEnvironment directly. This is wrapped in a
# try/except because gym.register() raises if called twice (e.g. on module
# reload in a notebook) — we don't want that to crash an import.
try:
    gym.register(
        id="PricingEnv-v0",
        entry_point="environment.pricing_env:PricingEnvironment",
        max_episode_steps=None,  # episode length is governed by days_remaining, not a step wrapper
    )
except gym.error.Error:
    logger.debug("PricingEnv-v0 already registered; skipping re-registration.")


if __name__ == "__main__":
    # Minimal manual smoke test — NOT a substitute for the Day 5 unit test
    # suite (tests/test_pricing_env.py), but useful for a quick sanity check
    # while developing this file interactively.
    logging.basicConfig(level=logging.INFO)

    env = PricingEnvironment(PricingEnvConfig(render_mode="human"))
    obs, info = env.reset(seed=42)
    env.render()
    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)
    print("Action 0 -> price:", env._action_to_price(0))
    print("Action 6 -> price:", env._action_to_price(6))

    try:
        env.step(env.action_space.sample())
    except NotImplementedError as e:
        print(f"\nExpected placeholder behaviour confirmed: {e}")