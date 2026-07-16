"""
Reward function for Dynamic Pricing Environment.
"""

MIN_PRICE = 100
MAX_PRICE = 1000


def calculate_reward(
    price,
    units_sold,
    remaining_inventory,
    remaining_days,
    season_over=False,
):
    """
    Calculate reward for one pricing step.
reward.py

This module contains the reward calculation logic for the
Dynamic Pricing Reinforcement Learning environment.

The reward tells the RL agent how "good" or "bad" its pricing
decision was at each step. The agent uses this feedback to
learn the best pricing policy over time.
"""


def calculate_reward(price, units_sold, inventory_left):
    """
    Calculate the reward for a single pricing decision.
Implements the reward function defined in the project design document
(Section 13 — Reward Design):

    r_t = (price_t * units_sold_t)
          - lambda_unsold  * unsold_penalty        (terminal step only)
          - lambda_discount * discount_depth_t^2 * units_sold_t * base_price
          + lambda_balance * inventory_pacing_bonus_t

Reward design is the single most consequential decision in this project —
the agent will optimize exactly what this function measures, not what we
intended it to measure. Every term below exists to counter a specific,
named failure mode; see the inline docstrings for which one.

This module has no dependency on Gymnasium or the environment class, by
design — it is a pure function of (price, units_sold, inventory, time),
independently unit-testable and independently tunable by the Data Scientist
role (see Stakeholders, design doc Section 7) without touching environment
mechanics.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RewardConfig:
    """
    Tunable weighting coefficients for the reward function.

    Attributes
    ----------
    lambda_unsold : float
        Weight on the terminal unsold-inventory penalty. Higher values
        push the agent harder toward clearing all inventory before the
        deadline, at the cost of potentially discounting more aggressively
        to guarantee sales — this is the primary "spoilage vs. margin"
        dial referenced by the Revenue Manager persona's safety bounds.
    lambda_discount : float
        Weight on the over-discounting penalty. Guards against the classic
        RL reward-hacking failure mode where an agent learns to discount
        to (near) zero every step to guarantee conversions, which
        maximizes short-term sale probability while destroying margin.
    lambda_balance : float
        Weight on the inventory-pacing shaping bonus. This exists purely
        to densify the learning signal early in training — the
        unsold-inventory penalty only fires on the terminal step of an
        episode that can be dozens of steps long, which is a very sparse
        signal for an agent to learn from. This bonus gives (small) dense
        feedback every step so gradient signal isn't limited to episode
        boundaries.
    unsold_penalty_price_multiplier : float
        Penalty per unsold unit, expressed as a multiple of base_price.
        A value of 1.0 means each unsold unit "costs" one full base-price
        unit of reward — reflecting that it represents genuinely lost
        revenue (Section 2.4 of the design doc: spoilage is permanent,
        unrecoverable loss, not merely deferred).
    """

    lambda_unsold: float = 1.0
    lambda_discount: float = 0.5
    lambda_balance: float = 0.1
    unsold_penalty_price_multiplier: float = 1.0

    def __post_init__(self) -> None:
        for name in (
            "lambda_unsold",
            "lambda_discount",
            "lambda_balance",
            "unsold_penalty_price_multiplier",
        ):
            value = getattr(self, name)
            if value < 0:
                raise ValueError(f"{name} must be >= 0, got {value}")


@dataclass(frozen=True)
class RewardBreakdown:
    """
    Transparent decomposition of a single step's reward.

    Returned (rather than just a bare float) so every term can be logged
    per step during training/evaluation — this is what lets the Data
    Scientist diagnose *why* an agent's reward changed (e.g. "revenue is up
    but so is discount_penalty" is a very different diagnosis from "revenue
    dropped"), and it is what a revenue manager needs to trust the policy
    rather than treat it as an opaque number.
    """

    revenue: float
    discount_penalty: float
    unsold_penalty: float
    balance_bonus: float
    total: float


def compute_reward(
    price: float,
    base_price: float,
    units_sold: int,
    remaining_inventory_after: int,
    initial_inventory: int,
    days_remaining_after: int,
    selling_horizon_days: int,
    terminated: bool,
    config: RewardConfig,
) -> RewardBreakdown:
    """
    Compute the reward for one environment step.

    Parameters
    ----------
    price : float
    units_sold : int
    remaining_inventory : int
    remaining_days : int
    season_over : bool

    Returns
    -------
    float
    """
https://github.com/abinayasundaramoorthi/dynamic-pricing-rl/pull/56/conflict?name=pricing_env%252Freward.py&base_oid=7cc093e56a2eceec9e7e3b56e7c6340ba3491a6a&head_oid=ba1f7c8d6ebe37af3dc1b87045c834d6356e3655
    reward = 0.0

    # Revenue
    revenue = price * units_sold
    reward += revenue

    # Overstock penalty
    if season_over and remaining_inventory > 0:
        reward -= remaining_inventory * 20

    # Understock penalty
    if remaining_inventory == 0 and remaining_days > 0:
        reward -= remaining_days * 15

    # Invalid pricing penalty
    if price < MIN_PRICE or price > MAX_PRICE:
        reward -= 100

    return reward
        The price set by the agent for this time step.
    units_sold : int
        The number of units sold at this price during this time step.
    inventory_left : int
        The number of units remaining in inventory after this sale.
        (Not used in the initial version, but kept as a parameter
        so it's ready for future improvements — see notes below.)

    Returns
    -------
    reward : float
        The reward value for this step. In the initial version,
        this is simply the revenue earned.
    """

    # ------------------------------------------------------------
    # INITIAL REWARD LOGIC (Week 1 baseline)
    # ------------------------------------------------------------
    # The simplest possible reward: revenue earned in this step.
    # Revenue = price * units_sold
    #
    # Example:
    #   price = 400, units_sold = 5
    #   revenue = 400 * 5 = 2000
    #
    # This encourages the agent to sell as many units as possible
    # at as high a price as possible.
    revenue = price * units_sold

    reward = revenue

    # ------------------------------------------------------------
    # FUTURE IMPROVEMENTS (not implemented yet)
    # ------------------------------------------------------------
    # The reward function can be made smarter later by adding
    # penalties or bonuses. Some ideas for future weeks:
    #
    # 1. Inventory Penalty:
    #    If too much inventory is left near the end of the season,
    #    subtract a penalty to discourage the agent from holding
    #    prices too high for too long (unsold stock = lost value).
    #
    #    Example (not active yet):
    #    if inventory_left > some_threshold and days_left < some_limit:
    #        reward -= inventory_penalty_weight * inventory_left
    #
    # 2. Customer Satisfaction Penalty:
    #    If prices change too frequently or spike suddenly, it could
    #    hurt customer trust. A penalty could be added for large
    #    price jumps between steps.
    #
    #    Example (not active yet):
    #    reward -= satisfaction_penalty_weight * abs(price - previous_price)
    #
    # 3. Overbooking Risk Penalty:
    #    If the agent sells too fast and risks running out of stock
    #    before the season ends (missing potential future high-price
    #    sales), a small penalty could balance this risk.
    #
    # These are intentionally left out for now to keep the first
    # version simple. They can be added once the basic agent is
    # working correctly with the simple revenue-based reward.
    # ------------------------------------------------------------

    return reward
        The price offered this step (post action-to-price conversion).
    base_price : float
        Reference price; used to normalize the discount and unsold
        penalties so their magnitude is comparable to revenue regardless
        of the absolute price scale configured for a given experiment.
    units_sold : int
        Units sold this step (already capped at available inventory by
        the demand simulator).
    remaining_inventory_after : int
        Inventory remaining AFTER this step's sale is applied.
    initial_inventory : int
        Inventory at the start of the episode; used to compute the actual
        sell-through fraction for the pacing bonus.
    days_remaining_after : int
        Days remaining AFTER this step (i.e. post-decrement).
    selling_horizon_days : int
        Total length of the selling season; used to compute the ideal
        (linear) pacing fraction for the pacing bonus.
    terminated : bool
        Whether this step ended the episode (deadline reached or inventory
        exhausted). The unsold-inventory penalty only applies here — it is
        a *terminal* cost, not a per-step cost, because unsold inventory
        only becomes an irreversible loss once no more selling days remain.
    config : RewardConfig
        Tunable weighting coefficients.

    Returns
    -------
    RewardBreakdown
        Full decomposition, including `.total` — the scalar reward to
        return from `PricingEnvironment.step()`.
    """
    if base_price <= 0:
        raise ValueError(f"base_price must be > 0, got {base_price}")
    if initial_inventory <= 0:
        raise ValueError(f"initial_inventory must be > 0, got {initial_inventory}")

    # --- Revenue term ------------------------------------------------- #
    # The direct, immediate business value captured this step.
    revenue = price * units_sold

    # --- Over-discounting penalty --------------------------------------#
    # discount_depth is 0 at/above base_price and grows toward 1 as price
    # falls toward zero. Squaring makes small discounts cheap and deep
    # discounts disproportionately expensive — a deliberately convex
    # shape so the agent isn't equally punished for a modest 5% discount
    # and a reward-hacking 90% discount.
    #
    # Scaled by (units_sold * base_price) rather than by realized revenue:
    # scaling by revenue would let the agent "escape" the penalty simply
    # by discounting further (since discounted revenue shrinks as price
    # drops), which would defeat the purpose of the penalty entirely.
    # Anchoring the scale to base_price keeps the penalty meaningful
    # regardless of how deep the discount goes.
    discount_depth = max(0.0, (base_price - price) / base_price)
    discount_penalty = (
        config.lambda_discount * (discount_depth**2) * units_sold * base_price
    )

    # --- Terminal unsold-inventory penalty ------------------------------#
    # Fires only at episode end, proportional to units left unsold.
    # This is the term responsible for long-horizon credit assignment:
    # it forces the agent to weigh today's "hold price high" decision
    # against a cost that is only realized potentially many steps later —
    # exactly the kind of temporal credit assignment a rule-based system
    # cannot do, and precisely why this is an RL problem (see Section 6 of
    # the design doc).
    unsold_penalty = 0.0
    if terminated and remaining_inventory_after > 0:
        unsold_penalty = (
            config.lambda_unsold
            * config.unsold_penalty_price_multiplier
            * base_price
            * remaining_inventory_after
        )

    # --- Inventory pacing bonus ----------------------------------------#
    # Dense, small-magnitude shaping signal: reward the agent for keeping
    # actual sell-through roughly in line with a naive linear pacing
    # curve (e.g. 50% of days elapsed -> roughly 50% of inventory sold).
    # This exists purely to give the agent *some* signal every step,
    # since the unsold penalty above is sparse (terminal-only) and would
    # otherwise leave early training with very little gradient to learn
    # from. lambda_balance defaults small (0.1) specifically so this
    # shaping term cannot dominate the true objective (revenue).
    if selling_horizon_days > 0:
        time_elapsed_fraction = 1.0 - (days_remaining_after / selling_horizon_days)
    else:
        time_elapsed_fraction = 1.0
    time_elapsed_fraction = min(max(time_elapsed_fraction, 0.0), 1.0)

    sold_fraction = 1.0 - (remaining_inventory_after / initial_inventory)
    sold_fraction = min(max(sold_fraction, 0.0), 1.0)

    pacing_gap = abs(sold_fraction - time_elapsed_fraction)
    balance_bonus = config.lambda_balance * base_price * (1.0 - pacing_gap)

    total = revenue - discount_penalty - unsold_penalty + balance_bonus

    return RewardBreakdown(
        revenue=revenue,
        discount_penalty=discount_penalty,
        unsold_penalty=unsold_penalty,
        balance_bonus=balance_bonus,
        total=total,
    )
