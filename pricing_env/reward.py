"""
Reward function for Dynamic Pricing Environment.
"""

from __future__ import annotations

from dataclasses import dataclass


MIN_PRICE = 100
MAX_PRICE = 1000


@dataclass(frozen=True)
class RewardConfig:
    """
    Tunable weighting coefficients for the reward function.
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
    Breakdown of reward components.
    """

    revenue: float
    discount_penalty: float
    unsold_penalty: float
    balance_bonus: float
    total: float


def calculate_reward(
    price,
    units_sold,
    remaining_inventory,
    remaining_days,
    season_over=False,
):
    """
    Simple reward calculation.
    """

    reward = 0.0

    revenue = price * units_sold
    reward += revenue

    if season_over and remaining_inventory > 0:
        reward -= remaining_inventory * 20

    if remaining_inventory == 0 and remaining_days > 0:
        reward -= remaining_days * 15

    if price < MIN_PRICE or price > MAX_PRICE:
        reward -= 100

    return reward


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
    Advanced reward calculation.
    """

    if base_price <= 0:
        raise ValueError(f"base_price must be > 0, got {base_price}")

    if initial_inventory <= 0:
        raise ValueError(
            f"initial_inventory must be > 0, got {initial_inventory}"
        )

    revenue = price * units_sold

    discount_depth = max(
        0.0,
        (base_price - price) / base_price,
    )

    discount_penalty = (
        config.lambda_discount
        * (discount_depth ** 2)
        * units_sold
        * base_price
    )

    unsold_penalty = 0.0

    if terminated and remaining_inventory_after > 0:
        unsold_penalty = (
            config.lambda_unsold
            * config.unsold_penalty_price_multiplier
            * base_price
            * remaining_inventory_after
        )

    if selling_horizon_days > 0:
        time_elapsed_fraction = (
            1.0
            - (
                days_remaining_after
                / selling_horizon_days
            )
        )
    else:
        time_elapsed_fraction = 1.0

    time_elapsed_fraction = min(
        max(time_elapsed_fraction, 0.0),
        1.0,
    )

    sold_fraction = (
        1.0
        - (
            remaining_inventory_after
            / initial_inventory
        )
    )

    sold_fraction = min(
        max(sold_fraction, 0.0),
        1.0,
    )

    pacing_gap = abs(
        sold_fraction - time_elapsed_fraction
    )

    balance_bonus = (
        config.lambda_balance
        * base_price
        * (1.0 - pacing_gap)
    )

    total = (
        revenue
        - discount_penalty
        - unsold_penalty
        + balance_bonus
    )

    return RewardBreakdown(
        revenue=revenue,
        discount_penalty=discount_penalty,
        unsold_penalty=unsold_penalty,
        balance_bonus=balance_bonus,
        total=total,
    )