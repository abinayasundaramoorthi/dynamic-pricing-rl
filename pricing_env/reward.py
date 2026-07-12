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