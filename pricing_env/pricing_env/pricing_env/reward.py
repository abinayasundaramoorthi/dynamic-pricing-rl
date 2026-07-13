"""
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

    Parameters
    ----------
    price : float
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
