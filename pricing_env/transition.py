"""
transition.py

Contains the state transition logic for the Dynamic Pricing RL environment.

Given the current state (inventory, days left) and the agent's chosen price,
this module calculates:
  - how many units get sold (demand)
  - the updated inventory
  - the reward earned
  - the next state
  - whether the episode has ended

This logic is used inside pricing_env.py's step() function.
"""

import numpy as np
from reward import calculate_reward


def apply_transition(price, inventory, days_left, base_demand=10):
    """
    Apply one pricing decision and calculate the resulting state transition.

    Parameters
    ----------
    price : float
        The price chosen by the agent for this time step.
    inventory : int
        Current inventory before this step.
    days_left : int
        Number of days remaining before this step.
    base_demand : float
        Baseline daily demand used in the demand simulation formula.

    Returns
    -------
    next_state : list
        [new_inventory, new_days_left] after applying this transition.
    reward : float
        Revenue earned this step.
    done : bool
        True if the episode has ended (no inventory left or no days left).
    info : dict
        Extra debug information (price used, units sold).
    """

    # --- 1. Apply pricing action ---
    # (price is already given directly here — the caller looks up the
    #  actual price value from the agent's chosen action index)

    # --- 2. Update demand ---
    # Simple demand simulation: higher price means fewer buyers.
    # We use an exponential decay based on price, then add randomness
    # (Poisson distribution) so it behaves like real, unpredictable customers.
    demand_rate = base_demand * np.exp(-0.003 * price)
    units_sold = int(np.random.poisson(demand_rate))

    # Can't sell more units than we actually have in stock.
    units_sold = min(units_sold, inventory)

    # --- 3. Update inventory ---
    new_inventory = inventory - units_sold

    # --- 4. Calculate reward using our reward function (reward.py) ---
    reward = calculate_reward(price, units_sold, new_inventory)

    # --- 5. Move time forward ---
    new_days_left = days_left - 1

    # --- 6. Generate next state ---
    next_state = [new_inventory, new_days_left]

    # --- 7. Validate state transition ---
    # Basic sanity checks to catch bugs early during development.
    assert new_inventory >= 0, "Inventory cannot go negative!"
    assert new_days_left >= 0, "Days left cannot go negative!"

    # --- 8. Check if episode is done ---
    done = (new_inventory <= 0) or (new_days_left <= 0)

    info = {
        "price": price,
        "units_sold": units_sold
    }

    return next_state, reward, done, info
