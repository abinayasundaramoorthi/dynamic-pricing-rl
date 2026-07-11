"""
Action Space Module

Defines the discrete pricing actions used by the RL agent.

Action Mapping
--------------
0 -> Decrease price by 10%
1 -> Decrease price by 5%
2 -> Keep current price
3 -> Increase price by 5%
4 -> Increase price by 10%
"""

# Mapping from action ID to percentage change
ACTION_MAP = {
    0: -0.10,
    1: -0.05,
    2: 0.00,
    3: 0.05,
    4: 0.10,
}


def apply_action(current_price: float, action: int) -> float:
    """
    Applies the selected action to the current price.

    Parameters
    ----------
    current_price : float
        Current selling price.

    action : int
        Discrete action selected by the RL agent.

    Returns
    -------
    float
        Updated price after applying the action.

    Raises
    ------
    ValueError
        If the action is invalid.
    """

    if action not in ACTION_MAP:
        raise ValueError(f"Invalid action: {action}")

    percentage_change = ACTION_MAP[action]
    updated_price = current_price * (1 + percentage_change)

    return round(updated_price, 2)

if __name__ == "__main__":
    price = 1000

    for action in ACTION_MAP:
        print(
            f"Action {action}: ₹{price} -> ₹{apply_action(price, action)}"
        )