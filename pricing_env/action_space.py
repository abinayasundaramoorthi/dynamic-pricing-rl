"""
action_space.py

Owns the discrete pricing action space: building the Gymnasium
Discrete action space, converting an action index into a concrete
price, and producing a human-readable label for each action.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from gymnasium import spaces


def build_action_space(config: Any) -> spaces.Discrete:
   
    return spaces.Discrete(len(config.price_adjustment_pct))


def action_to_price(
    action: int,
    current_price: float,
    price_adjustment_pct: Sequence[float],
    min_price: float,
    max_price: float,
) -> float:
    """
    Convert an action index into a bounded price.

    Parameters
    ----------
    action : int
        Index of the selected pricing action.
    current_price : float
        Current selling price.
    price_adjustment_pct : Sequence[float]
        Percentage adjustments for every action.
    min_price : float
        Minimum allowed price.
    max_price : float
        Maximum allowed price.

    Returns
    -------
    float
        The resulting price, clipped to [min_price, max_price].

    Raises
    ------
    ValueError
        If `action` is not a valid index into `price_adjustment_pct`.
    """
        
    

    if not (0 <= action < len(price_adjustment_pct)):
        raise ValueError(
            f"Invalid action {action}. "
            f"Must be between 0 and {len(price_adjustment_pct) - 1}."
        )

    pct_adjustment = price_adjustment_pct[action]
    new_price = current_price * (1 + pct_adjustment)

    return float(np.clip(new_price, min_price, max_price))


def describe_action(
    action: int,
    price_adjustment_pct: Sequence[float],
) -> str:
    
    
    

    if not (0 <= action < len(price_adjustment_pct)):
        raise ValueError(
            f"Invalid action {action}. "
            f"Must be between 0 and {len(price_adjustment_pct) - 1}."
        )

    pct = price_adjustment_pct[action]

    if pct < 0:
        return f"Discount {abs(pct):.0%}"

    if pct > 0:
        return f"Increase {pct:.0%}"
    return "Hold price"

    return "Hold Price"


if __name__ == "__main__":

    class DummyConfig:
        price_adjustment_pct = [-0.10, -0.05, 0.00, 0.05, 0.10]

    config = DummyConfig()

    space = build_action_space(config)

    print("Action Space:", space)

    current_price = 1000
    min_price = 500
    max_price = 1500

    for action in range(space.n):
        updated = action_to_price(
            action,
            current_price,
            config.price_adjustment_pct,
            min_price,
            max_price,
        )

        print(
            f"{action} -> "
            f"{describe_action(action, config.price_adjustment_pct)} "
            f"= ₹{updated:.2f}"
        )