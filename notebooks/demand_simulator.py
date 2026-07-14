"""
demand_simulator.py

A simple, standalone demand model: given a price, estimate how many
units/customers will buy today.

Kept intentionally simple (a linear price-demand relationship + random
noise) so it can be swapped out later for something more realistic
(e.g., a model calibrated on the real airline/retail datasets) without
changing how the rest of the environment calls it.
"""

import numpy as np


def simulate_demand(price, base_demand=10, max_price=200, noise_std=1.5):
    """
    Estimate demand (number of units customers want to buy) at a given price.

    Parameters
    ----------
    price : float
        The price being charged today.
    base_demand : float, default=10
        The average demand when price is very low (close to 0).
        This is the "ceiling" of expected demand.
    max_price : float, default=200
        The price at which demand effectively drops to zero.
        Used to scale how strongly price pushes demand down.
    noise_std : float, default=1.5
        Standard deviation of random noise added to demand, so the
        model isn't perfectly deterministic (real customer behavior
        always has some randomness day to day).

    Returns
    -------
    int
        The simulated demand for today, never negative.
    """

    # Step 1: Compute the "expected" demand using a simple LINEAR relationship.
    # As price rises from 0 -> max_price, expected_demand falls from
    # base_demand -> 0. This directly encodes the requirement:
    # "Lower price -> Higher expected demand, Higher price -> Lower expected demand."
    price_ratio = price / max_price          # 0.0 (free) to 1.0 (at max price)
    expected_demand = base_demand * (1 - price_ratio)

    # Step 2: Demand can never be negative in real life, so if price is at or
    # above max_price, we clip expected demand at 0 instead of letting it go negative.
    expected_demand = max(0.0, expected_demand)

    # Step 3: Add random variation using NumPy, since real customer demand
    # fluctuates day to day even at the same price (weather, competitor
    # promotions, random chance, etc). We use a Normal (Gaussian) distribution
    # centered on our expected_demand, with spread controlled by noise_std.
    noisy_demand = np.random.normal(loc=expected_demand, scale=noise_std)

    # Step 4: Demand must be a non-negative whole number (you can't have -2
    # customers, or 3.7 customers). Round to nearest integer and clip at 0.
    final_demand = max(0, round(noisy_demand))

    return int(final_demand)


if __name__ == "__main__":
    # Quick sanity test: show how demand behaves across a range of prices.
    # We expect demand to trend DOWN as price goes UP, with some random noise.
    print(f"{'Price':>8} | {'Simulated Demand':>17}")
    print("-" * 30)

    test_prices = [10, 50, 90, 130, 160, 200, 250]
    for p in test_prices:
        demand = simulate_demand(p)
        print(f"{p:>8} | {demand:>17}")

    # Also show that the SAME price gives different results each time,
    # proving the randomness is working as intended.
    print("\nRunning price=$90 five times to show random variation:")
    for i in range(5):
        print(f"  Run {i+1}: demand = {simulate_demand(90)}")