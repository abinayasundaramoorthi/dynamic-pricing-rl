"""
random_policy.py

Validates PricingEnvironment (pricing_env package) by running it with a
RANDOM pricing policy: every day, pick a price-adjustment action at random
(no intelligence at all) and see what total revenue that produces over a
full selling season.

This gives a BASELINE number. Any real RL agent trained against this
environment must beat this random-policy revenue, otherwise it isn't
learning anything useful - it would be no better than guessing.
"""

import sys
import os
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from pricing_env import PricingEnvironment, PricingEnvConfig


def run_random_policy(env, num_episodes=100, seed=42):
    """
    Runs the environment for several full episodes (selling seasons),
    choosing a completely random price-adjustment action every single day.

    Parameters
    ----------
    env : PricingEnvironment
        The environment to test.
    num_episodes : int, default=100
        How many full selling seasons to simulate. Using many episodes
        (not just 1) gives a reliable AVERAGE, since a single random run
        could get lucky or unlucky.
    seed : int
        Base random seed so results are reproducible.

    Returns
    -------
    dict
        Summary statistics: revenue and units-sold per episode, plus
        overall averages.
    """
    episode_revenues = []
    episode_units_sold = []
    episode_final_prices = []

    for episode in range(num_episodes):
        obs, info = env.reset(seed=seed + episode)
        terminated = truncated = False

        while not (terminated or truncated):
            # Pick a price-adjustment action completely at random, using
            # Gymnasium's own action_space.sample() - this is the
            # "no intelligence" baseline policy.
            random_action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(random_action)

        episode_revenues.append(info["episode_revenue"])
        units_sold = env.config.initial_inventory - obs[0]
        episode_units_sold.append(units_sold)
        episode_final_prices.append(info["current_price"])

    summary = {
        "num_episodes": num_episodes,
        "avg_revenue": float(np.mean(episode_revenues)),
        "std_revenue": float(np.std(episode_revenues)),
        "min_revenue": float(np.min(episode_revenues)),
        "max_revenue": float(np.max(episode_revenues)),
        "avg_units_sold": float(np.mean(episode_units_sold)),
        "avg_units_unsold": float(env.config.initial_inventory - np.mean(episode_units_sold)),
        "sellout_rate": float(np.mean([u >= env.config.initial_inventory for u in episode_units_sold])),
        "episode_revenues": episode_revenues,
    }
    return summary


if __name__ == "__main__":
    config = PricingEnvConfig()
    env = PricingEnvironment(config)

    print("Running RANDOM pricing policy validation...")
    print(f"Config: initial_inventory={config.initial_inventory}, "
          f"selling_horizon_days={config.selling_horizon_days}, "
          f"base_price=₹{config.base_price}, "
          f"num_actions={env.action_space.n} "
          f"(adjustments={list(config.price_adjustment_pct)})\n")

    results = run_random_policy(env, num_episodes=100)

    print("=" * 50)
    print("RANDOM POLICY BASELINE RESULTS (100 episodes)")
    print("=" * 50)
    print(f"Average revenue per episode: ₹{results['avg_revenue']:.2f}")
    print(f"Std deviation of revenue:    ₹{results['std_revenue']:.2f}")
    print(f"Min revenue:                 ₹{results['min_revenue']:.2f}")
    print(f"Max revenue:                 ₹{results['max_revenue']:.2f}")
    print(f"Average units sold:          {results['avg_units_sold']:.1f} / {config.initial_inventory}")
    print(f"Average units UNSOLD:        {results['avg_units_unsold']:.1f}")
    print(f"Sellout rate:                {results['sellout_rate']*100:.1f}% of episodes")