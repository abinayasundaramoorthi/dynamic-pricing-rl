
"""
random_policy.py
 
Validates the DynamicPricingEnv by running it with a RANDOM pricing policy:
every day, pick a price at random (no intelligence at all) and see what
total revenue that produces over a full selling season.
 
This gives us a BASELINE number. Any real RL agent we train later must
beat this random-policy revenue, otherwise it isn't learning anything
useful - it would be no better than guessing.
"""
 
import sys
import os
import numpy as np
 
# Allow importing environment.py from the project root, regardless of
# where this script is run from.
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from environment import DynamicPricingEnv
 
 
def run_random_policy(env, num_episodes=100, seed=42):
    """
    Runs the environment for several full episodes (selling seasons),
    choosing a completely random price every single day.
 
    Parameters
    ----------
    env : DynamicPricingEnv
        The environment to test.
    num_episodes : int, default=100
        How many full selling seasons to simulate. Using many episodes
        (not just 1) gives us a reliable AVERAGE, since a single random
        run could get lucky or unlucky.
    seed : int
        Random seed so results are reproducible.
 
    Returns
    -------
    dict
        Summary statistics: revenues per episode, units sold per episode,
        and overall averages.
    """
    np.random.seed(seed)
 
    episode_revenues = []      # total revenue earned in each episode
    episode_units_sold = []    # total units sold in each episode (out of total_inventory)
 
    for episode in range(num_episodes):
        obs, info = env.reset(seed=seed + episode)
        terminated = truncated = False
 
        while not (terminated or truncated):
            # Pick a price index completely at random, using Gymnasium's
            # own action_space.sample() - this is the "no intelligence"
            # baseline policy.
            random_action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(random_action)
 
        # After the episode ends, record what happened.
        episode_revenues.append(env.total_revenue)
        units_sold = env.total_inventory - env.inventory_remaining
        episode_units_sold.append(units_sold)
 
    summary = {
        "num_episodes": num_episodes,
        "avg_revenue": float(np.mean(episode_revenues)),
        "std_revenue": float(np.std(episode_revenues)),
        "min_revenue": float(np.min(episode_revenues)),
        "max_revenue": float(np.max(episode_revenues)),
        "avg_units_sold": float(np.mean(episode_units_sold)),
        "avg_units_unsold": float(env.total_inventory - np.mean(episode_units_sold)),
        "sellout_rate": float(np.mean([u >= env.total_inventory for u in episode_units_sold])),
        "episode_revenues": episode_revenues,
    }
    return summary
 
 
if __name__ == "__main__":
    env = DynamicPricingEnv()
 
    print("Running RANDOM pricing policy validation...")
    print(f"Config: total_inventory={env.total_inventory}, "
          f"selling_window={env.selling_window}, "
          f"price_levels={env.price_levels}\n")
 
    results = run_random_policy(env, num_episodes=100)
 
    print("=" * 50)
    print("RANDOM POLICY BASELINE RESULTS (100 episodes)")
    print("=" * 50)
    print(f"Average revenue per episode: ${results['avg_revenue']:.2f}")
    print(f"Std deviation of revenue:    ${results['std_revenue']:.2f}")
    print(f"Min revenue:                 ${results['min_revenue']:.2f}")
    print(f"Max revenue:                 ${results['max_revenue']:.2f}")
    print(f"Average units sold:          {results['avg_units_sold']:.1f} / {env.total_inventory}")
    print(f"Average units UNSOLD:        {results['avg_units_unsold']:.1f}")
    print(f"Sellout rate:                {results['sellout_rate']*100:.1f}% of episodes")