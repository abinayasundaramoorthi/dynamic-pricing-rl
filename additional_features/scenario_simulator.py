"""
scenario_simulator.py

What-If Scenario Simulator — lets someone type in a custom business
situation (inventory remaining, days remaining) and instantly see what
each pricing strategy would recommend, along with the projected outcome
if that strategy were followed for the rest of the season.

Unlike the other evaluation tools in this project (which run FULL random
episodes), this simulator answers a specific, one-off question: "given
THIS exact situation right now, what would each policy do next, and how
would the rest of the season likely play out under each one?"

Reuses the trained agents and baseline policies already built in
Issue #91 (policy_evaluator.py).
"""

import sys
import os
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'evaluation'))

from pricing_env.pricing_env import PricingEnvironment, PricingEnvConfig
from agents.q_learning_agent import QLearningAgent
from agents.dqn_agent import DQNAgent
from policy_evaluator import (
    make_fixed_price_policy,
    make_daily_discount_policy,
    make_random_policy,
    make_agent_policy,
)


def simulate_rest_of_season(env, policy_fn, inventory, days_remaining, seed=None):
    """
    Simulate how the REST of the season would likely play out, starting
    from a custom (inventory, days_remaining) situation, if a given
    policy is followed from here onward.

    This directly overrides the environment's internal state after
    reset(), rather than using the environment's normal starting
    conditions -- letting us answer "what if we were in THIS situation
    right now" rather than only "what if we start a fresh season."
    """
    observation, info = env.reset(seed=seed)

    # Override the environment's state to match the custom scenario
    env._state.remaining_inventory = int(inventory)
    env._state.days_remaining = int(days_remaining)
    observation = env._state.to_observation()

    total_revenue = 0.0
    total_reward = 0.0
    prices_charged = []
    done = False

    while not done:
        action = policy_fn(observation)
        next_observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        total_revenue += info["reward_breakdown"]["revenue"]
        total_reward += reward
        prices_charged.append(info["price"])

        observation = next_observation

    final_inventory = float(observation[0])
    sold_units = inventory - final_inventory

    return {
        "immediate_price": prices_charged[0] if prices_charged else None,
        "projected_total_revenue": total_revenue,
        "projected_total_reward": total_reward,
        "projected_units_sold": sold_units,
        "projected_unsold_inventory": final_inventory,
    }


def run_scenario(env, policies, inventory, days_remaining, num_simulations=20):
    """
    Run a custom scenario through every policy, averaging the projected
    outcome over several simulations (since demand is random) for a more
    reliable estimate than just one simulation run.
    """
    print(f"\n{'='*70}")
    print(f"SCENARIO: {int(inventory)} units remaining, {int(days_remaining)} days left")
    print(f"{'='*70}\n")

    results = {}
    for name, policy_fn in policies.items():
        outcomes = [
            simulate_rest_of_season(env, policy_fn, inventory, days_remaining, seed=1000 + i)
            for i in range(num_simulations)
        ]

        avg_revenue = np.mean([o["projected_total_revenue"] for o in outcomes])
        avg_reward = np.mean([o["projected_total_reward"] for o in outcomes])
        avg_sold = np.mean([o["projected_units_sold"] for o in outcomes])
        avg_unsold = np.mean([o["projected_unsold_inventory"] for o in outcomes])
        immediate_price = outcomes[0]["immediate_price"]

        results[name] = {
            "immediate_price": immediate_price,
            "avg_projected_revenue": avg_revenue,
            "avg_projected_reward": avg_reward,
            "avg_units_sold": avg_sold,
            "avg_unsold_inventory": avg_unsold,
        }

        print(f"{name}:")
        print(f"  Recommended price right now: ${immediate_price:.2f}")
        print(f"  Projected total revenue (avg of {num_simulations} sims): ${avg_revenue:,.2f}")
        print(f"  Projected units sold: {avg_sold:.1f} | Unsold: {avg_unsold:.1f}")
        print()

    best_policy = max(results.items(), key=lambda item: item[1]["avg_projected_revenue"])
    print(f"Best projected outcome: {best_policy[0]} "
          f"(${best_policy[1]['avg_projected_revenue']:,.2f} projected revenue)")

    return results


def setup_policies(env, train_episodes=1000):
    """Train the learned agents and build all 5 policies."""
    print("Training Q-Learning agent...")
    ql_agent = QLearningAgent(num_actions=env.action_space.n)
    ql_agent.train(env, num_episodes=train_episodes, verbose=False)

    print("Training DQN agent...")
    dqn_agent = DQNAgent(state_dim=2, action_dim=env.action_space.n)
    dqn_agent.train(env, num_episodes=train_episodes, verbose=False)

    return {
        "DQN": make_agent_policy(dqn_agent),
        "Q-Learning": make_agent_policy(ql_agent),
        "Fixed Price Policy": make_fixed_price_policy(env),
        "Discount Policy": make_daily_discount_policy(env),
        "Random Policy": make_random_policy(env),
    }


if __name__ == "__main__":
    env = PricingEnvironment(PricingEnvConfig())
    policies = setup_policies(env)

    # --- Example scenarios (edit these to try your own "what-if" situations) ---
    example_scenarios = [
        {"inventory": 20, "days_remaining": 5},
        {"inventory": 80, "days_remaining": 25},
        {"inventory": 3, "days_remaining": 10},
    ]

    for scenario in example_scenarios:
        run_scenario(env, policies, scenario["inventory"], scenario["days_remaining"])

    print("\n\nTo try your own scenario, edit the 'example_scenarios' list at the "
          "bottom of this file, or call run_scenario(env, policies, inventory, days_remaining) directly.")