"""
policy_evaluator.py

Multi-Policy Evaluation Engine.

Runs multiple pricing strategies (DQN, Q-Learning, and simple baselines)
through the SAME environment under IDENTICAL simulation conditions, so
their results can be fairly compared side by side.

"Identical conditions" means: for a given episode number, every policy
faces the exact same random demand sequence (arrivals, purchase
decisions) — achieved by resetting the environment with the same seed
for that episode number across all policies. Without this, differences
in results could just be due to one policy getting "luckier" demand,
rather than the policy actually being better.

Policies evaluated:
  - DQN Agent            (trained, greedy)
  - Q-Learning Agent      (trained, greedy, tuned defaults from Issue #59)
  - Fixed Price Policy    (always holds price at the starting base price)
  - Daily Discount Policy (applies the same fixed discount every day)
  - Random Policy         (picks a random price action every day)

Metrics collected per policy:
  - Revenue
  - Inventory Sold
  - Unsold Inventory
  - Average Selling Price
  - Episode Reward
"""

import sys
import os
import csv
import random
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from pricing_env.pricing_env import PricingEnvironment, PricingEnvConfig
from agents.q_learning_agent import QLearningAgent
from agents.dqn_agent import DQNAgent


class PolicyEvaluator:
    """
    Runs a given pricing policy (any function that maps an observation to
    an action) through the environment and collects standardized metrics.
    """

    def __init__(self, env):
        self.env = env

    def run_episode(self, action_selector, seed=None):
        """
        Run one full episode using the given policy function.

        Parameters
        ----------
        action_selector : callable
            A function: observation -> action (int).
        seed : int, optional
            Seed passed to env.reset() — using the same seed for
            different policies makes their results directly comparable
            (same random demand sequence for all of them).

        Returns
        -------
        metrics : dict
            episode_reward, revenue, inventory_sold, unsold_inventory,
            avg_selling_price for this one episode.
        """
        observation, info = self.env.reset(seed=seed)
        done = False

        total_reward = 0.0
        total_revenue = 0.0
        total_units_sold = 0
        prices_charged = []

        while not done:
            action = action_selector(observation)
            next_observation, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            total_reward += reward
            total_revenue += info["reward_breakdown"]["revenue"]
            total_units_sold += info["units_sold"]
            prices_charged.append(info["price"])

            observation = next_observation

        unsold_inventory = float(observation[0])  # remaining_inventory at episode end
        avg_selling_price = float(np.mean(prices_charged)) if prices_charged else 0.0

        return {
            "episode_reward": total_reward,
            "revenue": total_revenue,
            "inventory_sold": total_units_sold,
            "unsold_inventory": unsold_inventory,
            "avg_selling_price": avg_selling_price,
        }

    def evaluate_policy(self, action_selector, num_episodes=100, base_seed=1000):
        """
        Run a policy across many episodes and return averaged metrics.

        Uses the same sequence of seeds (base_seed, base_seed+1, ...) for
        every policy evaluated with this evaluator instance — this is
        what guarantees "identical simulation conditions" across policies
        being compared, as long as evaluate_policy() is called with the
        same base_seed for each one.
        """
        episode_records = [
            self.run_episode(action_selector, seed=base_seed + i)
            for i in range(num_episodes)
        ]

        summary = {
            "avg_episode_reward": float(np.mean([r["episode_reward"] for r in episode_records])),
            "avg_revenue": float(np.mean([r["revenue"] for r in episode_records])),
            "avg_inventory_sold": float(np.mean([r["inventory_sold"] for r in episode_records])),
            "avg_unsold_inventory": float(np.mean([r["unsold_inventory"] for r in episode_records])),
            "avg_selling_price": float(np.mean([r["avg_selling_price"] for r in episode_records])),
        }
        return summary, episode_records


# ------------------------------------------------------------
# POLICY DEFINITIONS
# ------------------------------------------------------------
def make_fixed_price_policy(env):
    """Always choose the action closest to 0% adjustment (holds price steady)."""
    pct_list = list(env.config.price_adjustment_pct)
    hold_index = int(np.argmin(np.abs(np.array(pct_list))))
    return lambda observation: hold_index


def make_daily_discount_policy(env):
    """Always apply the same, largest available discount every single day."""
    pct_list = list(env.config.price_adjustment_pct)
    deepest_discount_index = int(np.argmin(pct_list))  # most negative = deepest discount
    return lambda observation: deepest_discount_index


def make_random_policy(env):
    """Pick a completely random price action every day."""
    num_actions = env.action_space.n
    return lambda observation: random.randint(0, num_actions - 1)


def make_agent_policy(agent):
    """Wrap a trained agent (QLearningAgent or DQNAgent) as a policy function."""
    return lambda observation: agent.choose_action(observation, greedy=True)


def export_to_csv(results, filepath):
    """Export the policy comparison summary table to a CSV file."""
    fieldnames = [
        "policy", "avg_episode_reward", "avg_revenue",
        "avg_inventory_sold", "avg_unsold_inventory", "avg_selling_price",
    ]

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for policy_name, summary in results.items():
            row = {"policy": policy_name, **summary}
            writer.writerow(row)

    print(f"Results exported to {filepath}")


if __name__ == "__main__":
    NUM_EVAL_EPISODES = 100
    TRAIN_EPISODES = 1000

    env = PricingEnvironment(PricingEnvConfig())
    evaluator = PolicyEvaluator(env)

    # --- Train the learned agents first ---
    print("Training Q-Learning agent...")
    ql_agent = QLearningAgent(num_actions=env.action_space.n)  # tuned defaults (Issue #59)
    ql_agent.train(env, num_episodes=TRAIN_EPISODES, verbose=False)

    print("Training DQN agent...")
    dqn_agent = DQNAgent(state_dim=2, action_dim=env.action_space.n)
    dqn_agent.train(env, num_episodes=TRAIN_EPISODES, verbose=False)

    # --- Build all 5 policies ---
    policies = {
        "DQN Agent": make_agent_policy(dqn_agent),
        "Q-Learning Agent": make_agent_policy(ql_agent),
        "Fixed Price Policy": make_fixed_price_policy(env),
        "Daily Discount Policy": make_daily_discount_policy(env),
        "Random Policy": make_random_policy(env),
    }

    # --- Evaluate all policies under IDENTICAL conditions ---
    # Same base_seed for every policy -> each policy faces the exact same
    # sequence of random demand realizations, episode by episode.
    results = {}
    for name, policy_fn in policies.items():
        print(f"\nEvaluating: {name}...")
        summary, _ = evaluator.evaluate_policy(
            policy_fn, num_episodes=NUM_EVAL_EPISODES, base_seed=1000
        )
        results[name] = summary
        print(f"  Avg Reward: {summary['avg_episode_reward']:.2f} | "
              f"Avg Revenue: {summary['avg_revenue']:.2f} | "
              f"Unsold Inventory: {summary['avg_unsold_inventory']:.2f}")

    # --- Export ---
    os.makedirs("evaluation", exist_ok=True)
    export_to_csv(results, "evaluation/policy_evaluation_results.csv")

    print("\nAll pricing strategies evaluated successfully!")