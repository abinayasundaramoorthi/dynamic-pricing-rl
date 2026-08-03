"""
policy_comparison.py

Policy Performance Comparison Engine — builds on the Multi-Policy
Evaluation Engine (policy_evaluator.py, Issue #91) by adding ranking and
difference-from-baseline calculations, so the 5 pricing strategies can be
directly compared and ranked by business performance.

Baseline for "difference" calculations: Fixed Price Policy — the
simplest, non-learned strategy, representing what a business might do
with no dynamic pricing system at all. Every other policy's performance
is measured as an improvement (or decline) relative to this baseline.

Outputs:
  - evaluation/policy_ranking.csv (ranked comparison table)
"""

import sys
import os
import csv
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from pricing_env.pricing_env import PricingEnvironment, PricingEnvConfig
from agents.q_learning_agent import QLearningAgent
from agents.dqn_agent import DQNAgent
from evaluation.policy_evaluator import (
    PolicyEvaluator,
    make_fixed_price_policy,
    make_daily_discount_policy,
    make_random_policy,
    make_agent_policy,
)


TRAIN_EPISODES = 1000
EVAL_EPISODES = 100
BASELINE_POLICY_NAME = "Fixed Price Policy"


def run_all_policies(env):
    """Train the learned agents and evaluate all 5 policies under identical conditions."""
    evaluator = PolicyEvaluator(env)

    print("Training Q-Learning agent...")
    ql_agent = QLearningAgent(num_actions=env.action_space.n)
    ql_agent.train(env, num_episodes=TRAIN_EPISODES, verbose=False)

    print("Training DQN agent...")
    dqn_agent = DQNAgent(state_dim=2, action_dim=env.action_space.n)
    dqn_agent.train(env, num_episodes=TRAIN_EPISODES, verbose=False)

    policies = {
        "DQN": make_agent_policy(dqn_agent),
        "Q-Learning": make_agent_policy(ql_agent),
        "Fixed Price Policy": make_fixed_price_policy(env),
        "Discount Policy": make_daily_discount_policy(env),
        "Random Policy": make_random_policy(env),
    }

    results = {}
    for name, policy_fn in policies.items():
        print(f"Evaluating: {name}...")
        summary, _ = evaluator.evaluate_policy(
            policy_fn, num_episodes=EVAL_EPISODES, base_seed=2000
        )
        results[name] = summary

    return results


def build_ranking_table(results, initial_inventory):
    """
    Compute differences from baseline, sell-through rate, and rank every
    policy by average reward (highest = best = rank 1).
    """
    baseline = results[BASELINE_POLICY_NAME]

    table = []
    for policy_name, summary in results.items():
        revenue_diff = summary["avg_revenue"] - baseline["avg_revenue"]
        reward_diff = summary["avg_episode_reward"] - baseline["avg_episode_reward"]
        sell_through_rate = (summary["avg_inventory_sold"] / initial_inventory) * 100

        table.append({
            "policy": policy_name,
            "avg_reward": summary["avg_episode_reward"],
            "reward_difference": reward_diff,
            "avg_revenue": summary["avg_revenue"],
            "revenue_difference": revenue_diff,
            "inventory_sold": summary["avg_inventory_sold"],
            "sell_through_rate": sell_through_rate,
        })

    # Rank by average reward, highest first
    table.sort(key=lambda row: row["avg_reward"], reverse=True)
    for rank, row in enumerate(table, start=1):
        row["rank"] = rank

    return table


def export_ranking_csv(table, filepath):
    """Export the ranked comparison table to CSV."""
    fieldnames = [
        "rank", "policy", "avg_reward", "reward_difference",
        "avg_revenue", "revenue_difference", "inventory_sold", "sell_through_rate",
    ]

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in table:
            writer.writerow(row)

    print(f"\nPolicy ranking exported to {filepath}")


def print_comparison_table(table):
    """Print a readable comparison table to the console."""
    print(f"\n{'Rank':<5}{'Policy':<22}{'Avg Reward':>12}{'Reward Diff':>14}"
          f"{'Avg Revenue':>13}{'Revenue Diff':>14}{'Sell-through':>14}")
    print("-" * 96)
    for row in table:
        print(f"{row['rank']:<5}{row['policy']:<22}{row['avg_reward']:>12.2f}"
              f"{row['reward_difference']:>14.2f}{row['avg_revenue']:>13.2f}"
              f"{row['revenue_difference']:>14.2f}{row['sell_through_rate']:>13.1f}%")


if __name__ == "__main__":
    env = PricingEnvironment(PricingEnvConfig())

    print("Running all 5 policies under identical simulation conditions...\n")
    results = run_all_policies(env)

    print("\nBuilding ranking table...")
    ranking_table = build_ranking_table(results, env.config.initial_inventory)

    print_comparison_table(ranking_table)

    os.makedirs("evaluation", exist_ok=True)
    export_ranking_csv(ranking_table, "evaluation/policy_ranking.csv")

    winner = ranking_table[0]
    print(f"\nTop-ranked policy: {winner['policy']} "
          f"(Avg Reward: {winner['avg_reward']:.2f}, "
          f"Revenue improvement over {BASELINE_POLICY_NAME}: "
          f"{winner['revenue_difference']:+.2f})")

    print("\nAll pricing strategies compared and ranked successfully!")