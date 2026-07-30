"""
pricing_visualizations.py

Generates visualizations showing how pricing decisions evolve over time
and comparing performance across all 5 pricing strategies, for Issue #103.

Charts generated (saved to evaluation/pricing_charts/):
  1. price_trajectory.png       - price chosen each day, per policy
  2. revenue_trend.png          - cumulative revenue over the season, per policy
  3. inventory_remaining.png    - inventory level over the season, per policy
  4. daily_price_changes.png    - day-to-day price change, per policy
  5. policy_performance_comparison.png - average reward/revenue across policies (multi-episode)

Charts 1-4 use ONE representative episode per policy, all run with the
SAME seed, so every policy faces identical simulated demand — making
their pricing behavior directly, fairly comparable on the same chart.

Chart 5 reuses the multi-episode evaluation approach from Issues #91/#99
for a robust, averaged performance comparison (not just one episode).
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'evaluation'))

from pricing_env.pricing_env import PricingEnvironment, PricingEnvConfig
from agents.q_learning_agent import QLearningAgent
from agents.dqn_agent import DQNAgent
from policy_evaluator import (
    PolicyEvaluator,
    make_fixed_price_policy,
    make_daily_discount_policy,
    make_random_policy,
    make_agent_policy,
)


TRAIN_EPISODES = 1000
COMPARISON_EVAL_EPISODES = 100
TRAJECTORY_SEED = 5000  # shared seed so all policies face identical demand

POLICY_COLORS = {
    "DQN": "#2f9e64",
    "Q-Learning": "#3b5bdb",
    "Fixed Price Policy": "#f08c00",
    "Discount Policy": "#e03131",
    "Random Policy": "#868e96",
}


def run_trajectory_episode(env, action_selector, seed):
    """
    Run one full episode, recording per-day data (not just totals) --
    needed for the trajectory-style charts.

    Returns
    -------
    trajectory : dict of lists
        days, prices, daily_revenue, cumulative_revenue, inventory_after
    """
    observation, info = env.reset(seed=seed)
    done = False

    days, prices, daily_revenue, cumulative_revenue, inventory_after = [], [], [], [], []
    running_revenue = 0.0
    day_counter = 0

    while not done:
        action = action_selector(observation)
        next_observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        day_counter += 1
        revenue_today = info["reward_breakdown"]["revenue"]
        running_revenue += revenue_today

        days.append(day_counter)
        prices.append(info["price"])
        daily_revenue.append(revenue_today)
        cumulative_revenue.append(running_revenue)
        inventory_after.append(float(next_observation[0]))

        observation = next_observation

    return {
        "days": days,
        "prices": prices,
        "daily_revenue": daily_revenue,
        "cumulative_revenue": cumulative_revenue,
        "inventory_after": inventory_after,
    }


def train_agents(env):
    """Train the Q-Learning and DQN agents used across all charts."""
    print("Training Q-Learning agent...")
    ql_agent = QLearningAgent(num_actions=env.action_space.n)
    ql_agent.train(env, num_episodes=TRAIN_EPISODES, verbose=False)

    print("Training DQN agent...")
    dqn_agent = DQNAgent(state_dim=2, action_dim=env.action_space.n)
    dqn_agent.train(env, num_episodes=TRAIN_EPISODES, verbose=False)

    return ql_agent, dqn_agent


def build_policies(env, ql_agent, dqn_agent):
    return {
        "DQN": make_agent_policy(dqn_agent),
        "Q-Learning": make_agent_policy(ql_agent),
        "Fixed Price Policy": make_fixed_price_policy(env),
        "Discount Policy": make_daily_discount_policy(env),
        "Random Policy": make_random_policy(env),
    }


def collect_trajectories(env, policies):
    """Run one shared-seed episode per policy, collecting full trajectories."""
    trajectories = {}
    for name, policy_fn in policies.items():
        trajectories[name] = run_trajectory_episode(env, policy_fn, seed=TRAJECTORY_SEED)
    return trajectories


# ------------------------------------------------------------
# CHART 1: PRICE TRAJECTORY
# ------------------------------------------------------------
def chart_price_trajectory(trajectories, filepath):
    plt.figure(figsize=(10, 6))
    for name, traj in trajectories.items():
        plt.plot(traj["days"], traj["prices"], label=name,
                  color=POLICY_COLORS[name], linewidth=2, marker="o", markersize=3)
    plt.title("Price Trajectory Over the Selling Season", fontsize=14, fontweight="bold")
    plt.xlabel("Day")
    plt.ylabel("Price ($)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close()
    print(f"Saved: {filepath}")


# ------------------------------------------------------------
# CHART 2: REVENUE TREND (cumulative)
# ------------------------------------------------------------
def chart_revenue_trend(trajectories, filepath):
    plt.figure(figsize=(10, 6))
    for name, traj in trajectories.items():
        plt.plot(traj["days"], traj["cumulative_revenue"], label=name,
                  color=POLICY_COLORS[name], linewidth=2)
    plt.title("Cumulative Revenue Over the Selling Season", fontsize=14, fontweight="bold")
    plt.xlabel("Day")
    plt.ylabel("Cumulative Revenue ($)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close()
    print(f"Saved: {filepath}")


# ------------------------------------------------------------
# CHART 3: INVENTORY REMAINING
# ------------------------------------------------------------
def chart_inventory_remaining(trajectories, filepath):
    plt.figure(figsize=(10, 6))
    for name, traj in trajectories.items():
        plt.plot(traj["days"], traj["inventory_after"], label=name,
                  color=POLICY_COLORS[name], linewidth=2)
    plt.title("Inventory Remaining Over the Selling Season", fontsize=14, fontweight="bold")
    plt.xlabel("Day")
    plt.ylabel("Inventory Remaining (units)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close()
    print(f"Saved: {filepath}")


# ------------------------------------------------------------
# CHART 4: DAILY PRICE CHANGES
# ------------------------------------------------------------
def chart_daily_price_changes(trajectories, filepath):
    plt.figure(figsize=(10, 6))
    for name, traj in trajectories.items():
        prices = np.array(traj["prices"])
        daily_change = np.diff(prices, prepend=prices[0])
        plt.plot(traj["days"], daily_change, label=name,
                  color=POLICY_COLORS[name], linewidth=1.5, alpha=0.85)
    plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
    plt.title("Daily Price Changes Over the Selling Season", fontsize=14, fontweight="bold")
    plt.xlabel("Day")
    plt.ylabel("Price Change vs Previous Day ($)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close()
    print(f"Saved: {filepath}")


# ------------------------------------------------------------
# CHART 5: POLICY PERFORMANCE COMPARISON (multi-episode average)
# ------------------------------------------------------------
def chart_policy_performance_comparison(env, policies, filepath):
    evaluator = PolicyEvaluator(env)

    names, avg_rewards, avg_revenues = [], [], []
    for name, policy_fn in policies.items():
        summary, _ = evaluator.evaluate_policy(
            policy_fn, num_episodes=COMPARISON_EVAL_EPISODES, base_seed=3000
        )
        names.append(name)
        avg_rewards.append(summary["avg_episode_reward"])
        avg_revenues.append(summary["avg_revenue"])

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 6))
    bars1 = ax.bar(x - width / 2, avg_rewards, width, label="Avg Reward",
                    color=[POLICY_COLORS[n] for n in names], alpha=0.9)
    bars2 = ax.bar(x + width / 2, avg_revenues, width, label="Avg Revenue",
                    color=[POLICY_COLORS[n] for n in names], alpha=0.5)

    ax.set_title(f"Policy Performance Comparison ({COMPARISON_EVAL_EPISODES}-episode average)",
                 fontsize=14, fontweight="bold")
    ax.set_ylabel("Value")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    ax.axhline(0, color="black", linewidth=0.8)

    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close()
    print(f"Saved: {filepath}")


if __name__ == "__main__":
    env = PricingEnvironment(PricingEnvConfig())

    ql_agent, dqn_agent = train_agents(env)
    policies = build_policies(env, ql_agent, dqn_agent)

    output_dir = "evaluation/pricing_charts"
    os.makedirs(output_dir, exist_ok=True)

    print("\nRunning shared-seed trajectory episodes for all policies...")
    trajectories = collect_trajectories(env, policies)

    print("\nGenerating charts...")
    chart_price_trajectory(trajectories, os.path.join(output_dir, "price_trajectory.png"))
    chart_revenue_trend(trajectories, os.path.join(output_dir, "revenue_trend.png"))
    chart_inventory_remaining(trajectories, os.path.join(output_dir, "inventory_remaining.png"))
    chart_daily_price_changes(trajectories, os.path.join(output_dir, "daily_price_changes.png"))
    chart_policy_performance_comparison(
        env, policies, os.path.join(output_dir, "policy_performance_comparison.png")
    )

    print("\nAll pricing visualizations generated successfully!")