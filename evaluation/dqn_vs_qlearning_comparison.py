"""
dqn_vs_qlearning_comparison.py

Trains both a Q-Learning agent (tuned defaults from Issue #59) and a DQN
agent (Issues #63, #71, #75), evaluates both on identical metrics, and
documents the comparison for Issue #80.

Metrics compared:
  - Average Reward
  - Total Revenue
  - Inventory Utilization
  - Training Stability (reward variance in late training)
  - Episode Performance (early vs late training reward, best episode)

Outputs:
  - evaluation/dqn_vs_qlearning.md   (written summary + conclusions)
  - evaluation/comparison_metrics.csv (raw metrics, for further analysis)
"""

import sys
import os
import csv
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from pricing_env.pricing_env import PricingEnvironment, PricingEnvConfig
from agents.q_learning_agent import QLearningAgent
from agents.dqn_agent import DQNAgent


TRAIN_EPISODES = 1000
EVAL_EPISODES = 200


def run_qlearning(env):
    """Train and evaluate the Q-Learning agent using its tuned defaults (Issue #59)."""
    agent = QLearningAgent(num_actions=env.action_space.n)  # uses tuned defaults
    agent.train(env, num_episodes=TRAIN_EPISODES, verbose=False)
    eval_metrics = agent.evaluate(env, num_episodes=EVAL_EPISODES)

    rewards = np.array(agent.episode_rewards)
    return {
        "agent": "Q-Learning",
        "avg_reward": eval_metrics["avg_reward"],
        "avg_revenue": eval_metrics["avg_revenue"],
        "avg_inventory_utilization": eval_metrics["avg_inventory_utilization"],
        "early_train_avg": float(np.mean(rewards[:100])),
        "late_train_avg": float(np.mean(rewards[-100:])),
        "reward_std_late": float(np.std(rewards[-100:])),
        "best_episode_reward": float(np.max(rewards)),
        "states_or_params_learned": len(agent.q_table),
    }


def run_dqn(env):
    """Train and evaluate the DQN agent."""
    agent = DQNAgent(state_dim=2, action_dim=env.action_space.n)
    agent.train(env, num_episodes=TRAIN_EPISODES, verbose=False)
    eval_metrics = agent.evaluate(env, num_episodes=EVAL_EPISODES)

    rewards = np.array(agent.episode_rewards)
    total_params = sum(p.numel() for p in agent.q_network.parameters())
    return {
        "agent": "DQN",
        "avg_reward": eval_metrics["avg_reward"],
        "avg_revenue": eval_metrics["avg_revenue"],
        "avg_inventory_utilization": eval_metrics["avg_inventory_utilization"],
        "early_train_avg": float(np.mean(rewards[:100])),
        "late_train_avg": float(np.mean(rewards[-100:])),
        "reward_std_late": float(np.std(rewards[-100:])),
        "best_episode_reward": float(np.max(rewards)),
        "states_or_params_learned": total_params,
    }


def write_csv(qlearning_results, dqn_results, filepath):
    fieldnames = [
        "metric", "Q-Learning", "DQN"
    ]
    rows = [
        ("Average Reward", qlearning_results["avg_reward"], dqn_results["avg_reward"]),
        ("Average Revenue", qlearning_results["avg_revenue"], dqn_results["avg_revenue"]),
        ("Inventory Utilization (%)",
         qlearning_results["avg_inventory_utilization"] * 100,
         dqn_results["avg_inventory_utilization"] * 100),
        ("Early Training Avg Reward (first 100 ep)",
         qlearning_results["early_train_avg"], dqn_results["early_train_avg"]),
        ("Late Training Avg Reward (last 100 ep)",
         qlearning_results["late_train_avg"], dqn_results["late_train_avg"]),
        ("Reward Std Dev (last 100 ep, stability)",
         qlearning_results["reward_std_late"], dqn_results["reward_std_late"]),
        ("Best Episode Reward",
         qlearning_results["best_episode_reward"], dqn_results["best_episode_reward"]),
        ("States Learned (Q-table) / Parameters (DQN)",
         qlearning_results["states_or_params_learned"], dqn_results["states_or_params_learned"]),
    ]

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)
        for row in rows:
            writer.writerow(row)

    print(f"Metrics exported to {filepath}")


def generate_report(qlearning_results, dqn_results, filepath):
    reward_diff_pct = (
        (dqn_results["avg_reward"] - qlearning_results["avg_reward"])
        / abs(qlearning_results["avg_reward"]) * 100
        if qlearning_results["avg_reward"] != 0 else 0.0
    )
    revenue_diff_pct = (
        (dqn_results["avg_revenue"] - qlearning_results["avg_revenue"])
        / abs(qlearning_results["avg_revenue"]) * 100
        if qlearning_results["avg_revenue"] != 0 else 0.0
    )

    winner = "DQN" if dqn_results["avg_reward"] > qlearning_results["avg_reward"] else "Q-Learning"
    more_stable = (
        "DQN" if dqn_results["reward_std_late"] < qlearning_results["reward_std_late"]
        else "Q-Learning"
    )

    lines = []
    lines.append("# DQN vs Q-Learning Performance Comparison — Issue #80\n")
    lines.append(f"**Training episodes:** {TRAIN_EPISODES} (each agent)")
    lines.append(f"**Evaluation episodes:** {EVAL_EPISODES} (greedy, no exploration)\n")

    lines.append("## Agents Compared\n")
    lines.append(
        "**Q-Learning:** Tabular agent using a Q-table, with tuned "
        "hyperparameters from Issue #59 (learning rate=0.1, discount "
        "factor=0.95, epsilon decay=0.999).\n"
    )
    lines.append(
        "**DQN:** Neural network-based agent (Issue #63 architecture), "
        "trained using an experience replay buffer (Issue #71) and a "
        "target network for stability (Issue #75). Same core "
        "hyperparameters (discount factor=0.95, similar epsilon schedule) "
        "for a fair comparison, plus DQN-specific settings: learning "
        "rate=0.001, batch size=64, target network update every 500 steps.\n"
    )

    lines.append("## Performance Comparison\n")
    lines.append("| Metric | Q-Learning | DQN | Difference (DQN vs Q-Learning) |")
    lines.append("|---|---|---|---|")
    lines.append(
        f"| Average Reward | {qlearning_results['avg_reward']:.2f} | "
        f"{dqn_results['avg_reward']:.2f} | {reward_diff_pct:+.1f}% |"
    )
    lines.append(
        f"| Average Revenue | {qlearning_results['avg_revenue']:.2f} | "
        f"{dqn_results['avg_revenue']:.2f} | {revenue_diff_pct:+.1f}% |"
    )
    lines.append(
        f"| Inventory Utilization | "
        f"{qlearning_results['avg_inventory_utilization']*100:.1f}% | "
        f"{dqn_results['avg_inventory_utilization']*100:.1f}% | "
        f"{(dqn_results['avg_inventory_utilization']-qlearning_results['avg_inventory_utilization'])*100:+.1f}pp |"
    )
    lines.append("")

    lines.append("## Training Stability\n")
    lines.append(
        "Measured as the standard deviation of reward across the last "
        "100 training episodes — lower means more consistent, converged "
        "performance.\n"
    )
    lines.append("| Metric | Q-Learning | DQN |")
    lines.append("|---|---|---|")
    lines.append(
        f"| Reward Std Dev (last 100 episodes) | "
        f"{qlearning_results['reward_std_late']:.2f} | "
        f"{dqn_results['reward_std_late']:.2f} |"
    )
    lines.append(f"\n**More stable agent: {more_stable}**\n")

    lines.append("## Episode Performance\n")
    lines.append("| Metric | Q-Learning | DQN |")
    lines.append("|---|---|---|")
    lines.append(
        f"| Early Training Avg (first 100 ep) | "
        f"{qlearning_results['early_train_avg']:.2f} | "
        f"{dqn_results['early_train_avg']:.2f} |"
    )
    lines.append(
        f"| Late Training Avg (last 100 ep) | "
        f"{qlearning_results['late_train_avg']:.2f} | "
        f"{dqn_results['late_train_avg']:.2f} |"
    )
    lines.append(
        f"| Best Single Episode Reward | "
        f"{qlearning_results['best_episode_reward']:.2f} | "
        f"{dqn_results['best_episode_reward']:.2f} |"
    )
    lines.append(
        f"| States Learned (Q-table) / Network Parameters (DQN) | "
        f"{qlearning_results['states_or_params_learned']} | "
        f"{dqn_results['states_or_params_learned']} |"
    )
    lines.append("")

    lines.append("## Observations\n")
    lines.append(
        f"- **{winner}** achieved the higher average reward at evaluation "
        f"time ({reward_diff_pct:+.1f}% difference).\n"
        f"- Q-Learning's Q-table grew to "
        f"{qlearning_results['states_or_params_learned']} learned states, "
        f"each stored and updated independently. DQN instead uses a fixed "
        f"{dqn_results['states_or_params_learned']}-parameter neural "
        f"network, which can in principle generalize to states never "
        f"seen exactly during training — a key theoretical advantage of "
        f"DQN for larger or continuous state spaces, though this "
        f"project's state space (inventory × days remaining) is small "
        f"enough that Q-Learning can already represent it fairly "
        f"completely.\n"
        f"- {more_stable} showed lower reward variance in late training, "
        f"suggesting more consistent, converged behavior by the end of "
        f"training.\n"
    )

    lines.append("## Conclusion\n")
    lines.append(
        f"For this project's current state space size (2 dimensions, "
        f"bounded ranges), **{winner}** performed best on average reward "
        f"in this evaluation. Given the state space is small enough for "
        f"a Q-table to represent completely, Q-Learning's simpler, more "
        f"directly interpretable approach remains a strong and "
        f"computationally cheaper baseline. DQN's architecture is "
        f"nonetheless valuable groundwork: if the environment is later "
        f"extended with additional continuous features (e.g. real "
        f"competitor pricing, seasonal demand signals), a tabular "
        f"Q-table would no longer be feasible, and the DQN implementation "
        f"built here would become the necessary approach.\n"
    )

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Report written to {filepath}")


if __name__ == "__main__":
    env = PricingEnvironment(PricingEnvConfig())

    print("Training and evaluating Q-Learning agent...")
    qlearning_results = run_qlearning(env)
    print(f"  Avg Reward: {qlearning_results['avg_reward']:.2f} | "
          f"Avg Revenue: {qlearning_results['avg_revenue']:.2f}")

    print("\nTraining and evaluating DQN agent...")
    dqn_results = run_dqn(env)
    print(f"  Avg Reward: {dqn_results['avg_reward']:.2f} | "
          f"Avg Revenue: {dqn_results['avg_revenue']:.2f}")

    os.makedirs("evaluation", exist_ok=True)
    write_csv(qlearning_results, dqn_results, "evaluation/comparison_metrics.csv")
    generate_report(qlearning_results, dqn_results, "evaluation/dqn_vs_qlearning.md")

    print("\nDQN vs Q-Learning comparison complete!")
