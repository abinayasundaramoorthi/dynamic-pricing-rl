"""
hyperparameter_search.py

Tests multiple combinations of Q-Learning hyperparameters and records
which combination performs best, for Issue #47.

Parameters tested:
  - Learning rate (alpha)
  - Discount factor (gamma)
  - Epsilon decay rate
  - Initial epsilon

For each combination:
  1. Train a fresh agent for a fixed number of episodes.
  2. Evaluate the trained agent with NO exploration (pure greedy actions)
     over a separate set of episodes, measuring:
       - Average reward
       - Average revenue
       - Average inventory utilization

Results are written to evaluation/hyperparameter_results.md
"""

import sys
import os
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from pricing_env.pricing_env import PricingEnvironment, PricingEnvConfig
from agents.q_learning_agent import QLearningAgent


# ------------------------------------------------------------
# PARAMETER COMBINATIONS TO TEST
# ------------------------------------------------------------
# Each combination varies at least one of: alpha, gamma, epsilon_decay,
# initial epsilon — covering a "slow/careful" learner, a "fast/aggressive"
# learner, and a couple of middle-ground options.
PARAM_COMBINATIONS = [
    {
        "name": "Baseline",
        "learning_rate": 0.1,
        "discount_factor": 0.95,
        "epsilon": 1.0,
        "epsilon_decay": 0.995,
    },
    {
        "name": "Fast Learning Rate",
        "learning_rate": 0.3,
        "discount_factor": 0.95,
        "epsilon": 1.0,
        "epsilon_decay": 0.995,
    },
    {
        "name": "Low Discount Factor (short-term focus)",
        "learning_rate": 0.1,
        "discount_factor": 0.80,
        "epsilon": 1.0,
        "epsilon_decay": 0.995,
    },
    {
        "name": "Slow Epsilon Decay (more exploration)",
        "learning_rate": 0.1,
        "discount_factor": 0.95,
        "epsilon": 1.0,
        "epsilon_decay": 0.999,
    },
    {
        "name": "Fast Epsilon Decay + Lower Start",
        "learning_rate": 0.1,
        "discount_factor": 0.95,
        "epsilon": 0.5,
        "epsilon_decay": 0.98,
    },
]

TRAIN_EPISODES = 2000
EVAL_EPISODES = 200


def run_experiment(params, env_config):
    """Train and evaluate one hyperparameter combination."""
    env = PricingEnvironment(env_config)

    agent = QLearningAgent(
        num_actions=env.action_space.n,
        learning_rate=params["learning_rate"],
        discount_factor=params["discount_factor"],
        epsilon=params["epsilon"],
        epsilon_decay=params["epsilon_decay"],
    )

    start_time = time.time()
    agent.train(env, num_episodes=TRAIN_EPISODES, verbose=False)
    train_time = time.time() - start_time

    eval_metrics = agent.evaluate(env, num_episodes=EVAL_EPISODES)

    result = {
        "name": params["name"],
        "learning_rate": params["learning_rate"],
        "discount_factor": params["discount_factor"],
        "epsilon": params["epsilon"],
        "epsilon_decay": params["epsilon_decay"],
        "avg_reward": eval_metrics["avg_reward"],
        "avg_revenue": eval_metrics["avg_revenue"],
        "avg_inventory_utilization": eval_metrics["avg_inventory_utilization"],
        "states_learned": len(agent.q_table),
        "train_time_seconds": train_time,
    }
    return result


def generate_report(results, filepath):
    """Write all results into a clean markdown report."""

    # Sort by average reward, best first
    sorted_results = sorted(results, key=lambda r: r["avg_reward"], reverse=True)
    best = sorted_results[0]

    lines = []
    lines.append("# Q-Learning Hyperparameter Optimization Results — Issue #47\n")
    lines.append(f"**Training episodes per configuration:** {TRAIN_EPISODES}")
    lines.append(f"**Evaluation episodes per configuration:** {EVAL_EPISODES} (greedy, no exploration)\n")

    lines.append("## Parameter Combinations Tested\n")
    lines.append("| Configuration | Learning Rate (α) | Discount Factor (γ) | Initial ε | ε Decay |")
    lines.append("|---|---|---|---|---|")
    for r in results:
        lines.append(
            f"| {r['name']} | {r['learning_rate']} | {r['discount_factor']} | "
            f"{r['epsilon']} | {r['epsilon_decay']} |"
        )
    lines.append("")

    lines.append("## Performance Comparison\n")
    lines.append("| Configuration | Avg Reward | Avg Revenue | Inventory Utilization | States Learned | Train Time (s) |")
    lines.append("|---|---|---|---|---|---|")
    for r in sorted_results:
        lines.append(
            f"| {r['name']} | {r['avg_reward']:.2f} | {r['avg_revenue']:.2f} | "
            f"{r['avg_inventory_utilization']*100:.1f}% | {r['states_learned']} | "
            f"{r['train_time_seconds']:.1f} |"
        )
    lines.append("")

    lines.append("## Best Configuration\n")
    lines.append(f"**Winner: {best['name']}**\n")
    lines.append(f"- Learning rate (α): {best['learning_rate']}")
    lines.append(f"- Discount factor (γ): {best['discount_factor']}")
    lines.append(f"- Initial epsilon: {best['epsilon']}")
    lines.append(f"- Epsilon decay: {best['epsilon_decay']}")
    lines.append(f"- Average reward achieved: {best['avg_reward']:.2f}")
    lines.append(f"- Average revenue achieved: {best['avg_revenue']:.2f}")
    lines.append(f"- Inventory utilization: {best['avg_inventory_utilization']*100:.1f}%\n")

    lines.append("## Notes on Interpretation\n")
    lines.append(
        "- **Avg Reward** reflects the full reward signal (revenue minus "
        "discount/unsold penalties, plus pacing bonus) — this is what the "
        "agent actually optimizes for.\n"
        "- **Avg Revenue** is the raw price × units sold, useful for "
        "understanding real business impact separately from the shaped "
        "reward signal.\n"
        "- **Inventory Utilization** shows what fraction of starting stock "
        "was sold by the end of the season — very low utilization suggests "
        "the agent is pricing too high and leaving stock unsold; very high "
        "utilization achieved via heavy discounting may hurt revenue, which "
        "the Avg Revenue and discount penalty in Avg Reward would reveal.\n"
        "- Evaluation used greedy (no-exploration) actions so results "
        "reflect the agent's actual learned policy, not random exploration "
        "noise.\n"
    )

    lines.append("## Recommendation\n")
    lines.append(
        f"Going forward, the Q-Learning agent should default to the "
        f"**{best['name']}** configuration "
        f"(α={best['learning_rate']}, γ={best['discount_factor']}, "
        f"ε_start={best['epsilon']}, ε_decay={best['epsilon_decay']}) "
        f"for baseline training runs, based on the highest average reward "
        f"achieved across {EVAL_EPISODES} evaluation episodes.\n"
    )

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Report written to {filepath}")


if __name__ == "__main__":
    env_config = PricingEnvConfig()

    all_results = []
    for params in PARAM_COMBINATIONS:
        print(f"\nRunning experiment: {params['name']}...")
        result = run_experiment(params, env_config)
        all_results.append(result)
        print(f"  Avg Reward: {result['avg_reward']:.2f} | "
              f"Avg Revenue: {result['avg_revenue']:.2f} | "
              f"Utilization: {result['avg_inventory_utilization']*100:.1f}%")

    os.makedirs("evaluation", exist_ok=True)
    generate_report(all_results, "evaluation/hyperparameter_results.md")

    print("\nAll experiments complete!")