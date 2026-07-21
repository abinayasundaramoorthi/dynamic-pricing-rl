"""
optimize_and_document.py

Compares the agent's performance BEFORE hyperparameter tuning (original
defaults) against AFTER tuning (the optimized defaults now built into
q_learning_agent.py), for Issue #59.

Also checks stability of learning: whether reward variance in the late
stage of training is low (a sign the agent has converged to a consistent
policy rather than still fluctuating wildly).

Results are written to reports/q_learning_optimization.md
"""

import sys
import os
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from pricing_env.pricing_env import PricingEnvironment, PricingEnvConfig
from agents.q_learning_agent import QLearningAgent


TRAIN_EPISODES = 2000
EVAL_EPISODES = 200

# BEFORE tuning: the project's original starting defaults
BEFORE_PARAMS = {
    "learning_rate": 0.1,
    "discount_factor": 0.95,
    "epsilon": 1.0,
    "epsilon_decay": 0.995,
    "epsilon_min": 0.01,
    "exploration_strategy": "epsilon_greedy",
    "warmup_episodes": 0,
}

# AFTER tuning: matches the new defaults now built into QLearningAgent
AFTER_PARAMS = {
    "learning_rate": 0.1,
    "discount_factor": 0.95,
    "epsilon": 1.0,
    "epsilon_decay": 0.999,
    "epsilon_min": 0.01,
    "exploration_strategy": "epsilon_greedy",
    "warmup_episodes": 0,
}


def train_and_measure(params, env, num_episodes):
    """Train an agent with given params and return training + evaluation results."""
    agent = QLearningAgent(num_actions=env.action_space.n, **params)
    agent.train(env, num_episodes=num_episodes, verbose=False)

    eval_metrics = agent.evaluate(env, num_episodes=EVAL_EPISODES)

    rewards = np.array(agent.episode_rewards)
    late_rewards = rewards[-200:]
    stability_std = float(np.std(late_rewards))
    stability_cv = (
        stability_std / abs(np.mean(late_rewards)) * 100
        if np.mean(late_rewards) != 0 else 0.0
    )  # coefficient of variation, as a %, easier to interpret than raw std

    return {
        "avg_reward": eval_metrics["avg_reward"],
        "avg_revenue": eval_metrics["avg_revenue"],
        "avg_inventory_utilization": eval_metrics["avg_inventory_utilization"],
        "early_train_avg": float(np.mean(rewards[:100])),
        "late_train_avg": float(np.mean(rewards[-100:])),
        "stability_std": stability_std,
        "stability_cv_pct": stability_cv,
        "states_learned": len(agent.q_table),
    }


def generate_report(before, after, filepath):
    reward_improvement_pct = (
        (after["avg_reward"] - before["avg_reward"]) / abs(before["avg_reward"]) * 100
        if before["avg_reward"] != 0 else 0.0
    )
    revenue_improvement_pct = (
        (after["avg_revenue"] - before["avg_revenue"]) / abs(before["avg_revenue"]) * 100
        if before["avg_revenue"] != 0 else 0.0
    )

    lines = []
    lines.append("# Q-Learning Baseline Optimization — Issue #59\n")
    lines.append(f"**Training episodes:** {TRAIN_EPISODES}")
    lines.append(f"**Evaluation episodes:** {EVAL_EPISODES} (greedy, no exploration)\n")

    lines.append("## Hyperparameters Tuned\n")
    lines.append("| Parameter | Before (original default) | After (tuned) |")
    lines.append("|---|---|---|")
    lines.append(f"| Learning Rate (α) | {BEFORE_PARAMS['learning_rate']} | {AFTER_PARAMS['learning_rate']} |")
    lines.append(f"| Discount Factor (γ) | {BEFORE_PARAMS['discount_factor']} | {AFTER_PARAMS['discount_factor']} |")
    lines.append(f"| Initial Epsilon | {BEFORE_PARAMS['epsilon']} | {AFTER_PARAMS['epsilon']} |")
    lines.append(f"| Epsilon Decay | {BEFORE_PARAMS['epsilon_decay']} | {AFTER_PARAMS['epsilon_decay']} |")
    lines.append(f"| Epsilon Min | {BEFORE_PARAMS['epsilon_min']} | {AFTER_PARAMS['epsilon_min']} |")
    lines.append("")
    lines.append(
        "**Selected change:** slower epsilon decay (0.995 → 0.999). "
        "Learning rate and discount factor were tested at other values in "
        "Issue #47's broader search (e.g. α=0.3, γ=0.80) but the original "
        "values (α=0.1, γ=0.95) already performed best, so they are kept "
        "unchanged here — only epsilon decay is adjusted, since it showed "
        "the clearest, most consistent improvement across testing.\n"
    )

    lines.append("## Performance Comparison\n")
    lines.append("| Metric | Before Tuning | After Tuning | Improvement |")
    lines.append("|---|---|---|---|")
    lines.append(
        f"| Avg Reward | {before['avg_reward']:.2f} | {after['avg_reward']:.2f} | "
        f"{reward_improvement_pct:+.1f}% |"
    )
    lines.append(
        f"| Avg Revenue | {before['avg_revenue']:.2f} | {after['avg_revenue']:.2f} | "
        f"{revenue_improvement_pct:+.1f}% |"
    )
    lines.append(
        f"| Inventory Utilization | {before['avg_inventory_utilization']*100:.1f}% | "
        f"{after['avg_inventory_utilization']*100:.1f}% | "
        f"{(after['avg_inventory_utilization']-before['avg_inventory_utilization'])*100:+.1f}pp |"
    )
    lines.append(
        f"| States Learned | {before['states_learned']} | {after['states_learned']} | "
        f"{after['states_learned']-before['states_learned']:+d} |"
    )
    lines.append("")

    lines.append("## Learning Stability\n")
    lines.append(
        "Stability is measured using the coefficient of variation (CV) of "
        "reward across the last 200 training episodes — the standard "
        "deviation as a percentage of the mean. A **lower CV means more "
        "stable, consistent performance** (the agent has converged), while "
        "a high CV means the agent's results are still swinging widely "
        "from episode to episode.\n"
    )
    lines.append("| Metric | Before Tuning | After Tuning |")
    lines.append("|---|---|---|")
    lines.append(
        f"| Early training avg reward (first 100 episodes) | "
        f"{before['early_train_avg']:.2f} | {after['early_train_avg']:.2f} |"
    )
    lines.append(
        f"| Late training avg reward (last 100 episodes) | "
        f"{before['late_train_avg']:.2f} | {after['late_train_avg']:.2f} |"
    )
    lines.append(
        f"| Reward Std Dev (last 200 episodes) | "
        f"{before['stability_std']:.2f} | {after['stability_std']:.2f} |"
    )
    lines.append(
        f"| Coefficient of Variation | "
        f"{before['stability_cv_pct']:.1f}% | {after['stability_cv_pct']:.1f}% |"
    )
    lines.append("")

    stability_verdict = (
        "more stable" if after["stability_cv_pct"] < before["stability_cv_pct"]
        else "less stable"
    )
    lines.append(
        f"The tuned configuration shows **{stability_verdict}** learning "
        f"behavior compared to the original defaults, based on reward "
        f"variance in the late training stage.\n"
    )

    lines.append("## Conclusion\n")
    if reward_improvement_pct > 0:
        lines.append(
            f"Tuning the epsilon decay rate from 0.995 to 0.999 improved "
            f"average reward by **{reward_improvement_pct:.1f}%** "
            f"(revenue by {revenue_improvement_pct:.1f}%), while "
            f"maintaining {stability_verdict} learning behavior. "
            f"This configuration is now set as the **default** in "
            f"`agents/q_learning_agent.py`.\n"
        )
    else:
        lines.append(
            f"The tuned configuration did not show a clear improvement in "
            f"this run ({reward_improvement_pct:+.1f}%). Given natural "
            f"run-to-run variance in RL training, further repeated trials "
            f"are recommended before finalizing this as the default.\n"
        )

    lines.append("## Selected Hyperparameters (Final)\n")
    lines.append("```python")
    lines.append("QLearningAgent(")
    lines.append(f"    learning_rate={AFTER_PARAMS['learning_rate']},")
    lines.append(f"    discount_factor={AFTER_PARAMS['discount_factor']},")
    lines.append(f"    epsilon={AFTER_PARAMS['epsilon']},")
    lines.append(f"    epsilon_decay={AFTER_PARAMS['epsilon_decay']},")
    lines.append(f"    epsilon_min={AFTER_PARAMS['epsilon_min']},")
    lines.append(f"    exploration_strategy=\"{AFTER_PARAMS['exploration_strategy']}\",")
    lines.append(")")
    lines.append("```")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\nReport written to {filepath}")


if __name__ == "__main__":
    env = PricingEnvironment(PricingEnvConfig())

    print("Training BEFORE-tuning agent (original defaults)...")
    before_results = train_and_measure(BEFORE_PARAMS, env, TRAIN_EPISODES)
    print(f"  Avg Reward: {before_results['avg_reward']:.2f} | "
          f"Avg Revenue: {before_results['avg_revenue']:.2f} | "
          f"Stability CV: {before_results['stability_cv_pct']:.1f}%")

    print("\nTraining AFTER-tuning agent (optimized defaults)...")
    after_results = train_and_measure(AFTER_PARAMS, env, TRAIN_EPISODES)
    print(f"  Avg Reward: {after_results['avg_reward']:.2f} | "
          f"Avg Revenue: {after_results['avg_revenue']:.2f} | "
          f"Stability CV: {after_results['stability_cv_pct']:.1f}%")

    os.makedirs("reports", exist_ok=True)
    generate_report(before_results, after_results, "reports/q_learning_optimization.md")

    print("\nOptimization documentation complete!")