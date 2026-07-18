"""
exploration_comparison.py

Compares the ORIGINAL exploration strategy (plain epsilon-greedy, decay
starts immediately) against the IMPROVED exploration strategy (Boltzmann
action selection + exploration warm-up period), for Issue #51.

Uses the best baseline hyperparameters identified in Issue #47
(Slow Epsilon Decay config: alpha=0.1, gamma=0.95, epsilon_decay=0.999)
as the common foundation for both agents, so exploration strategy is the
only variable being changed.

Metrics compared:
  - Average reward (evaluation, no exploration)
  - Average revenue
  - Convergence behavior: how reward progressed across training
    (early vs late training performance, and how many states were
    explored — a proxy for how broadly the agent searched before settling)

Results are written to evaluation/exploration_comparison.md
"""

import sys
import os
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from pricing_env.pricing_env import PricingEnvironment, PricingEnvConfig
from agents.q_learning_agent import QLearningAgent


TRAIN_EPISODES = 2000
EVAL_EPISODES = 200

# Common base hyperparameters (best config found in Issue #47)
BASE_PARAMS = {
    "learning_rate": 0.1,
    "discount_factor": 0.95,
    "epsilon": 1.0,
    "epsilon_decay": 0.999,
    "epsilon_min": 0.01,
}


def train_and_collect(agent, env, num_episodes):
    """Train an agent and return convergence tracking data."""
    agent.train(env, num_episodes=num_episodes, verbose=False)

    rewards = np.array(agent.episode_rewards)
    early_avg = float(np.mean(rewards[:100]))
    late_avg = float(np.mean(rewards[-100:]))

    # Simple convergence check: first episode index where the rolling
    # 100-episode average reward gets within 5% of the final (late) average
    # -- a rough proxy for "how long it took to stabilize".
    convergence_episode = num_episodes  # default: never fully converged
    threshold = 0.05 * abs(late_avg) if late_avg != 0 else 1.0
    for i in range(100, num_episodes):
        rolling_avg = np.mean(rewards[i - 100:i])
        if abs(rolling_avg - late_avg) <= threshold:
            convergence_episode = i
            break

    return {
        "early_avg_reward": early_avg,
        "late_avg_reward": late_avg,
        "convergence_episode": convergence_episode,
        "states_explored": len(agent.q_table),
    }


def run_comparison():
    env = PricingEnvironment(PricingEnvConfig())

    # --- ORIGINAL exploration strategy ---
    print("Training ORIGINAL exploration agent (plain epsilon-greedy)...")
    original_agent = QLearningAgent(
        num_actions=env.action_space.n,
        exploration_strategy="epsilon_greedy",
        warmup_episodes=0,
        **BASE_PARAMS,
    )
    original_convergence = train_and_collect(original_agent, env, TRAIN_EPISODES)
    original_eval = original_agent.evaluate(env, num_episodes=EVAL_EPISODES)

    print(f"  Avg Reward: {original_eval['avg_reward']:.2f} | "
          f"Avg Revenue: {original_eval['avg_revenue']:.2f} | "
          f"States explored: {original_convergence['states_explored']}")

    # --- IMPROVED exploration strategy ---
    print("\nTraining IMPROVED exploration agent (Boltzmann + warm-up)...")
    improved_agent = QLearningAgent(
        num_actions=env.action_space.n,
        exploration_strategy="boltzmann_epsilon_greedy",
        warmup_episodes=200,
        temperature=1.0,
        temperature_decay=0.995,
        temperature_min=0.05,
        **BASE_PARAMS,
    )
    improved_convergence = train_and_collect(improved_agent, env, TRAIN_EPISODES)
    improved_eval = improved_agent.evaluate(env, num_episodes=EVAL_EPISODES)

    print(f"  Avg Reward: {improved_eval['avg_reward']:.2f} | "
          f"Avg Revenue: {improved_eval['avg_revenue']:.2f} | "
          f"States explored: {improved_convergence['states_explored']}")

    return {
        "original": {**original_eval, **original_convergence},
        "improved": {**improved_eval, **improved_convergence},
    }


def generate_report(results, filepath):
    orig = results["original"]
    imp = results["improved"]

    reward_improvement_pct = (
        (imp["avg_reward"] - orig["avg_reward"]) / abs(orig["avg_reward"]) * 100
        if orig["avg_reward"] != 0 else 0.0
    )
    revenue_improvement_pct = (
        (imp["avg_revenue"] - orig["avg_revenue"]) / abs(orig["avg_revenue"]) * 100
        if orig["avg_revenue"] != 0 else 0.0
    )

    lines = []
    lines.append("# Exploration Strategy Comparison — Issue #51\n")
    lines.append(f"**Training episodes:** {TRAIN_EPISODES}")
    lines.append(f"**Evaluation episodes:** {EVAL_EPISODES} (greedy, no exploration)")
    lines.append(
        f"**Common base hyperparameters:** alpha={BASE_PARAMS['learning_rate']}, "
        f"gamma={BASE_PARAMS['discount_factor']}, "
        f"epsilon_decay={BASE_PARAMS['epsilon_decay']} "
        f"(best config identified in Issue #47)\n"
    )

    lines.append("## Strategies Compared\n")
    lines.append(
        "**Original — plain epsilon-greedy:** when exploring, picks a "
        "completely random action. Epsilon decay begins immediately from "
        "episode 1.\n"
    )
    lines.append(
        "**Improved — Boltzmann + exploration warm-up:** when exploring, "
        "picks an action using softmax over current Q-values (still "
        "explores, but leans toward more promising actions instead of "
        "wasting attempts on actions with clear evidence of being bad). "
        "Additionally, epsilon is held at its starting value for the "
        "first 200 episodes (warm-up) before decay begins, giving the "
        "agent a guaranteed period of broad exploration before it starts "
        "narrowing its strategy down — intended to avoid settling into a "
        "poor local pricing habit too early.\n"
    )

    lines.append("## Performance Comparison\n")
    lines.append("| Metric | Original (epsilon-greedy) | Improved (Boltzmann + warm-up) | Change |")
    lines.append("|---|---|---|---|")
    lines.append(
        f"| Avg Reward | {orig['avg_reward']:.2f} | {imp['avg_reward']:.2f} | "
        f"{reward_improvement_pct:+.1f}% |"
    )
    lines.append(
        f"| Avg Revenue | {orig['avg_revenue']:.2f} | {imp['avg_revenue']:.2f} | "
        f"{revenue_improvement_pct:+.1f}% |"
    )
    lines.append(
        f"| Inventory Utilization | {orig['avg_inventory_utilization']*100:.1f}% | "
        f"{imp['avg_inventory_utilization']*100:.1f}% | "
        f"{(imp['avg_inventory_utilization']-orig['avg_inventory_utilization'])*100:+.1f}pp |"
    )
    lines.append(
        f"| States Explored (breadth) | {orig['states_explored']} | "
        f"{imp['states_explored']} | "
        f"{imp['states_explored']-orig['states_explored']:+d} |"
    )
    lines.append("")

    lines.append("## Convergence Behavior\n")
    lines.append("| Metric | Original | Improved |")
    lines.append("|---|---|---|")
    lines.append(
        f"| Early training avg reward (first 100 episodes) | "
        f"{orig['early_avg_reward']:.2f} | {imp['early_avg_reward']:.2f} |"
    )
    lines.append(
        f"| Late training avg reward (last 100 episodes) | "
        f"{orig['late_avg_reward']:.2f} | {imp['late_avg_reward']:.2f} |"
    )
    lines.append(
        f"| Episode where performance stabilized (within 5% of final) | "
        f"{orig['convergence_episode']} | {imp['convergence_episode']} |"
    )
    lines.append("")

    lines.append("## Interpretation\n")
    if reward_improvement_pct > 0:
        lines.append(
            f"The improved exploration strategy achieved a "
            f"**{reward_improvement_pct:.1f}% higher average reward** than "
            f"plain epsilon-greedy exploration. This supports the idea that "
            f"guiding exploration toward more promising actions (Boltzmann) "
            f"and delaying decay (warm-up) helps the agent discover a "
            f"better overall pricing policy rather than settling early "
            f"into a mediocre one.\n"
        )
    else:
        lines.append(
            f"In this run, the improved exploration strategy did not "
            f"outperform plain epsilon-greedy exploration "
            f"({reward_improvement_pct:.1f}% change). This may indicate "
            f"the warm-up period or temperature schedule needs further "
            f"tuning, or that this environment's state space is small "
            f"enough that plain epsilon-greedy already explores it "
            f"adequately.\n"
        )

    lines.append(
        f"States explored increased from {orig['states_explored']} to "
        f"{imp['states_explored']}, indicating "
        f"{'broader' if imp['states_explored'] > orig['states_explored'] else 'narrower'} "
        f"coverage of the state space under the improved strategy.\n"
    )

    lines.append("## Recommendation\n")
    if reward_improvement_pct > 0:
        lines.append(
            "Adopt the improved exploration strategy "
            "(Boltzmann action selection + 200-episode warm-up) as the "
            "default for future Q-Learning training runs, given its "
            "higher achieved reward at evaluation time.\n"
        )
    else:
        lines.append(
            "Continue using plain epsilon-greedy exploration for now, "
            "and revisit Boltzmann/warm-up tuning (e.g. different "
            "temperature decay rates or warm-up lengths) in a future "
            "iteration.\n"
        )

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\nReport written to {filepath}")


if __name__ == "__main__":
    results = run_comparison()

    os.makedirs("evaluation", exist_ok=True)
    generate_report(results, "evaluation/exploration_comparison.md")

    print("Comparison complete!")