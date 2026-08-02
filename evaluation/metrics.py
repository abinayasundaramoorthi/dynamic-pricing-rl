"""
metrics.py

Reusable evaluation utilities to analyze and compare agent performance on
PricingEnvironment. Works with ANY agent that exposes a callable
act(observation) -> action_index interface, so heuristic baselines and
trained RL agents can all be evaluated with the exact same metrics.

Tracked metrics:
- Average Episode Reward   (mean total reward per episode - note this is
                             NOT the same as revenue in this environment,
                             since reward = revenue - discount_penalty -
                             unsold_penalty + balance_bonus; see reward.py)
- Revenue per Episode      (the actual dollar revenue earned, separate
                             from reward, pulled from info['episode_revenue'])
- Inventory Utilization    (% of initial inventory actually sold)
- Price Trends             (which prices get charged, and when in the
                             season they tend to be charged, using the
                             human-readable action_label from info)
"""

import sys
import os
import json
import csv
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from pricing_env import PricingEnvironment


def collect_episode_metrics(env, act_fn, num_episodes=100, seed=999):
    """
    Runs act_fn through num_episodes full seasons and records detailed
    per-episode and per-step data.

    Parameters
    ----------
    env : PricingEnvironment
        The environment to evaluate in.
    act_fn : callable
        A function taking an observation (numpy array) and returning an
        action index (int). Wrap agent-specific act() methods in a lambda
        if needed (e.g. for agents that need extra args like a training flag).
    num_episodes : int
        Number of episodes to run.
    seed : int
        Base random seed for reproducibility.

    Returns
    -------
    dict
        Raw per-episode and per-step data, used by
        compute_summary_statistics() to produce the final metrics.
    """
    episode_rewards = []          # sum of step rewards (NOT the same as revenue here)
    episode_revenues = []         # actual dollar revenue, from info['episode_revenue']
    episode_units_sold = []
    all_prices_charged = []
    all_action_labels = []
    price_by_days_remaining = {}  # {days_remaining: [prices charged at that point]}

    for episode in range(num_episodes):
        obs, info = env.reset(seed=seed + episode)
        terminated = truncated = False
        episode_reward = 0.0

        while not (terminated or truncated):
            days_remaining_before = int(round(obs[1]))  # obs = [inventory, days_remaining]
            action = act_fn(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward

            price = info["price"]
            all_prices_charged.append(price)
            all_action_labels.append(info["action_label"])
            price_by_days_remaining.setdefault(days_remaining_before, []).append(price)

        episode_rewards.append(episode_reward)
        episode_revenues.append(info["episode_revenue"])
        units_sold = env.config.initial_inventory - obs[0]
        episode_units_sold.append(units_sold)

    return {
        "num_episodes": num_episodes,
        "initial_inventory": env.config.initial_inventory,
        "episode_rewards": episode_rewards,
        "episode_revenues": episode_revenues,
        "episode_units_sold": episode_units_sold,
        "all_prices_charged": all_prices_charged,
        "all_action_labels": all_action_labels,
        "price_by_days_remaining": price_by_days_remaining,
    }


def compute_summary_statistics(raw_metrics):
    """
    Takes the raw data from collect_episode_metrics() and computes the
    four required summary metrics: Average Episode Reward, Revenue per
    Episode stats, Inventory Utilization, and Price Trends.
    """
    rewards = np.array(raw_metrics["episode_rewards"])
    revenues = np.array(raw_metrics["episode_revenues"])
    units_sold = np.array(raw_metrics["episode_units_sold"])
    initial_inventory = raw_metrics["initial_inventory"]
    all_prices = np.array(raw_metrics["all_prices_charged"])

    # --- Average Episode Reward (RL reward signal, includes penalties/bonuses) ---
    avg_episode_reward = float(np.mean(rewards))
    std_episode_reward = float(np.std(rewards))

    # --- Revenue per Episode (actual dollar revenue, separate from reward) ---
    revenue_stats = {
        "avg": float(np.mean(revenues)),
        "std": float(np.std(revenues)),
        "min": float(np.min(revenues)),
        "max": float(np.max(revenues)),
        "median": float(np.median(revenues)),
    }

    # --- Inventory Utilization ---
    utilization_pct = units_sold / initial_inventory * 100
    inventory_utilization = {
        "avg_units_sold": float(np.mean(units_sold)),
        "avg_utilization_pct": float(np.mean(utilization_pct)),
        "sellout_rate_pct": float(np.mean(units_sold >= initial_inventory) * 100),
        "min_utilization_pct": float(np.min(utilization_pct)),
    }

    # --- Price Trends ---
    action_labels, label_counts = np.unique(raw_metrics["all_action_labels"], return_counts=True)
    action_distribution = {
        str(label): float(c / len(raw_metrics["all_action_labels"]) * 100)
        for label, c in zip(action_labels, label_counts)
    }

    price_by_days = raw_metrics["price_by_days_remaining"]
    all_days = sorted(price_by_days.keys(), reverse=True)  # most days left -> fewest
    if all_days:
        n = len(all_days)
        early_days = all_days[: max(1, n // 3)]
        mid_days = all_days[max(1, n // 3): max(2, 2 * n // 3)]
        late_days = all_days[max(2, 2 * n // 3):]

        def avg_price_for(day_group):
            prices = [p for d in day_group for p in price_by_days.get(d, [])]
            return float(np.mean(prices)) if prices else None

        price_trend_by_phase = {
            "early_season_avg_price": avg_price_for(early_days),
            "mid_season_avg_price": avg_price_for(mid_days),
            "late_season_avg_price": avg_price_for(late_days),
        }
    else:
        price_trend_by_phase = {}

    return {
        "average_episode_reward": avg_episode_reward,
        "std_episode_reward": std_episode_reward,
        "revenue_per_episode": revenue_stats,
        "inventory_utilization": inventory_utilization,
        "price_trends": {
            "action_distribution_pct": action_distribution,
            "by_season_phase": price_trend_by_phase,
        },
    }


def evaluate_agent(env, act_fn, agent_name, num_episodes=100, seed=999):
    """
    Convenience wrapper: runs collect_episode_metrics() then
    compute_summary_statistics() in one call, and tags the result with
    the agent's name for easy comparison across multiple agents.
    """
    raw = collect_episode_metrics(env, act_fn, num_episodes=num_episodes, seed=seed)
    summary = compute_summary_statistics(raw)
    summary["agent_name"] = agent_name
    summary["num_episodes"] = num_episodes
    return summary


def save_metrics_json(summary, filepath):
    """Saves a summary dict (from evaluate_agent) to a JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True) if os.path.dirname(filepath) else None
    with open(filepath, 'w') as f:
        json.dump(summary, f, indent=2)


def save_comparison_csv(summaries, filepath):
    """
    Saves a list of summary dicts (one per agent) to a single CSV file,
    one row per agent, for easy comparison/plotting in a spreadsheet.
    """
    rows = []
    for s in summaries:
        rows.append({
            "agent_name": s["agent_name"],
            "avg_episode_reward": round(s["average_episode_reward"], 2),
            "std_episode_reward": round(s["std_episode_reward"], 2),
            "avg_revenue": round(s["revenue_per_episode"]["avg"], 2),
            "avg_units_sold": round(s["inventory_utilization"]["avg_units_sold"], 2),
            "avg_utilization_pct": round(s["inventory_utilization"]["avg_utilization_pct"], 2),
            "sellout_rate_pct": round(s["inventory_utilization"]["sellout_rate_pct"], 2),
            "early_season_avg_price": s["price_trends"]["by_season_phase"].get("early_season_avg_price"),
            "late_season_avg_price": s["price_trends"]["by_season_phase"].get("late_season_avg_price"),
        })

    os.makedirs(os.path.dirname(filepath), exist_ok=True) if os.path.dirname(filepath) else None
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    # Self-test: evaluate a random policy end-to-end through the full
    # metrics pipeline, to confirm everything calculates correctly against
    # the real PricingEnvironment (not a mock).
    from pricing_env import PricingEnvConfig

    env = PricingEnvironment(PricingEnvConfig())

    random_summary = evaluate_agent(
        env,
        act_fn=lambda obs: env.action_space.sample(),
        agent_name="Random",
        num_episodes=100,
    )

    print("=== Metrics self-test: Random Policy ===")
    print(f"Average Episode Reward: {random_summary['average_episode_reward']:.2f}")
    print(f"Revenue per Episode - min/avg/max: "
          f"${random_summary['revenue_per_episode']['min']:.2f} / "
          f"${random_summary['revenue_per_episode']['avg']:.2f} / "
          f"${random_summary['revenue_per_episode']['max']:.2f}")
    print(f"Inventory Utilization: {random_summary['inventory_utilization']['avg_utilization_pct']:.1f}% "
          f"(sellout rate: {random_summary['inventory_utilization']['sellout_rate_pct']:.1f}%)")
    print(f"Price Trends by phase: {random_summary['price_trends']['by_season_phase']}")
    print(f"Action distribution: {random_summary['price_trends']['action_distribution_pct']}")