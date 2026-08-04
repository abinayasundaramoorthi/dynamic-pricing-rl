"""
export_results.py

Infrastructure for storing simulation outputs, so results from any policy
(random, heuristic, Q-Learning, DQN, etc.) can be exported in a
consistent format for visualization and business reporting.

Provides:
- SimulationResult - a schema (one record per simulated episode)
- export_to_csv() / export_to_json() - format-specific export utilities
- run_simulation() - runs a policy through PricingEnvironment and
  collects results in the SimulationResult schema, ready to export

Schema fields (one row per episode):
- episode_number     : which simulated season this is
- policy_name        : which policy produced this episode (e.g. "Random",
                        "Q-Learning", "DQN") - lets multiple policies'
                        results be combined in one file for comparison
- revenue             : actual dollar revenue earned that episode
                        (info["episode_revenue"])
- reward              : total RL reward earned that episode - NOT the
                         same as revenue in this environment (see
                         pricing_env/reward.py)
- inventory_remaining : units left unsold at the end of the episode
- selling_price       : AVERAGE price charged across the episode (a
                         single representative number per episode, since
                         price changes every step - see the module
                         docstring note below for why average was chosen)

Usage
-----
    from pricing_env import PricingEnvironment, PricingEnvConfig
    from evaluation.export_results import run_simulation, export_to_csv, export_to_json

    env = PricingEnvironment(PricingEnvConfig())
    results = run_simulation(env, act_fn=lambda obs: env.action_space.sample(),
                              policy_name="Random", num_episodes=100)
    export_to_csv(results, "evaluation/simulation_results.csv")
    export_to_json(results, "evaluation/simulation_results.json")
"""

import os
import csv
import json
from dataclasses import dataclass, asdict
from typing import List, Callable

import numpy as np


@dataclass
class SimulationResult:
    """
    One row of the evaluation result schema - a single simulated episode's
    outcome under a given policy.
    """
    episode_number: int
    policy_name: str
    revenue: float
    reward: float
    inventory_remaining: int
    selling_price: float


def run_simulation(
    env,
    act_fn: Callable,
    policy_name: str,
    num_episodes: int = 100,
    seed: int = 999,
) -> List[SimulationResult]:
    """
    Runs `act_fn` through `num_episodes` full seasons, and returns one
    SimulationResult per episode.

    `selling_price` is recorded as the AVERAGE price charged across the
    episode, not every individual price - the schema is one row per
    EPISODE (matching the "Episode Number" field), not one row per day,
    so a single representative price number is needed. Average was
    chosen over e.g. final price, since it better reflects the overall
    pricing behavior of the episode rather than just its last moment.

    Parameters
    ----------
    env : PricingEnvironment
    act_fn : callable
        A function taking an observation and returning an action index.
    policy_name : str
        Label identifying which policy produced these results (e.g.
        "Random", "Q-Learning", "DQN") - stored in every row, so results
        from multiple policies can be combined into one export and
        compared.
    num_episodes : int
    seed : int

    Returns
    -------
    List[SimulationResult]
    """
    results = []

    for episode in range(num_episodes):
        obs, info = env.reset(seed=seed + episode)
        terminated = truncated = False
        episode_reward = 0.0
        prices_this_episode = []

        while not (terminated or truncated):
            action = act_fn(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            prices_this_episode.append(info["price"])

        results.append(SimulationResult(
            episode_number=episode,
            policy_name=policy_name,
            revenue=float(info["episode_revenue"]),
            reward=float(episode_reward),
            inventory_remaining=int(obs[0]),
            selling_price=float(np.mean(prices_this_episode)) if prices_this_episode else 0.0,
        ))

    return results


def export_to_csv(results: List[SimulationResult], filepath: str) -> str:
    """
    Exports a list of SimulationResult records to a CSV file, one row
    per episode. Creates parent directories if needed. Appends to an
    existing file's policy comparison by simply writing multiple
    policies' results (concatenate the lists before calling this) into
    one combined file, rather than one file per policy.
    """
    if not results:
        raise ValueError("No results to export - `results` is empty.")

    parent = os.path.dirname(filepath)
    if parent:
        os.makedirs(parent, exist_ok=True)

    fieldnames = list(asdict(results[0]).keys())

    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))

    return filepath


def export_to_json(results: List[SimulationResult], filepath: str) -> str:
    """
    Exports a list of SimulationResult records to a JSON file, as a list
    of objects (one per episode). Creates parent directories if needed.
    """
    if not results:
        raise ValueError("No results to export - `results` is empty.")

    parent = os.path.dirname(filepath)
    if parent:
        os.makedirs(parent, exist_ok=True)

    with open(filepath, 'w') as f:
        json.dump([asdict(r) for r in results], f, indent=2)

    return filepath


if __name__ == "__main__":
    # Self-test / real data generation: run a few policies through the
    # real pricing_env package and export combined results, proving the
    # schema and both export utilities work end-to-end.
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

    from pricing_env import PricingEnvironment, PricingEnvConfig

    config = PricingEnvConfig()
    env = PricingEnvironment(config)

    print("Running simulations for multiple policies...")

    all_results: List[SimulationResult] = []

    # Random policy
    random_results = run_simulation(
        env, act_fn=lambda obs: env.action_space.sample(),
        policy_name="Random", num_episodes=100,
    )
    all_results.extend(random_results)
    print(f"  Random: {len(random_results)} episodes simulated")

    # "Hold price" heuristic - always take the middle (no-adjustment) action
    hold_action_index = len(config.price_adjustment_pct) // 2
    hold_results = run_simulation(
        env, act_fn=lambda obs: hold_action_index,
        policy_name="Hold Price", num_episodes=100,
    )
    all_results.extend(hold_results)
    print(f"  Hold Price: {len(hold_results)} episodes simulated")

    # Export combined results
    csv_path = os.path.join(os.path.dirname(__file__), "simulation_results.csv")
    json_path = os.path.join(os.path.dirname(__file__), "simulation_results.json")

    export_to_csv(all_results, csv_path)
    export_to_json(all_results, json_path)

    print(f"\nExported {len(all_results)} total episodes across {len(set(r.policy_name for r in all_results))} policies.")
    print(f"CSV:  {csv_path}")
    print(f"JSON: {json_path}")

    print(f"\nSample rows:")
    for r in all_results[:2] + all_results[-2:]:
        print(f"  {asdict(r)}")

    print(f"\nPer-policy summary:")
    for policy in set(r.policy_name for r in all_results):
        policy_results = [r for r in all_results if r.policy_name == policy]
        avg_revenue = np.mean([r.revenue for r in policy_results])
        avg_reward = np.mean([r.reward for r in policy_results])
        avg_price = np.mean([r.selling_price for r in policy_results])
        print(f"  {policy}: avg_revenue=${avg_revenue:.2f}, avg_reward={avg_reward:.2f}, avg_price=${avg_price:.2f}")