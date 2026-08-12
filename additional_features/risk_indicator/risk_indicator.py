"""
risk_indicator.py

Risk Indicator feature: measures how much REVENUE VARIABILITY a given
pricing situation carries, by running many simulated futures from that
point forward and looking at the spread of resulting outcomes.

This is deliberately different from a "confidence score" (which measures
how certain an agent is about its TOP price choice, e.g. based on the gap
between the best and second-best Q-values). This module instead asks:
"if we're in this situation and follow our policy from here, how
consistent is the eventual revenue likely to be?" - a business risk
question, not a model-certainty question.

How it works
------------
From a given (days_remaining, inventory_remaining) starting point, we run
N independent simulated seasons forward (each with a different random
seed, so demand plays out differently each time), letting the given
policy make every pricing decision along the way. We then look at the
DISTRIBUTION of final revenue across those N simulations:
  - a tight distribution (low spread) -> Low risk, outcome is predictable
  - a wide distribution (high spread) -> High risk, outcome is volatile

Usage
-----
    from pricing_env import PricingEnvironment, PricingEnvConfig
    from agents.q_learning_agent import QLearningAgent
    from evaluation.risk_indicator import assess_risk

    config = PricingEnvConfig()
    env = PricingEnvironment(config)
    agent = QLearningAgent.load("agents/saved_policy.pkl")

    result = assess_risk(
        env, act_fn=lambda obs: agent.select_greedy_action(obs),
        days_remaining=15, inventory_remaining=40,
    )
    print(result)
"""

import os
import sys
from dataclasses import dataclass
from typing import Callable

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


@dataclass
class RiskAssessment:
    """Result of a risk assessment for one (days_remaining, inventory_remaining) situation."""
    days_remaining: int
    inventory_remaining: int
    num_simulations: int
    mean_revenue: float
    std_revenue: float
    min_revenue: float
    max_revenue: float
    worst_case_10pct: float   # 10th percentile - a "things go badly" estimate
    coefficient_of_variation: float  # std / mean - the actual risk metric
    risk_level: str           # "Low" / "Medium" / "High"


def _classify_risk(coefficient_of_variation: float) -> str:
    """
    Buckets the coefficient of variation (std/mean - a scale-independent
    measure of relative spread) into a human-readable risk level.

    Coefficient of variation is used rather than raw std dev because raw
    std dev isn't comparable across situations with very different mean
    revenue (e.g. $500 std dev is huge risk on a $1,000 mean, but tiny
    risk on a $20,000 mean) - dividing by the mean fixes that.
    """
    if coefficient_of_variation < 0.15:
        return "Low"
    if coefficient_of_variation < 0.35:
        return "Medium"
    return "High"


def assess_risk(
    env,
    act_fn: Callable,
    days_remaining: int,
    inventory_remaining: int,
    num_simulations: int = 100,
    base_seed: int = 500_000,
) -> RiskAssessment:
    """
    Runs `num_simulations` independent simulated seasons starting from
    the given (days_remaining, inventory_remaining) situation, using
    `act_fn` to make every pricing decision, and returns a summary of
    how much the resulting revenue varies.

    Parameters
    ----------
    env : PricingEnvironment
    act_fn : callable
        A function taking an observation and returning an action index -
        e.g. a trained agent's `select_greedy_action`.
    days_remaining, inventory_remaining : int
        The situation to assess risk for.
    num_simulations : int
        How many simulated futures to run. More simulations = more
        reliable risk estimate, at the cost of more compute time.
    base_seed : int
        Starting seed; each simulation uses base_seed + i so results
        differ per run but are fully reproducible overall.

    Returns
    -------
    RiskAssessment
    """
    revenues = []

    for i in range(num_simulations):
        obs, info = env.reset(seed=base_seed + i)
        # Force the environment's internal state to the situation we're
        # assessing, rather than always starting from a full season -
        # this lets us ask "what if we're HERE, right now" rather than
        # only ever assessing risk from day one.
        env._state.days_remaining = days_remaining
        env._state.remaining_inventory = inventory_remaining
        obs = env._state.to_observation()

        terminated = truncated = False
        while not (terminated or truncated):
            action = act_fn(obs)
            obs, reward, terminated, truncated, info = env.step(action)

        revenues.append(info["episode_revenue"])

    revenues = np.array(revenues)
    mean_revenue = float(np.mean(revenues))
    std_revenue = float(np.std(revenues))
    cv = (std_revenue / mean_revenue) if mean_revenue > 0 else 0.0

    return RiskAssessment(
        days_remaining=days_remaining,
        inventory_remaining=inventory_remaining,
        num_simulations=num_simulations,
        mean_revenue=mean_revenue,
        std_revenue=std_revenue,
        min_revenue=float(np.min(revenues)),
        max_revenue=float(np.max(revenues)),
        worst_case_10pct=float(np.percentile(revenues, 10)),
        coefficient_of_variation=round(cv, 4),
        risk_level=_classify_risk(cv),
    )


def print_risk_report(result: RiskAssessment) -> None:
    """Pretty-prints a RiskAssessment to the console."""
    print(f"Risk Assessment: {result.days_remaining} days left, "
          f"{result.inventory_remaining} units remaining")
    print(f"  Simulations run:       {result.num_simulations}")
    print(f"  Expected revenue:      ${result.mean_revenue:,.2f}")
    print(f"  Revenue range:         ${result.min_revenue:,.2f} - ${result.max_revenue:,.2f}")
    print(f"  Worst-case (10th pct): ${result.worst_case_10pct:,.2f}")
    print(f"  Coefficient of variation: {result.coefficient_of_variation}")
    print(f"  Risk Level: {result.risk_level}")


if __name__ == "__main__":
    from pricing_env import PricingEnvironment, PricingEnvConfig
    from agents.q_learning_agent import QLearningAgent

    config = PricingEnvConfig()
    env = PricingEnvironment(config)

    checkpoint_path = os.path.join(os.path.dirname(__file__), '..', 'agents', 'q_table.pkl')
    if os.path.exists(checkpoint_path):
        agent = QLearningAgent.load(checkpoint_path, action_space=env.action_space)
        act_fn = lambda obs: agent.select_greedy_action(obs)
        print(f"Loaded trained agent from {checkpoint_path}\n")
    else:
        print(f"No trained agent found at {checkpoint_path} - using random policy instead.\n")
        act_fn = lambda obs: env.action_space.sample()

    # Assess risk at a few representative points in the season
    test_situations = [
        (30, 100),  # season start, full inventory
        (15, 60),   # halfway through, moderate inventory
        (5, 60),    # near deadline, lots of inventory left (risky!)
        (5, 5),     # near deadline, almost sold out (safe)
    ]

    for days, inv in test_situations:
        result = assess_risk(env, act_fn, days_remaining=days, inventory_remaining=inv, num_simulations=100)
        print_risk_report(result)
        print()