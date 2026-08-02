"""
baseline_random.py

Uniform-random pricing policy: every step, pick one of the configured
price-adjustment actions with equal probability, independent of the
observed state.

This is the weakest possible "no intelligence" baseline — any agent
(learned or rule-based) that cannot beat it has learned nothing useful.
It also gives the evaluation harness a sanity check on the environment
and reward function themselves: `RewardConfig`'s penalties should punish
random pricing hard (frequent deep discounts, poor pacing), which is a
useful independent confirmation that the reward shaping in
`pricing_env/reward.py` behaves as designed.

Note: `training/train_agent.py` already has a `random_policy(observation,
env)` function used as the Day 1 training placeholder. This class is
deliberately a separate, small wrapper around the same idea (sample from
`action_space`) rather than importing that function directly — the
evaluation harness's `Policy` protocol expects a stateful object with
`select_greedy_action(observation)` and `reset()`, and reaching into
`train_agent.py` (a *training* entry point, not a library module) from the
evaluation package would create an awkward cross-dependency between the
two independently-owned pipelines.
"""

from __future__ import annotations

import numpy as np
from gymnasium import spaces


class RandomPolicy:
    """
    Samples a uniformly random valid action every step.

    Parameters
    ----------
    action_space : spaces.Discrete
        The environment's action space to sample from.
    seed : int
        Seed for the policy's own RNG. Kept independent of the
        environment's RNG (`env.np_random`) so evaluation runs are
        reproducible from the policy side too — the same seed always
        produces the same sequence of random actions, regardless of how
        many times `select_greedy_action` is called across episodes.
    """

    name = "random"

    def __init__(self, action_space: spaces.Discrete, seed: int = 0) -> None:
        self.action_space = action_space
        self._rng = np.random.default_rng(seed)

    def select_greedy_action(self, observation: np.ndarray) -> int:
        """Ignore `observation` entirely; sample uniformly at random."""
        return int(self._rng.integers(0, self.action_space.n))

    def reset(self) -> None:
        """No internal episode state to reset (stateless policy)."""
        return None