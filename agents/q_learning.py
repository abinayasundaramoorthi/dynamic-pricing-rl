"""
q_learning.py

Tabular Q-Learning agent for the pricing environment.

Learns a (remaining_inventory, days_remaining) -> pricing-action policy via
the standard Q-Learning update rule (Watkins & Dayan, 1992):

    Q(s, a) <- Q(s, a) + alpha * (r + gamma * max_a' Q(s', a') - Q(s, a))
"""

from __future__ import annotations

import json
import logging
import pickle
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from gymnasium import spaces

logger = logging.getLogger(__name__)

StateKey = Tuple[int, int]  # (remaining_inventory, days_remaining)


class QLearningAgent:
    def __init__(
        self,
        action_space: spaces.Discrete,
        learning_rate: float,
        discount_factor: float,
        exploration_rate: float,
        exploration_min: float,
        exploration_decay: float,
        seed: int,
    ) -> None:
        if not (0.0 < learning_rate <= 1.0):
            raise ValueError(f"learning_rate must be in (0, 1], got {learning_rate}")
        if not (0.0 <= discount_factor < 1.0):
            raise ValueError(f"discount_factor must be in [0, 1), got {discount_factor}")
        if not (0.0 <= exploration_min <= exploration_rate <= 1.0):
            raise ValueError(
                "Require 0 <= exploration_min <= exploration_rate <= 1, got "
                f"exploration_rate={exploration_rate}, exploration_min={exploration_min}"
            )
        if not (0.0 < exploration_decay <= 1.0):
            raise ValueError(f"exploration_decay must be in (0, 1], got {exploration_decay}")

        self.action_space = action_space
        self.num_actions = int(action_space.n)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_min = exploration_min
        self.exploration_decay = exploration_decay

        self.q_table: Dict[StateKey, np.ndarray] = defaultdict(
            lambda: np.zeros(self.num_actions, dtype=np.float64)
        )
        self._rng = np.random.default_rng(seed)

    @staticmethod
    def _state_key(observation: np.ndarray) -> StateKey:
        return (int(round(float(observation[0]))), int(round(float(observation[1]))))

    def select_action(self, observation: np.ndarray) -> int:
        if self._rng.random() < self.exploration_rate:
            return int(self._rng.integers(0, self.num_actions))
        return self.select_greedy_action(observation)

    def select_greedy_action(self, observation: np.ndarray) -> int:
        state = self._state_key(observation)
        q_values = self.q_table[state]
        best = np.flatnonzero(q_values == q_values.max())
        return int(self._rng.choice(best))

    def update(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        terminated: bool,
    ) -> None:
        state = self._state_key(observation)
        next_state = self._state_key(next_observation)

        current_q = self.q_table[state][action]
        next_max_q = 0.0 if terminated else float(np.max(self.q_table[next_state]))
        td_target = reward + self.discount_factor * next_max_q
        self.q_table[state][action] = current_q + self.learning_rate * (td_target - current_q)

    def decay_exploration(self) -> None:
        self.exploration_rate = max(
            self.exploration_min, self.exploration_rate * self.exploration_decay
        )

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "q_table": dict(self.q_table),
            "num_actions": self.num_actions,
            "learning_rate": self.learning_rate,
            "discount_factor": self.discount_factor,
            "exploration_rate": self.exploration_rate,
            "exploration_min": self.exploration_min,
            "exploration_decay": self.exploration_decay,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)

        metadata = {
            "num_states_visited": len(self.q_table),
            "num_actions": self.num_actions,
            "learning_rate": self.learning_rate,
            "discount_factor": self.discount_factor,
            "final_exploration_rate": self.exploration_rate,
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        with open(str(path) + ".json", "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info("Saved Q-Learning policy | states_visited=%d | path=%s", len(self.q_table), path)

    @classmethod
    def load(cls, path: Path, action_space: spaces.Discrete, seed: int = 0) -> "QLearningAgent":
        with open(path, "rb") as f:
            payload = pickle.load(f)

        if payload["num_actions"] != action_space.n:
            raise ValueError(
                f"Saved policy has {payload['num_actions']} actions but the "
                f"given action_space has {action_space.n}. This policy was "
                "trained against a differently-configured environment and "
                "cannot be safely evaluated against this one."
            )

        agent = cls(
            action_space=action_space,
            learning_rate=payload["learning_rate"],
            discount_factor=payload["discount_factor"],
            exploration_rate=payload["exploration_rate"],
            exploration_min=payload["exploration_min"],
            exploration_decay=payload["exploration_decay"],
            seed=seed,
        )
        agent.q_table = defaultdict(
            lambda: np.zeros(agent.num_actions, dtype=np.float64),
            payload["q_table"],
        )
        logger.info("Loaded Q-Learning policy | states=%d | path=%s", len(agent.q_table), path)
        return agent