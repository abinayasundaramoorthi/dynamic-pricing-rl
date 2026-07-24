"""
dqn_agent.py

Deep Q-Network agent for the pricing environment (Mnih et al., 2015,
"Human-level control through deep reinforcement learning").

Replaces the tabular Q-table (agents/q_learning.py) with a small
feedforward neural network Q(s, a; theta), trained via two standard DQN
stabilization mechanisms:

  1. Experience replay: transitions are stored in a buffer and sampled in
     random minibatches, breaking the strong temporal correlation between
     consecutive environment steps that would otherwise destabilize SGD
     (consecutive transitions from one episode are highly similar; a
     network trained directly on them in sequence would overfit to
     whatever part of the state space that episode happened to visit).
  2. A separate target network Q(s, a; theta_target), synced periodically
     from the online network, used to compute TD targets. Without this,
     the network would be chasing a target that moves every single
     gradient step (bootstrapping off itself), a well-documented source
     of training divergence.

Why DQN instead of the tabular Q-table now: the Week 2 tabular agent
(agents/q_learning.py) works well because Phase 1's state space is tiny
(~3,000 reachable (inventory, days) states) — that was an explicit,
documented design choice, not an oversight. DQN is introduced here because
the roadmap's Phase 2 extensions (competitor price, customer segment,
seasonality — see the design doc) would make a tabular Q-table grow
combinatorially large and eventually impossible to enumerate; a function
approximator doesn't have that limitation.
"""

from __future__ import annotations

import json
import logging
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Deque, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces

logger = logging.getLogger(__name__)


class QNetwork(nn.Module):
    """
    Feedforward MLP mapping a (normalized) state vector to one Q-value
    per discrete action.

    Kept intentionally small (default two hidden layers of 64 units):
    this environment's observation is a 2-dimensional vector
    (remaining_inventory, days_remaining), not an image or a
    high-dimensional sensor reading. A larger network would only add
    unnecessary variance to training without any representational
    benefit — there simply isn't enough underlying structure in a
    2-dimensional input to justify more capacity.
    """

    def __init__(
        self, input_dim: int, num_actions: int, hidden_layer_sizes: List[int]
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        in_features = input_dim
        for hidden_size in hidden_layer_sizes:
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(nn.ReLU())
            in_features = hidden_size
        layers.append(nn.Linear(in_features, num_actions))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


@dataclass
class Transition:
    """One (s, a, r, s', done) transition stored in the replay buffer."""

    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """
    Fixed-size circular buffer of transitions, sampled uniformly at random.

    A `deque(maxlen=...)` automatically evicts the oldest transition once
    full — the standard, simplest DQN replay strategy. Prioritized replay
    (sampling transitions with larger TD-error more often) is a documented,
    common extension, but adds real complexity (a sum-tree structure,
    importance-sampling bias correction) that isn't justified yet for a
    state space this small and low-dimensional.
    """

    def __init__(self, capacity: int, seed: int) -> None:
        self._buffer: Deque[Transition] = deque(maxlen=capacity)
        self._rng = np.random.default_rng(seed)

    def push(self, transition: Transition) -> None:
        self._buffer.append(transition)

    def sample(self, batch_size: int) -> List[Transition]:
        indices = self._rng.choice(len(self._buffer), size=batch_size, replace=False)
        return [self._buffer[i] for i in indices]

    def __len__(self) -> int:
        return len(self._buffer)


class DQNAgent:
    """
    Deep Q-Network agent.

    Exposes `select_action()` / `select_greedy_action()` /
    `decay_exploration()` with the same names and meaning as
    `agents/q_learning.py`'s `QLearningAgent`, so the two agents are easy
    to compare conceptually. Learning itself works differently, though:
    tabular Q-Learning updates immediately from a single transition
    (`update()`); DQN instead accumulates transitions via `remember()` and
    periodically trains on a *sampled minibatch* via `train_step()` — a
    single transition alone isn't enough context for a stable gradient
    step, which is precisely why DQN needs the replay buffer.

    This is also why DQN is driven by its own dedicated training loop in
    `training/train_dqn.py` rather than being wired into
    `training.train_agent.build_agent()`: `train_agent.run_training()`'s
    per-step `agent.update(transition)` contract was built for tabular
    Q-Learning's single-transition update rule and doesn't fit DQN's
    batch-based, warm-up-gated training step cleanly. See
    `reports/dqn_architecture.md` for the full rationale.
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        action_space: spaces.Discrete,
        hidden_layer_sizes: List[int],
        learning_rate: float,
        discount_factor: float,
        exploration_rate: float,
        exploration_min: float,
        exploration_decay: float,
        batch_size: int,
        replay_buffer_size: int,
        min_replay_size_before_training: int,
        target_update_frequency: int,
        grad_clip_norm: Optional[float],
        device: str,
        seed: int,
    ) -> None:
        if not (0.0 < learning_rate <= 1.0):
            raise ValueError(f"learning_rate must be in (0, 1], got {learning_rate}")
        if not (0.0 <= discount_factor < 1.0):
            raise ValueError(
                f"discount_factor must be in [0, 1), got {discount_factor}"
            )
        if not (0.0 <= exploration_min <= exploration_rate <= 1.0):
            raise ValueError(
                "Require 0 <= exploration_min <= exploration_rate <= 1, got "
                f"exploration_rate={exploration_rate}, exploration_min={exploration_min}"
            )
        if not (0.0 < exploration_decay <= 1.0):
            raise ValueError(
                f"exploration_decay must be in (0, 1], got {exploration_decay}"
            )
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")
        if min_replay_size_before_training < batch_size:
            raise ValueError(
                "min_replay_size_before_training must be >= batch_size, got "
                f"{min_replay_size_before_training} < {batch_size}"
            )

        try:
            self.device = torch.device(device)
            if self.device.type == "cuda" and not torch.cuda.is_available():
                raise RuntimeError(
                    "device='cuda' requested but torch.cuda.is_available() "
                    "is False on this machine. Use device='cpu', or run on "
                    "a machine with a working CUDA installation."
                )
        except (RuntimeError, ValueError) as e:
            if "cuda" in str(e).lower():
                raise
            raise ValueError(f"Invalid torch device {device!r}: {e}") from e

        self.num_actions = int(action_space.n)
        self.obs_low = torch.tensor(
            observation_space.low, dtype=torch.float32, device=self.device
        )
        self.obs_high = torch.tensor(
            observation_space.high, dtype=torch.float32, device=self.device
        )
        self.obs_shape = observation_space.shape

        input_dim = int(np.prod(observation_space.shape))
        self.hidden_layer_sizes = list(hidden_layer_sizes)
        self.online_network = QNetwork(
            input_dim, self.num_actions, self.hidden_layer_sizes
        ).to(self.device)
        self.target_network = QNetwork(
            input_dim, self.num_actions, self.hidden_layer_sizes
        ).to(self.device)
        self.target_network.load_state_dict(self.online_network.state_dict())
        self.target_network.eval()  # target net is never trained directly

        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(
            self.online_network.parameters(), lr=learning_rate
        )

        self.replay_buffer = ReplayBuffer(replay_buffer_size, seed=seed)

        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_min = exploration_min
        self.exploration_decay = exploration_decay
        self.batch_size = batch_size
        self.min_replay_size_before_training = min_replay_size_before_training
        self.target_update_frequency = target_update_frequency
        self.grad_clip_norm = grad_clip_norm

        self._rng = np.random.default_rng(seed)
        self._gradient_step_count = 0

    def _normalize(self, observation: np.ndarray) -> torch.Tensor:
        """
        Min-max normalize the observation to roughly [0, 1] before feeding
        the network.

        Neural networks trained with SGD are notoriously sensitive to
        input scale: `remaining_inventory` can range up to hundreds while
        `days_remaining` ranges up to tens, and without normalization the
        larger-magnitude feature would dominate early gradients purely
        due to scale, not genuine importance to the Q-value. This is
        exactly the kind of practical engineering detail a tabular
        Q-table never needed (each (inventory, days) pair was just a dict
        key) — part of why DQN requires more careful setup than tabular
        Q-Learning for the same underlying problem.
        """
        obs_t = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        return (obs_t - self.obs_low) / (self.obs_high - self.obs_low + 1e-8)

    def select_action(self, observation: np.ndarray) -> int:
        """Epsilon-greedy action selection, matching QLearningAgent's
        interface and semantics exactly."""
        if self._rng.random() < self.exploration_rate:
            return int(self._rng.integers(0, self.num_actions))
        return self.select_greedy_action(observation)

    def select_greedy_action(self, observation: np.ndarray) -> int:
        """Act purely greedily (no exploration) — used for evaluation.
        Wrapped in `torch.no_grad()` since action selection should never
        build a backprop graph."""
        with torch.no_grad():
            state_t = self._normalize(observation).unsqueeze(0)
            q_values = self.online_network(state_t)
            return int(torch.argmax(q_values, dim=1).item())

    def remember(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        terminated: bool,
    ) -> None:
        """Store one transition in the replay buffer. `.copy()` on the
        arrays guards against the caller's observation being mutated by
        the environment on a later step before this transition is sampled
        for training."""
        self.replay_buffer.push(
            Transition(
                observation.copy(), action, reward, next_observation.copy(), terminated
            )
        )

    def train_step(self) -> Optional[float]:
        """
        Sample a minibatch from replay and perform one gradient step.

        Returns the scalar Huber loss for logging, or `None` if the
        replay buffer doesn't yet hold at least
        `min_replay_size_before_training` transitions — training on a
        near-empty, low-diversity buffer early on is a well-documented
        source of instability, so gradient steps are simply skipped
        during this warm-up window rather than trained on an
        unrepresentative sample.
        """
        if len(self.replay_buffer) < self.min_replay_size_before_training:
            return None

        batch = self.replay_buffer.sample(self.batch_size)
        states = torch.stack([self._normalize(t.state) for t in batch])
        actions = torch.tensor(
            [t.action for t in batch], dtype=torch.int64, device=self.device
        )
        rewards = torch.tensor(
            [t.reward for t in batch], dtype=torch.float32, device=self.device
        )
        next_states = torch.stack([self._normalize(t.next_state) for t in batch])
        dones = torch.tensor(
            [float(t.done) for t in batch], dtype=torch.float32, device=self.device
        )

        current_q = (
            self.online_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        )

        with torch.no_grad():
            next_q_max = self.target_network(next_states).max(dim=1).values
            td_target = rewards + self.discount_factor * next_q_max * (1.0 - dones)

        # Huber loss (smooth L1), not MSE: more robust to the occasional
        # large-magnitude TD error a squared loss would let dominate the
        # gradient, given this environment's reward scale can vary
        # substantially between a routine sale and a large terminal
        # unsold-inventory penalty.
        loss = F.smooth_l1_loss(current_q, td_target)

        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.online_network.parameters(), self.grad_clip_norm
            )
        self.optimizer.step()

        self._gradient_step_count += 1
        if self._gradient_step_count % self.target_update_frequency == 0:
            self.target_network.load_state_dict(self.online_network.state_dict())
            logger.debug(
                "Synced target network at gradient step %d", self._gradient_step_count
            )

        return float(loss.item())

    def decay_exploration(self) -> None:
        """Multiplicatively decay exploration_rate, floored at
        exploration_min. Called once per completed episode — identical
        contract to QLearningAgent.decay_exploration()."""
        self.exploration_rate = max(
            self.exploration_min, self.exploration_rate * self.exploration_decay
        )

    def save(self, path: Path) -> None:
        """
        Persist the online network's weights and hyperparameters to disk.

        Writes two files, mirroring `QLearningAgent.save()`'s pattern:
          - `<path>` : torch state_dict + hyperparameters (torch.save),
            for `load()` to reconstruct a working agent.
          - `<path>.json` : a human-readable sidecar (architecture,
            hyperparameters, timestamp) so a reviewer can sanity-check
            what was trained without loading the torch checkpoint.

        Only the ONLINE network's weights are saved, not the target
        network's — the target network is purely a training-stability
        device; at evaluation/deployment time only the online network's
        learned Q-values matter.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "online_state_dict": self.online_network.state_dict(),
            "hidden_layer_sizes": self.hidden_layer_sizes,
            "num_actions": self.num_actions,
            "obs_shape": tuple(self.obs_shape),
            "learning_rate": self.learning_rate,
            "discount_factor": self.discount_factor,
            "exploration_rate": self.exploration_rate,
            "exploration_min": self.exploration_min,
            "exploration_decay": self.exploration_decay,
        }
        torch.save(payload, path)

        metadata = {
            "hidden_layer_sizes": self.hidden_layer_sizes,
            "num_actions": self.num_actions,
            "learning_rate": self.learning_rate,
            "discount_factor": self.discount_factor,
            "final_exploration_rate": self.exploration_rate,
            "gradient_steps_taken": self._gradient_step_count,
            "replay_buffer_size_at_save": len(self.replay_buffer),
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        with open(str(path) + ".json", "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(
            "Saved DQN policy | gradient_steps=%d | path=%s",
            self._gradient_step_count,
            path,
        )

    @classmethod
    def load(
        cls,
        path: Path,
        observation_space: spaces.Box,
        action_space: spaces.Discrete,
        device: str = "cpu",
        seed: int = 0,
    ) -> "DQNAgent":
        """
        Reconstruct a DQNAgent from a file written by `save()`.

        Raises
        ------
        ValueError
            If the saved policy's action count or observation shape don't
            match the given spaces — loading a network trained against a
            differently-configured environment would silently produce
            nonsensical Q-values, since the network's input/output
            dimensions would no longer correspond to the environment's
            actual observation/action semantics.
        """
        payload = torch.load(path, map_location=device, weights_only=False)

        if payload["num_actions"] != action_space.n:
            raise ValueError(
                f"Saved policy has {payload['num_actions']} actions but the "
                f"given action_space has {action_space.n}. This policy was "
                "trained against a differently-configured environment."
            )
        if tuple(payload["obs_shape"]) != tuple(observation_space.shape):
            raise ValueError(
                f"Saved policy expects observation shape {payload['obs_shape']} "
                f"but the given observation_space has shape "
                f"{observation_space.shape}. This policy was trained against "
                "a differently-configured environment."
            )

        agent = cls(
            observation_space=observation_space,
            action_space=action_space,
            hidden_layer_sizes=payload["hidden_layer_sizes"],
            learning_rate=payload["learning_rate"],
            discount_factor=payload["discount_factor"],
            exploration_rate=payload["exploration_rate"],
            exploration_min=payload["exploration_min"],
            exploration_decay=payload["exploration_decay"],
            batch_size=1,  # irrelevant for a loaded, evaluation-only agent
            replay_buffer_size=1,
            min_replay_size_before_training=1,
            target_update_frequency=1,
            grad_clip_norm=None,
            device=device,
            seed=seed,
        )
        agent.online_network.load_state_dict(payload["online_state_dict"])
        agent.target_network.load_state_dict(payload["online_state_dict"])
        agent.online_network.eval()

        logger.info("Loaded DQN policy | path=%s", path)
        return agent