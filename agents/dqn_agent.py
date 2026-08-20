"""
dqn_agent.py

Full trainable DQN agent -- ties together the three components built in
previous issues:
  - QNetwork (dqn_network.py, Issue #63): the neural network approximating Q-values
  - ReplayBuffer / Transition (this file, Issue #70/#71): stores and samples
    past experiences
  - TargetNetworkManager (target_network.py, Issue #75): stabilizes training targets

This combines them into a complete training loop, matching the standard
DQN algorithm (Mnih et al., 2015):

  1. Choose an action using epsilon-greedy over the ONLINE network's Q-values
  2. Take the action, observe (reward, next_state, done)
  3. Store the experience in the replay buffer
  4. Sample a random mini-batch from the replay buffer
  5. Compute the target: reward + gamma * max(TARGET network's Q-values for next_state)
     (using 0 for the future term if the episode ended)
  6. Compute the loss between the online network's predicted Q-value for
     the action actually taken, and the target
  7. Backpropagate and update the online network's weights
  8. Periodically synchronize the target network to match the online network

Also tracks per-episode average training loss (Issue #84), so learning
progress can be visualized alongside episode reward.

Issue #70 -- "Integrate Experience Replay into the DQN Training Pipeline"
--------------------------------------------------------------------------
`training/train_dqn.py` (and `tests/test_replay_buffer_integration.py`)
drive this agent through three specific entry points:

  - `select_action(observation)`      -- epsilon-greedy action selection
  - `remember(state, action, reward, next_state, done)` -- push one
    `Transition` onto `self.replay_buffer`, defensively copying any
    array-like state so later in-place mutation of the caller's arrays
    can't corrupt what's stored.
  - `train_step()`                    -- one gradient step from a sampled
    mini-batch, or `None` if the buffer hasn't reached
    `min_replay_size_before_training` yet.
"""

from __future__ import annotations

import random
from collections import deque
from typing import List, NamedTuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .dqn_network import QNetwork
from .target_network import TargetNetworkManager


# --------------------------------------------------------------------------- #
# Transition / ReplayBuffer
# --------------------------------------------------------------------------- #
class Transition(NamedTuple):
    """A single stored experience: what happened at one environment step."""

    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """
    Fixed-size buffer that stores past `Transition`s and supports random
    mini-batch sampling for DQN training.

    Once `capacity` is reached, pushing a new transition automatically
    evicts the oldest one (via `collections.deque(maxlen=capacity)`), so
    memory usage never grows unbounded during a long training run.
    """

    def __init__(self, capacity: int, seed: Optional[int] = None):
        self.capacity = capacity
        self._buffer = deque(maxlen=capacity)
        self._rng = random.Random(seed)

    def push(self, transition: Transition) -> None:
        """Store one `Transition` in the buffer."""
        self._buffer.append(transition)

    def sample(self, batch_size: int) -> List[Transition]:
        """
        Randomly sample (without replacement, within this call) a
        mini-batch of `Transition`s currently stored in the buffer.
        """
        if batch_size > len(self._buffer):
            raise ValueError(
                f"Cannot sample batch_size={batch_size}; buffer only "
                f"contains {len(self._buffer)} transitions so far."
            )
        return self._rng.sample(list(self._buffer), batch_size)

    def is_ready(self, batch_size: int) -> bool:
        """Does the buffer have enough transitions to sample a full mini-batch?"""
        return len(self._buffer) >= batch_size

    def __len__(self) -> int:
        return len(self._buffer)


# --------------------------------------------------------------------------- #
# DQN Agent
# --------------------------------------------------------------------------- #
class DQNAgent:
    """A full DQN agent: neural network + replay buffer + target network + training loop."""

    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_layer_sizes=(64, 64),
        learning_rate=0.001,
        discount_factor=0.95,
        exploration_rate=1.0,
        exploration_decay=0.995,
        exploration_min=0.01,
        batch_size=64,
        replay_buffer_size=10000,
        min_replay_size_before_training=1000,
        target_update_frequency=500,
        grad_clip_norm=None,
        seed=None,
        device=None,
        # Backward-compatible aliases for the older constructor kwargs
        # used elsewhere in the codebase (evaluation/*, dashboard/*).
        hidden_dim=None,
        epsilon=None,
        epsilon_decay=None,
        epsilon_min=None,
        buffer_capacity=None,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = discount_factor

        self.epsilon = epsilon if epsilon is not None else exploration_rate
        self.epsilon_decay = epsilon_decay if epsilon_decay is not None else exploration_decay
        self.epsilon_min = epsilon_min if epsilon_min is not None else exploration_min

        self.batch_size = batch_size
        self.min_replay_size_before_training = min_replay_size_before_training
        self.grad_clip_norm = grad_clip_norm

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)

        # QNetwork (agents/dqn_network.py) is a fixed two-hidden-layer MLP;
        # the *first* configured hidden layer size sets the width of both
        # hidden layers (a config with a single-element list, e.g. [8] in
        # tests, or a symmetric list like [64, 64], both work naturally).
        hidden_layer_sizes = list(hidden_layer_sizes) if hidden_layer_sizes else [128]
        resolved_hidden_dim = hidden_dim if hidden_dim is not None else hidden_layer_sizes[0]

        self.q_network = QNetwork(state_dim, action_dim, resolved_hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

        capacity = buffer_capacity if buffer_capacity is not None else replay_buffer_size
        self.replay_buffer = ReplayBuffer(capacity=capacity, seed=seed)
        self.target_manager = TargetNetworkManager(
            self.q_network, update_frequency=target_update_frequency
        )

        self.episode_rewards = []
        self.epsilon_history = []
        self.episode_avg_loss = []  # average training loss per episode (Issue #84)

    # ----------------------------------------------------------------- #
    # Construction from a DQNConfig
    # ----------------------------------------------------------------- #
    @classmethod
    def from_config(cls, config, env) -> "DQNAgent":
        """Build a `DQNAgent` sized to `env`'s observation/action spaces from a `DQNConfig`."""
        return cls(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n,
            hidden_layer_sizes=config.hidden_layer_sizes,
            learning_rate=config.learning_rate,
            discount_factor=config.discount_factor,
            exploration_rate=config.exploration_rate,
            exploration_decay=config.exploration_decay,
            exploration_min=config.exploration_min,
            batch_size=config.batch_size,
            replay_buffer_size=config.replay_buffer_size,
            min_replay_size_before_training=config.min_replay_size_before_training,
            target_update_frequency=config.target_update_frequency,
            grad_clip_norm=config.grad_clip_norm,
            seed=config.seed,
            device=config.device,
        )

    # ----------------------------------------------------------------- #
    # Action selection
    # ----------------------------------------------------------------- #
    def select_action(self, observation, greedy: bool = False) -> int:
        """Epsilon-greedy action selection using the online network."""
        if (not greedy) and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        state_tensor = torch.tensor(
            np.array(observation), dtype=torch.float32, device=self.device
        )
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return int(torch.argmax(q_values).item())

    # Backward-compatible alias (older call sites use `choose_action`).
    choose_action = select_action

    def select_greedy_action(self, observation) -> int:
        """
        Pure-greedy action selection (no exploration), matching the
        `select_greedy_action(observation) -> int` signature every policy
        in `evaluation/evaluate_policies.py`'s `Policy` protocol must
        implement (see `QLearningAgent.select_greedy_action` and the
        `baselines/` classes for the same contract).
        """
        return self.select_action(observation, greedy=True)

    # ----------------------------------------------------------------- #
    # Experience storage
    # ----------------------------------------------------------------- #
    def remember(self, state, action, reward, next_state, done) -> None:
        """
        Store one experience in the replay buffer as a `Transition`.

        States are defensively copied (`np.array(...).copy()`) so that any
        later in-place mutation of the caller's `state`/`next_state`
        arrays (e.g. the environment reusing an internal buffer) can never
        corrupt what was actually stored.
        """
        transition = Transition(
            state=np.array(state, dtype=np.float32).copy(),
            action=int(action),
            reward=float(reward),
            next_state=np.array(next_state, dtype=np.float32).copy(),
            done=bool(done),
        )
        self.replay_buffer.push(transition)

    # ----------------------------------------------------------------- #
    # Learning
    # ----------------------------------------------------------------- #
    def train_step(self) -> Optional[float]:
        """
        Perform one gradient descent update using a sampled mini-batch.

        Returns `None` (a no-op) until the replay buffer has accumulated
        at least `min_replay_size_before_training` transitions -- training
        on a near-empty, low-diversity buffer is a well-documented source
        of early instability.
        """
        if len(self.replay_buffer) < self.min_replay_size_before_training:
            return None

        batch = self.replay_buffer.sample(self.batch_size)

        states = torch.tensor(
            np.array([t.state for t in batch]), dtype=torch.float32, device=self.device
        )
        actions = torch.tensor([t.action for t in batch], dtype=torch.int64, device=self.device)
        rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32, device=self.device)
        next_states = torch.tensor(
            np.array([t.next_state for t in batch]), dtype=torch.float32, device=self.device
        )
        dones = torch.tensor([float(t.done) for t in batch], dtype=torch.float32, device=self.device)

        q_values = self.q_network(states)
        predicted_q = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        target_q_values = self.target_manager.get_target_q_values(next_states)
        max_next_q = torch.max(target_q_values, dim=1)[0]
        target = rewards + self.gamma * max_next_q * (1 - dones)

        loss = self.loss_fn(predicted_q, target.detach())

        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(self.q_network.parameters(), self.grad_clip_norm)
        self.optimizer.step()

        self.target_manager.step()

        return float(loss.item())

    # Backward-compatible alias (older internal name).
    _train_step = train_step

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # Alias matching QLearningAgent's naming, for callers that treat both
    # agent types interchangeably.
    decay_exploration = decay_epsilon

    # ----------------------------------------------------------------- #
    # Older convenience API (train / evaluate a full episode loop
    # directly), kept for backward compatibility with evaluation and
    # dashboard scripts that construct a `DQNAgent` and call these
    # directly instead of going through `training/train_dqn.py`.
    # ----------------------------------------------------------------- #
    def train(self, env, num_episodes=1000, verbose=True):
        """Train the DQN agent for a number of episodes."""
        for episode in range(1, num_episodes + 1):
            observation, info = env.reset()
            done = False
            total_reward = 0
            episode_losses = []

            while not done:
                action = self.select_action(observation)
                next_observation, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                self.remember(observation, action, reward, next_observation, done)
                loss = self.train_step()
                if loss is not None:
                    episode_losses.append(loss)

                observation = next_observation
                total_reward += reward

            self.decay_epsilon()
            self.episode_rewards.append(total_reward)
            self.epsilon_history.append(self.epsilon)

            avg_loss_this_episode = float(np.mean(episode_losses)) if episode_losses else 0.0
            self.episode_avg_loss.append(avg_loss_this_episode)

            if verbose and episode % 100 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                avg_loss = np.mean(self.episode_avg_loss[-100:])
                print(f"Episode {episode}/{num_episodes} | "
                      f"Avg reward (last 100): {avg_reward:.2f} | "
                      f"Avg loss (last 100): {avg_loss:.4f} | "
                      f"Epsilon: {self.epsilon:.3f}")

        return self.episode_rewards

    def evaluate(self, env, num_episodes=100):
        """Evaluate the trained agent with NO exploration (pure greedy)."""
        total_rewards = []
        total_revenues = []
        utilizations = []

        for _ in range(num_episodes):
            observation, info = env.reset()
            done = False
            episode_reward = 0
            episode_revenue = 0
            initial_inventory = env.config.initial_inventory

            while not done:
                action = self.select_action(observation, greedy=True)
                next_observation, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                episode_reward += reward
                episode_revenue += info["reward_breakdown"]["revenue"]
                observation = next_observation

            remaining_inventory = observation[0]
            utilization = 1.0 - (remaining_inventory / initial_inventory)

            total_rewards.append(episode_reward)
            total_revenues.append(episode_revenue)
            utilizations.append(utilization)

        return {
            "avg_reward": float(np.mean(total_rewards)),
            "avg_revenue": float(np.mean(total_revenues)),
            "avg_inventory_utilization": float(np.mean(utilizations)),
        }

    def save_model(self, filepath="agents/dqn_model.pt"):
        torch.save(self.q_network.state_dict(), filepath)
        print(f"Model weights saved to {filepath}")

    def load_model(self, filepath="agents/dqn_model.pt"):
        self.q_network.load_state_dict(torch.load(filepath, map_location=self.device))
        self.q_network.eval()
        print(f"Model weights loaded from {filepath}")

    @classmethod
    def load(
        cls,
        filepath,
        observation_space=None,
        action_space=None,
        state_dim=None,
        action_dim=None,
        hidden_layer_sizes=(64, 64),
        device="cpu",
        seed=None,
    ) -> "DQNAgent":
        """
        Reconstruct a trained `DQNAgent` from a `save_model()` checkpoint
        (a plain `state_dict`, not the full agent object).

        Because only the network weights are persisted, the caller must
        supply the same `state_dim`/`action_dim` (or `observation_space`/
        `action_space`, from which those are derived) and
        `hidden_layer_sizes` the checkpoint was trained with --
        `torch.load_state_dict` fails loudly with a shape mismatch if
        they don't match, rather than silently loading garbage.

        `exploration_rate` is forced to 0 (pure greedy): a reloaded
        checkpoint is evaluated, never trained further, by every current
        caller (`evaluation/evaluate_policies.py`), so accidental random
        exploration during evaluation is a bug, not a feature.
        """
        resolved_state_dim = state_dim if state_dim is not None else (
            observation_space.shape[0] if observation_space is not None else None
        )
        resolved_action_dim = action_dim if action_dim is not None else (
            action_space.n if action_space is not None else None
        )
        if resolved_state_dim is None or resolved_action_dim is None:
            raise ValueError(
                "DQNAgent.load() requires either (state_dim, action_dim) or "
                "(observation_space, action_space) to reconstruct the network."
            )

        agent = cls(
            state_dim=resolved_state_dim,
            action_dim=resolved_action_dim,
            hidden_layer_sizes=hidden_layer_sizes,
            exploration_rate=0.0,
            exploration_min=0.0,
            seed=seed,
            device=device,
        )
        agent.load_model(str(filepath))
        return agent


if __name__ == "__main__":
    import sys
    import os

    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

    from pricing_env.pricing_env import PricingEnvironment, PricingEnvConfig

    env = PricingEnvironment(PricingEnvConfig())

    agent = DQNAgent(state_dim=2, action_dim=env.action_space.n)

    print("Training DQN agent...")
    agent.train(env, num_episodes=500)

    print("\nEvaluating trained DQN agent...")
    metrics = agent.evaluate(env, num_episodes=100)
    print("Evaluation metrics:", metrics)

    agent.save_model("agents/dqn_model.pt")