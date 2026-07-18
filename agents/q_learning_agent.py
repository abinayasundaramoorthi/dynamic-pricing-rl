"""
q_learning_agent.py

Baseline Q-Learning agent for the Dynamic Pricing RL project.

This agent learns a pricing policy by maintaining a Q-table: a lookup
table that stores a "quality score" (Q-value) for every (state, action)
combination it has seen. Over time, through trial and error, it learns
which price (action) is best for each situation (state).

Built to work with the team's Gymnasium-style PricingEnvironment
(pricing_env.py):
  - env.reset(seed=...) returns (observation, info)
  - env.step(action) returns (observation, reward, terminated, truncated, info)
  - env.action_space.n gives the number of discrete actions

Features:
  - Policy extraction, save/load, performance tracking
  - evaluate(): honest, no-exploration performance testing
  - IMPROVED EXPLORATION (Issue #51):
      * Exploration warm-up period — epsilon stays high for a fixed
        number of episodes before decay begins, so the agent gathers
        broad experience before starting to commit to a strategy
        (avoids getting stuck in a poor "local" pricing habit too early).
      * Boltzmann (softmax) action selection as an alternative to plain
        uniform-random exploration — when exploring, the agent still
        leans toward actions with higher known Q-values instead of
        wasting exploration on actions it already has strong evidence
        are bad (e.g. absurdly low or high prices).
"""

import random
import pickle
import numpy as np
from collections import defaultdict


class QLearningAgent:
    """A tabular Q-Learning agent with policy management and improved exploration."""

    def __init__(self, num_actions, learning_rate=0.1, discount_factor=0.95,
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01,
                 exploration_strategy="epsilon_greedy", warmup_episodes=0,
                 temperature=1.0, temperature_decay=0.995, temperature_min=0.05):
        """
        Parameters
        ----------
        num_actions : int
            Number of possible actions (price levels). Get from env.action_space.n
        learning_rate : float (alpha)
        discount_factor : float (gamma)
        epsilon : float
            Starting exploration probability.
        epsilon_decay : float
            Multiplicative decay applied each episode, once warm-up ends.
        epsilon_min : float
            Floor for epsilon — always keeps a little exploration.
        exploration_strategy : str
            "epsilon_greedy"          -> original: explore = pick a fully random action
            "boltzmann_epsilon_greedy" -> improved: explore = pick an action using
                                           softmax over current Q-values, so
                                           exploration still favors more promising
                                           actions instead of being fully blind.
        warmup_episodes : int
            Number of episodes to hold epsilon at its starting value before
            decay begins (exploration scheduling). 0 disables warm-up
            (original behavior — decay starts immediately).
        temperature : float
            Starting softmax temperature for Boltzmann exploration. Higher
            temperature = closer to uniform random; lower = more strongly
            favors the best-known action. Only used when
            exploration_strategy="boltzmann_epsilon_greedy".
        temperature_decay : float
            Multiplicative decay applied to temperature each episode
            (after warm-up), same idea as epsilon_decay.
        temperature_min : float
            Floor for temperature.
        """
        self.num_actions = num_actions
        self.alpha = learning_rate
        self.gamma = discount_factor

        self.epsilon = epsilon
        self.epsilon_start = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.exploration_strategy = exploration_strategy
        self.warmup_episodes = warmup_episodes
        self._episodes_seen = 0

        self.temperature = temperature
        self.temperature_start = temperature
        self.temperature_decay = temperature_decay
        self.temperature_min = temperature_min

        self.q_table = defaultdict(lambda: np.zeros(self.num_actions))

        self.episode_rewards = []
        self.epsilon_history = []
        self.episode_lengths = []

    def _discretize_state(self, observation):
        """Convert a NumPy observation into a clean integer tuple for the Q-table."""
        return tuple(int(round(x)) for x in observation)

    # ------------------------------------------------------------
    # ACTION SELECTION
    # ------------------------------------------------------------
    def _softmax_action(self, q_values):
        """
        Pick an action using Boltzmann (softmax) exploration: actions
        with higher Q-values are more likely to be picked, but it's
        still probabilistic (not purely greedy), so the agent keeps
        exploring — just more intelligently than pure random choice.
        """
        temperature = max(self.temperature, 1e-6)  # avoid divide-by-zero
        scaled = q_values / temperature
        scaled = scaled - np.max(scaled)  # numerical stability
        exp_values = np.exp(scaled)
        probabilities = exp_values / np.sum(exp_values)
        return int(np.random.choice(self.num_actions, p=probabilities))

    def choose_action(self, observation, greedy=False):
        """
        Choose an action.

        Parameters
        ----------
        greedy : bool
            If True, always pick the best known action (no exploration at
            all) — used by evaluate() to test the agent's learned policy
            honestly, regardless of exploration_strategy.
        """
        state = self._discretize_state(observation)
        q_values = self.q_table[state]

        if greedy:
            return int(np.argmax(q_values))

        explore = random.random() < self.epsilon

        if not explore:
            return int(np.argmax(q_values))

        # --- Exploring: HOW we explore depends on the strategy ---
        if self.exploration_strategy == "boltzmann_epsilon_greedy":
            return self._softmax_action(q_values)
        else:
            # original strategy: fully random action
            return random.randint(0, self.num_actions - 1)

    # ------------------------------------------------------------
    # Q-VALUE UPDATE
    # ------------------------------------------------------------
    def update(self, observation, action, reward, next_observation, done):
        """
        Q(s,a) = Q(s,a) + alpha * [reward + gamma * max(Q(s',a')) - Q(s,a)]
        """
        state = self._discretize_state(observation)
        next_state = self._discretize_state(next_observation)

        current_q = self.q_table[state][action]

        if done:
            target = reward
        else:
            best_next_q = np.max(self.q_table[next_state])
            target = reward + self.gamma * best_next_q

        self.q_table[state][action] = current_q + self.alpha * (target - current_q)

    # ------------------------------------------------------------
    # EXPLORATION SCHEDULING (warm-up + decay)
    # ------------------------------------------------------------
    def decay_epsilon(self):
        """
        Reduce epsilon (and temperature, if using Boltzmann exploration)
        after each episode.

        If warmup_episodes > 0, epsilon (and temperature) stay at their
        starting values for that many episodes first — giving the agent
        a guaranteed period of broad exploration before it starts
        narrowing down, which helps avoid settling into a poor "local"
        pricing habit too early, before it has seen enough of the
        situation space.
        """
        self._episodes_seen += 1

        if self._episodes_seen <= self.warmup_episodes:
            return  # still in warm-up — don't decay yet

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        if self.exploration_strategy == "boltzmann_epsilon_greedy":
            self.temperature = max(
                self.temperature_min, self.temperature * self.temperature_decay
            )

    # ------------------------------------------------------------
    # POLICY EXTRACTION
    # ------------------------------------------------------------
    def extract_policy(self):
        """Convert the learned Q-table into a simple state -> best action policy."""
        policy = {}
        for state, q_values in self.q_table.items():
            policy[state] = int(np.argmax(q_values))
        return policy

    # ------------------------------------------------------------
    # SAVE / LOAD Q-TABLE
    # ------------------------------------------------------------
    def save_policy(self, filepath="agents/saved_policy.pkl"):
        """Save the learned Q-table (and key agent settings) to disk."""
        data_to_save = {
            "q_table": dict(self.q_table),
            "num_actions": self.num_actions,
            "epsilon": self.epsilon,
            "alpha": self.alpha,
            "gamma": self.gamma,
        }

        with open(filepath, "wb") as f:
            pickle.dump(data_to_save, f)

        print(f"Policy saved to {filepath}")

    def load_policy(self, filepath="agents/saved_policy.pkl"):
        """Load a previously saved Q-table (and settings) back into this agent."""
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        self.q_table = defaultdict(lambda: np.zeros(self.num_actions))
        self.q_table.update(data["q_table"])

        self.num_actions = data["num_actions"]
        self.epsilon = data["epsilon"]
        self.alpha = data["alpha"]
        self.gamma = data["gamma"]

        print(f"Policy loaded from {filepath}")
        print(f"Loaded {len(self.q_table)} learned states.")

    # ------------------------------------------------------------
    # LEARNING LOOP
    # ------------------------------------------------------------
    def train(self, env, num_episodes=1000, verbose=True):
        """Train the agent by running it through many episodes."""
        for episode in range(1, num_episodes + 1):
            observation, info = env.reset()
            done = False
            total_reward = 0
            steps = 0

            while not done:
                action = self.choose_action(observation)
                next_observation, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                self.update(observation, action, reward, next_observation, done)

                observation = next_observation
                total_reward += reward
                steps += 1

            self.decay_epsilon()

            self.episode_rewards.append(total_reward)
            self.epsilon_history.append(self.epsilon)
            self.episode_lengths.append(steps)

            if verbose and episode % 100 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                print(f"Episode {episode}/{num_episodes} | "
                      f"Avg reward (last 100): {avg_reward:.2f} | "
                      f"Epsilon: {self.epsilon:.3f}")

        return self.episode_rewards

    # ------------------------------------------------------------
    # EVALUATION (no exploration — honest performance test)
    # ------------------------------------------------------------
    def evaluate(self, env, num_episodes=100):
        """
        Test the agent's learned policy with NO exploration (pure greedy
        actions), for fair comparison between agents/configurations.

        Returns
        -------
        metrics : dict
            avg_reward, avg_revenue, avg_inventory_utilization
        """
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
                action = self.choose_action(observation, greedy=True)
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

        metrics = {
            "avg_reward": float(np.mean(total_rewards)),
            "avg_revenue": float(np.mean(total_revenues)),
            "avg_inventory_utilization": float(np.mean(utilizations)),
        }
        return metrics

    def get_performance_summary(self):
        """Return a simple summary of training performance so far."""
        if not self.episode_rewards:
            return "No training data yet — call train() first."

        summary = {
            "total_episodes": len(self.episode_rewards),
            "avg_reward_first_100": np.mean(self.episode_rewards[:100]),
            "avg_reward_last_100": np.mean(self.episode_rewards[-100:]),
            "best_episode_reward": np.max(self.episode_rewards),
            "final_epsilon": self.epsilon,
            "states_learned": len(self.q_table),
        }
        return summary


if __name__ == "__main__":
    import sys
    import os

    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

    from pricing_env.pricing_env import PricingEnvironment, PricingEnvConfig

    env = PricingEnvironment(PricingEnvConfig())

    print("Training IMPROVED exploration agent (Boltzmann + warm-up)...")
    agent = QLearningAgent(
        num_actions=env.action_space.n,
        exploration_strategy="boltzmann_epsilon_greedy",
        warmup_episodes=200,
    )
    agent.train(env, num_episodes=1000)

    print("\nTraining complete!")
    print("Performance summary:", agent.get_performance_summary())

    policy = agent.extract_policy()
    print(f"\nExtracted policy covers {len(policy)} states.")

    agent.save_policy("agents/saved_policy.pkl")

    print("\nEvaluating trained agent...")
    eval_metrics = agent.evaluate(env, num_episodes=100)
    print("Evaluation metrics:", eval_metrics)