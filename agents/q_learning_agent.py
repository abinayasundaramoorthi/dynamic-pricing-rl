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
  - Configurable epsilon decay strategy
  - Policy extraction (turning the Q-table into a clear "best action per state" policy)
  - Saving/loading the learned Q-table (so training progress isn't lost)
  - Performance tracking (so we can plot how the agent improved over time)
  - evaluate(): test a trained agent's real performance with NO exploration,
    used for comparing different hyperparameter settings fairly (Issue #47)
"""

import random
import pickle
import numpy as np
from collections import defaultdict


class QLearningAgent:
    """A simple tabular Q-Learning agent with policy management."""

    def __init__(self, num_actions, learning_rate=0.1, discount_factor=0.95,
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.num_actions = num_actions
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_start = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.q_table = defaultdict(lambda: np.zeros(self.num_actions))

        self.episode_rewards = []
        self.epsilon_history = []
        self.episode_lengths = []

    def _discretize_state(self, observation):
        """Convert a NumPy observation into a clean integer tuple for the Q-table."""
        return tuple(int(round(x)) for x in observation)

    def choose_action(self, observation, greedy=False):
        """
        Choose an action using the epsilon-greedy strategy.

        Parameters
        ----------
        greedy : bool
            If True, always pick the best known action (no random exploration).
            Used during evaluate() to test the agent's learned policy honestly.
        """
        state = self._discretize_state(observation)

        if (not greedy) and random.random() < self.epsilon:
            action = random.randint(0, self.num_actions - 1)
        else:
            action = int(np.argmax(self.q_table[state]))

        return action

    def update(self, observation, action, reward, next_observation, done):
        """
        Update the Q-table using the Q-Learning update rule:
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

    def decay_epsilon(self):
        """Reduce epsilon after each episode using exponential decay."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def extract_policy(self):
        """Convert the learned Q-table into a simple state -> best action policy."""
        policy = {}
        for state, q_values in self.q_table.items():
            policy[state] = int(np.argmax(q_values))
        return policy

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

    def evaluate(self, env, num_episodes=100):
        """
        Test the agent's learned policy with NO exploration (pure greedy
        actions), so we get an honest measure of how good the agent
        actually is — used to fairly compare different hyperparameter
        settings against each other (Issue #47).

        Parameters
        ----------
        env : PricingEnvironment
        num_episodes : int
            Number of episodes to evaluate over.

        Returns
        -------
        metrics : dict
            avg_reward, avg_revenue, avg_inventory_utilization (0 to 1,
            fraction of starting inventory actually sold).
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
                # Pull actual revenue out of the reward breakdown, since
                # `reward` itself includes penalties/bonuses, not pure revenue.
                episode_revenue += info["reward_breakdown"]["revenue"]

                observation = next_observation

            # Inventory utilization: fraction of starting inventory actually sold
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
    agent = QLearningAgent(num_actions=env.action_space.n)

    print("Training Q-Learning agent...")
    agent.train(env, num_episodes=1000)

    print("\nTraining complete!")
    summary = agent.get_performance_summary()
    print("Performance summary:", summary)

    policy = agent.extract_policy()
    print(f"\nExtracted policy covers {len(policy)} states.")

    agent.save_policy("agents/saved_policy.pkl")

    new_agent = QLearningAgent(num_actions=env.action_space.n)
    new_agent.load_policy("agents/saved_policy.pkl")

    test_observation = np.array([50, 15], dtype=np.float32)
    action = new_agent.choose_action(test_observation)
    print(f"\nAction chosen by loaded agent for state {test_observation}: {action}")

    print("\nEvaluating trained agent...")
    eval_metrics = agent.evaluate(env, num_episodes=100)
    print("Evaluation metrics:", eval_metrics)