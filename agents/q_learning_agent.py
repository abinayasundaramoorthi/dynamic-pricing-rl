"""
q_learning_agent.py

Baseline Q-Learning agent for the Dynamic Pricing RL project.

This agent learns a pricing policy by maintaining a Q-table: a lookup
table that stores a "quality score" (Q-value) for every (state, action)
combination it has seen. Over time, through trial and error, it learns
which price-adjustment action is best for each situation (state).

This is the simple "notebook" style agent - a baseline before we move
to DQN (Deep Q-Network) in later weeks.

Compatible with the real `pricing_env.PricingEnvironment`, which uses the
modern Gymnasium 5-tuple API:
    reset(seed=...) -> (observation, info)
    step(action)    -> (observation, reward, terminated, truncated, info)
"""

import random
import pickle
from pathlib import Path

import numpy as np
from collections import defaultdict


class QLearningAgent:
    """A simple tabular Q-Learning agent."""

    def __init__(self, num_actions, learning_rate=0.1, discount_factor=0.95,
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.num_actions = num_actions
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.q_table = defaultdict(lambda: np.zeros(self.num_actions))

    def choose_action(self, state):
        """
        Epsilon-greedy action choice. `state` is rounded before being used
        as a dict key - the environment's observation is float32
        (e.g. [87.0, 14.0]), and rounding keeps the Q-table's effective
        state space finite and matches the way whole units/days are
        actually incremented, rather than treating every tiny floating
        point variation as a brand-new, never-seen-before state.
        """
        state = tuple(round(float(x)) for x in state)

        if random.random() < self.epsilon:
            action = random.randint(0, self.num_actions - 1)
        else:
            action = int(np.argmax(self.q_table[state]))

        return action

    def select_greedy_action(self, state):
        """
        Chooses the best known action for `state` with NO exploration
        (epsilon is ignored entirely). Used for evaluating a trained
        policy - e.g. after reloading from disk via `load()` - where we
        want to see the agent's best learned behaviour, not continued
        random exploration.
        """
        state = tuple(round(float(x)) for x in state)
        return int(np.argmax(self.q_table[state]))

    def update(self, state, action, reward, next_state, done):
        state = tuple(round(float(x)) for x in state)
        next_state = tuple(round(float(x)) for x in next_state)

        current_q = self.q_table[state][action]

        if done:
            target = reward
        else:
            best_next_q = np.max(self.q_table[next_state])
            target = reward + self.gamma * best_next_q

        self.q_table[state][action] = current_q + self.alpha * (target - current_q)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, filepath):
        """
        Saves the agent's learned Q-table and hyperparameters to a .pkl
        file, creating parent directories if needed.

        The Q-table is converted from a defaultdict to a plain dict
        before pickling - defaultdict's lambda default_factory cannot be
        pickled directly ("can't pickle local object"), so we save the
        plain dict of everything actually learned so far, and rebuild the
        defaultdict wrapper (with a fresh, equivalent lambda) on load().
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "q_table": dict(self.q_table),
            "num_actions": self.num_actions,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_decay": self.epsilon_decay,
            "epsilon_min": self.epsilon_min,
        }

        with open(filepath, "wb") as f:
            pickle.dump(data, f)

        return filepath

    @classmethod
    def load(cls, filepath, action_space=None, seed=None):
        """
        Loads a previously-saved agent from a .pkl file.

        Parameters
        ----------
        filepath : str or Path
            Path to the .pkl file written by `save()`.
        action_space : gymnasium.spaces.Discrete, optional
            If given, `action_space.n` is used for num_actions instead of
            the value stored in the file - lets a caller reconstruct an
            agent against a specific environment's action space rather
            than blindly trusting whatever was saved.
        seed : int, optional
            If given, reseeds Python's `random` module (used by
            `choose_action`'s exploration draw) for reproducible
            evaluation runs.

        Returns
        -------
        QLearningAgent
            A new agent instance with the saved Q-table restored.
        """
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        num_actions = action_space.n if action_space is not None else data["num_actions"]

        agent = cls(
            num_actions=num_actions,
            learning_rate=data["alpha"],
            discount_factor=data["gamma"],
            epsilon=data["epsilon"],
            epsilon_decay=data["epsilon_decay"],
            epsilon_min=data["epsilon_min"],
        )
        agent.q_table = defaultdict(lambda: np.zeros(num_actions), data["q_table"])

        if seed is not None:
            random.seed(seed)

        return agent

    def train(self, env, num_episodes=1000, seed=42, verbose=True):
        """
        Trains the agent for num_episodes full seasons.

        Tracks REWARD and REVENUE separately, since they are not the same
        thing in this environment - reward.py's compute_reward() subtracts
        discount/unsold penalties and adds a pacing bonus on top of raw
        revenue, so the RL training signal and the business dollar figure
        genuinely differ here.

        Returns
        -------
        (episode_rewards, episode_revenues) : tuple of lists
        """
        episode_rewards = []
        episode_revenues = []

        for episode in range(1, num_episodes + 1):
            state, info = env.reset(seed=seed + episode)
            terminated = truncated = False
            total_reward = 0.0

            while not (terminated or truncated):
                action = self.choose_action(state)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                self.update(state, action, reward, next_state, done)

                state = next_state
                total_reward += reward

            self.decay_epsilon()
            episode_rewards.append(total_reward)
            episode_revenues.append(info["episode_revenue"])

            if verbose and episode % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                avg_revenue = np.mean(episode_revenues[-100:])
                print(f"Episode {episode}/{num_episodes} | "
                      f"Avg reward (last 100): {avg_reward:.2f} | "
                      f"Avg revenue (last 100): ₹{avg_revenue:.2f} | "
                      f"Epsilon: {self.epsilon:.3f}")

        return episode_rewards, episode_revenues


if __name__ == "__main__":
    import sys
    import os

    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

    from pricing_env import PricingEnvironment, PricingEnvConfig

    config = PricingEnvConfig()
    env = PricingEnvironment(config)
    agent = QLearningAgent(num_actions=env.action_space.n)

    print("Training Q-Learning agent...")
    rewards, revenues = agent.train(env, num_episodes=1000)

    print("\nTraining complete!")
    print(f"Average reward in last 100 episodes: {np.mean(rewards[-100:]):.2f}")
    print(f"Average revenue in last 100 episodes: ₹{np.mean(revenues[-100:]):.2f}")
    print(f"Number of unique states learned: {len(agent.q_table)}")

    # --- Save, then reload into a FRESH agent, to prove the round-trip works ---
    save_path = os.path.join(os.path.dirname(__file__), "saved_policy.pkl")
    agent.save(save_path)
    print(f"\nSaved Q-table to: {save_path}")

    reloaded_agent = QLearningAgent.load(save_path, action_space=env.action_space, seed=42)
    print(f"Reloaded agent has {len(reloaded_agent.q_table)} states "
          f"(matches original: {len(reloaded_agent.q_table) == len(agent.q_table)})")

    # Evaluate the RELOADED agent greedily (no exploration) to confirm it
    # actually produces sensible behaviour after being read back from disk,
    # not just that the file was written successfully.
    eval_rewards, eval_revenues = [], []
    for ep in range(100):
        state, info = env.reset(seed=500_000 + ep)
        terminated = truncated = False
        total_reward = 0.0
        while not (terminated or truncated):
            action = reloaded_agent.select_greedy_action(state)
            state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
        eval_rewards.append(total_reward)
        eval_revenues.append(info["episode_revenue"])

    print(f"\nReloaded agent evaluation (100 episodes, greedy):")
    print(f"  Avg reward:  {np.mean(eval_rewards):.2f}")
    print(f"  Avg revenue: ₹{np.mean(eval_revenues):.2f}")