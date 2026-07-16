"""
q_learning_agent.py

Baseline Q-Learning agent for the Dynamic Pricing RL project.

This agent learns a pricing policy by maintaining a Q-table: a lookup
table that stores a "quality score" (Q-value) for every (state, action)
combination it has seen. Over time, through trial and error, it learns
which price (action) is best for each situation (state).

This is the simple "notebook" style agent — a baseline before we move
to DQN (Deep Q-Network) in later weeks.
"""

import random
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
        state = tuple(state)

        if random.random() < self.epsilon:
            action = random.randint(0, self.num_actions - 1)
        else:
            action = int(np.argmax(self.q_table[state]))

        return action

    def update(self, state, action, reward, next_state, done):
        state = tuple(state)
        next_state = tuple(next_state)

        current_q = self.q_table[state][action]

        if done:
            target = reward
        else:
            best_next_q = np.max(self.q_table[next_state])
            target = reward + self.gamma * best_next_q

        self.q_table[state][action] = current_q + self.alpha * (target - current_q)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def train(self, env, num_episodes=1000, verbose=True):
        episode_rewards = []

        for episode in range(1, num_episodes + 1):
            state = env.reset()
            done = False
            total_reward = 0

            while not done:
                action = self.choose_action(state)
                next_state, reward, done, info = env.step(action)

                self.update(state, action, reward, next_state, done)

                state = next_state
                total_reward += reward

            self.decay_epsilon()
            episode_rewards.append(total_reward)

            if verbose and episode % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                print(f"Episode {episode}/{num_episodes} | "
                      f"Avg revenue (last 100): {avg_reward:.2f} | "
                      f"Epsilon: {self.epsilon:.3f}")

        return episode_rewards


if __name__ == "__main__":
    import sys
    import os

    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'pricing_env'))

    from pricing_env import PricingEnv

    env = PricingEnv(initial_inventory=100, total_days=30)
    agent = QLearningAgent(num_actions=len(env.price_levels))

    print("Training Q-Learning agent...")
    rewards = agent.train(env, num_episodes=1000)

    print("\nTraining complete!")
    print(f"Average revenue in last 100 episodes: {np.mean(rewards[-100:]):.2f}")
    print(f"Number of unique states learned: {len(agent.q_table)}")