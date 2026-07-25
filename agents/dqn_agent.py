"""
dqn_agent.py

Full trainable DQN agent — ties together the three components built in
previous issues:
  - QNetwork (dqn_network.py, Issue #63): the neural network approximating Q-values
  - ReplayBuffer (replay_buffer.py, Issue #71): stores and samples past experiences
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
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .dqn_network import QNetwork
from .replay_buffer import ReplayBuffer
from .target_network import TargetNetworkManager


class DQNAgent:
    """A full DQN agent: neural network + replay buffer + target network + training loop."""

    def __init__(self, state_dim, action_dim, hidden_dim=128,
                 learning_rate=0.001, discount_factor=0.95,
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01,
                 buffer_capacity=10000, batch_size=64,
                 target_update_frequency=500, device=None):
        """
        Parameters
        ----------
        state_dim : int
        action_dim : int
        hidden_dim : int
        learning_rate : float
            Step size for the Adam optimizer updating the online network.
        discount_factor : float (gamma)
        epsilon, epsilon_decay, epsilon_min : float
            Same epsilon-greedy exploration scheme as QLearningAgent.
        buffer_capacity : int
            Max size of the replay buffer.
        batch_size : int
            Number of experiences sampled per training step.
        target_update_frequency : int
            Training steps between target network synchronizations.
        device : str, optional
            "cpu" or "cuda". Auto-detected if not given.
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.q_network = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)
        self.target_manager = TargetNetworkManager(
            self.q_network, update_frequency=target_update_frequency
        )

        self.episode_rewards = []
        self.epsilon_history = []

    def choose_action(self, observation, greedy=False):
        """Epsilon-greedy action selection using the online network."""
        if (not greedy) and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        state_tensor = torch.tensor(
            np.array(observation), dtype=torch.float32, device=self.device
        )
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return int(torch.argmax(q_values).item())

    def _train_step(self):
        """Perform one gradient descent update using a sampled mini-batch."""
        if not self.replay_buffer.is_ready(self.batch_size):
            return None  # not enough experiences yet

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.int64, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

        # Current Q-value predictions for the actions actually taken
        q_values = self.q_network(states)
        predicted_q = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target: reward + gamma * max Q(next_state) from the TARGET network
        # (not the online network -- this is the whole point of Issue #75).
        # If the episode ended (done=1), there's no future reward to add.
        target_q_values = self.target_manager.get_target_q_values(next_states)
        max_next_q = torch.max(target_q_values, dim=1)[0]
        target = rewards + self.gamma * max_next_q * (1 - dones)

        loss = self.loss_fn(predicted_q, target.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.target_manager.step()  # may or may not trigger a sync this step

        return loss.item()

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def train(self, env, num_episodes=1000, verbose=True):
        """Train the DQN agent for a number of episodes."""
        for episode in range(1, num_episodes + 1):
            observation, info = env.reset()
            done = False
            total_reward = 0

            while not done:
                action = self.choose_action(observation)
                next_observation, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                self.replay_buffer.push(observation, action, reward, next_observation, done)
                self._train_step()

                observation = next_observation
                total_reward += reward

            self.decay_epsilon()
            self.episode_rewards.append(total_reward)
            self.epsilon_history.append(self.epsilon)

            if verbose and episode % 100 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                print(f"Episode {episode}/{num_episodes} | "
                      f"Avg reward (last 100): {avg_reward:.2f} | "
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