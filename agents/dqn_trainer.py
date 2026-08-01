"""
dqn_trainer.py

Mini-Batch DQN Trainer

This module performs one Deep Q-Network training step using
experiences sampled from the replay buffer.

Implemented:
- Mini-batch sampling
- Predicted Q-value computation
- Target Q-value computation
- Loss calculation
- Backpropagation
- Online network update
"""
import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys
sys.path.append(os.path.dirname(__file__))


class DQNTrainer:
    """
    Handles one DQN training step.
    """

    def __init__(
        self,
        online_network,
        target_network,
        replay_buffer,
        learning_rate=1e-3,
        gamma=0.99,
        batch_size=64,
        device="cpu",
    ):

        self.online_network = online_network
        self.target_network = target_network
        self.replay_buffer = replay_buffer

        self.gamma = gamma
        self.batch_size = batch_size
        self.device = device

        self.loss_function = nn.MSELoss()

        self.optimizer = optim.Adam(
            self.online_network.parameters(),
            lr=learning_rate,
        )

    def train_step(self):
        """
        Perform one mini-batch DQN update.

        Returns
        -------
        float
            Training loss.
        """

        if len(self.replay_buffer) < self.batch_size:
            return None

        (
            states,
            actions,
            rewards,
            next_states,
            dones,
        ) = self.replay_buffer.sample(self.batch_size)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # -----------------------------
        # Current Q-values
        # -----------------------------
        current_q = self.online_network(states)

        predicted_q = current_q.gather(1, actions)

        # -----------------------------
        # Target Q-values
        # -----------------------------
        with torch.no_grad():

            next_q = self.target_network(next_states)

            max_next_q = next_q.max(dim=1, keepdim=True)[0]

            target_q = rewards + (
                self.gamma * max_next_q * (1 - dones)
            )

        # -----------------------------
        # Compute Loss
        # -----------------------------
        loss = self.loss_function(
            predicted_q,
            target_q,
        )

        # -----------------------------
        # Backpropagation
        # -----------------------------
        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()

        return loss.item()

if __name__ == "__main__":

    from replay_buffer import ReplayBuffer
    from dqn_network import QNetwork
    from target_network import TargetNetworkManager

    device = "cuda" if torch.cuda.is_available() else "cpu"

    state_dim = 2
    action_dim = 7

    online_net = QNetwork(state_dim, action_dim).to(device)

    target_manager = TargetNetworkManager(online_net)

    replay_buffer = ReplayBuffer(capacity=1000)

    trainer = DQNTrainer(
        online_network=online_net,
        target_network=target_manager.target_network,
        replay_buffer=replay_buffer,
        learning_rate=0.001,
        gamma=0.99,
        batch_size=16,
        device=device,
    )

    # Fill replay buffer with dummy experiences
    for i in range(100):

        state = [i, i + 1]

        action = i % action_dim

        reward = float(i)

        next_state = [i + 1, i + 2]

        done = (i == 99)

        replay_buffer.push(
            state,
            action,
            reward,
            next_state,
            done,
        )

    loss = trainer.train_step()

    print("\nMini-Batch Training Successful")
    print(f"Training Loss : {loss:.6f}")