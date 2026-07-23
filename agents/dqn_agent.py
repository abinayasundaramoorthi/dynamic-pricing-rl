"""
dqn_agent.py

DQN agent — wraps the QNetwork (dqn_network.py) and provides an interface
for choosing actions based on the network's predicted Q-values.

Scope of this file (Issue #63): initializing the network and getting
Q-value predictions / choosing actions from it (forward pass only).
Training the network (experience replay, target network, loss
calculation, backpropagation) is intentionally NOT included here — that
is the focus of a following issue, matching the project's Week 3 roadmap
of first building the network (this issue), then training it.
"""

import torch
import numpy as np

from .dqn_network import QNetwork


class DQNAgent:
    """
    Wraps a QNetwork and provides a simple interface for getting
    Q-value predictions and choosing actions — the DQN equivalent of
    QLearningAgent's choose_action(), but backed by a neural network
    instead of a Q-table.
    """

    def __init__(self, state_dim, action_dim, hidden_dim=128, device=None):
        """
        Parameters
        ----------
        state_dim : int
            Number of values describing the state (e.g. 2 for
            [inventory_remaining, days_remaining]).
        action_dim : int
            Number of possible discrete actions (price levels).
            Get this from env.action_space.n
        hidden_dim : int
            Number of units in each hidden layer of the network.
        device : str or torch.device, optional
            "cpu" or "cuda". Defaults to GPU if available, otherwise CPU.
        """
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.q_network = QNetwork(
            state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim
        ).to(self.device)

    def get_q_values(self, observation):
        """
        Run a forward pass through the network to get predicted Q-values
        for a given observation.

        Parameters
        ----------
        observation : array-like
            The current state, e.g. [inventory_remaining, days_remaining]
            (a NumPy array, list, or similar).

        Returns
        -------
        q_values : numpy.ndarray
            Predicted Q-value for each possible action.
        """
        state_tensor = torch.tensor(
            np.array(observation), dtype=torch.float32, device=self.device
        )

        # No gradient tracking needed here — this is just inference
        # (getting a prediction), not a training step.
        with torch.no_grad():
            q_values = self.q_network(state_tensor)

        return q_values.cpu().numpy()

    def choose_action(self, observation):
        """
        Choose the action with the highest predicted Q-value (greedy).

        Note: this is purely greedy for now — epsilon-greedy exploration,
        experience replay, and the training loop itself will be added in
        the follow-up training issue.

        Parameters
        ----------
        observation : array-like
            The current state.

        Returns
        -------
        action : int
            Index of the action with the highest predicted Q-value.
        """
        q_values = self.get_q_values(observation)
        return int(np.argmax(q_values))

    def save_model(self, filepath="agents/dqn_model.pt"):
        """Save the network's learned weights to disk."""
        torch.save(self.q_network.state_dict(), filepath)
        print(f"Model weights saved to {filepath}")

    def load_model(self, filepath="agents/dqn_model.pt"):
        """Load previously saved network weights."""
        self.q_network.load_state_dict(
            torch.load(filepath, map_location=self.device)
        )
        self.q_network.eval()
        print(f"Model weights loaded from {filepath}")


if __name__ == "__main__":
    # Quick smoke test: confirm the agent initializes and can choose actions.
    state_dim = 2
    action_dim = 7

    agent = DQNAgent(state_dim=state_dim, action_dim=action_dim)
    print(f"DQNAgent initialized on device: {agent.device}")
    print(agent.q_network)

    sample_observation = [50.0, 15.0]
    q_values = agent.get_q_values(sample_observation)
    print(f"\nQ-values for state {sample_observation}: {q_values}")

    action = agent.choose_action(sample_observation)
    print(f"Chosen action (highest Q-value): {action}")

    print("\nDQNAgent forward pass and action selection working correctly!")