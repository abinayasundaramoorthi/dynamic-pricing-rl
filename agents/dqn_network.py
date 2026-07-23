"""
dqn_network.py

Neural network architecture for the Deep Q-Network (DQN) agent.

Why a neural network instead of a Q-table?
Our tabular Q-Learning agent (q_learning_agent.py) stores a Q-value for
every (state, action) pair it has personally visited. This works, but it
has a fundamental limit: if the agent encounters a state it has never
seen exactly before, it has no idea what to do — it starts from zero.

A neural network instead LEARNS A FUNCTION that maps any state to
Q-values for every action — including states it has never seen exactly,
by recognizing patterns similar to states it has seen. This is what
allows DQN to scale to large or continuous state spaces.
"""

import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """
    A simple feedforward neural network that approximates Q-values.

    Architecture
    ------------
    Input Layer:
        Size = state_dim (number of numbers describing the state, e.g.
        2 for [inventory_remaining, days_remaining]). This is just the
        raw state — no hidden units here, it's the entry point of the data.

    Hidden Layers:
        Two fully-connected layers, 128 units each, with ReLU activation.

        Why 128 units? Our state space is small (2 numbers in Phase 1),
        so we don't need a huge network — 128 units per layer is a
        common, safe default that gives the network enough capacity to
        learn a non-trivial pricing function without being needlessly
        slow to train. This can be tuned later if the state space grows
        (e.g. if more features are added to the observation).

        Why 2 hidden layers? A single hidden layer can only learn
        relatively simple (near-linear) functions of the input. A second
        hidden layer lets the network combine features in more complex,
        non-linear ways — important here because the relationship between
        (inventory, days left) and the best price is not a simple
        straight-line relationship (e.g. "hold price high early, drop it
        sharply near the deadline" is a non-linear strategy).

        Why ReLU activation? ReLU (Rectified Linear Unit) is the standard
        default choice for hidden layers — it's computationally cheap and
        avoids the "vanishing gradient" problem that older activations
        (like sigmoid/tanh) can suffer from, which would make training
        slow or unstable.

    Output Layer:
        Size = action_dim (number of possible pricing actions). Each
        output number is the network's predicted Q-value for taking that
        specific action in the given input state.

        No activation function is applied to the output layer. Q-values
        can be any real number (positive or negative, since our reward
        function includes both revenue and penalties) — applying an
        activation like ReLU or sigmoid here would incorrectly restrict
        the range of values the network can predict.

    Weight Initialization
    ----------------------
    Hidden layers use He (Kaiming) initialization, which is specifically
    designed to work well with ReLU activations — it keeps the scale of
    values reasonable as they pass through many layers, helping training
    start smoothly instead of with wildly too-large or too-small values.
    The output layer uses a smaller Xavier initialization, appropriate
    since it has no activation function of its own.
    """

    def __init__(self, state_dim, action_dim, hidden_dim=128):
        """
        Parameters
        ----------
        state_dim : int
            Number of values describing the state (e.g. 2 for
            [inventory_remaining, days_remaining]).
        action_dim : int
            Number of possible discrete actions (price levels).
        hidden_dim : int
            Number of units in each hidden layer. Default: 128.
        """
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        # --- Define the layers ---
        self.hidden_layer_1 = nn.Linear(state_dim, hidden_dim)
        self.hidden_layer_2 = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, action_dim)

        self.activation = nn.ReLU()

        # --- Initialize weights ---
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Set starting weights using principled initialization schemes,
        rather than PyTorch's raw defaults, so training starts from a
        numerically well-behaved point.
        """
        # He (Kaiming) initialization for ReLU-activated hidden layers
        nn.init.kaiming_uniform_(self.hidden_layer_1.weight, nonlinearity="relu")
        nn.init.zeros_(self.hidden_layer_1.bias)

        nn.init.kaiming_uniform_(self.hidden_layer_2.weight, nonlinearity="relu")
        nn.init.zeros_(self.hidden_layer_2.bias)

        # Xavier initialization for the output layer (no activation function)
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, state):
        """
        Forward propagation: compute Q-values for every action, given
        an input state.

        Parameters
        ----------
        state : torch.Tensor
            Shape (batch_size, state_dim) or (state_dim,) for a single state.

        Returns
        -------
        q_values : torch.Tensor
            Shape (batch_size, action_dim) or (action_dim,) — the
            predicted Q-value for each possible action.
        """
        x = self.hidden_layer_1(state)
        x = self.activation(x)

        x = self.hidden_layer_2(x)
        x = self.activation(x)

        q_values = self.output_layer(x)  # no activation on output

        return q_values


if __name__ == "__main__":
    # Quick smoke test: confirm the network builds and runs correctly.
    state_dim = 2   # [inventory_remaining, days_remaining]
    action_dim = 7  # matches PricingEnvConfig's default price_adjustment_pct list

    network = QNetwork(state_dim=state_dim, action_dim=action_dim)
    print(network)

    # Single state input
    sample_state = torch.tensor([50.0, 15.0])
    q_values = network(sample_state)
    print(f"\nSingle state input shape: {sample_state.shape}")
    print(f"Q-values output: {q_values}")
    print(f"Q-values shape: {q_values.shape} (should be ({action_dim},))")

    # Batch input (multiple states at once)
    batch_states = torch.tensor([[50.0, 15.0], [80.0, 5.0], [10.0, 25.0]])
    batch_q_values = network(batch_states)
    print(f"\nBatch input shape: {batch_states.shape}")
    print(f"Batch Q-values shape: {batch_q_values.shape} (should be (3, {action_dim}))")

    print("\nNetwork initialized and forward pass executed successfully!")