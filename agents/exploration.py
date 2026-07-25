"""
Epsilon-Greedy Exploration Strategy

Implements the exploration policy for the
Deep Q-Network (DQN) agent.
"""

import random

from configs.dqn_config import DQNConfig


class EpsilonGreedy:
    """
    Implements epsilon-greedy exploration.
    """

    def __init__(
        self,
        initial_epsilon=DQNConfig.INITIAL_EPSILON,
        minimum_epsilon=DQNConfig.MIN_EPSILON,
        epsilon_decay=DQNConfig.EPSILON_DECAY
    ):

        self.epsilon = initial_epsilon

        self.minimum_epsilon = minimum_epsilon

        self.epsilon_decay = epsilon_decay

    # -----------------------------------------------------

    def select_action(
        self,
        q_values,
        action_space
    ):
        """
        Select action using epsilon-greedy policy.

        Parameters
        ----------
        q_values : list or numpy array
            Predicted Q-values.

        action_space : list
            Available actions.

        Returns
        -------
        int
            Selected action.
        """

        if random.random() < self.epsilon:

            action = random.choice(action_space)

        else:

            max_index = q_values.index(max(q_values))

            action = action_space[max_index]

        return action

    # -----------------------------------------------------

    def decay_epsilon(self):
        """
        Update epsilon after each episode.
        """

        self.epsilon = max(
            self.minimum_epsilon,
            self.epsilon * self.epsilon_decay
        )

        return self.epsilon

    # -----------------------------------------------------

    def reset(self):
        """
        Reset epsilon for a new experiment.
        """

        self.epsilon = DQNConfig.INITIAL_EPSILON

    # -----------------------------------------------------

    def get_epsilon(self):
        """
        Return current epsilon value.
        """

        return self.epsilon

    # -----------------------------------------------------

    def __str__(self):

        return (
            f"EpsilonGreedy("
            f"epsilon={self.epsilon:.4f}, "
            f"min={self.minimum_epsilon}, "
            f"decay={self.epsilon_decay})"
        )