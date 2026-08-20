"""
exploration.py

Implements the epsilon-greedy exploration strategy used by the DQN agent.
The module balances exploration (random actions) and exploitation
(best predicted action) during training.

Author: Nagammai
"""

import random
import json
from pathlib import Path


class EpsilonGreedyExplorer:
    """
    Implements epsilon-greedy exploration.

    With probability epsilon:
        choose a random action

    Otherwise:
        choose the greedy (best) action.

    Supports:
    - Exponential decay
    - Linear decay
    - Saving/loading exploration state
    """

    def __init__(
        self,
        initial_epsilon=1.0,
        minimum_epsilon=0.01,
        epsilon_decay=0.995,
        decay_strategy="exponential",
    ):

        if not 0 <= minimum_epsilon <= initial_epsilon <= 1:
            raise ValueError(
                "Epsilon values must satisfy "
                "0 <= min <= initial <= 1"
            )

        self.initial_epsilon = initial_epsilon
        self.minimum_epsilon = minimum_epsilon
        self.epsilon_decay = epsilon_decay
        self.decay_strategy = decay_strategy

        self.epsilon = initial_epsilon
        self.step_count = 0

    def choose_action(self, q_values):
        """
        Select an action using epsilon-greedy policy.

        Parameters
        ----------
        q_values : list or numpy.ndarray
            Predicted Q-values from the DQN.

        Returns
        -------
        int
            Selected action index.
        """

        self.step_count += 1

        if random.random() < self.epsilon:
            return random.randint(0, len(q_values) - 1)

        return max(range(len(q_values)), key=lambda i: q_values[i])

    def decay(self):
        """
        Update epsilon after every training step.
        """

        if self.decay_strategy == "exponential":

            self.epsilon = max(
                self.minimum_epsilon,
                self.epsilon * self.epsilon_decay,
            )

        elif self.decay_strategy == "linear":

            self.epsilon = max(
                self.minimum_epsilon,
                self.epsilon - self.epsilon_decay,
            )

        else:
            raise ValueError(
                "Unknown decay strategy."
            )

    def reset(self):
        """
        Reset epsilon to its initial value.
        """

        self.epsilon = self.initial_epsilon
        self.step_count = 0

    def get_epsilon(self):
        """
        Return the current epsilon value.
        """

        return self.epsilon

    def exploration_rate(self):
        """
        Percentage of exploration remaining.
        """

        return self.epsilon * 100

    def save_state(self, filepath="agents/checkpoints/exploration_state.json"):
        """
        Save exploration state.
        """

        Path(filepath).parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        data = {
            "epsilon": self.epsilon,
            "step_count": self.step_count,
        }

        with open(filepath, "w") as file:
            json.dump(data, file, indent=4)

    def load_state(self, filepath="agents/checkpoints/exploration_state.json"):
        """
        Load exploration state.
        """

        with open(filepath) as file:
            data = json.load(file)

        self.epsilon = data["epsilon"]
        self.step_count = data["step_count"]

    def summary(self):
        """
        Print current exploration information.
        """

        print("\n===== Exploration Summary =====")
        print(f"Current epsilon : {self.epsilon:.4f}")
        print(f"Steps           : {self.step_count}")
        print(f"Exploration %   : {self.exploration_rate():.2f}%")
        print("===============================\n")


if __name__ == "__main__":

    explorer = EpsilonGreedyExplorer()

    q_values = [12.5, 9.3, 15.2, 18.8, 13.1]

    for episode in range(10):

        action = explorer.choose_action(q_values)

        print(
            f"Episode {episode+1:02d} | "
            f"Action: {action} | "
            f"Epsilon: {explorer.get_epsilon():.4f}"
        )

        explorer.decay()

    explorer.summary()