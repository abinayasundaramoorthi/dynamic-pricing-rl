"""
Training Logger for Dynamic Pricing Reinforcement Learning

This module logs training metrics for each episode and automatically
creates a CSV file to store the logs if it does not already exist.
"""

import csv
import os


class TrainingLogger:
    """
    Logger class for recording Reinforcement Learning training metrics.
    """

    def __init__(self, log_file="evaluation/training_logs.csv"):
        """
        Initialize the logger.

        Parameters
        ----------
        log_file : str
            Path to the CSV log file.
        """

        self.log_file = log_file

        # Create evaluation directory if it doesn't exist
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)

        # Automatically create CSV file with header
        if not os.path.exists(self.log_file):
            with open(self.log_file, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([
                    "Episode",
                    "Total Reward",
                    "Inventory Remaining",
                    "Final Revenue"
                ])

    def log_episode(
        self,
        episode: int,
        total_reward: float,
        inventory_remaining: int,
        final_revenue: float
    ):
        """
        Log metrics for a single training episode.

        Parameters
        ----------
        episode : int
            Episode number.

        total_reward : float
            Total reward accumulated during the episode.

        inventory_remaining : int
            Inventory remaining after the episode.

        final_revenue : float
            Revenue generated during the episode.
        """

        with open(self.log_file, mode="a", newline="") as file:
            writer = csv.writer(file)

            writer.writerow([
                episode,
                total_reward,
                inventory_remaining,
                final_revenue
            ])

    def display_logs(self):
        """
        Display all logged training metrics.
        """

        if not os.path.exists(self.log_file):
            print("No training logs found.")
            return

        print("\n========== TRAINING LOGS ==========\n")

        with open(self.log_file, mode="r") as file:
            reader = csv.reader(file)

            for row in reader:
                print(row)

        print("\n===================================\n")


# -------------------------------------------------------
# Example Usage
# -------------------------------------------------------

if __name__ == "__main__":

    logger = TrainingLogger()

    # Sample Episode 1
    logger.log_episode(
        episode=1,
        total_reward=2450,
        inventory_remaining=18,
        final_revenue=3100
    )

    # Sample Episode 2
    logger.log_episode(
        episode=2,
        total_reward=2725,
        inventory_remaining=10,
        final_revenue=3450
    )

    # Sample Episode 3
    logger.log_episode(
        episode=3,
        total_reward=2980,
        inventory_remaining=5,
        final_revenue=3820
    )

    print("Training metrics logged successfully!")

    logger.display_logs()