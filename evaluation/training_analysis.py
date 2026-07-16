"""
Training Performance Analyzer

This module analyzes Reinforcement Learning training logs and computes
performance metrics for the Dynamic Pricing project.

"""

import os
import pandas as pd
import matplotlib.pyplot as plt


class TrainingAnalyzer:
    """
    Analyze RL training logs and generate performance reports.
    """

    def __init__(
        self,
        log_file="evaluation/training_logs.csv",
        report_file="evaluation/training_report.md",
    ):

        self.log_file = log_file
        self.report_file = report_file
        self.df = pd.read_csv(self.log_file)


    def calculate_metrics(self):
        """
        Calculate important training statistics.
        """

        self.total_episodes = len(self.df)

        self.average_reward = self.df["Total Reward"].mean()

        self.average_revenue = self.df["Final Revenue"].mean()

        self.average_inventory = self.df[
            "Inventory Remaining"
        ].mean()

        self.best_reward = self.df["Total Reward"].max()

        # Best episode
        best_row = self.df.loc[
            self.df["Total Reward"].idxmax()
        ]

        self.best_episode = int(best_row["Episode"])

        # Revenue improvement

        self.first_revenue = self.df.iloc[0]["Final Revenue"]

        self.last_revenue = self.df.iloc[-1]["Final Revenue"]

        self.revenue_improvement = (
            self.last_revenue
            - self.first_revenue
        )

    # --------------------------------------------------

    def print_summary(self):
        """
        Print training summary.
        """

        print("\n========== TRAINING SUMMARY ==========\n")

        print(f"Total Episodes : {self.total_episodes}")

        print(
            f"Average Reward : {self.average_reward:.2f}"
        )

        print(
            f"Average Revenue : {self.average_revenue:.2f}"
        )

        print(
            f"Average Inventory Remaining : "
            f"{self.average_inventory:.2f}"
        )

        print(
            f"Best Episode : {self.best_episode}"
        )

        print(
            f"Best Reward : {self.best_reward}"
        )

        print(
            f"Revenue Improvement : "
            f"{self.revenue_improvement}"
        )

        print("\n======================================\n")

    # Reward Trend

    def plot_reward_trend(self):
        """
        Plot reward trend over episodes.
        """

        plt.figure(figsize=(8, 5))

        plt.plot(
            self.df["Episode"],
            self.df["Total Reward"],
            marker="o",
            linewidth=2,
        )

        plt.title("Reward Trend")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")

        plt.grid(True)

        plt.tight_layout()

        plt.savefig(
            "evaluation/reward_trend.png",
            dpi=300
        )

        plt.close()

    # Revenue Trend

    def plot_revenue_trend(self):
        """
        Plot revenue trend over episodes.
        """

        plt.figure(figsize=(8, 5))

        plt.plot(
            self.df["Episode"],
            self.df["Final Revenue"],
            marker="o",
            linewidth=2,
        )

        plt.title("Revenue Trend")
        plt.xlabel("Episode")
        plt.ylabel("Final Revenue")

        plt.grid(True)

        plt.tight_layout()

        plt.savefig(
            "evaluation/revenue_trend.png",
            dpi=300
        )

        plt.close()

    # Inventory Trend

    def plot_inventory_trend(self):
        """
        Plot remaining inventory after every episode.
        """

        plt.figure(figsize=(8, 5))

        plt.plot(
            self.df["Episode"],
            self.df["Inventory Remaining"],
            marker="o",
            linewidth=2,
        )

        plt.title("Inventory Utilization")
        plt.xlabel("Episode")
        plt.ylabel("Inventory Remaining")

        plt.grid(True)

        plt.tight_layout()

        plt.savefig(
            "evaluation/inventory_trend.png",
            dpi=300
        )

        plt.close()

    def generate_graphs(self):
        """
        Generate all analysis graphs.
        """

        print("Generating Reward Trend...")

        self.plot_reward_trend()

        print("Generating Revenue Trend...")

        self.plot_revenue_trend()

        print("Generating Inventory Trend...")

        self.plot_inventory_trend()

        print("Graphs saved inside evaluation/.")

if __name__ == "__main__":

    analyzer = TrainingAnalyzer()

    analyzer.calculate_metrics()

    analyzer.print_summary()

    analyzer.generate_graphs()