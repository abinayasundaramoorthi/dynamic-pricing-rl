"""
Performance Metrics Collector

Reads training logs and prepares business metrics for
visualizing Reinforcement Learning training performance.
"""

import os
import pandas as pd


class MetricsCollector:
    """Collect training metrics and generate performance_metrics.csv."""

    def __init__(
        self,
        input_file="evaluation/training_logs.csv",
        output_file="evaluation/performance_metrics.csv",
        initial_inventory=25
    ):

        self.input_file = input_file
        self.output_file = output_file
        self.initial_inventory = initial_inventory

        if not os.path.exists(self.input_file):
            raise FileNotFoundError(
                f"Training log not found: {self.input_file}"
            )

        self.df = pd.read_csv(self.input_file)

    def calculate_metrics(self):
        """Calculate business metrics from training logs."""

        metrics = pd.DataFrame()

        # Episode Number
        metrics["Episode"] = self.df["Episode"]

        # Reward Trend
        metrics["Episode Reward"] = self.df["Total Reward"]

        # Revenue Trend
        metrics["Revenue"] = self.df["Final Revenue"]

        # Remaining Inventory
        metrics["Inventory Remaining"] = self.df["Inventory Remaining"]

        # Inventory Utilization (%)
        metrics["Inventory Utilization (%)"] = (
            (
                self.initial_inventory
                - self.df["Inventory Remaining"]
            )
            / self.initial_inventory
            * 100
        ).round(2)

        # Average Selling Price
        if "Average Selling Price" in self.df.columns:
            metrics["Average Selling Price"] = self.df[
                "Average Selling Price"
            ]
        else:
            metrics["Average Selling Price"] = (
                "N/A (Pending RL Integration)"
            )

        # Price Adjustment Frequency
        if "Price Action" in self.df.columns:

            freq = (
                self.df["Price Action"]
                .value_counts(normalize=True)
                .mul(100)
                .round(2)
                .to_dict()
            )

            metrics["Price Adjustment Frequency"] = (
                self.df["Price Action"]
                .map(lambda x: f"{freq[x]}%")
            )

        else:
            metrics["Price Adjustment Frequency"] = (
                "N/A (Pending RL Integration)"
            )

        self.metrics = metrics

    def save_metrics(self):
        """Save metrics to CSV."""

        self.metrics.to_csv(
            self.output_file,
            index=False
        )

        print("\nPerformance metrics generated successfully!")
        print(f"Saved to: {self.output_file}")

    def print_summary(self):
        """Display summary."""

        print("\n========== Performance Metrics ==========\n")

        print(f"Total Episodes : {len(self.metrics)}")

        print(
            f"Average Reward : "
            f"{self.metrics['Episode Reward'].mean():.2f}"
        )

        print(
            f"Average Revenue : "
            f"{self.metrics['Revenue'].mean():.2f}"
        )

        print(
            f"Average Inventory Utilization : "
            f"{self.metrics['Inventory Utilization (%)'].mean():.2f}%"
        )

        print("\n=========================================\n")


if __name__ == "__main__":

    collector = MetricsCollector()

    collector.calculate_metrics()

    collector.save_metrics()

    collector.print_summary()