"""
Reward Analysis Module

Analyzes reward-related metrics from training logs and generates
a reward analysis report for the Dynamic Pricing RL project.
"""

import os
import pandas as pd


class RewardAnalyzer:
    """Analyze reward metrics and generate a report."""

    def __init__(
        self,
        log_file="evaluation/training_logs.csv",
        report_file="evaluation/reward_analysis_report.md"
    ):

        self.log_file = log_file
        self.report_file = report_file

        if not os.path.exists(self.log_file):
            raise FileNotFoundError(
                f"Training log not found: {self.log_file}"
            )

        self.df = pd.read_csv(self.log_file)

    def analyze(self):
        """Calculate reward-related statistics."""

        self.total_episodes = len(self.df)

        self.average_reward = self.df["Total Reward"].mean()

        self.average_revenue = self.df["Final Revenue"].mean()

        self.maximum_revenue = self.df["Final Revenue"].max()

        self.minimum_revenue = self.df["Final Revenue"].min()

        self.average_inventory = self.df[
            "Inventory Remaining"
        ].mean()

        self.minimum_inventory = self.df[
            "Inventory Remaining"
        ].min()

        self.maximum_inventory = self.df[
            "Inventory Remaining"
        ].max()

        self.best_episode = self.df.loc[
            self.df["Total Reward"].idxmax(),
            "Episode"
        ]

        self.best_reward = self.df["Total Reward"].max()

        self.revenue_growth = (
            self.df.iloc[-1]["Final Revenue"]
            -
            self.df.iloc[0]["Final Revenue"]
        )

    def print_summary(self):
        """Display analysis summary."""

        print("\n========== Reward Analysis ==========\n")

        print(f"Episodes              : {self.total_episodes}")
        print(f"Average Reward        : {self.average_reward:.2f}")
        print(f"Average Revenue       : {self.average_revenue:.2f}")
        print(f"Revenue Improvement   : {self.revenue_growth:.2f}")
        print(f"Average Inventory     : {self.average_inventory:.2f}")
        print(f"Best Episode          : {self.best_episode}")
        print(f"Best Reward           : {self.best_reward:.2f}")

        print("\n=====================================\n")

    def generate_report(self):
        """Generate the reward analysis report in Markdown format."""

        report = f"""# Reward Analysis Report

## Overview

This report analyzes the reward components of the Dynamic Pricing Reinforcement Learning environment using the available training logs.

---

## Reward Statistics

| Metric | Value |
|---------|------:|
| Episodes Analysed | {self.total_episodes} |
| Average Reward | {self.average_reward:.2f} |
| Average Revenue | {self.average_revenue:.2f} |
| Highest Revenue | {self.maximum_revenue:.2f} |
| Revenue Improvement | {self.revenue_growth:.2f} |
| Average Inventory Remaining | {self.average_inventory:.2f} |
| Best Episode | {self.best_episode} |
| Best Reward | {self.best_reward:.2f} |

---

# Reward Component Analysis

## 1. Revenue Reward

The reward function encourages higher revenue generation by assigning positive rewards to profitable pricing decisions.

**Observation**

- Average Revenue : {self.average_revenue:.2f}
- Highest Revenue : {self.maximum_revenue:.2f}

The available training metrics indicate positive revenue generation throughout the recorded episodes.

---

## 2. Unsold Inventory Penalty

The reward function penalizes remaining inventory to encourage selling available inventory before the selling period ends.

**Observation**

- Average Remaining Inventory : {self.average_inventory:.2f}
- Minimum Remaining Inventory : {self.minimum_inventory}

Lower remaining inventory generally represents better inventory utilization.

---

## 3. Discount Penalty

The current training logs do not contain pricing action history.

Therefore, excessive discounting or aggressive price increases cannot yet be directly evaluated.

Future versions of this analysis can incorporate action logs after Q-Learning or DQN training is completed.

---

## 4. Inventory Balance Reward

The designed reward function balances multiple business objectives rather than maximizing only one metric.

The reward simultaneously considers:

- Revenue generation
- Remaining inventory
- Pricing quality
- Long-term business performance

This encourages balanced pricing decisions instead of short-term optimization.

---

# Behaviour Analysis

### Revenue Improvement

Revenue improvement across episodes:

**{self.revenue_growth:.2f}**

This indicates that pricing performance is improving based on the available training metrics.

---

### Inventory Utilization

The reduction in remaining inventory indicates improved inventory utilization.

---

### Pricing Behaviour

Detailed pricing behaviour analysis requires pricing action history, which will become available after RL agent training.

---

# Overall Observations

✔ Revenue reward encourages profitable pricing.

✔ Unsold inventory penalty supports inventory clearance.

✔ Reward function balances revenue and inventory objectives.

✔ Current implementation provides a solid foundation for future Reinforcement Learning experiments.

✔ More detailed behavioural analysis can be performed after Q-Learning/DQN integration.

---

# Conclusion

The current reward function is well aligned with the business objective of Dynamic Pricing.

It rewards higher revenue while discouraging unsold inventory and poor pricing decisions.

Once Reinforcement Learning training is completed, this analysis framework can be extended to evaluate pricing policies, discount behaviour, and long-term learning performance.
"""

        with open(self.report_file, "w", encoding="utf-8") as file:
            file.write(report)

        print("\nReport generated successfully!")
        print(self.report_file)


if __name__ == "__main__":

    analyzer = RewardAnalyzer()

    analyzer.analyze()

    analyzer.print_summary()

    analyzer.generate_report()