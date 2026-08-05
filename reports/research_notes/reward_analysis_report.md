# Reward Analysis Report

## Overview

This report analyzes the reward components of the Dynamic Pricing Reinforcement Learning environment using the available training logs.

---

## Reward Statistics

| Metric | Value |
|---------|------:|
| Episodes Analysed | 3 |
| Average Reward | 2718.33 |
| Average Revenue | 3456.67 |
| Highest Revenue | 3820.00 |
| Revenue Improvement | 720.00 |
| Average Inventory Remaining | 11.00 |
| Best Episode | 3 |
| Best Reward | 2980.00 |

---

# Reward Component Analysis

## 1. Revenue Reward

The reward function encourages higher revenue generation by assigning positive rewards to profitable pricing decisions.

**Observation**

- Average Revenue : 3456.67
- Highest Revenue : 3820.00

The available training metrics indicate positive revenue generation throughout the recorded episodes.

---

## 2. Unsold Inventory Penalty

The reward function penalizes remaining inventory to encourage selling available inventory before the selling period ends.

**Observation**

- Average Remaining Inventory : 11.00
- Minimum Remaining Inventory : 5

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

**720.00**

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
