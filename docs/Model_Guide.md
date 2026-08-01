# Model Guide

## Overview

The project compares multiple pricing strategies using Reinforcement Learning.

---

# Available Models

## 1. Static Pricing

Uses a fixed selling price throughout the season.

Advantages

- Simple
- Easy to implement

Limitations

- Cannot adapt to demand changes

---

## 2. Rule-Based Pricing

Adjusts prices according to predefined business rules.

Advantages

- Easy to understand
- Business friendly

Limitations

- No learning capability

---

## 3. Q-Learning

Uses a Q-table to learn the best pricing action.

Key Parameters

- Learning Rate
- Discount Factor
- Exploration Rate

Advantages

- Learns optimal pricing policy

Limitations

- Large state spaces become difficult

---

## 4. Deep Q-Network (DQN)

Uses a neural network to approximate Q-values.

Components

- Replay Buffer
- Target Network
- Neural Network
- Experience Replay

Hyperparameters

- Learning Rate
- Batch Size
- Replay Buffer Size
- Target Update Frequency

Advantages

- Handles larger state spaces
- Better scalability

---

# Training Workflow

```
Initialize Environment

↓

Initialize Agent

↓

Run Episodes

↓

Collect Experience

↓

Update Model

↓

Evaluate Policy
```

---

# Evaluation Metrics

- Total Reward
- Average Reward
- Revenue
- Inventory Utilization
- Sell-through Rate
- Average Selling Price

---

# Conclusion

The DQN model demonstrated improved business performance compared to traditional pricing strategies.