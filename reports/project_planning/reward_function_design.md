# Reward Function Design

## Objective

The reward function guides the Reinforcement Learning agent toward maximizing long-term business revenue while avoiding undesirable pricing behavior.

---

## Reward Components

### 1. Revenue

Revenue is earned whenever a customer purchases inventory.

Revenue = Price × Units Sold

---

### 2. Overstock Penalty

Unsold inventory at the end of the booking season represents lost revenue.

Penalty = Remaining Inventory × 20

Applied only when the selling season has ended.

---

### 3. Understock Penalty

Selling out too early prevents future high-value sales.

Penalty = Remaining Days × 15

Applied when inventory becomes zero before the booking period ends.

---

### 4. Invalid Pricing Penalty

The pricing agent should never generate unrealistic prices.

Penalty = 100

Applied when

- Price < Minimum Price
- Price > Maximum Price

---

## Final Reward Equation

Reward = Revenue − Overstock Penalty − Understock Penalty − Invalid Pricing Penalty

---

## Expected Agent Behaviour

The reward function encourages the RL agent to:

- maximize revenue
- avoid selling inventory too cheaply
- avoid excessive prices
- finish the booking season with minimal leftover inventory
- maintain realistic pricing decisions