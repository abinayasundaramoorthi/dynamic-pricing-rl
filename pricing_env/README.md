# Dynamic Pricing Gymnasium Environment

## Overview

This package contains a custom Gymnasium environment developed for the
Dynamic Pricing Reinforcement Learning project.

The environment simulates hotel inventory management where a Reinforcement
Learning agent dynamically adjusts prices to maximize total revenue while
efficiently utilizing inventory.

---

# Features

- Gymnasium compatible environment
- Dynamic pricing simulation
- Demand prediction integration
- Reward calculation
- Inventory tracking
- Action and observation spaces
- Compatible with Q-Learning and Deep Q-Network (DQN)

---

# Folder Structure

pricing_env/

├── __init__.py

├── pricing_env.py

├── action_space.py

├── demand_simulator.py

├── reward.py

├── state.py

└── README.md

---

# Environment Components

## PricingEnvironment

Main Gymnasium environment.

Responsibilities

- Reset environment
- Execute actions
- Calculate rewards
- Update inventory
- Return observations

---

## Action Space

Five discrete pricing actions.

| Action | Description |
|---------|-------------|
|0|Decrease Price by 10%|
|1|Decrease Price by 5%|
|2|Keep Current Price|
|3|Increase Price by 5%|
|4|Increase Price by 10%|

---

## Observation Space

Environment state contains

- Current Price
- Remaining Inventory
- Remaining Days

---

## Reward Function

Reward is calculated using

Revenue

minus

Inventory Penalty

minus

Discount Penalty

plus

Inventory Balance Bonus

---

## Demand Simulator

Simulates customer demand based on

- Selling Price
- Remaining Inventory
- Remaining Selling Days

---

## Reset

```python
observation, info = env.reset()
```

---

## Step

```python
observation, reward, terminated, truncated, info = env.step(action)
```

---

## Compatible Algorithms

- Rule Based Pricing
- Static Pricing
- Q-Learning
- Deep Q Network (DQN)

---

## Example

```python
from pricing_env import PricingEnvironment

env = PricingEnvironment()

observation, info = env.reset()

action = 2

observation, reward, terminated, truncated, info = env.step(action)
```

---

## License

Educational use only.