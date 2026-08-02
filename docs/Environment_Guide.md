# Environment Guide

## Overview

The Dynamic Pricing Reinforcement Learning project uses a custom Gymnasium-compatible environment that simulates hotel room pricing over a fixed selling horizon.

The environment enables reinforcement learning agents to dynamically adjust prices while maximizing revenue and improving inventory utilization.

---

# Environment Features

- Gymnasium-compatible interface
- Dynamic pricing actions
- Demand simulation
- Inventory tracking
- Reward calculation
- Episode termination

---

# Project Structure

```
pricing_env/
│
├── pricing_env.py
├── action_space.py
├── demand_simulator.py
├── reward.py
├── state.py
└── __init__.py
```

---

# Observation Space

The environment returns the current state as

```
[Remaining Inventory,
 Remaining Selling Days]
```

Example

```
[100.0, 30.0]
```

---

# Action Space

Five discrete pricing actions are supported.

| Action | Description |
|---------|-------------|
|0|Decrease price by 10%|
|1|Decrease price by 5%|
|2|Keep current price|
|3|Increase price by 5%|
|4|Increase price by 10%|

---

# Reset Environment

```python
from pricing_env import PricingEnvironment

env = PricingEnvironment()

state, info = env.reset()
```

---

# Step Through Environment

```python
next_state, reward, terminated, truncated, info = env.step(action)
```

---

# Episode Ends When

- Inventory becomes zero
- Selling horizon ends

---

# Verification

Environment initialization was verified successfully.

```
Environment initialized successfully!
[100. 30.]
```

---

# Purpose

The environment is used for:

- Rule-Based Pricing
- Static Pricing
- Q-Learning
- Deep Q-Network (DQN)
