# 34 Environment Design

## Objective

The Dynamic Pricing Environment is designed to simulate a real-world pricing scenario for industries such as airlines, hotels, and travel services. The environment enables a Reinforcement Learning (RL) agent to learn an optimal pricing policy by interacting with a simulated market.

The primary objective is to maximize cumulative revenue while efficiently managing limited inventory over a finite booking period.

---

# Environment Overview

The environment follows the standard Reinforcement Learning interaction cycle:

1. The environment initializes the inventory, booking period, and price.
2. The RL agent observes the current state.
3. The agent selects a pricing action.
4. The environment updates the price.
5. Customer demand is simulated.
6. Inventory is updated.
7. A reward is calculated.
8. The next state is returned.
9. The process continues until the episode ends.

---

# State Space

The state represents the current condition of the business.

Each state consists of:

| Variable | Description |
|-----------|-------------|
| Remaining Inventory | Number of unsold seats/rooms/products |
| Remaining Days | Number of days left before departure/check-in |
| Current Price | Current selling price |
| Current Demand (optional) | Estimated customer demand |

Example:

```text
State = (Inventory=50,
         Days=20,
         Price=₹1200)
```

The state changes after every pricing decision.

---

# Action Space

The RL agent selects one action from five discrete pricing actions.

| Action | Description |
|---------|-------------|
| 0 | Decrease price by 10% |
| 1 | Decrease price by 5% |
| 2 | Keep current price |
| 3 | Increase price by 5% |
| 4 | Increase price by 10% |

These actions are implemented in:

```text
environment/action_space.py
```

---

# Reward Function

The reward function measures how good a pricing decision is.

The reward considers multiple business objectives.

### Revenue

The agent earns reward whenever inventory is sold.

Revenue = Price × Units Sold

### Overstock Penalty

Unsold inventory at the end of the booking season reduces the reward.

### Understock Penalty

Selling all inventory too early results in a penalty because future revenue opportunities are lost.

### Invalid Pricing Penalty

Prices outside the allowed pricing range receive a penalty.

Overall reward:

```text
Reward =
Revenue
− Overstock Penalty
− Understock Penalty
− Invalid Pricing Penalty
```

The reward function is implemented in:

```text
pricing_env/reward.py
```

---

# Episode Flow

Each episode represents one complete booking season.

Example:

```text
Start Episode

↓

Initialize Environment

↓

Observe State

↓

Choose Action

↓

Update Price

↓

Simulate Demand

↓

Update Inventory

↓

Calculate Reward

↓

Generate Next State

↓

Episode Finished?

│

├── No → Continue

└── Yes → End Episode
```

---

# Environment Lifecycle

The environment follows the OpenAI Gymnasium lifecycle.

### 1. reset()

Initializes the environment.

Returns:

- Initial state
- Initial inventory
- Initial price
- Remaining booking days

---

### 2. step(action)

The agent selects a pricing action.

The environment:

- Updates price
- Simulates demand
- Updates inventory
- Calculates reward
- Generates next state

Returns:

```python
observation,
reward,
terminated,
truncated,
info
```

---

### 3. terminated

Returns **True** when:

- Inventory becomes zero
- Booking period ends

---

### 4. truncated

Returns **True** if the episode stops due to an external condition such as the maximum number of simulation steps.

---

# Project Architecture

```text
               RL Agent
                   │
                   ▼
        Select Pricing Action
                   │
                   ▼
        pricing_env.step(action)
                   │
     ┌─────────────┼─────────────┐
     │             │             │
     ▼             ▼             ▼
Action Space   Demand Model   Reward Function
     │             │             │
     └─────────────┼─────────────┘
                   ▼
           Update Environment
                   │
                   ▼
            Generate Next State
                   │
                   ▼
            Return Observation
```

# Business Objective

The environment is designed to train an RL agent that learns pricing strategies capable of:

- Maximizing total revenue
- Selling inventory before the booking deadline
- Avoiding excessive discounts
- Preventing unrealistic price increases
- Balancing customer demand with available inventory

---

# Future Enhancements

The current environment can be extended by adding:

- Competitor pricing
- Seasonal demand
- Customer segmentation
- Dynamic demand forecasting
- Continuous action space
- Deep Reinforcement Learning (DQN, PPO)

---

# Conclusion

The Dynamic Pricing Environment provides a realistic simulation for Reinforcement Learning research. It models the interaction between pricing decisions, customer demand, inventory management, and business objectives. This modular architecture enables easy integration with different RL algorithms while maintaining flexibility for future improvements.