# DQN Model Loading Guide

**File:** `models/dqn_weights.pth`
**Related Issue:** #107 — Package and Publish Trained DQN Model

---

## What This File Contains

`dqn_weights.pth` contains the **learned weights** of the trained DQN neural network — the numbers the network adjusted during training to become good at predicting Q-values for pricing decisions.

It does **not** contain the network's architecture (layer sizes, structure) — only the weights. To use this file, you need to first build a network with the **exact same architecture** it was trained with, then load these weights into it.

## Model Architecture Required

The saved weights match this exact architecture (from `agents/dqn_network.py`):

| Setting | Value |
|---|---|
| Input size (state_dim) | 2 (inventory_remaining, days_remaining) |
| Hidden layers | 2 layers, 128 units each |
| Output size (action_dim) | 7 (number of price adjustment options) |
| Activation | ReLU (hidden layers only) |

**Important:** If you change the network architecture in `dqn_network.py` (e.g. different hidden layer size), these saved weights will no longer load correctly — you'll need to retrain and re-save.

## How to Load and Use This Model

### Step 1: Import the network class
```python
from agents.dqn_network import QNetwork
import torch
```

### Step 2: Build a network with matching architecture
```python
network = QNetwork(state_dim=2, action_dim=7, hidden_dim=128)
```

### Step 3: Load the saved weights into it
```python
network.load_state_dict(torch.load("models/dqn_weights.pth", map_location="cpu"))
network.eval()  # switch to inference mode (disables training-only behavior)
```

### Step 4: Run inference (get a price recommendation)
```python
import numpy as np

state = torch.tensor([50.0, 15.0], dtype=torch.float32)  # [inventory, days_remaining]

with torch.no_grad():
    q_values = network(state)

recommended_action = int(torch.argmax(q_values).item())
print(f"Recommended action: {recommended_action}")
```

## Using It via DQNAgent (Simpler Alternative)

If you're working with the full `DQNAgent` class (`agents/dqn_agent.py`), it already wraps all of the above:

```python
from agents.dqn_agent import DQNAgent

agent = DQNAgent(state_dim=2, action_dim=7)
agent.load_model("models/dqn_weights.pth")

action = agent.choose_action(observation=[50.0, 15.0], greedy=True)
```

## Verification Performed

Before publishing, this model was verified with the following checks (see `models/package_model.py`):

1. **Weights saved successfully** — confirmed the file was written without errors.
2. **Loading verified on a fresh network** — the weights were loaded into a brand-new, randomly-initialized network (not the original training instance), confirming the saved file works independently, not just because the trained object was still in memory.
3. **Inference tested** — ran the loaded network on several sample states, confirming valid, non-NaN, non-Infinity Q-value outputs, and sensible recommended actions.

## Common Pitfalls

- **Mismatched architecture:** If you get a `size mismatch` error when loading, it means the network you built doesn't match the architecture used during training. Double-check `state_dim`, `action_dim`, and `hidden_dim` match exactly.
- **Forgetting `.eval()`:** Always call `network.eval()` after loading for inference — this ensures consistent, deterministic behavior (relevant for architectures using dropout/batch norm, and good practice regardless).
- **Device mismatch:** If you trained on GPU but are loading on a CPU-only machine, use `map_location="cpu"` in `torch.load()` (already included in the examples above) to avoid errors.