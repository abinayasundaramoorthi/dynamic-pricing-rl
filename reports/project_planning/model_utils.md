# Neural Network Utility Functions

## Purpose

This module provides reusable helper utilities required for Deep Q-Network (DQN) training.

## Features

- Random seed initialization
- Automatic CPU/GPU selection
- Neural network weight initialization
- Model summary
- Trainable parameter count
- Model checkpoint saving
- Model checkpoint loading

## Files

```
utils/
├── model_utils.py
└── checkpoint.py
```

## Usage

### Initialize Environment

```python
from utils.model_utils import set_seed, get_device

set_seed(42)
device = get_device()
```

### Initialize Model

```python
from utils.model_utils import initialize_weights

initialize_weights(model)
```

### Save Checkpoint

```python
from utils.checkpoint import save_checkpoint

save_checkpoint(
    model,
    optimizer,
    episode=100,
    reward=5200
)
```

### Load Checkpoint

```python
from utils.checkpoint import load_checkpoint

episode, reward = load_checkpoint(
    model,
    optimizer
)
```