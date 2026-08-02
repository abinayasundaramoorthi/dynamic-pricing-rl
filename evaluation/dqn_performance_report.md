# Deep Reinforcement Learning Performance Report — Issue #84

**Training episodes:** 1000 (each agent)
**Evaluation episodes:** 200 (greedy, no exploration)

## Summary Chart

![Performance Summary](..\dashboard\performance_summary.png)

The chart above shows (clockwise from top-left): episode reward progression during training, DQN training loss over time, average reward/revenue comparison, and inventory utilization comparison.

## 1. Average Reward

| Agent | Avg Reward (Evaluation) |
|---|---|
| Q-Learning | 13908.38 |
| DQN | 16939.27 |

**Difference:** DQN +21.8% vs Q-Learning

## 2. Episode Reward (Training Progress)

| Agent | Early Training Avg (first 100 ep) | Late Training Avg (last 100 ep) | Best Episode |
|---|---|---|---|
| Q-Learning | 10163.77 | 13183.09 | 22115.33 |
| DQN | 12432.37 | 16499.67 | 22671.91 |

## 3. Training Loss (DQN)

DQN's training loss (Mean Squared Error between predicted and target Q-values) started at approximately **704541.93** and settled to approximately **112771.65** by the end of training. A decreasing loss trend indicates the network is successfully learning to predict Q-values more accurately over time, rather than diverging or staying flat (which would indicate a training problem).

Note: Q-Learning has no equivalent "loss" metric, since it updates a Q-table directly using the Q-Learning update rule rather than training a neural network via gradient descent — this metric is DQN-specific.

## 4. Revenue

| Agent | Avg Revenue (Evaluation) |
|---|---|
| Q-Learning | 15353.82 |
| DQN | 17219.75 |

**Difference:** DQN +12.2% vs Q-Learning

## 5. Inventory Utilization

| Agent | Inventory Utilization |
|---|---|
| Q-Learning | 94.8% |
| DQN | 99.0% |

## 6. Comparison with Q-Learning — Summary

| Metric | Q-Learning | DQN | Winner |
|---|---|---|---|
| Avg Reward | 13908.38 | 16939.27 | DQN |
| Avg Revenue | 15353.82 | 17219.75 | DQN |
| Inventory Utilization | 94.8% | 99.0% | DQN |

## Key Observations

- **DQN** achieved the higher average reward overall (+21.8%), consistent with the dedicated comparison conducted in Issue #80.
- DQN's training loss trend (decreasing from ~704541.93 to ~112771.65) confirms the neural network trained successfully and did not diverge, validating the network architecture (Issue #63), replay buffer (Issue #71), and target network (Issue #75) all worked correctly together.
- Both agents achieved high inventory utilization (Q-Learning: 94.8%, DQN: 99.0%), indicating both successfully learned to sell down inventory before the selling season ends, rather than leaving significant unsold stock.
- Q-Learning's Q-table is fully interpretable (every state's learned values can be directly inspected), while DQN's knowledge is distributed across network weights — a trade-off between DQN's generalization ability and Q-Learning's transparency, worth considering for future deployment decisions.

## Conclusion

Both agents successfully learned effective dynamic pricing policies for this environment. DQN produced the strongest evaluation results in this run. Given the project's current small state space, Q-Learning remains a strong, simple, interpretable baseline, while the DQN implementation demonstrates the project is ready to scale to larger or more complex state spaces (e.g. additional continuous features) where a tabular approach would no longer be feasible.
