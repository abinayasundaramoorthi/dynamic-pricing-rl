# DQN vs Q-Learning Performance Comparison — Issue #80

**Training episodes:** 1000 (each agent)
**Evaluation episodes:** 200 (greedy, no exploration)

## Agents Compared

**Q-Learning:** Tabular agent using a Q-table, with tuned hyperparameters from Issue #59 (learning rate=0.1, discount factor=0.95, epsilon decay=0.999).

**DQN:** Neural network-based agent (Issue #63 architecture), trained using an experience replay buffer (Issue #71) and a target network for stability (Issue #75). Same core hyperparameters (discount factor=0.95, similar epsilon schedule) for a fair comparison, plus DQN-specific settings: learning rate=0.001, batch size=64, target network update every 500 steps.

## Performance Comparison

| Metric | Q-Learning | DQN | Difference (DQN vs Q-Learning) |
|---|---|---|---|
| Average Reward | 14494.25 | 15384.11 | +6.1% |
| Average Revenue | 15892.14 | 16076.10 | +1.2% |
| Inventory Utilization | 94.2% | 97.8% | +3.6pp |

## Training Stability

Measured as the standard deviation of reward across the last 100 training episodes — lower means more consistent, converged performance.

| Metric | Q-Learning | DQN |
|---|---|---|
| Reward Std Dev (last 100 episodes) | 5872.32 | 4180.40 |

**More stable agent: DQN**

## Episode Performance

| Metric | Q-Learning | DQN |
|---|---|---|
| Early Training Avg (first 100 ep) | 11283.41 | 13058.46 |
| Late Training Avg (last 100 ep) | 12954.98 | 15763.14 |
| Best Single Episode Reward | 22031.17 | 22349.60 |
| States Learned (Q-table) / Network Parameters (DQN) | 1903 | 17799 |

## Observations

- **DQN** achieved the higher average reward at evaluation time (+6.1% difference).
- Q-Learning's Q-table grew to 1903 learned states, each stored and updated independently. DQN instead uses a fixed 17799-parameter neural network, which can in principle generalize to states never seen exactly during training — a key theoretical advantage of DQN for larger or continuous state spaces, though this project's state space (inventory × days remaining) is small enough that Q-Learning can already represent it fairly completely.
- DQN showed lower reward variance in late training, suggesting more consistent, converged behavior by the end of training.

## Conclusion

For this project's current state space size (2 dimensions, bounded ranges), **DQN** performed best on average reward in this evaluation. Given the state space is small enough for a Q-table to represent completely, Q-Learning's simpler, more directly interpretable approach remains a strong and computationally cheaper baseline. DQN's architecture is nonetheless valuable groundwork: if the environment is later extended with additional continuous features (e.g. real competitor pricing, seasonal demand signals), a tabular Q-table would no longer be feasible, and the DQN implementation built here would become the necessary approach.
