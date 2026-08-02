# Q-Learning Baseline Optimization — Issue #59

**Training episodes:** 2000
**Evaluation episodes:** 200 (greedy, no exploration)

## Hyperparameters Tuned

| Parameter | Before (original default) | After (tuned) |
|---|---|---|
| Learning Rate (α) | 0.1 | 0.1 |
| Discount Factor (γ) | 0.95 | 0.95 |
| Initial Epsilon | 1.0 | 1.0 |
| Epsilon Decay | 0.995 | 0.999 |
| Epsilon Min | 0.01 | 0.01 |

**Selected change:** slower epsilon decay (0.995 → 0.999). Learning rate and discount factor were tested at other values in Issue #47's broader search (e.g. α=0.3, γ=0.80) but the original values (α=0.1, γ=0.95) already performed best, so they are kept unchanged here — only epsilon decay is adjusted, since it showed the clearest, most consistent improvement across testing.

## Performance Comparison

| Metric | Before Tuning | After Tuning | Improvement |
|---|---|---|---|
| Avg Reward | 13531.25 | 13753.82 | +1.6% |
| Avg Revenue | 14735.96 | 14983.98 | +1.7% |
| Inventory Utilization | 98.2% | 96.5% | -1.6pp |
| States Learned | 1602 | 1909 | +307 |

## Learning Stability

Stability is measured using the coefficient of variation (CV) of reward across the last 200 training episodes — the standard deviation as a percentage of the mean. A **lower CV means more stable, consistent performance** (the agent has converged), while a high CV means the agent's results are still swinging widely from episode to episode.

| Metric | Before Tuning | After Tuning |
|---|---|---|
| Early training avg reward (first 100 episodes) | 12457.44 | 10488.85 |
| Late training avg reward (last 100 episodes) | 13673.24 | 14108.06 |
| Reward Std Dev (last 200 episodes) | 4428.74 | 4525.31 |
| Coefficient of Variation | 32.1% | 33.2% |

The tuned configuration shows **less stable** learning behavior compared to the original defaults, based on reward variance in the late training stage.

## Conclusion

Tuning the epsilon decay rate from 0.995 to 0.999 improved average reward by **1.6%** (revenue by 1.7%), while maintaining less stable learning behavior. This configuration is now set as the **default** in `agents/q_learning_agent.py`.

## Selected Hyperparameters (Final)

```python
QLearningAgent(
    learning_rate=0.1,
    discount_factor=0.95,
    epsilon=1.0,
    epsilon_decay=0.999,
    epsilon_min=0.01,
    exploration_strategy="epsilon_greedy",
)
```