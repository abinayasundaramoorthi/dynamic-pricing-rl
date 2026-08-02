# Q-Learning Hyperparameter Optimization Results — Issue #47

**Training episodes per configuration:** 2000
**Evaluation episodes per configuration:** 200 (greedy, no exploration)

## Parameter Combinations Tested

| Configuration | Learning Rate (α) | Discount Factor (γ) | Initial ε | ε Decay |
|---|---|---|---|---|
| Baseline | 0.1 | 0.95 | 1.0 | 0.995 |
| Fast Learning Rate | 0.3 | 0.95 | 1.0 | 0.995 |
| Low Discount Factor (short-term focus) | 0.1 | 0.8 | 1.0 | 0.995 |
| Slow Epsilon Decay (more exploration) | 0.1 | 0.95 | 1.0 | 0.999 |
| Fast Epsilon Decay + Lower Start | 0.1 | 0.95 | 0.5 | 0.98 |

## Performance Comparison

| Configuration | Avg Reward | Avg Revenue | Inventory Utilization | States Learned | Train Time (s) |
|---|---|---|---|---|---|
| Slow Epsilon Decay (more exploration) | 13947.68 | 15388.88 | 95.1% | 1887 | 2.1 |
| Fast Learning Rate | 13765.89 | 14805.16 | 98.8% | 1688 | 2.1 |
| Baseline | 12973.98 | 14294.71 | 98.7% | 1661 | 2.1 |
| Low Discount Factor (short-term focus) | 12488.64 | 13695.24 | 99.3% | 1720 | 1.9 |
| Fast Epsilon Decay + Lower Start | 5343.00 | 8646.07 | 100.0% | 880 | 1.6 |

## Best Configuration

**Winner: Slow Epsilon Decay (more exploration)**

- Learning rate (α): 0.1
- Discount factor (γ): 0.95
- Initial epsilon: 1.0
- Epsilon decay: 0.999
- Average reward achieved: 13947.68
- Average revenue achieved: 15388.88
- Inventory utilization: 95.1%

## Notes on Interpretation

- **Avg Reward** reflects the full reward signal (revenue minus discount/unsold penalties, plus pacing bonus) — this is what the agent actually optimizes for.
- **Avg Revenue** is the raw price × units sold, useful for understanding real business impact separately from the shaped reward signal.
- **Inventory Utilization** shows what fraction of starting stock was sold by the end of the season — very low utilization suggests the agent is pricing too high and leaving stock unsold; very high utilization achieved via heavy discounting may hurt revenue, which the Avg Revenue and discount penalty in Avg Reward would reveal.
- Evaluation used greedy (no-exploration) actions so results reflect the agent's actual learned policy, not random exploration noise.

## Recommendation

Going forward, the Q-Learning agent should default to the **Slow Epsilon Decay (more exploration)** configuration (α=0.1, γ=0.95, ε_start=1.0, ε_decay=0.999) for baseline training runs, based on the highest average reward achieved across 200 evaluation episodes.
