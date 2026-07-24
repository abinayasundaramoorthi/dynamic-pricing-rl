# Baseline Comparison: Q-Learning Agent vs. Random Policy

## Objective
Compare the trained Q-Learning agent with the random policy baseline to verify that the agent has actually learned an effective pricing strategy, rather than performing no better than chance.

## Method
- **Random Policy:** price chosen completely at random each day via Gymnasium's `action_space.sample()` — no strategy
- **Q-Learning Policy:** the trained agent (`agents/q_table.pkl`) acting with exploration disabled (pure exploitation of its learned policy)
- **200 episodes per policy**, same seed range (444-643) for both, ensuring both policies were tested against the exact same sequence of simulated market conditions
- Metrics computed via the standardized `evaluation/metrics.py` utility

## Results

| Metric | Random Policy | Q-Learning Policy |
|---|---|---|
| Average Reward | $10,439.25 | $12,920.15 |
| Average Revenue | $10,439.25 | $12,920.15 |
| Average Units Sold | 100.0 / 100 | 100.0 / 100 |
| Average Inventory Remaining | 0.0 | 0.0 |
| Sellout Rate | 100% | 100% |

Full metrics, including standard deviation, min/max revenue, and price-trend breakdowns by season phase, are saved in `evaluation/comparison_metrics.csv`.

## Comparison

**Average Reward:** The Q-Learning policy achieved a **+23.8% higher average reward** than the random baseline ($12,920.15 vs. $10,439.25) — a gain of $2,480.90 per season on average.

**Revenue:** Since reward and revenue are identical in this environment (reward = units sold × price), the revenue comparison mirrors the reward result exactly: **+$2,480.90 average revenue improvement**, a **23.8% uplift**.

**Inventory Remaining:** No difference — both policies achieve a 100% sellout rate on average, leaving 0 units unsold. This means the Q-Learning agent's revenue advantage comes entirely from *how it prices*, not from selling additional inventory the random policy failed to sell.

## Observations

- The consistent, sizeable reward and revenue uplift (+23.8%) across 200 independent episodes confirms the Q-Learning agent has genuinely **learned an effective pricing strategy** — this is not noise or a lucky sample, since both policies were tested against identical simulated conditions episode-for-episode.
- Because both policies reach 100% sellout, the entire performance gap is attributable to **price selection quality**, not sales volume — the Q-Learning agent charges smarter prices at the right points in the season (as detailed in `evaluation/pricing_policy_report.md`, it learns to raise prices later in the season rather than pricing flat/randomly throughout).
- This result validates that the Q-Learning training process (`agents/q_learning_agent.py`, `notebooks/q_learning_training.ipynb`) produced a working policy, and confirms the environment and reward signal are structured correctly — a poorly-designed environment or reward function often fails to produce any meaningful separation between a trained agent and a random baseline, which is not what we observe here.

## Conclusion

The trained Q-Learning agent significantly and consistently outperforms the random policy baseline, verifying that the agent has learned genuine, effective pricing behavior rather than matching random performance. The +23.8% average reward improvement is achieved purely through smarter pricing decisions, since both policies sell out 100% of inventory on average.

## Acceptance Criteria Check

- ✅ **Comparison completed successfully** — both policies executed across 200 episodes each with no errors
- ✅ **Metrics exported** — `evaluation/comparison_metrics.csv` generated with full per-policy statistics
- ✅ **Summary prepared** — this document
