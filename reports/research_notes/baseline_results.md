# Baseline Results: Random Pricing Policy

## Objective
Validate the `DynamicPricingEnv` environment by running it with a random pricing policy, and establish a baseline revenue number for comparison against smarter strategies (heuristics, Q-Learning, DQN).

## Method
- Environment config: `total_inventory=100`, `selling_window=30 days`, `price_levels=[50, 90, 130, 160, 200]`, with time-dependent stochastic demand (customers become more price-tolerant as the deadline approaches)
- Policy: on each of the 30 days, a price is chosen **completely at random** from the 5 available price levels (no strategy, no adaptation to time or inventory remaining), using Gymnasium's own `action_space.sample()`
- Ran **100 independent episodes** (simulated selling seasons) for statistically reliable averages

## Results

| Metric | Value |
|---|---|
| Average revenue per episode | $10,598.30 |
| Std deviation of revenue | $1,040.68 |
| Min revenue (worst episode) | $7,860.00 |
| Max revenue (best episode) | $14,060.00 |
| Average units sold | 100.0 / 100 |
| Average units unsold | 0.0 |
| Sellout rate | 100.0% of episodes |

## Observations

- **Execution:** The random policy executed successfully across all 100 episodes with no errors, and rewards were generated on every step - confirming the environment's `reset()` and `step()` functions behave correctly end-to-end (both acceptance criteria met).
- **Full sellout under random pricing:** Unlike an earlier version of this environment (before time-dependent demand was added), every episode now sells out completely. This is because customers become significantly more price-tolerant as the deadline approaches, so demand surges late in the season regardless of price - this "urgency" effect is enough to clear inventory even under a policy with no strategy at all.
- **High variance:** Revenue still swings meaningfully between $7,860 and $14,060 (std dev ~$1,041) purely from random luck in *when* higher-demand days happen to line up with higher prices, even though every episode sells out.

## Conclusion

The environment is validated as working correctly. The random policy establishes a baseline of **$10,598.30 average revenue per episode with a 100% sellout rate**. This is the number that naive heuristics (fixed price, discount-based) and trained RL agents (Q-Learning, DQN) are compared against in later evaluation stages - see `evaluation/heuristic_results.md` and `notebooks/policy_evaluation.ipynb` for those comparisons.
