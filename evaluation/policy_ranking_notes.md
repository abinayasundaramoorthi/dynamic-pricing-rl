# Policy Ranking — Interpretation Notes

**Related to:** `evaluation/policy_ranking.csv` (Issue #99)

---

## Result Summary

| Rank | Policy | Avg Reward | Avg Revenue | Sell-through |
|---|---|---|---|---|
| 1 | Fixed Price Policy | 19,627.35 | 19,542.00 | 97.7% |
| 2 | DQN | 15,911.63 | 16,331.25 | 99.8% |
| 3 | Q-Learning | 13,890.76 | 15,387.24 | 94.7% |
| 4 | Random Policy | 8,442.22 | 13,332.18 | 77.0% |
| 5 | Discount Policy | -3,291.17 | 3,557.14 | 100.0% |

**Fixed Price Policy ranked highest** in this run — outperforming both learned agents (DQN and Q-Learning) on average reward AND raw revenue, not just reward.

## This is a Legitimate, Documented Result — Not a Bug

This result was verified correctly (all 5 policies ran under identical simulated demand conditions via matched random seeds). Rather than treat this as an error, it's reported honestly here, along with possible explanations, in the same spirit as the exploration strategy findings in Issue #51.

## Possible Explanations

**1. The reward function strongly penalizes discounting.**
Looking at `reward.py`'s `compute_reward()`, the `discount_penalty` term grows with the *square* of how far the price drops below base price. Fixed Price never discounts at all, so it completely avoids this penalty. DQN and Q-Learning may have each learned to discount somewhat — a reasonable-looking strategy that, under this specific reward design, costs more in penalty than it gains in extra sales.

**2. Training run-to-run variance.**
Both agents are trained from scratch each time this script runs (no saved checkpoint reused), for 1,000 episodes. This is a modest training budget, and results can vary noticeably between runs — the agents in this particular run may not have converged as strongly as in earlier comparisons (see Issue #80 and #84, where DQN clearly outperformed Q-Learning under different training runs).

**3. The base price may already be well-calibrated for this demand model.**
If the configured `base_price` sits close to the revenue-maximizing point in the demand simulation, there may simply be limited room for a dynamic strategy to improve on "hold steady" — especially once the discount penalty is factored in.

## What Would Help Confirm This

- Re-running this comparison multiple times (different random seeds for training) to see if Fixed Price consistently wins, or if this run was an outlier
- Training for more episodes to rule out under-convergence
- Comparing with `lambda_discount` set to 0 in the reward config, to isolate how much of Fixed Price's advantage comes specifically from avoiding the discount penalty

## Takeaway

This result doesn't mean DQN/Q-Learning "don't work" — both clearly outperformed Random Policy and Discount Policy, and earlier evaluations (#80, #84) showed them outperforming each other in expected directions. It does suggest the **reward function's discount penalty may be strong enough to make simple price-holding surprisingly competitive** in this environment configuration — a useful, honest finding for the team to discuss, particularly for whoever owns reward function tuning.