# RL Agent vs. Random Baseline: Comparison Summary

## Objective
Compare the trained Q-Learning agent's performance against the random pricing baseline, to quantify the concrete value added by reinforcement learning over having no pricing strategy at all.

## Method
- Trained agent: Q-Learning (`agents/q_table.pkl`), evaluated with exploration disabled (pure exploitation of the learned policy)
- Baseline: Random pricing, using Gymnasium's own `action_space.sample()` (no strategy)
- **200 evaluation episodes per policy**, same random seed range for both, for a fair, statistically reliable comparison
- Evaluated using the standardized `evaluation/metrics.py` utility (Average Episode Reward, Revenue per Episode, Inventory Utilization, Price Trends)

## Results

| Metric | Random Baseline | Q-Learning Agent | Difference |
|---|---|---|---|
| **Average Episode Reward (RL reward vs. random reward)** | $10,375.75 | $12,914.90 | **+$2,539.15 (+24.5%)** |
| Std Dev of Reward | $1,138.69 | $737.48 | Lower — more consistent |
| Min Revenue (worst episode) | $7,120.00 | $10,810.00 | Higher floor |
| Max Revenue (best episode) | $13,610.00 | $15,100.00 | Higher ceiling |
| **Revenue Generated (avg)** | $10,375.75 | $12,914.90 | +$2,539.15 |
| **Inventory Remaining (avg, unsold)** | 0.0 / 100 | 0.0 / 100 | No difference — both sell out |
| Sellout Rate | 100.0% | 100.0% | Identical |
| **Pricing Behaviour — Early Season Avg Price** | $126.56 | $120.97 | Q-Learning starts slightly lower |
| **Pricing Behaviour — Mid Season Avg Price** | $124.15 | $139.22 | Q-Learning rises |
| **Pricing Behaviour — Late Season Avg Price** | $126.23 | $158.90 | Q-Learning rises further |

## Comparison by Task Area

### RL Reward vs. Random Reward
The Q-Learning agent achieved a **24.5% higher average episode reward** than random pricing ($12,914.90 vs $10,375.75), a gain of $2,539.15 per season on average. The agent's reward distribution is also **tighter** (std dev $737.48 vs $1,138.69) — meaning it doesn't just perform better on average, it's also more *consistent*, with a notably higher worst-case outcome ($10,810 vs $7,120 in the worst episode observed).

### Revenue Generated
Since reward and revenue are identical in this environment (reward = units sold × price), the revenue comparison mirrors the reward comparison exactly: **+$2,539.15 average revenue gain per season**, a **24.5% improvement**.

### Inventory Remaining
Both policies fully sell out (100% sellout rate, 0 units remaining on average) under this environment's demand model, due to the late-season demand surge from the time-dependent demand function — this surge is strong enough that even random pricing eventually clears inventory. This means the Q-Learning agent's advantage comes entirely from **charging better prices**, not from selling more units (since both sell the same 100/100).

### Pricing Behaviour
This is where the real difference becomes visible. Random pricing shows a flat, unstructured pattern across the season (~$124-127 throughout, as expected from unweighted random sampling). **The Q-Learning agent shows a clear, learned upward trend** — starting slightly lower early in the season ($120.97) and rising through mid-season ($139.22) to a notably higher late-season average ($158.90). This indicates the agent learned to recognize and exploit the environment's time-dependent demand surge (customers becoming more price-tolerant as the deadline approaches), a pattern a random policy has no way to discover or act on.

## Summary

The trained Q-Learning agent clearly outperforms the random baseline: **+24.5% higher average revenue, lower variance, and a materially higher worst-case outcome**, achieved not by selling more inventory (both sell out completely) but by **learning when to charge more** — raising prices later in the season to capture the value of increasingly urgent late-arriving demand. This confirms the RL approach is learning genuine, exploitable structure in the pricing problem rather than matching random performance by chance.

## Acceptance Criteria Check

- ✅ **Trained agent completes evaluation episodes** — 200 episodes run successfully with no errors
- ✅ **Comparison metrics generated** — reward, revenue, inventory, and pricing behaviour all computed via `evaluation/metrics.py`
- ✅ **Results clearly show difference between policies** — +24.5% reward improvement and a clearly distinct pricing pattern (rising vs. flat) documented above
