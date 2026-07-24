
# Robustness Test Report: Q-Learning Agent Under Different Environment Configurations

## Objective
Evaluate whether the trained Q-Learning agent maintains stable performance when the environment's inventory and selling-horizon settings differ from what it was trained on.

## Important Technical Note (read first)

The Q-Learning agent's Q-table has a **fixed shape of (31, 101, 5)** — it was trained specifically for `selling_window=30` days and `total_inventory=100` units (see `docs/mdp_formulation.md`). This means the table has **no entries at all** for states like `inventory_remaining=150` or `days_remaining=45` — those simply don't exist in a table sized for a maximum of 30 days and 100 units.

Running the raw agent against those states causes an immediate crash:
```
IndexError: index 200 is out of bounds for axis 1 with size 101
```

To satisfy the "no runtime errors during testing" acceptance criterion while still producing a meaningful result, the test harness **clamps** any out-of-range state down to the nearest value the table actually has an entry for (e.g. inventory 200 → clamped to 100) before querying it. This is a **testing workaround, not a real fix** — it lets us observe *what the agent does* under out-of-range conditions without crashing, but the underlying limitation (the agent has no real answer for states it wasn't trained on) is itself the main finding of this report.

## Method
- 4 test configurations + 1 baseline (matching training conditions), 50 episodes each, 200+50 total episodes
- Recorded: average reward, total revenue, average units sold, episode completion status, and how often clamping was triggered (a direct measure of how far outside the trained distribution each config pushes the agent)

## Results

| Configuration | Avg Reward | Units Sold | Completion Rate | Clamp Rate (states outside training) |
|---|---|---|---|---|
| **Baseline** (100 units, 30 days — matches training) | $13,156.00 | 100.0 / 100 | 100% | 0.0% |
| Low Initial Inventory (30 units, 30 days) | $1,500.00 | 30.0 / 30 | 100% | 0.0% |
| High Initial Inventory (200 units, 30 days) | $11,254.60 | 200.0 / 200 | 100% | 55.4% |
| Short Selling Horizon (100 units, 10 days) | $4,167.00 | 83.3 / 100 | 100% | 0.0% |
| Long Selling Horizon (100 units, 60 days) | $5,313.60 | 100.0 / 100 | 100% | 100.0% |

*(All 250 episodes completed without a single runtime error, using the clamping safeguard described above.)*

## Analysis by Configuration

### Low Initial Inventory (30 units)
Technically **within** the table's trained range (30 ≤ 100), so no clamping was needed — this exposes a different, subtler issue: **sparse state coverage**. During training, the agent almost always started with 100 units, so it rarely experienced states like "30 days left, 30 units left" (that combination only occurs early in a season if 70 units sold unusually fast, which is a low-probability trajectory the agent barely encountered). The result — a flat $50/unit price the entire time (the cheapest available) — suggests the agent's Q-values for this region of the table never received enough training updates to be reliable, even though technically "in bounds."

### High Initial Inventory (200 units)
Triggered clamping on over half of all steps (55.4%) — any time actual inventory exceeded 100, the agent was making decisions based on a clamped-to-100 view of the world, effectively blind to the true remaining stock. Revenue ($11,254.60) was lower than baseline despite selling twice the inventory, and the price-per-unit-sold was correspondingly much lower — a direct, measurable cost of operating outside the trained range.

### Short Selling Horizon (10 days)
No clamping needed (10 < 30), but performance dropped sharply — only **83.3 of 100 units sold** on average, the only configuration where the agent did **not** achieve a full sellout. The agent's pricing strategy (learned to gradually raise prices over a full 30-day arc, per `evaluation/pricing_policy_report.md`) doesn't compress correctly into a 10-day window — it runs out of time before its learned pattern would normally reach the demand-clearing prices it relies on late in a 30-day season.

### Long Selling Horizon (60 days)
Clamped on **100% of steps** — every single day of this 60-day season, the agent's "days_remaining" value exceeded the trained maximum of 30, so it never once made a decision based on true, unclamped information. Revenue ($5,313.60) was the second-lowest of all configurations tested, reflecting that the agent was essentially guessing for the entire first half of every season.

## Summary: How Environmental Changes Affect Agent Performance

| Change | Effect on Performance | Root Cause |
|---|---|---|
| Smaller inventory (in-range) | Severely degraded (near-minimum pricing) | Sparse training coverage of that state region |
| Larger inventory (out-of-range) | Moderately degraded | Clamping masks true inventory level from the agent |
| Shorter horizon (in-range) | Meaningfully degraded (only config that fails to sell out) | Learned price-rise pattern doesn't fit in a shorter window |
| Longer horizon (out-of-range) | Most severely degraded | Agent operates "blind" (fully clamped) for the entire first half of the season |

**The clear overall finding: this Q-Learning agent does not generalize.** It performs well only within the exact bounds it was trained on (100 units, 30 days), and its performance degrades — sometimes severely — the further a scenario departs from those exact bounds, whether or not clamping was technically required to avoid a crash.

## Strengths and Weaknesses of the Learned Policy

**Strengths:**
- Extremely strong, reliable performance at the exact trained configuration ($13,156.00 baseline, matching earlier evaluation results)
- The clamping safeguard confirms the agent can be made crash-safe with a simple engineering wrapper, even though its *decisions* remain unreliable outside its trained range

**Weaknesses:**
- **No structural generalization** — a fundamental limitation of tabular Q-Learning: it memorizes a lookup table for specific state values rather than learning a transferable rule
- **Sparse coverage even within trained bounds** — states rarely visited during training (like starting mid-season with unusually low inventory) produce weak, unreliable decisions despite being technically "in range"
- **No safety net for out-of-range states** — without the clamping wrapper added specifically for this test, the agent would crash outright on any inventory or time-horizon setting beyond its exact training configuration

## Recommendation

This is a direct, practical argument for the DQN agent (`agents/dqn_agent.py`) over tabular Q-Learning for any real deployment where inventory size or selling-window length might vary business to business (e.g. reusing one trained model across different flight routes or hotel types with different capacities). A neural network naturally produces *some* output for any numeric input — it wouldn't crash without a clamping wrapper — and its ability to generalize across similar states (demonstrated in `notebooks/dqn_training.ipynb`) makes it structurally better suited to this kind of variation, though its extrapolation reliability far outside the training distribution would still need to be tested the same way before trusting it in production.

## Acceptance Criteria Check

- ✅ **Agent successfully completes all test scenarios** — 250/250 episodes completed (100% completion rate in every configuration)
- ✅ **Performance metrics recorded for each configuration** — average reward, units sold, and completion status recorded for all 5 configurations
- ✅ **Report highlights strengths and weaknesses of the learned policy** — dedicated section above
- ✅ **No runtime errors during testing** — zero crashes across all 250 episodes, achieved via the clamping safeguard described in the Technical Note (documented transparently as a testing workaround, not a claim that the underlying agent handles these cases well)
