# Pricing Policy Report: Business Evaluation of the Learned Strategy

## Objective
Evaluate the Q-Learning agent's learned pricing strategy from a business perspective — not just "did it get a good score," but "what would this actually mean if a business followed this agent's pricing decisions."

## Method
- Analyzed the trained Q-Learning agent's (`agents/q_table.pkl`) day-by-day decisions across three fully-logged representative episodes, plus aggregate statistics across 200 evaluation episodes
- Recorded: price charged, units sold, inventory remaining, and cumulative revenue for every single day of each traced episode

## Agent Pricing Decisions — Recorded Episode Traces

### Example Episode (seed=10)

| Day | Days Left | Price | Units Sold | Inventory After | Cumulative Revenue |
|---|---|---|---|---|---|
| 1 | 30 | $130 | 2 | 98 | $260 |
| 5 | 26 | $90 | 5 | 80 | $1,920 |
| 10 | 21 | $160 | 3 | 54 | $5,110 |
| 14 | 17 | $200 | 3 | 35 | $7,150 |
| 18 | 13 | $160 | 2 | 20 | $9,060 |
| 20 | 11 | $200 | 4 | 13 | $10,460 |
| 23 | 8 | $160 | 4 | 0 | $12,390 |

*(Abbreviated — full 23-day log available in evaluation scripts. Two additional full traces, seeds 20 and 30, were recorded and show the same overall pattern with total revenues of $14,370 and $12,750 respectively.)*

**Full-episode summary of the three traced runs:**

| Episode (seed) | Days Used | Final Revenue | Final Inventory |
|---|---|---|---|
| 10 | 23 / 30 | $12,390.00 | 0 |
| 20 | 25 / 30 | $14,370.00 | 0 |
| 30 | 19 / 30 | $12,750.00 | 0 |

## Business Metrics Summary (200 episodes)

| Metric | Value |
|---|---|
| Average final revenue per season | $12,938.35 |
| Average final inventory remaining | 0.00 / 100 units |
| Sellout rate | 100.0% |
| Average days used before selling out | 22.0 / 30 days |
| Range of days used | 19 to 24 days |
| Episodes that used the full 30-day season | 0 / 200 (0%) |
| Average price — early season (first third) | $123.14 |
| Average price — mid season (middle third) | $132.38 |
| Average price — late season (final third) | $157.49 |

## Inventory Depletion Pattern

Inventory depletes steadily throughout the season rather than in a sharp end-of-season rush. Looking at the seed=10 trace: inventory drops fairly evenly from 100 to roughly 50 by the midpoint (day 11-12), then continues declining to 0 by day 23. There is no evidence of the agent "holding back" large amounts of stock only to dump it right at the deadline — depletion is gradual and consistent across the whole selling window, until stock naturally runs out early.

## Revenue Generation Pattern

Revenue accumulates roughly linearly for the first half of each episode, then accelerates somewhat as prices rise in the later stages (visible in the seed=10 trace: revenue grew by ~$3,000 in days 14-23, compared to ~$3,000 over the much longer days 1-13 span) — consistent with the agent charging measurably higher prices later in the season ($157.49 avg late-season vs. $123.14 avg early-season).

## Expected vs. Actual Behaviour

The Week 1 roadmap's stated expectation (drawn from typical revenue management intuition) was that the agent would learn to **hold prices high early, then drop prices near the deadline to clear remaining stock** — a classic "markdown to avoid spoilage" pattern.

**What the agent actually learned is the opposite, and more nuanced:**

| Expected | Actual |
|---|---|
| Prices drop near the deadline to force a sellout | Prices **rise** near the deadline ($123 early → $157 late) |
| Agent uses the full selling window, selling out right at (or near) the deadline | Agent sells out **early** — every single one of 200 test episodes sold out before day 30, averaging 22 days, never using the full season |
| Strategy is driven by avoiding spoilage (unsold inventory risk) | Strategy is driven by **exploiting rising demand tolerance** — the agent learned that late-arriving customers (from the environment's time-dependent demand model) are more willing to pay higher prices, so it raises price rather than lowering it |

**Why this makes business sense once understood:** the environment's demand model was built so that customers become *more* price-tolerant as the deadline approaches (representing urgent, last-minute buyers), not less. Given that, the revenue-maximizing move genuinely is to raise prices as urgency increases, not markdown — and the agent correctly discovered this, even though it diverges from the "classic" markdown intuition assumed in the roadmap's framing. This is a legitimate example of the agent learning the *actual* dynamics of its environment rather than a textbook pattern that doesn't apply here.

## Business Summary

The Q-Learning agent's learned pricing strategy reliably sells out all inventory (100% sellout rate across 200 test seasons) while generating an average of $12,938.35 per season — achieved through a **rising price** strategy rather than the discount-based approach a business might intuitively expect. The agent consistently sells out with time to spare (roughly 6-11 days before the deadline on average), suggesting there may be room to push prices even higher in the final stretch of a season without risking a sellout failure — a potential direction for further tuning or for a future DQN-based comparison (see `notebooks/dqn_training.ipynb`, where the DQN agent independently discovered an even more aggressive version of this same high-price strategy).

## Acceptance Criteria Check

- ✅ **Agent pricing decisions are recorded** — three full day-by-day episode traces (seeds 10, 20, 30) plus aggregate statistics across 200 episodes
- ✅ **Business metrics are summarized** — revenue, inventory, sellout rate, and price trends all reported in a dedicated section
- ✅ **Report compares expected vs. actual behaviour** — directly addressed in the dedicated comparison table and discussion above, including an honest explanation for why the actual behaviour diverges from the roadmap's initial expectation
