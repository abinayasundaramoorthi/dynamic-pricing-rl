# Simulation Summary Template

*This is a reusable template. Copy this file and rename it per simulation run (e.g. `simulation_summary_2026-08-01.md`), then fill in each section using the output of `evaluation/export_results.py` for that run.*

---

## Run Info

- **Date / run ID:**
- **Data source:** `evaluation/simulation_results.csv` (or `.json`)
- **Policies compared:**
- **Episodes per policy:**
- **Environment config used** (from `PricingEnvConfig`):
  - `initial_inventory`:
  - `selling_horizon_days`:
  - `base_price`:
  - `price_adjustment_pct`:

## Methodology

Results were generated using `evaluation/export_results.py`'s `run_simulation()` function, which runs a given policy through `num_episodes` full seasons and records one row per episode using the `SimulationResult` schema: `episode_number`, `policy_name`, `revenue`, `reward`, `inventory_remaining`, `selling_price` (the episode's average price, since price changes daily but this schema stores one row per episode).

**Note:** `revenue` and `reward` are NOT the same number in this environment - `reward` includes discount/unsold penalties and a pacing bonus on top of raw `revenue` (see `pricing_env/reward.py`). Both are reported separately below.

## Results Summary

*(Fill in from the exported CSV/JSON, one row per policy compared)*

| Policy | Avg Revenue | Avg Reward | Avg Inventory Remaining | Avg Selling Price | Sellout Rate |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |

## Revenue Comparison

*(Which policy generated the most revenue on average? By how much, in absolute $ and %?)*

## Reward Comparison

*(Does the reward ranking match the revenue ranking, or do they diverge? If they diverge, why - e.g. one policy is racking up discount/unsold penalties that don't show up in raw revenue?)*

## Inventory and Pricing Behavior

*(How much inventory is typically left unsold per policy? Does a higher average selling price correlate with more or less unsold inventory? Is there a policy that sells out reliably vs. one that doesn't?)*

## Observations

*(Any surprising results? Does one policy dominate every metric, or are there trade-offs - e.g. higher revenue but lower sellout rate?)*

## Conclusion

*(1-2 sentence takeaway: which policy performed best overall, and what would you try next?)*

---

## Reference: Schema Definition

| Field | Type | Description |
|---|---|---|
| `episode_number` | int | Which simulated season this row represents |
| `policy_name` | str | Which policy produced this episode (lets multiple policies be compared in one file) |
| `revenue` | float | Actual dollar revenue earned that episode |
| `reward` | float | Total RL reward earned that episode - NOT the same as revenue (see Methodology above) |
| `inventory_remaining` | int | Units left unsold at the end of the episode |
| `selling_price` | float | Average price charged across the episode |
