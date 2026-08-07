# Week 2 Report: Baseline Strategies and Q-Learning

## Objective
Establish naive pricing baselines, then train a tabular Q-Learning agent to beat them.

## Work Completed

1. **Baseline Strategies** (`baselines/random_policy.py`, `baselines/heuristic_agents.py`)
   Built three baselines: Random pricing, Fixed Price ($130), and Discount $10/day. Fixed Price emerged as the strongest heuristic (~$13,000 avg revenue, 100% sellout), beating both Random and the Discount strategy - the discount agent underperformed because it dropped prices too early, before the environment's late-season demand surge made lower prices unnecessary.

2. **Tabular Q-Learning Agent** (`agents/q_learning_agent.py`)
   Implemented a Q-table mapping every (days_remaining, inventory_remaining) state to the best price action, trained via epsilon-greedy exploration over 20,000 episodes (alpha=0.1, gamma=0.99, epsilon decayed from 1.0 to 0.05).

3. **Results** (`notebooks/q_learning_training.ipynb`)
   Q-Learning reached ~$13,500 avg revenue, edging out Fixed Price. Learned strategy: raise prices later in the season (not discount), correctly exploiting the environment's late-season demand surge - the opposite pattern of the naive discount heuristic.

4. **Evaluation Framework** (`evaluation/metrics.py`, `evaluation/agent_vs_baseline.md`)
   Built reusable evaluation utilities tracking Average Reward, Revenue per Episode, Inventory Utilization, and Price Trends, generic across any agent. Confirmed Q-Learning beats Random by +24.5% average reward, with the entire gain traceable to smarter pricing decisions (both policies sell out 100% of inventory - the difference is purely in *how* they price, not how much they sell).

## Key Outcome
A trained RL agent that outperforms every naive heuristic, with the improvement traceable to a specific, interpretable learned behavior (rising prices near the deadline) rather than a black-box gain.

## Files Delivered
- `baselines/random_policy.py`
- `baselines/heuristic_agents.py`
- `agents/q_learning_agent.py`
- `evaluation/heuristic_results.md`
- `evaluation/metrics.py`
- `evaluation/agent_vs_baseline.md`
- `evaluation/pricing_policy_report.md`
- `notebooks/q_learning_training.ipynb`
