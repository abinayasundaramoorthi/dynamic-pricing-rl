# Week 2 Report — Yogeshwaran

**Project:** Dynamic Pricing using Reinforcement Learning
**Branch:** feature/yogesh
**Issues Completed:** #39, #43, #47, #51, #59

---

## Overview

Week 2 focused on building, enhancing, and tuning the Q-Learning agent — progressing from a basic working agent to a fully optimized baseline with documented hyperparameter tuning and exploration strategy experiments.

---

## Issue #39 — Implement Baseline Q-Learning Agent

**Objective:** Develop the first reinforcement learning agent to interact with the pricing environment.

**Work Completed:**
- Implemented a tabular Q-Learning agent using a Q-table (dictionary mapping state → action values)
- Implemented epsilon-greedy action selection (balancing exploration vs exploitation)
- Implemented the Q-value update rule and a full training loop
- Verified the agent initializes correctly, updates its Q-table after each step, and successfully interacts with the environment

**Deliverables:**
- `agents/q_learning_agent.py` (initial version)

---

## Issue #43 — Enhance Q-Learning Agent with Policy Management

**Objective:** Improve the baseline agent by adding policy handling and training stability features.

**Work Completed:**
- Implemented policy extraction — converting the learned Q-table into a clear "best action per state" policy
- Implemented saving and loading the learned Q-table (using pickle), so training progress can be reused without retraining from scratch
- Added performance tracking (episode rewards, epsilon history, episode lengths) for later analysis
- Verified exploration correctly reduces over episodes, and a loaded policy produces sensible, non-random actions

**Deliverables:**
- `agents/q_learning_agent.py` (updated)
- `agents/saved_policy.pkl`

**Note:** This issue also required resolving a significant merge conflict — `pricing_env.py` and `reward.py` had become corrupted when a teammate's more advanced Gymnasium-style environment was merged with an earlier version. Both files were restored to the team's intended advanced implementation, and the agent was updated to correctly interface with the Gymnasium-style API (5-value `step()` return, `action_space.n`, etc.).

---

## Issue #47 — Q-Learning Hyperparameter Optimization

**Objective:** Find suitable parameters that improve RL agent learning performance.

**Work Completed:**
- Tested 5 combinations of learning rate (α), discount factor (γ), initial epsilon, and epsilon decay
- Evaluated each configuration on average reward, average revenue, and inventory utilization using a dedicated no-exploration `evaluate()` method
- Identified the best-performing configuration (slower epsilon decay: 0.999) and documented full results

**Deliverables:**
- `evaluation/hyperparameter_search.py`
- `evaluation/hyperparameter_results.md`

---

## Issue #51 — Advanced Exploration Strategy

**Objective:** Improve agent exploration to avoid poor local pricing strategies.

**Work Completed:**
- Implemented Boltzmann (softmax) action selection as an alternative to plain random exploration — exploration now leans toward more promising actions instead of being fully blind
- Implemented exploration scheduling (a configurable warm-up period holding epsilon at its starting value before decay begins)
- Compared the original exploration strategy against the improved strategy, evaluating reward, revenue, and convergence behavior
- Documented findings honestly: the improved strategy did not outperform plain epsilon-greedy in this environment, with analysis of why

**Deliverables:**
- `agents/q_learning_agent.py` (updated with configurable exploration strategies)
- `evaluation/exploration_comparison.md`

---

## Issue #59 — Optimize Baseline Q-Learning Performance

**Objective:** Improve the learning performance of the Q-Learning agent through parameter tuning and policy optimization.

**Work Completed:**
- Set the tuned hyperparameters identified in Issue #47 (epsilon decay = 0.999) as the agent's new default values
- Ran a formal before/after comparison (original defaults vs tuned defaults), measuring average reward, revenue, and learning stability (reward variance in late training)
- Documented a modest but consistent improvement: +1.6% average reward, +1.7% average revenue

**Deliverables:**
- `agents/q_learning_agent.py` (final tuned defaults)
- `reports/q_learning_optimization.md`

---

## Additional Work (Beyond Assigned Issues)

- **Filename typo fix:** Corrected `demand_stimulator.py` → `demand_simulator.py`, including cleanup of a duplicate file left behind from the merge
- **Repo folder structure:** Created top-level `data/` folder and organized `reports/` into `project_planning/`, `technical_notes/`, and `diagrams/` per team conventions

---

## Summary

By the end of Week 2, the project had:
- A fully functional Q-Learning agent with policy save/load and performance tracking
- Documented, evidence-based hyperparameter tuning (not guesswork)
- Honest experimentation with exploration strategies, including a documented negative result
- A final, optimized baseline agent ready for comparison against more advanced methods (DQN) in later weeks
