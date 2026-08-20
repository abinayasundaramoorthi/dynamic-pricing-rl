# Week 1 Report — Yogeshwaran

**Project:** Dynamic Pricing using Reinforcement Learning
**Branch:** feature/yogesh
**Issues Completed:** #5, #12, #17, #29, #33

---

## Overview

Week 1 focused on building the foundational research and core environment components for the Dynamic Pricing RL project — covering RL theory research, algorithm selection, and the first working version of the pricing environment (reward logic, state transitions, and functional testing).

---

## Issue #5 — Reinforcement Learning Research

**Objective:** Study Reinforcement Learning basics before implementation.

**Work Completed:**
- Researched and documented core RL concepts: Agent, Environment, State, Action, Reward, Policy, and Episode
- Explained the difference between traditional Machine Learning and Reinforcement Learning
- Mapped each RL concept directly to the dynamic pricing use case (e.g. Agent = pricing engine, Reward = daily revenue)

**Deliverables:**
- `reports/project_planning/Dynamic_Pricing_Research.md` — 2-page written summary
- `reports/diagrams/state_action_reward.png` — visual diagram of the Agent-Environment interaction loop

---

## Issue #12 — Reinforcement Learning Algorithms Comparison

**Objective:** Compare algorithms suitable for the project and recommend one.

**Work Completed:**
- Researched and compared four RL algorithms: Q-Learning, SARSA, Deep Q-Network (DQN), and PPO
- Evaluated each on advantages, limitations, and suitable use cases
- Recommended a phased approach: Q-Learning as a simple baseline, upgrading to DQN for the final model (with PPO noted as a possible future improvement)

**Deliverables:**
- `reports/technical_notes/RL_Algorithm_Comparison.md`

---

## Issue #17 — Implement Reward Function Module

**Objective:** Create the reward calculation logic for the environment.

**Work Completed:**
- Implemented `calculate_reward(price, units_sold, inventory_left)` — initial reward defined as revenue (price × units sold)
- Documented planned future improvements (inventory penalty, customer satisfaction penalty, overbooking risk penalty) without implementing them yet, to keep the baseline simple and testable

**Deliverables:**
- `pricing_env/reward.py`

---

## Issue #29 — Implement State Transition Logic

**Objective:** Develop the complete state transition logic after pricing actions.

**Work Completed:**
- Implemented the full step sequence: apply pricing action → simulate demand → update inventory → calculate reward → generate next state
- Added validation checks (e.g. inventory and days remaining can never go negative) to catch bugs early
- Built a `PricingEnv` wrapper class providing the standard `reset()` / `step()` / `render()` interface

**Deliverables:**
- `pricing_env/transition.py`
- `pricing_env/pricing_env.py` (initial version)

---

## Issue #33 — Environment Functional Testing

**Objective:** Verify complete environment functionality.

**Work Completed:**
- Built and executed a Jupyter notebook testing `reset()`, `step()`, and `render()`
- Ran 5 full simulated episodes using random pricing actions, logging observations at every step
- Verified no runtime errors occurred and all acceptance criteria passed

**Deliverables:**
- `notebooks/environment_testing.ipynb` (executed, with outputs saved)

---

## Additional Work (Beyond Assigned Issues)

During Week 1, the following unplanned but necessary fixes were also completed to keep the branch working correctly:

- **Folder structure correction:** Moved `reward.py` from an incorrectly created `environment/` folder into the correct `pricing_env/` folder, per team folder standards
- **Merge conflict resolution:** Diagnosed and resolved a 3-way merge conflict that had corrupted `pricing_env.py` and `reward.py` after multiple team members' work was combined — restored the team's intended advanced Gymnasium-style environment without altering its logic
- **Filename typo fix:** Corrected `demand_stimulator.py` → `demand_simulator.py` to match what the environment's `__init__.py` expected

---

## Summary

By the end of Week 1, the project had:
- A documented understanding of RL fundamentals and a justified algorithm choice (Q-Learning → DQN)
- A working, tested pricing environment with reward calculation and state transition logic
- A verified, error-free environment ready for agent development in Week 2
