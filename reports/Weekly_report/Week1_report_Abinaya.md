# Week 1 Report — Travel & Hospitality Dynamic Pricing (Reinforcement Learning)

**Sprint:** Week 1, Days 1–5
**Status:** Complete
**Prepared by:** Abinaya

---

## 1. Objective Recap

Week 1's goal was to take the project from a business problem statement to a fully working, tested Gymnasium environment — the foundation every agent (baselines, Q-Learning, DQN) in Week 2 onward will train against.

---

## 2. Day-by-Day Summary

| Day | Planned Task | Actual Outcome |
|---|---|---|
| Day 1 | Business understanding & problem definition | `reports/Problem_Statement.md` completed — business background, MDP formulation rationale, state/action/reward design, stakeholders, KPIs, constraints, risks, and timeline. |
| Day 2 | MDP design | MDP formalization (state, action, reward, transition, discount factor, terminal condition) was carried into code directly as module-level docstrings in `state.py`, `action_space.py`, and `reward.py`, rather than as a separate standalone document. **Open item:** a dedicated `reports/MDP_Design.md` referenced in the original Week 1 plan was not produced as a standalone artifact — flagged below under Open Items. |
| Day 3 | Gym environment skeleton | `pricing_env/pricing_env.py` foundation shipped: `PricingEnvConfig`, `PricingEnvironment.__init__()`, `reset()`, `render()`, and a placeholder `step()` that raised `NotImplementedError` by design, so no agent could silently train against an incomplete environment. |
| Day 4 | Demand function + environment integration | `step()` fully implemented, integrating four independently-scoped modules: `state.py` (observation/state ownership), `action_space.py` (discrete pricing actions), `demand_simulator.py` (Poisson-arrival / logistic-acceptance stochastic demand model), and `reward.py` (revenue, discount-penalty, unsold-penalty, pacing-bonus reward function). Full episode loop verified end to end. |
| Day 5 | Code review, merge, refactor, docs, folder check, report | Completed — see Sections 3–6 below. |

---

## 3. Code Review Findings

A static-analysis pass (pyflakes) plus a manual review of every module in `pricing_env/` was performed prior to merge. Two categories of issues were found and corrected:

### 3.1 Filename/packaging bugs (blocking)

These were caught when reviewing the files as they existed on disk, prior to merge:

| Issue | Impact | Fix |
|---|---|---|
| Package init file was named `init.py` instead of `__init__.py` | Python did not recognize `pricing_env/` as a package at all; every relative import (`from .state import ...`, etc.) failed, and `from pricing_env import PricingEnvironment` raised `ImportError` | Renamed to `__init__.py` |
| Demand module was named `demand_stimulator.py` instead of `demand_simulator.py` | Every other module imports `from .demand_simulator import ...`; with the mismatched filename, that import could never resolve | Renamed to `demand_simulator.py` |

Both were verified by attempting the exact import that downstream agent code will use (`from pricing_env import PricingEnvironment`) before and after the fix — it failed with `ImportError` before, and succeeded after.

### 3.2 Static analysis findings (non-blocking, code quality)

| File | Issue | Fix |
|---|---|---|
| `state.py` | `dataclasses.field` imported but never used | Removed the unused import |
| `action_space.py` | `build_action_space(config: "Any")` referenced `Any` as a string-quoted forward reference without importing it, relying on a `# noqa: F821` suppression | Properly imported `Any` from `typing` and removed the string quoting and suppression comment |

Neither of these affected runtime behavior — both were caught by static analysis rather than test failures — but leaving them in place would have set a bad precedent for code quality as more contributors touch this codebase in Week 2.

---

## 4. Refactoring Performed

Beyond the two fixes above, no structural refactoring was judged necessary. The module boundaries established in Day 4 (`state.py` / `action_space.py` / `reward.py` / `demand_simulator.py` / `pricing_env.py`) were reviewed against the single-responsibility rationale documented in each file's module docstring and found to still hold:

- `state.py` remains the single owner of the observation space definition.
- `action_space.py` remains the single owner of action → price conversion, so every future agent (baselines, Q-Learning, DQN) prices identically for a given action index.
- `reward.py` and `demand_simulator.py` remain fully decoupled from Gymnasium and from each other, confirmed by their lack of cross-imports — both are independently unit-testable in isolation, which will matter when the Data Scientist role tunes reward weights or demand parameters ahead of Week 2 training.

No circular imports, no duplicated logic, and no dead code were found elsewhere in the package.

---

## 5. Folder Organization Verification

Verified against the actual repository tree:

```
dynamic-pricing-rl/
│   .gitignore
│   LICENSE
│   README.md
│   requirements.txt
│
├───agents/
├───baselines/
├───configs/
├───dashboard/
├───evaluation/
├───notebooks/
├───pricing_env/
│       __init__.py
│       pricing_env.py
│       state.py
│       action_space.py
│       reward.py
│       demand_simulator.py
├───reports/
│       Problem_Statement.md
│       Week1_Report.md
├───tests/
│       verify_day4_environment.py
└───utils/
```

`pricing_env/` is complete and internally consistent — all five module files present, correctly named, and correctly cross-referenced. `tests/` and its verification script are in place ahead of the Week 1 plan (originally scoped as part of Day 5 environment integration testing on Day 3–5, delivered here as a standalone, CI-ready script). `reports/` now contains both the Day 1 problem statement and this report.

---

## 6. Verification Results

The full verification suite (`tests/verify_day4_environment.py`) was re-run against the corrected, reviewed code prior to sign-off:

```
11/11 checks passed:
[PASS] Construction & reset() shape/dtype
[PASS] step() before reset() raises
[PASS] Invalid action raises
[PASS] Single step() returns well-formed 5-tuple
[PASS] step() after termination raises
[PASS] Episode always terminates, state stays non-negative
[PASS] Units sold never exceeds pre-step available inventory
[PASS] Same seed -> reproducible trajectory
[PASS] Revenue accounting is internally consistent
[PASS] Gymnasium check_env() API compliance
[PASS] Demand responds to price in the correct direction
```

Additionally confirmed:
- Every package listed in `requirements.txt` (`gymnasium`, `numpy`, `pandas`, `matplotlib`, `seaborn`, `torch`) installs and imports cleanly — the repository **builds successfully** with no missing or conflicting dependencies.
- `pyflakes pricing_env/` returns clean (no warnings) after the Section 3.2 fixes.

---

## 7. Acceptance Criteria — Status

| Criterion | Status | Evidence |
|---|---|---|
| All modules integrated | ✅ Met | `pricing_env.py` imports and calls into `state.py`, `action_space.py`, `reward.py`, and `demand_simulator.py`; full episode runs end to end (Section 6). |
| Repository builds successfully | ✅ Met | All `requirements.txt` dependencies install and import cleanly; package imports without error post-fix. |
| Documentation updated | ✅ Met | `README.md` updated (setup instructions, repo structure, verification instructions, Week 1 status); this report added to `reports/`. |

---

## 8. Open Items Carried Into Week 2

- **`reports/MDP_Design.md`** was referenced in the original Week 1 timeline as a standalone Day 2 deliverable but was not produced separately — the MDP formalization exists in code docstrings and `Problem_Statement.md` instead. Recommend either producing it retroactively for completeness or explicitly closing this out as "superseded by in-code documentation" before Week 2 kickoff.
- Reward function weights (`lambda_unsold`, `lambda_discount`, `lambda_balance` in `RewardConfig`) and demand parameters (`DemandConfig`) are currently set to reasonable Phase-1 defaults but have not yet been empirically calibrated against baseline agent behavior — this is explicitly scoped as Week 2 work (baseline agents + Q-Learning).
- `tests/verify_day4_environment.py` is a smoke-test script, not a `pytest` suite. Converting it to formal `pytest` test cases under `tests/` would let it run automatically in CI going forward.

---

## 9. Readiness for Week 2

The environment is functionally complete, tested, and stable. Week 2 can proceed directly to:
1. Implementing baseline agents (`baselines/baseline_fixed.py`, `baselines/baseline_rulebased.py`) against `PricingEnvironment`.
2. Implementing tabular Q-Learning (`agents/q_learning.py`) against the same environment.
3. Building the evaluation harness (`evaluation/evaluate_agents.py`) to compare them.

No environment-level blockers remain.