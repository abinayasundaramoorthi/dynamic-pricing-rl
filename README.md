# Dynamic Pricing with Reinforcement Learning

> **Update — single Flask app, Streamlit removed.** The project now runs
> as **one Flask application**: `python -m dashboard.web_dashboard.app`.
> The original Streamlit dashboard (`dashboard/dashboard_app.py`) has
> been removed — its data-loading logic lives on in
> `dashboard/data_contract.py`. The former second Flask app
> (`web_app/app.py`, for Human-Feedback-Learning + Digital-Twin
> features) is now merged in as the `legacy` blueprint, mounted at
> `/legacy` (`/legacy/human-feedback`, `/legacy/digital-twin`). The
> bundled demand-shock event calendar
> (`additional_features/demand_shock_detection/sample_events.csv`) now
> contains real Indian festival/holiday dates (Republic Day, Holi,
> Ganesh Chaturthi, Navratri, Dussehra, Diwali, Christmas & New Year)
> instead of placeholder template rows — see
> `dashboard/web_dashboard/OPTIMIZATION_NOTES.md` for the full history
> of these changes.

**Travel & Hospitality — Learning Optimal Pricing Policies for Perishable Inventory**

A Reinforcement Learning system that learns pricing policies for finite,
perishable inventory — airline seats, hotel rooms — by interacting with a
simulated booking market. The agent optimizes total season revenue rather
than any single transaction, adapting its price to remaining inventory,
demand signals, and the shrinking time before departure/check-in.

The project was built as a 4-week sprint, moving from problem formulation
to a fully validated, end-to-end system: a custom Gymnasium environment,
two learned agents (tabular Q-Learning and a Deep Q-Network), three
heuristic baselines, a 1,000-episode statistical evaluation framework, and
a business-facing Flask dashboard.

> **Project status:** Week 4 complete — validated end to end. One honest
> open finding is carried forward rather than hidden: in the current
> comparison, the fixed-price baseline outperforms both learned agents on
> revenue (see [Results](#results) and [`reports/release_notes.md`](reports/release_notes.md)
> for the full explanation before treating any number here as final).

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Weekly Progress](#weekly-progress)
  - [Week 1 — Environment & MDP Foundation](#week-1--environment--mdp-foundation)
  - [Week 2 — Baseline RL Pipeline & Q-Learning](#week-2--baseline-rl-pipeline--q-learning)
  - [Week 3 — Deep Q-Network Integration](#week-3--deep-q-network-integration)
  - [Week 4 — Evaluation, Dashboard & Release](#week-4--evaluation-dashboard--release)
- [Repository Structure](#repository-structure)
- [Known Issues & Open Items](#known-issues--open-items)
- [Documentation Index](#documentation-index)
- [Roadmap](#roadmap)

## What to do with this zip

Extract it into your project root. It will merge two things into your existing tree:
## Overview

| | |
|---|---|
| **Problem** | Static/manual pricing for perishable inventory (flights, hotel rooms) leaves revenue on the table — it can't adapt in real time to remaining inventory, demand, and time pressure. |
| **Approach** | Model pricing as a Markov Decision Process and train agents to learn a pricing policy through simulated interaction, benchmarked against realistic heuristic strategies. |
| **State** | `[remaining_inventory, days_remaining]` |
| **Action** | Discrete price adjustment relative to current price (7 tiers, −20% to +20%) |
| **Reward** | Step revenue, minus an over-discounting penalty and a terminal unsold-inventory penalty, plus a small inventory-pacing shaping bonus |
| **Episode end** | Inventory sells out, or the booking deadline is reached |
| **Agents compared** | DQN, Tabular Q-Learning, Random, Fixed Price, Time-Based Discount |

## Architecture

```
 Problem Statement & MDP Design (Week 1)
            │
            ▼
   PricingEnvironment (Gymnasium)
   state.py · action_space.py · reward.py · demand_simulator.py
            │
   ┌────────┴─────────┐
   ▼                   ▼
Q-Learning (Week 2)   DQN (Week 3)
tabular Q-table       feedforward net + replay buffer + target network
   │                   │
   └────────┬──────────┘
            ▼
  Baselines: Random · Fixed Price · Time-Based Discount (Week 4)
            │
            ▼
  Policy Evaluation Framework — 1,000 episodes × 5 policies (Week 4)
            │
            ▼
  Streamlit Business Dashboard (Week 4)
```

## Installation

- `additional_features/` — **updates** the Feature 2 files in place (Streamlit code removed, replaced with pure-Python data functions); Features 1 and 3 (`explainable_ai_pricing/`, `demand_shock_detection/`) are unchanged from before, plus one new file (`sample_events.csv`).
- `dashboard/web_dashboard/` — **brand new** folder, the Flask app itself.

```
your-project-root/
├── (all existing files — untouched)
│   └── dashboard/
│       ├── dashboard_app.py            <- unchanged, still Streamlit, still yours
│       ├── pricing_visualizations.py   <- unchanged
│       ├── ...
│       └── web_dashboard/              <- NEW: the Flask app
│           ├── __init__.py
│           ├── app.py
│           ├── services.py
│           ├── blueprints/
│           │   ├── __init__.py
│           │   ├── main.py             (HTML page routes)
│           │   └── api.py              (JSON API routes — all real data)
│           ├── templates/
│           │   ├── base.html
│           │   ├── index.html
│           │   ├── dashboard.html
│           │   ├── recommendations.html
│           │   ├── events.html
│           │   ├── simulator.html
│           │   └── errors/{404,500}.html
│           └── static/
│               ├── css/style.css
│               └── js/main.js
│
└── additional_features/
    ├── __init__.py
    ├── explainable_ai_pricing/          <- unchanged
    ├── demand_shock_detection/          <- unchanged + sample_events.csv (new)
    ├── advanced_dashboard/               <- REWRITTEN: pure data layer now, no Streamlit
    │   ├── __init__.py
    │   ├── kpi_components.py
    │   ├── charts.py
    │   └── export_utils.py
    ├── deployment_smoke_test.py          <- updated: tests the Flask app too
    └── tests/
        ├── __init__.py
        ├── test_explainable_ai_pricing.py
        ├── test_demand_shock_detection.py
        └── test_advanced_dashboard.py     <- rewritten for the new data layer + Flask
**Verify the install:**

```bash
python -m pytest tests/ -v
```

Runs 7 checks covering replay buffer storage, sampling, capacity eviction,
defensive array copying, and warm-up-gated training. For a full pipeline
check (environment → agents → evaluation → dashboard), see
[`reports/release_notes.md`](reports/release_notes.md), section
"How to validate this release."

## Usage

### Tabular Q-Learning

```bash
python -m training.train_agent --agent q_learning --episodes 5000
```

## New dependency

**Flask** is now a dependency (not in your original `requirements.txt`):

```bash
pip install flask
```

Nothing else changed — still pandas/numpy/torch/gymnasium/matplotlib/streamlit (Streamlit stays because your *original* dashboard still uses it; the *new* dashboard does not).

## Running the new Flask dashboard

From your project root:
Trains a feedforward network with experience replay and a target network,
saves it to `agents/checkpoints/dqn_policy.pt`, reloads it, and evaluates
it the same way.

> **Note:** DQN training is not currently bit-for-bit reproducible run to
> run — PyTorch's global RNG for network initialization is unseeded.
> Different runs can produce checkpoints with meaningfully different
> evaluated performance (see [Known Issues](#known-issues--open-items)).

### Experiment comparison suite

```bash
python -m training.run_experiment
```

Runs a suite of named hyperparameter configurations, writing each one's
results to `evaluation/results/<name>/`, plus a cross-experiment
`comparison_summary.csv`.

### Full policy evaluation (all 5 strategies)

```bash
python -m dashboard.web_dashboard.app
```

Then open **http://localhost:8080**. Override the port with the `PORT` environment variable if needed.

Human-Feedback-Learning and Digital-Twin-Simulation are available in the
same app, under `/legacy` (`/legacy/human-feedback`, `/legacy/digital-twin`).

Runs 1,000 simulated booking seasons for each of DQN, Q-Learning, Random,
Fixed Price, and Time-Based Discount, scores every policy against the
project's business KPIs (revenue uplift, sell-through, spoilage), and
writes `evaluation/evaluation_results.csv` (episode-level) plus
`evaluation/policy_evaluation_summary.csv` / `.json` (aggregate).
Add `--smoke-test` for a fast 20-episode pipeline check instead.

## What's real vs. what's a template
Reads the evaluation outputs above and renders policy performance,
pricing trends, and business-KPI scorecards in the Flask dashboard
(`/dashboard`). Run the evaluation command above at least once first —
the dashboard displays what's on disk, it doesn't simulate anything itself.

### Programmatic usage

Every number in the Flask dashboard comes from one of three real sources — nothing is `Math.random()` or hardcoded:

- **Evaluation CSVs** (`evaluation/evaluation_results.csv`, `evaluation/policy_evaluation_summary.csv`) — same files your original dashboard already reads.
- **Your real trained agent checkpoints** (`agents/checkpoints/dqn_policy.pt`, `q_learning_policy.pkl`) — the "Live AI Recommendation" and "What-If Simulator" pages run actual rollouts through your real `PricingEnvironment` with these checkpoints.
- **An event calendar file** (CSV or JSON) for the Demand Shock page. No real calendar of yours exists yet, so it falls back to `additional_features/demand_shock_detection/sample_events.csv` — **explicitly labeled as a sample** in the UI (a visible banner says so) and in the API response (`"is_sample_data": true`). Point the Events page at your own file's path to use real events instead.

One design change from the reference zip you liked: the "Competitor Price War" market-condition option was removed from the simulator, because this project's demand model has no competitor-pricing mechanism to simulate — adding it would have meant fabricating a feature that doesn't actually do anything. The two market conditions that remain ("High Demand" / "Low Demand") are real: they scale the project's actual `DemandConfig.base_daily_arrival_rate` parameter.

## Verification

Both run clean against this exact codebase, including your real trained checkpoints:

```bash
python -m pytest additional_features/tests -q
# 71 passed
## Results

**Latest 1,000-episode-per-policy comparison**, run after the DQN RNG-seeding
fix described in [Known Issues](#known-issues--open-items) (see
[`evaluation/policy_evaluation_summary.csv`](evaluation/policy_evaluation_summary.csv)
for the raw output and
[`evaluation/significance_results.csv`](evaluation/significance_results.csv)
for the paired significance tests below):

| Policy | Mean Revenue (₹) | Sell-Through | Spoilage |
|---|---:|---:|---:|
| Fixed Price | 1,865,694.85 | 97.8% | 2.2% |
| DQN | 1,608,924.81 | 95.3% | 4.7% |
| Q-Learning | 1,391,544.43 | 98.2% | 1.8% |
| Random | 1,357,007.35 | 84.8% | 15.2% |
| Time-Based Discount | 1,231,777.58 | 100.0% | 0.0% |

**What changed and why it matters:** the previously committed DQN checkpoint
evaluated at a mean revenue collapsed near the bottom of the table (see
[Known Issues](#known-issues--open-items) for the root cause and the fix).
After fixing that and retraining with the *unchanged* default
hyperparameters in `configs/dqn_config.py`, DQN's evaluated mean revenue
roughly **doubled** and it moved from last place to second place, ahead of
Q-Learning, Random, and Time-Based Discount — this is the actual learning
signal the algorithm was producing all along; it just wasn't being measured
correctly before.

**The fixed-price baseline still wins on mean revenue**, and this is no
longer just an observation — it's now backed by a paired significance test
(every policy is evaluated on the identical 1,000 simulated seasons, so the
comparison is paired, not independent samples). A paired t-test and a
Wilcoxon signed-rank test were both run (`evaluation/significance_testing.py`);
fixed-price beats every other policy at p < 0.0001 with a large effect size
(Cohen's d from −1.09 vs. DQN to −2.64 vs. Time-Based Discount). This is a
real, statistically supported result — not a display bug and not
attributable to noise — and is carried forward as the project's central
open finding rather than hidden. Neither agent's hyperparameters have been
exhaustively tuned; a preliminary experiment with a larger network and
tuned learning rate (`configs.dqn_config.get_optimized_dqn_config()`) raised
DQN's evaluation-only mean revenue further (≈ ₹1.73M over 200 greedy
episodes, vs. ≈ ₹1.62M for the default config, same seed) but was not
carried into the five-policy comparison above because its 128×128 network
doesn't match the 64×64 architecture the evaluation pipeline and dashboard
currently assume — closing that gap is listed under
[Known Issues](#known-issues--open-items).

---

## Weekly Progress

### Week 1 — Environment & MDP Foundation

**Goal:** Take the project from a business problem statement to a fully
working, tested Gymnasium environment — the foundation every agent trains
against from Week 2 onward.

| Day | Focus | Outcome |
|---|---|---|
| Day 1 | Business understanding & problem definition | Business background, MDP rationale, state/action/reward design, stakeholders, KPIs, constraints, risks, and timeline documented in `reports/project_planning/problem_statement.md`. |
| Day 2 | MDP design | State, action, reward, transition, discount factor, and terminal condition formalized directly in code docstrings (`state.py`, `action_space.py`, `reward.py`). |
| Day 3 | Gym environment skeleton | `PricingEnvConfig`, `__init__()`, `reset()`, `render()` shipped; `step()` deliberately raised `NotImplementedError` so no agent could train against an incomplete environment. |
| Day 4 | Demand model & full integration | `step()` fully implemented, wiring together `state.py`, `action_space.py`, `demand_simulator.py` (Poisson-arrival / logistic-acceptance stochastic demand), and `reward.py`. Full episode loop verified end to end. |
| Day 5 | Review, refactor, docs, sign-off | Code review, folder verification, README, and sprint report. |

**Code review findings & fixes:**
- Two blocking packaging bugs caught and fixed: `init.py` → `__init__.py` (package wasn't recognized at all), and `demand_stimulator.py` → `demand_simulator.py` (mismatched import target).
- Two non-blocking static-analysis issues resolved: an unused import in `state.py`, and an unimported `Any` type hint in `action_space.py`.

**Verification — 11/11 checks passed:**
```
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
All `requirements.txt` dependencies installed and imported cleanly; the
repository built successfully with no missing or conflicting dependencies.

**Delivered:** a functionally complete, tested, stable environment —
unblocked for baseline agents, Q-Learning, and the evaluation harness.

---

### Week 2 — Baseline RL Pipeline & Q-Learning

**Sprint goal:** Develop and integrate the baseline reinforcement learning
pipeline using the custom Gymnasium pricing environment and a Q-Learning
agent.

**Completed work:**

| Area | Delivered |
|---|---|
| RL training pipeline | Environment initialization, training configuration, entry point, episode execution — a complete training workflow. |
| Tabular Q-Learning agent | Q-table initialization, epsilon-greedy action selection, Q-value update rule, learning loop, hyperparameter configuration. |
| Environment integration | `reset()`, `step()`, reward calculation, state transitions, and action execution all verified against the live agent. |
| Metrics & logging | Episode reward, revenue, inventory utilization, and episode statistics captured. |
| Evaluation | Q-Learning benchmarked against a random policy; baseline metrics and a performance summary produced. |
| Documentation | README, training config docs, hyperparameter documentation, sprint summary. |

**Components shipped this week:** `agents/q_learning_agent.py`,
`training/train_agent.py`, `configs/training_config.py`,
`evaluation/training_logs.csv`, `evaluation/performance_summary.md`,
`utils/logger.py`, `dashboard/training_dashboard.py`.

**Testing summary:** environment, state, action space, reward function,
demand simulator, training pipeline, Q-Learning agent, logging, and
evaluation — all verified complete.

**Challenges encountered:** designing an effective reward function,
selecting suitable hyperparameters, integrating environment components,
validating training behavior, and organizing a modular project
architecture.

**Delivered:** a functional baseline reinforcement learning system, ready
for advanced RL development (DQN) in Week 3.

---

### Week 3 — Deep Q-Network Integration

**Goal:** Extend the project from a working tabular Q-Learning agent to a
Deep Q-Network — the scaling path needed once richer state variables
(competitor price, customer segment, seasonality) would make a tabular
Q-table impractically large.

**Issues closed:** #62, #70, #74, #78, #82

| Day | Issue | Outcome |
|---|---|---|
| Day 1 | #62 | `agents/dqn_agent.py` (network, replay buffer, target network), `configs/dqn_config.py`, `training/train_dqn.py`, and `reports/project_planning/dqn_architecture.md` built and verified end to end. |
| Day 3 | #70 | Experience replay was found already substantively built; added the genuinely missing piece — a 7-test suite (`tests/test_replay_buffer_integration.py`) validating storage, sampling, capacity eviction, defensive array copying, and warm-up-gated training. |
| Day 4 | #74 | Documented a real full-scale training run and a genuine gotcha: a short smoke-test run can evaluate as catastrophically bad if the replay buffer never clears its warm-up threshold — looks like a bug, isn't one. |
| — | #78 | Executed a fully-monitored 2,000-episode DQN run. Found and documented two real, non-blocking findings: training-time reward variance widens late in training even as the evaluated greedy policy stays strong (a known vanilla-DQN characteristic), and training is not bit-for-bit reproducible run-to-run because PyTorch's global RNG is unseeded. |
| Day 5 | #82 | Sprint report, README update, code review/refactor, repository verification. |

**Code review & refactor:**
- Found duplicate `build_environment()` / `verify_environment_compatibility()` logic across `train_agent.py` and `train_dqn.py`; extracted both into a shared `training/env_utils.py`.
- The refactor broke a test fixture (5 of 7 tests failed) — caught by re-running the full suite rather than assuming the refactor was safe. Fixed, then re-verified all 7 tests passing.
- `pyflakes` clean across `pricing_env/`, `agents/`, `configs/`, `training/`, `tests/`.

**Validation performed:**

| Pipeline | Result |
|---|---|
| Q-Learning (`train_agent.py`) | Executes cleanly; policy saves/reloads/evaluates correctly. |
| DQN (`train_dqn.py`) | Full 2,000-episode run — 54,880 gradient steps, zero errors; reloaded policy evaluated at ~$19,500 mean revenue across two independent held-out seed sets. |
| Experience replay | 7/7 tests passing, re-verified after refactor. |
| Experiment suite | Executes cleanly, results persist correctly. |

**Delivered:** a working DQN pipeline, integrated and validated alongside
Q-Learning, with no runtime or integration errors remaining.

---

### Week 4 — Evaluation, Dashboard & Release

**Goal:** Turn the two trained agents into an actual answer to the
project's core question — does either learned policy beat a simple
pricing rule? — by building a fair large-scale evaluation framework,
adding the naive baseline strategies still open from Week 3, giving the
results a business-facing presentation, and validating and releasing the
whole system.

**Issues closed:** #90, #98, #102, #106, #110

| Day | Issue | Outcome |
|---|---|---|
| Day 1 | #90 | Built `evaluation/evaluate_policies.py`, `configs/evaluation_config.py`, and `reports/policy_evaluation_design.md`. Closed Week 3's open item by building the three baselines (`baseline_random.py`, `baseline_fixed.py`, `baseline_timebased.py`) as a prerequisite. All five policies share one `select_greedy_action(observation)` interface, so the evaluation loop was written once, not five times. |
| Day 2 | #98 | Found no DQN checkpoint existed yet, trained one, then executed 1,000 episodes × 5 policies (5,000 simulations total), 0 execution failures. Results consolidated into `evaluation/evaluation_results.csv` and `evaluation/simulation_summary.md`. |
| Day 3 | #102 | Built the Streamlit business dashboard (`dashboard/dashboard_app.py`), validated with `AppTest` across 5 scenarios (normal load, filtered views, missing data, incompatible schema) — 0 unhandled exceptions. Architecture documented in `reports/dashboard_design.md`. |
| Day 4 | #106 | Re-exercised every component together, not just reviewed. Found the DQN checkpoint evaluating well below an earlier training run's reported result — most likely explained by the already-flagged unseeded DQN RNG — and reported this honestly rather than retraining until the number looked better. Independently re-confirmed on a second machine with numerically identical results, which also surfaced and corrected one misplaced/misnamed report file. |
| Day 5 | #110 | Delivered a capstone final project report and this restructured README (Project Overview / Installation / Usage / Results / Repository Structure). |

**Code review & fixes:**
- Corrected an evaluation output filename mismatch (`evaluation/results/policy_evaluation_episodes.csv` → `evaluation/evaluation_results.csv`) against the deliverable spec, re-verified via a wipe-and-rerun.
- Caught a mid-validation regression where running `--smoke-test` briefly overwrote the committed 1,000-episode results file; caught via a row-count check (expected 5,001 lines, found 101), and the full evaluation was re-run to restore it before anything downstream was generated from the truncated file.

**Validation performed:**

| Pipeline | Result |
|---|---|
| Environment | Built, `check_env`, `reset()`/`step()` exercised directly — pass. |
| DQN / Q-Learning agents | Checkpoint loaded fresh from disk, valid action produced — pass. |
| Baselines | All three built and evaluated successfully alongside the learned agents. |
| Policy evaluation | Full 1,000 × 5 run, 0 failures, output row counts verified. |
| Dashboard | Streamlit `AppTest`, 0 exceptions across 5 scenarios. |
| Cross-platform reproducibility | Independently re-run on Windows — numerically identical results to the Linux run, to the fractional cent. |
| `pytest tests/` | 7/7 passed. |

**Delivered:** every component (environment, both agents, all three
baselines, the evaluation framework, the dashboard) built, integrated, and
independently validated — including cross-platform reproduction — with
the DQN performance gap reported transparently as the project's central
open finding rather than concealed.

---

## Repository Structure

python -m additional_features.deployment_smoke_test
# RESULT: all checks passed — safe to deploy.
```

The smoke test now also boots the real Flask app with a test client and hits every page route plus the core API routes (including a real `/api/simulate` POST), so a broken template or route is caught before you ever run `python -m dashboard.web_dashboard.app` by hand.
dynamic-pricing-rl/
│   .gitignore
│   LICENSE
│   README.md
│   requirements.txt
│
├── agents/                          # Learned pricing agents
│       q_learning.py                   # Tabular Q-Learning agent
│       dqn_agent.py                     # Deep Q-Network (replay buffer, target network)
│       checkpoints/
│           q_learning_policy.pkl           # Trained Q-table + metadata JSON
│           dqn_policy.pt                    # Trained DQN weights + metadata JSON
│
├── baselines/                       # Heuristic pricing strategies (the comparison floor)
│       baseline_random.py
│       baseline_fixed.py
│       baseline_timebased.py
│
├── configs/
│       training_config.py             # Q-Learning training configuration
│       experiment_config.py            # Named hyperparameter-comparison experiments
│       dqn_config.py                    # DQN training configuration
│       evaluation_config.py              # Policy evaluation configuration (1,000-episode runs, KPIs)
│
├── dashboard/
│       dashboard_app.py                # Streamlit business dashboard
│
├── evaluation/
│       evaluate_policies.py            # Runs all 5 policies through 1,000 simulated seasons
│       evaluation_results.csv            # Episode-level results, latest run (generated)
│       policy_evaluation_summary.csv/.json # Aggregate per-policy stats + KPI scoring (generated)
│       simulation_summary.md              # Human-readable summary of the latest run (generated)
│
├── notebooks/                       # Exploratory analysis notebooks
│
├── pricing_env/                     # The core Gymnasium environment
│       __init__.py
│       pricing_env.py                  # PricingEnvironment — MDP loop (init/reset/step/render)
│       state.py                          # MDP state representation + observation space
│       action_space.py                    # Discrete pricing action space
│       reward.py                            # Reward function
│       demand_simulator.py                    # Stochastic demand model
│
├── reports/
│       Weekly_report/
│           Week1_report_Abinaya.md
│           Week2_report_Abinaya.md
│           Week3_report_Abinaya.md
│           Week4_report_Abinaya.md
│       project_planning/
│           problem_statement.md
│           rl_problem_formulation.md
│           dqn_architecture.md
│           dqn_trianing_workflow.md
│       training_execution_summary.md
│       policy_evaluation_design.md      # Architecture of the evaluation framework
│       dashboard_design.md                # Architecture of the business dashboard
│       release_notes.md                     # Final project release notes
│
├── tests/
│       test_replay_buffer_integration.py
│
├── training/
│       env_utils.py                    # Shared environment construction/verification
│       train_agent.py                    # Q-Learning training entry point
│       run_experiment.py                  # Experiment comparison workflow
│       train_dqn.py                         # DQN training entry point
│       dqn_training_results.md                # Full-scale DQN run results record
│
└── utils/                           # Shared helper functions
```

## Known Issues & Open Items

Carried forward transparently rather than hidden. Four of the five items
below have now been investigated and either fixed or resolved with real
evidence, rather than just re-described; each entry says exactly what was
done and what's still open.

- ✅ **DQN performance regression — root cause found and fixed, not just
  "seeded and hoped."** The actual bug was in `pricing_env/transition.py`
  (`apply_transition()`), which called the *global* `np.random.poisson`
  instead of the environment's seeded `self.np_random` generator. That
  function turned out to be **dead code** — nothing in the live training
  or evaluation path imports it — so it was not the cause of the committed
  checkpoint's bad performance, but it was a real reproducibility bug in
  its own right and has been left clearly documented rather than silently
  deleted. The live demand path (`pricing_env/demand_simulator.py`) was
  independently verified to already route every random draw through the
  environment's seeded `self.np_random`, and `agents/dqn_agent.py` already
  seeds both `random` and `torch` from `DQNConfig.seed` before constructing
  the Q-network. **Empirical reproducibility check:** two full training
  runs with `--seed 42` were diffed byte-for-byte —
  `agents/checkpoints/dqn_policy.pt` was **identical** across both runs,
  and a third run with `--seed 7` produced a different checkpoint, as
  expected. DQN training is reproducible in the current code. What
  actually explains the old, much worse committed checkpoint is simply
  that it was a stale artifact from an earlier, different run — retraining
  once, cleanly, with the existing (already-correct) seeding roughly
  **doubled** DQN's evaluated mean revenue (see [Results](#results)).
  One remaining gap: `random.seed` / `torch.manual_seed` are set, but
  `np.random.seed` is not — this hasn't caused an observed problem because
  the env's demand model uses its own seeded generator, not global numpy
  state, but a `utils/model.py::set_seed()` helper that covers `random`,
  `numpy`, and `torch` (plus CUDA determinism flags) already exists in the
  repo and is currently **unused** — wiring it into `train_dqn.py`/
  `train_agent.py` would close this defensively even though no failure
  from it has been observed.
- ✅ **Statistical significance testing — implemented.**
  `evaluation/significance_testing.py` runs a paired t-test and a Wilcoxon
  signed-rank test between every policy and a reference policy (default:
  `fixed_price`), using the seed column already in
  `evaluation/evaluation_results.csv` to pair episodes correctly. Output:
  `evaluation/significance_results.csv`. Covered by
  `evaluation/test_significance_testing.py` (4 tests, passing).
- ⚠️ **Fixed-price baseline still beats both learned agents on mean
  revenue** — this is now a *statistically confirmed* finding (p < 0.0001,
  large effect size against every alternative), not just an unverified
  observation, per the significance testing above. DQN closed most of the
  gap after the retrain (see [Results](#results)) but did not close all of
  it. This remains the project's central open research question.
- ⚠️ **Hyperparameter tuning — partially done, not fully integrated.**
  `configs/dqn_config.py` already contains a tuned configuration
  (`get_optimized_dqn_config()`, 128×128 network, lower learning rate). A
  training run with it was executed as part of this pass and evaluated
  higher than the default config on a 200-episode greedy check, but it was
  **not** carried into the five-policy comparison table because
  `configs/evaluation_config.py`'s `dqn_hidden_layer_sizes` (and the Flask
  dashboard's checkpoint loading) are hardcoded to the 64×64 architecture —
  loading the 128×128 checkpoint through the existing evaluation pipeline
  fails with a `state_dict` shape mismatch. Next step: either add an
  architecture field to the evaluation/dashboard config so it reads the
  checkpoint's own shape, or standardize on one architecture end-to-end.
  Q-Learning has not been separately tuned.
- **`notebooks/` and `utils/`** remain reserved for future use — `utils/`
  specifically now also contains the unused `set_seed()` helper noted
  above, which is a genuine candidate for near-term use rather than
  permanently reserved.

## Documentation Index

| Document | Purpose |
|---|---|
| `reports/project_planning/problem_statement.md` | Original business/technical problem definition |
| `reports/project_planning/rl_problem_formulation.md` | MDP formalization |
| `reports/project_planning/dqn_architecture.md` | DQN network/training architecture |
| `reports/policy_evaluation_design.md` | Evaluation framework design |
| `reports/dashboard_design.md` | Dashboard architecture |
| `reports/release_notes.md` | Full release summary, including the DQN known issue |
| `reports/Weekly_report/` | Detailed week-by-week sprint reports (source for this README) |
| `docs/Documentation_Index.md` | Additional environment, model, and dashboard guides |

## Roadmap

- ✅ **Week 1:** Environment design, MDP formulation, demand/reward model.
- ✅ **Week 2:** Tabular Q-Learning, experiment comparison workflow.
- ✅ **Week 3:** Deep Q-Network, experience replay, target network.
- ✅ **Week 4:** Baseline policies, 1,000-episode evaluation framework, business dashboard, full validation.
- ⏭ **Next:** Investigate and close the DQN performance gap (seed RNG,
  retrain, re-evaluate) before any production consideration; add
  statistical significance testing to the evaluation framework.

---

*Built over a 4-week sprint. See `reports/Weekly_report/` for the complete, day-by-day source reports this README summarizes.*
