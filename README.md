# Dynamic Pricing with Reinforcement Learning

**Travel & Hospitality — Learning Optimal Pricing Policies for Perishable Inventory**

A reinforcement learning system that learns dynamic pricing policies for finite, perishable inventory — airline seats, hotel rooms — by interacting with a simulated booking market. The agent optimizes total season revenue rather than any single transaction, adapting price to remaining inventory, demand signals, and the shrinking time before departure or check-in.

The project was delivered as a 4-week sprint, progressing from problem formulation to a fully validated, end-to-end system: a custom Gymnasium environment, two learned agents (tabular Q-Learning and a Deep Q-Network), three heuristic baselines, a 1,000-episode statistical evaluation framework, and a business-facing Flask dashboard.

**Project status:** Complete — validated end to end across all four sprint weeks. Results are reported transparently, including one open finding carried forward rather than hidden: the fixed-price baseline currently outperforms both learned agents on mean revenue (see [Results](#results)).

---

## Table of Contents

- [Team](#team)
- [Overview](#overview)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Project Management & Workflow](#project-management--workflow)
- [Weekly Progress](#weekly-progress)
- [Repository Structure](#repository-structure)
- [Known Issues & Open Items](#known-issues--open-items)
- [Documentation Index](#documentation-index)
- [Roadmap](#roadmap)
- [License](#license)

---

## Team

| Name | Role |
|---|---|
| Abinaya S | Team Leader |
| Himanshu Rawat | Team Member |
| Nagammai Subramaniyan | Team Member |
| N. Yogeshwaran | Team Member |

---

## Overview

| | |
|---|---|
| **Problem** | Static or manually-set pricing for perishable inventory (flights, hotel rooms) leaves revenue on the table — it cannot adapt in real time to remaining inventory, demand, and time pressure. |
| **Approach** | Model pricing as a Markov Decision Process (MDP) and train agents to learn a pricing policy through simulated interaction, benchmarked against realistic heuristic strategies. |
| **State** | `[remaining_inventory, days_remaining]` |
| **Action** | Discrete price adjustment relative to current price (7 tiers, −20% to +20%) |
| **Reward** | Step revenue, minus an over-discounting penalty and a terminal unsold-inventory penalty, plus a small inventory-pacing shaping bonus |
| **Episode end** | Inventory sells out, or the booking deadline is reached |
| **Agents compared** | DQN, Tabular Q-Learning, Random, Fixed Price, Time-Based Discount |

---

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
  Business Dashboard (Flask)
```

---

## Installation

**Prerequisites:** Python 3.9+

```bash
git clone https://github.com/<org>/dynamic-pricing-rl.git
cd dynamic-pricing-rl
pip install -r requirements.txt
```

**Dependencies:** `gymnasium`, `numpy`, `pandas`, `matplotlib`, `seaborn`, `torch`, `scipy`, `flask`, `pytest`

**Verify the install:**

```bash
python -m pytest tests/ -v
```

For a full pipeline check (environment → agents → evaluation → dashboard), see [`reports/release_notes.md`](reports/release_notes.md), section "How to validate this release."

---

## Usage

### Train the tabular Q-Learning agent

```bash
python -m training.train_agent --agent q_learning --episodes 5000
```

### Train the Deep Q-Network agent

```bash
python -m training.train_dqn
```

Trains a feedforward network with experience replay and a target network, saves it to `agents/checkpoints/dqn_policy.pt`, reloads it, and evaluates it.

### Run the experiment comparison suite

```bash
python -m training.run_experiment
```

Runs a suite of named hyperparameter configurations, writing each one's results to `evaluation/results/<name>/`, plus a cross-experiment `comparison_summary.csv`.

### Run the full policy evaluation (all 5 strategies)

```bash
python -m evaluation.evaluate_policies
```

Runs 1,000 simulated booking seasons for each of DQN, Q-Learning, Random, Fixed Price, and Time-Based Discount, scores every policy against the project's business KPIs (revenue uplift, sell-through, spoilage), and writes `evaluation/evaluation_results.csv` (episode-level) plus `evaluation/policy_evaluation_summary.csv` / `.json` (aggregate). Add `--smoke-test` for a fast 20-episode pipeline check instead.

### Launch the business dashboard

```bash
python -m dashboard.web_dashboard.app
```

Then open **http://localhost:8080** (override with the `PORT` environment variable). The dashboard reads the evaluation outputs above and renders policy performance, pricing trends, and business-KPI scorecards — run the evaluation command at least once first.

### Run the test suite

```bash
python -m pytest tests/ -v
python -m pytest additional_features/tests -q
```

---

## Results

**Latest 1,000-episode-per-policy comparison** (see [`evaluation/policy_evaluation_summary.csv`](evaluation/policy_evaluation_summary.csv) for the raw output and [`evaluation/significance_results.csv`](evaluation/significance_results.csv) for paired significance tests):

| Policy | Mean Revenue (₹) | Sell-Through | Spoilage |
|---|---:|---:|---:|
| Fixed Price | 1,865,694.85 | 97.8% | 2.2% |
| DQN | 1,608,924.81 | 95.3% | 4.7% |
| Q-Learning | 1,391,544.43 | 98.2% | 1.8% |
| Random | 1,357,007.35 | 84.8% | 15.2% |
| Time-Based Discount | 1,231,777.58 | 100.0% | 0.0% |

**Key findings:**

- DQN's evaluated mean revenue roughly doubled after fixing a checkpoint/seeding issue, moving it from last place to second — ahead of Q-Learning, Random, and Time-Based Discount.
- The fixed-price baseline still leads on mean revenue. This is backed by paired significance testing (paired t-test and Wilcoxon signed-rank test, since every policy is evaluated on the same 1,000 simulated seasons): fixed-price beats every other policy at p < 0.0001, with a large effect size (Cohen's d from −1.09 vs. DQN to −2.64 vs. Time-Based Discount).
- Neither learned agent has been exhaustively tuned. A preliminary experiment with a larger network and tuned learning rate raised DQN's evaluation-only mean revenue further (≈ ₹1.73M over 200 greedy episodes vs. ≈ ₹1.62M for the default configuration), but was not carried into the five-policy comparison above due to an architecture mismatch with the current evaluation pipeline (tracked under [Known Issues](#known-issues--open-items)).

This gap between the learned agents and the fixed-price baseline is treated as the project's central open research question and is reported transparently rather than hidden.

---

## Project Management & Workflow

The project was run as a 4-week agile sprint using GitHub for source control and project tracking:

- **Kanban Board (GitHub Projects):** All work was tracked on a Kanban board with **To Do → In Progress → Review → Done** columns. Each task was created as a GitHub Issue, assigned to a team member, and moved across the board as it progressed, giving the team a single source of truth for what was planned, in flight, and completed each week.
- **Issue-driven development:** Features and fixes were scoped as individual GitHub Issues (e.g. `#62`, `#70`, `#74`, `#78`, `#82`, `#90`, `#98`, `#102`, `#106`, `#110`), each closed only after its deliverable was verified and tested.
- **Branching & review:** Work was developed on feature branches and merged into `dev` via Pull Requests, with code review carried out before merge.
- **Task division:** Work was split by MDP/environment design, agent implementation (Q-Learning, DQN), evaluation framework, dashboard/UI, and documentation/reporting, with weekly rotation of reporting responsibility across the team.
- **Sprint cadence:** Each week closed with a review and refactor pass, a written sprint report, and a Kanban board reconciliation to move completed cards to Done before the next week's planning began.

This workflow kept task ownership clear across all four team members and produced a traceable history from problem statement to final, validated release.

---

## Weekly Progress

### Week 1 — Environment & MDP Foundation

**Goal:** Take the project from a business problem statement to a fully working, tested Gymnasium environment — the foundation every agent trains against from Week 2 onward.

| Day | Focus | Outcome |
|---|---|---|
| Day 1 | Business understanding & problem definition | Business background, MDP rationale, state/action/reward design, stakeholders, KPIs, constraints, risks, and timeline documented. |
| Day 2 | MDP design | State, action, reward, transition, discount factor, and terminal condition formalized in code. |
| Day 3 | Gym environment skeleton | `PricingEnvConfig`, `__init__()`, `reset()`, `render()` shipped. |
| Day 4 | Demand model & full integration | `step()` fully implemented and wired to the demand simulator and reward function; full episode loop verified end to end. |
| Day 5 | Review, refactor, docs, sign-off | Code review, folder verification, README, and sprint report. |

**Verification:** 11/11 checks passed, covering construction, reset shape/dtype, invalid-action handling, reproducibility, revenue accounting, and Gymnasium API compliance.

**Delivered:** a functionally complete, tested, stable environment.

---

### Week 2 — Baseline RL Pipeline & Q-Learning

**Goal:** Develop and integrate a baseline reinforcement learning pipeline using the custom environment and a tabular Q-Learning agent.

| Area | Delivered |
|---|---|
| RL training pipeline | Environment initialization, training configuration, entry point, episode execution. |
| Tabular Q-Learning agent | Q-table initialization, epsilon-greedy action selection, Q-value update rule, hyperparameter configuration. |
| Environment integration | `reset()`, `step()`, reward calculation, and state transitions verified against the live agent. |
| Metrics & logging | Episode reward, revenue, inventory utilization, and episode statistics captured. |
| Evaluation | Q-Learning benchmarked against a random policy. |

**Delivered:** a functional baseline reinforcement learning system, ready for DQN development.

---

### Week 3 — Deep Q-Network Integration

**Goal:** Extend the project from tabular Q-Learning to a Deep Q-Network — the scaling path needed once richer state variables would make a Q-table impractically large.

**Issues closed:** #62, #70, #74, #78, #82

- Built `agents/dqn_agent.py` (network, replay buffer, target network), `configs/dqn_config.py`, `training/train_dqn.py`.
- Added a 7-test suite (`tests/test_replay_buffer_integration.py`) validating storage, sampling, capacity eviction, and warm-up-gated training.
- Executed a fully-monitored 2,000-episode DQN run — 54,880 gradient steps, zero errors.
- Refactored duplicated environment-setup logic in `train_agent.py`/`train_dqn.py` into a shared `training/env_utils.py`.

**Delivered:** a working DQN pipeline, integrated and validated alongside Q-Learning.

---

### Week 4 — Evaluation, Dashboard & Release

**Goal:** Turn the two trained agents into an answer to the project's core question — does either learned policy beat a simple pricing rule? — via a fair large-scale evaluation framework, business-facing dashboard, and full release validation.

**Issues closed:** #90, #98, #102, #106, #110

- Built the three heuristic baselines and a shared `select_greedy_action(observation)` interface across all five policies.
- Ran 1,000 episodes × 5 policies (5,000 simulations total), 0 execution failures.
- Built the business dashboard, validated across normal load, filtered views, missing data, and schema-mismatch scenarios.
- Re-exercised every component together, added statistical significance testing, and independently re-confirmed results on a second machine.

**Delivered:** every component (environment, both agents, all three baselines, evaluation framework, dashboard) built, integrated, and independently validated, with the DQN performance gap reported transparently rather than concealed.

---

## Repository Structure

```
dynamic-pricing-rl/
│   .gitignore
│   LICENSE
│   README.md
│   requirements.txt
│
├── agents/                    # Learned pricing agents
│   ├── q_learning_agent.py        # Tabular Q-Learning agent
│   ├── dqn_agent.py                # Deep Q-Network (replay buffer, target network)
│   └── checkpoints/
│       ├── q_learning_policy.pkl       # Trained Q-table + metadata
│       └── dqn_policy.pt                # Trained DQN weights + metadata
│
├── baselines/                 # Heuristic pricing strategies (the comparison floor)
│   ├── baseline_random.py
│   ├── baseline_fixed.py
│   └── baseline_timebased.py
│
├── configs/
│   ├── training_config.py         # Q-Learning training configuration
│   ├── experiment_config.py        # Named hyperparameter-comparison experiments
│   ├── dqn_config.py                # DQN training configuration
│   └── evaluation_config.py          # Policy evaluation configuration
│
├── dashboard/
│   └── web_dashboard/              # Flask business dashboard
│
├── evaluation/
│   ├── evaluate_policies.py        # Runs all 5 policies through 1,000 simulated seasons
│   ├── evaluation_results.csv        # Episode-level results (generated)
│   ├── policy_evaluation_summary.csv/.json  # Aggregate stats + KPI scoring (generated)
│   └── significance_testing.py       # Paired statistical significance tests
│
├── notebooks/                 # Exploratory analysis notebooks
│
├── pricing_env/                # The core Gymnasium environment
│   ├── __init__.py
│   ├── pricing_env.py              # PricingEnvironment — MDP loop (init/reset/step/render)
│   ├── state.py                     # MDP state representation + observation space
│   ├── action_space.py               # Discrete pricing action space
│   ├── reward.py                       # Reward function
│   └── demand_simulator.py               # Stochastic demand model
│
├── reports/
│   ├── Weekly_report/               # Week 1–4 sprint reports, per team member
│   ├── project_planning/              # Problem statement, MDP formulation, architecture docs
│   ├── research_notes/                 # Comparative and evaluation research notes
│   ├── dashboard_design.md
│   └── release_notes.md                 # Final project release notes
│
├── tests/
│   ├── test_replay_buffer_integration.py
│   └── test_web_dashboard_api.py
│
├── training/
│   ├── env_utils.py             # Shared environment construction/verification
│   ├── train_agent.py             # Q-Learning training entry point
│   ├── run_experiment.py           # Experiment comparison workflow
│   └── train_dqn.py                  # DQN training entry point
│
├── additional_features/       # Explainable AI, demand-shock detection, human feedback, digital twin
│
└── utils/                     # Shared helper functions
```

---

## Known Issues & Open Items

Carried forward transparently rather than hidden:

- **DQN performance regression — investigated and fixed.** Root cause traced to a reproducibility bug in `pricing_env/transition.py` (unrelated to the live training path) plus a stale committed checkpoint. Retraining cleanly with the existing, correct seeding roughly doubled DQN's evaluated mean revenue.
- **Statistical significance testing — implemented.** `evaluation/significance_testing.py` runs paired t-tests and Wilcoxon signed-rank tests between every policy and a reference policy, covered by 4 passing unit tests.
- **Fixed-price baseline still beats both learned agents on mean revenue.** Statistically confirmed (p < 0.0001, large effect size against every alternative). This remains the project's central open research question.
- **Hyperparameter tuning — partially done, not fully integrated.** A tuned DQN configuration (128×128 network) evaluated higher on a preliminary check but is not yet compatible with the current evaluation pipeline's fixed 64×64 architecture assumption.
- **`np.random.seed` — not wired in.** A `set_seed()` helper covering `random`, `numpy`, and `torch` already exists in `utils/model.py` but is currently unused in the training entry points.

---

## Documentation Index

| Document | Purpose |
|---|---|
| `reports/project_planning/problem_statement.md` | Original business/technical problem definition |
| `reports/project_planning/rl_problem_formulation.md` | MDP formalization |
| `reports/project_planning/dqn_architecture.md` | DQN network/training architecture |
| `reports/dashboard_design.md` | Dashboard architecture |
| `reports/release_notes.md` | Full release summary, including known issues |
| `reports/Weekly_report/` | Detailed week-by-week sprint reports, per team member |
| `docs/Documentation_Index.md` | Additional environment, model, and dashboard guides |

---

## Roadmap

- ✅ **Week 1:** Environment design, MDP formulation, demand/reward model.
- ✅ **Week 2:** Tabular Q-Learning, experiment comparison workflow.
- ✅ **Week 3:** Deep Q-Network, experience replay, target network.
- ✅ **Week 4:** Baseline policies, 1,000-episode evaluation framework, business dashboard, full validation.

---

## License

This project is licensed under the MIT License — see [`LICENSE`](LICENSE) for details.

---

*Built over a 4-week sprint by Abinaya S, Himanshu Rawat, Nagammai Subramaniyan, and N. Yogeshwaran. See `reports/Weekly_report/` for the complete, day-by-day source reports this README summarizes.*
