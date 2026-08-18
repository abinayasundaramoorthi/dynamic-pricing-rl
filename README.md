# additional_features + dashboard/web_dashboard — Flask Edition

This is an update to the previous `additional_features` delivery. Two things changed, both requested directly:

1. **No more Streamlit for the new dashboard.** The Advanced Revenue Management Dashboard (Feature 2) is now a **Flask** app with Bootstrap 5 + Chart.js, built from the design you liked (`revenue-ai-production.zip`).
2. **No duplicate top-level dashboard folder.** The Flask app lives at `dashboard/web_dashboard/`, nested inside your existing `dashboard/` folder — not a second top-level `web_dashboard/` folder next to it.

Your original `dashboard/dashboard_app.py` (Streamlit) and every other core project file are **byte-for-byte untouched**. This is purely additive.
# Dynamic Pricing with Reinforcement Learning

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
a business-facing Streamlit dashboard.

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

Your original Streamlit dashboard still runs exactly as before:
Runs 1,000 simulated booking seasons for each of DQN, Q-Learning, Random,
Fixed Price, and Time-Based Discount, scores every policy against the
project's business KPIs (revenue uplift, sell-through, spoilage), and
writes `evaluation/evaluation_results.csv` (episode-level) plus
`evaluation/policy_evaluation_summary.csv` / `.json` (aggregate).
Add `--smoke-test` for a fast 20-episode pipeline check instead.

### Business dashboard

```bash
python -m streamlit run dashboard/dashboard_app.py
```

## What's real vs. what's a template
Reads the evaluation outputs above and renders policy performance,
pricing trends, and business-KPI scorecards. Run the evaluation command
at least once first — the dashboard displays what's on disk, it doesn't
simulate anything itself.

### Programmatic usage

```python
from pricing_env import PricingEnvironment, PricingEnvConfig

env = PricingEnvironment(PricingEnvConfig(render_mode="human"))
obs, info = env.reset(seed=42)

terminated = truncated = False
total_reward = 0.0
while not (terminated or truncated):
    action = env.action_space.sample()  # replace with a trained agent's policy
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward

print(f"Episode revenue: ${info['episode_revenue']:.2f}")
```

---

## Repository structure

```
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

## Environment summary

- **One step = one day.** Each day the agent chooses a discrete price
  action; a stochastic number of customers arrive and decide to purchase
  based on that price; inventory, revenue, and reward update accordingly.
- **State:** `[remaining_inventory, days_remaining]`.
- **Action:** discrete percentage price adjustment relative to the current
  price (default: 7 tiers from −20% to +20%).
- **Reward:** revenue earned this step, minus an over-discounting penalty
  and a terminal unsold-inventory penalty, plus a small inventory-pacing
  shaping bonus.
- **Episode ends** when inventory sells out or the deadline is reached,
  whichever comes first.

## Policies compared

Five pricing strategies are evaluated head-to-head, on identical simulated
demand, via `evaluation/evaluate_policies.py`:

| Policy | File | Approach |
|---|---|---|
| DQN | `agents/dqn_agent.py` | Feedforward network, experience replay, target network |
| Q-Learning | `agents/q_learning.py` | Tabular Q-table over discretized `(inventory, days)` states |
| Random | `baselines/baseline_random.py` | Uniform-random action every step (the floor) |
| Fixed Price | `baselines/baseline_fixed.py` | Never discounts off `base_price` (the legacy-system stand-in) |
| Time-Based Discount | `baselines/baseline_timebased.py` | Rule-based discount depth driven by deadline urgency + sell-through pacing |

**Latest 1,000-episode-per-policy comparison** (see
`evaluation/simulation_summary.md` for full detail):

| Policy | Mean Revenue | Sell-Through | Spoilage |
|---|---:|---:|---:|
| Fixed Price | $19,558.60 | 97.8% | 2.2% |
| Q-Learning | $15,356.83 | 87.5% | 12.5% |
| Random | $14,225.89 | 84.8% | 15.2% |
| Time-Based Discount | $12,913.07 | 100.0% | 0.0% |
| DQN | $7,136.26 | 100.0% | 0.0% |

**Read this before trusting it at face value:** in this comparison, the
fixed-price baseline currently beats every learned policy on revenue, and
DQN in particular is selling out but at very deep discounts. This is a real
result from the checkpoints currently committed to the repo, not a display
bug — see `reports/Release_Notes.md` for why, and for what it means before
this is presented as "the RL agent beats the baseline."

Neither agent's hyperparameters have been empirically tuned yet — both use
literature-standard defaults, documented in their respective config files
(`configs/training_config.py`, `configs/dqn_config.py`).

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
