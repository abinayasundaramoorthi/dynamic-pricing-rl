# dynamic-pricing-rl

Reinforcement Learning for Dynamic Pricing in Travel & Hospitality.

A Reinforcement Learning agent that learns optimal pricing policies for
finite, perishable inventory (airline seats, hotel rooms) by interacting
with a simulated booking market — maximizing total season revenue instead
of any single transaction, and adapting to remaining inventory, demand, and
the shrinking time before departure/check-in.

**Status:** Week 4 complete — project validated end to end. Environment,
tabular Q-Learning, Deep Q-Network, three heuristic baselines, a
1,000-episode policy evaluation framework, and a business-facing dashboard
are all built, integrated, and verified together.
See [`reports/Release_Notes.md`](reports/Release_Notes.md) for the full
release summary (including a known issue with the current DQN checkpoint's
performance — read that before treating the numbers below as final), and
[`reports/project_planning/problem_statement.md`](reports/project_planning/problem_statement.md)
for the original business/technical design.

---

## Setup

```bash
git clone <repo-url>
cd dynamic-pricing-rl
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

## Verify the project is working

```bash
python -m pytest tests/ -v
```

Runs 7 checks covering replay buffer storage, sampling, capacity eviction,
defensive array copying, and warm-up-gated training.

For a fuller check that the whole pipeline (environment → agents →
evaluation → dashboard) actually works end to end on your machine, see
[`reports/Release_Notes.md`](reports/Release_Notes.md), Section "How to
validate this release."

## Quick usage

### Tabular Q-Learning

```bash
python -m training.train_agent --agent q_learning --episodes 5000
```

Trains a Q-table, saves it to `agents/checkpoints/q_learning_policy.pkl`,
reloads it from disk, and evaluates the reloaded policy on held-out seeds.

### Deep Q-Network

```bash
python -m training.train_dqn --episodes 2000
```

Trains a small feedforward network with experience replay and a target
network, saves it to `agents/checkpoints/dqn_policy.pt`, reloads it from
disk, and evaluates the reloaded policy the same way.

**Note:** DQN training is not currently bit-for-bit reproducible run-to-run
(PyTorch's global RNG for network initialization is unseeded) — different
runs of this command can produce checkpoints with meaningfully different
evaluated performance. See `reports/Release_Notes.md` for a concrete
example of how large that spread can be.

### Experiment comparison suite

```bash
python -m training.run_experiment
```

Runs a suite of named hyperparameter configurations and writes each one's
results independently to `evaluation/results/<name>/`, plus a
cross-experiment `comparison_summary.csv`.

### Large-scale policy evaluation (all 5 strategies)

```bash
python -m evaluation.evaluate_policies
```

Runs 1,000 simulated booking seasons for each of DQN, Q-Learning, Random,
Fixed Price, and Time-Based Discount, scores every policy against this
project's business KPIs (revenue uplift, sell-through, spoilage), and
writes `evaluation/evaluation_results.csv` (episode-level) and
`evaluation/policy_evaluation_summary.csv`/`.json` (aggregate). See
`reports/policy_evaluation_design.md` for the architecture and
`evaluation/simulation_summary.md` for the latest run's results.

Add `--smoke-test` for a fast 20-episode pipeline check instead of the full
1,000-episode run.

### Business dashboard

```bash
streamlit run dashboard/dashboard_app.py
```

Reads the evaluation outputs above and renders policy performance, pricing
trends, and business-KPI scorecards. Run the evaluation command above at
least once first — the dashboard displays whatever it finds on disk, it
doesn't simulate anything itself. See `reports/dashboard_design.md` for
the architecture.

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
├── agents/
│       q_learning.py               # Tabular Q-Learning agent
│       dqn_agent.py                 # Deep Q-Network agent (replay buffer, target network)
│       checkpoints/
│           q_learning_policy.pkl        # Trained Q-table + metadata JSON
│           dqn_policy.pt                 # Trained DQN weights + metadata JSON
├── baselines/                       # Naive/heuristic pricing strategies, evaluated
│       __init__.py                    # alongside the learned agents as the comparison floor
│       baseline_random.py
│       baseline_fixed.py
│       baseline_timebased.py
├── configs/
│       training_config.py          # Q-Learning training configuration
│       experiment_config.py         # Named hyperparameter-comparison experiments
│       dqn_config.py                 # DQN training configuration
│       evaluation_config.py           # Policy evaluation configuration (1,000-episode runs, business KPIs)
├── dashboard/
│       dashboard_app.py             # Streamlit business dashboard (policy performance, pricing trends, KPIs)
├── evaluation/
│       evaluate_policies.py         # Evaluation workflow — runs all 5 policies through 1,000 simulated seasons
│       evaluation_results.csv         # Episode-level results, latest run (generated)
│       policy_evaluation_summary.csv   # Aggregate per-policy stats + KPI scoring (generated)
│       simulation_summary.md            # Human-readable summary of the latest evaluation run (generated)
├── notebooks/                       # Reserved for exploratory analysis notebooks
├── pricing_env/                     # The core Gymnasium environment
│       __init__.py
│       pricing_env.py                 # PricingEnvironment — MDP loop (init/reset/step/render)
│       state.py                        # MDP state representation + observation space
│       action_space.py                  # Discrete pricing action space
│       reward.py                         # Reward function
│       demand_simulator.py                # Stochastic demand model
├── reports/
│       Weekly_report/
│           Week1_report_Abinaya.md
│           Week2_report_Abinaya.md
│           Week3_report_Abinaya.md
│       project_planning/
│           problem_statement.md
│           rl_problem_formulation.md
│           dqn_architecture.md
│           dqn_trianing_workflow.md
│       training_execution_summary.md
│       policy_evaluation_design.md    # Architecture of the evaluation framework
│       dashboard_design.md              # Architecture of the business dashboard
│       Release_Notes.md                  # Final project release notes
├── tests/
│       test_replay_buffer_integration.py
├── training/
│       env_utils.py                   # Shared environment construction/verification
│       train_agent.py                  # Q-Learning training entry point
│       run_experiment.py                # Experiment comparison workflow
│       train_dqn.py                       # DQN training entry point
│       dqn_training_results.md             # Full-scale DQN run results record
└── utils/                                    # Reserved for shared helper functions
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

- **Week 1:** Environment design, MDP formulation, demand/reward model. Complete.
- **Week 2:** Tabular Q-Learning, experiment comparison workflow. Complete.
- **Week 3:** Deep Q-Network, experience replay, target network. Complete.
- **Week 4:** Baseline policies, 1,000-episode policy evaluation framework,
  business-facing dashboard, full project validation. Complete.
- **Next:** investigate and close the DQN performance gap flagged above
  (see `reports/Release_Notes.md`, Known Issues) before any production
  consideration.