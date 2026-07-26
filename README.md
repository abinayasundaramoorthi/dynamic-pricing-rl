# dynamic-pricing-rl

Reinforcement Learning for Dynamic Pricing in Travel & Hospitality.

A Reinforcement Learning agent that learns optimal pricing policies for
finite, perishable inventory (airline seats, hotel rooms) by interacting
with a simulated booking market — maximizing total season revenue instead
of any single transaction, and adapting to remaining inventory, demand, and
the shrinking time before departure/check-in.

**Status:** Week 3 complete. Environment, tabular Q-Learning, and Deep
Q-Network all built, integrated, and verified.
See [`reports/Week3_Sprint_Report.md`](reports/Week3_Sprint_Report.md) for
the current sprint summary, and
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

## Verify the environment and replay buffer

```bash
python -m pytest tests/test_replay_buffer_integration.py -v
```

Runs 7 checks covering replay buffer storage, sampling, capacity eviction,
defensive array copying, and warm-up-gated training.

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

### Experiment comparison suite

```bash
python -m training.run_experiment
```

Runs a suite of named hyperparameter configurations and writes each one's
results independently to `evaluation/results/<name>/`, plus a
cross-experiment `comparison_summary.csv`.

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
│       q_learning.py        # Tabular Q-Learning agent
│       dqn_agent.py          # Deep Q-Network agent (replay buffer, target network)
├── baselines/                # Reserved for naive heuristic agents (not yet implemented —
│                              # see Week3_Sprint_Report.md, Open Items)
├── configs/
│       training_config.py    # Q-Learning training configuration
│       experiment_config.py   # Named hyperparameter-comparison experiments
│       dqn_config.py            # DQN training configuration
├── dashboard/                  # Business-facing results dashboard (planned, Week 4)
├── evaluation/                  # Experiment results output (generated at runtime)
├── notebooks/                    # Exploratory analysis notebooks
├── pricing_env/                   # The core Gymnasium environment
│       __init__.py
│       pricing_env.py              # PricingEnvironment — MDP loop (init/reset/step/render)
│       state.py                     # MDP state representation + observation space
│       action_space.py               # Discrete pricing action space
│       reward.py                      # Reward function
│       demand_simulator.py             # Stochastic demand model
├── reports/
│       Weekly_report/
│           Week1_report_Abinaya.md
│           Week2_report_Abinaya.md
│       project_planning/
│           problem_statement.md
│           rl_problem_formulation.md
│           dqn_architecture.md
│           dqn_trianing_workflow.md
│       training_execution_summary.md
│       Week3_Sprint_Report.md
├── tests/
│       test_replay_buffer_integration.py
├── training/
│       env_utils.py                   # Shared environment construction/verification
│       train_agent.py                  # Q-Learning training entry point
│       run_experiment.py                # Experiment comparison workflow
│       train_dqn.py                       # DQN training entry point
│       dqn_training_results.md             # Latest full-scale DQN run results
└── utils/                                    # Shared helper functions (reserved)
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

## Agents

| Agent | File | Approach | Latest verified result |
|---|---|---|---|
| Q-Learning | `agents/q_learning.py` | Tabular Q-table over discretized `(inventory, days)` states | Beats random policy by ~9% revenue on held-out seeds |
| DQN | `agents/dqn_agent.py` | Feedforward network, experience replay, target network | ~$19,500 mean evaluation revenue over 200 held-out episodes (two independent seed sets agree within 0.3%) |

Neither agent's hyperparameters have been empirically tuned yet — both use
literature-standard defaults, documented honestly in their respective
config files (`configs/training_config.py`, `configs/dqn_config.py`).
DQN training is also not currently bit-for-bit reproducible run-to-run
(PyTorch's global RNG for network initialization is unseeded) — see
`reports/Week3_Sprint_Report.md`, Open Items.

## Roadmap

- **Week 1:** Environment design, MDP formulation, demand/reward model. Complete.
- **Week 2:** Tabular Q-Learning, experiment comparison workflow. Complete.
- **Week 3:** Deep Q-Network, experience replay, target network. Complete.
- **Week 4:** Policy evaluation across simulated seasons, Q-Learning vs.
  DQN comparison, price-trajectory visualization, business-facing
  dashboard.