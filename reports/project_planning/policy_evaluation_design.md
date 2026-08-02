# Policy Evaluation Framework — Design & Architecture

**Issue:** #90 — Design and Integrate Policy Evaluation Framework
**Status:** Implemented and verified (smoke-tested end to end; full 1,000-episode run supported)
**Prerequisite reading:** `reports/project_planning/problem_statement.md` (Section 18–19, KPIs and success criteria), `agents/dqn_agent.py` / `agents/q_learning.py` (the two learned policies this framework evaluates), `pricing_env/reward.py` (what each policy is actually optimizing against)

---

## 1. Purpose

Every other Week 1–3 deliverable produced *one* policy at a time: the environment, the two baseline heuristics implied by the design doc, tabular Q-Learning, and DQN. None of that work, on its own, answers the question the project exists to answer: **is the learned pricing policy actually better than what a revenue manager would do without it?**

This framework is the single place that question gets answered, concretely and reproducibly:

- All five pricing strategies (`dqn`, `q_learning`, `random`, `fixed_price`, `time_based_discount`) are run through **the exact same simulation protocol** — 1,000 simulated booking seasons, on identical demand realizations.
- Results are reduced to the **business KPIs** the project defined for itself (Section 19 of the problem statement: revenue uplift, sell-through, spoilage), not just RL-internal metrics like mean reward.
- Everything is persisted (episode-level and summary) so results are auditable and reproducible from a checked-in config, not regenerated ad hoc in a notebook each time someone wants a number.

---

## 2. Scope of this deliverable

Issue #90 asked for three things, and this is where each lives:

| Deliverable | File | What it is |
|---|---|---|
| Evaluation workflow | `evaluation/evaluate_policies.py` | Orchestrates building policies, running episodes, aggregating results, scoring against KPIs, persisting output |
| Evaluation configuration | `configs/evaluation_config.py` | `EvaluationConfig` — one frozen dataclass fully specifying a run: environment, episode count, seeds, which policies, checkpoint paths, business KPI thresholds |
| Architecture documentation | `reports/policy_evaluation_design.md` | This document |

A fourth, unlisted-but-necessary piece was added because it didn't exist yet: `baselines/` (`baseline_random.py`, `baseline_fixed.py`, `baseline_timebased.py`). The problem statement's Week 2 milestone table (Section 21, M1/M2) already named this directory and these two files as planned deliverables; they were never implemented as part of that earlier work, so "Integrate ... Random Policy, Fixed Price Policy, Discount Policy" (issue #90's task list) required building them first. See Section 5 below.

---

## 3. Why evaluation needs its own config, not `TrainingConfig`/`DQNConfig`

`configs/training_config.py` and `configs/dqn_config.py` each describe **how to train one agent**. Neither has any notion of "compare five differently-sourced policies against a shared metric set" — that's a different unit of work with a genuinely different parameter set:

- Which policies to include, and where their checkpoints live (`policies_to_evaluate`, `dqn_checkpoint_path`, `q_learning_checkpoint_path`)
- How many simulated seasons and what seed sequence (`num_episodes`, `base_seed`, `episode_seeds()`)
- Business KPI thresholds to score against (`business_kpis`)
- Which policy is the "1.0x" comparison baseline for revenue uplift (`reference_policy_for_uplift`)

None of `TrainingConfig`'s fields (learning rate, exploration schedule, ...) are meaningful here — a heuristic baseline has no learning rate — and none of `EvaluationConfig`'s fields are meaningful there. This is the same reasoning `configs/dqn_config.py`'s own module docstring gives for why `DQNConfig` isn't a subclass of `TrainingConfig`; `EvaluationConfig` follows the established project convention rather than inventing a new one: frozen dataclass, validated in `__post_init__`, one canonical `get_default_evaluation_config()` factory.

What *is* deliberately shared: the same environment configuration (`initial_inventory=100`, `selling_horizon_days=30`, `base_price=200.0`) already used by `get_final_training_config()` and `get_default_dqn_config()`. This is what makes the comparison an apples-to-apples evaluation of *algorithms* on a fixed problem instance, not a comparison confounded by different environment settings between training and evaluation.

`EvaluationConfig.episode_seeds()` is the mechanism that guarantees a fair comparison at the implementation level: it returns one ordered list of `num_episodes` seeds, and every policy's evaluation loop is driven off that exact same list. This makes it structurally impossible for two policies in a single run to silently diverge onto different demand realizations — the one requirement that makes a head-to-head revenue comparison scientifically valid (see `pricing_env/pricing_env.py`'s `reset()` docstring on why seeding the whole episode this way is what makes the Week 4 protocol valid in the first place).

---

## 4. Why a `Policy` protocol instead of five special cases

`evaluate_policies.py` needs to run `DQNAgent`, `QLearningAgent`, and three baseline classes through *one* evaluation loop. That was only possible because both learned agents already exposed a `select_greedy_action(observation) -> int` method with identical semantics (no exploration, act purely on the current state) — a convention `agents/dqn_agent.py` was deliberately built to match `agents/q_learning.py`'s existing interface (see that module's own docstring). The three baseline classes in `baselines/` were built to satisfy the exact same method signature, plus a no-op-by-default `reset()` hook for any policy that carries per-episode state (none of the current five need it, but the hook keeps the interface future-proof for e.g. a recurrent policy without changing `run_episode()`).

`evaluate_policies.py` expresses this as a `typing.Protocol` (`Policy`) purely for static readability — it is not a base class any policy inherits from, and none of the five classes needed to change to satisfy it. `run_episode()` and `evaluate_policy()` are therefore written once, against `Policy`, and never branch on which concrete policy type they were given.

---

## 5. The three baseline policies (`baselines/`)

Each baseline is stateless or near-stateless, requires no checkpoint, and exists purely to establish "how much of the RL agents' performance is actually attributable to learning" versus a naive or rule-based alternative:

- **`RandomPolicy`** — samples uniformly from the action space every step, ignoring the observation entirely. The floor: anything that can't beat this has learned nothing.
- **`FixedPricePolicy`** — always selects the zero-adjustment ("hold price") action, keeping the effective price pinned at `base_price` for the whole season. This is the "legacy static pricing system" stand-in the problem statement's Executive Summary explicitly names as the thing RL is meant to outperform, and is the default `reference_policy_for_uplift` for that reason.
- **`TimeBasedDiscountPolicy`** — the strongest non-learned baseline. It computes a target discount depth from two interpretable signals (time-to-deadline urgency, and how far sell-through is lagging a naive linear pacing curve — the same pacing-gap quantity `reward.py`'s `balance_bonus` term penalizes), then snaps to whichever configured price-adjustment action is closest to that target. This mirrors how a revenue manager would reason about the problem manually, without any ML, and is the bar a "smart heuristic" sets — a bar that's meaningfully harder for the RL agents to clear than beating `RandomPolicy` or `FixedPricePolicy`.

None of the three needed a checkpoint or a `checkpoint_dir` entry in `EvaluationConfig` — they're fully specified by their constructor arguments, which `build_policies()` derives directly from `EvaluationConfig.env_config`.

---

## 6. Evaluation workflow (`evaluate_policies.py`)

```
EvaluationConfig
      │
      ▼
build_environment()  ──►  verify_environment_compatibility()   (reused from training/env_utils.py)
      │
      ▼
build_policies()  ──►  {dqn?, q_learning?, random, fixed_price, time_based_discount}
      │                  (missing checkpoints are logged + skipped, not fatal)
      ▼
episode_seeds()  ──►  [seed_1, ..., seed_1000]           (one shared list, all policies)
      │
      ▼
evaluate_policy() per policy  ──►  run_episode() × 1000   (greedy action selection, no exploration)
      │
      ▼
summarize_policy() per policy  ──►  mean/std revenue & reward, sell-through %, spoilage %, ...
      │
      ▼
score_against_business_kpis()  ──►  revenue uplift %, pass/fail vs. Section 19 targets
      │
      ▼
save_results()  ──►  evaluation/results/{policy_evaluation_episodes.csv, _summary.csv, _summary.json}
```

Design choices worth calling out:

- **Greedy action selection only.** `run_episode()` always calls `select_greedy_action()`, never `select_action()` (which would apply epsilon-greedy exploration on the learned agents). Evaluation measures what a deployed policy would actually do, not exploratory behavior — the same reasoning `train_agent.py`'s and `train_dqn.py`'s own post-training `evaluate_agent()` functions already use.
- **Missing checkpoints degrade gracefully, not fatally.** `build_policies()` logs a warning and excludes a policy whose checkpoint file doesn't exist (e.g. running this before `train_dqn.py` has ever been executed), rather than raising. This is what makes "Evaluation pipeline initializes successfully" hold true on a fresh checkout, while the warning still makes the gap impossible to miss in the logs. `run_evaluation()` only raises if *every* requested policy fails to build — there being nothing at all to evaluate is the one condition that should stop the run.
- **Business KPIs are a scoring overlay, not a redefinition of what's measured.** `score_against_business_kpis()` is a separate function from `summarize_policy()` specifically so the raw summary table (mean revenue, sell-through, etc.) stays interpretable on its own; the KPI pass/fail columns and revenue-uplift percentage are derived on top of it, using the thresholds from `EvaluationConfig.business_kpis` (which default to the project's own Section 19 numbers: >10% revenue uplift, >95% sell-through, <5% spoilage).
- **Episode-level results are persisted, not just the summary.** `policy_evaluation_episodes.csv` has one row per (policy, seed) — every simulated season's outcome — so any statistical test beyond mean/std (e.g. a paired comparison across identical seeds) can be run later without re-executing the simulation.

---

## 7. Business KPIs evaluated

Pulled directly from `reports/project_planning/problem_statement.md`, Section 19 ("Success Criteria"), and implemented as `BusinessKPIConfig`:

| KPI | Default target | Field |
|---|---|---|
| Revenue improvement over the fixed-price baseline | > 10% | `target_revenue_uplift_pct` |
| Sell-through rate | > 95% | `target_sell_through_pct` |
| Inventory spoilage | < 5% | `max_spoilage_pct` |

Every summary row also reports mean/std reward, mean/median revenue, mean price, and mean discount depth — RL-internal and price-trajectory diagnostics useful for debugging a policy's *behavior*, even for policies that clear (or miss) the three business KPIs above.

---

## 8. Configuration (`configs/evaluation_config.py`)

`EvaluationConfig` follows the same frozen-dataclass, `__post_init__`-validated pattern as `TrainingConfig` and `DQNConfig`. Two factory functions are provided:

- `get_default_evaluation_config()` — the canonical 1,000-episode, all-five-policies configuration used for reported results.
- `get_smoke_test_evaluation_config()` — a 20-episode variant of the same configuration, used to verify the pipeline initializes and runs end to end quickly (`python -m evaluation.evaluate_policies --smoke-test`), without waiting on a full run. Not a source of reportable numbers.

`EvaluationConfig.__post_init__` validates that `policies_to_evaluate` only names known policies (`ALL_POLICY_NAMES`), contains no duplicates, and that `reference_policy_for_uplift` is itself one of the policies being evaluated — the same "fail loud at config-construction time, not deep inside a 1,000-episode run" philosophy used throughout this codebase (e.g. `PricingEnvConfig`, `RewardConfig`).

---

## 9. Usage

```bash
# Full 1,000-episode evaluation of all five policies
python -m evaluation.evaluate_policies

# Fast pipeline sanity check (20 episodes)
python -m evaluation.evaluate_policies --smoke-test

# Override episode count or the policy subset
python -m evaluation.evaluate_policies --episodes 200
python -m evaluation.evaluate_policies --policies random fixed_price time_based_discount
```

Results are written to `evaluation/results/`:

- `policy_evaluation_episodes.csv` — every (policy, seed) episode's outcome.
- `policy_evaluation_summary.csv` / `.json` — aggregated per-policy statistics and business-KPI scoring.

---

## 10. Acceptance criteria — how this deliverable satisfies them

| Acceptance criterion | How it's satisfied |
|---|---|
| Evaluation pipeline initializes successfully | `build_environment()` + `verify_environment_compatibility()` run first and fail loudly on a malformed environment; policy construction degrades gracefully on missing checkpoints rather than crashing the whole run. Verified via `--smoke-test`. |
| All policies are integrated | `dqn`, `q_learning`, `random`, `fixed_price`, `time_based_discount` are all constructible and run through the identical `run_episode()` loop via the shared `Policy` interface. |
| Simulation configuration supports 1,000 episodes | `EvaluationConfig.num_episodes` defaults to 1000 (`get_default_evaluation_config()`); `episode_seeds()` generates the full, shared seed list regardless of count. |
| Evaluation workflow documented | This document. |

---

## 11. Known limitations / follow-up work

- **DQN checkpoint dependency.** The evaluation framework can only score `dqn` if `training/train_dqn.py` has already been run and written `agents/checkpoints/dqn_policy.pt`. As of this deliverable, no such checkpoint has yet been committed — running the evaluation as-is will log a warning and evaluate the remaining four policies. This is expected, not a bug: training DQN to convergence is separate, already-scoped work.
- **No statistical significance testing yet.** The summary table reports mean/std per policy but does not yet run a paired significance test (e.g. paired t-test across identical seeds) to confirm an observed revenue difference is not noise. The episode-level CSV is structured (identical seed column across policies) specifically so this can be added later without re-running the simulation.
- **Single environment configuration per run.** This framework evaluates all policies against one `PricingEnvConfig` per invocation. Comparing robustness across multiple environment configurations (e.g. varying `initial_inventory` or demand volatility) would mean running this framework multiple times with different configs — a natural extension via `configs/experiment_config.py`'s existing sweep pattern, not yet wired into `EvaluationConfig`.