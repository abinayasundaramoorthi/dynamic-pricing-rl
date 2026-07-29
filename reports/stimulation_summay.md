# Large-Scale Policy Evaluation — Simulation Summary

**Issue:** #98 — Execute Large-Scale Policy Evaluation Pipeline
**Pipeline used:** `evaluation/evaluate_policies.py` (built in #90)
**Run date:** 2026-07-29
**Status:** ✅ Completed successfully — 0 execution failures

---

## 1. Objective

Execute the policy evaluation pipeline from issue #90 for real: run 1,000 simulated booking seasons for every implemented pricing strategy, monitor the run to completion, verify all simulations actually finished, and consolidate the output into a single results file and this summary.

---

## 2. Execution issues encountered and resolved

**Issue 1 — missing DQN checkpoint.** Before the first run, `agents/checkpoints/` contained a Q-Learning checkpoint but no DQN checkpoint — `training/train_dqn.py` had never been executed. Running the evaluation pipeline as-is would have silently skipped the DQN agent (by design, per #90 — missing checkpoints degrade gracefully rather than crashing the run), which would have violated this issue's "All pricing strategies evaluated" acceptance criterion.

**Resolution:** ran `python -m training.train_dqn` (default config: 2,000 training episodes, ε-greedy 1.0→0.05, replay buffer 50k, target-network sync every 25 episodes) to convergence. Training completed in ~4m6s and produced `agents/checkpoints/dqn_policy.pt`, confirmed loadable via the save/load round-trip check `train_dqn.py` runs automatically.

**Issue 2 — output filename didn't match the deliverable spec.** The pipeline originally wrote episode-level results to `evaluation/results/policy_evaluation_episodes.csv`, requiring a manual copy/rename to produce `evaluation/evaluation_results.csv` as named in this issue's deliverables.

**Resolution:** updated `evaluation/evaluate_policies.py`'s `save_results()` and `configs/evaluation_config.py`'s `results_dir` default so the pipeline writes `evaluation/evaluation_results.csv` directly, under that exact name, with no manual step. Re-ran the full pipeline afterward to confirm the file is produced automatically (see Section 4).

No other execution issues occurred.

---

## 3. Execution log

Command: `python -m evaluation.evaluate_policies`
Config: `get_default_evaluation_config()` — 1,000 episodes, all 5 policies, seeds `5,000,001`–`5,000,1000` shared identically across every policy.
Wall-clock time: **~11 seconds** for all 5,000 episodes (all policies act greedily — no training happens during evaluation).

| Order | Policy | Episodes run | Progress checkpoints logged |
|---|---|---|---|
| 1 | `dqn` | 1,000/1,000 | 100, 200, ..., 1000 ✅ |
| 2 | `q_learning` | 1,000/1,000 | 100, 200, ..., 1000 ✅ |
| 3 | `random` | 1,000/1,000 | 100, 200, ..., 1000 ✅ |
| 4 | `fixed_price` | 1,000/1,000 | 100, 200, ..., 1000 ✅ |
| 5 | `time_based_discount` | 1,000/1,000 | 100, 200, ..., 1000 ✅ |

No exceptions, no dropped episodes, no policy build failures. `run_evaluation()` completed and returned a fully-populated summary for all five policies.

---

## 4. Verification of simulation completion

- `evaluation/evaluation_results.csv` (this issue's deliverable) contains **5,000 data rows** (+1 header row) — exactly `5 policies × 1,000 episodes`, confirmed by direct row count, and produced automatically by the pipeline (no manual file operations).
- Every policy's episode count in the summary is exactly `1000` (see `num_episodes` column, `evaluation/policy_evaluation_summary.csv`).
- Every one of the 5,000 episodes reached `terminated` or `truncated` in the environment — `evaluate_policy()` only appends a result after `run_episode()` returns, and `run_episode()` only returns after the episode's `while not (terminated or truncated)` loop exits — i.e. every simulated booking season actually ran to completion, none were cut short.

**1,000 simulations completed successfully, for all 5 pricing strategies. No execution failures.**

---

## 5. Consolidated results

| Policy | Mean Revenue | Std Revenue | Revenue Uplift vs. `fixed_price` | Sell-Through % | Spoilage % |
|---|---:|---:|---:|---:|---:|
| **fixed_price** | $19,558.60 | $851.78 | 0.0% (reference) | 97.8% | 2.2% |
| **q_learning** | $15,356.83 | $3,257.33 | -21.5% | 87.5% | 12.5% |
| **random** | $14,225.89 | $3,434.55 | -27.3% | 84.8% | 15.2% |
| **time_based_discount** | $12,913.07 | $2,476.34 | -34.0% | 100.0% | 0.0% |
| **dqn** | $7,136.26 | $598.85 | -63.5% | 100.0% | 0.0% |

Full episode-level detail (one row per policy × seed, 5,000 rows) is in `evaluation/evaluation_results.csv`. Aggregated statistics and business-KPI pass/fail flags are also written automatically to `evaluation/policy_evaluation_summary.csv` / `.json` by the pipeline.

### Business KPI scoring (targets from `configs/evaluation_config.py`'s `BusinessKPIConfig`)

| Policy | Revenue uplift > 10%? | Sell-through > 95%? | Spoilage < 5%? |
|---|:---:|:---:|:---:|
| fixed_price | — (reference) | ✅ | ✅ |
| q_learning | ❌ | ❌ | ❌ |
| random | ❌ | ❌ | ❌ |
| time_based_discount | ❌ | ✅ | ✅ |
| dqn | ❌ | ✅ | ✅ |

---

## 6. Observations (not execution issues — the pipeline ran correctly)

The run itself completed cleanly, but the *results* are worth flagging honestly rather than smoothing over:

- **`fixed_price` currently outperforms every other policy on revenue**, including both RL agents. This is not a pipeline defect — it's a real result from the currently-trained checkpoints.
- **`dqn` sells through 100% of inventory but at very low prices** (mean price ~$81 vs. a $200 base price, ~60% mean discount depth) — consistent with a policy that has converged to "discount aggressively to guarantee a sale" rather than one that has learned to balance price against demand elasticity. `q_learning`'s revenue is also below the fixed-price and random baselines in this run.
- This suggests the DQN and Q-Learning checkpoints evaluated here are undertrained or under-tuned (reward scale, learning rate, or episode count) relative to what's needed to beat the heuristic baselines — a training/tuning question for the agent-development side of the project, not something in scope for this evaluation-execution issue. The evaluation pipeline's job was to measure this accurately and consolidate it, which it did.

---

## 7. Deliverables produced

```
evaluation/
├── evaluation_results.csv     # 5,000 rows: full episode-level results, all 5 policies
│                               # written automatically by evaluate_policies.py — no manual step
└── simulation_summary.md      # this document
```

(The pipeline also writes `evaluation/policy_evaluation_summary.csv` and `.json` directly alongside these as part of its normal operation — the machine-readable aggregate backing Section 5's table above.)