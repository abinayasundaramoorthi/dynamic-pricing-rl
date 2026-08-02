# Week 4 Report — Policy Evaluation, Dashboard & Final Release

**Sprint:** Week 4, Days 1–5
**Status:** Complete (engineering scope) — two manual GitHub actions remain, see Section 7
**Issues closed this sprint:** #90, #98, #102, #106, #110

---

## 1. Objective Recap

Week 4's goal was to turn the two trained agents from Weeks 2–3 into an
actual answer to the project's core question: does either learned policy
beat a simple pricing rule? That required building an evaluation framework
capable of comparing all policies fairly at scale, adding the naive
baseline strategies the original roadmap called for (still open at the end
of Week 3 — see that sprint's Open Items), giving the results a
business-facing presentation, and then validating and releasing the whole
project as one integrated system.

---

## 2. Day-by-Day Summary

| Day | Issue | Planned Task | Actual Outcome |
|---|---|---|---|
| Day 1 | #90 | Design and integrate a policy evaluation framework | `evaluation/evaluate_policies.py`, `configs/evaluation_config.py`, and `reports/policy_evaluation_design.md` built. Also closed Week 3's open item: `baselines/` was still empty, so `baselines/baseline_random.py`, `baseline_fixed.py`, and `baseline_timebased.py` were built here as a prerequisite — the evaluation framework needed something to evaluate. All five policies (2 learned + 3 baseline) share one `select_greedy_action(observation)` interface, so the evaluation loop was written once, not five times. |
| Day 2 | #98 | Execute the full 1,000-episode evaluation pipeline | Found no DQN checkpoint existed in the repo yet (Week 3 closed with only a Q-Learning checkpoint committed) — trained one via `training/train_dqn.py` as the prerequisite to actually run this issue's task. Then executed 1,000 episodes × 5 policies (5,000 total simulations), 0 execution failures, results consolidated into `evaluation/evaluation_results.csv` and `evaluation/simulation_summary.md`. |
| Day 3 | #102 | Integrate a business dashboard and visualization pipeline | `dashboard/dashboard_app.py` (Streamlit) built to read the Day 2 evaluation outputs and render policy performance, pricing trends, and business-KPI scorecards. Validated with Streamlit's `AppTest` harness across five scenarios (normal load, filtered views, missing data, incompatible schema) — 0 unhandled exceptions in any of them. `reports/dashboard_design.md` documents the architecture, including the compatibility checks between the evaluation output and the dashboard's expected schema. |
| Day 4 | #106 | Validate the complete project and prepare for release | Every component (environment, both agents, evaluation pipeline, dashboard) independently re-exercised together, not just reviewed. Found a real issue in the process: the DQN checkpoint trained on Day 2 evaluates at $7,136 mean revenue, well below the ~$19,500 an earlier training run (issue #78) had reported — most likely explained by DQN training's already-documented lack of RNG seeding (Week 3 Open Items). Reported honestly in `reports/Release_Notes.md` rather than retrained-until-better. Independently re-confirmed on a second machine (Windows), with numerically identical evaluation results, which also surfaced one misplaced deliverable file (`reports/stimulation_summay.md`, a misnamed/misplaced copy of `evaluation/simulation_summary.md`) — corrected via `git mv`. |
| Day 5 | #110 | Finalize project integration and prepare repository submission | `reports/Final_Project_Report.md` (capstone summary of all four weeks) and a restructured `README.md` (explicit Project Overview / Installation / Usage / Results / Repository Structure sections) delivered. Code-level integration confirmed complete; PR review/merge, issue closing, and Kanban board updates flagged as manual GitHub actions outside this sprint's tooling — see Section 7. |

---

## 3. Code Review and Refactor

### 3.1 Naming/path correction caught during cross-platform validation

`evaluate_policies.py`'s `save_results()` originally wrote episode-level
output to `evaluation/results/policy_evaluation_episodes.csv` — functional,
but not the exact filename (`evaluation_results.csv`, directly under
`evaluation/`) the #98 deliverable spec named. Fixed by changing
`EvaluationConfig.results_dir`'s default and the output filename in
`save_results()`, then re-running the full pipeline from a clean state to
confirm the correct file appears automatically with no manual rename step
— re-verified via a wipe-and-rerun before this was considered fixed.

### 3.2 Regression caught mid-validation, fixed same day

While validating the evaluation pipeline (#106), running its
`--smoke-test` mode (20 episodes) — as part of confirming it writes output
correctly — briefly overwrote the committed 1,000-episode
`evaluation_results.csv` with 20-episode data, since both write to the
same filename by design. Caught immediately via a row-count check (5,001
lines expected, found 101), and the full 1,000-episode evaluation was
re-run to restore the correct file before anything was finalized. No
downstream deliverable (dashboard, reports) had been generated from the
truncated file in the meantime.

### 3.3 Static analysis / compile checks

`py_compile` clean across `baselines/`, `configs/evaluation_config.py`,
`evaluation/evaluate_policies.py`, and `dashboard/dashboard_app.py` after
all changes this sprint.

---

## 4. Validation Performed

| Pipeline | Verified how | Result |
|---|---|---|
| `pricing_env` | Built, `check_env`, `reset()`/`step()` exercised directly (#106) | Pass |
| DQN Agent | Checkpoint loaded from disk into a fresh agent instance, valid action produced | Pass |
| Q-Learning Agent | Same load-and-act check as DQN | Pass |
| Baseline policies | All three (`random`, `fixed_price`, `time_based_discount`) built and evaluated successfully alongside the learned agents | Pass |
| Policy Evaluation | Full 1,000-episode × 5-policy run, 0 execution failures, output row counts verified (5,001 lines) | Pass |
| Dashboard | Streamlit `AppTest` — script actually executes against real evaluation output, 0 exceptions, across 5 scenarios | Pass |
| Cross-platform reproducibility | Full evaluation independently re-run on Windows (PowerShell, Python 3.14.2) by the repo owner | Numerically identical results to the Linux run, to the fractional cent |
| `pytest tests/` | Re-run at the start of #106's validation pass | 7/7 passed |

---

## 5. Repository Structure Verification

```
dynamic-pricing-rl/
├── agents/
│   ├── q_learning.py
│   ├── dqn_agent.py
│   └── checkpoints/
│       ├── q_learning_policy.pkl        (Week 2)
│       └── dqn_policy.pt                 (retrained this sprint — Day 2)
├── baselines/                (built this sprint — Day 1, closes Week 3's open item)
│   ├── __init__.py
│   ├── baseline_random.py
│   ├── baseline_fixed.py
│   └── baseline_timebased.py
├── configs/
│   ├── training_config.py
│   ├── experiment_config.py
│   ├── dqn_config.py
│   └── evaluation_config.py    (new — Day 1)
├── dashboard/
│   └── dashboard_app.py         (new — Day 3)
├── evaluation/
│   ├── evaluate_policies.py     (new — Day 1)
│   ├── evaluation_results.csv     (generated — Day 2)
│   ├── policy_evaluation_summary.csv/.json  (generated — Day 2)
│   └── simulation_summary.md       (generated — Day 2)
├── pricing_env/               (Week 1, unchanged this sprint)
├── reports/
│   ├── Weekly_report/
│   │   ├── Week1_report_Abinaya.md
│   │   ├── Week2_report_Abinaya.md
│   │   ├── Week3_report_Abinaya.md
│   │   └── Week4_report_Abinaya.md   (this document)
│   ├── project_planning/       (Week 1, unchanged)
│   ├── training_execution_summary.md   (Week 3)
│   ├── policy_evaluation_design.md      (new — Day 1)
│   ├── dashboard_design.md               (new — Day 3)
│   ├── Release_Notes.md                    (new — Day 4)
│   └── Final_Project_Report.md              (new — Day 5)
├── tests/
│   └── test_replay_buffer_integration.py   (Week 3, unchanged, re-verified passing)
├── training/                    (Week 2–3, unchanged this sprint)
└── utils/, notebooks/             (still empty — see Section 6)
```

No structural issues found beyond the one caught and fixed in Section 3.1/4
(a misnamed `reports/stimulation_summay.md`, corrected to
`evaluation/simulation_summary.md`).

---

## 6. Open Items Carried Forward

- **DQN performance regression relative to Week 3's reported result.** The
  checkpoint currently committed evaluates at $7,136 mean revenue vs. the
  ~$19,500 documented in Week 3's `training_execution_summary.md`. Root
  cause is very likely Week 3's already-flagged, still-unresolved issue:
  DQN training's PyTorch RNG is unseeded, so different runs can converge
  to meaningfully different policies. Recommended fix (seed the RNG,
  retrain multiple seeds, re-evaluate) is documented in
  `reports/Final_Project_Report.md`, Section 6.3, but not yet implemented.
- **Neither learned agent currently beats the fixed-price baseline** on
  mean revenue in the 1,000-episode comparison. This is the project's
  central open finding heading out of Week 4 — everything needed to
  measure this cleanly now exists and works; closing the gap is the next
  phase's actual research work, not an engineering task.
- **No statistical significance testing** (e.g. paired t-test across
  identical seeds) is run on the evaluation results yet. The episode-level
  CSV is structured to support adding this later without re-running any
  simulations.
- **`notebooks/` and `utils/` remain empty**, as they have since Week 1 —
  not required by any deliverable through Week 4, still reserved for
  future use.
- **Neither agent's hyperparameters have been empirically tuned** — both
  still use literature-standard defaults, a carryover from every prior
  week's reports.

---

## 7. Manual Actions Required (cannot be performed from here)

Issue #110's acceptance criteria include GitHub/project-management actions
outside what code-level work can do directly:

- **Review and merge open Pull Requests into `dev`.**
- **Close GitHub Issues #90, #98, #102, #106, #110.** Based on the
  validation in Sections 3–4 above, all five are functionally complete and
  safe to close, with the Section 6 items explicitly carried forward as
  new/existing issues rather than silently dropped.
- **Move completed Kanban cards to Done.** Same five issues.

---

## 8. Readiness for Final Submission

All code-level acceptance criteria across #90, #98, #102, #106, and #110
are met: every component (environment, both agents, all three baselines,
the evaluation framework, the dashboard) is built, integrated, and
independently validated, including cross-platform reproduction. The
repository is documented (README, five architecture/planning reports, this
sprint report, release notes, and the final project report) and structurally
clean. The three manual GitHub actions in Section 7, plus the DQN
performance follow-up in Section 6, are the only items remaining before
this project is fully closed out.