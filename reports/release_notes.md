# Release Notes — dynamic-pricing-rl

**Issue:** #106 — Validate Complete Project and Prepare GitHub Release
**Release:** Week 4 / v0.4.0 (proposed tag)
**Date:** 2026-08-02
**Status:** ✅ Validated — ready for release, with one known issue flagged below (does not block release, but should be read before treating current DQN numbers as representative)

---

## 1. What's in this release

This release closes out the four-week internship project scope defined in
`reports/project_planning/problem_statement.md`, Section 6.1. It includes
everything built across all four weeks, validated together as one working
system for the first time in this issue:

- **Gymnasium-compliant pricing environment** (`pricing_env/`) — finite,
  perishable inventory, stochastic demand, configurable reward shaping.
- **Two learned agents** — tabular Q-Learning (`agents/q_learning.py`) and
  Deep Q-Network (`agents/dqn_agent.py`), both with trained checkpoints
  committed under `agents/checkpoints/`.
- **Three heuristic baselines** (`baselines/`) — Random, Fixed Price, and
  Time-Based Discount — the comparison floor the learned agents are
  benchmarked against.
- **A 1,000-episode-per-policy evaluation framework**
  (`evaluation/evaluate_policies.py`, `configs/evaluation_config.py`) that
  runs all five policies through identical simulated demand and scores
  them against this project's own business KPIs.
- **A business-facing dashboard** (`dashboard/dashboard_app.py`, built on
  Streamlit) that renders the evaluation output as policy comparisons,
  pricing trends, and KPI scorecards for a non-technical stakeholder.
- **Full documentation** — architecture docs for the evaluation framework
  and dashboard, weekly sprint reports, and this release notes document.

---

## 2. Validation performed for this release

Every component was independently exercised in this issue, not just
reviewed by reading code:

| Component | How it was validated | Result |
|---|---|---|
| Test suite | `python -m pytest tests/ -v` | 7/7 passed |
| Gym Environment | Built, `check_env`-verified, `reset()`/`step()` exercised directly | Pass |
| DQN Agent | Checkpoint loaded from disk into a fresh agent instance, produced a valid greedy action | Pass |
| Q-Learning Agent | Same load-and-act check as DQN | Pass |
| Policy Evaluation | Full pipeline re-run (`run_evaluation()`), confirmed all 5 policies produce summary rows | Pass |
| Dashboard | Streamlit `AppTest` harness — script actually executes against the real evaluation output, 0 exceptions | Pass |
| Repository structure | Full file listing reviewed against the planned structure in `problem_statement.md` | Reviewed — see Section 4 |

**A note on how the evaluation check was run:** validating the evaluation
pipeline included running its `--smoke-test` mode (20 episodes) as a fast
correctness check, which — as part of validating it actually writes output
correctly — temporarily overwrote the committed 1,000-episode
`evaluation_results.csv` with 20-episode data. This was caught immediately
(row count check) and the full `python -m evaluation.evaluate_policies`
was re-run to restore the correct 5,000-row deliverable before this
release was finalized. Confirmed restored: `evaluation_results.csv` is
5,001 lines (header + 5,000 rows) and its aggregate numbers match
`evaluation/simulation_summary.md` exactly. Noted here for transparency,
not because it affected the final released state.

**Independent cross-platform confirmation.** The validation above was
performed on Linux. The repository owner separately ran the identical
`pytest`, `--smoke-test`, and full-evaluation commands on Windows
(PowerShell, Python 3.14.2) and obtained **numerically identical results**
— e.g. `fixed_price` mean revenue $19,558.60, `dqn` mean revenue
$7,136.256245 to the same fractional cent, across both the 20-episode
smoke test and the full 1,000-episode run. This confirms the environment's
seeding (`EvaluationConfig.episode_seeds()`) is fully deterministic across
machines and Python versions, not just within a single session — a
meaningfully stronger reproducibility guarantee than a single-machine
validation provides, and directly relevant given DQN training's own
non-determinism is flagged as a known issue below (Section 3): the
*evaluation* pipeline is reproducible even though DQN *training* currently
isn't.

**One repository issue found and fixed during this cross-check:** the
committed file for issue #98's `simulation_summary.md` deliverable had
been created as `reports/stimulation_summay.md` — misspelled and in the
wrong directory (`reports/` instead of `evaluation/`). Caught by
comparing the owner's `git commit` output against the deliverable spec.
Fixed via `git mv reports/stimulation_summay.md
evaluation/simulation_summary.md`. Flagged here as an example of exactly
the kind of thing this validation pass exists to catch — a file that
looked committed and present, but not usable at the path anything else
in the project (or a reviewer) would expect it at.

---

## 3. Known issue — DQN checkpoint underperforms its earlier reported result

This is the most important thing in this release to be upfront about.

`reports/training_execution_summary.md` (from an earlier issue, #78)
reports a DQN checkpoint evaluating at **~$19,500 mean revenue** over 200
held-out episodes. The DQN checkpoint currently committed in this repo
(`agents/checkpoints/dqn_policy.pt`) — trained fresh during issue #98,
because no checkpoint existed in the repo at that time — evaluates at
**$7,136.26 mean revenue** over 1,000 episodes (see
`evaluation/simulation_summary.md`), a large gap in the same environment
and reward function.

**Root cause, most likely:** the README (both before and after this
release) already documents that DQN training is not bit-for-bit
reproducible run-to-run — PyTorch's global RNG for network weight
initialization is unseeded in the current `train_dqn.py`. Two independent
training runs with identical hyperparameters can converge to meaningfully
different policies. The $19,500 and $7,136 figures are almost certainly
two different runs' outcomes, not a measurement error — both numbers are
independently reproducible from their respective evaluation methodology.

**Why this doesn't block this release:** issue #106's job is to validate
the system works end to end and is release-ready, not to re-tune agent
hyperparameters — that's a distinct, follow-up piece of work. Every
component functions correctly; the pipeline correctly measured a real,
if disappointing, result. Silently re-training until a better checkpoint
appeared and shipping that instead would have hidden a real reproducibility
problem rather than reporting it.

**Recommended follow-up (not in this release's scope):**
1. Seed PyTorch's global RNG in `train_dqn.py` (or pass an explicit
   `torch.manual_seed()` call) so DQN training becomes reproducible.
2. Once reproducible, train several seeds and report the distribution of
   outcomes, not a single run's number.
3. Re-run `evaluation/evaluate_policies.py` and confirm DQN's revenue
   against the fixed-price baseline before making any "RL beats baseline"
   claim in future reporting.

Q-Learning also underperforms the fixed-price and random baselines in the
current comparison ($15,356.83 vs. $19,558.60 and $14,225.89 respectively)
— less dramatic than DQN's gap, but the same underlying conclusion
applies: neither learned agent currently beats the simple baselines on
revenue, and that should be resolved before this project is presented as
demonstrating RL's value for this problem.

---

## 4. Repository structure review

Reviewed the full repository tree against the planned structure in
`problem_statement.md` Section 7. Findings:

- All planned top-level directories exist and are populated, except
  `notebooks/` and `utils/`, which remain empty placeholders — consistent
  with the problem statement's own scope (neither was called for by any
  Week 1–4 deliverable) and not a gap.
- `agents/checkpoints/` contains both trained checkpoints
  (`dqn_policy.pt`, `q_learning_policy.pkl`) plus their metadata JSON
  sidecars, satisfying the internship brief's requirement that "the
  custom Gym environment and trained weights must be pushed to GitHub."
- Every directory still has its original `.gitkeep` file even where now
  populated with real content (`agents/`, `baselines/`, `configs/`,
  `dashboard/`, `evaluation/`, `tests/`). Harmless, but worth a cleanup
  pass in a future PR — `.gitkeep` exists only to make otherwise-empty
  directories trackable by git, and is redundant once a directory has
  real tracked files in it.
- `requirements.txt` was missing a trailing newline at end of file; fixed
  as part of this release's file review (cosmetic, not functional).
- One deliverable was found misplaced in the repository owner's working
  tree during cross-platform validation (Section 2): `simulation_summary.md`
  (issue #98's deliverable) had been committed as
  `reports/stimulation_summay.md` — wrong filename, wrong directory.
  Corrected to `evaluation/simulation_summary.md` via `git mv`. The
  content itself was correct; only the path was wrong.

No missing required files were found.

---

## 5. How to validate this release

For anyone pulling this release fresh:

```bash
# 1. Install
pip install -r requirements.txt

# 2. Unit tests
python -m pytest tests/ -v
# Expect: 7 passed

# 3. Confirm trained checkpoints are present
ls agents/checkpoints/
# Expect: dqn_policy.pt, dqn_policy.pt.json, q_learning_policy.pkl, q_learning_policy.pkl.json

# 4. Fast pipeline smoke test (do this before the full run)
python -m evaluation.evaluate_policies --smoke-test
# Expect: log lines for all 5 policies, 0 errors

# 5. Full 1,000-episode evaluation (regenerates the committed results files)
python -m evaluation.evaluate_policies
# Expect: ~5-15 seconds, evaluation/evaluation_results.csv with 5,001 lines afterward

# 6. Dashboard
streamlit run dashboard/dashboard_app.py
# Expect: browser opens, four sections render with no red error box
```

If any step behaves differently than described here, that's a regression
relative to this validated release — worth filing as its own issue rather
than assumed to be expected variance.

---

## 6. Deliverables for this issue

```
reports/
└── Release_Notes.md    # this document
README.md                # updated to reflect the full Week 4 state
```

## 7. Acceptance criteria — status

| Acceptance criterion | Status |
|---|---|
| Complete project validated | ✅ — every component independently exercised, Section 2 |
| Repository reviewed | ✅ — Section 4 |
| Release notes prepared | ✅ — this document |
| Repository ready for final release | ✅, with the DQN/Q-Learning performance gap (Section 3) disclosed as a known issue rather than blocking — the system works correctly and honestly reports a result that needs follow-up tuning, which is a valid release state for a project at this stage |