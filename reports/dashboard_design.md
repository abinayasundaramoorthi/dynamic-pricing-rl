# Business Dashboard & Visualization Pipeline — Design & Architecture

**Issue:** #102 — Integrate Business Dashboard and Visualization Pipeline
**Status:** Implemented and validated end to end (see Section 6)
**Prerequisite reading:** `reports/policy_evaluation_design.md` (#90 — what produces the data this dashboard consumes), `evaluation/simulation_summary.md` (#98 — the actual evaluation run this dashboard was validated against)

---

## 1. Purpose

#90 designed the evaluation framework. #98 executed it and produced real numbers. Neither of those makes the results legible to the audience the project was built for — the problem statement's stakeholder table names a **Revenue Manager** who "reviews the RL agent's pricing strategy on a monitoring dashboard" and a **Business Analyst** who "benchmarks RL performance against current pricing strategies," neither of whom is expected to open a 5,000-row CSV.

This dashboard is that final layer: it reads what the evaluation pipeline already wrote to disk and renders it as policy comparisons, pricing trends, and business-KPI scorecards. It runs no simulations and holds no evaluation logic of its own — see Section 3 for why that separation is deliberate.

---

## 2. Scope of this deliverable

| Deliverable | File | What it is |
|---|---|---|
| Dashboard application | `dashboard/dashboard_app.py` | Streamlit app: loads evaluation outputs, verifies they're compatible, renders four sections |
| Architecture documentation | `reports/dashboard_design.md` | This document |

`dashboard/` existed only as a placeholder (`.gitkeep`) before this issue — this is the first real content in it, matching the `README.md` structure's "planned, Week 4" note.

**New dependency:** `streamlit` (added to `requirements.txt`). Chosen over building a static-HTML/matplotlib report because the problem statement explicitly calls for "a lightweight results dashboard" (Section 6.1) that a Revenue Manager "reviews" — implying something they open and interact with (filter by policy, drill into distributions), not a static image. Streamlit was the lowest-friction way to get an interactive, filterable dashboard without a separate frontend build step, consistent with this project's existing bias toward minimal, single-file, `python -m`-runnable entry points (`training/train_dqn.py`, `evaluation/evaluate_policies.py`).

---

## 3. Why the dashboard has zero simulation/evaluation logic

`dashboard_app.py` never imports `PricingEnvironment`, never constructs a policy, and never calls `evaluate_policies.run_evaluation()`. It reads exactly two files:

```
evaluation/evaluation_results.csv          (episode-level, from #98)
evaluation/policy_evaluation_summary.csv   (aggregate, from #98)
```

This is a deliberate architectural boundary, not an oversight:

- **Re-running 5,000 episodes just to render a chart would be wasteful** and would make dashboard load time depend on simulation time (currently ~11s, but not guaranteed to always be that fast as the environment or agents grow).
- **It keeps "did the evaluation run correctly" (#90/#98's concern) separate from "is the result displayed correctly" (#102's concern).** If a number on the dashboard looks wrong, this separation means the first question is always "is the CSV wrong, or is the rendering wrong" — a diagnosable split, not a monolith where both failure modes look identical.
- **It matches how a Revenue Manager would actually use this in practice**: evaluation is expensive-ish and run periodically (e.g. after retraining an agent); the dashboard is opened frequently to review whatever the latest run produced. Decoupling them means the dashboard loads instantly regardless of when the underlying evaluation last ran.

The cost of this boundary is that the dashboard is only as fresh as the last evaluation run — made explicit via the sidebar's "Reload evaluation data" button (re-reads the CSVs from disk; does not re-run any simulation) and the file-not-found fallback (Section 5) if no evaluation has ever been run.

---

## 4. Compatibility verification (the actual "Verify compatibility" task)

Issue #102 explicitly lists "Verify compatibility between: Policy Evaluation, Business Metrics, Visualization Components" as a task, not just a nice-to-have. This is implemented as real, executable checks in `dashboard_app.py`, not just a design-time assumption:

1. **Policy Evaluation ↔ this dashboard's schema** (`load_episode_results()`, `load_summary_results()`): each loader checks the CSV actually contains every column the dashboard's charts and tables read (`REQUIRED_EPISODE_COLUMNS`, `REQUIRED_SUMMARY_COLUMNS`) before returning it, raising a specific `DataCompatibilityError` naming exactly which column is missing if `evaluate_policies.py` ever changes its output schema.
2. **Policy Evaluation ↔ Policy Evaluation** (`verify_cross_source_compatibility()`): confirms `evaluation_results.csv` and `policy_evaluation_summary.csv` actually describe the same run — same set of policies in both, and each policy's row count in the episode-level file matches the `num_episodes` the summary file claims. This catches a stale or partially-regenerated file (e.g. one file re-run, the other not).
3. **Policy Evaluation ↔ Business Metrics**: the summary CSV already contains `meets_*_target` boolean columns, computed by `evaluate_policies.py` at evaluation time using `BusinessKPIConfig`. Rather than trusting those flags blindly, the dashboard re-derives them from a fresh `BusinessKPIConfig()` instance and confirms they agree — if someone changes a KPI threshold in `configs/evaluation_config.py` without re-running the evaluation pipeline, this check catches the resulting inconsistency instead of silently displaying stale pass/fail flags next to a KPI target that no longer matches.

All three checks run once, at dashboard load, before any chart renders. A failure in any of them surfaces as `st.error()` with the specific mismatch (see Section 5), not a stack trace.

---

## 5. Graceful degradation (no integration issues in practice)

Three failure modes were identified and are handled without crashing the app (verified via `AppTest`, Section 6):

| Situation | Behavior |
|---|---|
| Evaluation has never been run (no CSVs exist yet) | `st.warning()` with the exact command to run (`python -m evaluation.evaluate_policies`) — not a raw `FileNotFoundError` traceback |
| CSVs exist but are from an incompatible/older pipeline version | `st.error()` naming the specific missing column(s) or mismatch, per Section 4 |
| User deselects every policy in the sidebar filter | `st.info()` prompting them to select at least one — not an empty/broken chart |

This is the same "fail loud with a specific, actionable message, never a bare crash" philosophy already used in `evaluate_policies.py`'s `build_policies()` (missing checkpoints) and `EvaluationConfig.__post_init__` (invalid config), applied at the presentation layer.

---

## 6. Dashboard layout and structure

```
┌─────────────────────────────────────────────────────────┐
│  Header: title + 4 KPI cards                             │
│  (top policy, its revenue, reference baseline, # policies)│
├─────────────────────────────────────────────────────────┤
│  Policy Performance                                       │
│    - Bar chart: mean revenue per policy                   │
│    - Full comparison table (sortable, all metrics + flags)│
├─────────────────────────────────────────────────────────┤
│  Pricing Trends                                            │
│    - Bar chart: mean price charged per policy              │
│    - Bar chart: mean discount depth per policy              │
│    - Expandable: per-episode revenue distribution (filterable)│
├─────────────────────────────────────────────────────────┤
│  Business Metrics                                           │
│    - One scorecard per policy: revenue uplift / sell-through│
│      / spoilage, each flagged against BusinessKPIConfig      │
└─────────────────────────────────────────────────────────┘
         ▲
         │ Sidebar: policy multiselect filter + reload button
```

Layout decisions:

- **Header-first, KPI-cards-first.** A Revenue Manager opening this dashboard should see "which policy currently wins and by how much" in the first five seconds, before any chart. This mirrors `simulation_summary.md`'s own Section 5 (consolidated results) as the highest-priority information, just rendered instead of written prose.
- **Policy Performance before Pricing Trends before Business Metrics.** Ordered from "which policy is best" (the headline question) → "how does it achieve that" (price/discount behavior, the mechanism) → "does it actually meet our targets" (the KPI scorecard, the accountability check). This ordering intentionally separates *ranking* from *goal attainment* — a policy can be the best of a bad set without clearing any KPI target (as the current run shows; see Section 7), and collapsing those two questions into one section would obscure that.
- **Sidebar filter, not per-section filters.** One multiselect controls every section, so switching "compare only the two RL agents" vs. "everyone" doesn't require re-filtering four separate times.
- **Revenue distribution is collapsed behind an expander**, not shown by default — it's the most detailed/least immediately-actionable view (a Revenue Manager's first question is "who wins," not "what's the full distribution shape"), consistent with progressive disclosure rather than front-loading every chart.

---

## 7. Validation performed (issue #102's "Validate end-to-end dashboard execution")

This wasn't just written and assumed to work — it was actually executed and checked, using Streamlit's `AppTest` harness (which runs the real script and surfaces any exception, rather than only checking that a web server process starts):

1. **Normal load against the real #98 evaluation output** (`evaluation/evaluation_results.csv`, 5,000 rows; `policy_evaluation_summary.csv`, 5 rows) — 0 exceptions, all four sections rendered, 19 `st.metric` widgets populated, 1 comparison dataframe rendered.
2. **Sidebar filter interaction** — deselecting all policies produces the expected `st.info()` prompt and zero metrics (not a broken chart); re-selecting a 2-policy subset correctly re-renders with 10 metrics (scoped to just those two).
3. **Missing-data path** — pointed the dashboard at an empty `evaluation/` directory (simulating a fresh checkout where evaluation has never run): 0 exceptions, correct `st.warning()` with the exact remediation command.
4. **Schema-incompatible data path** — fed the dashboard a hand-truncated `evaluation_results.csv` missing 8 of the required columns: 0 exceptions, correct `st.error()` naming every missing column.
5. **Manual smoke test** — `streamlit run dashboard/dashboard_app.py --server.headless true`, confirmed the server starts and serves HTTP 200.

All five checks passed with zero unhandled exceptions — the concrete basis for this issue's "Dashboard loads successfully" and "No integration issues" acceptance criteria.

---

## 8. Usage

```bash
# One-time: make sure evaluation has actually been run
python -m evaluation.evaluate_policies

# Launch the dashboard
streamlit run dashboard/dashboard_app.py
```

Opens in the browser at `http://localhost:8501` by default. Use the sidebar to filter which policies are shown, or click "Reload evaluation data" after re-running the evaluation pipeline to pick up fresh numbers without restarting the Streamlit process.

---

## 9. Acceptance criteria — how this deliverable satisfies them

| Acceptance criterion | How it's satisfied |
|---|---|
| Dashboard loads successfully | Verified via `AppTest` (0 exceptions across 5 scenarios, Section 6) and a manual `streamlit run` smoke test (HTTP 200) |
| Evaluation outputs integrated correctly | Reads `evaluation_results.csv` and `policy_evaluation_summary.csv` directly, with schema and cross-source compatibility checks (Section 4) run before any rendering |
| Dashboard architecture documented | This document |
| No integration issues | All three identified failure modes (missing data, incompatible schema, empty filter) degrade gracefully with an actionable message, not a crash (Section 5, verified in Section 6) |

---

## 10. Known limitations / follow-up work

- **No auto-refresh / file-watching.** The dashboard reads the CSVs once per session (cached via `st.cache_data`) and only re-reads on an explicit "Reload evaluation data" click. A future version could watch the file's mtime and prompt automatically when it changes.
- **No historical run comparison.** This dashboard shows only the single most recent evaluation run's output. Comparing "this week's retrained DQN vs. last week's" would require the evaluation pipeline to write timestamped/versioned output directories — not something `evaluate_policies.py` currently does, and out of scope for this issue.
- **Price-trajectory-over-time charts are not included.** The current `evaluation_results.csv` schema stores per-episode *aggregates* (`mean_price`, `mean_discount_depth_pct`), not the full day-by-day price trajectory within an episode. A true "price over the 30-day selling horizon" chart would require `evaluate_policies.py` to additionally persist per-step price data — a change to the evaluation pipeline (#90/#98's domain), not something this dashboard can add on its own without that upstream data.