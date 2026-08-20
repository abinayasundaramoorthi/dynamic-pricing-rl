# Flask Dashboard Performance Fix — Notes

## Which app this is

Your project has **three** web UIs. Per your own `README.md` ("Usage" →
"Full policy evaluation" section) and `dashboard/web_dashboard/__init__.py`,
the Flask app for live RL pricing recommendations is:

```
python -m dashboard.web_dashboard.app
```

(`web_app/app.py` is a separate, smaller Flask app used only for the
human-feedback and digital-twin features — it doesn't touch the RL
agents, so it wasn't the source of the reported slowness and was left
untouched. `dashboard/dashboard_app.py` is the original Streamlit
business dashboard, also untouched and still runnable exactly as
before with `streamlit run dashboard/dashboard_app.py`.)

## What was actually causing the slowness

I inspected the real code before changing anything and benchmarked it
(`time.perf_counter()`, fresh Python subprocess per run — see numbers
below). Two things stood out; a third followed from digging further.

**1. The Flask app was importing Streamlit for no reason.**
`dashboard/web_dashboard/blueprints/api.py` imported
`load_episode_results`/`load_summary_results` from
`dashboard/dashboard_app.py` — the Streamlit dashboard — just to reuse
two CSV-loading functions. `additional_features/advanced_dashboard/
{charts,kpi_components}.py` did the same thing to reuse
`POLICY_DISPLAY_NAMES`. Because `dashboard_app.py` does
`import streamlit as st` at module scope, **every** Flask process start
paid the cost of importing the entire Streamlit package (and Streamlit's
own `st.cache_data` decorator warns/behaves oddly outside a real
Streamlit runtime — see the `"No runtime found"` warnings in the old
logs) even though the Flask dashboard never renders anything with it.

**2. The DQN checkpoint's first-load cost was landing on the user's
first real click, not on app startup.** `services.py` already caches
loaded agents correctly (`model.eval()`, `torch.no_grad()`, a
process-local `_dqn_agent_cache` dict) — that part of your code was
already right. But because loading was *lazy*, the first person to hit
"Get Recommendation" after the server started was the one who paid the
~0.8–1.1s cost of reading the checkpoint off disk and PyTorch's one-time
kernel warmup. Every click after that was already fast (~1–2ms) — the
problem was specifically "my first click is slow," which matches what
you described.

**3. `debug=True` was hardcoded in `app.run(...)`.** Flask's debug mode
enables the Werkzeug auto-reloader, which is a development convenience
(watches loaded modules for changes and restarts the process) with real
runtime overhead, especially once `torch`/`streamlit` are loaded into
`sys.modules`. It has no business being on by default for a "just run
it" local app.

### What was *not* the problem

Contrary to what's common in RL demo projects, this codebase was **not**
doing any of these things — I checked directly:
- Not retraining or reloading the model per request.
- Not running large evaluations from a live recommendation click
  (`/api/simulate` runs one real episode rollout, tens of steps — cheap
  by design; the actual 1,000-episode evaluation lives in
  `evaluation/evaluate_policies.py`, run offline, never from a route).
- `model.eval()` / `torch.no_grad()` were already in place.
- Page routes (`blueprints/main.py`) already render a static shell and
  fetch data client-side via `fetch()` — no full-page reloads, no
  server-side heavy computation on page load.

## Files changed

| File | Change |
|---|---|
| `dashboard/data_contract.py` | **New.** Streamlit-free module holding the shared CSV-loading logic, required columns, and `POLICY_DISPLAY_NAMES`/`display_name()` — with its own lightweight, mtime-based cache (no `st.cache_data`, no Streamlit import). |
| `dashboard/dashboard_app.py` | Now imports the data contract from `dashboard/data_contract.py` instead of defining it locally, and keeps its `@st.cache_data` wrapping around it. **Behavior is identical** — still runs exactly the same via `streamlit run dashboard/dashboard_app.py`. |
| `additional_features/advanced_dashboard/charts.py` | Imports `POLICY_DISPLAY_NAMES` from `dashboard.data_contract` instead of `dashboard.dashboard_app`. |
| `additional_features/advanced_dashboard/kpi_components.py` | Same change, for `POLICY_DISPLAY_NAMES` / `display_name`. |
| `dashboard/web_dashboard/blueprints/api.py` | Imports `load_episode_results`/`load_summary_results` from `dashboard.data_contract`. Added `GET /api/health`. |
| `dashboard/web_dashboard/services.py` | Added `warm_agents()` (eager-loads any available agent checkpoint once, at startup) and `is_dqn_agent_loaded()` / `is_q_learning_agent_loaded()` (cheap, no-I/O cache checks used by `/api/health`). |
| `dashboard/web_dashboard/app.py` | Calls `services.warm_agents()` inside `create_app()`; logs startup timing with `time.perf_counter()`; `debug`/reloader now come from `FLASK_DEBUG` env var (default off) instead of being hardcoded `True`; added `threaded=True`. |
| `tests/test_web_dashboard_api.py` | **New.** Covers `/api/health`, `/api/recommendation` (dqn + q_learning, valid/invalid input), missing-checkpoint → 404, and repeated-request cache reuse. |

Nothing in `pricing_env/`, `agents/`, `training/`, or `evaluation/` was
touched — the RL system (Q-Learning, DQN, environment, reward, action
space, checkpoints) is exactly as you built it.

## Before vs after (measured, not invented)

Measured with `time.perf_counter()` in a **fresh Python subprocess per
run** (so import caching from a shared process can't skew the numbers),
using the actual checkpoints in `agents/checkpoints/`.

| Operation | Before | After | Improvement |
|---|---:|---:|---:|
| Process startup (`create_app()` incl. imports) | 2.13s | 2.41s | -0.28s (see note) |
| Homepage (`GET /`) | 0.007s | 0.008s | ~no change (was already fast) |
| **First** `/api/recommendation` click after startup | **0.840s** | **0.0024s** | **~350× faster** |
| Repeat `/api/recommendation` clicks | 0.0016s | 0.0014s | ~no change (already cached) |
| `/api/kpis` (first load) | 0.0094s | 0.0087s | ~no change (already cached) |
| **Startup + time until first click returns** | **2.97s** | **2.41s** | **~19% faster**, and the slow part moves to *before* you ever touch the browser |
| Streamlit imported into the Flask process? | Yes | **No** | — |

**Note on startup:** total startup went up slightly (~0.28s) because the
DQN checkpoint load that used to happen lazily on your first click now
happens eagerly during `create_app()` instead. That's a deliberate
trade: the RL model has to be loaded *somewhere* before it can serve a
recommendation — the fix moves that one-time cost to happen once, while
the server is starting (before you're looking at the page), instead of
during your first interaction with it. Net effect: by the time the page
finishes loading in your browser, the first click is already instant.

## How to run it (VS Code / Windows)

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m dashboard.web_dashboard.app
```

Open **http://localhost:8080**. Override the port with `set PORT=8081`
(Windows) before running if 8080 is taken. Leave `FLASK_DEBUG` unset for
normal fast use; set `set FLASK_DEBUG=1` only while actively editing
routes/templates and wanting auto-reload.

If a checkpoint hasn't been trained yet, `/api/recommendation` (and
`/api/health`'s `dqn_loaded`/`q_learning_loaded` fields) will tell you
plainly instead of failing silently — train with:

```bash
python -m training.train_agent --agent q_learning --episodes 5000
python -m training.train_dqn
```

## Remaining bottlenecks / honest limitations

- **`import torch` itself costs ~1.8s**, independent of anything in this
  project — that's PyTorch's own import time on this machine, and it's
  paid once at process startup regardless of caching strategy. If you
  never need DQN specifically, running only with the Q-Learning agent
  available would avoid this, but the project is built to support both,
  so I didn't remove that dependency.
- `create_app()` now blocks briefly on agent warm-up before Flask starts
  accepting connections. For this project's size (one process, local
  dev use) that's a good trade — if you ever run multiple worker
  processes in production, each worker pays this warm-up independently
  once (the same as it already would have lazily).
- I did not change `evaluation/evaluate_policies.py`'s 1,000-episode
  runs — those are correctly kept offline/CLI-only and were never wired
  into a live route to begin with.

## Trade-offs introduced

- Startup is very slightly slower so that every user-facing click is
  fast — a deliberate, stated trade-off, not something to "fix" further
  without giving up the win it buys.
- `debug=False` by default means code changes require a manual restart
  now (previously the always-on reloader picked them up automatically).
  Set `FLASK_DEBUG=1` while developing to get the old auto-reload
  behavior back.

---

## Round 2 — Streamlit removed, single Flask app, real India event data

A follow-up pass, in response to three explicit asks: remove Streamlit
entirely, run everything as one Flask app (no duplicates), and use real
India-specific data for testing live recommendations.

### 1. Streamlit removed entirely

- Deleted `dashboard/dashboard_app.py` (the Streamlit business
  dashboard) and removed `streamlit` from `requirements.txt`.
- `dashboard/data_contract.py` — added in Round 1 specifically so the
  Flask app never had to import Streamlit — remains the single home
  for the shared CSV-loading logic. It has no framework dependency at
  all, so nothing needed to change there.
- Fixed the two places that still imported the now-deleted file:
  `additional_features/deployment_smoke_test.py` and
  `additional_features/tests/test_advanced_dashboard.py`, both now
  import from `dashboard.data_contract` instead.
- Verified: `"streamlit" in sys.modules` is `False` after
  `create_app()` runs, and after every request.

### 2. Down to one Flask app

The project previously ran a second, separate Flask app —
`web_app/app.py` — for the Human-Feedback-Learning and
Digital-Twin-Simulation features. That's now merged into the main app:

- New blueprint `dashboard/web_dashboard/blueprints/legacy.py` (the
  route handlers and business logic are unchanged from the original
  `web_app/app.py` — only the Flask-app-vs-Blueprint wiring changed),
  registered in `create_app()` at `url_prefix="/legacy"`.
- Its templates/static/data moved from the old standalone `web_app/`
  folder to `dashboard/web_dashboard/legacy/{templates,static,evaluation}/`.
  Templates were namespaced into a `legacy/` subfolder
  (`legacy/templates/legacy/*.html`) — **required**, not cosmetic:
  Flask's app-level template loader takes priority over a blueprint's
  own loader, so without namespacing, the legacy blueprint's
  `base.html`/`dashboard.html`/etc. were silently shadowed by the main
  dashboard's identically-named templates, and every `/legacy/*` page
  rendered the wrong UI (right route, wrong template — the kind of bug
  that's easy to ship if you only check the HTTP status code).
- Every previously-hardcoded URL in those templates/JS
  (`/human-feedback`, `/digital-twin`, `/api/feedback`, `/api/simulate`,
  `url_for('static', ...)`) was updated to the new `/legacy/...`
  paths / `legacy.static` endpoint, so nothing 404s.
- Old standalone `web_app/` folder deleted.
- There is now exactly one `Flask(__name__)` instantiation in the
  entire project (`dashboard/web_dashboard/app.py`).
- New tests (`tests/test_web_dashboard_api.py`): confirm Streamlit
  never gets imported, confirm the `legacy` blueprint's routes/static
  files work, and confirm `create_app()` returns one `Flask` instance
  owning `main.*`, `api.*`, and `legacy.*` endpoints together.

### 3. Real India event data

`additional_features/demand_shock_detection/sample_events.csv` — the
bundled fallback event calendar used by `/api/alerts` when no custom
event source is supplied — contained two clearly-labeled placeholder
rows ("SAMPLE - Regional Tech Conference", etc.). Replaced with real
2026 Indian festival/holiday dates, verified via web search rather than
recalled from memory (Diwali's date in particular shifts by weeks
year-to-year on the lunisolar calendar, so it isn't guessable):

| Event | 2026 date |
|---|---|
| Republic Day | 26 Jan |
| Holi | 4 Mar |
| Independence Day | 15 Aug |
| Ganesh Chaturthi | 14 Sep |
| Navratri (start) | 11 Oct |
| Dussehra | 20 Oct |
| Diwali (Lakshmi Puja) | 8 Nov |
| Bhai Dooj | 10 Nov |
| Christmas & New Year Peak | 25 Dec |

**Honest caveat on `expected_impact`/`confidence`:** the *dates* above
are real, sourced facts. The demand-lift multipliers and confidence
values are my own reasonable analyst-style estimates for a travel/
hospitality context (e.g. Diwali highest at +40%, Republic Day/
Independence Day more modest at +15-20%) — not pulled from any specific
market-research dataset, because I don't have one. This matches how
`event_schema.py` already documents this field: *"Analyst-supplied...
this module never invents or estimates it"* — meaning the module itself
doesn't invent numbers, but whoever fills in the CSV always is
supplying a judgment call, real dataset or not. If you have your own
market data for these dates, replace the `expected_impact`/`confidence`
columns with real figures; the dates and event names don't need to
change.

**Try it live:** with the Flask app running,

```
GET /api/alerts?lookahead=180&reference_date=2026-08-20
```

returns real upcoming festivals (Ganesh Chaturthi, Navratri, Dussehra,
Diwali, Bhai Dooj) with computed risk levels and suggested pricing
adjustments — this is genuine test data for the live-recommendation
demand-shock feature, not a fabricated example.
