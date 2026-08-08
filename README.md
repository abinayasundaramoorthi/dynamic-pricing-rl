# additional_features + dashboard/web_dashboard — Flask Edition

This is an update to the previous `additional_features` delivery. Two things changed, both requested directly:

1. **No more Streamlit for the new dashboard.** The Advanced Revenue Management Dashboard (Feature 2) is now a **Flask** app with Bootstrap 5 + Chart.js, built from the design you liked (`revenue-ai-production.zip`).
2. **No duplicate top-level dashboard folder.** The Flask app lives at `dashboard/web_dashboard/`, nested inside your existing `dashboard/` folder — not a second top-level `web_dashboard/` folder next to it.

Your original `dashboard/dashboard_app.py` (Streamlit) and every other core project file are **byte-for-byte untouched**. This is purely additive.

## What to do with this zip

Extract it into your project root. It will merge two things into your existing tree:

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
```

## New dependency

**Flask** is now a dependency (not in your original `requirements.txt`):

```bash
pip install flask
```

Nothing else changed — still pandas/numpy/torch/gymnasium/matplotlib/streamlit (Streamlit stays because your *original* dashboard still uses it; the *new* dashboard does not).

## Running the new Flask dashboard

From your project root:

```bash
python -m dashboard.web_dashboard.app
```

Then open **http://localhost:8080**. Override the port with the `PORT` environment variable if needed.

Your original Streamlit dashboard still runs exactly as before:

```bash
python -m streamlit run dashboard/dashboard_app.py
```

## What's real vs. what's a template

Every number in the Flask dashboard comes from one of three real sources — nothing is `Math.random()` or hardcoded:

- **Evaluation CSVs** (`evaluation/evaluation_results.csv`, `evaluation/policy_evaluation_summary.csv`) — same files your original dashboard already reads.
- **Your real trained agent checkpoints** (`agents/checkpoints/dqn_policy.pt`, `q_learning_policy.pkl`) — the "Live AI Recommendation" and "What-If Simulator" pages run actual rollouts through your real `PricingEnvironment` with these checkpoints.
- **An event calendar file** (CSV or JSON) for the Demand Shock page. No real calendar of yours exists yet, so it falls back to `additional_features/demand_shock_detection/sample_events.csv` — **explicitly labeled as a sample** in the UI (a visible banner says so) and in the API response (`"is_sample_data": true`). Point the Events page at your own file's path to use real events instead.

One design change from the reference zip you liked: the "Competitor Price War" market-condition option was removed from the simulator, because this project's demand model has no competitor-pricing mechanism to simulate — adding it would have meant fabricating a feature that doesn't actually do anything. The two market conditions that remain ("High Demand" / "Low Demand") are real: they scale the project's actual `DemandConfig.base_daily_arrival_rate` parameter.

## Verification

Both run clean against this exact codebase, including your real trained checkpoints:

```bash
python -m pytest additional_features/tests -q
# 71 passed

python -m additional_features.deployment_smoke_test
# RESULT: all checks passed — safe to deploy.
```

The smoke test now also boots the real Flask app with a test client and hits every page route plus the core API routes (including a real `/api/simulate` POST), so a broken template or route is caught before you ever run `python -m dashboard.web_dashboard.app` by hand.
