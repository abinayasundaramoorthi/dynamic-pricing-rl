"""
dashboard.web_dashboard package

The project's ONE Flask web application. Originally added alongside two
other apps — a Streamlit dashboard (`dashboard/dashboard_app.py`) and a
second Flask app (`web_app/app.py`) for the human-feedback and
digital-twin features. Both have since been merged in or removed:
Streamlit is no longer a project dependency (its data-loading logic
lives on in `dashboard/data_contract.py`, which this package already
used), and the human-feedback/digital-twin routes are now the `legacy`
blueprint below, mounted at `/legacy`.

Contents
--------
    app.py                      -- `create_app()` Flask application factory
    services.py                 -- shared business logic (agent loading/
                                    caching, real environment rollouts,
                                    event loading) used by the API blueprint
    blueprints/main.py          -- RL pricing dashboard HTML page routes
    blueprints/api.py           -- JSON API routes (real data only —
                                    every number comes from this project's
                                    real environment, real trained agents,
                                    or real evaluation CSVs)
    blueprints/legacy.py        -- Human-Feedback-Learning + Digital-Twin
                                    -Simulation routes, mounted at
                                    `/legacy` (templates/static/data for
                                    these live in `dashboard/web_dashboard/
                                    legacy/`, kept separate from the main
                                    dashboard's own `templates/`/`static/`)
    templates/                  -- Jinja2 templates (Bootstrap 5 + Chart.js)
    static/                     -- CSS/JS assets

Data layer
----------
All KPI computation and chart-series building is delegated to the
existing, framework-agnostic
`additional_features.advanced_dashboard.{kpi_components,charts,export_utils}`
modules — this package contains only Flask routing/wiring, not a second
copy of that logic.

Running the app
----------------
From the project root (so the `dashboard`/`additional_features`/
`pricing_env`/... packages are importable):

    python -m dashboard.web_dashboard.app

or, with Flask's own CLI:

    set FLASK_APP=dashboard.web_dashboard.app:create_app   (Windows)
    export FLASK_APP=dashboard.web_dashboard.app:create_app  (macOS/Linux)
    flask run --port 8080
"""
