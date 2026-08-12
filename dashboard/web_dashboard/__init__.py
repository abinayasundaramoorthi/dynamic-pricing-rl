"""
dashboard.web_dashboard package

A Flask web dashboard, nested inside the project's existing `dashboard/`
folder alongside the original Streamlit app (`dashboard/dashboard_app.py`,
`dashboard/pricing_visualizations.py`, etc.) — neither of which this
package modifies.

Why Flask, and why here (not a new top-level folder)
------------------------------------------------------
This app is intentionally placed at `dashboard/web_dashboard/` rather
than a separate top-level `web_dashboard/` folder, so the project has
ONE `dashboard/` home for all dashboard code instead of two
similarly-named top-level folders. It uses Flask + server-rendered
Jinja templates + Chart.js (not Streamlit) for the "Advanced Revenue
Management Dashboard Enhancement" feature, per project requirements.

Contents
--------
    app.py                    -- `create_app()` Flask application factory
    services.py                -- shared business logic (agent loading/
                                   caching, real environment rollouts,
                                   event loading) used by the API blueprint
    blueprints/main.py          -- HTML page routes
    blueprints/api.py           -- JSON API routes (real data only —
                                   every number comes from this project's
                                   real environment, real trained agents,
                                   or real evaluation CSVs)
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
