"""
advanced_dashboard package

Public API for Feature 2 — Advanced Revenue Management Dashboard
Enhancement.

This package is now a pure DATA layer: KPI computation
(`kpi_components.py`) and Chart.js-ready series builders (`charts.py`),
plus CSV/JSON export byte helpers (`export_utils.py`). Nothing here
depends on a UI framework — the actual dashboard UI lives in
`dashboard/web_dashboard/` (a Flask app with Bootstrap + Chart.js
templates), whose `blueprints/api.py` imports these functions to build
its JSON responses.

    from additional_features.advanced_dashboard import kpi_components, charts, export_utils

    summary_headline = kpi_components.summary_headline(summary_df)
    revenue_series = charts.revenue_trend_series(episodes_df)
"""

from . import charts, export_utils, kpi_components

__all__ = [
    "kpi_components",
    "charts",
    "export_utils",
]
