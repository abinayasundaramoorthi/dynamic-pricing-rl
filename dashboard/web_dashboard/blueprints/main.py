"""
blueprints/main.py

HTML page routes for the Flask web dashboard. Every page renders a
static shell template; the actual data is fetched client-side from
`blueprints/api.py`'s JSON endpoints (see each template's
`{% block extra_js %}`), so a page load is always fast and the data
shown is always current, not baked into the server-rendered HTML at
request time.
"""

from flask import Blueprint, render_template

main_bp = Blueprint("main", __name__)


@main_bp.route("/")
def index():
    """Landing page with AI insights summary."""
    return render_template("index.html")


@main_bp.route("/dashboard")
def dashboard():
    """Main Revenue Management Dashboard."""
    return render_template("dashboard.html")


@main_bp.route("/recommendations")
def recommendations():
    """AI Recommendation and Explainability panel."""
    return render_template("recommendations.html")


@main_bp.route("/events")
def events():
    """Event calendar and Demand Shock alerts."""
    return render_template("events.html")


@main_bp.route("/simulator")
def simulator():
    """What-if scenario simulator."""
    return render_template("simulator.html")
