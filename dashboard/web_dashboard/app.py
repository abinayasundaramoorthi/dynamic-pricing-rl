"""
app.py

Flask application factory for the Dynamic Pricing RL project. This is
now the ONE Flask app for the whole project — the RL pricing dashboard
(Explainable AI Pricing, Revenue Management, Demand-Shock Alerts) plus
the Human-Feedback-Learning and Digital-Twin-Simulation features
(previously a second, separate Flask app at `web_app/app.py`, now the
`legacy` blueprint registered below at url_prefix "/legacy"). The
original Streamlit business dashboard (`dashboard/dashboard_app.py`)
has been removed — its data-loading logic lives on in
`dashboard/data_contract.py`, which this app already used.

Run it from the project root with:

    python -m dashboard.web_dashboard.app

which starts the Flask dev server on http://localhost:8080 by default
(override with the PORT environment variable).

Performance-relevant environment variables (all optional):

    FLASK_DEBUG=1   Enable Flask's debugger + auto-reloader (off by
                     default). The reloader is a development convenience,
                     not something a "just run it and use it" local app
                     needs — leaving it on by default was previously
                     adding startup/runtime overhead for no benefit
                     during normal use. Turn it on only while actively
                     editing template/route code.
    PORT=8080        Port to listen on.
"""

import os
import sys
import time
from pathlib import Path

from flask import Flask, render_template

# Ensure the project root (the directory containing pricing_env/, agents/,
# additional_features/, dashboard/, ...) is importable, exactly like the
# rest of this project's entry-point scripts already assume when run
# directly.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def create_app() -> Flask:
    """Application factory (standard Flask pattern) — build and configure the Flask app."""
    start = time.perf_counter()

    app = Flask(__name__, static_folder="static", template_folder="templates")

    app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-key-change-in-production")
    app.config["JSON_SORT_KEYS"] = False

    from dashboard.web_dashboard.blueprints.api import api_bp
    from dashboard.web_dashboard.blueprints.legacy import legacy_bp
    from dashboard.web_dashboard.blueprints.main import main_bp
    from dashboard.web_dashboard import services

    app.register_blueprint(main_bp)
    app.register_blueprint(api_bp, url_prefix="/api")
    # Human-Feedback-Learning + Digital-Twin-Simulation, mounted under
    # /legacy so their routes (was "/", "/human-feedback", "/digital-twin",
    # "/api/feedback", "/api/simulate") never collide with the main
    # dashboard's own "/" or "/api/simulate" routes.
    app.register_blueprint(legacy_bp, url_prefix="/legacy")

    @app.errorhandler(404)
    def page_not_found(_e):
        return render_template("errors/404.html"), 404

    @app.errorhandler(500)
    def internal_server_error(_e):
        return render_template("errors/500.html"), 500

    # Warm the agent cache now, at startup, instead of leaving it to load
    # lazily on whichever request happens to be first. Without this, the
    # first live recommendation a person actually clicks pays the full
    # checkpoint-load + PyTorch first-call warmup cost; every request
    # after that is already fast (services.py caches loaded agents), so
    # this just moves that one-time cost out of the user-facing path.
    # Missing checkpoints are not a startup error — same as any request,
    # they're skipped (see services.warm_agents docstring).
    warm_start = time.perf_counter()
    warmed = services.warm_agents()
    warm_elapsed = time.perf_counter() - warm_start
    for agent_name, ok in warmed.items():
        if ok:
            app.logger.info("Warmed %s agent at startup (%.3fs total warm-up).", agent_name, warm_elapsed)
        else:
            app.logger.warning(
                "No trained '%s' checkpoint found — /api/recommendation?agent=%s "
                "will 404 until it's trained.",
                agent_name,
                agent_name,
            )

    app.logger.info("Flask app ready in %.3fs (startup).", time.perf_counter() - start)
    return app


if __name__ == "__main__":
    flask_app = create_app()
    port = int(os.environ.get("PORT", 8080))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    # threaded=True: lets one slow/streaming request (e.g. a CSV export)
    # not block a concurrent, unrelated request (e.g. a live recommendation)
    # on Flask's single-threaded dev server default.
    flask_app.run(host="0.0.0.0", port=port, debug=debug, use_reloader=debug, threaded=True)
