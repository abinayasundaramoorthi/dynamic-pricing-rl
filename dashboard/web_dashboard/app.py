"""
app.py

Flask application factory for the Advanced Revenue Management Dashboard
(Feature 2), nested inside the project's existing `dashboard/` folder.

Run it from the project root with:

    python -m dashboard.web_dashboard.app

which starts the Flask dev server on http://localhost:8080 by default
(override with the PORT environment variable).
"""

import os
import sys
from pathlib import Path

from flask import Flask, render_template

# Ensure the project root (the directory containing pricing_env/, agents/,
# additional_features/, dashboard/, ...) is importable, exactly like the
# rest of this project's entry-point scripts (e.g. dashboard/dashboard_app.py)
# already assume when run directly.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def create_app() -> Flask:
    """Application factory (standard Flask pattern) — build and configure the Flask app."""
    app = Flask(__name__, static_folder="static", template_folder="templates")

    app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-key-change-in-production")
    app.config["JSON_SORT_KEYS"] = False

    from dashboard.web_dashboard.blueprints.api import api_bp
    from dashboard.web_dashboard.blueprints.main import main_bp

    app.register_blueprint(main_bp)
    app.register_blueprint(api_bp, url_prefix="/api")

    @app.errorhandler(404)
    def page_not_found(_e):
        return render_template("errors/404.html"), 404

    @app.errorhandler(500)
    def internal_server_error(_e):
        return render_template("errors/500.html"), 500

    return app


if __name__ == "__main__":
    flask_app = create_app()
    port = int(os.environ.get("PORT", 8080))
    flask_app.run(host="0.0.0.0", port=port, debug=True)
