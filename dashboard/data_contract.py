"""
data_contract.py

Home for the evaluation-output "data contract" (file paths, required
columns, human-readable policy names) and the actual CSV loading logic,
shared by:

  * dashboard/web_dashboard/                             (the Flask app)
  * additional_features/advanced_dashboard/{charts,kpi_components}.py

History
-------
This logic originally lived in `dashboard/dashboard_app.py`, a
Streamlit dashboard that has since been removed from the project (the
project now runs a single Flask app; see
`dashboard/web_dashboard/__init__.py`). It was pulled out into this
standalone module before that removal specifically so the Flask app
never had to import Streamlit just to reuse two constants and two CSV
loaders — that's no longer a concern now that Streamlit is gone
entirely, but this module remains the single, correct home for this
logic rather than duplicating it back into `blueprints/api.py`.

Caching
-------
`load_episode_results` / `load_summary_results` are cached in-process,
keyed by path and invalidated automatically when the underlying file's
modification time changes — so re-running the evaluation pipeline and
refreshing the dashboard picks up new results without a process
restart.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent

EVALUATION_RESULTS_PATH = REPO_ROOT / "evaluation" / "evaluation_results.csv"
EVALUATION_SUMMARY_PATH = REPO_ROOT / "evaluation" / "policy_evaluation_summary.csv"

REQUIRED_EPISODE_COLUMNS = {
    "policy", "seed", "total_reward", "final_revenue", "units_sold",
    "initial_inventory", "sell_through_pct", "spoilage_pct", "mean_price",
    "mean_discount_depth_pct", "steps",
}
REQUIRED_SUMMARY_COLUMNS = {
    "policy", "num_episodes", "mean_revenue", "std_revenue",
    "mean_sell_through_pct", "mean_spoilage_pct", "mean_price",
    "mean_discount_depth_pct", "revenue_uplift_pct",
    "meets_revenue_uplift_target", "meets_sell_through_target",
    "meets_spoilage_target",
}

POLICY_DISPLAY_NAMES: Dict[str, str] = {
    "dqn": "DQN Agent",
    "q_learning": "Q-Learning Agent",
    "random": "Random Policy",
    "fixed_price": "Fixed Price Policy",
    "time_based_discount": "Time-Based Discount Policy",
}


def display_name(policy: str) -> str:
    """Human-readable label for a policy key, falling back to the key itself."""
    return POLICY_DISPLAY_NAMES.get(policy, policy)


class DataCompatibilityError(Exception):
    """Raised when an evaluation output file exists but doesn't match the
    schema callers expect — e.g. produced by an older/incompatible
    version of `evaluation/evaluate_policies.py`. Distinguished from a
    plain `FileNotFoundError` so callers can give a different message for
    "wrong shape of data" versus "no data yet"."""


_cache_lock = threading.Lock()
# path -> (mtime at load time, loaded DataFrame)
_csv_cache: Dict[Path, Tuple[float, pd.DataFrame]] = {}


def _load_validated_csv(path: Path, required_columns: set) -> pd.DataFrame:
    """Read `path` as CSV, validate its columns, and cache the result
    in-process until the file's mtime changes (so this is safe to call
    on every request without re-reading/re-parsing the file each time)."""
    if not path.exists():
        raise FileNotFoundError(str(path))

    mtime = path.stat().st_mtime
    with _cache_lock:
        cached = _csv_cache.get(path)
        if cached is not None and cached[0] == mtime:
            return cached[1]

    df = pd.read_csv(path)
    missing = required_columns - set(df.columns)
    if missing:
        raise DataCompatibilityError(
            f"{path} is missing expected column(s) {sorted(missing)}. "
            "This usually means it was produced by an older version of "
            "evaluation/evaluate_policies.py — re-run the evaluation "
            "pipeline to regenerate it."
        )

    with _cache_lock:
        _csv_cache[path] = (mtime, df)
    return df


def load_episode_results(path: Path = EVALUATION_RESULTS_PATH) -> pd.DataFrame:
    """Load the episode-level results CSV (cached; see module docstring)."""
    return _load_validated_csv(path, REQUIRED_EPISODE_COLUMNS)


def load_summary_results(path: Path = EVALUATION_SUMMARY_PATH) -> pd.DataFrame:
    """Load the per-policy aggregate summary CSV (cached; see module docstring)."""
    return _load_validated_csv(path, REQUIRED_SUMMARY_COLUMNS)
