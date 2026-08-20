"""
additional_features package

Container package for three new, self-contained feature modules added on
top of the existing Dynamic Pricing RL project, without modifying any
existing file:

    additional_features.explainable_ai_pricing   — human/developer-readable
        explanations for why the RL agent recommended a given price,
        built only from data the project's environment and agents
        already produce (EnvState/info dict, Q-values, reward
        breakdown). No competitor pricing, special-event, or customer
        behaviour data is fabricated — this project does not track
        those signals, so the explainer simply omits them.

    additional_features.advanced_dashboard        — reusable Flask +
        Chart.js dashboard components (KPI cards, trend charts,
        baseline comparisons, recommendation timeline, CSV/JSON export)
        that read the exact same `evaluation/evaluation_results.csv`
        and `evaluation/policy_evaluation_summary.csv` contract that
        `dashboard/data_contract.py` defines. These are importable
        building blocks consumed by `dashboard/web_dashboard/` (the
        project's one Flask app).

    additional_features.demand_shock_detection     — an early-warning
        module that flags known local demand-shock events (festivals,
        conferences, holidays, ...) ahead of time from a configurable
        CSV/JSON event source, and produces advisory pricing/exploration
        adjustments. It does not call into the RL agent directly; it
        only produces reusable, importable recommendations.

Each sub-package matches the existing project's convention (see
`pricing_env/__init__.py`, `baselines/__init__.py`): a package-level
`__init__.py` re-exports the public API so callers can write

    from additional_features.explainable_ai_pricing import PricingExplainer
    from additional_features.advanced_dashboard import charts, kpi_components
    from additional_features.demand_shock_detection import DemandShockDetector

without reaching into submodules directly.
"""

from . import advanced_dashboard, demand_shock_detection, explainable_ai_pricing

__all__ = [
    "explainable_ai_pricing",
    "advanced_dashboard",
    "demand_shock_detection",
]
