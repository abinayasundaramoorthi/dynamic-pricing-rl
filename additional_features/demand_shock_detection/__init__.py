"""
demand_shock_detection package

Public API for Feature 3 — Demand-Shock / Local Event Early-Warning.

Import from the package root, matching the existing project's convention
(see `pricing_env/__init__.py`):

    from additional_features.demand_shock_detection import (
        DemandShockDetector, DemandShockConfig, DemandShockAlert,
        Event, EventSource, CSVEventSource, JSONEventSource,
    )
"""

from .event_schema import CSVEventSource, Event, EventSource, JSONEventSource
from .shock_detector import DemandShockAlert, DemandShockConfig, DemandShockDetector

__all__ = [
    "DemandShockDetector",
    "DemandShockConfig",
    "DemandShockAlert",
    "Event",
    "EventSource",
    "CSVEventSource",
    "JSONEventSource",
]
