"""
event_schema.py

Defines the event data model and pluggable event-source loaders for
Feature 3 (Demand-Shock / Local Event Early-Warning).

Design intent
-------------
`EventSource` is an abstract base class with a single method, `load()`.
`CSVEventSource` and `JSONEventSource` are the two concrete
implementations required now. Adding a future API-backed source (e.g.
`RestApiEventSource`) means writing one new class that implements
`load() -> List[Event]` — nothing in `shock_detector.py` or anywhere
else in the project needs to change, because `DemandShockDetector`
(shock_detector.py) only ever depends on the `EventSource` interface,
never on a concrete loader.

This module has no dependency on Gymnasium, PyTorch, or Streamlit — it
is a pure data-loading layer, independently testable, matching the
existing project's convention of keeping I/O and domain logic in
separate, narrowly-scoped files (see `pricing_env/demand_simulator.py`'s
module docstring for the same design rationale).
"""

from __future__ import annotations

import csv
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import List, Optional, Union

# Required columns/keys every event source must supply. Kept as a module
# constant (rather than inlined per-loader) so CSV and JSON loaders stay
# in lockstep and any future loader validates against the same contract.
REQUIRED_EVENT_FIELDS = ("name", "date", "category", "expected_impact")


@dataclass(frozen=True)
class Event:
    """
    One known local demand-shock event, as supplied by the caller's own
    event calendar (CSV/JSON file) — never fabricated by this module.

    Attributes
    ----------
    name : str
        Event name, e.g. "Jaipur Literature Festival".
    event_date : date
        Calendar date the event occurs / peaks.
    category : str
        Free-text event category (e.g. "festival", "conference",
        "holiday", "concert", "sporting_event", "weather",
        "school_holiday", "local_event"). Not restricted to a fixed enum
        — the project owner's event calendar defines its own taxonomy;
        this module does not hardcode or assume a closed category list.
    expected_impact : float
        Analyst-supplied expected demand multiplier for this event, e.g.
        `1.25` for an expected 25% demand lift (matching the project's
        own Jaipur Lit Fest example: "25% rate lift, near-full
        occupancy"). This number comes from the caller's market
        intelligence — this module never invents or estimates it.
    confidence : Optional[float]
        Analyst-supplied confidence in `expected_impact`, in [0, 1]. If
        omitted, `shock_detector.py` derives a conservative default
        rather than fabricating a specific number.
    notes : Optional[str]
        Free-text notes carried through to the alert's explanation.
    """

    name: str
    event_date: date
    category: str
    expected_impact: float
    confidence: Optional[float] = None
    notes: Optional[str] = None

    def __post_init__(self) -> None:
        if self.expected_impact <= 0:
            raise ValueError(
                f"expected_impact must be > 0 (a multiplier on baseline demand), "
                f"got {self.expected_impact} for event {self.name!r}"
            )
        if self.confidence is not None and not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"confidence must be in [0, 1], got {self.confidence} for event {self.name!r}"
            )


class EventSource(ABC):
    """
    Abstract interface every event source must implement.

    Adding a new source type (e.g. a REST API feed) later requires only
    a new subclass of this ABC implementing `load()` — no changes to
    `DemandShockDetector` or any caller of `DemandShockDetector`.
    """

    @abstractmethod
    def load(self) -> List[Event]:
        """Return every event this source currently knows about."""
        raise NotImplementedError


def _parse_date(value: str) -> date:
    """Parse an ISO-8601 date string (YYYY-MM-DD); fails loudly on bad input."""
    return datetime.strptime(value.strip(), "%Y-%m-%d").date()


def _validate_row(row: dict, source_description: str) -> None:
    missing = [field for field in REQUIRED_EVENT_FIELDS if field not in row or row[field] in (None, "")]
    if missing:
        raise ValueError(
            f"{source_description} is missing required field(s) {missing} "
            f"in row {row!r}. Required fields: {list(REQUIRED_EVENT_FIELDS)}."
        )


class CSVEventSource(EventSource):
    """
    Loads events from a CSV file with columns:
    `name, date, category, expected_impact[, confidence, notes]`

    `date` must be ISO-8601 (`YYYY-MM-DD`). Extra columns are ignored;
    missing required columns raise `ValueError` immediately (fail loud,
    matching this project's existing config-validation style — see
    `pricing_env/pricing_env.py::PricingEnvConfig.__post_init__`).
    """

    def __init__(self, filepath: Union[str, Path]) -> None:
        self.filepath = Path(filepath)
        if not self.filepath.exists():
            raise FileNotFoundError(f"CSV event source not found: {self.filepath}")

    def load(self) -> List[Event]:
        events: List[Event] = []
        with open(self.filepath, "r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                _validate_row(row, f"CSV event source '{self.filepath}'")
                events.append(
                    Event(
                        name=row["name"].strip(),
                        event_date=_parse_date(row["date"]),
                        category=row["category"].strip(),
                        expected_impact=float(row["expected_impact"]),
                        confidence=float(row["confidence"]) if row.get("confidence") else None,
                        notes=row.get("notes") or None,
                    )
                )
        return events


class JSONEventSource(EventSource):
    """
    Loads events from a JSON file containing a list of objects:

        [
          {"name": "...", "date": "YYYY-MM-DD", "category": "...",
           "expected_impact": 1.25, "confidence": 0.8, "notes": "..."},
          ...
        ]

    Same validation contract as `CSVEventSource` — required fields are
    checked explicitly and a malformed file fails loudly rather than
    silently producing an incomplete event list.
    """

    def __init__(self, filepath: Union[str, Path]) -> None:
        self.filepath = Path(filepath)
        if not self.filepath.exists():
            raise FileNotFoundError(f"JSON event source not found: {self.filepath}")

    def load(self) -> List[Event]:
        with open(self.filepath, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, list):
            raise ValueError(
                f"JSON event source '{self.filepath}' must contain a top-level list "
                f"of event objects, got {type(payload).__name__}"
            )
        events: List[Event] = []
        for row in payload:
            _validate_row(row, f"JSON event source '{self.filepath}'")
            events.append(
                Event(
                    name=str(row["name"]).strip(),
                    event_date=_parse_date(str(row["date"])),
                    category=str(row["category"]).strip(),
                    expected_impact=float(row["expected_impact"]),
                    confidence=float(row["confidence"]) if row.get("confidence") is not None else None,
                    notes=row.get("notes") or None,
                )
            )
        return events
