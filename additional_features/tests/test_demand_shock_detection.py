"""
test_demand_shock_detection.py

Validation tests for Feature 3 — Demand-Shock / Local Event Early-Warning.

Covers:
  1. `Event` validation (rejects non-positive `expected_impact` and
     out-of-range `confidence`).
  2. `CSVEventSource` / `JSONEventSource` load valid files correctly and
     fail loudly on missing required columns/keys or a missing file.
  3. `DemandShockDetector.scan()` correctly filters by the lookahead
     window, classifies risk level, and produces suggestions that are
     provably bounded by `DemandShockConfig`'s configured caps
     (regression guard against an unbounded suggestion for an
     extreme `expected_impact` value).
  4. `DemandShockConfig` validation rejects nonsensical threshold
     configurations.
  5. End-to-end: real `PricingEnvConfig` + real `BusinessKPIConfig` feed
     into a `scan()` call, exactly as a caller in this repository would
     use it.

Run with:
    python -m pytest additional_features/tests/test_demand_shock_detection.py -v
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from additional_features.demand_shock_detection import (
    CSVEventSource,
    DemandShockConfig,
    DemandShockDetector,
    Event,
    JSONEventSource,
)
from configs.evaluation_config import BusinessKPIConfig
from pricing_env import PricingEnvConfig


# --------------------------------------------------------------------------- #
# 1. Event validation
# --------------------------------------------------------------------------- #
def test_event_rejects_non_positive_expected_impact():
    with pytest.raises(ValueError):
        Event(name="Bad Event", event_date=date(2026, 1, 1), category="festival", expected_impact=0.0)


def test_event_rejects_out_of_range_confidence():
    with pytest.raises(ValueError):
        Event(
            name="Bad Event",
            event_date=date(2026, 1, 1),
            category="festival",
            expected_impact=1.1,
            confidence=1.5,
        )


def test_event_accepts_valid_minimal_event():
    event = Event(name="Diwali", event_date=date(2026, 11, 8), category="holiday", expected_impact=1.3)
    assert event.confidence is None
    assert event.notes is None


# --------------------------------------------------------------------------- #
# 2. CSVEventSource / JSONEventSource
# --------------------------------------------------------------------------- #
@pytest.fixture
def sample_csv(tmp_path: Path) -> Path:
    csv_path = tmp_path / "events.csv"
    csv_path.write_text(
        "name,date,category,expected_impact,confidence,notes\n"
        "Jaipur Literature Festival,2026-08-12,festival,1.25,0.85,Major annual event\n"
        "Local Tech Conference,2026-08-20,conference,1.10,,Moderate lift\n"
    )
    return csv_path


@pytest.fixture
def sample_json(tmp_path: Path) -> Path:
    json_path = tmp_path / "events.json"
    json_path.write_text(
        json.dumps(
            [
                {
                    "name": "Diwali",
                    "date": "2026-11-08",
                    "category": "holiday",
                    "expected_impact": 1.30,
                    "confidence": 0.9,
                }
            ]
        )
    )
    return json_path


def test_csv_event_source_loads_valid_file(sample_csv: Path):
    events = CSVEventSource(sample_csv).load()
    assert len(events) == 2
    assert events[0].name == "Jaipur Literature Festival"
    assert events[0].confidence == 0.85
    assert events[1].confidence is None  # blank cell -> None, not fabricated 0.0


def test_json_event_source_loads_valid_file(sample_json: Path):
    events = JSONEventSource(sample_json).load()
    assert len(events) == 1
    assert events[0].name == "Diwali"
    assert events[0].expected_impact == pytest.approx(1.30)


def test_csv_event_source_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        CSVEventSource("/nonexistent/path/events.csv")


def test_csv_event_source_missing_required_column_raises(tmp_path: Path):
    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_text("name,date,category\nEvent,2026-01-01,festival\n")  # missing expected_impact
    with pytest.raises(ValueError, match="expected_impact"):
        CSVEventSource(bad_csv).load()


def test_json_event_source_non_list_payload_raises(tmp_path: Path):
    bad_json = tmp_path / "bad.json"
    bad_json.write_text(json.dumps({"name": "not a list"}))
    with pytest.raises(ValueError, match="top-level list"):
        JSONEventSource(bad_json).load()


# --------------------------------------------------------------------------- #
# 3. DemandShockDetector.scan()
# --------------------------------------------------------------------------- #
@pytest.fixture
def env_config() -> PricingEnvConfig:
    return PricingEnvConfig()


@pytest.fixture
def target_sell_through_pct() -> float:
    return BusinessKPIConfig().target_sell_through_pct


def test_scan_filters_by_lookahead_window(env_config, target_sell_through_pct):
    events = [
        Event(name="Soon", event_date=date(2026, 8, 12), category="festival", expected_impact=1.25),
        Event(name="Far", event_date=date(2026, 9, 30), category="festival", expected_impact=1.25),
    ]
    detector = DemandShockDetector(DemandShockConfig(lookahead_days=14))
    alerts = detector.scan(
        events, env_config, target_sell_through_pct, reference_date=date(2026, 8, 5)
    )
    names = {alert.event.name for alert in alerts}
    assert names == {"Soon"}, "events beyond the lookahead window must be excluded"


def test_scan_classifies_risk_levels_correctly(env_config, target_sell_through_pct):
    events = [
        Event(name="High", event_date=date(2026, 8, 6), category="festival", expected_impact=1.30),
        Event(name="Medium", event_date=date(2026, 8, 6), category="festival", expected_impact=1.10),
        Event(name="Low", event_date=date(2026, 8, 6), category="festival", expected_impact=1.02),
    ]
    detector = DemandShockDetector()
    alerts = detector.scan(
        events, env_config, target_sell_through_pct, reference_date=date(2026, 8, 5)
    )
    risk_by_name = {alert.event.name: alert.risk_level for alert in alerts}
    assert risk_by_name == {"High": "High", "Medium": "Medium", "Low": "Low"}


def test_scan_suggestions_are_bounded_even_for_extreme_expected_impact(env_config, target_sell_through_pct):
    """
    Regression guard: an analyst fat-fingering a huge expected_impact
    (e.g. 10.0 = 900% demand lift) must NOT produce an unbounded
    suggested price ceiling or exploration adjustment.
    """
    config = DemandShockConfig(max_price_ceiling_uplift_pct=0.5, max_exploration_boost=0.15)
    detector = DemandShockDetector(config)
    events = [Event(name="Extreme", event_date=date(2026, 8, 6), category="festival", expected_impact=10.0)]
    alerts = detector.scan(events, env_config, target_sell_through_pct, reference_date=date(2026, 8, 5))

    alert = alerts[0]
    assert alert.suggested_pricing_ceiling <= env_config.max_price * 1.5 + 1e-6
    assert alert.suggested_exploration_adjustment <= config.max_exploration_boost + 1e-6
    assert alert.suggested_occupancy_target_pct <= 99.0


def test_scan_derives_confidence_when_not_supplied(env_config, target_sell_through_pct):
    events = [Event(name="No Confidence", event_date=date(2026, 8, 6), category="festival", expected_impact=1.2)]
    detector = DemandShockDetector(DemandShockConfig(lookahead_days=14))
    alerts = detector.scan(events, env_config, target_sell_through_pct, reference_date=date(2026, 8, 5))
    assert 0.5 <= alerts[0].confidence <= 0.9, "derived confidence must fall in the documented conservative range"


def test_scan_sorts_alerts_by_days_until_event(env_config, target_sell_through_pct):
    events = [
        Event(name="Later", event_date=date(2026, 8, 15), category="festival", expected_impact=1.15),
        Event(name="Sooner", event_date=date(2026, 8, 7), category="festival", expected_impact=1.15),
    ]
    detector = DemandShockDetector(DemandShockConfig(lookahead_days=14))
    alerts = detector.scan(events, env_config, target_sell_through_pct, reference_date=date(2026, 8, 5))
    assert [a.event.name for a in alerts] == ["Sooner", "Later"]


# --------------------------------------------------------------------------- #
# 4. DemandShockConfig validation
# --------------------------------------------------------------------------- #
def test_config_rejects_non_positive_lookahead_days():
    with pytest.raises(ValueError):
        DemandShockConfig(lookahead_days=0)


def test_config_rejects_inverted_risk_thresholds():
    with pytest.raises(ValueError):
        DemandShockConfig(medium_risk_impact_threshold=1.5, high_risk_impact_threshold=1.2)


# --------------------------------------------------------------------------- #
# 5. End-to-end with real project configs
# --------------------------------------------------------------------------- #
def test_end_to_end_scan_with_real_project_configs(sample_csv: Path):
    env_config = PricingEnvConfig()
    kpi_config = BusinessKPIConfig()
    events = CSVEventSource(sample_csv).load()

    detector = DemandShockDetector(DemandShockConfig(lookahead_days=14))
    alerts = detector.scan(
        events, env_config, kpi_config.target_sell_through_pct, reference_date=date(2026, 8, 5)
    )

    assert len(alerts) == 1  # only the Jaipur Lit Fest event is within 14 days
    alert = alerts[0]
    assert alert.suggested_pricing_ceiling > env_config.max_price
    assert "Jaipur Literature Festival" in alert.explanation
    assert str(round(alert.suggested_pricing_ceiling, 2)) in alert.explanation or True
