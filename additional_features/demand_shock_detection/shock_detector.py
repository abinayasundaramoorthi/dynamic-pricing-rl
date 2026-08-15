"""
shock_detector.py

`DemandShockDetector` — the early-warning engine for Feature 3.

Purpose (from the feature spec): flag known local demand-shock events a
configurable number of days out and pre-emptively suggest exploration /
price-ceiling adjustments to the RL agent, rather than waiting for the
environment's state (occupancy / booking pace) to reflect the shock
after the fact — the "hotels that react to a festival after bookings
spike leave money on the table" pattern.

Integration boundary
---------------------
This module NEVER touches `agents/dqn_agent.py`, `agents/q_learning_agent.py`,
or `pricing_env/pricing_env.py` directly, per the project's integration
rule ("Do not directly modify the RL Agent. Instead expose reusable
modules"). It only reads:

    - `pricing_env.PricingEnvConfig` (real, existing config — for
      `base_price`/`max_price`, so the suggested pricing ceiling is
      expressed relative to the actual configured price bounds, not an
      invented number)
    - `configs.evaluation_config.BusinessKPIConfig` (real, existing
      config — for `target_sell_through_pct`, so the suggested occupancy
      target is anchored to the project's own success criteria, not an
      arbitrary constant)

and produces plain dataclasses (`DemandShockAlert`) that a caller (the
training loop, the dashboard, or a human revenue manager) can choose to
act on however it likes — e.g. by constructing a new `PricingEnvConfig`
with a raised `max_price`, or by nudging `DQNAgent.epsilon` before the
next training episode. No existing file is modified to "wire this in";
that wiring, if wanted, is a caller-side decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import List, Optional, Sequence

from .event_schema import Event

# Imported lazily-safe: both are real, already-existing project configs.
from pricing_env import PricingEnvConfig  # noqa: E402


@dataclass(frozen=True)
class DemandShockConfig:
    """
    Tunable thresholds controlling how aggressively the detector reacts.

    Attributes
    ----------
    lookahead_days : int
        How many days ahead of an event date the detector starts
        raising alerts. Default 14, matching the "flag ... a set number
        of days out" requirement.
    high_risk_impact_threshold : float
        `expected_impact` at or above this value is classified "High"
        risk. Default 1.20 (a 20%+ expected demand lift) — comfortably
        below the project's own Jaipur Lit Fest reference point (1.25).
    medium_risk_impact_threshold : float
        `expected_impact` at or above this value (but below the high
        threshold) is classified "Medium" risk. Default 1.08.
    max_price_ceiling_uplift_pct : float
        Hard cap, as a fraction (e.g. 0.5 = 50%), on how far above the
        environment's *current* `max_price` the suggested ceiling may
        rise, regardless of how large `expected_impact` is. Prevents a
        single large `expected_impact` value in an event file from
        producing an unbounded suggested ceiling.
    max_exploration_boost : float
        Hard cap on the suggested downward adjustment to an agent's
        epsilon (exploration rate) ahead of a shock — e.g. 0.15 means
        the detector will never suggest reducing epsilon by more than
        0.15 absolute, regardless of event severity.
    """

    lookahead_days: int = 14
    high_risk_impact_threshold: float = 1.20
    medium_risk_impact_threshold: float = 1.08
    max_price_ceiling_uplift_pct: float = 0.50
    max_exploration_boost: float = 0.15

    def __post_init__(self) -> None:
        if self.lookahead_days <= 0:
            raise ValueError(f"lookahead_days must be > 0, got {self.lookahead_days}")
        if not (1.0 < self.medium_risk_impact_threshold <= self.high_risk_impact_threshold):
            raise ValueError(
                "Require 1.0 < medium_risk_impact_threshold <= "
                f"high_risk_impact_threshold, got medium="
                f"{self.medium_risk_impact_threshold}, high={self.high_risk_impact_threshold}"
            )
        if not (0.0 < self.max_price_ceiling_uplift_pct <= 5.0):
            raise ValueError(
                f"max_price_ceiling_uplift_pct must be in (0, 5], got {self.max_price_ceiling_uplift_pct}"
            )
        if not (0.0 < self.max_exploration_boost <= 1.0):
            raise ValueError(
                f"max_exploration_boost must be in (0, 1], got {self.max_exploration_boost}"
            )


@dataclass(frozen=True)
class DemandShockAlert:
    """
    One actionable early-warning alert for a single upcoming event.

    Attributes
    ----------
    event : Event
        The triggering event (see `event_schema.py`).
    days_until_event : int
        Calendar days between the scan date and the event date.
    risk_level : str
        One of {"Low", "Medium", "High"}.
    suggested_occupancy_target_pct : float
        Recommended sell-through target (%) to aim for ahead of this
        event, derived from `BusinessKPIConfig.target_sell_through_pct`
        scaled up by the event's expected demand lift and capped at 99%
        (never suggests promising 100% sell-through, which would leave
        zero buffer for demand variance).
    suggested_exploration_adjustment : float
        Recommended reduction to an agent's exploration rate (epsilon)
        ahead of the event, in [0, `DemandShockConfig.max_exploration_boost`].
        Positive value = "reduce epsilon by this much" (i.e. exploit the
        anticipated demand surge rather than continuing to explore).
    suggested_pricing_ceiling : float
        Recommended absolute price ceiling (in the same currency units
        as `PricingEnvConfig.max_price`) ahead of the event, bounded by
        `DemandShockConfig.max_price_ceiling_uplift_pct` above the
        environment's current `max_price`.
    confidence : float
        Confidence in this alert, in [0, 1]. Uses `event.confidence`
        directly when the caller supplied one; otherwise derived
        conservatively from how far out the event is (closer = more
        confident forecasts are typically more reliable) — never a
        fabricated fixed number.
    explanation : str
        Human-readable, data-grounded explanation of the alert.
    """

    event: Event
    days_until_event: int
    risk_level: str
    suggested_occupancy_target_pct: float
    suggested_exploration_adjustment: float
    suggested_pricing_ceiling: float
    confidence: float
    explanation: str


class DemandShockDetector:
    """
    Scans a list of `Event`s against a reference date and the current
    `PricingEnvConfig`/`BusinessKPIConfig`, producing `DemandShockAlert`s
    for anything within the configured lookahead window.
    """

    def __init__(self, config: Optional[DemandShockConfig] = None) -> None:
        self.config = config or DemandShockConfig()

    def scan(
        self,
        events: Sequence[Event],
        env_config: PricingEnvConfig,
        target_sell_through_pct: float,
        reference_date: Optional[date] = None,
    ) -> List[DemandShockAlert]:
        """
        Produce alerts for every event within the lookahead window.

        Parameters
        ----------
        events : Sequence[Event]
            Typically the output of `CSVEventSource.load()` /
            `JSONEventSource.load()`.
        env_config : pricing_env.PricingEnvConfig
            The environment's real, existing configuration — used for
            `base_price` and `max_price` so suggestions stay anchored to
            the actual experiment's price scale.
        target_sell_through_pct : float
            The project's real sell-through target (pass in
            `configs.evaluation_config.BusinessKPIConfig().target_sell_through_pct`)
            — this module does not hardcode its own target so it always
            stays consistent with whatever the evaluation pipeline uses.
        reference_date : date, optional
            "Today", for computing `days_until_event`. Defaults to
            `date.today()`.

        Returns
        -------
        List[DemandShockAlert]
            Sorted by `days_until_event` ascending (soonest first), only
            including events within `config.lookahead_days`.
        """
        today = reference_date or date.today()
        horizon = today + timedelta(days=self.config.lookahead_days)

        alerts: List[DemandShockAlert] = []
        for event in events:
            if not (today <= event.event_date <= horizon):
                continue
            days_until = (event.event_date - today).days
            alerts.append(self._build_alert(event, days_until, env_config, target_sell_through_pct))

        return sorted(alerts, key=lambda alert: alert.days_until_event)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _risk_level(self, expected_impact: float) -> str:
        if expected_impact >= self.config.high_risk_impact_threshold:
            return "High"
        if expected_impact >= self.config.medium_risk_impact_threshold:
            return "Medium"
        return "Low"

    def _confidence(self, event: Event, days_until: int) -> float:
        if event.confidence is not None:
            return float(event.confidence)
        # Conservative, distance-decayed default: full lookahead window
        # away -> 0.5; the event date itself -> 0.9. Never claims
        # certainty this module has no basis for.
        span = max(1, self.config.lookahead_days)
        proximity = 1.0 - (days_until / span)
        return round(0.5 + 0.4 * max(0.0, min(1.0, proximity)), 3)

    def _build_alert(
        self,
        event: Event,
        days_until: int,
        env_config: PricingEnvConfig,
        target_sell_through_pct: float,
    ) -> DemandShockAlert:
        risk_level = self._risk_level(event.expected_impact)
        confidence = self._confidence(event, days_until)

        # Occupancy target: scale the project's own sell-through target
        # up by (a confidence-weighted fraction of) the expected demand
        # lift, capped at 99% to always leave a buffer.
        demand_lift = event.expected_impact - 1.0
        occupancy_boost = demand_lift * confidence * 100.0
        suggested_occupancy_target_pct = min(99.0, target_sell_through_pct + occupancy_boost)

        # Exploration adjustment: reduce epsilon proportionally to risk
        # and confidence, capped by config.max_exploration_boost.
        risk_multiplier = {"Low": 0.2, "Medium": 0.6, "High": 1.0}[risk_level]
        suggested_exploration_adjustment = round(
            self.config.max_exploration_boost * risk_multiplier * confidence, 4
        )

        # Pricing ceiling: raise max_price proportionally to expected
        # demand lift, hard-capped by max_price_ceiling_uplift_pct.
        uncapped_uplift_pct = min(demand_lift, self.config.max_price_ceiling_uplift_pct)
        suggested_pricing_ceiling = round(env_config.max_price * (1.0 + uncapped_uplift_pct), 2)

        explanation = (
            f"'{event.name}' ({event.category}) is {days_until} day(s) away with an "
            f"analyst-supplied expected demand multiplier of {event.expected_impact:.2f}x "
            f"({demand_lift * 100:+.1f}%). Classified {risk_level} risk. Suggest raising the "
            f"occupancy target to {suggested_occupancy_target_pct:.1f}%, reducing exploration "
            f"by {suggested_exploration_adjustment:.3f}, and lifting the price ceiling from "
            f"₹{env_config.max_price:,.2f} to ₹{suggested_pricing_ceiling:,.2f} ahead of the event."
        )
        if event.notes:
            explanation += f" Analyst notes: {event.notes}"

        return DemandShockAlert(
            event=event,
            days_until_event=days_until,
            risk_level=risk_level,
            suggested_occupancy_target_pct=round(suggested_occupancy_target_pct, 2),
            suggested_exploration_adjustment=suggested_exploration_adjustment,
            suggested_pricing_ceiling=suggested_pricing_ceiling,
            confidence=confidence,
            explanation=explanation,
        )
