"""
demand_simulator.py

Stochastic demand model for the pricing environment.

Models each day as a two-stage random process, standard in the revenue
management literature (see Talluri & van Ryzin, "The Theory and Practice of
Revenue Management"):

  1. Customer arrivals: the number of potential customers who consider
     buying today is Poisson-distributed, with a mean that grows as the
     deadline approaches (captures the well-documented late-booking demand
     surge in airline/hotel booking curves).
  2. Purchase decision: each arriving customer independently accepts the
     offered price with a probability given by a logistic function of
     price relative to the reference (base) price — the standard
     "logit demand" curve used in dynamic pricing research. Higher prices
     monotonically reduce acceptance probability; the curve is centered so
     that price == base_price yields roughly a 50% acceptance rate.

Units actually sold = Binomial(arrivals, acceptance_probability), capped at
remaining inventory (you cannot sell more than you have).

This module has NO dependency on Gymnasium or the environment class — it is
a pure, independently testable simulation component, which is exactly why it
was scoped as its own file: it can be unit-tested and calibrated in
isolation from the RL machinery around it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DemandConfig:
    """
    Configuration for the stochastic demand model.

    Attributes
    ----------
    base_daily_arrival_rate : float
        Expected number of potential customers per day at baseline
        (no urgency multiplier applied). Must be > 0.
    price_elasticity : float
        Steepness of the logistic acceptance-probability curve. Higher
        values mean customers are more price-sensitive (acceptance drops
        off more sharply as price rises above base_price). Must be > 0.
    min_acceptance_prob : float
        Floor on acceptance probability, even at very high prices. Real
        markets are never perfectly price-elastic — some customers will
        always buy regardless of price (e.g. urgent business travel).
    max_acceptance_prob : float
        Ceiling on acceptance probability, even at very low prices —
        avoids a degenerate "free lunch" case where the model predicts
        certain sale for arbitrarily low price.
    urgency_growth : float
        Maximum extra fractional boost to expected arrivals as the
        deadline approaches (applied linearly over the selling horizon).
        E.g. 0.8 means arrivals can be up to 80% higher on the final day
        than on day one, modeling the last-minute booking surge.
    """

    base_daily_arrival_rate: float = 5.0
    price_elasticity: float = 3.0
    min_acceptance_prob: float = 0.02
    max_acceptance_prob: float = 0.98
    urgency_growth: float = 0.8

    def __post_init__(self) -> None:
        if self.base_daily_arrival_rate <= 0:
            raise ValueError(
                f"base_daily_arrival_rate must be > 0, got {self.base_daily_arrival_rate}"
            )
        if self.price_elasticity <= 0:
            raise ValueError(
                f"price_elasticity must be > 0, got {self.price_elasticity}"
            )
        if not (0.0 <= self.min_acceptance_prob < self.max_acceptance_prob <= 1.0):
            raise ValueError(
                "Require 0 <= min_acceptance_prob < max_acceptance_prob <= 1, got "
                f"min={self.min_acceptance_prob}, max={self.max_acceptance_prob}"
            )
        if self.urgency_growth < 0:
            raise ValueError(f"urgency_growth must be >= 0, got {self.urgency_growth}")


@dataclass(frozen=True)
class SaleOutcome:
    """
    Result of simulating one day of demand.

    Attributes
    ----------
    units_sold : int
        Actual units sold today, already capped at remaining inventory.
    arrivals : int
        Simulated number of potential customers who considered buying.
    acceptance_probability : float
        The price-driven probability each arriving customer purchased.
    expected_demand : float
        arrivals_mean * acceptance_probability — a deterministic demand
        intensity signal (not the realized random units_sold), used for
        the qualitative demand_level label and for info/logging.
    demand_level : str
        One of {"Low", "Medium", "High"} — a human-readable summary of
        expected_demand relative to the configured baseline arrival rate.
    """

    units_sold: int
    arrivals: int
    acceptance_probability: float
    expected_demand: float
    demand_level: str


class DemandSimulator:
    """
    Stateless (aside from config) stochastic demand generator.

    Deliberately does not hold its own RNG — the caller (PricingEnvironment)
    passes in `self.np_random` on every call. This is essential for
    reproducibility: Gymnasium's `reset(seed=...)` reseeds `self.np_random`,
    and routing all randomness through that single generator is what makes
    an entire episode (and the Week 4 "1,000 simulated booking seasons"
    evaluation) exactly reproducible from a seed. If this class held its own
    independent `np.random.default_rng()`, episodes would not be
    reproducible via the environment's seed alone.
    """

    def __init__(self, config: DemandConfig) -> None:
        self.config = config

    def sample(
        self,
        price: float,
        reference_price: float,
        remaining_inventory: int,
        days_remaining: int,
        selling_horizon_days: int,
        rng: np.random.Generator,
    ) -> SaleOutcome:
        """
        Simulate one day of demand and return the resulting sale outcome.

        Parameters
        ----------
        price : float
            The price in effect today (post action-to-price conversion).
        reference_price : float
            The base/reference price the acceptance curve is centered on
            (typically `config.base_price` from PricingEnvConfig).
        remaining_inventory : int
            Units available to sell today; caps units_sold.
        days_remaining : int
            Days left until the deadline, used to compute urgency.
        selling_horizon_days : int
            Total length of the selling season, used to normalize urgency.
        rng : np.random.Generator
            The environment's seeded random generator (`self.np_random`).
            Passing this explicitly (rather than using a module-level RNG)
            is what makes results reproducible per-episode.

        Returns
        -------
        SaleOutcome
        """
        if remaining_inventory <= 0:
            # No inventory left to sell — short-circuit rather than sampling
            # a meaningless arrival process. Keeps behaviour well-defined
            # even if step() is ever called on a sold-out state.
            return SaleOutcome(
                units_sold=0,
                arrivals=0,
                acceptance_probability=0.0,
                expected_demand=0.0,
                demand_level="Low",
            )

        urgency_multiplier = self._time_urgency_multiplier(
            days_remaining, selling_horizon_days
        )
        expected_arrivals = self.config.base_daily_arrival_rate * urgency_multiplier
        arrivals = int(rng.poisson(expected_arrivals))

        acceptance_probability = self._acceptance_probability(price, reference_price)

        raw_units_sold = (
            int(rng.binomial(arrivals, acceptance_probability)) if arrivals > 0 else 0
        )
        units_sold = min(raw_units_sold, remaining_inventory)

        expected_demand = expected_arrivals * acceptance_probability
        demand_level = self._demand_level_label(expected_demand)

        return SaleOutcome(
            units_sold=units_sold,
            arrivals=arrivals,
            acceptance_probability=acceptance_probability,
            expected_demand=expected_demand,
            demand_level=demand_level,
        )

    def _time_urgency_multiplier(
        self, days_remaining: int, selling_horizon_days: int
    ) -> float:
        """
        Compute the arrival-rate multiplier reflecting late-booking urgency.

        Linearly scales from 1.0 (start of season, days_remaining ==
        selling_horizon_days) up to (1.0 + urgency_growth) as
        days_remaining approaches 0. Linear was chosen over an exponential
        curve deliberately: it is the simplest model that captures the
        qualitative real-world pattern (more urgency near the deadline)
        without introducing extra hyperparameters to calibrate before we
        have real booking-curve data to fit against. Replacing this with a
        curve fitted to real data is an explicit Phase 2 enhancement.
        """
        if selling_horizon_days <= 0:
            return 1.0
        time_elapsed_frac = 1.0 - (days_remaining / selling_horizon_days)
        time_elapsed_frac = float(np.clip(time_elapsed_frac, 0.0, 1.0))
        return 1.0 + self.config.urgency_growth * time_elapsed_frac

    def _acceptance_probability(self, price: float, reference_price: float) -> float:
        """
        Logistic acceptance probability, centered on reference_price.

        p(price) = 1 / (1 + exp(elasticity * (price / reference_price - 1)))

        At price == reference_price, p ≈ 0.5. Above it, p decays smoothly
        toward min_acceptance_prob; below it, p rises smoothly toward
        max_acceptance_prob. The logistic form is standard in the dynamic
        pricing literature because it is smooth and monotonic everywhere,
        which avoids the reward discontinuities a piecewise/step demand
        curve would introduce during agent training.
        """
        if reference_price <= 0:
            raise ValueError(f"reference_price must be > 0, got {reference_price}")
        ratio = price / reference_price
        logistic = 1.0 / (1.0 + np.exp(self.config.price_elasticity * (ratio - 1.0)))
        return float(
            np.clip(
                logistic,
                self.config.min_acceptance_prob,
                self.config.max_acceptance_prob,
            )
        )

    def _demand_level_label(self, expected_demand: float) -> str:
        """
        Bucket expected demand into a qualitative Low/Medium/High label,
        relative to the configured baseline arrival rate. Purely for
        interpretability (info dict, rendering) — never used in reward
        or training logic, so changing these thresholds cannot silently
        alter agent behaviour.
        """
        fraction_of_baseline = expected_demand / self.config.base_daily_arrival_rate
        if fraction_of_baseline < 0.4:
            return "Low"
        if fraction_of_baseline < 0.8:
            return "Medium"
        return "High"