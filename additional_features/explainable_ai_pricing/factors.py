"""
factors.py

Extracts interpretable pricing factors from the *real* data the existing
`pricing_env.PricingEnvironment` already produces on every `step()` call:

    observation : np.ndarray            -> [remaining_inventory, days_remaining]
                                            (pricing_env/state.py: EnvState.to_observation)
    info        : Dict[str, Any]        -> current_price, current_step,
                                            episode_revenue, last_units_sold,
                                            last_demand_level, initial_inventory,
                                            selling_horizon_days, action,
                                            action_label, price, units_sold,
                                            arrivals, acceptance_probability,
                                            demand_level, reward_breakdown
                                            (pricing_env/state.py: EnvState.to_info,
                                            pricing_env/pricing_env.py: step())

This module deliberately does NOT reference competitor pricing, special
events, or customer-behaviour segmentation: the project's environment,
state, reward, and demand modules do not compute or expose any of those
signals. Per the project's explainability requirement ("if a variable
does not exist in the repository, do not fabricate it — gracefully
ignore it"), this module only ever derives factors from fields that
provably exist in `EnvState`/`info` above.

Every function here is a pure function of its inputs — no Gymnasium,
Streamlit, or PyTorch dependency — matching the existing project's
convention of keeping domain logic independently unit-testable
(see `pricing_env/reward.py`, `pricing_env/demand_simulator.py`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class PriceFactor:
    """
    One interpretable factor that contributed to a pricing decision.

    Attributes
    ----------
    key : str
        Stable machine-readable identifier (e.g. "occupancy_pressure").
        Used by the JSON/developer explanation and by any caller that
        wants to key off a specific factor rather than parse prose.
    label : str
        Human-readable name (e.g. "Occupancy pressure").
    direction : str
        One of {"raises_price", "lowers_price", "neutral"} — the
        qualitative effect this factor had on the pricing decision.
    weight : float
        Non-negative importance score. Weights across all factors
        returned by `extract_pricing_factors` are normalized to sum to
        1.0, so `weight` can be read directly as "share of the
        explanation attributable to this factor".
    raw_value : float
        The underlying numeric value the factor was computed from
        (e.g. the actual sell-through percentage), for the developer
        explanation and for any downstream consumer that wants the
        number, not just the narrative.
    narrative : str
        One-sentence, data-grounded description of this specific factor
        (includes the actual numbers observed this step — never a
        generic template sentence).
    """

    key: str
    label: str
    direction: str
    weight: float
    raw_value: float
    narrative: str


def _time_elapsed_fraction(days_remaining: float, selling_horizon_days: float) -> float:
    """
    Fraction of the selling season elapsed so far.

    Mirrors the exact calculation in `pricing_env/reward.py::compute_reward`
    (`time_elapsed_fraction`) so the explainer's notion of "pacing" stays
    numerically consistent with what the agent is actually rewarded on.
    """
    if selling_horizon_days <= 0:
        return 1.0
    frac = 1.0 - (float(days_remaining) / float(selling_horizon_days))
    return float(np.clip(frac, 0.0, 1.0))


def _sold_fraction(remaining_inventory: float, initial_inventory: float) -> float:
    """
    Fraction of initial inventory already sold.

    Mirrors `pricing_env/reward.py::compute_reward`'s `sold_fraction`
    calculation exactly, for the same reason as `_time_elapsed_fraction`.
    """
    if initial_inventory <= 0:
        return 0.0
    frac = 1.0 - (float(remaining_inventory) / float(initial_inventory))
    return float(np.clip(frac, 0.0, 1.0))


def extract_pricing_factors(
    observation: Sequence[float],
    info: Dict[str, Any],
    previous_price: Optional[float] = None,
) -> List[PriceFactor]:
    """
    Derive a ranked list of `PriceFactor`s from one real environment step.

    Parameters
    ----------
    observation : Sequence[float]
        The 2-element observation returned by `PricingEnvironment.step()`
        / `.reset()`: `[remaining_inventory, days_remaining]`.
    info : Dict[str, Any]
        The `info` dict returned alongside `observation`. Must contain at
        least the keys `PricingEnvironment.step()` always sets:
        `initial_inventory`, `selling_horizon_days`, `price`, `units_sold`,
        `arrivals`, `acceptance_probability`, `demand_level`,
        `reward_breakdown` (a dict with `revenue`, `discount_penalty`,
        `unsold_penalty`, `balance_bonus`).
    previous_price : float, optional
        The price in effect *before* this step's action, used to describe
        the price change itself (e.g. "raised 8%"). If omitted, the
        price-change factor is skipped rather than guessed.

    Returns
    -------
    List[PriceFactor]
        Factors sorted by descending `weight` (most influential first).
        Weights are normalized to sum to 1.0 across the returned list.

    Raises
    ------
    KeyError
        If a required key is missing from `info` — this module never
        silently substitutes a fabricated default for data the
        environment is supposed to provide.
    """
    remaining_inventory = float(observation[0])
    days_remaining = float(observation[1])
    initial_inventory = float(info["initial_inventory"])
    selling_horizon_days = float(info["selling_horizon_days"])
    price = float(info["price"])
    units_sold = int(info["units_sold"])
    arrivals = int(info["arrivals"])
    acceptance_probability = float(info["acceptance_probability"])
    demand_level = str(info["demand_level"])
    reward_breakdown = info["reward_breakdown"]

    sold_fraction = _sold_fraction(remaining_inventory, initial_inventory)
    time_fraction = _time_elapsed_fraction(days_remaining, selling_horizon_days)
    pacing_gap = sold_fraction - time_fraction  # >0 = ahead of pace, <0 = behind

    raw_factors: List[PriceFactor] = []

    # --- 1. Occupancy / sell-through pressure --------------------------- #
    occupancy_pct = sold_fraction * 100.0
    occupancy_weight = abs(sold_fraction - 0.5) * 2.0  # 0 at 50% sold, 1 at 0%/100%
    raw_factors.append(
        PriceFactor(
            key="occupancy_pressure",
            label="Occupancy / sell-through",
            direction="raises_price" if sold_fraction > 0.5 else "lowers_price",
            weight=occupancy_weight,
            raw_value=occupancy_pct,
            narrative=(
                f"{occupancy_pct:.1f}% of initial inventory "
                f"({initial_inventory:.0f} units) is already sold, "
                f"leaving {remaining_inventory:.0f} units remaining."
            ),
        )
    )

    # --- 2. Booking pace vs. the deadline (pacing gap) ------------------ #
    # Same pacing_gap the reward function's balance_bonus penalizes —
    # ahead of pace supports holding/raising price, behind pace supports
    # discounting to avoid terminal unsold-inventory penalty.
    pace_weight = min(1.0, abs(pacing_gap) * 2.0)
    if pacing_gap > 0.01:
        pace_direction = "raises_price"
        pace_desc = (
            f"Sales are running {pacing_gap * 100:.1f} percentage points "
            f"AHEAD of the linear booking pace expected at this point in "
            f"the {selling_horizon_days:.0f}-day selling horizon."
        )
    elif pacing_gap < -0.01:
        pace_direction = "lowers_price"
        pace_desc = (
            f"Sales are running {abs(pacing_gap) * 100:.1f} percentage "
            f"points BEHIND the linear booking pace expected at this "
            f"point in the {selling_horizon_days:.0f}-day selling horizon."
        )
    else:
        pace_direction = "neutral"
        pace_desc = "Sales are almost exactly on the expected linear booking pace."
    raw_factors.append(
        PriceFactor(
            key="booking_pace",
            label="Booking pace vs. deadline",
            direction=pace_direction,
            weight=pace_weight,
            raw_value=pacing_gap * 100.0,
            narrative=pace_desc,
        )
    )

    # --- 3. Demand intensity (arrivals + acceptance probability) -------- #
    # acceptance_probability > 0.5 means the just-charged price cleared
    # the market more easily than the 50%-at-base-price reference point
    # (pricing_env/demand_simulator.py::_acceptance_probability).
    demand_weight = abs(acceptance_probability - 0.5) * 2.0
    raw_factors.append(
        PriceFactor(
            key="demand_intensity",
            label=f"Demand level ({demand_level})",
            direction="raises_price" if acceptance_probability > 0.5 else "lowers_price",
            weight=demand_weight,
            raw_value=acceptance_probability,
            narrative=(
                f"{arrivals} potential customers arrived this step with a "
                f"{acceptance_probability * 100:.1f}% price-acceptance "
                f"probability, resulting in {units_sold} unit(s) sold — "
                f"demand is classified as '{demand_level}'."
            ),
        )
    )

    # --- 4. Time urgency (deadline proximity) ---------------------------- #
    urgency_weight = time_fraction  # closer to deadline -> more urgency
    raw_factors.append(
        PriceFactor(
            key="time_urgency",
            label="Deadline proximity",
            direction="lowers_price" if time_fraction > 0.7 and sold_fraction < time_fraction else "neutral",
            weight=urgency_weight * 0.6,  # secondary factor: scaled down vs. direct pacing/occupancy signals
            raw_value=days_remaining,
            narrative=(
                f"{days_remaining:.0f} of {selling_horizon_days:.0f} selling "
                f"days remain ({time_fraction * 100:.1f}% of the horizon elapsed)."
            ),
        )
    )

    # --- 5. Reward composition (why the agent was rewarded this way) ---- #
    discount_penalty = float(reward_breakdown["discount_penalty"])
    unsold_penalty = float(reward_breakdown["unsold_penalty"])
    balance_bonus = float(reward_breakdown["balance_bonus"])
    revenue = float(reward_breakdown["revenue"])
    penalty_total = discount_penalty + unsold_penalty
    if penalty_total > 0 or balance_bonus > 0:
        reward_weight = min(1.0, penalty_total / max(revenue, 1.0) + balance_bonus / max(revenue, 1.0))
        parts = [f"₹{revenue:,.2f} revenue"]
        if discount_penalty > 0:
            parts.append(f"−₹{discount_penalty:,.2f} discount penalty")
        if unsold_penalty > 0:
            parts.append(f"−₹{unsold_penalty:,.2f} unsold-inventory penalty")
        if balance_bonus > 0:
            parts.append(f"+₹{balance_bonus:,.2f} pacing bonus")
        raw_factors.append(
            PriceFactor(
                key="reward_composition",
                label="Reward breakdown",
                direction="lowers_price" if discount_penalty > balance_bonus else "raises_price",
                weight=reward_weight,
                raw_value=penalty_total,
                narrative="This step's reward was composed of " + ", ".join(parts) + ".",
            )
        )

    # --- 6. Price change itself (if we know the previous price) --------- #
    if previous_price is not None and previous_price > 0:
        pct_change = (price - previous_price) / previous_price
        if abs(pct_change) > 1e-6:
            raw_factors.append(
                PriceFactor(
                    key="price_change",
                    label="Price change applied",
                    direction="raises_price" if pct_change > 0 else "lowers_price",
                    weight=min(1.0, abs(pct_change) * 3.0),
                    raw_value=pct_change * 100.0,
                    narrative=(
                        f"Price moved from ₹{previous_price:,.2f} to ₹{price:,.2f} "
                        f"({pct_change * 100:+.1f}%)."
                    ),
                )
            )

    # --- Normalize weights to sum to 1.0, rank descending ---------------- #
    total_weight = sum(f.weight for f in raw_factors)
    if total_weight <= 0:
        normalized = [PriceFactor(**{**f.__dict__, "weight": 1.0 / len(raw_factors)}) for f in raw_factors]
    else:
        normalized = [
            PriceFactor(**{**f.__dict__, "weight": f.weight / total_weight}) for f in raw_factors
        ]

    return sorted(normalized, key=lambda f: f.weight, reverse=True)
