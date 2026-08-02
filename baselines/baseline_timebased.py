"""
baseline_timebased.py

Rule-based, time-and-inventory-aware discount policy — the "revenue
manager's heuristic" baseline referenced in the design doc (Section 21,
M2: "Implement heuristic time/inventory-based discount rules"). This is
the strongest non-learned baseline in the comparison set: unlike
`RandomPolicy` and `FixedPricePolicy`, it *does* react to state, using two
simple, interpretable signals that mirror how a human revenue manager
would reason about the problem without any ML:

  1. Time urgency — discount more aggressively as the deadline approaches
     (`time_elapsed_fraction`), independent of how sales are pacing.
  2. Pacing pressure — discount more aggressively when sell-through is
     lagging a naive linear pacing curve (`pacing_gap` = the same
     "actual sold vs. ideal-by-now" gap `pricing_env/reward.py`'s
     `balance_bonus` term measures), and *not* when ahead of pace.

The two signals combine into a single target discount depth, which is
then snapped to the nearest actually-available action in
`price_adjustment_pct` — a real deployment cannot offer a price
adjustment that isn't in its allowed set, and neither can this baseline.

Because this rule reacts every step to (remaining_inventory,
days_remaining) rather than compounding a fixed daily percentage off
whatever the current price happens to be, it does not carry the classic
"10%-off-every-day" failure mode of drifting to $0 over a long horizon —
the target is always freshly computed relative to `base_price`.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from gymnasium import spaces


class TimeBasedDiscountPolicy:
    """
    Discounts more aggressively as the deadline nears and/or as sell-through
    falls behind a naive linear pacing curve.

    Parameters
    ----------
    action_space : spaces.Discrete
        The environment's action space (used to validate the resolved
        action index every step).
    price_adjustment_pct : Sequence[float]
        The environment's configured price adjustments. The policy always
        selects whichever entry is numerically closest to its computed
        target discount for the current state.
    initial_inventory : int
        Starting inventory for the episode — needed to convert the raw
        `remaining_inventory` observation into a sold fraction.
    selling_horizon_days : int
        Total selling season length — needed to convert the raw
        `days_remaining` observation into a time-elapsed fraction.
    max_discount : float
        Upper bound (as a positive fraction, e.g. 0.5 = 50%) on the
        target discount this policy will ever aim for, applied before
        snapping to the nearest available action. Keeps the heuristic
        from chasing a discount deeper than any real action offers, and
        keeps its behavior comparable in spirit to a real-world "we never
        discount more than X%" business rule. Default 0.5.
    time_urgency_weight : float
        Weight on the time-based urgency term (grows as the deadline
        nears, independent of pacing). Default 0.15 — a mild, steady
        discount drift toward the deadline even for a perfectly-paced
        episode.
    pacing_weight : float
        Weight on the pacing-shortfall term (fires only when behind
        pace). Default 1.5 — the dominant driver of this policy's
        discount depth, since falling behind pace is the more urgent,
        directly actionable signal for a revenue manager.
    """

    name = "time_based_discount"

    def __init__(
        self,
        action_space: spaces.Discrete,
        price_adjustment_pct: Sequence[float],
        initial_inventory: int,
        selling_horizon_days: int,
        max_discount: float = 0.5,
        time_urgency_weight: float = 0.15,
        pacing_weight: float = 1.5,
    ) -> None:
        if initial_inventory <= 0:
            raise ValueError(
                f"initial_inventory must be > 0, got {initial_inventory}"
            )
        if selling_horizon_days <= 0:
            raise ValueError(
                f"selling_horizon_days must be > 0, got {selling_horizon_days}"
            )
        if not (0.0 < max_discount <= 1.0):
            raise ValueError(f"max_discount must be in (0, 1], got {max_discount}")

        self.action_space = action_space
        self._price_adjustment_pct = np.asarray(price_adjustment_pct, dtype=np.float64)
        self.initial_inventory = int(initial_inventory)
        self.selling_horizon_days = int(selling_horizon_days)
        self.max_discount = max_discount
        self.time_urgency_weight = time_urgency_weight
        self.pacing_weight = pacing_weight

    def _target_discount(self, remaining_inventory: float, days_remaining: float) -> float:
        """Compute the desired discount depth in [0, max_discount] for the
        current state, before snapping to an available action."""
        time_elapsed_fraction = 1.0 - (days_remaining / self.selling_horizon_days)
        time_elapsed_fraction = min(max(time_elapsed_fraction, 0.0), 1.0)

        sold_fraction = 1.0 - (remaining_inventory / self.initial_inventory)
        sold_fraction = min(max(sold_fraction, 0.0), 1.0)

        # Positive when sales are lagging the naive linear pacing curve
        # (i.e. more time has elapsed than inventory sold) — the same
        # quantity `reward.py`'s balance_bonus penalizes deviation from.
        pacing_gap = max(0.0, time_elapsed_fraction - sold_fraction)

        # Urgency accelerates non-linearly toward the deadline, mirroring
        # how a revenue manager gets progressively more willing to cut
        # price the closer the unsellable date gets.
        time_urgency = time_elapsed_fraction**2

        target = (
            self.time_urgency_weight * time_urgency
            + self.pacing_weight * pacing_gap
        )
        return float(min(max(target, 0.0), self.max_discount))

    def select_greedy_action(self, observation: np.ndarray) -> int:
        remaining_inventory = float(observation[0])
        days_remaining = float(observation[1])

        target_discount = self._target_discount(remaining_inventory, days_remaining)
        # Discounts are negative adjustments in price_adjustment_pct, so
        # compare against -target_discount and snap to the closest
        # available action.
        action = int(
            np.argmin(np.abs(self._price_adjustment_pct - (-target_discount)))
        )
        if not self.action_space.contains(action):
            raise RuntimeError(
                f"Resolved action index {action} is not a valid member of "
                f"action_space={self.action_space}; this indicates a "
                "mismatch between price_adjustment_pct and the environment "
                "this policy was constructed against."
            )
        return action

    def reset(self) -> None:
        """No internal episode state to reset (state is derived fresh from
        each observation)."""
        return None