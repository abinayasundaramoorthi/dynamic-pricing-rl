"""
baseline_fixed.py

Static single-price policy: offer the environment's `base_price` for the
entire selling season and never adjust it, regardless of remaining
inventory or days left.

This is the classic "legacy pricing system" baseline referenced throughout
the project design doc (reports/project_planning/problem_statement.md,
Section 1: "Traditional fixed and rule-based pricing systems cannot adapt
fast enough..."). Concretely, it means always selecting the action whose
`price_adjustment_pct` entry is `0.0` (the "Hold price" action —
see `pricing_env/action_space.py:describe_action`), every single step.

Because `action_to_price()` applies adjustments relative to the *current*
price (not always the original base price — see that function's
docstring), always choosing the zero-adjustment action is what keeps the
effective price pinned at `base_price` for the whole episode; picking any
other action even once would compound away from it.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from gymnasium import spaces


class FixedPricePolicy:
    """
    Always selects the "hold price" action, keeping price pinned at
    `base_price` for the full episode.

    Parameters
    ----------
    action_space : spaces.Discrete
        The environment's action space (used only to validate the
        resolved hold-action index is in range).
    price_adjustment_pct : Sequence[float]
        The environment's configured price adjustments
        (`PricingEnvConfig.price_adjustment_pct`). Must contain exactly
        one `0.0` entry — the "no adjustment" action — or construction
        fails loudly rather than silently picking an arbitrary action.
    """

    name = "fixed_price"

    def __init__(
        self,
        action_space: spaces.Discrete,
        price_adjustment_pct: Sequence[float],
    ) -> None:
        self.action_space = action_space
        hold_indices = [
            i for i, pct in enumerate(price_adjustment_pct) if pct == 0.0
        ]
        if len(hold_indices) != 1:
            raise ValueError(
                "FixedPricePolicy requires price_adjustment_pct to contain "
                f"exactly one 0.0 ('hold price') entry, found {len(hold_indices)} "
                f"in {list(price_adjustment_pct)!r}."
            )
        self._hold_action = hold_indices[0]
        if not action_space.contains(self._hold_action):
            raise ValueError(
                f"Resolved hold action index {self._hold_action} is not a "
                f"valid member of action_space={action_space}."
            )

    def select_greedy_action(self, observation: np.ndarray) -> int:
        """Ignore `observation`; always hold at base_price."""
        return self._hold_action

    def reset(self) -> None:
        """No internal episode state to reset (stateless policy)."""
        return None