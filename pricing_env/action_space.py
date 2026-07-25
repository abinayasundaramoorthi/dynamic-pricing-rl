"""
action_space.py

Owns the discrete pricing action space: building the Gymnasium
`spaces.Discrete`, converting an action index into a concrete price, and
producing a human-readable label for each action (used in rendering and
policy-inspection tooling, per the interpretability non-functional
requirement in the design doc).

Why discrete actions, and why the mapping lives here (not in pricing_env.py):
Every agent — baselines, tabular Q-Learning, DQN — needs to agree on exactly
what "action 3" means in dollars. Centralizing that conversion in one module
guarantees the baselines team, the Q-Learning owner, and the evaluation
harness are all pricing identically for the same action index, instead of
each reimplementing (and potentially drifting from) the same formula.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from gymnasium import spaces


def build_action_space(config: Any) -> spaces.Discrete:
    """
    Build the discrete action space from the configured price adjustments.

    One action index per entry in `config.price_adjustment_pct`. The size
    of the space is derived from the config rather than hardcoded, so a
    future experiment can widen/narrow the action set (e.g. finer-grained
    5% steps) purely via config, with no code changes here.
    """
    return spaces.Discrete(len(config.price_adjustment_pct))


def action_to_price(
    action: int,
    current_price: float,
    price_adjustment_pct: Sequence[float],
    min_price: float,
    max_price: float,
) -> float:
    """
    Convert a discrete action index into a concrete, bounded price.

    Parameters
    ----------
    action : int
        Index into `price_adjustment_pct`. Must be in
        [0, len(price_adjustment_pct) - 1].
    current_price : float
        The price the adjustment is applied relative to. We deliberately
        adjust relative to the *current* price (not the fixed base price)
        so an agent can compound adjustments across steps (e.g. two
        consecutive "-10%" actions produce a steeper discount than one),
        which is the more expressive, RL-friendly formulation.
    price_adjustment_pct : Sequence[float]
        Ordered fractional adjustments, e.g. -0.10 = 10% discount.
    min_price, max_price : float
        Hard bounds. Prices are clipped, never rejected — an agent
        exploring an extreme action should get a saturated (but still
        valid) price, not a crashed episode. This keeps training stable:
        a hard failure on an out-of-bounds price would inject a
        discontinuity into the reward landscape that an agent could not
        learn to avoid without first crashing many episodes.

    Returns
    -------
    float
        The resulting price, clipped to [min_price, max_price].

    Raises
    ------
    ValueError
        If `action` is not a valid index into `price_adjustment_pct`.
    """
    if not (0 <= action < len(price_adjustment_pct)):
        raise ValueError(
            f"Invalid action {action!r}. Must be an integer in "
            f"[0, {len(price_adjustment_pct) - 1}]."
        )
    pct_adjustment = price_adjustment_pct[action]
    raw_price = current_price * (1.0 + pct_adjustment)
    return float(np.clip(raw_price, min_price, max_price))


def describe_action(action: int, price_adjustment_pct: Sequence[float]) -> str:
    """
    Produce a human-readable label for an action, e.g. "Discount 10%".

    Used by `render()` and by policy-inspection notebooks (Week 4
    "Price Trajectory" analysis) so a revenue manager reviewing agent
    behaviour sees "Discount 10%" rather than a bare integer index —
    directly supporting the interpretability requirement in the design doc.

    Generated dynamically from the configured percentage (rather than a
    hardcoded label table) so it stays correct even if the action set is
    reconfigured to a different number of price tiers.
    """
    if not (0 <= action < len(price_adjustment_pct)):
        raise ValueError(
            f"Invalid action {action!r}. Must be an integer in "
            f"[0, {len(price_adjustment_pct) - 1}]."
        )
    pct = price_adjustment_pct[action]
    if pct < 0:
        return f"Discount {abs(pct):.0%}"
    if pct > 0:
        return f"Increase {pct:.0%}"
    return "Hold price"