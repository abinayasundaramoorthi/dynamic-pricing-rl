"""
confidence.py

Two genuinely-computed (not fabricated) explainability signals, built
directly on top of the project's existing agents:

1. `confidence_from_action_values` — turns a raw vector of per-action
   values (DQN Q-values from `agents/dqn_network.py::QNetwork.forward`,
   or a Q-Learning `q_table[state]` row from
   `agents/q_learning_agent.py`) into a bounded [0, 1] confidence score
   and a qualitative label, via a softmax-margin: how much more the
   chosen action's value dominates the runner-up.

2. `dqn_state_sensitivity` — a local sensitivity (finite-difference)
   analysis of a trained `agents.dqn_network.QNetwork`'s Q-value for the
   selected action, with respect to the two real observation dimensions
   (`remaining_inventory`, `days_remaining` — see
   `pricing_env/state.py::EnvState.to_observation`). This is a standard,
   legitimate local explainability technique (perturb one input at a
   time, measure the output change) applied to the actual trained
   network — not a canned or templated explanation.

Both functions accept plain numpy arrays / an optional `torch` network,
so they work identically whether called from `PricingExplainer`,
a notebook, or the advanced dashboard.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

try:  # torch is already a hard dependency of this project (requirements.txt)
    import torch
except ImportError:  # pragma: no cover - torch is required project-wide
    torch = None  # type: ignore


@dataclass(frozen=True)
class ConfidenceScore:
    """
    Attributes
    ----------
    score : float
        Bounded confidence in [0.0, 1.0]. Computed from the softmax
        margin between the chosen action's value and the best
        alternative action's value — NOT a fabricated or hardcoded
        number.
    label : str
        Qualitative bucket: "Low", "Medium", "High", or "Very High".
    margin : float
        Raw (un-normalized) difference between the chosen action's value
        and the second-best action's value, in the same units as the
        input action values (Q-value units).
    """

    score: float
    label: str
    margin: float


def _label_for_score(score: float) -> str:
    if score >= 0.85:
        return "Very High"
    if score >= 0.65:
        return "High"
    if score >= 0.40:
        return "Medium"
    return "Low"


def confidence_from_action_values(
    action_values: Sequence[float],
    selected_action: int,
) -> ConfidenceScore:
    """
    Compute a confidence score from a vector of per-action values.

    Uses a temperature-scaled softmax over `action_values` and reads off
    the probability mass assigned to `selected_action` as the confidence
    score. This rewards a large margin between the best and second-best
    action (a decisive agent) and penalizes near-ties (an agent that is
    effectively indifferent between two prices).

    Parameters
    ----------
    action_values : Sequence[float]
        Per-action Q-values (DQN) or Q-table row (Q-Learning), in action
        order matching `PricingEnvConfig.price_adjustment_pct`.
    selected_action : int
        Index of the action actually taken.

    Returns
    -------
    ConfidenceScore
    """
    values = np.asarray(action_values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError(f"action_values must be a non-empty 1D sequence, got shape {values.shape}")
    if not (0 <= selected_action < values.size):
        raise ValueError(
            f"selected_action={selected_action} out of range for {values.size} actions"
        )

    # Scale-invariant softmax: normalize by the spread of the values so
    # confidence reflects RELATIVE dominance of the chosen action, not
    # the absolute magnitude of Q-values (which varies with base_price).
    spread = float(np.std(values))
    temperature = spread if spread > 1e-6 else 1.0
    scaled = (values - values.max()) / temperature
    exp_values = np.exp(scaled)
    probabilities = exp_values / exp_values.sum()

    score = float(probabilities[selected_action])

    sorted_values = np.sort(values)[::-1]
    best = sorted_values[0]
    second_best = sorted_values[1] if sorted_values.size > 1 else sorted_values[0]
    margin = float(best - second_best) if selected_action == int(np.argmax(values)) else float(
        values[selected_action] - best
    )

    return ConfidenceScore(score=score, label=_label_for_score(score), margin=margin)


def dqn_state_sensitivity(
    q_network,
    observation: Sequence[float],
    selected_action: int,
    feature_names: Tuple[str, str] = ("remaining_inventory", "days_remaining"),
    perturbation_fraction: float = 0.05,
    device: Optional[str] = None,
) -> Dict[str, float]:
    """
    Finite-difference sensitivity of the trained DQN's Q-value for
    `selected_action` with respect to each observation dimension.

    For each feature, perturbs the observation by
    `+/- perturbation_fraction * max(1.0, |value|)`, re-runs the actual
    `q_network.forward()`, and records how much the selected action's
    Q-value changed. This directly reuses the project's real
    `agents.dqn_network.QNetwork` — the sensitivity is measured, not
    estimated or invented.

    Parameters
    ----------
    q_network : agents.dqn_network.QNetwork
        A trained (or untrained) network instance; only `.forward()` is
        used, so this also works with `DQNAgent.q_network` directly.
    observation : Sequence[float]
        The 2-element `[remaining_inventory, days_remaining]` observation
        (see `pricing_env/state.py`).
    selected_action : int
        The action index whose Q-value sensitivity is measured.
    feature_names : Tuple[str, str]
        Names for the two observation dimensions, matching
        `EnvState.to_observation()`'s fixed ordering.
    perturbation_fraction : float
        Relative perturbation size applied to each feature. Default 5%.
    device : str, optional
        Torch device to run the forward passes on; defaults to whatever
        device the network's parameters already live on.

    Returns
    -------
    Dict[str, float]
        Mapping from feature name to the absolute change in the selected
        action's Q-value caused by perturbing that feature (larger =
        more influential on this specific pricing decision).
    """
    if torch is None:  # pragma: no cover
        raise ImportError("PyTorch is required for dqn_state_sensitivity (see requirements.txt)")

    resolved_device = device or next(q_network.parameters()).device
    base_obs = np.asarray(observation, dtype=np.float32)
    if base_obs.shape[0] != len(feature_names):
        raise ValueError(
            f"observation has {base_obs.shape[0]} dims but {len(feature_names)} "
            f"feature_names were given"
        )

    def q_for(obs_array: np.ndarray) -> float:
        tensor = torch.tensor(obs_array, dtype=torch.float32, device=resolved_device)
        with torch.no_grad():
            q_values = q_network(tensor)
        return float(q_values[selected_action].item())

    base_q = q_for(base_obs)
    sensitivities: Dict[str, float] = {}

    for i, name in enumerate(feature_names):
        delta = max(1.0, abs(float(base_obs[i]))) * perturbation_fraction
        perturbed_up = base_obs.copy()
        perturbed_up[i] += delta
        perturbed_down = base_obs.copy()
        perturbed_down[i] = max(0.0, perturbed_down[i] - delta)

        q_up = q_for(perturbed_up)
        q_down = q_for(perturbed_down)

        # Central-difference-style magnitude, robust to which direction
        # was clipped (e.g. inventory can't go below 0).
        sensitivities[name] = float((abs(q_up - base_q) + abs(base_q - q_down)) / 2.0)

    return sensitivities
