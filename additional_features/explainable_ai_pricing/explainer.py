"""
explainer.py

`PricingExplainer` — the single entry point for Feature 1 (Explainable AI
Pricing Insights). Ties together:

    factors.py     -> extract_pricing_factors()   (rule-based, always available)
    confidence.py  -> confidence_from_action_values(), dqn_state_sensitivity()
                       (available whenever the caller has action values / a
                       trained QNetwork; gracefully omitted otherwise)

into one `PricingExplanation` result offering four views onto the same
underlying (real, non-fabricated) data, exactly as required:

    .human_readable   -- narrative paragraph for a revenue manager
    .to_json()        -- machine-readable JSON string
    .developer_view()  -- dict of raw numeric internals for engineers/logs
    .factors           -- ranked List[PriceFactor] (feature importance)
    .confidence         -- ConfidenceScore
    .recommendation_summary -- one-line headline (e.g. "Price increased by 8%")

Usage
-----
    from additional_features.explainable_ai_pricing import PricingExplainer

    explainer = PricingExplainer()
    explanation = explainer.explain(
        observation=obs,
        info=info,
        previous_price=previous_price,
        action_values=q_values,          # optional: DQN Q-values or Q-table row
        q_network=agent.q_network,        # optional: enables sensitivity analysis
        selected_action=info["action"],
    )
    print(explanation.human_readable)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .confidence import ConfidenceScore, confidence_from_action_values, dqn_state_sensitivity
from .factors import PriceFactor, extract_pricing_factors


@dataclass
class PricingExplanation:
    """
    Full explanation of one pricing decision, in every format the
    project's explainability requirement calls for.

    Attributes
    ----------
    recommendation_summary : str
        One-line headline, e.g. "Price increased by 8.0% to ₹19,838.11".
    human_readable : str
        Multi-sentence narrative combining the summary, the top-ranked
        factors, and the confidence assessment — written for a revenue
        manager, not an engineer.
    factors : List[PriceFactor]
        Full ranked factor-importance list (see `factors.py`).
    confidence : Optional[ConfidenceScore]
        Present only when `action_values` was supplied to `.explain()`.
    sensitivity : Optional[Dict[str, float]]
        Present only when both `q_network` and `selected_action` were
        supplied — finite-difference Q-value sensitivity per state
        feature (see `confidence.dqn_state_sensitivity`).
    raw : Dict[str, Any]
        The raw `observation`/`info` this explanation was built from,
        kept verbatim for traceability/logging.
    """

    recommendation_summary: str
    human_readable: str
    factors: List[PriceFactor]
    confidence: Optional[ConfidenceScore] = None
    sensitivity: Optional[Dict[str, float]] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    def developer_view(self) -> Dict[str, Any]:
        """
        Dict of raw numeric internals, intended for logging/debugging —
        every number an engineer would need to verify this explanation
        by hand, with nothing summarized away.
        """
        return {
            "recommendation_summary": self.recommendation_summary,
            "factors": [
                {
                    "key": f.key,
                    "label": f.label,
                    "direction": f.direction,
                    "weight": round(f.weight, 6),
                    "raw_value": f.raw_value,
                }
                for f in self.factors
            ],
            "confidence": (
                {
                    "score": round(self.confidence.score, 6),
                    "label": self.confidence.label,
                    "margin": self.confidence.margin,
                }
                if self.confidence is not None
                else None
            ),
            "sensitivity": self.sensitivity,
            "raw": self.raw,
        }

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serializable dict combining every view (used by `to_json()`)."""
        return {
            "recommendation_summary": self.recommendation_summary,
            "human_readable": self.human_readable,
            "confidence": (
                {"score": self.confidence.score, "label": self.confidence.label}
                if self.confidence is not None
                else None
            ),
            "feature_importance": [
                {
                    "key": f.key,
                    "label": f.label,
                    "direction": f.direction,
                    "weight": round(f.weight, 4),
                    "narrative": f.narrative,
                }
                for f in self.factors
            ],
            "sensitivity": self.sensitivity,
            "developer": self.developer_view(),
        }

    def to_json(self, indent: int = 2) -> str:
        """Full JSON explanation, as required by the feature spec."""
        return json.dumps(self.to_dict(), indent=indent, default=str)


class PricingExplainer:
    """
    Stateless orchestrator that turns one environment step's
    `(observation, info)` — optionally alongside the agent's per-action
    values and/or trained Q-network — into a `PricingExplanation`.

    Deliberately holds no reference to any specific agent or environment
    instance (mirrors `pricing_env/reward.py` and `demand_simulator.py`'s
    "pure function of its inputs" design) so it can explain a DQN agent,
    a Q-Learning agent, or a logged historical step identically.
    """

    def explain(
        self,
        observation: Sequence[float],
        info: Dict[str, Any],
        previous_price: Optional[float] = None,
        action_values: Optional[Sequence[float]] = None,
        selected_action: Optional[int] = None,
        q_network: Optional[Any] = None,
    ) -> PricingExplanation:
        """
        Build a full `PricingExplanation` for one pricing decision.

        Parameters
        ----------
        observation : Sequence[float]
            `[remaining_inventory, days_remaining]`, as returned by
            `PricingEnvironment.step()`/`.reset()`.
        info : Dict[str, Any]
            The `info` dict returned alongside `observation` by
            `PricingEnvironment.step()`.
        previous_price : float, optional
            Price in effect before this step, for describing the price
            change itself. If omitted, that factor is skipped.
        action_values : Sequence[float], optional
            Per-action Q-values (`DQNAgent.q_network(obs)` output) or a
            Q-Learning `q_table[state]` row. If given (together with
            `selected_action`), a `ConfidenceScore` is computed.
        selected_action : int, optional
            The action actually taken this step. Required for
            `action_values`/`q_network`-based analysis; defaults to
            `info["action"]` when present and not explicitly given.
        q_network : agents.dqn_network.QNetwork, optional
            A trained DQN network. If given (together with
            `selected_action`), a state-sensitivity breakdown is
            computed via `confidence.dqn_state_sensitivity`.

        Returns
        -------
        PricingExplanation
        """
        if selected_action is None:
            selected_action = info.get("action")

        factors = extract_pricing_factors(observation, info, previous_price=previous_price)

        confidence: Optional[ConfidenceScore] = None
        if action_values is not None and selected_action is not None:
            confidence = confidence_from_action_values(action_values, int(selected_action))

        sensitivity: Optional[Dict[str, float]] = None
        if q_network is not None and selected_action is not None:
            sensitivity = dqn_state_sensitivity(q_network, observation, int(selected_action))

        recommendation_summary = self._build_summary(info, previous_price)
        human_readable = self._build_narrative(
            recommendation_summary, factors, confidence
        )

        return PricingExplanation(
            recommendation_summary=recommendation_summary,
            human_readable=human_readable,
            factors=factors,
            confidence=confidence,
            sensitivity=sensitivity,
            raw={"observation": list(map(float, observation)), "info": _json_safe(info)},
        )

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _build_summary(info: Dict[str, Any], previous_price: Optional[float]) -> str:
        price = float(info["price"])
        action_label = info.get("action_label", "Price update")
        if previous_price and previous_price > 0:
            pct_change = (price - previous_price) / previous_price
            if abs(pct_change) < 1e-6:
                return f"Price held at ₹{price:,.2f}."
            direction = "increased" if pct_change > 0 else "decreased"
            return f"Price {direction} by {abs(pct_change) * 100:.1f}% to ₹{price:,.2f}."
        return f"{action_label}: price set to ₹{price:,.2f}."

    @staticmethod
    def _build_narrative(
        summary: str,
        factors: List[PriceFactor],
        confidence: Optional[ConfidenceScore],
    ) -> str:
        top_factors = [f for f in factors if f.weight >= 0.12][:4] or factors[:2]
        reasons = "\n".join(f"- {f.narrative}" for f in top_factors)
        lines = [summary, "", "Reasons:", reasons]
        if confidence is not None:
            lines.append("")
            lines.append(f"Confidence: {confidence.label} ({confidence.score * 100:.0f}%).")
        return "\n".join(lines)


def _json_safe(info: Dict[str, Any]) -> Dict[str, Any]:
    """Shallow-convert an info dict to plain JSON-serializable types."""
    safe: Dict[str, Any] = {}
    for key, value in info.items():
        if isinstance(value, dict):
            safe[key] = {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in value.items()}
        elif isinstance(value, (int, float, str, bool)) or value is None:
            safe[key] = value
        else:
            safe[key] = str(value)
    return safe
