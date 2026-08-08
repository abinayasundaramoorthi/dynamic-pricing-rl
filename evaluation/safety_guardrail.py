"""
safety_guardrail.py

Safety Guardrail System — a protective layer that sits between the RL
agent's raw price decision and the actual price that gets used, checking
it against business safety rules before allowing it through.

Just like a car has both an engine (the RL agent, deciding what to do)
AND a seatbelt (this guardrail, protecting against bad outcomes even if
the engine does something wrong), this module doesn't try to make the
agent smarter -- it protects the business from any single risky decision
the agent might make, whether from a training bug, an unusual situation
it wasn't well-trained for, or any other cause.

Three safety rules are enforced:
  1. Minimum/Maximum price bounds -- price can never leave a safe range
  2. Maximum daily price change -- price can't jump too much in one day
  3. Statistical anomaly detection -- price can't wildly deviate from
     recent pricing history, even if within the absolute bounds
"""

import numpy as np


class PricingGuardrail:
    """
    Validates a proposed price against safety rules, and returns an
    approved (possibly corrected) price along with a clear explanation
    of any rule that was triggered.
    """

    def __init__(self, min_price=None, max_price=None,
                 max_daily_change_pct=0.15, history_window=5,
                 anomaly_std_threshold=2.5):
        """
        Parameters
        ----------
        min_price, max_price : float, optional
            Absolute price floor/ceiling. If None, that bound is not enforced.
        max_daily_change_pct : float
            Maximum allowed price change from the previous day, as a
            fraction (0.15 = 15%). Protects against sudden price shocks
            that could damage customer trust.
        history_window : int
            How many recent prices to remember for anomaly detection.
        anomaly_std_threshold : float
            How many standard deviations from recent average price counts
            as a statistical anomaly. Lower = stricter (catches more
            "unusual" prices); higher = more lenient.
        """
        self.min_price = min_price
        self.max_price = max_price
        self.max_daily_change_pct = max_daily_change_pct
        self.history_window = history_window
        self.anomaly_std_threshold = anomaly_std_threshold

        self.price_history = []

    def check_price(self, proposed_price, previous_price=None):
        """
        Check a proposed price against all safety rules.

        Parameters
        ----------
        proposed_price : float
            The price the RL agent (or any policy) wants to set.
        previous_price : float, optional
            Yesterday's price, used for the max-daily-change rule.

        Returns
        -------
        result : dict
            original_price, approved_price, was_overridden (bool),
            violations (list of str, empty if none triggered)
        """
        violations = []
        safe_price = proposed_price

        # --- Rule 1: Absolute min/max bounds ---
        if self.min_price is not None and safe_price < self.min_price:
            violations.append(
                f"Price ${safe_price:.2f} is below the minimum allowed "
                f"price of ${self.min_price:.2f}."
            )
            safe_price = self.min_price

        if self.max_price is not None and safe_price > self.max_price:
            violations.append(
                f"Price ${safe_price:.2f} is above the maximum allowed "
                f"price of ${self.max_price:.2f}."
            )
            safe_price = self.max_price

        # --- Rule 2: Maximum daily price change ---
        if previous_price is not None and previous_price > 0:
            change_pct = (safe_price - previous_price) / previous_price
            if abs(change_pct) > self.max_daily_change_pct:
                direction = 1 if change_pct > 0 else -1
                capped_price = previous_price * (1 + direction * self.max_daily_change_pct)
                violations.append(
                    f"Price change of {change_pct*100:+.1f}% from previous price "
                    f"(${previous_price:.2f}) exceeds the maximum allowed daily "
                    f"change of {self.max_daily_change_pct*100:.0f}%. "
                    f"Capped to ${capped_price:.2f}."
                )
                safe_price = capped_price

        # --- Rule 3: Statistical anomaly vs recent price history ---
        if len(self.price_history) >= 3:
            mean_price = float(np.mean(self.price_history))
            std_price = float(np.std(self.price_history))

            if std_price > 0:
                z_score = (safe_price - mean_price) / std_price
                if abs(z_score) > self.anomaly_std_threshold:
                    bound_price = mean_price + np.sign(z_score) * self.anomaly_std_threshold * std_price
                    violations.append(
                        f"Price ${safe_price:.2f} is a statistical outlier compared "
                        f"to recent pricing (z-score={z_score:.2f}, recent avg="
                        f"${mean_price:.2f}). Capped to ${bound_price:.2f}."
                    )
                    safe_price = float(bound_price)

        return {
            "original_price": proposed_price,
            "approved_price": safe_price,
            "was_overridden": violations != [],
            "violations": violations,
        }

    def record_price(self, price):
        """
        Record an actual (approved) price into the rolling history, used
        for future anomaly detection. Call this after check_price()
        approves a price and it's actually used.
        """
        self.price_history.append(price)
        if len(self.price_history) > self.history_window:
            self.price_history.pop(0)


def wrap_policy_with_guardrail(policy_fn, guardrail, price_adjustment_pct):
    """
    Wrap any existing policy function (like the ones from
    policy_evaluator.py) with the guardrail, so every action it chooses
    gets safety-checked before being used.

    Since our environment only accepts discrete price actions (not any
    arbitrary price), if the guardrail overrides the price, this finds
    the closest available discrete action to the approved safe price.

    Parameters
    ----------
    policy_fn : callable
        observation -> action (int), e.g. from policy_evaluator.py
    guardrail : PricingGuardrail
    price_adjustment_pct : list of float
        The environment's price adjustment percentages (same order as actions).

    Returns
    -------
    guarded_policy_fn : callable
        observation -> action (int), now safety-checked
    """
    state = {"previous_price": None, "base_price": None}

    def guarded_policy(observation):
        proposed_action = policy_fn(observation)
        proposed_pct = price_adjustment_pct[proposed_action]

        reference_price = state["previous_price"] or state["base_price"] or 200.0
        proposed_price = reference_price * (1 + proposed_pct)

        result = guardrail.check_price(proposed_price, state["previous_price"])

        if result["was_overridden"]:
            print(f"[GUARDRAIL] Overrode price ${result['original_price']:.2f} "
                  f"-> ${result['approved_price']:.2f}")
            for v in result["violations"]:
                print(f"  - {v}")

            # Find the closest available action to the approved safe price
            candidate_prices = [reference_price * (1 + pct) for pct in price_adjustment_pct]
            distances = [abs(cp - result["approved_price"]) for cp in candidate_prices]
            final_action = int(np.argmin(distances))
        else:
            final_action = proposed_action

        final_price = reference_price * (1 + price_adjustment_pct[final_action])
        guardrail.record_price(final_price)
        state["previous_price"] = final_price

        return final_action

    return guarded_policy


if __name__ == "__main__":
    # ------------------------------------------------------------
    # DEMONSTRATION: each rule catching a violation
    # ------------------------------------------------------------
    print("=== Demonstration: Safety Guardrail Rules ===\n")

    guardrail = PricingGuardrail(
        min_price=100, max_price=400,
        max_daily_change_pct=0.15,
        history_window=5, anomaly_std_threshold=2.0,
    )

    print("--- Rule 1: Minimum/Maximum bounds ---")
    result = guardrail.check_price(proposed_price=50)  # below min_price=100
    print(f"Proposed: $50 -> Approved: ${result['approved_price']:.2f}")
    for v in result["violations"]:
        print(f"  - {v}")

    result = guardrail.check_price(proposed_price=500)  # above max_price=400
    print(f"\nProposed: $500 -> Approved: ${result['approved_price']:.2f}")
    for v in result["violations"]:
        print(f"  - {v}")

    print("\n--- Rule 2: Maximum daily price change (15%) ---")
    result = guardrail.check_price(proposed_price=280, previous_price=200)  # +40% jump
    print(f"Proposed: $280 (from $200) -> Approved: ${result['approved_price']:.2f}")
    for v in result["violations"]:
        print(f"  - {v}")

    print("\n--- Rule 3: Statistical anomaly detection ---")
    guardrail2 = PricingGuardrail(anomaly_std_threshold=2.0)
    # Build up a stable price history
    for p in [200, 202, 198, 201, 199]:
        guardrail2.record_price(p)

    result = guardrail2.check_price(proposed_price=350)  # wild outlier vs stable history
    print(f"Recent history: [200, 202, 198, 201, 199]")
    print(f"Proposed: $350 -> Approved: ${result['approved_price']:.2f}")
    for v in result["violations"]:
        print(f"  - {v}")

    print("\n--- A normal, safe price (no violations expected) ---")
    # Note: with a very tightly clustered history like [200, 202, 198, 201, 199],
    # the standard deviation is small, so even modest-looking price changes can
    # register as statistically unusual. Using a price very close to the recent
    # average here to demonstrate the "no violation" case clearly.
    result = guardrail2.check_price(proposed_price=200)
    print(f"Proposed: $200 -> Approved: ${result['approved_price']:.2f} "
          f"(Overridden: {result['was_overridden']})")

    print("\nAll guardrail rules demonstrated successfully!")