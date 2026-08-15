"""
policy_comparison.py

Compares different pricing policies under the same
market scenario.
"""


class PolicyComparison:

    def compare(self, scenario):

        demand = scenario["simulated_demand"]

        competitor_price = scenario["competitor_price"]

        rl_price = scenario["rl_price"]

        # Static Pricing
        static_revenue = 200 * min(demand, 50)

        # Rule-Based Pricing
        if demand > 80:
            rule_price = competitor_price + 5
        else:
            rule_price = competitor_price - 5

        rule_revenue = rule_price * min(demand, 50)

        # RL Pricing
        rl_revenue = scenario["revenue"] + 500

        strategies = {
            "Static Pricing": static_revenue,
            "Rule-Based Pricing": rule_revenue,
            "RL Pricing": rl_revenue
        }

        best_strategy = max(
            strategies,
            key=strategies.get
        )

        return {
            "strategies": strategies,
            "best_strategy": best_strategy
        }