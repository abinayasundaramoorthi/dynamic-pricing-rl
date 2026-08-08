"""
risk_analyzer.py

Analyzes business risks for every simulated scenario.
"""


class RiskAnalyzer:

    def analyze(self, scenario):

        demand = scenario["simulated_demand"]

        weather = scenario["weather"]

        event = scenario["event"]

        if weather == "Rainy":

            return {
                "risk": "HIGH",
                "reason": "Bad weather may reduce bookings.",
                "recommendation": "Reduce room price by 5%."
            }

        if demand < 50:

            return {
                "risk": "MEDIUM",
                "reason": "Low customer demand.",
                "recommendation": "Offer promotional discounts."
            }

        if event == "Festival":

            return {
                "risk": "LOW",
                "reason": "High demand expected.",
                "recommendation": "Increase room price."
            }

        return {
            "risk": "LOW",
            "reason": "Stable market conditions.",
            "recommendation": "Maintain current pricing."
        }