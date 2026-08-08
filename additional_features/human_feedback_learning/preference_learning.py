"""
Learns manager pricing preferences.
"""

from .feedback_memory import FeedbackMemory


class PreferenceLearning:

    def __init__(self, memory: FeedbackMemory):

        self.memory = memory

    # Existing Method
    def preferred_adjustment(self):

        if len(self.memory.records) == 0:
            return 0

        adjustments = []

        for record in self.memory.records:

            adjustments.append(
                record.manager_price -
                record.ai_price
            )

        return sum(adjustments) / len(adjustments)

    # Existing Method
    def recommend_price(self, ai_price):

        adjustment = self.preferred_adjustment()

        return round(ai_price + adjustment, 2)

    # -------------------------------
    # NEW METHODS
    # -------------------------------

    def manager_preference(self):

        if len(self.memory.records) == 0:
            return "No feedback available."
        accepted = 0

        for record in self.memory.records:
            if record.decision == "Accepted":
                accepted += 1

        acceptance = accepted / len(self.memory.records)
        if acceptance >= 0.80:
            return "Manager usually accepts AI recommendations."

        elif acceptance >= 0.50:
            return "Manager occasionally modifies AI pricing."

        else:
            return "Manager frequently overrides AI recommendations."

    def acceptance_percentage(self):
        """
        Percentage of AI recommendations accepted.
        """

        if len(self.memory.records) == 0:
            return 0

        accepted = sum(
            1
            for record in self.memory.records
            if record.decision == "Accepted"
        )

        return round(
            accepted / len(self.memory.records) * 100,
            2
        )

    def business_insight(self):
        """
        Generates a simple business insight.
        """

        acceptance = self.acceptance_percentage()

        if acceptance >= 80:

            return (
                 "AI pricing aligns well with business expectations."
            )

        elif  acceptance >= 50:

            return (
               "AI pricing is partially aligned. Some manual adjustments are still required."
            )

        else:

            return (
                 "Managers frequently modify AI prices. "
                "The AI model should learn from additional feedback."
            )