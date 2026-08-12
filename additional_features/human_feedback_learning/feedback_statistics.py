"""
Business statistics for manager feedback.
"""

from .feedback_memory import FeedbackMemory, FeedbackRecord


class FeedbackStatistics:

    def __init__(self, memory: FeedbackMemory):

        self.memory = memory

    def acceptance_rate(self):

        total = len(self.memory.records)

        if total == 0:
            return 0

        accepted = sum(
            r.decision == "Accepted"
            for r in self.memory.records
        )

        return accepted / total * 100

    def average_difference(self):

        total = len(self.memory.records)

        if total == 0:
            return 0

        diff = sum(
            abs(r.ai_price - r.manager_price)
            for r in self.memory.records
        )

        return diff / total

    def total_feedback(self):

        return len(self.memory.records)