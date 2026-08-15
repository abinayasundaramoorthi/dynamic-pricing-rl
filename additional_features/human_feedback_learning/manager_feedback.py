"""
Manager Feedback Module

Allows a revenue manager to review AI pricing
recommendations.
"""

from datetime import datetime

from .feedback_memory import FeedbackMemory, FeedbackRecord


class ManagerFeedback:

    def __init__(self):

        self.memory = FeedbackMemory()

    def submit_feedback(
        self,
        ai_price,
        manager_price,
        scenario,
    ):

        if manager_price == ai_price:
            decision = "Accepted"

        elif manager_price > ai_price:
            decision = "Increased"

        else:
            decision = "Reduced"

        feedback = FeedbackRecord(
            ai_price=ai_price,
            manager_price=manager_price,
            decision=decision,
            scenario=scenario,
            timestamp=str(datetime.now())
        )

        self.memory.add_feedback(feedback)

        return decision