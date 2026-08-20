"""
integration.py

Example integration with pricing agent.
"""
from additional_features.human_feedback_learning.manager_feedback import ManagerFeedback

class FeedbackIntegration:

    def __init__(self):

        self.feedback = ManagerFeedback()

    def review_price(self, price):

        decision = self.feedback.review(price)

        return decision