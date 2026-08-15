"""
dashboard_data.py

Formats feedback statistics for dashboard display.
"""

from .feedback_statistics import FeedbackStatistics


class DashboardData:

    def __init__(self, history):
        self.statistics = FeedbackStatistics(history)

    def build_dashboard(self):

        return {
            "Acceptance Rate":
                f"{self.statistics.acceptance_rate():.2f}%",

            "Average Price Difference":
                round(self.statistics.average_difference(), 2),

            "Total Decisions":
                self.statistics.total_feedback(),

            "Approved Decisions":
                self.statistics.total_approved(),

            "Modified Decisions":
                self.statistics.total_modified(),
        }


if __name__ == "__main__":

    history = [
        {"accepted": True, "difference": 0},
        {"accepted": False, "difference": 10},
        {"accepted": True, "difference": 0},
    ]

    dashboard = DashboardData(history)

    print(dashboard.build_dashboard())