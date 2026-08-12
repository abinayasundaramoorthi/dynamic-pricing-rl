"""
Generate manager feedback report.
"""

from .feedback_statistics import FeedbackStatistics


class FeedbackAnalyzer:

    def __init__(self, statistics: FeedbackStatistics):

        self.statistics = statistics

    def generate_report(self):

        print("=" * 50)

        print("MANAGER FEEDBACK REPORT")

        print("=" * 50)

        print(
            "Acceptance Rate:",
            round(
                self.statistics.acceptance_rate(),
                2
            ),
            "%"
        )

        print(
            "Average Price Difference:",
            round(
                self.statistics.average_difference(),
                2
            )
        )

        print(
            "Total Decisions:",
            self.statistics.total_feedback()
        )   