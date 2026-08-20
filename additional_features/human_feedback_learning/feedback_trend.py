"""
feedback_trend.py

Shows manager decision trend.
"""


class FeedbackTrend:

    @staticmethod
    def display(memory):

        print("\n")
        print("=" * 60)
        print(" MANAGER DECISION TREND ")
        print("=" * 60)

        for i, record in enumerate(memory.records, start=1):

            print(
                f"{i}. "
                f"{record.scenario:<20}"
                f"{record.decision}"
            )

        print("=" * 60)