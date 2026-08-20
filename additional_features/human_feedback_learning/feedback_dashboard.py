"""
feedback_dashboard.py

Displays business dashboard for manager feedback.
"""


class FeedbackDashboard:

    @staticmethod
    def display(memory):

        accepted = 0
        increased = 0
        reduced = 0

        for record in memory.records:

            if record.decision == "Accepted":
                accepted += 1

            elif record.decision == "Increased":
                increased += 1

            elif record.decision == "Reduced":
                reduced += 1

        print("\n")
        print("=" * 60)
        print(" MANAGER FEEDBACK DASHBOARD ")
        print("=" * 60)

        print(f"Total Feedback      : {len(memory.records)}")
        print(f"Accepted Decisions  : {accepted}")
        print(f"Price Increased     : {increased}")
        print(f"Price Reduced       : {reduced}")

        print("=" * 60)