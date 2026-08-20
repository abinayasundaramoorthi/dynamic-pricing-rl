"""
feedback_export.py

Exports feedback history to CSV.
"""

import csv
import os


class FeedbackExporter:

    @staticmethod
    def export(memory):

        os.makedirs("evaluation", exist_ok=True)

        with open(
            "evaluation/manager_feedback.csv",
            "w",
            newline=""
        ) as file:

            writer = csv.writer(file)

            writer.writerow([
                "Scenario",
                "AI Price",
                "Manager Price",
                "Decision",
                "Timestamp"
            ])

            for record in memory.records:

                writer.writerow([
                    record.scenario,
                    record.ai_price,
                    record.manager_price,
                    record.decision,
                    record.timestamp
                ])

        print("\nFeedback exported to evaluation/manager_feedback.csv")