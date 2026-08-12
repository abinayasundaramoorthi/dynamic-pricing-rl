"""
feedback_report.py

Generates a report from feedback statistics.
"""

from datetime import datetime


def generate_feedback_report(stats, filename="reports/human_feedback_report.txt"):

    with open(filename, "w") as file:

        file.write("Human Feedback Learning Report\n")
        file.write("=" * 40 + "\n")
        file.write(f"Generated: {datetime.now()}\n\n")

        total = sum(stats.values())

        file.write(f"Total Feedback: {total}\n\n")

        for key, value in stats.items():
            file.write(f"{key}: {value}\n")

    print("Feedback report generated.")