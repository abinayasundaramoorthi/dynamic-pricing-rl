"""
Stores manager feedback history.

This module keeps every pricing decision made by the
manager so future learning modules can analyse patterns.
"""

from dataclasses import dataclass, asdict
from typing import List
import csv
import os


@dataclass
class FeedbackRecord:
    ai_price: float
    manager_price: float
    decision: str
    scenario: str
    timestamp: str


class FeedbackMemory:

    def __init__(self):
        self.records: List[FeedbackRecord] = []

    def add_feedback(self, feedback: FeedbackRecord):
        self.records.append(feedback)

    def total_feedback(self):
        return len(self.records)

    def export_csv(self, filename="evaluation/feedback_history.csv"):

        os.makedirs("evaluation", exist_ok=True)

        with open(filename, "w", newline="") as file:

            writer = csv.DictWriter(
                file,
                fieldnames=FeedbackRecord.__annotations__.keys()
            )

            writer.writeheader()

            for record in self.records:
                writer.writerow(asdict(record))