"""
feedback_visualizer.py

Visualizes manager feedback statistics.
"""

import matplotlib.pyplot as plt


class FeedbackVisualizer:
    """
    Creates simple charts for manager feedback analysis.
    """

    @staticmethod
    def plot_feedback(stats):
        labels = list(stats.keys())
        values = list(stats.values())

        plt.figure(figsize=(6, 4))
        plt.bar(labels, values)

        plt.title("Manager Feedback Summary")
        plt.xlabel("Feedback Type")
        plt.ylabel("Count")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":

    sample_stats = {
        "Approved": 8,
        "Modified": 4,
        "Rejected": 2
    }

    FeedbackVisualizer.plot_feedback(sample_stats)