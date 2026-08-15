"""
feedback_logger.py

Simple logger for manager feedback events.
"""

from datetime import datetime


class FeedbackLogger:

    def __init__(self, logfile="feedback_log.txt"):
        self.logfile = logfile

    def log(self, message):

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open(self.logfile, "a") as file:
            file.write(f"[{timestamp}] {message}\n")

    def log_feedback(
        self,
        ai_price,
        manager_price,
        decision,
    ):

        self.log(
            f"AI={ai_price}, "
            f"Manager={manager_price}, "
            f"Decision={decision}"
        )


if __name__ == "__main__":

    logger = FeedbackLogger()

    logger.log_feedback(
        220,
        235,
        "Modified"
    )

    print("Log created.")