class FeedbackExplanation:

    @staticmethod
    def explain(record):

        if record.decision == "Accepted":

            return (
                "AI recommendation was accepted."
            )

        if record.decision == "Reduced":

            return (
                "Manager reduced price because demand was expected to be low."
            )

        return (
            "Manager increased price because demand was expected to be high."
        )