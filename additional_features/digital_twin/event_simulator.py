class EventSimulator:

    def demand_multiplier(
        self,
        event
    ):

        if event == "Festival":
            return 1.5

        if event == "Holiday":
            return 1.3

        if event == "Conference":
            return 1.4

        return 1.0