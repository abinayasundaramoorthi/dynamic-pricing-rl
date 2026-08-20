class WeatherSimulator:

    def demand_multiplier(
        self,
        weather
    ):

        if weather == "Sunny":
            return 1.2

        if weather == "Rainy":
            return 0.8

        return 1.0