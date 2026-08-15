from .weather_simulator import WeatherSimulator
from .event_simulator import EventSimulator


class MarketSimulator:

    def __init__(self):

        self.weather = WeatherSimulator()
        self.event = EventSimulator()

    def simulate(
        self,
        scenario
    ):

        multiplier = (
            self.weather.demand_multiplier(
                scenario["weather"]
            )
            *
            self.event.demand_multiplier(
                scenario["event"]
            )
        )

        demand = int(
            scenario["expected_demand"]
            * multiplier
        )

        return demand