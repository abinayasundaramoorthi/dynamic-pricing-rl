import random


class ScenarioGenerator:

    def generate(self):

        return {
            "event": random.choice([
                "Normal Day",
                "Festival",
                "Holiday",
                "Conference"
            ]),

            "weather": random.choice([
                "Sunny",
                "Rainy",
                "Cloudy"
            ]),

            "competitor_price": random.randint(180,260),

            "expected_demand": random.randint(30,100)
        }