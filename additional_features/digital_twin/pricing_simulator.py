import random


class PricingSimulator:

    def simulate(self, demand, competitor_price):

        rl_price = competitor_price + random.randint(-10, 25)
        minimum_rooms = min(40, demand)
        rooms_sold = random.randint(minimum_rooms, demand)
        revenue = rl_price * rooms_sold

        return {
            "rl_price": rl_price,
            "rooms_sold": rooms_sold,
            "revenue": revenue
        }