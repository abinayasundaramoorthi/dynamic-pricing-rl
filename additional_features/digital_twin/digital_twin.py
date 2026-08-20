from .scenario_generator import ScenarioGenerator
from .market_simulator import MarketSimulator
from .pricing_simulator import PricingSimulator

class DigitalTwin:

    def __init__(self):

        self.pricing = PricingSimulator()
        self.generator = ScenarioGenerator()
        self.market = MarketSimulator()

    def run(self):

        scenario = self.generator.generate()

        demand = self.market.simulate(
            scenario
        )
        pricing = self.pricing.simulate(demand,scenario["competitor_price"])
        scenario.update(pricing)
        scenario["simulated_demand"] = demand
        
        return scenario