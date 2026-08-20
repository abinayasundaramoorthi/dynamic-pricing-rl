from .digital_twin import DigitalTwin


class SimulationRunner:

    def __init__(self):

        self.twin = DigitalTwin()

    def run(
        self,
        episodes=5
    ):

        results = []

        for _ in range(episodes):

            results.append(
                self.twin.run()
            )

        return results