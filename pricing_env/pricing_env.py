"""
pricing_env.py

Custom Reinforcement Learning environment for the Dynamic Pricing project.

This environment simulates selling a fixed inventory of items (e.g. tickets)
over a limited selling season. At each step, the agent sets a price, and the
environment simulates how many units get sold, updates inventory, and returns
the new state, reward, and whether the episode is done.

State  = [inventory_left, days_left]
Action = index into a list of possible price levels
Reward = revenue earned this step (see reward.py)

The actual state transition math lives in transition.py — this class is a
thin wrapper providing the standard reset() / step() / render() interface.
"""

from transition import apply_transition


class PricingEnv:
    """A simple custom environment for dynamic pricing RL."""

    def __init__(self, initial_inventory=100, total_days=30,
                 price_levels=None, base_demand=10):
        """
        Parameters
        ----------
        initial_inventory : int
            Number of units available at the start of each episode.
        total_days : int
            Number of days in one selling season (one episode).
        price_levels : list of float
            The discrete list of prices the agent can choose from.
            Example: [200, 300, 400, 500, 600]
        base_demand : float
            Baseline daily demand used in the demand simulation formula.
        """
        self.initial_inventory = initial_inventory
        self.total_days = total_days
        self.price_levels = price_levels if price_levels is not None else [200, 300, 400, 500, 600]
        self.base_demand = base_demand

        self.inventory = None
        self.days_left = None

    def reset(self):
        """Start a new episode (new selling season)."""
        self.inventory = self.initial_inventory
        self.days_left = self.total_days
        return self._get_state()

    def step(self, action):
        """
        Apply one pricing decision and move the environment forward by one day.

        Parameters
        ----------
        action : int
            Index into self.price_levels, chosen by the agent.

        Returns
        -------
        next_state, reward, done, info  (see transition.apply_transition)
        """
        price = self.price_levels[action]

        next_state, reward, done, info = apply_transition(
            price=price,
            inventory=self.inventory,
            days_left=self.days_left,
            base_demand=self.base_demand
        )

        # Update the environment's internal state
        self.inventory, self.days_left = next_state

        return next_state, reward, done, info

    def render(self):
        """Print the current state in a human-readable way."""
        print(f"Inventory left: {self.inventory} | Days left: {self.days_left}")

    def _get_state(self):
        """Return the current state as [inventory_left, days_left]."""
        return [self.inventory, self.days_left]
