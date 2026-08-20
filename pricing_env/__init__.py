"""
pricing_env package

Public API for the Travel & Hospitality dynamic pricing environment.
Import from the package root (`from pricing_env import PricingEnvironment`)
rather than reaching into submodules directly — this is what lets us
refactor internals (e.g. splitting demand_simulator.py further) later
without breaking every agent/training script that imports this package.
"""

from .action_space import action_to_price, build_action_space, describe_action
from .demand_simulator import DemandConfig, DemandSimulator, SaleOutcome
from .pricing_env import PricingEnvConfig, PricingEnvironment
from .reward import RewardBreakdown, RewardConfig, compute_reward
from .state import EnvState, build_observation_space

__all__ = [
    "PricingEnvironment",
    "PricingEnvConfig",
    "EnvState",
    "build_observation_space",
    "build_action_space",
    "action_to_price",
    "describe_action",
    "DemandConfig",
    "DemandSimulator",
    "SaleOutcome",
    "RewardConfig",
    "RewardBreakdown",
    "compute_reward",
]