# Week 1 Report: Designing the MDP and Gym Environment

## Objective
Formulate the dynamic pricing problem as a Markov Decision Process (MDP) and build a custom Gymnasium environment to simulate it.

## Work Completed

1. **MDP Formulation** (`docs/mdp_formulation.md`)
   Formally defined the State space (days remaining, inventory remaining), Action space (5 discrete price levels: $50-$200), Reward function (units sold x price), transition dynamics, and discount factor (gamma = 0.99).

2. **Custom Gymnasium Environment** (`environment/pricing_env.py`)
   Built `DynamicPricingEnv`, properly subclassing `gymnasium.Env` with formal `action_space` and `observation_space` definitions, following the standard `reset()`/`step()` API contract. Validated against Gymnasium's official `check_env()` compliance checker - passed, including reproducibility under a fixed random seed.

3. **Stochastic Demand Model** (`environment/demand_simulator.py`)
   Implemented `simulate_demand(price)`: purchase probability decreases as price increases, with random variation via NumPy. Upgraded to be time-dependent - customers become more price-tolerant as the deadline approaches, simulating real last-minute booking urgency.

4. **Validation** (`notebooks/gym_env_validation.ipynb`)
   Confirmed via testing: (a) demand decreases as price increases, (b) demand increases as time runs out at a fixed price (urgency effect), (c) fully reproducible results under a fixed seed.

## Key Outcome
A working, industry-standard-compliant simulation environment ready to train RL agents against, with all core assumptions validated through direct testing.

## Dependencies Added
`gymnasium`, `numpy`

## Files Delivered
- `docs/mdp_formulation.md`
- `environment/pricing_env.py`
- `environment/demand_simulator.py`
- `environment/__init__.py`
- `notebooks/gym_env_validation.ipynb`
- `requirements.txt`
