# Week 2 Progress Report

## **Project Title:** Reinforcement Learning for Dynamic Pricing

# Overview

During the second week of the project, I focused on developing and integrating the core components of the **Reinforcement Learning environment for Dynamic Pricing**. The work involved implementing state representation, demand simulation, environment transition logic, integrating the previously developed action space and reward function, and validating the complete pricing environment through testing.

---

# Task 1: Implement State Space Module

## Objective

To define the information that the Reinforcement Learning agent observes before making a pricing decision.

## Work Completed

Implemented the **State Space Module** to represent the current condition of the hotel pricing environment.

The state representation includes important business factors such as:

* Current price
* Remaining inventory
* Time remaining
* Demand-related information
* Other relevant market conditions

The state is structured so that the RL agent can use the information to select an appropriate pricing action.

## Deliverables

* `pricing_env/state.py`

---

# Task 2: Implement Demand Simulator

## Objective

To simulate realistic customer demand based on pricing and market conditions.

## Work Completed

Developed the **Demand Simulator** to generate demand values for different pricing situations.

The simulator considers the relationship between:

* Room price
* Customer demand
* Market conditions
* Pricing changes

This allows the RL environment to simulate how customer demand may change when the hotel increases or decreases its room price.

The demand simulator provides the environment with realistic demand values required for inventory and revenue calculations.

## Deliverables

* `pricing_env/demand_simulator.py`

---

# Task 3: Implement Environment Transition Logic

## Objective

To define how the environment changes from the current state to the next state after the RL agent performs an action.

## Work Completed

Implemented the **Transition Logic** responsible for generating the next environment state.

The transition process includes:

1. Receiving the current state.
2. Receiving the pricing action selected by the agent.
3. Updating the room price.
4. Simulating customer demand.
5. Updating the remaining inventory.
6. Updating the time step.
7. Generating the next state.

This establishes the basic state-transition mechanism required for the Markov Decision Process.

## Deliverables

* `pricing_env/transition.py`

---

# Task 4: Integrate Action Space and Reward Function

## Objective

To connect the previously implemented Action Space and Reward Function with the other environment components.

## Work Completed

Integrated the following modules:

* State Space
* Action Space
* Demand Simulator
* Reward Function
* Transition Logic

The environment now follows the basic RL interaction cycle:

```text
Current State
      ↓
RL Agent
      ↓
Select Action
      ↓
Update Price
      ↓
Simulate Demand
      ↓
Update Inventory
      ↓
Calculate Reward
      ↓
Generate Next State
```

This integration allows the pricing environment to respond to actions selected by an RL agent.

## Deliverables

* Integrated environment modules
* Updated project environment structure

---

# Task 5: Implement Pricing Environment Step Function

## Objective

To implement the main interaction function between the RL agent and the pricing environment.

## Work Completed

Implemented the `step()` functionality of the pricing environment.

The step function performs the following operations:

* Accepts the action selected by the RL agent.
* Applies the corresponding pricing change.
* Simulates customer demand.
* Updates inventory.
* Calculates the reward.
* Generates the next observation.
* Determines whether the episode has terminated or been truncated.
* Returns additional information required for analysis.

The environment follows the standard interaction pattern:

```python
observation, reward, terminated, truncated, info
```

## Deliverables

* `pricing_env/pricing_env.py`

---

# Task 6: Environment Integration and Testing

## Objective

To verify that all components of the Dynamic Pricing environment work together correctly.

## Work Completed

Performed integration testing of the environment modules.

Validated:

* Correct action execution.
* Correct price updates.
* Correct inventory updates.
* Demand generation.
* Reward calculation.
* Next-state generation.
* Episode termination conditions.
* Interaction between the RL agent and environment.

Test cases were used to verify that different pricing actions produce the expected changes in price, demand, inventory, and reward.

## Deliverables

* Environment integration tests
* Test results
* Bug fixes and validation

---

# Technical Skills Applied

During this week, the following concepts and technologies were explored and applied:

* Reinforcement Learning
* Markov Decision Process (MDP)
* State Representation
* Action Space
* Reward Engineering
* Demand Simulation
* Environment Transition Logic
* Inventory Management
* Revenue Optimization
* Python Programming
* Object-Oriented Programming
* Modular Software Architecture
* Unit Testing
* Git and GitHub Workflow

---

# Key Learnings

Throughout Week 2, I gained a better understanding of:

* How an RL agent interacts with an environment.
* How states represent the current business situation.
* How actions affect hotel pricing.
* How pricing changes can influence simulated demand.
* How inventory changes after customer bookings.
* How rewards guide the RL agent toward better pricing decisions.
* How transition logic connects one environment state to the next.
* How different Python modules can be integrated into a complete RL environment.

---

# Challenges Faced

The major challenges during Week 2 included:

* Designing a meaningful state representation for hotel pricing.
* Understanding how demand should respond to price changes.
* Connecting multiple environment modules correctly.
* Ensuring inventory does not become negative.
* Maintaining correct state transitions after each action.
* Integrating the reward function with the environment.
* Testing different pricing scenarios and identifying implementation issues.

---

# Outcome

By the end of Week 2, the major components required for the **Dynamic Pricing Reinforcement Learning environment** were implemented and integrated. The environment can now process pricing actions, simulate demand, update inventory, calculate rewards, and generate the next state.

This provides the required foundation for developing and training the Reinforcement Learning agent in the following stages of the project.

---

# Next Steps

During the upcoming weeks, the focus will shift towards:

* Implementing Q-Learning.
* Developing the RL pricing agent.
* Training the agent using the Dynamic Pricing environment.
* Comparing RL pricing with baseline pricing strategies.
* Evaluating revenue and occupancy performance.
* Developing visualization and performance metrics.
* Improving the pricing strategy based on experimental results.

---

# Summary

| Task                    | Work                                                | Status    |
| ----------------------- | --------------------------------------------------- | --------- |
| State Space             | Implemented state representation                    | Completed |
| Demand Simulator        | Implemented demand generation                       | Completed |
| Transition Logic        | Implemented state transitions                       | Completed |
| Environment Integration | Integrated state, action, demand and reward modules | Completed |
| `step()` Function       | Implemented RL environment interaction              | Completed |
| Environment Testing     | Tested pricing environment components               | Completed |

---

**Week 2 Status: Successfully Completed**
