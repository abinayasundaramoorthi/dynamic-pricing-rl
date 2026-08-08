# Week 3 Progress Report

## **Project Title:** Reinforcement Learning for Dynamic Pricing

# Overview

During the third week of the project, I focused on developing the **Reinforcement Learning pricing agent**, understanding the learning process, and evaluating the agent's pricing decisions against baseline strategies. The work involved studying Q-Learning concepts, implementing the agent, connecting it with the Dynamic Pricing environment, running training episodes, and analyzing the resulting pricing behavior and revenue performance.

---

# Task 1: Study Q-Learning for Dynamic Pricing

## Objective

To understand how Q-Learning can be used to learn the optimal pricing action for different hotel market conditions.

## Work Completed

Studied the fundamentals of **Q-Learning**, including:

* States
* Actions
* Rewards
* Q-values
* Exploration
* Exploitation
* Learning rate
* Discount factor
* Epsilon-greedy strategy

The Q-Learning process was mapped to the hotel pricing problem, where the agent learns which pricing action provides better long-term revenue for a particular market state.

The basic learning process can be represented as:

```text
Current State
      ↓
Select Pricing Action
      ↓
Environment
      ↓
Receive Reward
      ↓
Observe Next State
      ↓
Update Q-Value
      ↓
Select Better Action
```

---

# Task 2: Implement Q-Learning Agent

## Objective

To develop an RL agent capable of learning pricing decisions from interactions with the Dynamic Pricing environment.

## Work Completed

Implemented the Q-Learning agent with the required learning parameters:

* Learning rate
* Discount factor
* Exploration rate
* Exploration decay
* Number of actions
* Q-value storage

The agent uses an **epsilon-greedy strategy** to balance exploration and exploitation.

During the initial training stages, the agent explores different pricing actions. As training progresses, the exploration rate decreases, allowing the agent to increasingly select actions that have produced better rewards.

## Deliverables

* Q-Learning agent implementation
* Agent configuration
* Q-value learning mechanism

---

# Task 3: Integrate RL Agent with Pricing Environment

## Objective

To connect the Q-Learning agent with the previously developed Dynamic Pricing environment.

## Work Completed

Integrated the RL agent with the environment so that the agent can continuously interact with the pricing system.

The interaction follows:

```text
State
  ↓
Q-Learning Agent
  ↓
Pricing Action
  ↓
Dynamic Pricing Environment
  ↓
Demand Simulation
  ↓
Inventory Update
  ↓
Reward Calculation
  ↓
Next State
  ↓
Q-Value Update
```

The agent receives the current state, selects a pricing action, receives a reward, and updates its knowledge based on the outcome.

---

# Task 4: Implement Training Process

## Objective

To train the RL agent through repeated interactions with the Dynamic Pricing environment.

## Work Completed

Implemented the training workflow using multiple episodes.

During each episode:

1. The environment is initialized.
2. The current state is provided to the agent.
3. The agent selects a pricing action.
4. The environment applies the action.
5. Demand is simulated.
6. Inventory is updated.
7. Revenue and reward are calculated.
8. The next state is generated.
9. The Q-value is updated.
10. The process continues until the episode ends.

Training results were collected for further evaluation.

## Deliverables

* RL training workflow
* Episode-based training
* Training performance data

---

# Task 5: Evaluate RL Pricing Against Baseline Strategies

## Objective

To determine whether the RL-based pricing strategy provides better business performance compared with traditional pricing approaches.

## Work Completed

Compared RL pricing with baseline strategies such as:

* Static/Flat Pricing
* Time-Based Pricing
* Demand-Based or Rule-Based Pricing

The comparison focused on business metrics including:

* Total Revenue
* Average Revenue
* Occupancy
* Rooms Sold
* Pricing behavior
* Overall reward

The evaluation helps determine whether the RL agent is learning meaningful pricing decisions rather than simply changing prices randomly.

---

# Task 6: Analyze Pricing Agent Performance

## Objective

To understand the behavior of the trained RL agent and identify areas for improvement.

## Work Completed

Analyzed:

* Reward progression across episodes.
* Pricing actions selected by the agent.
* Changes in room prices.
* Revenue generated.
* Inventory utilization.
* Agent exploration and exploitation behavior.

The training results were used to identify whether the agent was gradually improving its pricing decisions.

---

# Task 7: Performance Visualization

## Objective

To visualize the learning and pricing performance of the RL agent.

## Work Completed

Prepared performance visualizations and analysis for metrics such as:

* Episode reward
* Revenue
* Pricing behavior
* Occupancy
* Inventory utilization
* Comparison between RL and baseline strategies

These visualizations make it easier to understand the learning behavior of the agent and communicate the results during project reviews.

---

# Technical Skills Applied

During Week 3, the following concepts and technologies were explored and applied:

* Q-Learning
* Reinforcement Learning
* Epsilon-Greedy Strategy
* Exploration vs Exploitation
* Q-Value Updates
* RL Agent Development
* RL Environment Integration
* Model Training
* Revenue Optimization
* Baseline Comparison
* Performance Evaluation
* Data Analysis
* Python Programming
* Matplotlib/Visualization
* Git and GitHub Workflow

---

# Key Learnings

Throughout Week 3, I gained a better understanding of:

* How an RL agent learns from rewards.
* How Q-values represent the expected usefulness of actions.
* How exploration and exploitation affect learning.
* How an RL agent can learn pricing strategies without manually defining every pricing rule.
* How to connect an RL agent with a custom environment.
* How training episodes influence agent behavior.
* How to evaluate an RL pricing strategy using business KPIs.
* The importance of comparing RL performance with baseline approaches.

---

# Challenges Faced

The major challenges during Week 3 included:

* Understanding the Q-Learning update process.
* Selecting appropriate learning parameters.
* Balancing exploration and exploitation.
* Integrating the agent with the existing environment.
* Ensuring the agent receives valid states and actions.
* Interpreting reward and revenue changes during training.
* Comparing RL performance fairly against baseline pricing strategies.

---

# Outcome

By the end of Week 3, the **Reinforcement Learning pricing agent and training workflow** were developed and connected with the Dynamic Pricing environment.

The agent can interact with the environment, select pricing actions, receive rewards, update its learned Q-values, and gradually improve its pricing decisions through repeated training episodes.

The evaluation and visualization work also provides a foundation for comparing the RL strategy with traditional pricing approaches.

---

# Next Steps

During the upcoming week, the focus will shift towards extending the project with **additional business-oriented features** and improving the interpretability and practical usability of the pricing system.

Planned work includes:

* Implementing a **Human-in-the-Loop Learning** feature.
* Allowing a revenue manager to review AI pricing recommendations.
* Recording manager decisions and feedback.
* Learning manager pricing preferences from historical feedback.
* Developing a **Digital Twin Market Simulator**.
* Simulating different market scenarios such as festivals, holidays, conferences, and weather conditions.
* Comparing RL pricing with alternative pricing strategies.
* Performing risk analysis and generating business recommendations.
* Preparing additional-feature outputs and reports.

---

# Summary

| Task                    | Work                                             | Status    |
| ----------------------- | ------------------------------------------------ | --------- |
| Q-Learning Study        | Studied Q-Learning and RL pricing concepts       | Completed |
| RL Agent                | Implemented Q-Learning pricing agent             | Completed |
| Environment Integration | Connected agent with Dynamic Pricing environment | Completed |
| Training                | Implemented episode-based RL training            | Completed |
| Baseline Evaluation     | Compared RL with traditional pricing strategies  | Completed |
| Performance Analysis    | Analyzed reward, revenue and pricing behavior    | Completed |
| Visualization           | Prepared training and performance visualizations | Completed |

---

**Week 3 Status: Successfully Completed**
