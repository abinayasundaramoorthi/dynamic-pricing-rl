# Week 1 Progress Report

**Project Title:** Reinforcement Learning for Dynamic Pricing   
---

# Overview

During the first week of the project, I focused on understanding the fundamentals of **Dynamic Pricing**, studying existing industry practices, and contributing to the development of the Reinforcement Learning (RL) environment. The work completed this week involved literature review, implementation of core environment modules, enhancement of the reward mechanism, and preparation of technical documentation to support future development.

---

# Task 1: Literature Survey on Dynamic Pricing (#4)

## Objective

To understand the concept of dynamic pricing, its applications, business benefits, challenges, and the role of Artificial Intelligence in pricing optimization.

## Work Completed

- Studied the fundamentals of Dynamic Pricing.
- Identified real-world applications across multiple industries.
- Researched the advantages and limitations of dynamic pricing.
- Explored how Artificial Intelligence and Machine Learning are used in pricing strategies.
- Collected and documented references from research papers, industry articles, and technical resources.

## Deliverables

- Dynamic Pricing Literature Survey Report
- References from academic and industry sources

---

# Task 2: Existing Dynamic Pricing Systems Study (#11)

## Objective

To analyze how leading companies implement dynamic pricing in real-world business environments.

## Work Completed

Conducted case studies on the following industries:

- Uber
- Amazon
- Airline Industry
- Hotel Industry
- Food Delivery Platforms

For each case study, the following aspects were analyzed:

- Pricing factors
- Pricing strategy
- Business objective
- Revenue optimization approach

Prepared a comparative analysis highlighting similarities and differences among various dynamic pricing systems.

## Deliverables

- Industry Case Studies Report
- Comparative Analysis Table
- Business Insights

---

# Task 3: Implement Action Space Module (#16)

## Objective

To define the pricing actions available to the Reinforcement Learning agent.

## Work Completed

Implemented the **Action Space Module** responsible for updating prices based on the agent's selected action.

Implemented five discrete pricing actions:

| Action | Description |
|---------|-------------|
| 0 | Decrease price by 10% |
| 1 | Decrease price by 5% |
| 2 | Keep current price |
| 3 | Increase price by 5% |
| 4 | Increase price by 10% |

Developed the helper function:

```python
apply_action(current_price, action)
```

which updates the current price according to the selected pricing action.

## Deliverables

- `pricing_env/action_space.py`

---

# Task 4: Enhance Reward Function (#30)

## Objective

To improve the reward calculation by incorporating realistic business objectives into the Reinforcement Learning environment.

## Work Completed

Designed a reward function that considers multiple business factors instead of relying solely on revenue.

The reward function includes:

- Revenue generated
- Remaining inventory
- Overstock penalty
- Understock penalty
- Invalid pricing penalty

Documented the reward equation and explained how each component influences the learning process.

## Deliverables

- `pricing_env/reward.py`
- `reports/project_planning/reward_function_design.md`

---

# Task 5: Environment Documentation (#34)

## Objective

To document the complete Reinforcement Learning environment for future development and integration.

## Work Completed

Prepared detailed documentation describing:

- Environment architecture
- State Space
- Action Space
- Reward Function
- Episode Flow
- Environment Lifecycle

Included an architecture diagram illustrating the interaction between the RL agent and the environment.

## Deliverables

- `reports/project_planning/environment_design.md`

---

# Technical Skills Applied

During this week, the following concepts and technologies were explored and applied:

- Reinforcement Learning Fundamentals
- Markov Decision Process (MDP)
- Dynamic Pricing Concepts
- Revenue Management
- Reward Engineering
- Python Programming
- Modular Software Design
- Technical Documentation
- Markdown Documentation
- Git and GitHub Workflow

---

# Key Learnings

Throughout the first week, I gained a strong understanding of:

- The importance of Dynamic Pricing in travel and hospitality.
- How Reinforcement Learning differs from traditional Machine Learning.
- The role of states, actions, rewards, and environment in RL.
- Designing modular components for an RL environment.
- Business-oriented reward engineering.
- Writing technical documentation for collaborative software development.

---

# Challenges Faced

- Understanding Reinforcement Learning concepts and terminology.
- Designing a reward function that balances multiple business objectives.
- Understanding how different environment components interact within an RL framework.
- Learning Git workflows for collaborative development.

---

# Outcome

By the end of Week 1, I successfully completed the assigned research, implementation, and documentation tasks. The work contributes to the foundation of the Dynamic Pricing Reinforcement Learning project by establishing essential environment components and supporting documentation for future RL model integration.

---

# Next Steps

During the upcoming weeks, the focus will shift towards:

- Building the complete Reinforcement Learning environment.
- Integrating all environment modules.
- Implementing Q-Learning.
- Developing a Deep Q-Network (DQN).
- Training and evaluating the pricing agent.
- Visualizing learning performance and pricing behavior.

---

# Summary

| Task ID | Task | Status |
|----------|------|--------|
| #4 | Literature Survey on Dynamic Pricing | Completed |
| #11 | Existing Dynamic Pricing Systems Study | Completed |
| #16 | Implement Action Space Module | Completed |
| #30 | Enhance Reward Function | Completed |
| #34 | Environment Documentation | Completed |

---

**Week 1 Status:** Successfully Completed