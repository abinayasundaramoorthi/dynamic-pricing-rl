# Week 2 Sprint Report

## Project

**Travel & Hospitality – Reinforcement Learning for Dynamic Pricing**

---

# Sprint Information

| Sprint | Week 2 |
|---------|--------|
| Duration | 5 Days |
| Sprint Goal | Develop and integrate the baseline Reinforcement Learning pipeline using a custom Gymnasium pricing environment and Q-Learning agent. |

---

# Sprint Objectives

The primary objective of Week 2 was to transition from environment development to reinforcement learning implementation by integrating the pricing environment with a baseline Q-Learning agent. The sprint focused on establishing a complete training workflow, evaluating agent performance, and preparing the project for future algorithm enhancements.

---

# Completed Work

## 1. RL Training Pipeline

Completed the initial reinforcement learning training workflow.

Completed:

- Environment initialization
- Training configuration
- Training entry point
- Episode execution
- Training workflow integration

---

## 2. Baseline Q-Learning Agent

Implemented the baseline Q-Learning agent.

Completed:

- Q-table initialization
- Epsilon-greedy action selection
- Q-value update rule
- Learning loop
- Hyperparameter configuration

---

## 3. Environment Integration

Integrated the custom Gymnasium pricing environment with the RL agent.

Verified:

- reset()
- step()
- reward calculation
- state transitions
- action execution

---

## 4. Training Metrics & Logging

Implemented utilities for monitoring training performance.

Captured:

- Episode reward
- Revenue
- Inventory utilization
- Episode statistics

---

## 5. Evaluation

Evaluated the baseline agent performance.

Completed:

- Random policy comparison
- Baseline metrics
- Performance summary
- Training observations

---

## 6. Documentation

Updated project documentation including:

- README
- Training configuration
- Hyperparameter documentation
- Sprint summary

---

# Repository Components Completed

```text
agents/
    q_learning_agent.py

training/
    train_agent.py

configs/
    training_config.py

pricing_env/
    pricing_env.py
    state.py
    action_space.py
    reward.py
    demand_simulator.py

evaluation/
    training_logs.csv
    performance_summary.md

utils/
    logger.py

dashboard/
    training_dashboard.py
```

---

# Testing Summary

The following components were verified during Week 2.

| Component | Status |
|-----------|--------|
| Pricing Environment | ✅ Completed |
| State Representation | ✅ Completed |
| Action Space | ✅ Completed |
| Reward Function | ✅ Completed |
| Demand Simulator | ✅ Completed |
| Training Pipeline | ✅ Completed |
| Q-Learning Agent | ✅ Completed |
| Logging | ✅ Completed |
| Evaluation | ✅ Completed |

---

# Sprint Outcome

At the end of Week 2, the project successfully supports:

- Custom Gymnasium pricing environment
- Baseline Q-Learning agent
- End-to-end training workflow
- Reward computation
- State transitions
- Training metrics collection
- Performance evaluation
- Baseline comparison

The project is now ready for advanced reinforcement learning development.

---

# Challenges Encountered

- Designing an effective reward function.
- Selecting suitable hyperparameters.
- Integrating environment components.
- Validating training behaviour.
- Organizing modular project architecture.

---

# Lessons Learned

During this sprint, the team gained practical experience in:

- Reinforcement Learning fundamentals
- Gymnasium environment development
- Q-Learning implementation
- Modular software architecture
- GitHub collaboration using Issues, Pull Requests, and Kanban workflows
- Sprint planning and task management

---

# Week 3 Goals

The objectives for the next sprint include:

- Improve Q-Learning performance
- Implement Deep Q-Network (DQN)
- Optimize reward function
- Tune hyperparameters
- Improve demand modelling
- Expand evaluation metrics
- Enhance dashboard visualizations

---

# Sprint Status

**Sprint:** Week 2

**Status:** ✅ Successfully Completed

The repository now contains a functional baseline reinforcement learning system and is ready to begin advanced RL development in Week 3.