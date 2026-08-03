# Week 3 Report — Yogeshwaran

**Project:** Dynamic Pricing using Reinforcement Learning
**Branch:** feature/yogesh
**Issues Completed:** #63, #71, #75, #80, #84

---

## Overview

Week 3 focused on building a complete Deep Q-Network (DQN) implementation from scratch — the neural network architecture, experience replay, and target network stabilization — then tying all three together into a full training loop and evaluating it against the Week 2 Q-Learning baseline.

---

## Issue #63 — Implement Deep Q-Network Neural Network Model

**Objective:** Replace the Q-table with a neural network capable of approximating Q-values for continuous and high-dimensional state spaces.

**Work Completed:**
- Designed and implemented a feedforward neural network (`QNetwork`) with an input layer, two hidden layers (128 units each, ReLU activation), and an output layer producing one Q-value per possible price action
- Implemented forward propagation and principled weight initialization (He/Kaiming for hidden layers, Xavier for the output layer)
- Documented the reasoning behind every architecture choice (layer sizes, activation function, no output activation)
- Verified the network initializes correctly and produces valid Q-value outputs for both single states and batches

**Deliverables:**
- `agents/dqn_network.py`
- `agents/dqn_agent.py` (initial inference-only version)

---

## Issue #71 — Implement Experience Replay Buffer

**Objective:** Develop the replay memory module that stores and samples experiences for DQN training.

**Work Completed:**
- Implemented a fixed-capacity `ReplayBuffer` class storing (state, action, reward, next_state, done) experiences
- Implemented random mini-batch sampling, breaking the harmful correlation between consecutive training samples
- Configured buffer capacity with automatic eviction of the oldest experiences once full
- Verified storage, sampling, and capacity limits all work correctly through automated tests

**Deliverables:**
- `agents/replay_buffer.py`

---

## Issue #75 — Implement Target Network for Deep Q-Learning

**Objective:** Develop the Target Network to stabilize DQN training by periodically synchronizing model parameters.

**Work Completed:**
- Implemented `TargetNetworkManager`, maintaining a separate, periodically-updated copy of the online network
- Implemented configurable synchronization frequency and a validation method to confirm sync correctness
- Verified the target network stays frozen between updates and correctly synchronizes at the configured interval

**Deliverables:**
- `agents/target_network.py`

---

## Issue #80 — Evaluate DQN Performance Against Q-Learning

**Objective:** Compare the performance of the Deep Q-Network with the previously implemented Q-Learning agent using predefined evaluation metrics.

**Work Completed:**
- Built the missing full DQN training loop, connecting the network (#63), replay buffer (#71), and target network (#75) into one working trainable agent
- Trained both a Q-Learning agent and a DQN agent under matching conditions
- Compared average reward, revenue, inventory utilization, training stability, and episode performance
- **Result:** DQN outperformed Q-Learning by +6.1% average reward in this evaluation

**Deliverables:**
- `agents/dqn_agent.py` (full trainable version)
- `evaluation/dqn_vs_qlearning.md`
- `evaluation/comparison_metrics.csv`

**Note:** This issue also required diagnosing and fixing recurring merge conflict corruption in `action_space.py` and `reward.py`, traced to an unresolved GitHub Pull Request.

---

## Issue #84 — Prepare Deep Reinforcement Learning Performance Report

**Objective:** Summarize DQN performance using quantitative metrics and visual analysis for project documentation.

**Work Completed:**
- Added per-episode training loss tracking to the DQN agent
- Generated a 4-panel summary chart: episode reward progression, training loss curve, reward/revenue comparison, and inventory utilization comparison
- Documented all required metrics (average reward, episode reward, training loss, revenue, inventory utilization) with comparison to Q-Learning
- **Result:** DQN outperformed Q-Learning by +21.8% average reward in this evaluation run, with training loss confirmed decreasing over time (validating stable learning)

**Deliverables:**
- `dashboard/performance_summary.png`
- `evaluation/dqn_performance_report.md`

---

## Key Technical Achievement

By the end of Week 3, the project had two fully independent, trained, and compared RL approaches:
- **Q-Learning** (Week 2) — simple, interpretable, table-based
- **DQN** (Week 3) — neural network-based, capable of generalizing to unseen states

Both approaches were evaluated fairly under matching conditions across two separate comparison runs (#80, #84), with DQN showing a consistent — though variable in magnitude — performance advantage.

---

## Summary

Week 3 delivered a complete, working DQN implementation from first principles — network architecture, experience replay, and target network stabilization — culminating in two independent evaluations confirming DQN's advantage over the Week 2 Q-Learning baseline.
