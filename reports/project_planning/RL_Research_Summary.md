# Reinforcement Learning Research — Issue #5

**Project:** Dynamic Pricing using Reinforcement Learning
**Prepared by:** Yogeshwaran
**Task:** Study Reinforcement Learning basics

---

## 1. What is Reinforcement Learning (RL)?

Reinforcement Learning is a type of machine learning where a computer program (called an **Agent**) learns to make good decisions by **trial and error**.

The agent performs an action, observes what happens, and receives a **reward** (positive or negative). Over time, the agent learns which actions lead to the best rewards and improves its decision-making — similar to how a person learns a game by playing it repeatedly.

In our project, the agent will learn how to **price tickets/rooms/seats** so that total revenue is maximized over a selling season.

---

## 2. Difference Between ML and RL

| Aspect | Traditional Machine Learning | Reinforcement Learning |
|---|---|---|
| Learns from | Labeled historical data (input → correct output) | Trial and error, through interaction |
| Feedback | Given upfront (correct answers known) | Received after taking an action (reward) |
| Goal | Predict a value or class correctly | Maximize long-term cumulative reward |
| Example | Predict house price from features | Learn best price to set at each moment |
| Data dependency | Needs a fixed dataset | Generates its own experience by interacting with an environment |

**Simple way to remember:** Traditional ML predicts an answer from existing data. RL *discovers* a strategy (policy) by acting and learning from outcomes.

---

## 3. Core Concepts of RL

### Agent
The decision-maker. In our project, the agent is the **pricing algorithm** that decides what price to set.

### Environment
The world the agent interacts with. In our project, this is the **simulated market** — customers, competitor prices, inventory, and time remaining.

### State (S)
A snapshot of the current situation the agent sees before deciding. For our project:
Example: `[60 tickets left, 15 days remaining]`

### Action (A)
The decision the agent makes at a given state. For our project:

Example: Set price to ₹400

### Reward (R)
The feedback the agent receives after taking an action. For our project:

Example: 8 tickets sold at ₹400 → Reward = ₹3200

### Policy (π)
The agent's strategy — a rule that maps states to actions. The goal of training is to find the **best policy**, i.e., the pricing rule that earns the most total revenue.

### Episode
One complete run of the environment from start to finish. In our project, one episode = **one full selling season** (e.g., 30 days, from full inventory to season end).

---

## 4. RL Workflow (Simple Flow)
See accompanying diagram: `rl_workflow_diagram.svg`

---

## 5. How This Applies to Our Project (Dynamic Pricing)

| RL Concept | Our Project's Version |
|---|---|
| Agent | Pricing engine |
| Environment | Simulated ticket/hotel booking market |
| State | Inventory left + days remaining |
| Action | Price to set today |
| Reward | Revenue earned today |
| Episode | One full booking season |
| Policy | The learned pricing strategy |

The objective: train the agent across thousands of simulated episodes so it learns **when to hold prices high** (early, plenty of inventory) and **when to drop prices** (near deadline, to clear remaining stock) — maximizing total revenue better than fixed-price or simple discount rules.

---

## 6. Summary

Reinforcement Learning lets us build a pricing agent that **learns by simulation** rather than following fixed rules. By repeatedly playing out selling seasons, adjusting prices, and observing revenue outcomes, the agent gradually discovers an optimal pricing policy — the foundation for the Q-Learning and Deep Q-Network (DQN) work planned in Weeks 2 and 3.
