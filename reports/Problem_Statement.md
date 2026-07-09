# Travel & Hospitality — Reinforcement Learning for Dynamic Pricing
## Project Design Document (Enterprise Edition)

**Document Type:** Project Charter & Technical Design Specification
**Domain:** Travel, Airline & Hospitality Revenue Management
**Prepared For:** Engineering Team, Data Science Team, Business & Executive Stakeholders
**Status:** Approved for Implementation — Sprint 1 Kickoff
**Version:** 2.0 (Enterprise-Enhanced)

---

## Table of Contents

1. Executive Summary
2. Business Background
3. Problem Statement
4. Business Objectives
5. Project Objectives
6. Project Scope
7. Stakeholders
8. Business KPIs
9. Expected Deliverables
10. Reinforcement Learning Formulation
11. State Space
12. Action Space
13. Reward Design
14. Project Constraints
15. Non-Functional Requirements
16. Assumptions
17. Risks
18. Success Metrics
19. Success Criteria
20. Week 1 Project Timeline
21. Week 2 Milestones
22. Repository Structure
23. Project Architecture
24. References

---

## 1. Executive Summary

Airlines, hotels, and other travel and hospitality operators sell a fixed, perishable inventory over a finite selling window. Every seat or room left unsold at departure or check-in represents permanently lost revenue, while every unit sold too cheaply, too early, represents value left on the table. Traditional fixed and rule-based pricing systems cannot adapt fast enough or reason far enough ahead to resolve this tension optimally.

This project designs and implements a **Reinforcement Learning (RL) agent** that learns an optimal, adaptive pricing policy through interaction with a simulated booking market. The agent is trained to maximize **cumulative season revenue**, not individual transaction value, directly addressing the core weakness of legacy pricing approaches. The initiative proceeds in two phases within the first sprint: (1) formulating the problem as a Markov Decision Process and building a custom Gym/Gymnasium simulation environment, and (2) training and evaluating tabular Q-Learning and Deep Q-Network (DQN) agents against fixed and rule-based baselines.

The expected outcome is a validated, documented, and version-controlled prototype demonstrating measurable revenue uplift over legacy pricing strategies in simulation — a decision-support asset that revenue management stakeholders can evaluate for further investment toward production deployment.

---

## 2. Business Background

The travel and hospitality sector operates under a distinctive economic constraint: **inventory is both finite and perishable**. Unlike physical retail goods, an unsold airline seat or hotel room cannot be warehoused and sold later — its value expires the moment the flight departs or the date passes. This has made **Revenue Management (RM)** one of the most mature applied optimization disciplines in the industry, tracing back to airline deregulation in the late 1970s and the development of statistical forecasting and bid-price control systems still in wide use today.

While these legacy RM systems represented a major advance over flat pricing, they are largely built on **periodic statistical forecasting refreshed on daily or weekly cycles**, combined with manually authored business rules for adjusting price tiers. This leaves a structural gap: none of these systems *learn*, in the machine learning sense, an end-to-end policy that directly optimizes cumulative revenue outcomes from experience. Meanwhile, adjacent industries — most notably ride-sharing (e.g., Uber's surge pricing) and e-commerce — have moved toward algorithmically-learned, continuously adapting pricing, establishing a competitive and technological expectation that travel and hospitality pricing will follow a similar trajectory.

This project is scoped as the **first phase** of exploring that trajectory: a controlled, simulation-based proof of concept that establishes whether an RL-based approach can outperform the pricing logic broadly representative of current industry practice, before any investment is made in integrating real transactional data or production systems.

---

## 3. Problem Statement

Selling finite inventory over a limited time period is a sequential optimization problem under uncertainty, and current pricing approaches fail to solve it well:

- **Fixed pricing** applies one price for the entire season, ignoring both demand fluctuation and the shrinking time-to-deadline, leaving revenue uncaptured during high-demand periods and inventory unsold during low-demand periods.
- **Rule-based pricing** ("discount 10% if fewer than 5 days remain") reacts to the current state but cannot reason about the long-term, cumulative consequence of a pricing decision on total season revenue, and does not scale as more state variables are introduced.
- **Demand is uncertain and dynamic**, shaped by seasonality, competitor behavior, and idiosyncratic customer willingness to pay, none of which is well captured by static logic.

The consequence is a persistent tension between two failure modes: **inventory spoilage** (unsold seats/rooms at the deadline — permanent, unrecoverable revenue loss) and **revenue leakage** (inventory sold too early at prices below what the market would have borne). Resolving this tension requires a system that reasons about pricing as a sequence of interdependent decisions across the entire selling horizon — precisely the class of problem Reinforcement Learning is designed to solve, since an RL agent optimizes cumulative long-term reward rather than any single transaction outcome.

---

## 4. Business Objectives

- Demonstrate measurable simulated revenue uplift from an RL-based pricing policy relative to fixed and rule-based pricing baselines.
- Reduce simulated inventory spoilage (unsold units remaining at episode termination).
- Reduce reliance on manual, reactive, last-minute discounting.
- Produce a decision-support prototype suitable for evaluation by revenue management and executive stakeholders as a candidate for further investment.

---

## 5. Project Objectives

### 5.1 Primary Objective
Design, build, and evaluate an RL agent that learns an optimal dynamic pricing policy for finite, perishable inventory, maximizing cumulative revenue per selling season relative to traditional baselines.

### 5.2 Secondary Objectives
- Produce a reusable, extensible simulation environment adaptable to other perishable-inventory pricing domains (event ticketing, car rental, cargo capacity).
- Establish a rigorous, reproducible evaluation framework comparing multiple algorithmic approaches.

### 5.3 Technical Objectives
- Formalize the pricing problem as a Markov Decision Process with a well-specified state space, action space, and reward function.
- Implement a stochastic demand simulator responsive to price, inventory, and time-to-deadline.
- Implement and train tabular Q-Learning and Deep Q-Network agents.

### 5.4 Research Objectives
- Empirically compare the sample efficiency, convergence stability, and final policy quality of tabular Q-Learning versus DQN.
- Validate that the learned policy reflects economically sound behavior (e.g., discounting near the deadline, holding price when inventory is scarce) through qualitative policy inspection.

### 5.5 Operational Objectives
- Deliver a version-controlled, documented GitHub repository suitable for handoff to future engineering iterations.
- Ensure modularity sufficient to plug in real historical booking data in a subsequent phase.

---

## 6. Project Scope

### 6.1 In Scope
- Custom Gym/Gymnasium environment simulating single-resource, finite-inventory, finite-horizon pricing.
- Stochastic demand model driven by price, remaining inventory, and days-until-deadline.
- Baseline heuristic pricing strategies (fixed price, time-based discounting, rule-based).
- Tabular Q-Learning and Deep Q-Network agents.
- Policy evaluation across simulated episodes, with statistical comparison against baselines.
- Visualization of price trajectories, revenue distributions, and learning curves.
- Business-oriented summary reporting and a lightweight results dashboard.

### 6.2 Out of Scope (Phase 1)
- Integration with live/production booking systems or real transactional data.
- Multi-resource, network-level revenue management (multiple flights/room types simultaneously).
- Multi-agent/competitive pricing simulation.
- Real-time deployment, API productionization, or online learning in a live environment.
- Regulatory or fare-class tariff-filing logic.

### 6.3 Future Enhancements (Phase 2+)
- Continuous action spaces via policy-gradient methods (PPO, SAC).
- Competitor price as an observable/simulated state variable.
- Network-level, multi-resource optimization.
- Calibration against real historical booking curves.
- Offline/batch RL trained directly from logged historical pricing data.

---

## 7. Stakeholders

| Stakeholder | Responsibilities |
|---|---|
| **Revenue Manager** | Defines pricing guardrails and safety bounds; reviews the RL agent's pricing strategy on a monitoring dashboard; approves policy changes before any production consideration |
| **Data Scientist** | Designs and tunes the reward function and state representation; evaluates model convergence and policy quality; guards against reward hacking |
| **Business Analyst** | Collects and validates business requirements; defines and validates KPIs; benchmarks RL performance against current pricing strategies |
| **ML Engineer** | Builds the Gym/Gymnasium environment; implements and trains RL agents; owns the evaluation and (future) deployment pipeline |
| **Project Manager** | Owns sprint planning, timeline, and deliverable tracking; coordinates across technical and business stakeholders |

---

## 8. Business KPIs

| KPI | Description |
|---|---|
| Total Revenue | Aggregate revenue generated per simulated selling season |
| Average Selling Price | Mean transaction price across all sales in a season |
| Occupancy / Sell-Through Rate | Percentage of total inventory sold by the terminal step |
| Inventory Utilization | Efficiency of inventory depletion relative to an ideal pacing curve |
| Average Booking Lead Time | Mean time-to-deadline at which sales occur |
| Customer Conversion Rate | Proportion of demand opportunities converted into sales at the offered price |
| Revenue Per Available Room/Seat (RevPAR-equivalent) | Total revenue normalized by total available inventory — the industry-standard yield metric |

---

## 9. Expected Deliverables

Upon completion, the project will produce:

- **Custom Gym Environment** — a fully functional `gym.Env` subclass simulating the booking process, with configurable inventory size, horizon length, and demand parameters.
- **RL Agents** — trained tabular Q-Learning and Deep Q-Network agents, with saved model weights / Q-tables committed to `agents/checkpoints/` and pushed to GitHub, as required by the internship brief.
- **Pricing Policy** — the learned state-to-price mapping, inspectable independent of training code.
- **Evaluation Results** — quantitative comparison (mean revenue, variance, spoilage rate) between RL agents and baseline strategies across a large number of simulated episodes.
- **Visualizations** — training/learning curves, price-trajectory plots, revenue distribution comparisons.
- **Performance Reports** — business-relevant summaries of technical results.
- **GitHub Repository** — clean, modular, version-controlled codebase.
- **Documentation** — this design document plus supporting technical documentation.

---

## 10. Reinforcement Learning Formulation

Dynamic pricing under a finite inventory and finite selling horizon is naturally modeled as a **Markov Decision Process (MDP)**, because the outcome of each pricing decision depends only on the current state — not the full history of how that state was reached — and each decision affects the state available for all subsequent decisions. This Markov property is what enables solution via Q-Learning-family methods, and is precisely why a supervised, single-shot prediction approach is insufficient: pricing decisions are coupled across time through the shared, depleting inventory constraint.

- **Environment** — the simulated booking market: a finite-inventory, finite-horizon system that accepts a price action each time step and returns a stochastic sale/no-sale outcome, updated inventory, and reward.
- **Agent** — the RL algorithm (tabular Q-Learning or DQN) selecting a pricing action at each time step.
- **Episode** — one complete simulated selling season, from inventory opening to the terminal deadline.
- **State (s)** — the observable information available at a given time step (Section 11).
- **Action (a)** — the pricing decision at each time step (Section 12).
- **Reward (r)** — the scalar feedback signal after each action (Section 13).
- **Policy (π)** — the learned mapping from state to pricing action; the final project deliverable.
- **Transition Function (P)** — the stochastic rule governing state evolution, driven primarily by the demand model and the deterministic decrement of time-remaining.
- **Discount Factor (γ)** — set close to 1 (e.g., 0.95–0.99) so the agent weighs the entire remaining season, not just the immediate next sale, avoiding short-sighted pricing.
- **Terminal Condition** — episode ends when time-remaining reaches zero or all inventory is sold, whichever occurs first; unsold inventory at termination yields zero further revenue.

---

## 11. State Space

| Variable | Description | Example Value | Why It Matters |
|---|---|---|---|
| Remaining Inventory | Number of unsold units left | 42 of 150 | Determines urgency to sell versus room to hold price high |
| Days Until Departure/Check-in | Countdown of time steps remaining | 12 days | Governs discounting urgency as the deadline approaches |
| Current Price | Price currently in effect | $245 | Provides continuity for the next pricing decision |
| Demand Level | Current market demand intensity | High / Medium / Low | Distinguishes "hold price" from "discount" conditions |
| Booking Rate | Recent rate of sales | 3 units/day | Leading indicator of pacing versus ideal sell-through curve |
| Seasonality | Calendar-driven demand pattern | Peak season | Demand baseline shifts substantially with seasonality |
| Competitor Price *(optional, Phase 2)* | Estimated competitor price | $260 | Customers are price-comparison sensitive |
| Customer Segment *(optional, Phase 2)* | Expected demand mix | 60% leisure | Segments differ in price sensitivity and lead time |
| Market Trend *(optional, Phase 2)* | Directional demand signal | Trending up | Helps anticipate near-term shifts |

The Phase 1 **core state vector** is `[remaining_inventory, days_until_departure]`, extended with `current_price`, `demand_level`, and `seasonality` as the environment matures. Tabular Q-Learning requires discretized bins; DQN accepts the continuous/normalized representation directly — a practical justification for progressing from Q-Learning to DQN as state complexity grows.

---

## 12. Action Space

Two formulations were considered: **discrete** pricing actions (a fixed menu of price levels) and **continuous** pricing actions (a real-valued price within a bounded range).

**Chosen approach: Discrete Action Space (Phase 1).**

**Justification:** Discrete actions are compatible with tabular Q-Learning, which requires an enumerable action set to construct a Q-table, keeping the early implementation tractable. They also mirror how travel pricing is often operationalized in practice (fare classes, rate tiers), making the learned policy auditable by a revenue manager. Discretization also simplifies debugging of the reward function and demand model in early sprints. Continuous action spaces (via PPO/SAC) are an explicit Phase 2 enhancement once the discrete formulation is validated.

| Action ID | Description | Price Adjustment | Resulting Price (Base = $200) |
|---|---|---|---|
| 0 | Deep discount | −20% | $160 |
| 1 | Moderate discount | −10% | $180 |
| 2 | Slight discount | −5% | $190 |
| 3 | Hold current price | 0% | $200 |
| 4 | Slight increase | +5% | $210 |
| 5 | Moderate increase | +10% | $220 |
| 6 | Premium increase | +20% | $240 |

---

## 13. Reward Design

Reward design is the single most consequential decision in this project, because the agent will optimize exactly what the reward function measures — not what the designers intended it to measure. A poorly specified reward can produce policies that score well but are economically unsound (e.g., discounting aggressively to guarantee sales, destroying margin over the full season).

**Proposed initial reward function**, evaluated at each time step *t*:

```
r_t = (price_t × sale_t)
      − λ_1 · (unsold_penalty if terminal and inventory_remaining > 0)
      − λ_2 · (discount_depth_t)^2 · sale_t
      + λ_3 · inventory_balance_bonus_t
```

- **`price_t × sale_t`** — primary revenue term; `sale_t ∈ {0,1}` indicates whether a unit sold at the chosen price.
- **Unsold penalty** — applied at the terminal step, proportional to unsold units, reflecting that spoiled inventory is permanently lost revenue; discourages holding price too high for too long.
- **Over-discounting penalty** — a convex penalty on discount depth when a sale occurs, discouraging default maximal discounting purely to guarantee conversions.
- **Inventory balancing bonus** — small shaping reward for tracking a healthy sell-through pace, providing denser feedback than the sparse terminal signal alone.
- **`λ_1, λ_2, λ_3`** — tunable weighting coefficients, calibrated empirically in Weeks 1–2.

This design aligns the agent's immediate incentive with the business objective, forces long-horizon credit assignment (via the terminal spoilage penalty and high discount factor), and guards against the well-known RL failure mode of reward hacking via trivial, unsound policies.

---

## 14. Project Constraints

- Only simulated data is used in Phase 1; no live transactional feed.
- Limited computational resources — training must be feasible on standard development hardware within the sprint timeline.
- Discrete pricing actions only (Phase 1); continuous control deferred.
- No competitor interaction modeling in Phase 1.
- Inventory is finite and non-replenishable within an episode.
- The selling horizon (time-to-deadline) is finite and fixed per episode.

---

## 15. Non-Functional Requirements

| Requirement | Description |
|---|---|
| Scalability | Environment and agent code should support larger state/action spaces without architectural rework |
| Maintainability | Modular, well-documented code that can be extended by engineers not involved in the original build |
| Modularity | Clear separation between environment, agents, training, and evaluation logic |
| Reproducibility | Fixed random seeds and logged configurations to ensure experiments can be reliably re-run |
| Interpretability | Policy behavior must be inspectable and explainable to non-technical stakeholders (e.g., via price-trajectory plots) |
| Training Stability | Learning curves should show consistent, non-divergent improvement across runs |
| Extensibility | Architecture should accommodate future algorithms (PPO/SAC) and additional state variables with minimal refactoring |

---

## 16. Assumptions

| Assumption | Reason |
|---|---|
| Demand follows a stochastic distribution responsive to price, inventory, and time | Simulates realistic customer purchase uncertainty |
| Inventory is fixed at the start of each episode | Reflects the airline/hotel finite-capacity constraint |
| Single product / single resource is modeled in Phase 1 | Keeps the MVP tractable before network-level scaling |
| Synthetic demand model reasonably approximates real-world booking behavior | Necessary to validate RL methodology before real-data integration |
| Discretized state and action spaces are sufficiently granular for Phase 1 | Balances policy quality against tabular Q-Learning tractability |

---

## 17. Risks

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| Reward hacking (agent maximizes reward via unsound pricing behavior) | Medium | High | Continuously inspect qualitative policy behavior (price trajectories) alongside quantitative reward |
| Sparse reward / slow convergence | Medium | Medium | Use reward shaping (inventory balancing bonus) to densify feedback |
| Unrealistic demand simulator | Medium | Medium | Explicitly document simulator assumptions; flag real-data calibration as Phase 2 |
| Q-table scalability limits | High (by design) | Medium | Introduce DQN specifically at the point tabular methods become impractical |
| DQN training instability | Medium | Medium | Use experience replay and target networks to stabilize training |

---

## 18. Success Metrics

### 18.1 Business Metrics
- Total Revenue per Episode (Season) — primary business KPI.
- Occupancy / Sell-Through Rate — percentage of total inventory sold by the terminal step.
- Inventory Utilization — efficiency of depletion relative to ideal pacing.

### 18.2 Machine Learning Metrics
- Average Reward per Episode, tracked over training to confirm an improving policy.
- Policy improvement over baselines — percentage revenue uplift versus fixed-price and rule-based agents on identical evaluation episodes.

### 18.3 Reinforcement Learning Metrics
- Episode reward trend — smoothed reward curve across training.
- Learning stability — variance of reward across episodes/runs.
- Convergence — the point at which policy/Q-values stabilize.

### 18.4 Engineering Metrics
- Training time — wall-clock time to convergence for each agent.
- Inference time — latency to produce a pricing action given a state.

---

## 19. Success Criteria

| Metric | Target |
|---|---|
| Revenue Improvement over Baselines | > 10% |
| Episode Reward Trend | Consistently increasing |
| Sell-Through Rate | > 95% |
| Inventory Spoilage | < 5% |
| Policy Convergence | Stable (low variance across final training runs) |
| Training Budget | Sufficient training episodes per agent to reach reward-curve convergence (tracked, not fixed in advance) |
| Evaluation Protocol | 1,000 simulated booking seasons per agent, per the internship brief's Week 4 evaluation requirement (DQN vs. naive baselines) |

---

## 20. Week 1 Project Timeline

| Day | Task | Description | Expected Output | GitHub Deliverable | Est. Hours |
|---|---|---|---|---|---|
| Day 1 | Business Understanding & Problem Definition | Document business problem, objectives, scope, stakeholders, and KPIs | `Problem_Statement.md` | `reports/Problem_Statement.md` | 4 |
| Day 2 | MDP Design | Formally define state, action, reward, transition, and discount factor | MDP design specification | `reports/MDP_Design.md` | 5 |
| Day 3 | Gym Environment Skeleton | Implement `gym.Env` subclass with `reset()`, `step()`, `observation_space`, `action_space` | Working environment skeleton | `pricing_env/pricing_env.py` | 6 |
| Day 4 | Demand Function Implementation | Implement stochastic demand model responsive to price, inventory, time | Demand simulation module | `pricing_env/demand_model.py` | 6 |
| Day 5 | Environment Integration & Unit Testing | Wire demand model into environment; write unit tests for state transitions and reward | Passing test suite | `tests/test_pricing_env.py` | 5 |
| Day 6 | Environment Validation Runs | Run random-agent rollouts to sanity-check environment dynamics and reward distribution | Validation plots/logs | `notebooks/env_validation.ipynb` | 4 |
| Day 7 | Documentation & Week 1 Review | Document environment design decisions and finalize Week 1 deliverables | Environment documentation | `reports/Environment_Design.md` | 3 |

---

## 21. Week 2 Milestones

| Milestone | Task | Description | Expected Output | GitHub Deliverable |
|---|---|---|---|---|
| M1 | Fixed Pricing Baseline | Implement a static single-price agent | Baseline agent + episode results | `baselines/baseline_fixed.py` |
| M2 | Time-Based Discount Baseline | Implement heuristic time/inventory-based discount rules (e.g., 10% daily discount) | Baseline agent + episode results | `baselines/baseline_timebased.py` |
| M3 | Tabular Q-Learning Implementation | Implement Q-table, epsilon-greedy exploration, Q-update rule | Q-Learning agent class | `agents/q_learning.py` |
| M4 | Q-Learning Training Run | Train agent across simulated episodes; log reward curve; save trained Q-table | Training logs, reward curve plot, trained Q-table | `training/train_q_learning.py`, `agents/checkpoints/q_table.pkl` |
| M5 | Evaluation Harness | Build standardized evaluation loop comparing all agents on identical episode seeds | Evaluation results table | `evaluation/evaluate_agents.py` |
| M6 | Comparative Analysis | Compare Q-Learning vs. baselines on revenue, occupancy, spoilage | Comparison report + plots | `reports/Week2_Comparison.md` |
| M7 | Documentation | Document Q-Learning design decisions, hyperparameters, and results | Updated technical documentation | `reports/QLearning_Design.md` |

---

## 22. Repository Structure

```
dynamic-pricing-rl/
│
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
│
├── pricing_env/            # Custom Gym/Gymnasium environment and demand simulation logic
├── baselines/              # Naive heuristic agents (fixed price, time-based / 10% daily discount)
├── agents/                 # Q-Learning and DQN agent implementations
│   └── checkpoints/        # Trained Q-tables and DQN weights, committed to GitHub per brief requirement
├── training/                # Training loops and experiment orchestration (train_q_learning.py, train_dqn.py)
├── evaluation/              # Evaluation harness — 1,000-season comparison of RL agents vs. baselines
├── configs/                 # YAML/JSON configuration files for environment and agent hyperparameters
├── plots/                   # Generated visualizations (reward curves, price-trajectory plots)
├── dashboard/                # Business-facing results dashboard
├── notebooks/                # Exploratory analysis and environment validation notebooks
├── utils/                    # Shared helper functions (logging, seeding, metrics)
├── tests/                    # Unit and integration tests
└── reports/                  # Planning documents, design specs, and business-facing reports
```

**Folder rationale:**
- `pricing_env/` isolates the MDP/simulation logic, decoupled from any specific agent implementation, so the same environment can be reused across Q-Learning, DQN, and future policy-gradient agents.
- `baselines/` is kept separate from `agents/` so the naive heuristic strategies the brief requires (fixed price, time-based discounting) are never conflated with the learned policies they are meant to be benchmarked against.
- `agents/checkpoints/` gives trained weights and Q-tables an explicit, version-controlled home — the internship brief specifically requires that "the custom Gym environment and trained weights must be pushed to GitHub."
- `training/` and `evaluation/` are split intentionally: training produces a policy, evaluation independently assesses it (including the brief's 1,000-simulated-season comparison), preventing evaluation logic from leaking into or biasing training.
- `configs/` externalizes hyperparameters from code for reproducible experimentation.
- `plots/` gives the Week 4 Matplotlib/Seaborn price-trajectory and reward-curve outputs a consistent, predictable location.
- `notebooks/` supports exploratory work and environment validation without polluting production code.
- `tests/` ensures environment correctness is verified before any agent is trained against it, since a flawed environment invalidates all downstream results.
- `reports/` centralizes planning and analytical documentation so business stakeholders have a single reference point independent of code.

---

## 23. Project Architecture

```
+---------------------------+
|      Business Problem     |
|  (Finite, Perishable      |
|   Inventory Pricing)      |
+-------------+-------------+
              |
              v
+---------------------------+
|   Gym/Gymnasium             |
|   Environment                |
|   (State, Reward, Terminal) |
+-------------+-------------+
              |
              v
+---------------------------+
|        RL Agent             |
|  (Q-Learning / DQN)          |
+-------------+-------------+
              |
              v
+---------------------------+
|         Action               |
|     (Price Selection)        |
+-------------+-------------+
              |
              v
+---------------------------+
|   Demand Simulation          |
| (Stochastic Purchase Model)  |
+-------------+-------------+
              |
              v
+---------------------------+
|          Reward              |
|  (Revenue − Penalties        |
|   + Shaping Bonus)           |
+-------------+-------------+
              |
              v
+---------------------------+
|          Learning            |
|  (Q-value / Weight Update)   |
+-------------+-------------+
              |
              v
+---------------------------+
|      Updated Policy          |
+-------------+-------------+
              |
              v
+---------------------------+
|   Revenue Maximization       |
|   (Business Outcome)         |
+---------------------------+
```

This loop repeats for every time step within an episode, and every episode across the full training run, allowing the agent to iteratively refine its pricing policy based on accumulated experience rather than a single static rule set.

---

## 24. References

- Sutton, R. S., & Barto, A. G. — *Reinforcement Learning: An Introduction*
- Mnih, V., et al. — *Human-Level Control through Deep Reinforcement Learning* (Deep Q-Network)
- OpenAI / Farama Foundation — *Gymnasium Documentation*
- Littlewood, K. — *Forecasting and Control of Passenger Bookings* (Littlewood's Rule, airline revenue management)
- Bellman, R. — *Dynamic Programming* (Bellman Equation)
- Talluri, K. T., & van Ryzin, G. J. — *The Theory and Practice of Revenue Management*

---
