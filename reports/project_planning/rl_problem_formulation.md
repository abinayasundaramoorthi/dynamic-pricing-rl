# RL Problem Formulation — Dynamic Pricing for Travel & Hospitality

**Project:** Project 2 — Travel & Hospitality: Reinforcement Learning for Dynamic Pricing
**Deliverable:** `reports/project_planning/rl_problem_formulation.md`
**Sprint / Day:** Week 1, Day 2 — MDP Design
**Team Size:** 4 members
**Status:** Draft for team review → finalize by end of Day 2

---

## 1. Objective of This Document

Day 1 established the business problem, objectives, scope, stakeholders, and KPIs (`reports/Problem_Statement.md`).

Day 2 converts that business understanding into a **formal Reinforcement Learning problem**. This document:

1. Finalizes the business scenario the environment will simulate.
2. Formally defines the **State Space**, **Action Space**, and **Reward Function**.
3. Documents the **transition dynamics** and **discount factor**.
4. Diagrams the **RL workflow** (agent–environment interaction loop).
5. Feeds directly into tomorrow's work: coding `pricing_env/pricing_env.py` (Day 3).

---

## 2. Finalized Business Scenario

Three candidate scenarios were considered — **Airline seat pricing**, **Hotel room pricing**, and **Retail/event-ticket pricing**. All three share the same underlying structure (finite, perishable inventory sold over a finite horizon), which is exactly why a single MDP formulation works for all of them.

**Decision: Hotel Room Pricing**, used as the primary framing for Phase 1, with the environment built generically enough to relabel as "airline seats" with zero code changes.

| Criterion | Airline | Hotel | Retail/Event |
|---|---|---|---|
| Inventory perishability | Yes | Yes | Yes |
| Familiarity / intuitiveness for stakeholder demos | Medium | High | Medium |
| Booking horizon length | Often long (weeks–months) | Short–medium (days–weeks) — easier to simulate short episodes for fast training iteration | Variable |
| Data/parameter availability for demand curve assumptions | Complex (fare classes) | Simpler (single room-night rate) | Simpler |
| Team decision | — | **Selected** | — |

**Rationale:** A single hotel room-type, single-property scenario keeps the Phase 1 MVP tractable (Section 6.1/6.2 of the project design doc — single resource, no network-level RM), while still being economically representative of the same spoilage-vs-leakage tension described in the business background. The environment naming/interface will remain generic (`resource_id`, `unit_price`) so the same code can later represent an airline seat or event ticket without refactoring — satisfying the "reusable, extensible simulation environment" secondary objective.

**Concrete framing used going forward:**
- **Property:** a single hotel, single room type.
- **Selling season:** a fixed booking horizon per episode (e.g., 30 days out from check-in).
- **Inventory:** a fixed number of rooms available at the start of the season (e.g., 100 rooms), non-replenishable within an episode.
- **Goal:** learn a pricing policy that maximizes total revenue from that room type across the booking horizon.

---

## 3. Markov Decision Process (MDP) — Formal Definition

The problem is modeled as a discrete-time, finite-horizon MDP: **⟨S, A, P, R, γ, T⟩**

| Component | Definition |
|---|---|
| **S** | State space — see Section 4 |
| **A** | Action space — see Section 5 |
| **P(s' \| s, a)** | Transition function — stochastic, driven by the demand model (Section 6) |
| **R(s, a)** | Reward function — see Section 7 |
| **γ** | Discount factor — see Section 8 |
| **T** | Terminal condition — see Section 9 |

**Why an MDP (and not supervised learning):** the outcome of any single pricing decision depends only on the current state, not the full history — but each decision changes the state (inventory, time remaining) available to every future decision. This coupling of sequential decisions through a shared, depleting resource is exactly what supervised, single-shot price prediction cannot capture, and exactly what Q-Learning-family methods are designed to optimize (cumulative, not per-transaction, reward).

### 3.1 Mathematical Objective

Formally, the goal of training is to learn an optimal pricing policy **π\*** that maximizes the expected cumulative discounted reward over an episode:

```
π* = argmax_π  E_π [ Σ_t γ^t · r_t ]
```

where:
- **π** is the pricing policy (state → action mapping) being learned,
- **γ** is the discount factor (Section 8),
- **r_t** is the reward received at time step *t* (Section 7),
- the expectation is taken over the stochasticity of the demand model (Section 6) and, if used, the agent's exploration policy.

This is the standard RL control objective, and it is what both the tabular Q-Learning update rule and the DQN loss function are ultimately approximating.

---

## 4. State Space (S)

### 4.1 Phase 1 Core State Vector

```
s_t = [remaining_inventory, days_until_checkin]
```

This is the minimum sufficient state to make the environment non-trivial and is what will be implemented first in `pricing_env.py` (Day 3), consistent with the MVP spec.

### 4.2 Full State Variable Table

| Variable | Included In | Type | Example | Why It Matters |
|---|---|---|---|---|
| `remaining_inventory` | Phase 1 (core) | Discrete int, 0–100 | 42 of 100 rooms | Determines urgency to sell vs. room to hold price high |
| `days_until_checkin` | Phase 1 (core) | Discrete int, 0–30 | 12 days | Governs discounting urgency as the deadline approaches |
| `current_price` | Phase 1 (extended) | Discrete/continuous | $180 | Gives the agent continuity/context for its next move |
| `demand_level` | Phase 1 (extended) | Categorical: Low/Med/High | High | Distinguishes "hold price" from "discount" conditions |
| `booking_rate` | Phase 1 (extended) | float, rooms/day | 3.2/day | Leading indicator of pacing vs. an ideal sell-through curve |
| `seasonality` | Phase 1 (extended) | Categorical | Peak season | Shifts the demand baseline substantially |
| `competitor_price` | Phase 2 | float | $195 | Customers are price-comparison sensitive |
| `customer_segment_mix` | Phase 2 | Categorical/vector | 60% leisure | Segments differ in price sensitivity and lead time |
| `market_trend` | Phase 2 | Categorical | Trending up | Anticipates near-term demand shifts |

### 4.3 Discretization Note

Tabular Q-Learning (Week 2) requires a **discretized, enumerable** state space — `remaining_inventory` and `days_until_checkin` are naturally integer-bounded, so binning is straightforward (e.g., inventory bucketed in groups of 5). DQN (Week 3) will accept the continuous/normalized version of the same features directly, which is the practical reason the project progresses from Q-Learning → DQN as state complexity grows, rather than starting with DQN.

### 4.4 Observation Space (Implementation Bridge for Day 3)

To bridge this design into the `gym.Env` implementation on Day 3, the observation space will be defined as:

```
For Phase 1 (core state):

  Discrete formulation (Tabular Q-Learning):
      observation_space = Discrete(inventory_bins × day_bins)
      e.g. remaining_inventory binned into groups of 5 (0-20 bins)
           × days_until_checkin as 0-30 (31 bins)

  Continuous/normalized formulation (DQN):
      observation_space = Box(
          low  = [0, 0],
          high = [max_inventory, max_days],
          shape = (2,),
          dtype = float32
      )
```

The tabular agent (Week 2) will consume the discretized version; the DQN agent (Week 3) will consume the normalized `Box` version directly. Both wrap the same underlying environment state — only the `observation_space` and the state-encoding function differ, so `pricing_env.py` should expose the raw `[remaining_inventory, days_until_checkin]` tuple internally and let each agent's wrapper handle its own encoding.

---

## 5. Action Space (A)

**Decision: Discrete action space for Phase 1.** Continuous control (PPO/SAC) is an explicit Phase 2 enhancement, only after the discrete formulation is validated.

**Why discrete first:**
- Compatible with tabular Q-Learning, which needs an enumerable action set to build a Q-table.
- Mirrors how hotel/airline pricing is actually operationalized (rate tiers / fare classes), which makes the learned policy auditable by the Revenue Manager persona.
- Simplifies debugging of the reward function and demand model while the team is still validating the environment (Days 5–6).

| Action ID | Description | Price Adjustment | Resulting Price (Base = $200/night) |
|---|---|---|---|
| 0 | Deep discount | −20% | $160 |
| 1 | Moderate discount | −10% | $180 |
| 2 | Slight discount | −5% | $190 |
| 3 | Hold current price | 0% | $200 |
| 4 | Slight increase | +5% | $210 |
| 5 | Moderate increase | +10% | $220 |
| 6 | Premium increase | +20% | $240 |

Safety bounds (a floor/ceiling on price) will be exposed as environment configuration so the Revenue Manager persona's "upper/lower safety bounds" requirement from Day 1 is satisfiable without changing the action set itself.

### 5.1 Action Implementation Detail

The agent itself never outputs a dollar price — it only outputs an **action ID (0–6)**. The environment is responsible for translating that ID into an actual room price using the configured `base_price` and the adjustment percentage table in Section 5:

```
chosen_price = base_price × (1 + adjustment_pct[action_id])
```

Keeping this conversion inside the environment (not the agent) means the pricing logic — base price, tier granularity, adjustment percentages — can be changed via `configs/` without touching any RL algorithm code, and the same Q-Learning/DQN implementation can be pointed at a different property or price ladder unmodified.

---

## 6. Transition Dynamics (P)

The transition from `s_t` to `s_{t+1}` is driven by two mechanisms:

1. **Deterministic time decay:** `days_until_checkin` decrements by exactly 1 each step, regardless of action.
2. **Stochastic demand response:** given the chosen price, a purchase occurs with probability `p(price_t, remaining_inventory_t, days_until_checkin_t)` — decreasing in price, and modulated by scarcity and time-to-deadline. If a sale occurs, `remaining_inventory` decrements by 1 and `current_price` updates to the chosen price; otherwise inventory is unchanged.

The exact functional form of the demand probability (e.g., logistic in price, with time/inventory modifiers) is the subject of Day 4 (`pricing_env/demand_model.py`) and will be documented separately once calibrated; this document fixes the *interface* the demand model must satisfy: `P(sale | price, inventory, days_remaining) → [0, 1]`.

### 6.1 Worked Transition Examples

Concrete examples make the abstract transition/reward rules easier to sanity-check before any code is written:

| Current State (inventory, days) | Action | Demand Outcome | Next State (inventory, days) | Reward |
|---|---|---|---|---|
| (100, 30) | Hold price ($200) | Sale | (99, 29) | +200 |
| (99, 29) | +20% ($240) | No sale | (99, 28) | 0 |
| (99, 28) | −10% ($180) | Sale | (98, 27) | +180 |
| (5, 1) | −20% ($160) | Sale | (4, 0) | +160 |
| (4, 0) | *(terminal — days = 0)* | — | Episode ends | −λ1 × 4 (unsold penalty on the 4 remaining rooms) |

These rows illustrate the full loop end-to-end: `days_until_checkin` always decrements by 1 regardless of the sale outcome; `remaining_inventory` only decrements on a sale; and the terminal step is where the spoilage penalty from Section 7 actually gets applied.

---

## 7. Reward Function (R)

Reward design is the single most consequential decision in this formulation — the agent will optimize exactly what is measured, not what the team intends. An underspecified reward risks a policy that scores well in training but is economically unsound (e.g., always discounting to guarantee sales).

**Proposed reward at each time step *t*:**

```
r_t = (price_t × sale_t)
      − λ1 · (unsold_penalty, applied only if terminal AND inventory_remaining > 0)
      − λ2 · (discount_depth_t)^2 · sale_t
      + λ3 · inventory_balance_bonus_t
```

| Term | Purpose |
|---|---|
| `price_t × sale_t` | Primary revenue term — the direct business objective (`sale_t ∈ {0,1}`) |
| Unsold penalty (terminal only) | Penalizes ending the episode with unsold rooms — spoilage is permanently lost revenue; discourages holding price too long |
| Over-discounting penalty (convex) | Discourages the agent from defaulting to maximum discounts purely to guarantee a sale each step |
| Inventory balancing bonus | Dense shaping signal that rewards tracking a healthy sell-through pace, rather than relying only on the sparse terminal spoilage signal |
| `λ1, λ2, λ3` | Tunable weights — calibrated empirically across Weeks 1–2, logged in `configs/` for reproducibility |

This reward aligns the agent's immediate incentive with cumulative revenue, forces long-horizon credit assignment via the terminal penalty and a high discount factor, and is an explicit guard against reward hacking (Data Scientist persona's stated concern in the Day 1 requirements).

---

## 8. Discount Factor (γ)

`γ` is set close to 1 (proposed range **0.95–0.99**) so the agent weighs the value of the *entire remaining season*, not just the immediately next sale. A low γ would bias the agent toward short-sighted, immediate-sale-maximizing behavior (e.g., always discounting), which directly conflicts with the cumulative-revenue objective in Section 4 of the business background.

---

## 9. Terminal Condition (T)

An episode ends when **either**:
- `days_until_checkin` reaches 0, **or**
- `remaining_inventory` reaches 0,

whichever occurs first. Any inventory unsold at termination yields zero further revenue for that episode (this is what the unsold penalty in the reward function penalizes).

---

## 10. RL Workflow Diagram

```
                    ┌─────────────────────────────────────┐
                    │        Business Problem             │
                    │  Finite, perishable hotel-room      │
                    │     inventory over a booking        │
                    │             horizon                 │
                    └───────────────┬─────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────────┐
                    │   Environment (pricing_env)       │
                    │   State s_t = [inventory,         │
                    │        days_until_checkin, ...]   │
                    └───────────────┬───────────────────┘
                                    │  observes s_t
                                    ▼
                    ┌────────────────────────────────────┐
                    │             Agent                  │
                    │   Tabular Q-Learning  /  DQN       │
                    │   (epsilon-greedy policy)          │
                    └───────────────┬────────────────────┘
                                    │  selects action a_t
                                    ▼
                    ┌────────────────────────────────────┐
                    │      Action: Set Price Level       │
                    │   (Action IDs 0–6, Section 5)      │
                    └───────────────┬────────────────────┘
                                    │
                                    ▼
                    ┌────────────────────────────────────┐
                    │     Demand Simulation Model        │
                    │  Stochastic sale / no-sale draw    │
                    │  P(sale | price, inv, days_left)   │
                    └───────────────┬────────────────────┘
                                    │
                                    ▼
                    ┌────────────────────────────────────┐
                    │            Reward r_t              │
                    │  Revenue − penalties + shaping     │
                    │            (Section 7)             │
                    └───────────────┬────────────────────┘
                                    │
                                    ▼
                    ┌────────────────────────────────────┐
                    │      New State s_(t+1)             │
                    │  inventory & days_until_checkin    │
                    │            updated                 │
                    └───────────────┬────────────────────┘
                                    │  (s_t, a_t, r_t, s_t+1) stored
                                    ▼
                    ┌─────────────────────────────────────┐
                    │        Learning Update              │
                    │  Q-table update (Q-Learning) or     │
                    │  gradient step on replay buffer     │
                    │            (DQN)                    │
                    └───────────────┬─────────────────────┘
                                    │
                                    ▼
                    ┌─────────────────────────────────────┐
                    │        Updated Policy π             │
                    └───────────────┬─────────────────────┘
                                    │  loop continues until
                                    │  terminal condition (Sec. 9)
                                    ▼
                    ┌─────────────────────────────────────┐
                    │   Revenue Maximization              │
                    │      (Business Outcome)             │
                    └─────────────────────────────────────┘
```

The loop above (`s_t → a_t → r_t → s_(t+1) → learning update`) repeats once per time step within an episode, and across every episode of training, letting the agent refine its policy from accumulated experience rather than a fixed rule set.

---

## 11. Team Sign-Off Checklist (Day 2)

| Item | Owner (suggested role) | Status |
|---|---|---|
| Business scenario finalized (Hotel) | Business Analyst | ☐ Confirm with team |
| State space agreed (core + extended) | Data Scientist | ☐ Confirm with team |
| Action space agreed (7 discrete tiers) | Data Scientist + Revenue Manager persona | ☐ Confirm with team |
| Reward function structure agreed | Data Scientist | ☐ Confirm with team |
| γ and terminal condition agreed | ML Engineer | ☐ Confirm with team |
| Workflow diagram reviewed | Whole team | ☐ Confirm with team |
| Ready to hand off to Day 3 (`pricing_env.py` skeleton) | ML Engineer | ☐ Confirm with team |

---

## 12. Next Steps (→ Day 3)

This formulation is the direct input to Day 3: implementing the `gym.Env` subclass (`pricing_env/pricing_env.py`) with `reset()`, `step()`, `observation_space` built from Section 4, and `action_space` built from Section 5. Reward logic from Section 7 will be implemented as a standalone, testable function so it can be unit-tested independently in Day 5 (`tests/test_pricing_env.py`).

---

## 13. Project Assumptions

These make the Phase 1 scope explicit and give the team a shared reference point for what the environment does and does not model:

- Single hotel property.
- Single room type (no room-mix or upgrade logic).
- Fixed inventory at the start of each episode (non-replenishable within the episode).
- One pricing decision per day (daily time step granularity).
- Customer demand is stochastic and responsive to price, inventory, and time-to-deadline.
- No cancellations or no-shows within an episode.
- No overbooking.
- No competitor price reactions or competitive dynamics (deferred to Phase 2, Section 4.2).

---

## 14. References

- Sutton, R. S., & Barto, A. G. — *Reinforcement Learning: An Introduction*
- Bellman, R. — *Dynamic Programming* (Bellman Equation, discount factor formalism)
- Littlewood, K. — *Forecasting and Control of Passenger Bookings* (Littlewood's Rule)
- Talluri, K. T., & van Ryzin, G. J. — *The Theory and Practice of Revenue Management*
- OpenAI / Farama Foundation — *Gymnasium Documentation*