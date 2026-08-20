# Dataset and Environment Decision

## Objective
Select the most suitable data source and environment for training the Reinforcement Learning dynamic pricing agent.

---

## 1. Evaluation

### Airline Datasets
Evaluated three options (Flight Price Prediction, Flight Prices, Airlines Flights Data). All three provide `Price` and `Days_left` (days before departure), which maps directly to the "days remaining" part of our RL state. However, none provide a seats-remaining/inventory field directly — a full inventory dataset (`saadharoon27/airlines-dataset`) exists as a relational database but requires joining multiple tables to reconstruct seat depletion over time.

**Selected: `shubhambathwal/flight-price-prediction`** — chosen for its direct `Days_left` vs `Price` relationship, useful for validating that the simulator's price-decay pattern matches real-world airline behavior.

### Hotel Datasets
Evaluated `hotel-booking-demand`, `hotel-reservations-classification-dataset`, and `hotel-reservations-data`. All contain `lead_time` (days before arrival) and price (`adr` / `avg_price_per_room`), but **no dataset publishes total room capacity** — hotels do not disclose inventory the way airlines disclose seat maps. Capacity would need to be assumed or derived from maximum observed daily bookings.

**Not selected for this project** — no hotel dataset was downloaded; airline and retail data were judged sufficient to cover both the time-decay and inventory-depletion patterns needed.

### Retail Datasets
Evaluated `retail-store-inventory-forecasting-dataset`, `retail-store-inventory-and-demand-forecasting`, and `retail-price-optimization-case-study`. The first was the strongest match found across all three domains: it contains `Inventory Level`, `Units Sold`, `Price`, `Demand Forecast`, `Discount`, `Competitor Pricing`, and `Revenue` in a single daily row per product — no table joins or assumptions required.

**Selected: `anirudhchauhan/retail-store-inventory-forecasting-dataset`** — chosen because it is the only evaluated dataset with real inventory level and price recorded together, directly usable to calibrate the simulator's demand/inventory dynamics.

### Summary Table

| Domain | Has Price? | Has Days-Remaining/Time? | Has Inventory? | Verdict |
|---|---|---|---|---|
| Airline | Yes | Yes | Only via table joins | Partial — used for price/time pattern |
| Hotel | Yes | Yes (lead_time) | No (must assume capacity) | Not used |
| Retail | Yes | Yes (daily) | Yes, directly | Used for inventory/demand pattern |

---

## 2. Decide

**Use a real dataset?**
Partially. The two selected datasets (flight price prediction, retail store inventory) will be used to *validate and calibrate* the simulator — e.g., confirming that price should decrease as days-remaining shrinks (from the airline data), and modeling how inventory depletes in response to demand and price (from the retail data). Neither dataset alone contains price + inventory + time-remaining together, so neither can directly drive RL training on its own.

**Use a synthetic simulator?**
Yes. Because no single real dataset combines all three required signals (price, inventory, time-remaining), the RL agent must be trained inside a synthetic environment. This environment's demand and depletion logic will be informed by the patterns observed in the two real datasets above, rather than built on pure assumption.

**Design a custom Gymnasium environment?**
Yes. No pre-built Gym or Gymnasium environment models dynamic pricing with finite inventory and a shrinking selling window. A custom environment must be built regardless of data source. It will be structured using Gymnasium's conventions (`gymnasium.Env` subclass, `action_space`/`observation_space` via `gymnasium.spaces`, standard `reset()`/`step()` methods) rather than the deprecated OpenAI Gym, since Gymnasium is the actively maintained standard and keeps the environment compatible with libraries like `stable-baselines3` for future upgrades (e.g., moving from Q-Learning to DQN).

---

## 3. Recommendation

**Recommended approach: A hybrid strategy — real datasets for calibration, a custom Gymnasium-based synthetic environment for training.**

Use `shubhambathwal/flight-price-prediction` to validate the time-decay pricing pattern (price vs. days-left) and `anirudhchauhan/retail-store-inventory-forecasting-dataset` to calibrate the inventory-depletion and demand-response logic. Build the actual RL training environment as a custom class following the Gymnasium API, with demand and inventory dynamics parameterized from patterns observed in these two datasets rather than arbitrary assumptions. This avoids the two extremes of relying on incomplete real data (no dataset has all required fields) or a fully arbitrary synthetic simulator (patterns grounded in real data instead of guesswork).
