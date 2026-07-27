# Business KPI Definitions

## Issue #92

Implement Business Metrics Calculator

---

# Objective

Translate reinforcement learning performance into business-oriented
Key Performance Indicators (KPIs).

---

# KPIs

## 1. Total Revenue

**Definition**

Total income generated during the selling period.

**Formula**

Total Revenue = Sum of all revenues

---

## 2. Average Revenue

**Definition**

Average revenue earned per episode or simulation.

**Formula**

Average Revenue = Total Revenue / Number of Episodes

---

## 3. Revenue Growth

**Definition**

Percentage increase in revenue between the initial and final periods.

**Formula**

Revenue Growth (%) =
((Final Revenue − Initial Revenue) / Initial Revenue) × 100

---

## 4. Inventory Utilization

**Definition**

Percentage of inventory used during the selling horizon.

**Formula**

Inventory Utilization (%) =
((Initial Inventory − Remaining Inventory)
/ Initial Inventory) × 100

---

## 5. Sell-through Rate

**Definition**

Percentage of available inventory successfully sold.

**Formula**

Sell-through Rate (%) =
Units Sold / Initial Inventory × 100

---

## 6. Average Inventory Usage

**Definition**

Average inventory level observed throughout the selling horizon.

**Formula**

Average Inventory Usage =
Average of recorded inventory levels

---

## 7. Average Selling Price

**Definition**

Average selling price achieved across all transactions.

**Formula**

Average Selling Price =
Average of all selling prices

---

# Reusability

The KPI functions are implemented independently of any specific
reinforcement learning algorithm.

They can be used to evaluate:

- Rule-Based Pricing
- Time-Based Pricing
- Q-Learning
- Deep Q-Network (DQN)
- Future Reinforcement Learning Models

---

# Conclusion

These business KPIs enable objective comparison of different pricing
policies using financial and operational performance metrics.