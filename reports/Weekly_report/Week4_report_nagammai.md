# Week 4 Progress Report

## **Project Title:** Reinforcement Learning for Dynamic Pricing

# Overview

During the fourth week of the project, I focused on extending the Dynamic Pricing Reinforcement Learning system with **additional business-oriented features** to make the solution more practical, explainable, and suitable for real-world hotel revenue management.

Two major additional functionalities were developed:

1. **Human-in-the-Loop Learning**
2. **Digital Twin Market Simulator**

The Human-in-the-Loop feature allows a revenue manager to review and modify AI-generated pricing recommendations, while the Digital Twin creates a simulated hotel market where different pricing strategies and business scenarios can be tested safely before applying them to real customers.

---

# Task 1: Implement Human-in-the-Loop Learning

## Objective

To incorporate human expertise into the AI pricing process by allowing a revenue manager to review, accept, increase, or reduce AI-generated prices.

## Work Completed

Developed a **Human Feedback Learning** module that connects AI pricing recommendations with manager decisions.

The feature supports three major types of manager feedback:

* **Accepted** – Manager agrees with the AI recommendation.
* **Increased** – Manager increases the AI-recommended price.
* **Reduced** – Manager decreases the AI-recommended price.

For example:

```text
AI Recommended Price : ₹260
Manager Price        : ₹275
Decision             : Increased
```

The manager's decision is stored in the feedback memory for future analysis.

## Deliverables

* `additional_features/human_feedback_learning/manager_feedback.py`
* `additional_features/human_feedback_learning/feedback_memory.py`

---

# Task 2: Implement Feedback Statistics

## Objective

To measure how closely AI pricing recommendations align with the decisions made by the revenue manager.

## Work Completed

Implemented feedback statistics to calculate:

* Acceptance Rate
* Average Price Difference
* Total Manager Decisions

For example, after collecting manager decisions, the system generates a summary such as:

```text
Acceptance Rate     : 20.00%
Average Difference  : ₹13.00
Total Decisions     : 5
```

These metrics help identify whether the AI pricing strategy is aligned with business expectations.

## Deliverables

* `feedback_statistics.py`
* `feedback_report.py`
* `feedback_dashboard.py`

---

# Task 3: Implement Manager Preference Learning

## Objective

To allow the system to learn the pricing preferences of the revenue manager from previous feedback.

## Work Completed

Implemented a **Preference Learning** component that calculates the average adjustment made by the manager compared with the AI recommendation.

For example:

```text
AI Price              : ₹250
Learned Adjustment    : -₹1
Updated Recommendation: ₹249
```

This allows historical manager decisions to influence future pricing recommendations.

The system therefore moves from simply collecting feedback to actually **learning from human decisions**.

## Deliverables

* `preference_learning.py`
* `feedback_analyzer.py`
* `feedback_explanation.py`
* `integration.py`

---

# Task 4: Feedback Logging and Export

## Objective

To maintain a historical record of manager decisions for analysis and future learning.

## Work Completed

Implemented feedback logging and export functionality.

Manager feedback can be stored in structured formats such as:

* CSV
* JSON
* Internal feedback memory

The stored information includes:

* AI price
* Manager price
* Decision
* Scenario
* Timestamp

This creates a historical dataset that can be used to understand manager behavior and improve future pricing recommendations.

## Deliverables

* `feedback_logger.py`
* `feedback_export.py`
* `manager_feedback.csv`
* `manager_feedback.json`
* `feedback_history.csv`

---

# Task 5: Implement Digital Twin Market Simulator

## Objective

To create a simulated hotel market where different business scenarios can be tested without affecting real customers or hotel operations.

## Work Completed

Developed a **Digital Twin Market Simulator** that generates different hotel market conditions.

The simulator considers scenarios such as:

* Normal Day
* Festival
* Holiday
* Conference

It also simulates different weather conditions such as:

* Sunny
* Cloudy
* Rainy

The simulator additionally generates competitor prices and expected customer demand.

Example:

```text
Event             : Festival
Weather           : Sunny
Competitor Price  : ₹227
Expected Demand   : 59
Simulated Demand  : 106
RL Price          : ₹228
Rooms Sold        : 78
Revenue           : ₹17,784
```

This allows the RL pricing strategy to be evaluated under different market conditions.

## Deliverables

* `additional_features/digital_twin/digital_twin.py`
* `scenario_generator.py`
* `event_simulator.py`
* `weather_simulator.py`
* `competitor_simulator.py`
* `market_simulator.py`

---

# Task 6: Policy Comparison

## Objective

To determine whether RL-based pricing performs better than traditional pricing approaches under different simulated market conditions.

## Work Completed

Implemented a policy comparison module that compares:

* Static Pricing
* Rule-Based Pricing
* RL Pricing

For each simulated scenario, the system calculates the revenue generated by each strategy and identifies the best-performing strategy.

Example:

```text
Static Pricing       : ₹10,000
Rule-Based Pricing   : ₹11,990
RL Pricing           : ₹11,340

Best Strategy        : Rule-Based Pricing
```

This provides a business-oriented evaluation of the RL pricing strategy.

## Deliverables

* `policy_comparison.py`

---

# Task 7: Risk Analysis

## Objective

To identify potentially risky market conditions and provide pricing recommendations.

## Work Completed

Implemented a risk analyzer that evaluates simulated scenarios based on factors such as:

* Weather conditions
* Demand levels
* Market conditions

The system classifies scenarios into risk levels such as:

* Low
* High

It also provides recommendations for the revenue manager.

Example:

```text
Risk Level       : HIGH
Reason           : Bad weather may reduce bookings.
Recommendation   : Reduce room price by 5%.
```

This makes the Digital Twin more useful as a **decision-support system** rather than only a simulation tool.

## Deliverables

* `risk_analyzer.py`

---

# Task 8: Business Simulation Report

## Objective

To summarize the results of the Digital Twin simulations using meaningful business KPIs.

## Work Completed

Developed a business report containing:

* Total Revenue
* Average Revenue
* Average Demand
* Average RL Price
* Average Occupancy
* Sell-through Rate
* Highest Revenue Scenario
* Lowest Revenue Scenario

The highest-performing scenario is also displayed with additional business information.

Example:

```text
Highest Revenue Scenario

Event      : Holiday
Weather    : Cloudy
Demand     : 107
RL Price   : ₹272
Revenue    : ₹22,032
```

This provides richer information than reporting revenue alone.

## Deliverables

* `simulation_report.py`
* `twin_dashboard.py`

---

# Task 9: AI Business Recommendations

## Objective

To convert simulation results into actionable business recommendations.

## Work Completed

Added an AI recommendation section based on the simulated market conditions.

Example recommendations include:

```text
AI Recommendation

• Increase prices during Festivals.
• Maintain prices during Normal Days.
• Offer discounts during Rainy weather.
• Continuously monitor competitor pricing.
• Run Digital Twin simulations before applying new pricing.
```

This helps bridge the gap between machine learning output and practical revenue management decisions.

---

# Task 10: Export Digital Twin Results

## Objective

To store simulation results in a structured format for further analysis and evaluation.

## Work Completed

Implemented CSV export functionality for Digital Twin simulation results.

The exported data contains useful information such as:

* Event
* Weather
* Competitor Price
* Demand
* RL Price
* Rooms Sold
* Revenue
* Policy Winner
* Risk Level

This allows the results to be analyzed later using Python, Excel, dashboards, or other visualization tools.

## Deliverables

* `export_results.py`
* `evaluation/digital_twin_results.csv`

---

# Task 11: Dashboard Summary

## Objective

To provide a quick overview of the overall performance of the Digital Twin simulation.

## Work Completed

Added a dashboard summary containing:

```text
DASHBOARD SUMMARY

Best Strategy        : RL Pricing
Highest Revenue      : ₹22,032
Average Revenue      : ₹15,997.40
Average Occupancy    : 68.40%
Total Revenue        : ₹79,987
High Risk Scenarios  : 2
Low Risk Scenarios   : 3
```

This allows the evaluator or revenue manager to understand the overall simulation performance quickly.

---

# Technical Skills Applied

During Week 4, the following concepts and technologies were explored and applied:

* Human-in-the-Loop Machine Learning
* Reinforcement Learning
* Dynamic Pricing
* Manager Feedback Learning
* Preference Learning
* Feedback Analysis
* Digital Twin Concepts
* Market Simulation
* Scenario Simulation
* Risk Analysis
* Policy Comparison
* Revenue Optimization
* Business KPI Analysis
* Python Programming
* CSV and JSON Data Handling
* Data Visualization
* Modular Software Design
* Git and GitHub Workflow

---

# Key Learnings

Throughout Week 4, I gained a better understanding of:

* How human expertise can complement Reinforcement Learning.
* How manager feedback can be stored and used for future pricing decisions.
* How AI systems can learn pricing preferences from historical human decisions.
* How Digital Twins can be used to test business strategies safely.
* How different market conditions can affect hotel demand and revenue.
* How RL pricing can be compared with traditional pricing strategies.
* How risk analysis can support business decision-making.
* How machine learning outputs can be converted into actionable business recommendations.
* The importance of explainability and human oversight in AI-based business systems.

---

# Challenges Faced

The major challenges during Week 4 included:

* Integrating additional features without modifying the main RL project unnecessarily.
* Managing Python package imports across the new feature modules.
* Designing a consistent feedback data structure.
* Handling manager feedback and preference calculations correctly.
* Designing realistic market scenarios for the Digital Twin.
* Ensuring simulated demand and inventory remain within valid ranges.
* Comparing multiple pricing strategies consistently.
* Generating meaningful business recommendations from simulation results.
* Managing separate Git branches for additional feature development.

---

# Outcome

By the end of Week 4, two major additional functionalities were successfully developed and tested:

### 1. Human-in-the-Loop Learning

The system can:

```text
AI Pricing Recommendation
          ↓
Revenue Manager Review
          ↓
Accept / Increase / Reduce
          ↓
Store Feedback
          ↓
Analyze Manager Preferences
          ↓
Learn Pricing Adjustment
          ↓
Improved Price Recommendation
```

### 2. Digital Twin Market Simulator

The system can:

```text
Generate Market Scenario
          ↓
Simulate Event + Weather
          ↓
Generate Competitor Price
          ↓
Simulate Demand
          ↓
Apply RL Pricing
          ↓
Calculate Rooms Sold & Revenue
          ↓
Compare Pricing Policies
          ↓
Analyze Risk
          ↓
Generate Business Recommendation
```

These features extend the original RL pricing system from a basic pricing agent into a more practical **AI-powered revenue decision-support system**.

---

# Next Steps

The upcoming work will focus on improving the usability and presentation of the developed features.

Planned activities include:

* Developing a web-based interface for the additional features.
* Integrating the Human-in-the-Loop Learning results into a dashboard.
* Displaying Digital Twin simulation results visually.
* Adding charts for revenue, demand, occupancy, and policy comparison.
* Improving business recommendations.
* Testing the features with additional scenarios.
* Integrating the additional features with the overall project dashboard where appropriate.
* Preparing final project documentation and demonstration materials.

---

# Summary

| Task                    | Work                                                   | Status    |
| ----------------------- | ------------------------------------------------------ | --------- |
| Human Feedback Learning | Implemented manager feedback mechanism                 | Completed |
| Feedback Statistics     | Implemented acceptance and pricing-difference analysis | Completed |
| Preference Learning     | Learned manager pricing adjustments                    | Completed |
| Feedback Logging        | Stored manager decisions in CSV/JSON                   | Completed |
| Digital Twin            | Developed simulated hotel market                       | Completed |
| Scenario Simulation     | Added events, weather and competitor conditions        | Completed |
| Policy Comparison       | Compared Static, Rule-Based and RL pricing             | Completed |
| Risk Analysis           | Added scenario risk evaluation                         | Completed |
| Business Report         | Generated simulation KPIs and insights                 | Completed |
| AI Recommendations      | Generated actionable pricing recommendations           | Completed |
| Result Export           | Exported simulation results to CSV                     | Completed |
| Dashboard Summary       | Added overall business performance summary             | Completed |

---

**Week 4 Status: Successfully Completed**
