**Project:** Travel & Hospitality – Reinforcement Learning for Dynamic Pricing

## Objective

To understand what dynamic pricing is, where it is used in the real world, and how AI/ML techniques (including reinforcement learning) are applied to build modern pricing systems.


## 1. What is Dynamic Pricing?

Dynamic pricing is a revenue management strategy where a business changes the price of a product or service based on real-time conditions instead of keeping it fixed. Prices go up when demand is high or inventory is scarce, and come down when demand is weak or a business needs to clear stock before a deadline (e.g., before a flight departs or a hotel date passes).

It is also called surge pricing, demand pricing, or time-based pricing. Instead of relying only on cost-plus rules, businesses now use algorithms that continuously factor in demand signals, competitor prices, inventory levels, customer behavior, time remaining, and even external data like weather or seasonality to decide the "right" price at any given moment.

This is directly relevant to our project: the airline/hotel booking system we are building is a classic dynamic pricing problem, where the agent must balance charging a high price against the risk of unsold inventory as the deadline approaches.

## 2. Real-World Applications

1. **Airlines** – Ticket prices change based on days left until departure, seat availability, and route demand. Prices typically rise as departure nears if seats are filling up fast, but drop if too many seats remain unsold close to the deadline.

2. **Ride-sharing (Uber, Lyft)** – Fares surge during high-demand periods (rain, holidays, rush hour) to balance rider demand with available driver supply, and to pull more drivers onto the road when needed.

3. **Hotels & Hospitality** – Room rates rise during peak season or local events and fall during off-season to maximize occupancy and revenue per available room (RevPAR).

4. **E-commerce (Amazon, Walmart)** – Product prices are adjusted frequently, sometimes multiple times a day, based on competitor pricing, stock levels, and customer browsing/purchase behavior.

5. **Event & Entertainment Ticketing** – Concert, sports, and movie ticket prices shift based on how close the event is, seat category demand, and remaining inventory.

## 3. Benefits of Dynamic Pricing

- **Higher revenue and better margins** – prices are set closer to what each customer segment is actually willing to pay, rather than one flat rate for everyone.

- **Reduced spoilage/unsold inventory** – perishable inventory (flight seats, hotel rooms, event tickets) is more likely to sell before it becomes worthless.

- **Faster response to market changes** – businesses can react to competitor moves, demand spikes, or supply shocks in near real time instead of manually repricing.

- **Improved customer experience in some cases** – buyers can find lower prices during low-demand periods.

## 4. Challenges of Dynamic Pricing

- **Customer perception of unfairness** – customers dislike seeing different prices for the same product, which can hurt trust and brand loyalty.

- **Regulatory and ethical concerns** – personalized or surge pricing raises questions about price discrimination and consumer protection, and is drawing increasing scrutiny from regulators.

- **Data and infrastructure requirements** – reliable dynamic pricing needs continuous, clean data pipelines (competitor prices, demand signals, inventory) which are costly to build and maintain.

- **Risk of model exploitation/instability** – poorly designed pricing algorithms (especially RL-based ones) can find "edge-case" strategies that maximize short-term reward but hurt long-term customer relationships or violate business constraints.

## 5. How AI is Used in Pricing

Modern dynamic pricing has moved from simple rule-based logic (e.g., "discount 10% every day") to data-driven and learning-based systems:

- **Demand forecasting models** (regression, gradient boosting) predict how many units will sell at a given price point.

- **Machine learning classification/regression** is used to estimate price elasticity and customer willingness to pay from historical and behavioral data.

- **Reinforcement Learning (RL)** treats pricing as a sequential decision-making problem: the "agent" (pricing algorithm) observes a state (e.g., remaining inventory, days left), takes an action (sets a price), and receives a reward (revenue earned). Over many simulated episodes, algorithms like **Q-Learning**, **Deep Q-Networks (DQN)**, and **Proximal Policy Optimization (PPO)** learn a pricing policy that maximizes cumulative long-term revenue rather than just the next sale — which is exactly the approach our project will use.

- RL-based pricing is especially valuable for problems with a hard deadline and finite inventory (like our airline/hotel case) because it naturally learns to lower prices as the deadline approaches if inventory remains unsold, without being explicitly told to do so.

## 6. Key Takeaways for This Project

- Our RL agent's "state" (remaining inventory, days until departure) and "action" (price level) map directly onto how real airline and hotel companies think about the problem.

- Q-Learning is a good starting baseline; DQN/PPO become necessary once the state space grows too large for a simple table.

- We should keep in mind fairness and stability concerns (from the research) when designing the reward function, so the agent doesn't learn exploitative or unstable pricing behavior.

---

## References

1. Wikipedia. "Dynamic Pricing." https://en.wikipedia.org/wiki/Dynamic_pricing
2. Harvard Business School Online. "Dynamic Pricing: What It Is & Why It's Important." https://online.hbs.edu/blog/post/what-is-dynamic-pricing
3. Salesforce. "What Is Dynamic Pricing? How It Works, With Examples." https://www.salesforce.com/blog/sales/dynamic-pricing/
4. Brookings Institution. "What is dynamic pricing, and why do consumers need better protections?" https://www.brookings.edu/articles/what-is-dynamic-pricing-and-why-do-consumers-need-better-protections/
5. ScienceDirect. "Dynamic pricing: Definition, implications for managers, and future research directions." https://www.sciencedirect.com/science/article/abs/pii/S0022435923000544
6. Gadde, N., Mohapatra, A., Dey, S., Das, I., Bhatia, V., Reddy, G. "Optimizing Dynamic Pricing through Reinforcement Learning: Techniques, Case Studies, and Implementation Challenges." International Journal of Science and Research, Vol. 13, Issue 11, 2024. https://www.ijsr.net/getabstract.php?paperid=SR241028211931
7. "Dynamic Retail Pricing via Q-Learning – A Reinforcement Learning Framework for Enhanced Revenue Management." arXiv, 2024. https://arxiv.org/html/2411.18261v1
8. "The AI-driven evolution of dynamic pricing: a semi-systematic review and novel hierarchical classification." Taylor & Francis, 2026. https://www.tandfonline.com/doi/full/10.1080/29966892.2026.2682621