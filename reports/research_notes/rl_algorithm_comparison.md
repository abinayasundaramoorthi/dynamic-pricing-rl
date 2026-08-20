# Reinforcement Learning Algorithm Comparison — Issue #12

**Project:** Dynamic Pricing using Reinforcement Learning
**Prepared by:** Yogeshwaran
**Objective:** Compare algorithms suitable for the project and recommend one

---

## 1. Q-Learning

**Simple idea:** The agent keeps a table (called a Q-table) that stores a score for every possible (state, action) combination. It updates this table after every action, slowly learning which price works best for each situation.

**Advantages**
- Simple to understand and implement
- Works well when the number of states and actions is small
- Guaranteed to find the best policy eventually, given enough training

**Limitations**
- Needs a table for every possible state — becomes impossible to manage when states are large or continuous (e.g., many combinations of inventory + days left)
- Doesn't generalize — if a new, unseen state appears, the agent has no idea what to do
- Slow to scale for real-world problems

**Suitable use cases**
- Small, simple environments with limited state/action combinations
- Good as a **first baseline model** to prove the concept before moving to something more powerful

---

## 2. Deep Q-Network (DQN)

**Simple idea:** Same core idea as Q-Learning, but instead of a table, it uses a **Neural Network** to predict the score for each action. This lets it handle much larger and more complex situations.

**Advantages**
- Can handle large or continuous state spaces (like our project's many inventory/day combinations)
- Learns patterns instead of memorizing every state individually
- Well-documented, widely used, many tutorials and libraries available (e.g., PyTorch, Stable-Baselines3)

**Limitations**
- Training can be unstable — needs techniques like Experience Replay and Target Networks to work properly
- Requires more tuning (hyperparameters) than Q-Learning
- Takes longer to train and more compute power

**Suitable use cases**
- Medium-to-complex environments with large state spaces
- Problems where a simple table isn't practical — like our dynamic pricing problem, where inventory and days can have many combinations

---

## 3. PPO (Proximal Policy Optimization) — High Level

**Simple idea:** Instead of learning a score for each action (like Q-Learning/DQN), PPO directly learns a **policy** — a strategy that outputs the best action to take. It updates this policy carefully, in small safe steps, to avoid drastic bad changes during training.

**Advantages**
- Very stable and reliable training compared to DQN
- Works well for both discrete actions (like price levels) and continuous actions (like an exact price value)
- Considered one of the strongest general-purpose RL algorithms today, widely used in industry

**Limitations**
- More complex to implement than Q-Learning or DQN
- Requires more understanding of policy-based methods (a different approach than value-based methods like Q-Learning/DQN)
- Usually needs more training episodes to converge compared to simpler methods

**Suitable use cases**
- Complex environments needing stable, reliable training
- Projects with continuous action spaces (e.g., setting an exact price like ₹437 instead of choosing from fixed price levels)

---

## 4. SARSA (State-Action-Reward-State-Action)

**Simple idea:** Very similar to Q-Learning, but with one key difference: SARSA updates its table based on the action it **actually took next**, while Q-Learning updates based on the **best possible next action** (even if it doesn't take it). This makes SARSA more "cautious."

**Advantages**
- Learns safer policies — good for situations where mistakes are costly
- Simple to implement, similar to Q-Learning
- Useful when the environment has risk (e.g., avoiding actions that could cause big losses)

**Limitations**
- Same table-based scaling problem as Q-Learning — doesn't handle large state spaces well
- Tends to learn more conservative (less aggressive) strategies, which may earn slightly less revenue in exchange for safety
- Less commonly used in modern RL projects compared to DQN/PPO

**Suitable use cases**
- Environments where safety and caution matter more than maximum reward
- Simple problems, similar scale to Q-Learning

---

## 5. Comparison Table

| Algorithm | Handles Large State Space? | Training Stability | Complexity to Implement | Best For |
|---|---|---|---|---|
| Q-Learning | ❌ No | ✅ Stable | ⭐ Easy | Simple baseline |
| SARSA | ❌ No | ✅ Stable | ⭐ Easy | Safety-focused simple problems |
| DQN | ✅ Yes | ⚠️ Needs tuning | ⭐⭐ Medium | Large discrete state spaces |
| PPO | ✅ Yes | ✅ Stable | ⭐⭐⭐ Harder | Complex/continuous action spaces |

---

## 6. Recommendation for Our Project

For the **Dynamic Pricing project**, our state space (inventory remaining × days remaining) is too large to manage with a simple table, ruling out plain Q-Learning and SARSA for the final model.

**Recommended approach:**
1. **Start with Q-Learning** as a simple baseline (Week 2) — easy to implement and helps prove the basic concept works before adding complexity.
2. **Move to DQN** as the main model (Week 3) — it directly matches our project roadmap, handles our large state space well, and has strong community support and tutorials for a beginner-friendly implementation.
3. **PPO is noted as a possible future improvement** — if time permits, it could be explored for even more stable training or if we later want to set continuous prices (e.g., exact ₹ values) instead of fixed price levels.

**In short: Q-Learning → DQN is the practical path for this project's timeline, with PPO as an optional stretch goal.**
