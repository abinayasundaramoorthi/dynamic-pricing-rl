# Week 3 Report: Deep Reinforcement Learning (DQN)

## Objective
Replace the Q-table with a neural network (Deep Q-Network) to handle larger state spaces and improve on tabular Q-Learning's performance.

## Work Completed

1. **Experience Replay Buffer** (`agents/replay_buffer.py`)
   Implemented a fixed-capacity buffer storing (state, action, reward, next_state, done) transitions, with random batch sampling - breaks the correlation between consecutive training steps that would otherwise destabilize neural network training.

2. **DQN Agent** (`agents/dqn_agent.py`)
   Built a small feedforward Q-network (2 inputs -> 64 -> 64 -> 5 outputs) using PyTorch, with a target network (updated every 500 steps) and epsilon-greedy exploration - the two standard techniques that make DQN training stable.

3. **Training and Results** (`notebooks/dqn_training.ipynb`)
   Trained for 3,000 episodes (vs. 20,000 needed for tabular Q-Learning - the network's ability to generalize across similar states meant far less training data was required). Final result: **$18,289.30 average revenue**, beating every prior strategy (Fixed Price $13,000, Q-Learning $13,512).

4. **Key Finding**
   DQN learned to hold prices near the maximum ($200) almost the entire season, deliberately trading a lower guaranteed sellout rate (only ~19% of episodes fully sold out) for much higher realized prices - exploiting the late-season demand surge more aggressively than either heuristic or Q-Learning discovered. This is a genuinely more sophisticated, revenue-first strategy that directly optimizes the actual business objective (revenue) rather than a proxy like sellout rate.

## Key Outcome
The best-performing pricing strategy built in the project, requiring less training data than tabular Q-Learning while achieving significantly higher revenue, at the cost of a real, honestly-reported trade-off (lower sellout rate).

## Dependencies Added
`torch`

## Files Delivered
- `agents/replay_buffer.py`
- `agents/dqn_agent.py`
- `notebooks/dqn_training.ipynb`
