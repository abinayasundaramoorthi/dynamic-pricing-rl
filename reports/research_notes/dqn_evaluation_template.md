# DQN Evaluation Template

*This is a reusable template. Copy this file and rename it per experiment (e.g. `dqn_evaluation_run_003.md`), then fill in each section using the outputs of `evaluation/dqn_metrics.py` for that run.*

---

## Experiment Info

- **Run ID / date:**
- **Purpose of this run:** *(e.g. "baseline hyperparameters", "testing higher learning rate", "testing larger hidden layer")*
- **Config used:**
  - `hidden_dim`:
  - `lr`:
  - `gamma`:
  - `batch_size`:
  - `target_update_freq`:
  - `epsilon_start` / `epsilon_end` / `epsilon_decay_episodes`:
  - `num_episodes`:

## Methodology

This evaluation uses `evaluation/dqn_metrics.py`'s `DQNTrainingMonitor` class, called once per episode during training via `monitor.log_episode(...)`, logging six metrics per episode: Episode Reward, Average Reward (rolling), Revenue, Inventory Utilization, Loss per Episode, and Exploration Rate (Epsilon). Full per-episode data is saved to `episode_log.csv`; an aggregate summary is saved to `training_summary.json`. See the module docstring in `dqn_metrics.py` for the exact integration pattern.

## Results Summary

*(Fill in from `training_summary.json` after the run)*

| Metric | Value |
|---|---|
| Total episodes | |
| Avg reward (overall) | |
| Avg reward (first 10% of training) | |
| Avg reward (last 10% of training) | |
| Reward improvement | |
| Avg inventory utilization | |
| Avg loss (first 10% of training) | |
| Avg loss (last 10% of training) | |
| Final epsilon | |

## Learning Curve

*(Paste or embed a plot of episode_reward / rolling average reward over episodes here — see `notebooks/dqn_training.ipynb` for the standard plotting code using `rolling_average()`)*

## Loss Curve

*(Paste or embed a plot of loss per episode over training here. Expected pattern: loss should generally trend downward or stabilize, not diverge/explode — a diverging loss curve indicates unstable training, e.g. learning rate too high)*

## Exploration Rate (Epsilon) Curve

*(Paste or embed a plot of epsilon over episodes — should show the expected decay from `epsilon_start` to `epsilon_end` over `epsilon_decay_episodes`, then flat)*

## Observations

*(Answer these explicitly for each run)*

- Did average reward improve meaningfully from early to late training? By how much?
- Did the loss curve stabilize, or show signs of instability (spikes, divergence)?
- Did inventory utilization/sellout behavior change over the course of training, or was it stable throughout?
- Does the final policy's behavior match expectations, or reveal anything surprising? *(cross-reference with `evaluation/pricing_policy_report.md`-style analysis if doing a full policy inspection)*

## Comparison to Previous Runs

*(If this isn't the first experiment, compare key numbers to prior runs here — e.g. a small table of avg_reward_last_10pct across run IDs, to track whether hyperparameter changes are actually helping)*

| Run ID | Avg Reward (last 10%) | Notes |
|---|---|---|
| | | |

## Conclusion

*(1-2 sentence takeaway: did this configuration work well? What would you try next?)*

---

## Reference: Metric Definitions

- **Episode Reward** — total reward accumulated in a single episode (= total revenue in this environment)
- **Average Reward** — rolling average of Episode Reward over a configurable window (default 100 episodes), used to visualize the overall learning trend without the noise of individual episodes
- **Revenue** — tracked identically to Episode Reward here, kept as a separate field for clarity and in case reward shaping is introduced later (e.g. penalties) that would make reward and revenue diverge
- **Inventory Utilization** — percentage of total starting inventory actually sold by the end of the episode
- **Loss per Episode** — the average of all DQN training-step losses (MSE between predicted and target Q-values) that occurred during that episode; `None` for early episodes before the replay buffer has enough data to begin training
- **Exploration Rate (Epsilon)** — the probability of taking a random (exploratory) action during that episode, per the epsilon-greedy schedule defined in `agents/dqn_agent.py`
