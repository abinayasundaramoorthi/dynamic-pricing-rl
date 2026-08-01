"""
dqn_logger.py

Logging utility to record Deep Q-Network training progress, episode by
episode, and export the results to CSV for future performance analysis
and visualization.

Logs seven fields per episode:
- Episode Number
- Episode Reward
- Average Reward     (rolling average, smooths out episode-to-episode noise)
- Revenue             (the actual dollar revenue earned - NOT the same as
                        reward in this environment, since reward.py's
                        compute_reward() subtracts discount/unsold
                        penalties and adds a pacing bonus on top of raw
                        revenue)
- Epsilon Value       (current exploration rate)
- Training Loss       (average loss across that episode's gradient updates)
- Replay Buffer Size  (how many transitions are currently stored - useful
                        to confirm the buffer is filling up as expected,
                        and to see when it hits its capacity ceiling)

Usage (inside a training loop, e.g. training/train_dqn.py):

    logger = DQNLogger()

    for episode in range(num_episodes):
        obs, info = env.reset(seed=seed + episode)
        terminated = truncated = False
        episode_losses = []

        while not (terminated or truncated):
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            agent.remember(obs, action, reward, next_obs, terminated)
            loss = agent.train_step()
            if loss is not None:
                episode_losses.append(loss)
            obs = next_obs

        logger.log(
            episode=episode,
            episode_reward=episode_reward,
            revenue=info["episode_revenue"],
            epsilon=agent.exploration_rate,
            loss=np.mean(episode_losses) if episode_losses else None,
            buffer_size=len(agent.replay_buffer),
        )

    logger.export_csv('evaluation/dqn_training_logs.csv')
"""

import os
import csv
import numpy as np


class DQNLogger:
    def __init__(self, rolling_window=100):
        """
        Parameters
        ----------
        rolling_window : int
            Number of recent episodes used to compute the "Average Reward"
            column.
        """
        self.rolling_window = rolling_window
        self.records = []
        self._reward_history = []  # running list, used to compute the rolling average incrementally

    def log(self, episode, episode_reward, revenue, epsilon, loss, buffer_size):
        """
        Records one episode's worth of training metrics. Call this once
        per completed episode, right after the episode ends.

        Parameters
        ----------
        episode : int
            Episode number (0-based).
        episode_reward : float
            Total RL reward earned during this episode. NOT the same as
            `revenue` in this environment - reward includes
            discount/unsold penalties and a pacing bonus on top of raw
            revenue (see pricing_env/reward.py).
        revenue : float
            Actual dollar revenue earned during this episode, typically
            read from `info["episode_revenue"]`.
        epsilon : float or None
            The exploration rate used during this episode. None is
            handled gracefully (e.g. when running a non-learning
            placeholder policy that has no epsilon).
        loss : float or None
            Average training loss across this episode's gradient steps.
            None is handled gracefully (e.g. very early episodes before
            the replay buffer has enough data to train on yet).
        buffer_size : int
            Current number of transitions stored in the replay buffer
            at the time this episode ended.
        """
        self._reward_history.append(episode_reward)
        recent = self._reward_history[-self.rolling_window:]
        average_reward = float(np.mean(recent))

        self.records.append({
            "episode_number": episode,
            "episode_reward": episode_reward,
            "average_reward": average_reward,
            "revenue": revenue,
            "epsilon_value": epsilon,
            "training_loss": loss,
            "replay_buffer_size": buffer_size,
        })

    def export_csv(self, filepath):
        """
        Writes all logged records to a CSV file, creating parent
        directories if needed. This is the "log file generated
        automatically" deliverable.
        """
        if not self.records:
            raise ValueError("No episodes have been logged yet - nothing to export.")

        parent = os.path.dirname(filepath)
        if parent:
            os.makedirs(parent, exist_ok=True)

        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.records[0].keys())
            writer.writeheader()
            writer.writerows(self.records)

        return filepath

    def __len__(self):
        return len(self.records)


if __name__ == "__main__":
    # Self-test: run a real training/evaluation session through the
    # logger, end to end, against the ACTUAL pricing_env package, and
    # export the actual deliverable CSV (evaluation/dqn_training_logs.csv).
    #
    # NOTE: this uses the random policy as a stand-in, NOT a real DQN
    # agent - there is currently no confirmed-working DQNAgent for the
    # pricing_env package in this codebase. `epsilon` and `loss` are
    # logged as None for every episode, honestly reflecting that no real
    # training is happening. `buffer_size` here tracks the cumulative
    # number of (state, action, reward, next_state) transitions seen so
    # far across the run - a meaningful stand-in even without a real
    # agent, since it's exactly what a real replay buffer would have
    # received had `agent.remember()` been called on each step. Once a
    # working DQN agent exists, swap the random-action line below for
    # `agent.select_action(obs)` / `agent.remember(...)` /
    # `agent.train_step()` and pass its real epsilon/loss/buffer values
    # instead, per the usage example in this file's module docstring.
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

    from pricing_env import PricingEnvironment, PricingEnvConfig

    config = PricingEnvConfig()
    env = PricingEnvironment(config)

    logger = DQNLogger()

    NUM_EPISODES = 300
    print(f"Running a {NUM_EPISODES}-episode self-test of the logger "
          f"(random policy - no DQN agent confirmed for this package yet)...")

    transitions_seen = 0

    for episode in range(NUM_EPISODES):
        obs, info = env.reset(seed=42 + episode)
        terminated = truncated = False
        episode_reward = 0.0

        while not (terminated or truncated):
            action = env.action_space.sample()  # stand-in for agent.select_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            transitions_seen += 1  # stand-in for agent.remember(...) growing the replay buffer

        logger.log(
            episode=episode,
            episode_reward=episode_reward,           # the real RL reward signal
            revenue=info["episode_revenue"],           # the real dollar revenue - genuinely different in this environment
            epsilon=None,   # no real training occurring in this self-test
            loss=None,      # no real training occurring in this self-test
            buffer_size=transitions_seen,
        )

        if (episode + 1) % 100 == 0:
            recent_avg = logger.records[-1]["average_reward"]
            print(f"  Episode {episode + 1}/{NUM_EPISODES} - rolling avg reward: ${recent_avg:.2f}, "
                  f"transitions seen: {transitions_seen}")

    output_path = os.path.join(os.path.dirname(__file__), '..', 'evaluation', 'dqn_training_logs.csv')
    logger.export_csv(output_path)

    print(f"\nLogged {len(logger)} episodes.")
    print(f"Exported to: {output_path}")
    print(f"\nFirst 3 rows:")
    for r in logger.records[:3]:
        print(f"  {r}")
    print(f"\nLast 3 rows:")
    for r in logger.records[-3:]:
        print(f"  {r}")
    print("\nNOTE: epsilon/loss are None above because this self-test used a "
          "random policy, not a real DQN agent. replay_buffer_size tracks "
          "cumulative transitions seen as a mechanical stand-in. Once "
          "agents/dqn_agent.py is confirmed working for pricing_env, pass "
          "its real epsilon/loss/buffer values into log() per the usage "
          "example in this file's module docstring.")