"""
dqn_metrics.py

Evaluation framework for monitoring DQN training performance during
future experiments. Unlike evaluation/metrics.py (which evaluates an
already-trained agent AFTER the fact), this module is designed to be
called DURING training, once per episode, to track how the agent is
progressing in real time.

Metrics tracked per episode:
- Episode Reward       (total reward/revenue for that single episode)
- Average Reward       (rolling average over a configurable window)
- Revenue              (identical to reward in this environment, tracked
                         separately since business revenue and RL reward
                         aren't always the same thing in general)
- Inventory Utilization (% of inventory sold that episode)
- Loss per Episode     (average training loss across that episode's
                         gradient updates - only available for DQN,
                         since tabular Q-Learning has no loss)
- Exploration Rate (Epsilon) (the epsilon value used during that episode)

Usage (inside a training loop):

    monitor = DQNTrainingMonitor(save_dir='evaluation/dqn_runs/run_001')

    for episode in range(num_episodes):
        obs, info = env.reset(seed=seed + episode)
        terminated = truncated = False
        episode_losses = []

        while not (terminated or truncated):
            action = agent.act(obs, episode=episode, training=True)
            next_obs, reward, terminated, truncated, info = env.step(action)
            agent.remember(obs, action, reward, next_obs, terminated)
            loss = agent.train_step()
            if loss is not None:
                episode_losses.append(loss)
            obs = next_obs

        monitor.log_episode(
            episode=episode,
            reward=env.total_revenue,
            units_sold=env.total_inventory - env.inventory_remaining,
            total_inventory=env.total_inventory,
            avg_loss=np.mean(episode_losses) if episode_losses else None,
            epsilon=agent.get_epsilon(episode),
        )

    monitor.save()
    monitor.print_summary()
"""

import os
import csv
import json
import numpy as np


class DQNTrainingMonitor:
    def __init__(self, save_dir, rolling_window=100):
        """
        Parameters
        ----------
        save_dir : str
            Directory where result files (CSV, JSON, plots) will be saved.
            Created automatically if it doesn't exist.
        rolling_window : int
            Window size for the "Average Reward" rolling metric - matches
            the smoothing window convention used in
            notebooks/dqn_training.ipynb for visual consistency.
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.rolling_window = rolling_window

        # One entry per episode, in order - the core data store.
        self.records = []

    def log_episode(self, episode, reward, units_sold, total_inventory,
                     avg_loss=None, epsilon=None):
        """
        Records one episode's worth of metrics. Call this once per
        completed episode during training.

        Parameters
        ----------
        episode : int
            Episode index (0-based).
        reward : float
            Total reward (= total revenue) earned in this episode.
        units_sold : int
            Units sold this episode.
        total_inventory : int
            The environment's total starting inventory, used to compute
            utilization %.
        avg_loss : float or None
            Average training loss across this episode's gradient steps.
            None if no training steps occurred yet (e.g. replay buffer
            still warming up) - handled gracefully, not treated as an error.
        epsilon : float or None
            The exploration rate used during this episode. None if not
            applicable (e.g. evaluating with training=False).
        """
        utilization_pct = (units_sold / total_inventory * 100) if total_inventory else 0.0

        self.records.append({
            "episode": episode,
            "episode_reward": reward,
            "revenue": reward,  # identical in this environment; separate field for clarity/extensibility
            "units_sold": units_sold,
            "inventory_utilization_pct": utilization_pct,
            "loss": avg_loss,
            "epsilon": epsilon,
        })

    def get_rolling_average_reward(self):
        """Returns the rolling-average reward series, for plotting/inspection."""
        rewards = [r["episode_reward"] for r in self.records]
        if len(rewards) < self.rolling_window:
            return []
        return list(np.convolve(rewards, np.ones(self.rolling_window) / self.rolling_window, mode='valid'))

    def get_summary(self):
        """
        Computes aggregate summary statistics across all logged episodes -
        useful for a quick end-of-training printout or for embedding
        into an evaluation report.
        """
        if not self.records:
            return {"error": "No episodes logged yet."}

        rewards = np.array([r["episode_reward"] for r in self.records])
        utilizations = np.array([r["inventory_utilization_pct"] for r in self.records])
        losses = np.array([r["loss"] for r in self.records if r["loss"] is not None])
        epsilons = np.array([r["epsilon"] for r in self.records if r["epsilon"] is not None])

        n = len(rewards)
        first_chunk = max(1, n // 10)  # first 10% of episodes
        last_chunk = max(1, n // 10)   # last 10% of episodes

        return {
            "num_episodes": n,
            "avg_reward_overall": float(np.mean(rewards)),
            "avg_reward_first_10pct": float(np.mean(rewards[:first_chunk])),
            "avg_reward_last_10pct": float(np.mean(rewards[-last_chunk:])),
            "reward_improvement": float(np.mean(rewards[-last_chunk:]) - np.mean(rewards[:first_chunk])),
            "avg_inventory_utilization_pct": float(np.mean(utilizations)),
            "avg_loss_first_10pct": float(np.mean(losses[:max(1, len(losses)//10)])) if len(losses) else None,
            "avg_loss_last_10pct": float(np.mean(losses[-max(1, len(losses)//10):])) if len(losses) else None,
            "final_epsilon": float(epsilons[-1]) if len(epsilons) else None,
        }

    def save(self):
        """Saves both a full per-episode CSV log and a summary JSON file."""
        csv_path = os.path.join(self.save_dir, "episode_log.csv")
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.records[0].keys())
            writer.writeheader()
            writer.writerows(self.records)

        summary_path = os.path.join(self.save_dir, "training_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(self.get_summary(), f, indent=2)

        return csv_path, summary_path

    def print_summary(self):
        """Prints a human-readable summary to the console - handy for a quick sanity check at the end of a training run."""
        s = self.get_summary()
        if "error" in s:
            print(s["error"])
            return

        print(f"{'='*50}")
        print(f"DQN Training Summary ({s['num_episodes']} episodes)")
        print(f"{'='*50}")
        print(f"Avg reward (overall):        ${s['avg_reward_overall']:.2f}")
        print(f"Avg reward (first 10%):      ${s['avg_reward_first_10pct']:.2f}")
        print(f"Avg reward (last 10%):       ${s['avg_reward_last_10pct']:.2f}")
        print(f"Reward improvement:          ${s['reward_improvement']:.2f}")
        print(f"Avg inventory utilization:   {s['avg_inventory_utilization_pct']:.1f}%")
        if s['avg_loss_first_10pct'] is not None:
            print(f"Avg loss (first 10%):        {s['avg_loss_first_10pct']:.4f}")
            print(f"Avg loss (last 10%):         {s['avg_loss_last_10pct']:.4f}")
        if s['final_epsilon'] is not None:
            print(f"Final epsilon:               {s['final_epsilon']:.4f}")


if __name__ == "__main__":
    # Self-test: run a SHORT real DQN training session through the full
    # monitoring framework, to confirm logging, summary computation, and
    # saving all work correctly end-to-end (satisfies "ready for
    # integration with DQN training").
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from environment import DynamicPricingEnv
    from agents.dqn_agent import DQNAgent, make_state_normalizer

    env = DynamicPricingEnv()
    normalizer = make_state_normalizer(env.selling_window, env.total_inventory)
    agent = DQNAgent(state_dim=2, num_actions=env.action_space.n, state_normalizer=normalizer,
                      epsilon_decay_episodes=100)

    monitor = DQNTrainingMonitor(save_dir=os.path.join(os.path.dirname(__file__), 'dqn_runs', 'self_test'))

    NUM_TEST_EPISODES = 150  # short run, just to prove the framework works end-to-end
    print(f"Running a {NUM_TEST_EPISODES}-episode self-test of the monitoring framework...")

    for episode in range(NUM_TEST_EPISODES):
        obs, info = env.reset(seed=42 + episode)
        terminated = truncated = False
        episode_losses = []

        while not (terminated or truncated):
            action = agent.act(obs, episode=episode, training=True)
            next_obs, reward, terminated, truncated, info = env.step(action)
            agent.remember(obs, action, reward, next_obs, terminated)
            loss = agent.train_step()
            if loss is not None:
                episode_losses.append(loss)
            obs = next_obs

        monitor.log_episode(
            episode=episode,
            reward=env.total_revenue,
            units_sold=env.total_inventory - env.inventory_remaining,
            total_inventory=env.total_inventory,
            avg_loss=float(np.mean(episode_losses)) if episode_losses else None,
            epsilon=agent.get_epsilon(episode),
        )

    csv_path, summary_path = monitor.save()
    monitor.print_summary()
    print(f"\nSaved episode log to: {csv_path}")
    print(f"Saved summary to: {summary_path}")