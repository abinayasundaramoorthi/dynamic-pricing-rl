"""
dqn_dashboard.py

Generates visualizations to monitor DQN training progress from a training
log CSV (produced by utils/dqn_logger.py, e.g. evaluation/dqn_training_logs.csv).

Produces 5 charts, one per tracked metric:
  1. Episode Reward       - raw per-episode reward, every episode
  2. Average Reward       - the rolling-average column already computed
                             by DQNLogger, smoothing out episode noise
  3. Training Loss        - average loss per episode
  4. Exploration Rate     - epsilon decay curve
  5. Replay Buffer Growth - how many transitions have accumulated over time

Each chart is saved as its own PNG file under an output directory, ready
to drop into a sprint report or slide deck.

Usage
-----
    python -m dashboard.dqn_dashboard
    python -m dashboard.dqn_dashboard --csv path/to/logs.csv --output-dir path/to/plots
"""

import os
import argparse

import pandas as pd
import matplotlib.pyplot as plt


def load_training_log(csv_path: str) -> pd.DataFrame:
    """
    Loads a training log CSV (as produced by utils/dqn_logger.py) into a
    DataFrame. Raises a clear error if the file doesn't exist or is
    missing expected columns, rather than failing with a cryptic pandas
    error deep inside a plotting function.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Training log not found at '{csv_path}'. "
            f"Run utils/dqn_logger.py first to generate it."
        )

    df = pd.read_csv(csv_path)

    expected_columns = {
        "episode_number", "episode_reward", "average_reward",
        "epsilon_value", "training_loss", "replay_buffer_size",
    }
    missing = expected_columns - set(df.columns)
    if missing:
        raise ValueError(
            f"Training log at '{csv_path}' is missing expected column(s): {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    return df


def _save_or_skip(df: pd.DataFrame, column: str, title: str, ylabel: str,
                   color: str, save_path: str) -> bool:
    """
    Shared plotting helper for the simple line charts below. Returns
    True if a chart was actually saved, False if it was skipped because
    the column had no real data (e.g. epsilon/loss are entirely None
    when the log came from a random-policy self-test rather than real
    DQN training) - skipping rather than saving a blank/misleading chart.
    """
    if df[column].isna().all():
        print(f"  Skipped '{title}': column '{column}' has no data "
              f"(all values are empty - likely a non-training run).")
        return False

    plt.figure(figsize=(9, 5))
    plt.plot(df["episode_number"], df[column], color=color, linewidth=1.2)
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()
    print(f"  Saved: {save_path}")
    return True


def plot_episode_reward(df: pd.DataFrame, output_dir: str) -> str:
    path = os.path.join(output_dir, "episode_reward.png")
    _save_or_skip(df, "episode_reward", "Episode Reward per Episode",
                  "Reward", "#4C72B0", path)
    return path


def plot_average_reward(df: pd.DataFrame, output_dir: str) -> str:
    path = os.path.join(output_dir, "average_reward.png")
    _save_or_skip(df, "average_reward", "Rolling Average Reward",
                  "Average Reward", "#55A868", path)
    return path


def plot_training_loss(df: pd.DataFrame, output_dir: str) -> str:
    path = os.path.join(output_dir, "training_loss.png")
    _save_or_skip(df, "training_loss", "Training Loss per Episode",
                  "Loss (MSE)", "#C44E52", path)
    return path


def plot_exploration_rate(df: pd.DataFrame, output_dir: str) -> str:
    path = os.path.join(output_dir, "exploration_rate.png")
    _save_or_skip(df, "epsilon_value", "Exploration Rate (Epsilon) Decay",
                  "Epsilon", "#8172B2", path)
    return path


def plot_replay_buffer_growth(df: pd.DataFrame, output_dir: str) -> str:
    path = os.path.join(output_dir, "replay_buffer_growth.png")
    _save_or_skip(df, "replay_buffer_size", "Replay Buffer Growth",
                  "Transitions Stored", "#DD8452", path)
    return path


def generate_all_plots(csv_path: str, output_dir: str) -> list:
    """
    Loads the training log and generates all 5 charts, saving each as a
    PNG under `output_dir` (created automatically if it doesn't exist).

    Returns
    -------
    list of str
        Paths to every chart that was actually saved (charts skipped due
        to missing data, per `_save_or_skip`, are not included).
    """
    os.makedirs(output_dir, exist_ok=True)
    df = load_training_log(csv_path)

    print(f"Loaded {len(df)} episodes from '{csv_path}'.")
    print(f"Generating charts into '{output_dir}'...")

    saved_paths = []
    for plot_fn in [
        plot_episode_reward,
        plot_average_reward,
        plot_training_loss,
        plot_exploration_rate,
        plot_replay_buffer_growth,
    ]:
        path = plot_fn(df, output_dir)
        if os.path.exists(path):
            saved_paths.append(path)

    print(f"\n{len(saved_paths)} of 5 charts generated successfully.")
    return saved_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate DQN training visualizations from a training log CSV."
    )
    parser.add_argument(
        "--csv", type=str, default="evaluation/dqn_training_logs.csv",
        help="Path to the training log CSV (default: evaluation/dqn_training_logs.csv).",
    )
    parser.add_argument(
        "--output-dir", type=str, default="evaluation/dqn_training_plots",
        help="Directory to save chart PNGs into (default: evaluation/dqn_training_plots).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_all_plots(args.csv, args.output_dir)


if __name__ == "__main__":
    main()