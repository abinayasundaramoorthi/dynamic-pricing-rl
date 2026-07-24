"""
Checkpoint Utility

Save and load Deep Q-Network checkpoints.
"""

import os
import torch


def save_checkpoint(
    model,
    optimizer,
    episode,
    reward,
    filepath="checkpoints/dqn_checkpoint.pth"
):
    """
    Save model checkpoint.
    """

    os.makedirs(
        os.path.dirname(filepath),
        exist_ok=True
    )

    torch.save(
        {
            "episode": episode,
            "reward": reward,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict()
        },
        filepath
    )

    print(f"\nCheckpoint saved:")
    print(filepath)


def load_checkpoint(
    model,
    optimizer=None,
    filepath="checkpoints/dqn_checkpoint.pth",
    device="cpu"
):
    """
    Load model checkpoint.
    """

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Checkpoint not found:\n{filepath}"
        )

    checkpoint = torch.load(
        filepath,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    if optimizer is not None:
        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

    print("\nCheckpoint loaded successfully.")

    return (
        checkpoint["episode"],
        checkpoint["reward"]
    )


def checkpoint_exists(
    filepath="checkpoints/dqn_checkpoint.pth"
):
    """
    Check whether checkpoint exists.
    """

    return os.path.exists(filepath)