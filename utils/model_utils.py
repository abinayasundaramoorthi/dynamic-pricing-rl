"""
Model Utility Functions

Reusable utility functions for Deep Q-Network (DQN) training.

Features:
- Random seed initialization
- Device configuration
- Neural network weight initialization
"""

import random
import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int = 42):
    """
    Initialize random seeds for reproducibility.
    """

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Random seed initialized: {seed}")


def get_device():
    """
    Select CPU or GPU automatically.
    """

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    return device


def initialize_weights(model):
    """
    Initialize neural network weights using Xavier initialization.
    """

    for layer in model.modules():

        if isinstance(layer, nn.Linear):

            nn.init.xavier_uniform_(layer.weight)

            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    print("Model weights initialized successfully.")


def count_parameters(model):
    """
    Return total trainable parameters.
    """

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def print_model_summary(model):
    """
    Display model information.
    """

    print("\n========== Model Summary ==========\n")

    print(model)

    print(
        f"\nTrainable Parameters : "
        f"{count_parameters(model):,}"
    )

    print("\n===================================\n")