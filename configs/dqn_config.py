"""
DQN Configuration

Contains reusable configuration parameters
for Deep Q-Network training.
"""


class DQNConfig:
    """Configuration values for DQN."""

    # -------------------------------
    # Exploration Parameters
    # -------------------------------

    INITIAL_EPSILON = 1.0

    MIN_EPSILON = 0.05

    EPSILON_DECAY = 0.995

    # -------------------------------
    # DQN Hyperparameters
    # (Future Integration)
    # -------------------------------

    GAMMA = 0.99

    LEARNING_RATE = 0.001

    BATCH_SIZE = 64

    TARGET_UPDATE = 100

    MEMORY_SIZE = 10000

    RANDOM_SEED = 42

    DEVICE = "auto"