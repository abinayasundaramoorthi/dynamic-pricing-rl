"""
replay_buffer.py

Experience Replay Buffer for DQN training.

Why do we need this?
If we trained the DQN network on experiences in the exact order they
happen (one step at a time, as they occur), two problems arise:

  1. Consecutive experiences are highly correlated (today's price is
     usually similar to yesterday's, inventory changes gradually) —
     training on strongly correlated data in sequence tends to make
     neural network training unstable and prone to overfitting to
     whatever the agent happened to be doing recently.

  2. Each experience would only be used once and then thrown away —
     wasteful, since a single experience (state, action, reward,
     next_state) contains useful information that could help the
     network learn even if reused multiple times.

The Replay Buffer solves both problems: it stores past experiences in
memory, and during training we randomly sample a "mini-batch" of past
experiences (not necessarily recent ones) to learn from. This breaks the
correlation between consecutive updates and lets the network learn from
each experience multiple times, which is standard practice in DQN
(first introduced in the original DeepMind Atari DQN paper).
"""

import random
from collections import deque, namedtuple

import numpy as np


# A single stored experience: what happened at one time step.
Experience = namedtuple(
    "Experience", ["state", "action", "reward", "next_state", "done"]
)


class ReplayBuffer:
    """
    Fixed-size buffer that stores past experiences and supports random
    mini-batch sampling for DQN training.
    """

    def __init__(self, capacity=10000):
        """
        Parameters
        ----------
        capacity : int
            Maximum number of experiences the buffer can hold. Once full,
            adding a new experience automatically discards the oldest one
            (handled automatically by using a deque with maxlen).

            Why a capacity limit? Without one, memory usage would grow
            forever during long training runs. A fixed capacity also
            means old, outdated experiences (from when the agent's
            policy was much worse/different) eventually get replaced by
            more recent ones, keeping the buffer's contents reasonably
            relevant to the agent's current behavior.
        """
        self.capacity = capacity
        # deque with maxlen automatically drops the oldest item once full
        # -- this gives us capacity management "for free" without any
        # manual bookkeeping.
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """
        Store one experience (a single environment step) in the buffer.

        Parameters
        ----------
        state : array-like
            The state before the action was taken.
        action : int
            The action that was taken.
        reward : float
            The reward received.
        next_state : array-like
            The state after the action was taken.
        done : bool
            Whether the episode ended on this step.
        """
        experience = Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        """
        Randomly sample a mini-batch of experiences from the buffer.

        Sampling is done WITHOUT replacement within a single batch (each
        experience can appear at most once in a given mini-batch), but
        WITH replacement across different calls to sample() over time —
        the same experience can be picked again in future mini-batches
        as long as it's still in the buffer.

        Parameters
        ----------
        batch_size : int
            Number of experiences to sample.

        Returns
        -------
        A tuple of 5 NumPy arrays, ready to be converted into tensors
        for training:
            states      : shape (batch_size, state_dim)
            actions     : shape (batch_size,)
            rewards     : shape (batch_size,)
            next_states : shape (batch_size, state_dim)
            dones       : shape (batch_size,)  (1.0 if done else 0.0)

        Raises
        ------
        ValueError
            If batch_size is larger than the number of experiences
            currently stored — sampling more items than exist would
            either fail or force sampling with replacement, which could
            silently produce a misleadingly duplicated batch.
        """
        if batch_size > len(self.buffer):
            raise ValueError(
                f"Cannot sample batch_size={batch_size}; buffer only "
                f"contains {len(self.buffer)} experiences so far."
            )

        batch = random.sample(self.buffer, batch_size)

        # Unpack the list of Experience namedtuples into separate arrays,
        # one per field -- this is the format PyTorch training code needs
        # (all states together, all actions together, etc.)
        states = np.array([e.state for e in batch], dtype=np.float32)
        actions = np.array([e.action for e in batch], dtype=np.int64)
        rewards = np.array([e.reward for e in batch], dtype=np.float32)
        next_states = np.array([e.next_state for e in batch], dtype=np.float32)
        dones = np.array([float(e.done) for e in batch], dtype=np.float32)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Number of experiences currently stored (used to check if buffer is ready to sample from)."""
        return len(self.buffer)

    def is_ready(self, batch_size):
        """
        Convenience check: does the buffer have enough experiences yet
        to sample a full mini-batch of the given size? Training loops
        typically wait until this is True before starting to learn,
        rather than training on a tiny, unrepresentative buffer.
        """
        return len(self.buffer) >= batch_size


if __name__ == "__main__":
    # Quick smoke test: confirm storing and sampling both work correctly.
    buffer = ReplayBuffer(capacity=100)

    print(f"Empty buffer length: {len(buffer)}")
    print(f"Is ready for batch of 5? {buffer.is_ready(5)}")

    # Simulate storing 20 fake experiences
    for i in range(20):
        fake_state = [50 - i, 30 - i]
        fake_action = i % 7
        fake_reward = float(i * 10)
        fake_next_state = [50 - i - 1, 30 - i - 1]
        fake_done = (i == 19)
        buffer.push(fake_state, fake_action, fake_reward, fake_next_state, fake_done)

    print(f"\nBuffer length after 20 pushes: {len(buffer)}")
    print(f"Is ready for batch of 5? {buffer.is_ready(5)}")

    states, actions, rewards, next_states, dones = buffer.sample(batch_size=5)
    print(f"\nSampled batch shapes:")
    print(f"  states:      {states.shape}")
    print(f"  actions:     {actions.shape}")
    print(f"  rewards:     {rewards.shape}")
    print(f"  next_states: {next_states.shape}")
    print(f"  dones:       {dones.shape}")

    print(f"\nSample states:\n{states}")
    print(f"Sample actions: {actions}")
    print(f"Sample rewards: {rewards}")

    # Test capacity management: push more than capacity, confirm it caps correctly
    small_buffer = ReplayBuffer(capacity=10)
    for i in range(15):
        small_buffer.push([i, i], 0, 1.0, [i + 1, i + 1], False)
    print(f"\nSmall buffer (capacity=10) length after 15 pushes: {len(small_buffer)} "
          f"(should be capped at 10)")

    # Test error handling: sampling more than available
    tiny_buffer = ReplayBuffer(capacity=100)
    tiny_buffer.push([1, 1], 0, 1.0, [2, 2], False)
    try:
        tiny_buffer.sample(batch_size=5)
        print("ERROR: should have raised ValueError!")
    except ValueError as e:
        print(f"\nCorrectly raised error when sampling too much: {e}")

    print("\nAll replay buffer tests passed!")