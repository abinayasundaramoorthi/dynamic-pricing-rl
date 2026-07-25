"""
test_replay_buffer_integration.py

Validation tests for issue #70 — "Integrate Experience Replay into the DQN
Training Pipeline."

These tests turn the manual smoke-test check ("run train_dqn.py and eyeball
the logs") into something automated and repeatable. They cover the three
acceptance criteria from the issue directly:

  1. Replay buffer integrated successfully.
  2. Experiences are stored correctly.
  3. Training pipeline executes without integration errors.

Run with:
    python -m pytest tests/test_replay_buffer_integration.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from agents.dqn_agent import DQNAgent, ReplayBuffer, Transition
from configs.dqn_config import DQNConfig, get_default_dqn_config
from pricing_env import PricingEnvironment
from training.train_dqn import build_agent, build_environment, run_training


# --------------------------------------------------------------------------- #
# 1. ReplayBuffer in isolation (push / sample / capacity eviction)
# --------------------------------------------------------------------------- #
def _dummy_transition(i: int) -> Transition:
    return Transition(
        state=np.array([float(i), float(i)]),
        action=i % 3,
        reward=float(i),
        next_state=np.array([float(i + 1), float(i + 1)]),
        done=False,
    )


def test_replay_buffer_stores_and_samples() -> None:
    buffer = ReplayBuffer(capacity=10, seed=0)
    assert len(buffer) == 0

    for i in range(5):
        buffer.push(_dummy_transition(i))
    assert len(buffer) == 5

    batch = buffer.sample(batch_size=3)
    assert len(batch) == 3
    assert all(isinstance(t, Transition) for t in batch)


def test_replay_buffer_evicts_oldest_at_capacity() -> None:
    """Pushing past `capacity` should silently evict the oldest transition,
    never grow unbounded, and never raise."""
    capacity = 5
    buffer = ReplayBuffer(capacity=capacity, seed=0)

    for i in range(capacity * 3):
        buffer.push(_dummy_transition(i))

    assert len(buffer) == capacity
    # the most recently pushed transition must still be present
    stored_rewards = {t.reward for t in buffer._buffer}
    assert float(capacity * 3 - 1) in stored_rewards
    # the earliest transitions must have been evicted
    assert 0.0 not in stored_rewards


# --------------------------------------------------------------------------- #
# 2. Experiences are stored correctly via DQNAgent.remember()
# --------------------------------------------------------------------------- #
@pytest.fixture()
def small_config() -> DQNConfig:
    """A cheap config for fast, deterministic tests -- short episodes,
    tiny buffer/network, low warm-up threshold."""
    base = get_default_dqn_config()
    from dataclasses import replace

    return replace(
        base,
        num_episodes=3,
        max_steps_per_episode=5,
        hidden_layer_sizes=[8],
        batch_size=4,
        replay_buffer_size=100,
        min_replay_size_before_training=8,
        target_update_frequency=5,
        log_every_n_episodes=1,
        num_eval_episodes=2,
        seed=0,
    )


@pytest.fixture()
def env(small_config: DQNConfig) -> PricingEnvironment:
    e = build_environment(small_config)
    yield e
    e.close()


@pytest.fixture()
def agent(small_config: DQNConfig, env: PricingEnvironment) -> DQNAgent:
    return build_agent(small_config, env)


def test_remember_pushes_one_transition_per_call(
    agent: DQNAgent, env: PricingEnvironment
) -> None:
    observation, _info = env.reset(seed=0)
    assert len(agent.replay_buffer) == 0

    action = agent.select_action(observation)
    next_observation, reward, terminated, _truncated, _info = env.step(action)
    agent.remember(observation, action, reward, next_observation, terminated)

    assert len(agent.replay_buffer) == 1
    stored = agent.replay_buffer._buffer[0]
    np.testing.assert_array_equal(stored.state, observation)
    np.testing.assert_array_equal(stored.next_state, next_observation)
    assert stored.action == action
    assert stored.reward == reward
    assert stored.done == terminated


def test_remember_copies_arrays_defensively(
    agent: DQNAgent, env: PricingEnvironment
) -> None:
    """Mutating the caller's observation *after* remember() must not corrupt
    the stored transition -- this is what the `.copy()` calls in
    DQNAgent.remember() exist to guarantee."""
    observation, _info = env.reset(seed=0)
    action = agent.select_action(observation)
    next_observation, reward, terminated, _truncated, _info = env.step(action)

    agent.remember(observation, action, reward, next_observation, terminated)
    observation[:] = -999.0
    next_observation[:] = -999.0

    stored = agent.replay_buffer._buffer[0]
    assert not np.any(stored.state == -999.0)
    assert not np.any(stored.next_state == -999.0)


# --------------------------------------------------------------------------- #
# 3. Training pipeline executes end-to-end without integration errors
# --------------------------------------------------------------------------- #
def test_train_step_is_noop_below_warmup_threshold(agent: DQNAgent) -> None:
    assert agent.train_step() is None


def test_train_step_returns_loss_once_buffer_is_warm(
    agent: DQNAgent, env: PricingEnvironment, small_config: DQNConfig
) -> None:
    observation, _info = env.reset(seed=0)
    for _ in range(small_config.min_replay_size_before_training):
        action = agent.select_action(observation)
        next_observation, reward, terminated, truncated, _info = env.step(action)
        agent.remember(observation, action, reward, next_observation, terminated)
        observation = next_observation
        if terminated or truncated:
            observation, _info = env.reset(seed=0)

    loss = agent.train_step()
    assert loss is not None
    assert isinstance(loss, float)
    assert loss >= 0.0


def test_full_training_loop_runs_without_error(
    env: PricingEnvironment, small_config: DQNConfig, agent: DQNAgent
) -> None:
    """End-to-end pipeline check: environment -> replay buffer -> agent,
    for a few short episodes, exactly as issue #70 asks to validate."""
    rewards, revenues = run_training(env, small_config, agent)

    assert len(rewards) == small_config.num_episodes
    assert len(revenues) == small_config.num_episodes
    assert all(np.isfinite(r) for r in rewards)
    # buffer should have accumulated real experience, not stayed empty
    assert len(agent.replay_buffer) > 0
    assert len(agent.replay_buffer) <= small_config.replay_buffer_size