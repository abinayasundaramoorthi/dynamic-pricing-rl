# DQN Training Architecture

**Issue:** #62 — Design and Integrate the Deep Q-Network Training Architecture
**Status:** Implemented and verified
**Prerequisite reading:** `reports/Problem_Statement.md` (Section 6, MDP formulation), `agents/q_learning.py` (the tabular agent this replaces for Phase 2 scale)

---

## 1. Why DQN, and why now

The Week 2 tabular Q-Learning agent (`agents/q_learning.py`) works well because Phase 1's state space is small: `[remaining_inventory, days_remaining]` has roughly 3,000 reachable combinations. That was an explicit, documented design choice — a dict-keyed Q-table is simple, fast to train, and trivially interpretable.

It does not scale. The project's own roadmap (`reports/Problem_Statement.md`, Section 4.3, Future Enhancements) names several Phase 2 state variables — competitor price, customer segment, seasonality — that would each multiply the size of the state space. A tabular Q-table over a combined state space with even a few of these added would become impractical to enumerate, store, or get sufficient visit counts for. A **function approximator** (a neural network estimating Q-values directly from state features) doesn't have this limitation: it generalizes across similar states instead of requiring every one to be visited individually.

DQN (Mnih et al., 2015) is the standard, well-understood entry point into that family of methods, which is why it's introduced here ahead of continuous-control alternatives (PPO/SAC, already flagged as Phase 2 work) — it's the smallest step up from tabular Q-Learning that still requires solving the two core stability problems any value-based deep RL method has to solve.

---

## 2. Review of the existing Q-Learning workflow

Before designing DQN's architecture, the existing tabular pipeline was reviewed for what to keep and what necessarily changes:

| Aspect | Q-Learning (`agents/q_learning.py`) | DQN (`agents/dqn_agent.py`) | Kept the same? |
|---|---|---|---|
| Action selection | Epsilon-greedy over `q_table[state]` | Epsilon-greedy over `network(state)` | **Yes** — same `select_action()` / `select_greedy_action()` interface and semantics |
| Exploration decay | Multiplicative, per episode, floored at `exploration_min` | Identical | **Yes** — same `decay_exploration()` method |
| Learning update | Immediate, from a single transition (`update()`) | Batched, from a sampled minibatch (`train_step()`), gated by a replay-buffer warm-up | **No** — this is the fundamental difference; see Section 4 |
| Value representation | `dict[(inventory, days)] -> np.ndarray[num_actions]` | `QNetwork` (MLP) mapping normalized state -> Q-values | **No** — replaced by a function approximator |
| Persistence | `pickle` of the Q-table dict + JSON metadata sidecar | `torch.save` of the network's `state_dict` + JSON metadata sidecar | **Yes** — same two-file pattern (binary checkpoint + human-readable sidecar) |
| Config style | Frozen dataclass, validated in `__post_init__`, `get_*_config()` factory function | Identical pattern, separate dataclass | **Yes** — deliberately mirrored, see Section 3 |

The takeaway: everything about *how an agent's behavior is invoked and inspected* carries over unchanged. Everything about *how the agent's values are represented and updated* has to change, because that's precisely the part tabular methods can't scale.

---

## 3. Configuration design (`configs/dqn_config.py`)

`DQNConfig` is a **separate** frozen dataclass from `TrainingConfig`, not a subclass or shared base. This was a deliberate choice: Q-Learning and DQN have genuinely different parameter sets. DQN needs network architecture (`hidden_layer_sizes`), a replay buffer (`replay_buffer_size`, `min_replay_size_before_training`), a batch size, and a target-network update cadence — none of which tabular Q-Learning has any use for. Sharing one config class would mean each agent type carries fields meaningless to it, which is worse for readability than two small, focused configs.

What *is* shared deliberately: the same environment configuration (`initial_inventory=100`, `selling_horizon_days=30`, `base_price=200.0`) as `training_config.get_final_training_config()`. This is what makes a later Q-Learning-vs-DQN comparison an apples-to-apples comparison of the *learning algorithms*, rather than a comparison confounded by two different problem instances.

`get_default_dqn_config()` follows the same "one canonical function, not scattered `DQNConfig()` calls" pattern already established by `training_config.get_default_training_config()`.

---

## 4. Why DQN needs its own training loop (`training/train_dqn.py`)

`training/train_agent.py`'s `run_training()` loop was built around tabular Q-Learning's update contract: **one transition in, one Q-table cell updated, immediately.** DQN's update rule is structurally different in a way that doesn't fit that contract:

- A single transition isn't enough context for a stable gradient step — DQN needs a **minibatch sampled from a history of past transitions** (experience replay), specifically to break the strong temporal correlation between consecutive steps of the same episode, which would otherwise cause the network to overfit to whatever narrow part of the state space that episode happened to visit.
- DQN needs a **warm-up period** — no gradient steps at all until the replay buffer holds enough transitions (`min_replay_size_before_training`) to sample a reasonably diverse minibatch from.
- DQN needs a **target network**, synced from the online network only every `target_update_frequency` gradient steps — without it, the TD target moves every single gradient step (the network bootstraps against itself), a well-documented cause of divergence.

Retrofitting `run_training()` to handle all of this via optional parameters would have made a currently-simple, single-transition loop much harder to read for the tabular case, to serve a training pattern the tabular agent doesn't need. A separate entry point keeps both loops legible for what they actually do. Everything *around* the loop — environment construction, Gymnasium compliance verification, train → save → reload-from-disk → evaluate — deliberately mirrors `train_agent.py`'s structure exactly, so the two pipelines remain easy to compare and maintain side by side.

---

## 5. Network architecture (`agents/dqn_agent.py::QNetwork`)

A small feedforward MLP: `Linear(2, 64) -> ReLU -> Linear(64, 64) -> ReLU -> Linear(64, num_actions=7)`.

This is deliberately small. The observation is a 2-dimensional vector (`remaining_inventory`, `days_remaining`), not an image or a high-dimensional sensor reading — there isn't enough underlying structure in two numbers to justify a larger network, and a larger one would only add unnecessary variance to training. `hidden_layer_sizes` is configurable (`DQNConfig.hidden_layer_sizes`, default `[64, 64]`) specifically so this can be widened later if Phase 2 adds more state variables, without a code change.

**Input normalization** matters here in a way it never did for the tabular agent: `remaining_inventory` ranges up to hundreds while `days_remaining` ranges up to tens, and an unnormalized network would let the larger-magnitude feature dominate early gradients purely due to scale. `DQNAgent._normalize()` min-max scales both to roughly `[0, 1]` using the environment's own `observation_space.low`/`.high` bounds before every forward pass.

---

## 6. Stabilization mechanisms

| Mechanism | Why it's needed |
|---|---|
| Experience replay (`ReplayBuffer`, `deque(maxlen=...)`, uniform random sampling) | Breaks temporal correlation between consecutive transitions; a network trained directly on sequential steps would overfit to one episode's narrow trajectory |
| Target network, synced every `target_update_frequency` gradient steps | Keeps the TD target fixed between syncs, instead of a target that moves every gradient step (bootstrapping against a constantly-shifting reference) |
| Replay warm-up (`min_replay_size_before_training`) | Avoids training on a near-empty, low-diversity buffer, a documented source of early instability |
| Huber loss (`F.smooth_l1_loss`), not MSE | More robust to the occasional large-magnitude TD error — this environment's reward can range from a routine sale to a large terminal unsold-inventory penalty, and a squared loss would let outliers dominate the gradient |
| Gradient norm clipping (`grad_clip_norm`, default 10.0) | Additional guard against any single large gradient destabilizing the network |

---

## 7. Persistence and evaluation

Same two-artifact pattern as `QLearningAgent`: `agents/checkpoints/dqn_policy.pt` (the online network's `state_dict` plus hyperparameters, loadable via `torch.load`) and `dqn_policy.pt.json` (a human-readable sidecar — architecture, hyperparameters, gradient steps taken, timestamp — so a reviewer can sanity-check what was trained without touching the binary checkpoint).

`DQNAgent.load()` validates that the saved policy's action count *and* observation shape match the environment being evaluated against, raising `ValueError` on a mismatch — loading a network trained against a differently-shaped environment would otherwise silently produce meaningless Q-values, since the network's input/output dimensions would no longer correspond to the environment's actual semantics.

`train_dqn.py`'s `main()` follows the exact same train → save → **reload from a fresh instance** → evaluate sequence as `train_agent.py`, for the same reason: evaluating the in-memory agent object would only prove training ran, not that the save/load round-trip actually produces a usable file.

---

## 8. Verification performed

- `pyflakes` clean across `pricing_env/`, `agents/`, `configs/`, `training/`.
- Full pipeline run (300 episodes, `min_replay_size_before_training=200` for a fast test): 7,323 gradient steps taken, training-episode revenue trended from a mean of $13,895 (first 50 episodes) to $16,091 (last 50 episodes).
- Reloaded (from-disk) greedy policy evaluated over 300 held-out episodes: **mean revenue $19,547**, notably above the tabular Q-Learning agent's evaluated result on the same environment configuration.
- Save/load mismatch safety: both a wrong action-space size and a wrong observation shape are correctly rejected with a clear `ValueError` on load.
- `DQNConfig` validation: confirmed to reject invalid `batch_size`, an inconsistent `min_replay_size_before_training`/`batch_size` pair, an unsupported `device`, empty `hidden_layer_sizes`, and non-positive `target_update_frequency`.
- Coexistence: `training/train_agent.py` (Q-Learning) re-tested after this work and confirmed unaffected.

## 9. Open items for Week 3+

- No hyperparameter tuning has been performed for DQN yet — `get_default_dqn_config()`'s values (`learning_rate=1e-3`, `batch_size=64`, `target_update_frequency=500`, etc.) are standard starting points from the DQN literature, the same honest caveat already on record for the Q-Learning agent's `get_final_training_config()`.
- No direct, apples-to-apples Q-Learning vs. DQN comparison report exists yet (both now share the same environment configuration specifically to enable one) — this is natural follow-up work for the evaluation harness.
- `evaluation/` still has no standalone multi-policy comparison script; both agents currently only support in-pipeline evaluation via their own `train_*.py` entry point.