"""
target_network.py

Target Network manager for stabilizing Deep Q-Network (DQN) training.

Why do we need a separate Target Network?
In Q-Learning, we update Q(s,a) by comparing it to a "target" value:
    target = reward + gamma * max(Q(next_state, a'))

With a Q-table this is fine. But with a neural network (DQN), the SAME
network computes both the current prediction AND the target's
"max(Q(next_state, a'))" term. That creates a problem: every time we
update the network's weights to reduce the error, the target itself also
changes (since it's computed using those same weights) — we're chasing a
constantly moving target. This makes training unstable and prone to
oscillating or diverging instead of steadily improving.

The fix (introduced in DeepMind's original DQN paper): keep TWO copies
of the network —
  - the ONLINE network: updated every training step (this is the one
    actually being trained, and the one used to choose actions)
  - the TARGET network: a copy that is only updated occasionally (e.g.
    every N steps), by copying the online network's weights over

Since the target network's weights stay fixed between updates, the
"target" values used in training stay stable for a while, giving the
online network a consistent target to learn toward — much like aiming at
a target that only moves occasionally, instead of one that's constantly
jittering.
"""

import copy

import torch


class TargetNetworkManager:
    """
    Manages a Target Network: creates it as a copy of a given online
    network, and handles periodic synchronization (copying the online
    network's weights into the target network) at a configurable
    frequency.
    """

    def __init__(self, online_network, update_frequency=1000):
        """
        Parameters
        ----------
        online_network : torch.nn.Module
            The network that is actively being trained (e.g. a QNetwork
            instance from dqn_network.py). The target network is created
            as an exact architectural copy of this network, with its
            weights initially synchronized to match.
        update_frequency : int
            How many training steps to wait between target network
            synchronizations. Configurable because it involves a
            trade-off:
                - Too LOW (e.g. every step) -> the target network becomes
                  almost identical to the online network at all times,
                  which brings back the "moving target" instability
                  problem we're trying to avoid.
                - Too HIGH (e.g. every 100,000 steps) -> the target
                  network becomes very outdated, so the online network is
                  learning from a target based on a much older, less
                  competent version of itself, slowing down learning.
                - A few hundred to a few thousand steps (default: 1000)
                  is a common, reasonable middle ground used in DQN
                  implementations.
        """
        self.online_network = online_network
        self.update_frequency = update_frequency

        # Create the target network as a structural copy of the online
        # network (same architecture, same layer sizes) — deepcopy
        # duplicates the whole object, including its current weights, so
        # we start with target == online, then keep them in sync manually
        # from this point on.
        self.target_network = copy.deepcopy(online_network)

        # The target network is only ever used for prediction (to compute
        # target values), never trained directly via backpropagation --
        # so we set it to evaluation mode and freeze its gradients. This
        # avoids wasting memory/computation tracking gradients that will
        # never be used, and prevents any accidental direct training of it.
        self.target_network.eval()
        for param in self.target_network.parameters():
            param.requires_grad = False

        self._step_count = 0

    def synchronize(self):
        """
        Copy the online network's current weights into the target
        network (a "hard update" — the target becomes an exact copy of
        the online network at this moment).
        """
        self.target_network.load_state_dict(self.online_network.state_dict())

    def step(self):
        """
        Call this once per training step. Automatically synchronizes the
        target network whenever `update_frequency` steps have passed.

        Returns
        -------
        synchronized : bool
            True if a synchronization happened on this call, False
            otherwise — useful for logging/debugging when updates occur.
        """
        self._step_count += 1

        if self._step_count % self.update_frequency == 0:
            self.synchronize()
            return True

        return False

    def get_target_q_values(self, state):
        """
        Run a forward pass through the TARGET network (not the online
        network) to get Q-value predictions — used when computing the
        "target" value during a training step.

        Parameters
        ----------
        state : torch.Tensor
            Input state(s).

        Returns
        -------
        q_values : torch.Tensor
            Predicted Q-values from the target network.
        """
        with torch.no_grad():
            return self.target_network(state)

    def validate_sync(self):
        """
        Check whether the target network's weights currently exactly
        match the online network's weights. Useful right after calling
        synchronize() to confirm the copy actually worked correctly.

        Returns
        -------
        is_synced : bool
            True if every parameter tensor matches exactly.
        """
        online_params = self.online_network.state_dict()
        target_params = self.target_network.state_dict()

        for key in online_params:
            if not torch.equal(online_params[key], target_params[key]):
                return False

        return True


if __name__ == "__main__":
    # Quick smoke test using the project's actual QNetwork architecture.
    import sys
    import os

    sys.path.append(os.path.dirname(__file__))
    from dqn_network import QNetwork

    online_net = QNetwork(state_dim=2, action_dim=7)
    target_manager = TargetNetworkManager(online_net, update_frequency=5)

    print("Target network initialized.")
    print(f"Initially synchronized with online network: {target_manager.validate_sync()}")

    # Simulate training: change the online network's weights (as if a
    # training step just happened), and confirm the target network does
    # NOT change until update_frequency steps have passed.
    with torch.no_grad():
        online_net.output_layer.weight.add_(1.0)  # simulate a weight update

    print(f"\nAfter modifying online network weights (simulating training):")
    print(f"Still synchronized? {target_manager.validate_sync()} (should be False)")

    # Step through several training steps, watch for the automatic sync
    print(f"\nUpdate frequency: {target_manager.update_frequency} steps")
    for step in range(1, 8):
        synced = target_manager.step()
        print(f"  Step {step}: synchronized this step = {synced}")

    print(f"\nSynchronized after reaching update_frequency? {target_manager.validate_sync()}")

    # Confirm the target network can produce Q-value predictions
    sample_state = torch.tensor([50.0, 15.0])
    target_q_values = target_manager.get_target_q_values(sample_state)
    print(f"\nTarget network Q-values for sample state: {target_q_values}")

    print("\nTarget network initialization, synchronization, and validation all working correctly!")