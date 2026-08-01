"""
baselines package

Naive / heuristic pricing strategies used as the comparison floor for the
learned agents (tabular Q-Learning, DQN). Every policy in this package
exposes the same minimal interface — `select_greedy_action(observation)`
and `reset()` — so `evaluation/evaluate_policies.py` can drive them
through the exact same evaluation loop it uses for the RL agents, with no
special-casing. See `PolicyAdapter` in `evaluation/evaluate_policies.py`
for how a bare Python callable (RL agent or baseline) is wrapped into a
uniform `Policy` protocol.

Three baselines are implemented here, matching the Week 2 milestones
(reports/project_planning/problem_statement.md, Section 21) and issue #90's
integration requirement:

  - RandomPolicy            (baseline_random.py)   — uniform random action
  - FixedPricePolicy        (baseline_fixed.py)    — never move off base_price
  - TimeBasedDiscountPolicy (baseline_timebased.py) — rule-based discount
    schedule driven by days-remaining and sell-through pacing

None of these baselines are trained — they need no checkpoint, no replay
buffer, and no exploration schedule. They exist purely to answer "is the
learned policy actually better than what a revenue manager would do with
no ML at all?"
"""

from .baseline_fixed import FixedPricePolicy
from .baseline_random import RandomPolicy
from .baseline_timebased import TimeBasedDiscountPolicy

__all__ = [
    "RandomPolicy",
    "FixedPricePolicy",
    "TimeBasedDiscountPolicy",
]