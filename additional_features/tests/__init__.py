"""
additional_features.tests package

Pytest suite for the three new feature modules under `additional_features/`.
Matches the existing project's test conventions (see `tests/
test_replay_buffer_integration.py`): pytest, `from __future__ import
annotations`, and a module docstring stating exactly what acceptance
criteria the file validates.

Run the full suite from the repository root with:

    python -m pytest additional_features/tests/ -v

These tests exercise the real project components wherever possible
(the actual `PricingEnvironment`, the actual trained checkpoints in
`agents/checkpoints/`, the actual `evaluation/*.csv` files) rather than
mocks, so a green run here is real evidence the feature modules work
against this specific repository — not just against synthetic fixtures.
"""
