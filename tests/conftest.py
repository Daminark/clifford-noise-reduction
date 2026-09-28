"""Slow-test and Hypothesis configuration for the test suite.

Tests marked ``slow`` are skipped unless ``--runslow`` is given.
The Hypothesis profile is chosen by the ``HYPOTHESIS_PROFILE`` environment
variable: ``dev`` (the default) for quick local runs, ``ci`` for the full
suite.
Neither profile has a per-example deadline, since building and sampling a
Stim circuit takes long enough that a deadline would fail spuriously.
"""

import os

import pytest
from hypothesis import settings

settings.register_profile("dev", max_examples=20, deadline=None)
settings.register_profile("ci", max_examples=200, deadline=None)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "dev"))


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if config.getoption("--runslow"):
        return
    skip_slow = pytest.mark.skip(reason="slow test; use --runslow to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
