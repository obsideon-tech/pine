"""Example budgets for property tests, scaled by the Hypothesis profile (tests/conftest.py).

Tests write ``@settings(max_examples=examples(60))``. Under ``HYPOTHESIS_PROFILE=thorough``
(the weekly scheduled run) every budget is multiplied by ``THOROUGH_SCALE``; an explicit
``max_examples`` would otherwise override the profile and the weekly run would do no more
work than CI.
"""

import os

THOROUGH_SCALE = 10


def examples(n: int) -> int:
    return n * THOROUGH_SCALE if os.getenv("HYPOTHESIS_PROFILE") == "thorough" else n
