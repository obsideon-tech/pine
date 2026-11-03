import os
from pathlib import Path

import pytest
from hypothesis import HealthCheck, settings

# Hypothesis profiles (PINE_PLAN.md §7.3, §10 D2), chosen with HYPOTHESIS_PROFILE:
# - "dev" (default): Hypothesis defaults, random examples.
# - "ci": derandomized, so a CI failure reproduces exactly on rerun and locally.
# - "thorough": random seeds and more examples, for the weekly scheduled run that keeps
#   searching for counterexamples. Tests that set max_examples explicitly keep their count
#   but still draw new examples each run.
# Every profile prints the reproduction blob on failure.
settings.register_profile("dev", print_blob=True)
settings.register_profile(
    "ci", derandomize=True, print_blob=True, suppress_health_check=[HealthCheck.too_slow]
)
settings.register_profile("thorough", max_examples=1000, print_blob=True, deadline=None)
settings.load_profile(os.getenv("HYPOTHESIS_PROFILE", "dev"))


FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES
