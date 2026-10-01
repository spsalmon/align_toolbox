import pytest

from tests.data_analysis.proportion_simulation import simulate_proportions


@pytest.fixture(scope="session")
def simulated_proportions():
    return simulate_proportions(seed=1)
