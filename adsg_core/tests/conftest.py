import pytest
from adsg_core.graph.adsg_nodes import *
from adsg_core.optimization.assign_enc.cache import reset_caches


@pytest.fixture(scope='session', autouse=True)
def reset_cache():
    reset_caches()


@pytest.fixture
def n():
    return [NamedNode(str(i)) for i in range(50)]
