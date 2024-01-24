import pytest
import timeit
import itertools
from adsg_core.optimization.assign_enc.matrix import *
from adsg_core.optimization.assign_enc.selector import *
from adsg_core.optimization.assign_enc.cache import reset_caches


@pytest.fixture
def gen_one_per_existence():
    src = [Node([1], repeated_allowed=False) for _ in range(2)]
    tgt = [Node([0, 1], repeated_allowed=False) for _ in range(3)]
    exist = []
    for i_src, i_tgt in itertools.product(list(range(len(src)))+[len(src)], list(range(len(tgt)))+[len(tgt)]):
        exist.append(NodeExistence(src_exists=[i == i_src for i in range(len(src))],
                                   tgt_exists=[i == i_tgt for i in range(len(tgt))]))
    return AggregateAssignmentMatrixGenerator.create(src=src, tgt=tgt, existence=NodeExistencePatterns(exist))


@pytest.fixture(scope='session', autouse=True)
def reset_cache():
    reset_caches()


def do_initialize_numba():
    selector = EncoderSelector(MatrixGenSettings(src=[Node([0, 1])], tgt=[Node([0, 1])]))

    d = None
    for i in range(2):
        EncoderSelector._numba_initialized = False
        t = timeit.default_timer()
        selector.initialize_numba()
        delta_t = timeit.default_timer()-t
        if d is None:
            d = delta_t
        else:
            assert delta_t < d


do_initialize_numba()
