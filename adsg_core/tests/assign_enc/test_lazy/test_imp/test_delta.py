import itertools
import numpy as np
from adsg_core.optimization.assign_enc.matrix import *
from adsg_core.optimization.assign_enc.lazy.imputation.delta import *
from adsg_core.optimization.assign_enc.lazy.encodings.conn_idx import *
from adsg_core.optimization.assign_enc.lazy.encodings.direct_matrix import *


def test_imputer():
    encoder = LazyDirectMatrixEncoder(LazyDeltaImputer())
    encoder.set_settings(MatrixGenSettings(src=[Node([0, 1, 2]), Node(min_conn=0)], tgt=[Node([0, 1]), Node(min_conn=1)]))

    dv = encoder.design_vars
    assert [d.n_opts for d in dv] == [2, 3, 2, 3]

    dv, mat = encoder.get_matrix([1, 2, 1, 0])
    assert np.all(dv == [1, 1, 0, 0])
    assert np.all(mat == np.array([[1, 1], [0, 0]]))


def test_max_tries():
    encoder = LazyDirectMatrixEncoder(LazyDeltaImputer(n_max_tries=10000))
    encoder.set_settings(MatrixGenSettings(src=[Node([1], repeated_allowed=False)],
                                           tgt=[Node([0, 1], repeated_allowed=False) for _ in range(20)]))
    assert encoder.get_imputation_ratio() > 50000

    n_invalid = 0
    for _ in range(5):
        dv_rand = encoder.get_random_design_vector()
        dv, mat = encoder.get_matrix(dv_rand)
        if mat[0, 0] == -1:
            assert np.all(dv == dv_rand)
            assert np.all(mat == -1)
            n_invalid += 1
    assert n_invalid > 0


def test_one_to_one(gen_one_per_existence: AggregateAssignmentMatrixGenerator):
    g = gen_one_per_existence
    encoder = LazyConnIdxMatrixEncoder(LazyDeltaImputer(), FlatConnCombsEncoder())
    encoder.set_settings(g.settings)
    assert len(encoder.design_vars) == 0

    assert encoder.get_n_design_points() == 1
    assert encoder.get_imputation_ratio() == 1.2
    assert encoder.get_information_index() == 1

    for i, existence in enumerate(gen_one_per_existence.existence_patterns.patterns):
        dv, mat = encoder.get_matrix([], existence=existence)
        assert dv == []
        assert mat.shape[0] == (len(gen_one_per_existence.src) if i not in [3, 7] else 0)


def test_different_sizes_bounds_imp():
    patterns = [
        NodeExistence(),
        NodeExistence(src_exists=[True, False], tgt_exists=[True, False]),
    ]
    settings = MatrixGenSettings(src=[Node(min_conn=1) for _ in range(2)], tgt=[Node([0, 1, 2]), Node(min_conn=0)],
                                 existence=NodeExistencePatterns(patterns=patterns))
    encoder = LazyDirectMatrixEncoder(LazyDeltaImputer())
    encoder.set_settings(settings)
    assert len(encoder.design_vars) == 4
    assert [dv.n_opts for dv in encoder.design_vars] == [3, 3, 3, 3]

    for existence in patterns:
        dv_seen = set()
        matrix_seen = set()
        for dv in itertools.product(*[list(range(dv.n_opts)) for dv in encoder.design_vars]):
            dv_imp, matrix = encoder.get_matrix(list(dv), existence=existence)
            dv_seen.add(tuple(dv_imp))
            matrix_seen.add(tuple(matrix.ravel()))
        assert len(dv_seen) == len(matrix_seen)
