import itertools
import numpy as np
from adsg_core.optimization.assign_enc.matrix import *
from adsg_core.optimization.assign_enc.lazy.imputation.first import *
from adsg_core.optimization.assign_enc.lazy.encodings.conn_idx import *
from adsg_core.optimization.assign_enc.lazy.encodings.direct_matrix import *


def test_imputer():
    encoder = LazyDirectMatrixEncoder(LazyFirstImputer())
    encoder.set_settings(MatrixGenSettings(src=[Node([0, 1]), Node([0, 1])], tgt=[Node([0, 1])]))

    dv, mat = encoder.get_matrix([1, 1])
    assert np.all(dv == [0, 0])
    assert np.all(mat == np.array([[0], [0]]))

    encoder.set_settings(MatrixGenSettings(src=[Node([0, 1]), Node([0, 1])], tgt=[Node([0, 1]), Node(min_conn=1)]))

    dv, mat = encoder.get_matrix([1, 1, 1, 1])
    assert np.all(dv == [0, 1, 0, 0])
    assert np.all(mat == np.array([[0, 1], [0, 0]]))


def test_one_to_one(gen_one_per_existence: AggregateAssignmentMatrixGenerator):
    g = gen_one_per_existence
    encoder = LazyConnIdxMatrixEncoder(LazyFirstImputer(), FlatConnCombsEncoder())
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
    encoder = LazyDirectMatrixEncoder(LazyFirstImputer())
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
