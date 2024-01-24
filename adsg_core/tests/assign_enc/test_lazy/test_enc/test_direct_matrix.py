import itertools
import numpy as np
from adsg_core.optimization.assign_enc.matrix import *
from adsg_core.optimization.assign_enc.lazy.imputation.delta import *
from adsg_core.optimization.assign_enc.lazy.imputation.first import *
from adsg_core.optimization.assign_enc.lazy.encodings.direct_matrix import *
from adsg_core.optimization.assign_enc.lazy.imputation.constraint_violation import *
from tests.test_lazy_encoding import check_lazy_conditionally_active


def test_encoding():
    encoder = LazyDirectMatrixEncoder(LazyConstraintViolationImputer())
    settings = MatrixGenSettings(src=[Node([0, 1, 2]), Node(min_conn=0)], tgt=[Node([0, 1]), Node(min_conn=1)])
    assert np.all(settings.get_max_conn_matrix() == np.array([
        [1, 2],
        [1, 2],
    ]))

    encoder.set_settings(settings)
    assert encoder._matrix_gen.count_all_matrices() == 21

    dv = encoder.design_vars
    assert len(dv) == 4
    assert [d.n_opts for d in dv] == [2, 3, 2, 3]

    _, mat = encoder.get_matrix([0, 1, 0, 0])
    assert np.all(mat == np.array([[0, 1], [0, 0]]))

    _, mat = encoder.get_matrix([0, 1, 0, 1])
    assert np.all(mat == np.array([[0, 1], [0, 1]]))

    _, mat = encoder.get_matrix([0, 2, 1, 1])
    assert np.all(mat == np.array([[0, 2], [1, 1]]))

    assert encoder.get_n_design_points() == 36
    assert encoder.get_imputation_ratio() == 36/21
    assert encoder.get_distance_correlation()

    check_lazy_conditionally_active(encoder)


def test_encoder_excluded():
    encoder = LazyDirectMatrixEncoder(LazyConstraintViolationImputer())
    src = [Node([0, 1, 2]), Node(min_conn=0)]
    tgt = [Node([0, 1]), Node(min_conn=1)]
    encoder.set_settings(MatrixGenSettings(src=src, tgt=tgt, excluded=[(src[1], tgt[0])]))

    dv = encoder.design_vars
    assert len(dv) == 3
    assert [d.n_opts for d in dv] == [2, 3, 3]

    _, mat = encoder.get_matrix([0, 1, 0])
    assert np.all(mat == np.array([[0, 1], [0, 0]]))

    _, mat = encoder.get_matrix([0, 2, 1])
    assert np.all(mat == np.array([[0, 2], [0, 1]]))

    check_lazy_conditionally_active(encoder)


def test_encoder_no_repeat():
    encoder = LazyDirectMatrixEncoder(LazyConstraintViolationImputer())
    src = [Node([0, 1, 2]), Node(min_conn=0, repeated_allowed=False)]
    tgt = [Node([0, 1], repeated_allowed=False), Node(min_conn=1)]
    settings = MatrixGenSettings(src=src, tgt=tgt)
    assert np.all(settings.get_max_conn_matrix() == np.array([
        [1, 2],
        [1, 1],
    ]))
    encoder.set_settings(settings)

    dv = encoder.design_vars
    assert len(dv) == 4
    assert [d.n_opts for d in dv] == [2, 3, 2, 2]

    _, mat = encoder.get_matrix([0, 1, 0, 0])
    assert np.all(mat == np.array([[0, 1], [0, 0]]))

    _, mat = encoder.get_matrix([0, 2, 0, 1])
    assert np.all(mat == np.array([[0, 2], [0, 1]]))

    check_lazy_conditionally_active(encoder)


def test_encoder_existence():
    encoder = LazyDirectMatrixEncoder(LazyDeltaImputer())
    src = [Node([0, 1, 2]), Node(min_conn=0, repeated_allowed=False)]
    tgt = [Node([0, 1], repeated_allowed=False), Node(min_conn=1)]
    exist = NodeExistencePatterns([
        NodeExistence(),
        NodeExistence(tgt_exists=[True, False]),
    ])
    encoder.set_settings(MatrixGenSettings(src, tgt, existence=exist))

    assert len(encoder._existence_design_vars) == 2
    assert len(encoder._existence_design_vars[exist.patterns[0]]) == 4
    assert len(encoder._existence_design_vars[exist.patterns[1]]) == 2

    dv = encoder.design_vars
    assert len(dv) == 4
    assert [d.n_opts for d in dv] == [2, 3, 2, 2]

    _, mat = encoder.get_matrix([0, 1, 0, 0])
    assert np.all(mat == np.array([[0, 1], [0, 0]]))
    _, mat = encoder.get_matrix([0, 1, 0, 0], existence=NodeExistence())
    assert np.all(mat == np.array([[0, 1], [0, 0]]))
    _, mat = encoder.get_matrix([0, 1, 0, 0], existence=exist.patterns[0])
    assert np.all(mat == np.array([[0, 1], [0, 0]]))

    _, mat = encoder.get_matrix([1, 0, 0, 0], existence=exist.patterns[0])
    assert np.all(mat == np.array([[1, 0], [0, 1]]))
    _, mat = encoder.get_matrix([1, 0, 1, 0], existence=exist.patterns[0])
    assert np.all(mat == np.array([[1, 0], [0, 1]]))

    dv, mat = encoder.get_matrix([1, 0, 0, 0], existence=exist.patterns[1])
    assert np.all(dv == [1, 0, -1, -1])
    assert np.all(mat == np.array([[1, 0], [0, 0]]))
    dv, mat = encoder.get_matrix([1, 0, 0, 1], existence=exist.patterns[1])
    assert np.all(dv == [1, 0, -1, -1])
    assert np.all(mat == np.array([[1, 0], [0, 0]]))

    check_lazy_conditionally_active(encoder)


def test_large_matrix():
    encoder = LazyDirectMatrixEncoder(LazyDeltaImputer())
    src = [Node([0, 1], repeated_allowed=False) for _ in range(6)]
    tgt = [Node([0, 1], repeated_allowed=False) for _ in range(6)]
    encoder.set_settings(MatrixGenSettings(src, tgt))

    assert len(encoder.design_vars) == 36
    assert encoder.get_n_design_points() > 0


def test_one_to_one(gen_one_per_existence: AggregateAssignmentMatrixGenerator):
    g = gen_one_per_existence
    encoder = LazyDirectMatrixEncoder(LazyDeltaImputer())
    encoder.set_settings(g.settings)
    assert len(encoder.design_vars) == 1
    assert encoder.design_vars[0].n_opts == 2

    assert encoder.get_n_design_points() == 2
    assert encoder.get_imputation_ratio() == 2.4
    assert encoder.get_information_index() == 1
    assert encoder.get_distance_correlation() == 1

    check_lazy_conditionally_active(encoder)

    for i, existence in enumerate(gen_one_per_existence.existence_patterns.patterns):
        dv, mat = encoder.get_matrix([1], existence=existence)
        assert mat.shape[0] == (len(gen_one_per_existence.src) if i not in [3, 7] else 0)


def test_one_to_one_first(gen_one_per_existence: AggregateAssignmentMatrixGenerator):
    g = gen_one_per_existence
    encoder = LazyDirectMatrixEncoder(LazyFirstImputer())
    encoder.set_settings(g.settings)
    assert len(encoder.design_vars) == 1
    assert encoder.design_vars[0].n_opts == 2

    check_lazy_conditionally_active(encoder)

    for i, existence in enumerate(gen_one_per_existence.existence_patterns.patterns):
        dv, mat = encoder.get_matrix([1], existence=existence)
        assert mat.shape[0] == (len(gen_one_per_existence.src) if i not in [3, 7] else 0)


def test_one_to_one_cv(gen_one_per_existence: AggregateAssignmentMatrixGenerator):
    g = gen_one_per_existence
    encoder = LazyDirectMatrixEncoder(LazyConstraintViolationImputer())
    encoder.set_settings(g.settings)
    assert len(encoder.design_vars) == 1
    assert encoder.design_vars[0].n_opts == 2

    check_lazy_conditionally_active(encoder)

    for i, existence in enumerate(gen_one_per_existence.existence_patterns.patterns):
        dv, mat = encoder.get_matrix([1], existence=existence)
        assert mat.shape[0] == len(gen_one_per_existence.src)


def test_different_sizes_bounds():
    patterns = [
        NodeExistence(),
        NodeExistence(src_exists=[True, False], tgt_exists=[True, False]),
    ]
    settings = MatrixGenSettings(src=[Node(min_conn=0) for _ in range(2)], tgt=[Node([0, 1]), Node(min_conn=0)],
                                 existence=NodeExistencePatterns(patterns=patterns))
    encoder = LazyDirectMatrixEncoder(LazyFirstImputer())
    encoder.set_settings(settings)
    assert len(encoder.design_vars) == 4
    assert [dv.n_opts for dv in encoder.design_vars] == [2, 3, 2, 3]

    for existence in patterns:
        dv_seen = set()
        matrix_seen = set()
        for dv in itertools.product(*[list(range(dv.n_opts)) for dv in encoder.design_vars]):
            dv_imp, matrix = encoder.get_matrix(list(dv), existence=existence)
            dv_seen.add(tuple(dv_imp))
            matrix_seen.add(tuple(matrix.ravel()))
        assert len(dv_seen) == len(matrix_seen)

    check_lazy_conditionally_active(encoder)
