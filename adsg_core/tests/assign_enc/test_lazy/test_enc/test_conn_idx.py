import numpy as np
from adsg_core.optimization.assign_enc.matrix import *
from adsg_core.optimization.assign_enc.lazy.imputation.closest import *
from adsg_core.optimization.assign_enc.lazy.encodings.conn_idx import *
from adsg_core.optimization.assign_enc.lazy.imputation.constraint_violation import *
from adsg_core.tests.assign_enc.test_lazy_encoding import check_lazy_conditionally_active


def test_encoding():
    encoder = LazyConnIdxMatrixEncoder(LazyConstraintViolationImputer(), FlatConnCombsEncoder())
    src, tgt = [Node([0, 1, 2]) for _ in range(3)], [Node(min_conn=1, repeated_allowed=False) for _ in range(2)]
    encoder.set_settings(MatrixGenSettings(src=src, tgt=tgt, excluded=[(src[0], tgt[1])],
                                           existence=NodeExistencePatterns([NodeExistence(src_n_conn_override={0: [2]})])))
    assert len(encoder.design_vars) == 0

    encoder.set_settings(MatrixGenSettings(src=src, tgt=tgt, excluded=[(src[0], tgt[1])],
                                           existence=NodeExistencePatterns([NodeExistence(src_n_conn_override={0: [1]})])))
    assert len(encoder.design_vars) == 2

    dv, mat = encoder.get_matrix([2, 2], existence=encoder.existence_patterns.patterns[0])
    assert np.all(dv == [2, 2])
    assert np.all(mat == np.array([[1, 0], [0, 1], [0, 1]]))

    dv, mat = encoder.get_matrix([0, 0], existence=encoder.existence_patterns.patterns[0])
    assert np.all(dv == [0, 0])
    assert mat[0, 0] == -1

    assert encoder.get_distance_correlation()
    check_lazy_conditionally_active(encoder)


def test_encoding_transpose():
    encoder = LazyConnIdxMatrixEncoder(LazyConstraintViolationImputer(), FlatConnCombsEncoder(), by_src=False)
    src, tgt = [Node([0, 1, 2]) for _ in range(3)], [Node(min_conn=1, repeated_allowed=False) for _ in range(2)]
    encoder.set_settings(MatrixGenSettings(src=src, tgt=tgt, excluded=[(src[0], tgt[1])],
                                           existence=NodeExistencePatterns([NodeExistence(src_n_conn_override={0: [2]})])))
    assert len(encoder.design_vars) == 2
    _, mat = encoder.get_matrix([0, 0], existence=encoder.existence_patterns.patterns[0])
    assert mat[0, 0] == -1

    encoder.set_settings(MatrixGenSettings(src=src, tgt=tgt, excluded=[(src[0], tgt[1])],
                                           existence=NodeExistencePatterns([NodeExistence(src_n_conn_override={0: [1]})])))
    assert len(encoder.design_vars) > 0

    dv, mat = encoder.get_matrix([0, 2], existence=encoder.existence_patterns.patterns[0])
    assert np.all(dv == [0, 2])
    assert np.all(mat == np.array([[1, 0], [0, 1], [0, 1]]))

    dv, mat = encoder.get_matrix([2, 2], existence=encoder.existence_patterns.patterns[0])
    assert np.all(dv == [2, 2])
    assert mat[0, 0] == -1

    check_lazy_conditionally_active(encoder)


def test_encoding_amount_first():
    encoder = LazyConnIdxMatrixEncoder(LazyConstraintViolationImputer(), FlatConnCombsEncoder(), amount_first=True)
    src, tgt = [Node([0, 1, 2]) for _ in range(3)], [Node(min_conn=1, repeated_allowed=False) for _ in range(2)]
    encoder.set_settings(MatrixGenSettings(src=src, tgt=tgt, excluded=[(src[0], tgt[1])],
                                           existence=NodeExistencePatterns([NodeExistence(src_n_conn_override={0: [2]})])))
    assert len(encoder.design_vars) == 0

    encoder.set_settings(MatrixGenSettings(src=src, tgt=tgt, excluded=[(src[0], tgt[1])],
                                           existence=NodeExistencePatterns([NodeExistence(src_n_conn_override={0: [1]})])))
    assert len(encoder.design_vars) == 4

    dv, mat = encoder.get_matrix([0, 0, 0, 0], existence=encoder.existence_patterns.patterns[0])
    assert np.all(dv == [-1, -1, -1, -1])
    assert mat[0, 0] == -1

    dv, mat = encoder.get_matrix([1, 1, 1, 1], existence=encoder.existence_patterns.patterns[0])
    assert np.all(dv == [1, -1, 1, -1])
    assert np.all(mat == np.array([[1, 0], [0, 1], [0, 1]]))

    check_lazy_conditionally_active(encoder)


def test_encoding_grouped():
    encoder = LazyConnIdxMatrixEncoder(LazyConstraintViolationImputer(), GroupedConnCombsEncoder())
    src, tgt = [Node([0, 1, 2]) for _ in range(3)], [Node(min_conn=1, repeated_allowed=False) for _ in range(2)]
    encoder.set_settings(MatrixGenSettings(src=src, tgt=tgt, excluded=[(src[0], tgt[1])],
                                           existence=NodeExistencePatterns([NodeExistence(src_n_conn_override={0: [2]})])))
    assert len(encoder.design_vars) == 0

    encoder.set_settings(MatrixGenSettings(src=src, tgt=tgt, excluded=[(src[0], tgt[1])],
                                           existence=NodeExistencePatterns([NodeExistence(src_n_conn_override={0: [1]})])))
    assert len(encoder.design_vars) == 4

    dv, mat = encoder.get_matrix([0, 1, 0, 1], existence=encoder.existence_patterns.patterns[0])
    assert np.all(dv == [0, 1, 0, 1])
    assert np.all(mat == np.array([[1, 0], [0, 1], [0, 1]]))

    dv, mat = encoder.get_matrix([0, 0, 0, 0], existence=encoder.existence_patterns.patterns[0])
    assert np.all(dv == [0, 0, 0, 0])
    assert mat[0, 0] == -1

    check_lazy_conditionally_active(encoder)


def test_encoding_conn_idx():
    for binary in [False, True]:
        encoder = LazyConnIdxMatrixEncoder(LazyConstraintViolationImputer(), ConnIdxCombsEncoder(binary=binary))
        src, tgt = [Node([0, 1, 2]) for _ in range(3)], [Node(min_conn=1, repeated_allowed=False) for _ in range(2)]
        encoder.set_settings(MatrixGenSettings(src=src, tgt=tgt, excluded=[(src[0], tgt[1])],
                                               existence=NodeExistencePatterns([NodeExistence(src_n_conn_override={0: [2]})])))
        assert len(encoder.design_vars) == 0

        encoder.set_settings(MatrixGenSettings(src=src, tgt=tgt, excluded=[(src[0], tgt[1])],
                                               existence=NodeExistencePatterns([NodeExistence(src_n_conn_override={0: [1]})])))
        assert len(encoder.design_vars) == (6 if binary else 4)
        if binary:
            continue

        dv, mat = encoder.get_matrix([1, 0, 1, 1], existence=encoder.existence_patterns.patterns[0])
        assert np.all(dv == [1, 0, 1, 1])
        assert np.all(mat == np.array([[1, 0], [1, 0], [1, 1]]))

        dv, mat = encoder.get_matrix([0, 1, 0, 1], existence=encoder.existence_patterns.patterns[0])
        assert np.all(dv == [0, -1, 0, -1])
        assert mat[0, 0] == -1

        check_lazy_conditionally_active(encoder)


def test_one_to_one(gen_one_per_existence: AggregateAssignmentMatrixGenerator):
    for binary in [False, True]:
        g = gen_one_per_existence
        encoder = LazyConnIdxMatrixEncoder(LazyClosestImputer(), FlatConnCombsEncoder(binary=binary))
        encoder.set_settings(g.settings)
        assert len(encoder.design_vars) == 0

        assert encoder.get_n_design_points() == 1
        assert encoder.get_imputation_ratio() == 1.2
        assert encoder.get_information_index() == 1
        assert encoder.get_distance_correlation() == 1

        for i, existence in enumerate(gen_one_per_existence.existence_patterns.patterns):
            dv, mat = encoder.get_matrix([], existence=existence)
            assert dv == []
            assert mat.shape[0] == (len(gen_one_per_existence.src) if i not in [3, 7] else 0)

        check_lazy_conditionally_active(encoder)
