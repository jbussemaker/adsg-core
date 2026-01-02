import pytest
import numpy as np
from adsg_core.optimization.assign_enc.matrix import *
from adsg_core.optimization.assign_enc.encoding import *
from adsg_core.optimization.assign_enc.eager.imputation.first import *
from adsg_core.optimization.assign_enc.eager.encodings import DirectMatrixEncoder


def test_filter_design_vectors():
    design_vectors = np.array([
        [0, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 1, 2, 3],
        [0, 1, 2, 4],
    ], dtype=int)

    def _assert_indices(vector, indices):
        assert np.all(np.where(filter_design_vectors(design_vectors, vector))[0] == indices)

    _assert_indices([0, 0, 0, 0], [0])
    _assert_indices([0, 1, 2, 3], [2])
    _assert_indices([0, 1, 1, 0], [])

    _assert_indices([0, 0, None, None], [0])
    _assert_indices([None, 0, None, None], [0])
    _assert_indices([0, 1, None, None], [1, 2, 3])
    _assert_indices([0, 1, 2, None], [2, 3])
    _assert_indices([0, 1, None, 3], [2])


class DirectEncoder(EagerEncoder):

    def _encode(self, matrix: np.ndarray) -> np.ndarray:
        n_mat = matrix.shape[0]
        return np.array([np.arange(0, n_mat)]).T


def test_encoder():
    enc = DirectEncoder(EagerImputer())
    with pytest.raises(RuntimeError):
        enc.get_matrix([0])

    matrix = np.random.randint(0, 3, (10, 2, 3), dtype=int)
    enc = DirectEncoder(EagerImputer(), matrix)
    exist = NodeExistence()
    assert enc.matrix is not matrix
    assert list(enc.matrix.values())[0] is matrix
    assert exist in enc.matrix
    assert enc.matrix[exist] is matrix
    assert enc.n_mat_max == 10

    assert enc._design_vectors[exist].shape == (10, 1)
    assert len(enc.design_vars) == 1
    assert enc.design_vars[0].n_opts == 10
    assert not enc.design_vars[0].conditionally_active
    check_conditionally_active(enc)

    assert enc.get_n_design_points() == 10
    assert enc.get_imputation_ratio() == 1.

    for _ in range(100):
        val = enc.design_vars[0].get_random()
        assert val >= 0
        assert val < 10

        dv = enc.get_random_design_vector()
        assert len(dv) == 1
        assert dv[0] >= 0
        assert dv[0] < 10

    assert enc.get_matrix_index([0]) == (0, exist)
    assert enc.is_valid_vector([0])
    assert enc.is_valid_vector([0, 0])
    assert enc.is_valid_vector([0, 1])

    dv, mat = enc.get_matrix([0])
    assert dv == [0]
    assert np.all(mat == matrix[0, :, :])

    dv, mat = enc.get_matrix([0, 0])
    assert dv == [0, -1]
    assert np.all(mat == matrix[0, :, :])

    dv, mat = enc.get_matrix([0, 1])
    assert dv == [0, -1]
    assert np.all(mat == matrix[0, :, :])

    assert enc.is_valid_vector([9])
    assert not enc.is_valid_vector([10])
    assert not enc.is_valid_vector([-1])

    dv, mat = enc.get_matrix([10])
    assert dv == [9]
    assert np.all(mat == matrix[9, :, :])
    dv, mat = enc.get_matrix([-1])
    assert dv == [0]
    assert np.all(mat == matrix[0, :, :])

    matrix_mask = np.ones((10,), dtype=bool)
    matrix_mask[0] = False
    with pytest.raises(NotImplementedError):
        enc.get_matrix([0], matrix_mask=matrix_mask)

    matrix_mask = np.zeros((10,), dtype=bool)
    assert not enc.is_valid_vector([0], matrix_mask=matrix_mask)
    dv, mat = enc.get_matrix([0], matrix_mask=matrix_mask)
    assert np.all(dv == [-1])
    assert np.all(mat == 0)


def check_conditionally_active(encoder: EagerEncoder):
    if len(encoder.design_vars) == 0:
        return

    any_inactive = np.zeros((len(encoder.design_vars),), dtype=bool)
    for _, des_vectors in encoder.padded_design_vectors.items():
        if des_vectors.shape[0] == 0 or des_vectors.shape[1] == 0:
            any_inactive_existence = np.ones((len(encoder.design_vars),), dtype=bool)
        else:
            any_inactive_existence = np.any(des_vectors == X_INACTIVE_VALUE, axis=0)
        any_inactive[:len(any_inactive_existence)] |= any_inactive_existence

    assert np.all(any_inactive == [dv.conditionally_active for dv in encoder.design_vars])


def test_information_index():
    assert Encoder.calc_information_index([]) == 1
    assert Encoder.calc_information_index([3]) == 0
    assert Encoder.calc_information_index([2]) == 1
    assert Encoder.calc_information_index([3, 2]) == pytest.approx(.63, abs=1e-3)
    assert Encoder.calc_information_index([2, 2, 2]) == 1
    assert Encoder.calc_information_index([10]) == 0

    assert Encoder.calc_information_index([2, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 2, 2, 2, 2,
                                           2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 3, 2, 3, 2, 3]) <= 1.


class TwoEncoder(EagerEncoder):

    def _encode(self, matrix: np.ndarray) -> np.ndarray:
        n_mat = matrix.shape[0]
        if n_mat <= 4:
            return np.array([np.arange(0, n_mat)]).T

        n_mat_half = int(np.ceil(n_mat/2))
        dv1 = np.tile(np.arange(0, n_mat_half), 2)[:n_mat]
        dv2 = np.repeat([0, 1], n_mat_half)[:n_mat]
        return np.column_stack([dv1, dv2])


def test_encoder_impute():
    matrix = np.random.randint(0, 3, (11, 2, 3), dtype=int)
    enc = TwoEncoder(FirstImputer())
    enc.matrix = matrix
    assert len(enc.design_vars) == 2
    assert enc.design_vars[0].n_opts == 6
    assert enc.design_vars[1].n_opts == 2
    assert not enc.design_vars[0].conditionally_active
    assert not enc.design_vars[1].conditionally_active
    exist = NodeExistence()
    assert enc.matrix[exist] is matrix
    assert enc.n_mat_max == 11

    assert enc.get_information_index() == enc.calc_information_index([6, 2])
    assert enc.get_information_index() == pytest.approx(.386, abs=1e-3)

    assert not enc.is_valid_vector([5, 1])
    assert enc.get_matrix_index([5, 1])[0] is None
    dv, mat = enc.get_matrix([5, 1])
    assert np.all(dv == enc._design_vectors[exist][0, :])
    assert np.all(mat == matrix[0, :, :])

    matrix_mask = np.ones((11,), dtype=bool)
    matrix_mask[0] = False
    dv, mat = enc.get_matrix([0, 0], matrix_mask=matrix_mask)
    assert np.all(dv == enc._design_vectors[exist][1, :])
    assert np.all(mat == matrix[1, :, :])

    assert np.all(enc.padded_design_vectors[exist] == enc._design_vectors[exist])


def test_encoder_existence():
    exist1 = NodeExistence()
    exist2 = NodeExistence(src_exists=[True, False])
    matrix_map = {
        exist1: np.random.random((10, 3, 2)),
        exist2: np.random.random((4, 3, 2)),
    }
    enc = TwoEncoder(FirstImputer())
    enc.matrix = matrix_map
    assert enc.n_mat_max == 10

    assert len(enc.design_vars) == 2
    assert enc.design_vars[0].n_opts == 5
    assert enc.design_vars[1].n_opts == 2
    assert not enc.design_vars[0].conditionally_active
    assert enc.design_vars[1].conditionally_active

    assert enc._design_vectors[exist1].shape[1] == 2
    assert enc._design_vectors[exist2].shape[1] == 1

    assert enc.is_valid_vector([0, 0], existence=exist1)
    assert enc.is_valid_vector([0, 1], existence=exist1)
    assert enc.is_valid_vector([2, 1], existence=exist1)

    assert enc.get_matrix([0, 0], existence=exist1)

    assert enc.is_valid_vector([0, 0], existence=exist2)
    assert enc.is_valid_vector([0, 1], existence=exist2)
    assert enc.is_valid_vector([2, 0], existence=exist2)

    assert enc._has_existence(exist1)
    assert not enc._has_existence(NodeExistence(tgt_exists=[True, False]))

    assert enc.get_matrix_index([0, 0], existence=exist1)[0] == 0
    assert enc.get_matrix_index([1, 0], existence=exist2)[0] == 1

    dv, mat = enc.get_matrix([1, 1], existence=exist2)  # Correct last des var value
    assert dv == [1, -1]
    assert np.all(mat == enc._matrix[exist2][1, :, :])

    dv, mat = enc.get_matrix([4, 1], existence=exist2)  # Impute
    assert dv == [0, -1]
    assert np.all(mat == enc._matrix[exist2][0, :, :])


class DuplicateEncoder(EagerEncoder):

    def _encode(self, matrix: np.ndarray) -> np.ndarray:
        n_mat = matrix.shape[0]
        return np.array([np.repeat(np.arange(0, n_mat), 2)[:n_mat]]).T


def test_duplicate_encoder():
    matrix = np.random.randint(0, 3, (10, 2, 3))
    enc = DuplicateEncoder(EagerImputer())
    with pytest.raises(RuntimeError):
        enc.matrix = matrix


class LowerThanZeroEncoder(EagerEncoder):

    def _encode(self, matrix: np.ndarray) -> np.ndarray:
        n_mat = matrix.shape[0]
        return np.column_stack([np.arange(0, n_mat)-2, np.arange(0, n_mat)])


class HigherThanZeroEncoder(EagerEncoder):

    def _encode(self, matrix: np.ndarray) -> np.ndarray:
        n_mat = matrix.shape[0]
        return np.column_stack([np.arange(0, n_mat)+1, np.arange(0, n_mat)+1])


def test_non_zero_encoder():
    matrix = np.random.randint(0, 3, (10, 2, 3))
    with pytest.raises(RuntimeError):
        LowerThanZeroEncoder(FirstImputer(), matrix)
    with pytest.raises(RuntimeError):
        HigherThanZeroEncoder(FirstImputer(), matrix)


def test_flatten_matrix():
    matrix = np.random.randint(0, 3, (10, 4, 3))
    flattened = flatten_matrix(matrix)
    assert flattened.shape == (10, 4*3)


def test_normalize_dvs():
    des_vectors = np.array([
        [0, 1, 0, 2, -1],
        [1, 2, 3, 0,  0],
        [2, 3, 2, 0,  2],
    ])

    assert np.all(EagerEncoder.normalize_design_vectors(des_vectors, remove_gaps=False) == np.array([
        [0, 0, 0, 2, -1],
        [1, 1, 3, 0, 0],
        [2, 2, 2, 0, 2],
    ]))

    assert np.all(EagerEncoder.normalize_design_vectors(des_vectors) == np.array([
        [0, 0, 0, 1, -1],
        [1, 1, 2, 0, 0],
        [2, 2, 1, 0, 1],
    ]))


class DirectZeroEncoder(EagerEncoder):

    def _encode(self, matrix: np.ndarray) -> np.ndarray:
        n_mat = matrix.shape[0]
        if n_mat == 1:
            return np.empty((0, 0), dtype=int)
        return np.array([np.arange(0, n_mat)]).T


def test_encoder_zero_dvs():
    src = [Node([1], repeated_allowed=False) for _ in range(2)]
    tgt = [Node([1], repeated_allowed=False) for _ in range(2)]
    exist = NodeExistencePatterns([
        NodeExistence(),
        NodeExistence(src_exists=[True, False], tgt_exists=[True, False]),
    ])
    gen = AggregateAssignmentMatrixGenerator.create(src, tgt, existence=exist)
    matrix_map = gen.get_agg_matrix()
    assert len(matrix_map) == 2
    assert matrix_map[exist.patterns[1]].shape[0] == 1

    encoder = DirectZeroEncoder(EagerImputer())
    encoder.matrix = matrix_map

    assert len(encoder.design_vars) == 1
    assert encoder.design_vars[0].n_opts == 2
    assert encoder.design_vars[0].conditionally_active

    dv, mat = encoder.get_matrix([0], existence=exist.patterns[0])
    assert dv == [0]
    assert np.all(mat == np.array([[1, 0], [0, 1]]))
    dv, mat = encoder.get_matrix([0], existence=exist.patterns[1])
    assert dv == [-1]
    assert np.all(mat == np.array([[1, 0], [0, 0]]))

    padded_dvs = encoder.padded_design_vectors
    assert len(padded_dvs) == 2
    assert np.all(padded_dvs[NodeExistence()] == np.array([np.arange(2)]).T)
    assert np.all(padded_dvs[exist.patterns[1]].shape == (0, 1))


def test_one_to_one(gen_one_per_existence: AggregateAssignmentMatrixGenerator):
    encoder = DirectZeroEncoder(EagerImputer())
    encoder.matrix = gen_one_per_existence.get_agg_matrix()
    assert len(encoder.design_vars) == 0

    assert encoder.get_n_design_points() == 1
    assert encoder.get_imputation_ratio() == 1.2
    assert encoder.get_information_index() == 1
    assert encoder.get_distance_correlation() == 1.
    assert encoder.padded_design_vectors is not None

    for i, existence in enumerate(gen_one_per_existence.existence_patterns.patterns):
        if i == 3 or i == 7:  # One of the sources exist but no target
            with pytest.raises(NotImplementedError):
                encoder.get_matrix([], existence=existence)
            continue

        dv, mat = encoder.get_matrix([], existence=existence)
        assert dv == []
        assert mat.shape[0] == len(gen_one_per_existence.src)


def test_no_conn(gen_one_per_existence: AggregateAssignmentMatrixGenerator):
    gen = AggregateAssignmentMatrixGenerator.create(
        src=gen_one_per_existence.src, tgt=gen_one_per_existence.tgt,
        existence=NodeExistencePatterns(patterns=[gen_one_per_existence.existence_patterns.patterns[-5]]))

    assert gen.count_all_matrices(max_by_existence=False) == 0
    encoder = DirectZeroEncoder(EagerImputer())
    encoder.matrix = gen.get_agg_matrix()
    assert len(encoder.design_vars) == 0

    assert encoder.get_n_design_points() == 1
    assert np.isinf(encoder.get_imputation_ratio())
    assert encoder.get_information_index() == 1
    assert encoder.get_distance_correlation() == 1.
    assert encoder.padded_design_vectors is not None


def test_existence_infeasible():
    src = [Node([1, 2], repeated_allowed=True)]
    tgt = [Node([1, 2], repeated_allowed=True)]
    exist = [
        NodeExistence(),
        NodeExistence(src_exists=[False]),
    ]
    gen = AggregateAssignmentMatrixGenerator.create(src=src, tgt=tgt, existence=NodeExistencePatterns(exist))

    assert gen.count_all_matrices() == 2
    encoder = DirectMatrixEncoder(EagerImputer())
    encoder.matrix = gen.get_agg_matrix()

    assert len(encoder.design_vars) == 1
    check_conditionally_active(encoder)
