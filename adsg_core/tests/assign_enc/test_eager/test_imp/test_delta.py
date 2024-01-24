import numpy as np
from adsg_core.optimization.assign_enc.matrix import *
from adsg_core.optimization.assign_enc.eager.imputation.delta import *
from adsg_core.optimization.assign_enc.eager.encodings.direct_matrix import *


def test_imputer():
    matrix = np.array([
        [[0, 0, 0, 0]],
        [[0, 1, 0, 0]],
        [[1, 0, 0, 0]],
        [[1, 1, 0, 0]],
        [[1, 1, 1, 1]],
        [[1, 1, 1, 2]],
    ])
    encoder = DirectMatrixEncoder(DeltaImputer(), matrix)

    dv, mat = encoder.get_matrix([0, 0, 0, 0])
    assert np.all(dv == [0, 0, 0, 0])
    assert np.all(mat == matrix[0, :, :])

    dv, mat = encoder.get_matrix([0, 0, 1, 0])
    assert np.all(dv == [0, 0, 0, 0])
    assert np.all(mat == matrix[0, :, :])

    dv, mat = encoder.get_matrix([1, 1, 1, 0])
    assert np.all(dv == [1, 1, 1, 1])
    assert np.all(mat == matrix[4, :, :])

    dv, mat = encoder.get_matrix([1, 1, 2, 0])
    assert np.all(dv == [1, 1, 1, 1])
    assert np.all(mat == matrix[4, :, :])


def test_imputer_masked():
    matrix = np.array([
        [[0, 0, 0, 0]],
        [[0, 1, 0, 0]],
        [[1, 0, 0, 0]],
        [[1, 1, 0, 0]],
        [[1, 1, 1, 1]],
        [[1, 1, 1, 2]],
    ])
    encoder = DirectMatrixEncoder(DeltaImputer(), matrix)
    mask = np.array([False, True, True, True, False, True], dtype=bool)

    dv, mat = encoder.get_matrix([0, 0, 0, 0], matrix_mask=mask)
    assert np.all(dv == [0, 1, 0, 0])
    assert np.all(mat == matrix[1, :, :])

    dv, mat = encoder.get_matrix([1, 1, 2, 0], matrix_mask=mask)
    assert np.all(dv == [1, 1, 1, 2])
    assert np.all(mat == matrix[5, :, :])


def test_imputer_random():
    matrix = np.random.randint(0, 3, (10, 3, 4))
    encoder = DirectMatrixEncoder(DeltaImputer(), matrix)

    n_imp = 0
    n_imp_invalid = 0
    for _ in range(10):
        dv = encoder.get_random_design_vector()
        if not encoder.is_valid_vector(dv):
            dv_imp, mat = encoder.get_matrix(dv)
            if np.all(dv == dv_imp):
                n_imp_invalid += 1
                assert np.all(mat == -1)
            else:
                n_imp += 1
                assert np.any(dv != dv_imp)
            assert mat.shape == (3, 4)

    assert n_imp > 0 or n_imp_invalid > 0


def test_one_to_one(gen_one_per_existence: AggregateAssignmentMatrixGenerator):
    encoder = DirectMatrixEncoder(DeltaImputer())
    encoder.matrix = gen_one_per_existence.get_agg_matrix()
    assert len(encoder.design_vars) == 0

    assert encoder.get_n_design_points() == 1
    assert encoder.get_imputation_ratio() == 1.2
    assert encoder.get_information_index() == 1

    for i, existence in enumerate(gen_one_per_existence.existence_patterns.patterns):
        dv, mat = encoder.get_matrix([], existence=existence)
        assert dv == []
        assert mat.shape[0] == (len(gen_one_per_existence.src) if i not in [3, 7] else 0)
