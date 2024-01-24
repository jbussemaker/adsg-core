import numpy as np
from adsg_core.optimization.assign_enc.matrix import *
from adsg_core.optimization.assign_enc.eager.imputation.auto_mod import *
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
    encoder = DirectMatrixEncoder(AutoModImputer(), matrix)

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
    encoder = DirectMatrixEncoder(AutoModImputer(), matrix)
    mask = np.array([False, True, True, True, False, True], dtype=bool)

    dv, mat = encoder.get_matrix([0, 0, 0, 0], matrix_mask=mask)
    assert np.all(dv == [0, 1, 0, 0])
    assert np.all(mat == matrix[1, :, :])

    dv, mat = encoder.get_matrix([1, 1, 1, 0], matrix_mask=mask)
    assert np.all(dv == [1, 1, 1, 2])
    assert np.all(mat == matrix[5, :, :])


def test_imputer_random_masked():
    matrix = np.random.randint(0, 3, (10, 3, 4))
    encoder = DirectMatrixEncoder(AutoModImputer(), matrix)

    for i_active in range(10):
        mask = np.zeros((10,), dtype=bool)
        mask[i_active] = True

        n_imp = 0
        for _ in range(100):
            dv = encoder.get_random_design_vector()
            if not encoder.is_valid_vector(dv, matrix_mask=mask):
                n_imp += 1
                dv_imp, mat = encoder.get_matrix(dv, matrix_mask=mask)
                assert np.any(dv != dv_imp)
                assert mat.shape == (3, 4)

        assert n_imp > 0


def test_imputer_reverse():
    matrix = np.array([
        [[0, 0, 0, 0]],
        [[0, 1, 0, 0]],
        [[1, 0, 0, 0]],
        [[1, 1, 0, 0]],
        [[1, 1, 1, 1]],
        [[1, 1, 1, 2]],
    ])
    encoder = DirectMatrixEncoder(AutoModImputer(reverse=True), matrix)

    dv, mat = encoder.get_matrix([0, 0, 0, 0])
    assert np.all(dv == [0, 0, 0, 0])
    assert np.all(mat == matrix[0, :, :])

    dv, mat = encoder.get_matrix([0, 0, 1, 0])
    assert np.all(dv == [0, 0, 0, 0])
    assert np.all(mat == matrix[0, :, :])

    dv, mat = encoder.get_matrix([1, 1, 1, 0])
    assert np.all(dv == [1, 1, 0, 0])
    assert np.all(mat == matrix[3, :, :])

    dv, mat = encoder.get_matrix([1, 1, 2, 0])
    assert np.all(dv == [1, 1, 0, 0])
    assert np.all(mat == matrix[3, :, :])

    dv, mat = encoder.get_matrix([0, 0, 0, 2])
    assert np.all(dv == [1, 1, 1, 2])
    assert np.all(mat == matrix[5, :, :])


def test_imputer_random():
    matrix = np.random.randint(0, 3, (10, 3, 4))
    encoder = DirectMatrixEncoder(AutoModImputer(), matrix)

    n_imp = 0
    for _ in range(100):
        dv = encoder.get_random_design_vector()
        if not encoder.is_valid_vector(dv):
            n_imp += 1
            dv_imp, mat = encoder.get_matrix(dv)
            assert np.any(dv != dv_imp)
            assert mat.shape == (3, 4)

    assert n_imp > 0


def test_one_to_one(gen_one_per_existence: AggregateAssignmentMatrixGenerator):
    encoder = DirectMatrixEncoder(AutoModImputer())
    encoder.matrix = gen_one_per_existence.get_agg_matrix()
    assert len(encoder.design_vars) == 0

    assert encoder.get_n_design_points() == 1
    assert encoder.get_imputation_ratio() == 1.2
    assert encoder.get_information_index() == 1

    for i, existence in enumerate(gen_one_per_existence.existence_patterns.patterns):
        dv, mat = encoder.get_matrix([], existence=existence)
        assert dv == []
        assert mat.shape[0] == (len(gen_one_per_existence.src) if i not in [3, 7] else 0)
