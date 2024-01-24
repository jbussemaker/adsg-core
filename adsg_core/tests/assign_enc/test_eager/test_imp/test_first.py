import numpy as np
from adsg_core.optimization.assign_enc.matrix import *
from adsg_core.optimization.assign_enc.eager.imputation.first import *
from adsg_core.optimization.assign_enc.eager.encodings.direct_matrix import *


def test_imputer():
    matrix = np.random.randint(0, 3, (10, 3, 4))
    encoder = DirectMatrixEncoder(FirstImputer(), matrix)

    n_imp = 0
    for _ in range(100):
        dv = encoder.get_random_design_vector()
        if not encoder.is_valid_vector(dv):
            n_imp += 1
            dv_imp, mat = encoder.get_matrix(dv)
            assert np.all(dv_imp == list(encoder._design_vectors.values())[0][0, :])
            assert np.all(mat == list(encoder._matrix.values())[0][0, :, :])

    assert n_imp > 0


def test_imputer_masked():
    matrix = np.random.randint(0, 5, (10, 3, 4))
    encoder = DirectMatrixEncoder(FirstImputer(), matrix)

    for _ in range(10):
        mask = np.random.randint(0, 2, (10,), dtype=bool)
        if np.all(~mask):
            continue
        i_first_valid = np.where(mask)[0][0]

        n_imp = 0
        for _ in range(100):
            dv = encoder.get_random_design_vector()
            if not encoder.is_valid_vector(dv, matrix_mask=mask):
                n_imp += 1
                dv_imp, mat = encoder.get_matrix(dv, matrix_mask=mask)
                assert np.all(dv_imp == list(encoder._design_vectors.values())[0][i_first_valid, :])
                assert np.all(mat == list(encoder._matrix.values())[0][i_first_valid, :, :])

        assert n_imp > 0


def test_one_to_one(gen_one_per_existence: AggregateAssignmentMatrixGenerator):
    encoder = DirectMatrixEncoder(FirstImputer())
    encoder.matrix = gen_one_per_existence.get_agg_matrix()
    assert len(encoder.design_vars) == 0

    assert encoder.get_n_design_points() == 1
    assert encoder.get_imputation_ratio() == 1.2
    assert encoder.get_information_index() == 1

    for i, existence in enumerate(gen_one_per_existence.existence_patterns.patterns):
        dv, mat = encoder.get_matrix([], existence=existence)
        assert dv == []
        assert mat.shape[0] == (len(gen_one_per_existence.src) if i not in [3, 7] else 0)
