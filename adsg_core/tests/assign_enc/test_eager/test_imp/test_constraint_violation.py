import numpy as np
from adsg_core.optimization.assign_enc.matrix import *
from adsg_core.optimization.assign_enc.assignment_manager import *
from adsg_core.optimization.assign_enc.eager.encodings.direct_matrix import *
from adsg_core.optimization.assign_enc.eager.imputation.constraint_violation import *


def test_imputer():
    matrix = np.random.randint(0, 3, (10, 3, 4))
    encoder = DirectMatrixEncoder(ConstraintViolationImputer(), matrix)

    n_imp = 0
    for _ in range(100):
        dv = encoder.get_random_design_vector()
        if not encoder.is_valid_vector(dv):
            n_imp += 1
            dv_imp, mat = encoder.get_matrix(dv)
            assert np.all(dv == dv_imp)
            assert mat.shape == (3, 4)
            assert np.all(mat == -1)

    assert n_imp > 0


def test_assignment_manager():
    encoder = DirectMatrixEncoder(ConstraintViolationImputer())
    manager = AssignmentManager(MatrixGenSettings(src=[Node([1], repeated_allowed=False) for _ in range(2)],
                                tgt=[Node([1], repeated_allowed=False) for _ in range(2)]), encoder=encoder)
    assert manager.encoder.n_mat_max == 2
    assert len(manager.design_vars) == 4

    dv, is_active, mat = manager.get_matrix([1, 0, 1, 0])
    assert np.all(dv == [1, 0, 1, 0])
    assert np.all(is_active)
    assert np.all(mat == -1)

    dv, is_active, conn_idx = manager.get_conn_idx([1, 0, 1, 0])
    assert np.all(dv == [1, 0, 1, 0])
    assert np.all(is_active)
    assert conn_idx is None
    dv, is_active, conns = manager.get_conns([1, 0, 1, 0])
    assert np.all(dv == [1, 0, 1, 0])
    assert np.all(is_active)
    assert conns is None


def test_one_to_one(gen_one_per_existence: AggregateAssignmentMatrixGenerator):
    encoder = DirectMatrixEncoder(ConstraintViolationImputer())
    encoder.matrix = gen_one_per_existence.get_agg_matrix()
    assert len(encoder.design_vars) == 0

    assert encoder.get_n_design_points() == 1
    assert encoder.get_imputation_ratio() == 1.2
    assert encoder.get_information_index() == 1

    for i, existence in enumerate(gen_one_per_existence.existence_patterns.patterns):
        dv, mat = encoder.get_matrix([], existence=existence)
        assert dv == []
        assert mat.shape[0] == (len(gen_one_per_existence.src) if i not in [3, 7] else 0)
