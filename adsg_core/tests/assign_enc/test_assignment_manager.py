import numpy as np
from adsg_core.optimization.assign_enc.matrix import *
from adsg_core.optimization.assign_enc.encoding import *
from adsg_core.optimization.assign_enc.eager.imputation.first import *
from adsg_core.optimization.assign_enc.assignment_manager import *


class FlattenEncoder(EagerEncoder):

    def _encode(self, matrix: np.ndarray) -> np.ndarray:
        n_mat, n_src, n_tgt = matrix.shape
        n_dv = n_src*n_tgt

        # Map matrix elements to design vector values
        design_vectors = matrix.reshape(n_mat, n_dv)

        # Move design vector values so that the first value is always 0
        design_vectors -= np.min(design_vectors, axis=0)
        return design_vectors


def test_manager():
    src = [Node([1, 2]), Node([1])]
    tgt = [Node([0, 1]), Node(min_conn=1)]
    enc = FlattenEncoder(FirstImputer())

    manager = AssignmentManager(MatrixGenSettings(src, tgt), enc)
    assert manager.matrix is not None
    assert len(manager.matrix) == 1
    assert list(manager.matrix.values())[0].shape == (6, 2, 2)
    assert manager._matrix_gen.count_all_matrices() == 6

    assert len(manager.design_vars) == 4

    seen_dvs = set()
    for _ in range(100):
        dv = manager.get_random_design_vector()
        imp_dv, is_active, conn_idx = manager.get_conn_idx(dv)
        imp_dv, is_active, conn = manager.get_conns(dv)

        assert imp_dv is not None
        assert is_active is not None
        assert conn is not None
        assert len(conn) > 0

        assert len(conn_idx) == len(conn)
        assert [(manager._matrix_gen.src[i], manager._matrix_gen.tgt[j]) for i, j in conn_idx] == conn

        seen_dvs.add(tuple(imp_dv))

        dv_corr, is_act_corr = manager.correct_vector(dv)
        assert np.all(dv_corr == imp_dv)
        assert np.all(is_act_corr == is_active)
        dv_corr, is_act_corr = manager.correct_vector(imp_dv)
        assert np.all(dv_corr == imp_dv)
        assert np.all(is_act_corr == is_active)

    assert len(seen_dvs) < 100

    all_x = manager.get_all_design_vectors()[NodeExistence()]
    for x in all_x:
        imp_x, _, _ = manager.get_matrix(x)
        assert np.all(imp_x == x)

    is_cond_all = np.any(all_x == X_INACTIVE_VALUE, axis=0)
    assert np.all(is_cond_all == [dv.conditionally_active for dv in manager.design_vars])
