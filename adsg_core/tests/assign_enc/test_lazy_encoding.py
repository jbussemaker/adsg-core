import itertools
import numpy as np
from typing import *
from adsg_core.optimization.assign_enc.matrix import *
from adsg_core.optimization.assign_enc.lazy_encoding import *
from adsg_core.optimization.assign_enc.assignment_manager import *


class DummyImputer(LazyImputer):

    def _impute(self, vector: DesignVector, matrix: Optional[np.ndarray], existence: NodeExistence,
                validate: Callable[[np.ndarray], bool], tried_vectors: Set[Tuple[int, ...]]) -> Tuple[DesignVector, np.ndarray]:
        assert not validate(matrix)
        assert tuple(vector) in tried_vectors
        imputed_dv = np.array(vector)*0+X_INACTIVE_VALUE
        imputed_matrix = matrix*0-1
        return imputed_dv, imputed_matrix


class DummyLazyEncoder(LazyEncoder):

    def __init__(self, imputer, n_opts=2):
        self._n_opts = n_opts
        super().__init__(imputer)

    def _encode(self, existence: NodeExistence) -> List[DiscreteDV]:
        n_dv = len(self.src)*len(self.tgt)
        return [DiscreteDV(n_opts=self._n_opts, conditionally_active=False) for _ in range(n_dv)]

    def _decode(self, vector: DesignVector, existence: NodeExistence) -> Optional[Tuple[DesignVector, np.ndarray]]:
        return vector, np.reshape(np.array(vector), (self.n_src, self.n_tgt))


def test_lazy_encoder():
    src, tgt = [Node([0, 1]), Node([0, 1])], [Node([0, 1]), Node([0, 1])]
    enc = DummyLazyEncoder(DummyImputer())
    enc.set_settings(MatrixGenSettings(src, tgt))
    assert enc.src == src
    assert enc.tgt == tgt
    assert enc.n_src == 2
    assert enc.n_tgt == 2
    assert enc.ex is None

    dv = enc.design_vars
    assert len(dv) == 4
    assert all([d.n_opts == 2 for d in dv])

    assert np.all(enc._decode([0, 1, 1, 1], NodeExistence(np.array([True]*2), np.array([True]*2)))[1] == np.array([[0, 1], [1, 1]]))

    dv, matrix, _ = enc._decode_vector([0, 1, 1, 1])
    assert np.all(dv == [0, 1, 1, 1])
    assert np.all(matrix == np.array([[0, 1], [1, 1]]))

    assert not enc.is_valid_vector([0, 1, 1, 1])
    assert enc.is_valid_vector([0, 1, 1, 0])
    assert enc.is_valid_vector([1, 0, 0, 1])
    assert not enc.is_valid_vector([0, 2, 1, 1])

    assert enc.is_valid_vector([0, 1, 1, 0, 0])
    assert not enc.is_valid_vector([0, 1, 1, 0, 1])

    dv, matrix = enc.get_matrix([0, 1, 1, 0])
    assert np.all(dv == [0, 1, 1, 0])
    assert np.all(matrix == np.array([[0, 1], [1, 0]]))
    dv, matrix = enc.get_matrix([0, 2, 1, 0])
    assert np.all(dv == [0, 1, 1, 0])
    assert np.all(matrix == np.array([[0, 1], [1, 0]]))

    dv, matrix = enc.get_matrix([0, 1, 1, 0, 0])
    assert np.all(dv == [0, 1, 1, 0, -1])
    assert np.all(matrix == np.array([[0, 1], [1, 0]]))
    dv, matrix = enc.get_matrix([0, 1, 1, 0, 1])
    assert np.all(dv == [0, 1, 1, 0, -1])
    assert np.all(matrix == np.array([[0, 1], [1, 0]]))

    assert enc.get_conn_idx(matrix) == [(0, 1), (1, 0)]
    assert enc.get_conns(matrix) == [(src[0], tgt[1]), (src[1], tgt[0])]

    dv, matrix = enc.get_matrix([0, 1, 1, 1])
    assert np.all(dv == [-1, -1, -1, -1])
    assert np.all(matrix == -1)

    assert enc._matrix_gen.count_all_matrices() == 7
    assert enc.get_n_design_points() == 16
    assert enc.get_imputation_ratio() == (16/7)
    for _ in range(3):
        assert enc.get_distance_correlation()
        assert enc.get_distance_correlation(minimum=True)

    check_lazy_conditionally_active(enc)


def check_lazy_conditionally_active(encoder: LazyEncoder):
    if len(encoder.design_vars) == 0:
        return

    any_inactive = np.zeros((len(encoder.design_vars),), dtype=bool)
    for _, des_vectors in encoder.get_all_design_vectors().items():
        any_inactive_existence = np.any(des_vectors == X_INACTIVE_VALUE, axis=0)
        any_inactive[:len(any_inactive_existence)] |= any_inactive_existence

    assert np.all(any_inactive == [dv.conditionally_active for dv in encoder.design_vars])


def test_lazy_imputer_none_exist():
    enc = DummyLazyEncoder(DummyImputer())
    enc.set_settings(MatrixGenSettings(src=[Node([0, 1]), Node([0, 1])], tgt=[Node([0, 1]), Node([0, 1])]))

    _, matrix, _ = enc._decode_vector([0, 1, 1, 0])
    assert np.all(matrix == np.array([[0, 1], [1, 0]]))

    dv, matrix = enc.get_matrix([0, 1, 1, 0], NodeExistence(src_exists=[False, False], tgt_exists=[False, False]))
    assert np.all(dv == [-1, -1, -1, -1])
    assert np.all(matrix == 0)

    dv, matrix = enc.get_matrix(np.array([0, 1, 1, 0]),
                                NodeExistence(src_exists=[False, False], tgt_exists=[False, False]))
    assert np.all(dv == [-1, -1, -1, -1])
    assert np.all(matrix == 0)

    all_x = enc.get_all_design_vectors()[NodeExistence()]
    assert all_x.shape == (7, 4)
    matrix_seen = set()
    for x in all_x:
        imp_x, matrix = enc.get_matrix(x)
        matrix_seen.add(tuple(matrix.ravel()))
        assert np.all(imp_x == x)
    assert len(matrix_seen) == all_x.shape[0]


def test_lazy_imputer_no_exist_correct():
    enc = DummyLazyEncoder(DummyImputer())
    enc.set_settings(MatrixGenSettings(src=[Node([0, 1]), Node([0, 1])], tgt=[Node([0, 1]), Node([0, 1])]))

    dv, matrix = enc.get_matrix([0, 1, 1, 1], NodeExistence(src_exists=[True, False]))
    assert np.all(dv == [0, 1, 1, 1])
    assert np.all(matrix == np.array([[0, 1], [0, 0]]))

    dv, matrix = enc.get_matrix([0, 1, 1, 1], NodeExistence(tgt_exists=[True, False]))
    assert np.all(dv == [0, 1, 1, 1])
    assert np.all(matrix == np.array([[0, 0], [1, 0]]))


def test_lazy_imputer_cache():
    enc = DummyLazyEncoder(DummyImputer())
    enc.set_settings(MatrixGenSettings(src=[Node([0, 1]), Node([0, 1])], tgt=[Node([0, 1]), Node([0, 1])]))

    def _stop_impute(*args, **kwargs):
        raise RuntimeError

    for _ in range(2):
        n_imputed = 0
        for dv in itertools.product(*[list(range(dv.n_opts)) for dv in enc.design_vars]):
            dv = list(dv)
            imp_dv, matrix = enc.get_matrix(dv)
            if matrix[0, 0] == -1:
                n_imputed += 1

        assert len(enc._imputer._impute_cache) == n_imputed
        enc._imputer._impute = _stop_impute


def test_lazy_assignment_manager():
    src = [Node([0, 1]), Node([0, 1])]
    tgt = [Node([0, 1]), Node([0, 1])]
    manager = LazyAssignmentManager(MatrixGenSettings(src, tgt), DummyLazyEncoder(DummyImputer()))

    dv = manager.design_vars
    assert len(dv) == 4
    assert all([d.n_opts == 2 for d in dv])

    dv, is_active, matrix = manager.get_matrix([0, 1, 1, 0])
    assert np.all(dv == [0, 1, 1, 0])
    assert np.all(is_active)
    assert np.all(matrix == np.array([[0, 1], [1, 0]]))

    dv_, is_active_ = manager.correct_vector([0, 1, 1, 0])
    assert np.all(dv_ == dv)
    assert np.all(is_active_)
    dv_, is_active_ = manager.correct_vector([0, 1, 1, 1])
    assert np.all(dv_ == [0, 0, 0, 0])
    assert np.all(~is_active_)

    _, _, conn_idx = manager.get_conn_idx([0, 1, 1, 0])
    assert conn_idx == [(0, 1), (1, 0)]
    _, _, conn_idx = manager.get_conn_idx([0, 1, 1, 1])
    assert not conn_idx


def test_one_to_one(gen_one_per_existence: AggregateAssignmentMatrixGenerator):
    g = gen_one_per_existence
    encoder = DummyLazyEncoder(DummyImputer())
    encoder.set_settings(g.settings)
