import numpy as np
from typing import *
from adsg_core.optimization.assign_enc.lazy_encoding import *

__all__ = ['EnumOrdinalEncoder']


class EnumOrdinalEncoder(QuasiLazyEncoder):
    """Defines one design variable with the same number of options as the number of possible assignment patterns."""

    def _encode_matrix(self, matrix: np.ndarray, existence: NodeExistence) -> List[DiscreteDV]:
        n_mat = matrix.shape[0]
        if n_mat <= 1:
            return []
        return [DiscreteDV(n_opts=n_mat, conditionally_active=False)]

    def _decode_matrix(self, vector: DesignVector, matrix: np.ndarray, existence: NodeExistence) \
            -> Optional[Tuple[DesignVector, np.ndarray]]:
        i_mat = vector[0] if len(vector) > 0 else 0
        if i_mat >= matrix.shape[0]:
            return [matrix.shape[0]-1], matrix[-1, :, :]
        return vector, matrix[i_mat, :, :]

    def _do_generate_random_dv_mat(self, n: int, existence: NodeExistence, matrix: np.ndarray,
                                   design_vars: List[DiscreteDV]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if n < matrix.shape[0]:
            i_selected = np.random.choice(matrix.shape[0], n, replace=False)
        else:
            i_selected = np.arange(matrix.shape[0])

        design_vectors = np.array([i_selected]).T
        matrices = matrix[i_selected, :, :]
        return design_vectors, matrices

    def _do_get_all_design_vectors(self, existence: NodeExistence, matrix: np.ndarray, design_vars: List[DiscreteDV]) \
            -> np.ndarray:
        design_vectors = np.array([np.arange(matrix.shape[0])]).T
        return design_vectors

    def __repr__(self):
        return f'{self.__class__.__name__}({self._imputer!r})'

    def __str__(self):
        return f'Enum Ord + {self._imputer!s}'
