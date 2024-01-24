import numpy as np
from typing import *
from adsg_core.optimization.assign_enc.lazy_encoding import *

__all__ = ['LazyConstraintViolationImputer']


class LazyConstraintViolationImputer(LazyImputer):
    """Does not do imputation, but sends a matrix with -1 as values to indicate an invalid design point.
    The optimization problem should represent this as a violated constraint."""

    def impute(self, vector: DesignVector, matrix: Optional[np.ndarray], existence: NodeExistence,
               tried_vectors: set = None) -> Tuple[DesignVector, np.ndarray]:
        invalid_matrix = -np.ones((len(self._matrix_gen.src), len(self._matrix_gen.tgt)), dtype=int)
        return vector, invalid_matrix

    def _impute(self, *args, **kwargs):
        raise RuntimeError

    def __repr__(self):
        return f'{self.__class__.__name__}()'

    def __str__(self):
        return f'Constr Vio Imp'
