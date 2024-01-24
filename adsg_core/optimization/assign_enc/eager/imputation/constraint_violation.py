import numpy as np
from typing import *
from adsg_core.optimization.assign_enc.encoding import *

__all__ = ['ConstraintViolationImputer']


class ConstraintViolationImputer(EagerImputer):
    """Does not do imputation, but sends a matrix with -1 as values to indicate an invalid design point.
    The optimization problem should represent this as a violated constraint."""

    def impute(self, vector: DesignVector, existence: NodeExistence, matrix_mask: MatrixSelectMask) -> Tuple[DesignVector, np.ndarray]:
        matrix = self._get_matrix(existence)
        if matrix.shape[0] == 0:
            return vector, np.zeros((0, 0), dtype=int)
        invalid_matrix = matrix[0, :, :]*0-1
        return vector, invalid_matrix

    def __repr__(self):
        return f'{self.__class__.__name__}()'

    def __str__(self):
        return f'Constr Vio Imp'
