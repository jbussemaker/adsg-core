import numpy as np
from typing import *
from adsg_core.optimization.assign_enc.encoding import *

__all__ = ['FirstImputer']


class FirstImputer(EagerImputer):
    """Imputer that simply chooses the first possible matrix."""

    def impute(self, vector: DesignVector, existence: NodeExistence, matrix_mask: MatrixSelectMask) -> Tuple[DesignVector, np.ndarray]:
        i_where_ok = np.where(matrix_mask)[0]
        if len(i_where_ok) == 0:
            return vector, np.zeros((0, 0), dtype=int)
        i_mat = i_where_ok[0]
        return self._return_imputation(i_mat, existence)

    def __repr__(self):
        return f'{self.__class__.__name__}()'

    def __str__(self):
        return f'First Imp'
