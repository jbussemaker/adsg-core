import numpy as np
from typing import *
from adsg_core.optimization.assign_enc.encoding import *

__all__ = ['ClosestImputer']


class ClosestImputer(EagerImputer):
    """Imputes by looking for the closest valid design vector based on euclidean or Mahattan distance."""

    def __init__(self, euclidean=True):
        super().__init__()
        self.euclidean = euclidean

    def impute(self, vector: DesignVector, existence: NodeExistence, matrix_mask: MatrixSelectMask) \
            -> Tuple[DesignVector, np.ndarray]:
        design_vectors = self._get_design_vectors(existence)[matrix_mask, :]

        if len(design_vectors) == 0:
            return vector, np.zeros((0, 0), dtype=int)

        elements, target = design_vectors, np.array(vector)
        if self.euclidean:
            dist = self._calc_dist_euclidean(elements, target)
        else:
            dist = self._calc_dist_manhattan(elements, target)
        i_min_dist = int(np.argmin(dist))

        i_mask = np.where(matrix_mask)[0]
        return self._return_imputation(i_mask[i_min_dist], existence)

    @staticmethod
    def _calc_dist_euclidean(elements: np.ndarray, target: np.ndarray) -> np.ndarray:
        return np.sqrt(np.sum((elements-target)**2, axis=1))

    @staticmethod
    def _calc_dist_manhattan(elements: np.ndarray, target: np.ndarray) -> np.ndarray:
        return np.sum(np.abs(elements-target), axis=1)

    def __repr__(self):
        return f'{self.__class__.__name__}(euclidean={self.euclidean})'

    def __str__(self):
        return f'Closest {"Euc" if self.euclidean else "Mht"} Imp'
