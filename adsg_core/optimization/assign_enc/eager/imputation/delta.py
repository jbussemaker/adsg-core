import itertools
import numpy as np
from typing import *
from adsg_core.optimization.assign_enc.encoding import *

__all__ = ['DeltaImputer']


class DeltaImputer(EagerImputer):
    """Imputes by modifying the design vector (gradually increasing the amount of mods) until a valid one is found."""

    n_max_tries = 10000

    def __init__(self):
        super().__init__()
        self._dv_idx_map = {}

    def initialize(self, *args, **kwargs):
        super().initialize(*args, **kwargs)
        self._dv_idx_map = {existence: {tuple(dv): i for i, dv in enumerate(dvs)}
                            for existence, dvs in self._design_vectors_zero.items()}

    def impute(self, vector: DesignVector, existence: NodeExistence, matrix_mask: MatrixSelectMask) -> Tuple[DesignVector, np.ndarray]:
        design_vectors = self._get_design_vectors(existence)[matrix_mask, :]
        if len(vector) == 0 or len(design_vectors) == 0:
            return vector, np.zeros((0, 0), dtype=int)

        # Determine delta values to try out
        def _sort_by_dist(des_var_delta):
            i_sorted = np.argsort(np.abs(des_var_delta))
            return des_var_delta[i_sorted]

        delta_values = [_sort_by_dist(np.arange(dv.n_opts)-vector[i]) for i, dv in enumerate(self._design_vars)]

        dv_map = self._dv_idx_map[existence]
        n_tries, n_tries_max = 0, self.n_max_tries
        first = True
        for dv_delta in itertools.product(*delta_values):
            if first:  # Skip first as this one has a delta of 0
                first = False
                continue
            dv = tuple(vector+np.array(dv_delta))

            # Check if design vector exists
            if dv in dv_map:
                idx = dv_map[dv]
                if matrix_mask[idx]:
                    return self._return_imputation(idx, existence)

            # Limit the amount of tries
            n_tries += 1
            if n_tries > n_tries_max:
                invalid_matrix = -np.ones(self._get_matrix(existence).shape[1:], dtype=int)
                return vector, invalid_matrix

        return vector, np.zeros((0, 0), dtype=int)

    @staticmethod
    def _calc_dist_euclidean(elements: np.ndarray, target: np.ndarray) -> np.ndarray:
        return np.sqrt(np.sum((elements-target)**2, axis=1))

    @staticmethod
    def _calc_dist_manhattan(elements: np.ndarray, target: np.ndarray) -> np.ndarray:
        return np.sum(np.abs(elements-target), axis=1)

    def __repr__(self):
        return f'{self.__class__.__name__}()'

    def __str__(self):
        return f'Delta Imp'
