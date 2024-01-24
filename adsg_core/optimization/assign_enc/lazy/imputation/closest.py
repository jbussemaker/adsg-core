import itertools
import numpy as np
from typing import *
from adsg_core.optimization.assign_enc.lazy_encoding import *
from adsg_core.optimization.assign_enc.lazy.imputation.delta import *

__all__ = ['LazyClosestImputer']


class LazyClosestImputer(LazyDeltaImputer):
    """Imputes by looking for the closest valid design vector based on euclidean or Mahattan distance.
    Eagerly evaluates itertools.product, so might need too much memory/time."""

    def __init__(self, euclidean=True, n_max_tries=None):
        self.euclidean = euclidean
        super().__init__(n_max_tries=n_max_tries)

    def yield_dv_delta_product(self, delta_values):
        # Sort all delta values by distance (skip first as it has a delta of 0)
        dist_func = self._calc_dist_euclidean if self.euclidean else self._calc_dist_manhattan
        dv_deltas = [np.array(dv_delta) for dv_delta in itertools.product(*delta_values)][1:]  # Skip first as it has a delta of 0
        dv_dist = dist_func(np.array(dv_deltas))

        # Sort by distance
        i_dv_dist = np.argsort(dv_dist)
        for i_dv in i_dv_dist:
            yield dv_deltas[i_dv]

    @staticmethod
    def _calc_dist_euclidean(deltas: np.ndarray) -> np.ndarray:
        return np.sqrt(np.sum(deltas**2, axis=1))

    @staticmethod
    def _calc_dist_manhattan(deltas: np.ndarray) -> np.ndarray:
        return np.sum(np.abs(deltas), axis=1)

    def __repr__(self):
        return f'{self.__class__.__name__}(euclidean={self.euclidean})'

    def __str__(self):
        return f'Closest {"Euc" if self.euclidean else "Mht"} Imp'
