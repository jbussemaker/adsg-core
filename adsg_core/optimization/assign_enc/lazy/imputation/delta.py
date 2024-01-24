import itertools
import numpy as np
from typing import *
from adsg_core.optimization.assign_enc.lazy_encoding import *

__all__ = ['LazyDeltaImputer']


class LazyDeltaImputer(LazyImputer):
    """Imputes by looking for the first valid design vector created by iterating over delta values."""

    _n_max_tries_default = 10000

    def __init__(self, n_max_tries=None):
        self.n_max_tries = n_max_tries if n_max_tries is not None else self._n_max_tries_default
        super().__init__()

    def _impute(self, vector: DesignVector, matrix: Optional[np.ndarray], existence: NodeExistence,
                validate: Callable[[np.ndarray], bool], tried_vectors: Set[Tuple[int, ...]]) -> Tuple[DesignVector, np.ndarray]:

        # Check if we have any design variables
        if len(vector) == 0:
            return vector, np.zeros((0, 0), dtype=int)

        # Determine delta values to try out
        def _sort_by_dist(des_var_delta):
            i_sorted = np.argsort(np.abs(des_var_delta))
            return des_var_delta[i_sorted]

        delta_values = [_sort_by_dist(np.arange(dv.n_opts)-vector[i])
                        for i, dv in enumerate(self._get_des_vars(existence))]

        # Loop over all delta values
        n_tries, n_tries_max = 0, self.n_max_tries
        for dv_delta in self.yield_dv_delta_product(delta_values):
            dv = vector+np.array(dv_delta)

            # Validate associated matrix
            results = self._decode(dv, existence)
            if results is not None:
                dv, matrix = results
                if validate(matrix):
                    return dv, matrix

            # Limit the amount of tries
            n_tries += 1
            if n_tries > n_tries_max:
                invalid_matrix = -np.ones((len(self._matrix_gen.src), len(self._matrix_gen.tgt)), dtype=int)
                return vector, invalid_matrix

        return vector, np.zeros((0, 0), dtype=int)

    def yield_dv_delta_product(self, delta_values):
        first = True
        for dv_delta in itertools.product(*delta_values):
            if first:  # Skip first as this one has a delta of 0
                first = False
                continue
            yield dv_delta

    def __repr__(self):
        return f'{self.__class__.__name__}()'

    def __str__(self):
        return f'Delta Imp'
