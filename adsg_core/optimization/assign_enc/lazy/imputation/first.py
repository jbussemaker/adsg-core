import itertools
import numpy as np
from typing import *
from adsg_core.optimization.assign_enc.lazy_encoding import *

__all__ = ['LazyFirstImputer']


class LazyFirstImputer(LazyImputer):
    """Imputer that chooses the first possible matrix."""

    def _impute(self, vector: DesignVector, matrix: Optional[np.ndarray], existence: NodeExistence,
                validate: Callable[[np.ndarray], bool], tried_vectors: Set[Tuple[int, ...]]) \
            -> Tuple[DesignVector, np.ndarray]:

        if len(vector) == 0:
            return vector, np.zeros((0, 0), dtype=int)

        cache_key = ('found_first', hash(existence))
        if cache_key in self._impute_cache:
            return self._impute_cache[cache_key]

        # Loop through possible design vectors until one is found that has not been tried yet
        for dv in itertools.product(*[list(range(dv.n_opts)) for dv in self._get_des_vars(existence)[::-1]]):
            # Validate this design vector and associated matrix
            vector = np.array(dv[::-1])
            vector, matrix = self._decode(vector, existence)
            if validate(matrix):
                self._impute_cache[cache_key] = vector, matrix
                return vector, matrix

        self._impute_cache[cache_key] = vector, np.zeros((0, 0), dtype=int)
        return self._impute_cache[cache_key]

    def __repr__(self):
        return f'{self.__class__.__name__}()'

    def __str__(self):
        return f'First Imp'
