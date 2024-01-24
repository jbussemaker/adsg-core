import numpy as np
from typing import *
from adsg_core.optimization.assign_enc.lazy_encoding import *

__all__ = ['LazyDirectMatrixEncoder']


class LazyDirectMatrixEncoder(LazyEncoder):
    """
    Defines one design variable for each matrix element.
    Uses the following logic to determine the amount of maximum connections per source-target pair:
    - If the connection is excluded, max connections is 0
    - If for either the source or target repeat_allowed is False, max connections is 1
    - Otherwise, the max connections is min(source_max, target_max, overall_max)

    The minimum number of connections is always assumed to be 0.
    If the nr of max connections is 0, the design variable is skipped.
    """

    def __init__(self, imputer: LazyImputer):
        self._dv_idx_map = {}
        super().__init__(imputer)

    def _encode_prepare(self):
        self._dv_idx_map = {}

    def _encode(self, existence: NodeExistence) -> List[DiscreteDV]:
        matrix_gen = self._matrix_gen
        max_conn_mat = matrix_gen.get_max_conn_mat(existence)

        dvs = []
        self._dv_idx_map[existence] = dv_idx_map = []
        for i in range(self.n_src):
            if not existence.has_src(i):
                continue
            for j in range(self.n_tgt):
                if not existence.has_tgt(j):
                    continue

                # Get maximum number of connections
                max_conn = max_conn_mat[i, j]
                if max_conn == 0:
                    continue

                # Define design variable
                if max_conn >= 1:
                    dvs.append(DiscreteDV(n_opts=max_conn+1, conditionally_active=False))
                    dv_idx_map.append((i, j))

        return dvs

    def _decode(self, vector: DesignVector, existence: NodeExistence) -> Optional[Tuple[DesignVector, np.ndarray]]:
        matrix = self.get_empty_matrix()
        dv_map = self._dv_idx_map.get(existence, [])
        for i_dv, (i, j) in enumerate(dv_map):
            matrix[i, j] = vector[i_dv]
        return vector, matrix

    def __repr__(self):
        return f'{self.__class__.__name__}({self._imputer!r})'

    def __str__(self):
        return f'Lazy Direct Matrix + {self._imputer!s}'
