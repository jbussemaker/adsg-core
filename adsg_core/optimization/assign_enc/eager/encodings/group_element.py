import numpy as np
from typing import Optional
from adsg_core.optimization.assign_enc.eager.encodings.grouped_base import *

__all__ = ['ElementGroupedEncoder', 'ConnIdxGroupedEncoder']


class ElementGroupedEncoder(GroupedEncoder):
    """Group by the value of each design variable"""

    def _get_grouping_values(self, matrix: np.ndarray) -> np.ndarray:
        return self.flatten_matrix(matrix)

    def __repr__(self):
        return f'{self.__class__.__name__}({self._imputer!r})'

    def __str__(self):
        normalize_str = ' Norm Grp' if self.normalize_within_group else ''
        return f'Group By Element{normalize_str} + {self._imputer!s}'


class ConnIdxGroupedEncoder(GroupedEncoder):
    """Group by the connection indices (positions); should work well for permutations"""

    def __init__(self, *args, by_src=True, binary=False, **kwargs):
        self.by_src = by_src
        self.normalize_within_group = True
        self.binary = binary
        super().__init__(*args, **kwargs)

    def _get_ordinal_conv_base(self) -> Optional[int]:
        if self.binary:
            return 2

    def _get_grouping_values(self, matrix: np.ndarray) -> np.ndarray:
        if matrix.shape[0] == 0:
            return np.empty((0, 0), dtype=int)

        self.normalize_within_group = True
        conn_idx_values = []
        for i_from in range(matrix.shape[1] if self.by_src else matrix.shape[2]):
            from_matrix = matrix[:, i_from, :] if self.by_src else matrix[:, :, i_from]
            conn_idx = self.get_conn_indices(from_matrix)
            if conn_idx is not None:
                conn_idx_values.append(conn_idx)

        if len(conn_idx_values) == 0:
            return np.empty((matrix.shape[0], 0), dtype=int)
        return np.column_stack(conn_idx_values)

    @classmethod
    def get_conn_indices(cls, conn_matrix: np.ndarray) -> Optional[np.ndarray]:
        n_conn_max = np.max(np.sum(conn_matrix, axis=1))
        if n_conn_max == 0:
            return
        conn_idx = -np.ones((conn_matrix.shape[0], n_conn_max), dtype=int)
        for i, row in enumerate(conn_matrix):
            cls._set_conn_indices(row, conn_idx[i, :])
        return conn_idx+1

    @staticmethod
    def _set_conn_indices(conn_arr: np.ndarray, tgt: np.ndarray):
        offset = 0
        for i, n in enumerate(conn_arr):
            if n > 0:
                tgt[offset:offset+n] = i
                offset += n

    def __repr__(self):
        return f'{self.__class__.__name__}({self._imputer!r}, by_src={self.by_src}, binary={self.binary})'

    def __str__(self):
        bin_str = ' Bin' if self.binary else ''
        return f'Group By {"Src" if self.by_src else "Tgt"} Conn Idx{bin_str} + {self._imputer!s}'
