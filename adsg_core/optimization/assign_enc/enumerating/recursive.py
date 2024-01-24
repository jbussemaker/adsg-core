import string
import numpy as np
from typing import *
from adsg_core.optimization.assign_enc.lazy_encoding import *

__all__ = ['EnumRecursiveEncoder']


class EnumRecursiveEncoder(QuasiLazyEncoder):
    """Defines design variables where each has a fixed nr of options to cover all possible assignment patterns."""

    _str_lookup = {char: i for i, char in enumerate(string.digits+string.ascii_uppercase)}

    def __init__(self, *args, n_divide=2, **kwargs):
        self.n_divide = max(2, n_divide)
        super().__init__(*args, **kwargs)
        self._dv_last = {}
        self._dv_inactive_key = {}

    def _encode_prepare(self):
        super()._encode_prepare()
        self._dv_last = {}
        self._dv_inactive_key = {}

    def _encode_matrix(self, matrix: np.ndarray, existence: NodeExistence) -> List[DiscreteDV]:
        n_mat = matrix.shape[0]
        if n_mat <= 1:
            return []

        n = self.n_divide
        n_var = int(np.ceil(np.log(n_mat)/np.log(n)))

        # Get design vector values that lead to inactive variables (due to nr cutoff)
        dv_last = np.array(self.base_repr_int(n_mat-1, n))
        i_inactive = np.where(dv_last == 0)[0]
        if len(i_inactive) > 0:
            left_side_values = dv_last[:i_inactive[-1]+1].copy()
            self._dv_inactive_key[existence] = (i_inactive, left_side_values)
            dv_last[i_inactive] = X_INACTIVE_VALUE
        self._dv_last[existence] = dv_last

        n_opts = np.ones((n_var,), dtype=int)*n
        n_opts[0] = dv_last[0]+1
        return [DiscreteDV(n_opts=n_opt, conditionally_active=dv_last[i] == X_INACTIVE_VALUE)
                for i, n_opt in enumerate(n_opts)]

    def _decode_matrix(self, vector: DesignVector, matrix: np.ndarray, existence: NodeExistence) \
            -> Optional[Tuple[DesignVector, np.ndarray]]:
        if len(vector) == 0:
            return vector, matrix[0, :, :]

        i_mat = np.sum((self.n_divide**np.arange(len(vector)))*vector[::-1])
        if i_mat >= matrix.shape[0]:
            return self._dv_last[existence], matrix[-1, :, :]

        vectors = np.array([vector])
        self._set_inactive(vectors, existence)
        vector = vectors[0, :]

        return vector, matrix[i_mat, :, :]

    @classmethod
    def base_repr_int(cls, value: int, base: int) -> List[int]:
        return [cls._str_lookup[char] for char in (np.binary_repr(value) if base == 2 else np.base_repr(value, base))]

    def _do_generate_random_dv_mat(self, n: int, existence: NodeExistence, matrix: np.ndarray,
                                   design_vars: List[DiscreteDV]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if n < matrix.shape[0]:
            i_selected = np.random.choice(matrix.shape[0], n, replace=False)
        else:
            i_selected = np.arange(matrix.shape[0])

        design_vectors = self._encode_design_vectors(i_selected, len(design_vars), existence)
        matrices = matrix[i_selected, :, :]
        return design_vectors, matrices

    def _do_get_all_design_vectors(self, existence: NodeExistence, matrix: np.ndarray, design_vars: List[DiscreteDV]) \
            -> np.ndarray:
        design_vectors = self._encode_design_vectors(np.arange(matrix.shape[0]), len(design_vars), existence)
        return design_vectors

    def _encode_design_vectors(self, i_selected, n_dv: int, existence: NodeExistence) -> np.ndarray:
        design_vectors = np.zeros((len(i_selected), n_dv), dtype=int)
        for i_dv, i_sel in enumerate(i_selected):
            base_repr = self.base_repr_int(i_sel, self.n_divide)
            design_vectors[i_dv, -len(base_repr):] = base_repr

        self._set_inactive(design_vectors, existence)
        return design_vectors

    def _set_inactive(self, design_vectors: np.ndarray, existence: NodeExistence):
        dv_inactive_key = self._dv_inactive_key.get(existence)
        if dv_inactive_key is not None:
            i_inactive, left_side_values = dv_inactive_key
            i_dv_inactive = np.all(design_vectors[:, :len(left_side_values)] == left_side_values, axis=1)
            for ix in i_inactive:
                design_vectors[i_dv_inactive, ix] = X_INACTIVE_VALUE

    def __repr__(self):
        return f'{self.__class__.__name__}({self._imputer!r})'

    def __str__(self):
        return f'Enum Rec {self.n_divide} + {self._imputer!s}'
