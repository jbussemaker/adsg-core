import numpy as np
from adsg_core.optimization.assign_enc.encoding import *

__all__ = ['DirectMatrixEncoder']


class DirectMatrixEncoder(EagerEncoder):
    """Defines one design variable for each matrix element."""

    def __init__(self, *args, remove_gaps=True, **kwargs):
        self.remove_gaps = remove_gaps
        super().__init__(*args, **kwargs)

    def _encode(self, matrix: np.ndarray) -> np.ndarray:
        # Map matrix elements to design vector values
        design_vectors = self.flatten_matrix(matrix)

        # Normalize design vectors (move to 0 and optionally remove value gaps)
        return self.normalize_design_vectors(design_vectors, remove_gaps=self.remove_gaps)

    def __repr__(self):
        return f'{self.__class__.__name__}({self._imputer!r}, remove_gaps={self.remove_gaps})'

    def __str__(self):
        remove_gaps_str = ' Rem Gaps' if self.remove_gaps else ''
        return f'Direct Matrix{remove_gaps_str} + {self._imputer!s}'
