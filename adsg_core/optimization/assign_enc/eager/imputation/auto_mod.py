import numpy as np
from typing import *
from adsg_core.optimization.assign_enc.encoding import *

__all__ = ['AutoModImputer']


class AutoModImputer(EagerImputer):
    """
    Imputes design vectors by moving from the start or end to change design variables if they are too restrictive:
    For each design variable (from start or end):
    - Create a new design vector only fixing the values up to that design vector
    - Filter out design vectors matching this partial vector
    - If none were found: try to assign different values to the design variable (skipping the current value)
      - If a value is found that matches 1 or more design vectors, stop the search for a different value
    - If exactly one design vector was found to comply: return this as the imputed design vector
    - If more than one design vectors were found: move to the next design variable
    """

    def __init__(self, reverse=False):
        super().__init__()
        self.reverse = reverse

    def impute(self, vector: DesignVector, existence: NodeExistence, matrix_mask: MatrixSelectMask) -> Tuple[DesignVector, np.ndarray]:
        if len(vector) == 0:
            return vector, np.zeros((0, 0), dtype=int)

        mask = matrix_mask
        partial_vector: PartialDesignVector = [None for _ in range(len(vector))]
        for i_dv in (reversed(range(len(vector))) if self.reverse else range(len(vector))):
            dv_partial_vector = partial_vector.copy()
            dv_partial_vector[i_dv] = vector[i_dv]

            # Find design vectors that match the new partial design vector
            dv_mask = self._filter_design_vectors(dv_partial_vector, existence) & mask
            i_mask, = np.where(dv_mask)

            # If we have found no valid design vectors for this partial vector, modify the value until we have found one
            if len(i_mask) == 0:
                for dv_value in range(self._design_vars[i_dv].n_opts):
                    # Skip current value
                    if dv_value == vector[i_dv]:
                        continue

                    # Find design vectors that match the modified partial design vector
                    dv_partial_vector[i_dv] = dv_value
                    dv_mask = self._filter_design_vectors(dv_partial_vector, existence) & mask
                    i_mask, = np.where(dv_mask)

                    if len(i_mask) > 0:
                        break

                if len(i_mask) == 0:
                    raise RuntimeError('Cannot find any valid design vectors!')

            # If we have found exactly one vector, we can stop searching and return this one
            if len(i_mask) == 1:
                return self._return_imputation(i_mask[0], existence)

            # Otherwise, continue with the next design variable
            else:
                mask = dv_mask
                partial_vector = dv_partial_vector

        # Only happens if there are duplicate design vectors
        raise RuntimeError('Multiple possible design vectors found! Check if there are any duplicates')

    def __repr__(self):
        return f'{self.__class__.__name__}(reverse={self.reverse})'

    def __str__(self):
        return f'Auto Mod Imp{" Rev" if self.reverse else ""}'
