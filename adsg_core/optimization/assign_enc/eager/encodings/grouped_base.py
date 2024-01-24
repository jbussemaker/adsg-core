import itertools
import numpy as np
from typing import Optional
from adsg_core.optimization.assign_enc.encoding import *
from collections import defaultdict

__all__ = ['GroupedEncoder', 'GroupByIndexEncoder']


class GroupedEncoder(EagerEncoder):
    """
    Base class for an encoder that recursively separates the matrix based on some grouping criteria, until all
    matrices are associated to one unique design vector.

    By default, grouping values are normalized within sub-groups, leading to much less value options per design
    variable. To disable, set normalize_within_group=False.
    """

    def __init__(self, *args, normalize_within_group=True, **kwargs):
        self.normalize_within_group = normalize_within_group
        super().__init__(*args, **kwargs)

    def _encode(self, matrix: np.ndarray) -> np.ndarray:
        return self.group_by_values(
            self._get_grouping_values(matrix), normalize_within_group=self.normalize_within_group,
            ordinal_base=self._get_ordinal_conv_base())

    def _get_ordinal_conv_base(self) -> Optional[int]:
        pass

    @classmethod
    def group_by_values(cls, group_by_values: np.ndarray, normalize_within_group=True,
                        ordinal_base: int = None, n_declared_start=None) -> np.ndarray:
        """
        Get design vectors that uniquely map to different value combinations. Example:
        [[1 2],      [[0 0],
         [1 3],  -->  [0 1],
         [2 2],       [1 -1],
         [3 2],       [2 -1],  # -1 means inactive

        Optionally normalize only after determining all groups.
        Optionally convert the grouped design vector values to some other base,
        e.g. for base 2: [[1, 2, 3, 4]].T --> [[0, 0], [0, 1], [1, 0], [1, 1]]"""
        design_vectors = np.empty(group_by_values.shape, dtype=int)
        row_mask_list = [np.arange(group_by_values.shape[0])]

        detect_imp_ratio = cls._early_detect_high_imp_ratio
        n_valid = group_by_values.shape[0]
        if n_valid == 0:
            return np.zeros((0, 0), dtype=int)

        n_declared_total = n_declared_start if n_declared_start is not None else 1
        if detect_imp_ratio is not None and (n_declared_total/n_valid) >= detect_imp_ratio:
            raise DetectedHighImpRatio(cls, n_declared_total/n_valid)

        # Loop over columns
        col_is_active = np.ones((group_by_values.shape[1],), dtype=bool)
        for i_col in range(group_by_values.shape[1]):

            # Loop over current sub-divisions
            next_row_masks = []
            grouping_needed = False
            all_inactive = True
            col_values = set()
            for row_mask in row_mask_list:

                # Check if there are multiple design vectors in this subdivision
                group_by_values_sub = group_by_values[row_mask, i_col]
                if len(group_by_values_sub) == 1:
                    # All subsequent columns of this design vector will also be inactive
                    design_vectors[row_mask, i_col:] = X_INACTIVE_VALUE
                    continue

                # Check if there are multiple unique values in this sub-division
                value_dict = defaultdict(list)
                for value_idx, value in enumerate(group_by_values_sub):
                    value_dict[value].append(value_idx)
                unique_values = sorted(list(value_dict.keys()))

                if unique_values[0] < 0:
                    raise ValueError('Values to group by should not contain negative values!')

                if len(unique_values) == 1:
                    next_row_masks.append(row_mask)
                    design_vectors[row_mask, i_col] = X_INACTIVE_VALUE
                    grouping_needed = True
                    continue
                all_inactive = False

                # Set design vector values and get next indices for the subset of the different values
                for value_idx, value in enumerate(unique_values):
                    next_row_mask = row_mask[value_dict[value]]

                    assigned_value = value_idx if normalize_within_group else value
                    design_vectors[next_row_mask, i_col] = assigned_value
                    col_values.add(assigned_value)

                    if len(next_row_mask) == 1:
                        design_vectors[next_row_mask, i_col+1:] = X_INACTIVE_VALUE
                    else:
                        grouping_needed = True
                        next_row_masks.append(next_row_mask)

            # Detect high imputation ratios early
            if detect_imp_ratio is not None:
                # Extend the total declared values (which will correspond to the nr of options for the design vars)
                n_declared_total *= max(1, len(col_values))

                # Calculate minimum imputation ratio that can be expected: the imputation ratio can only become higher
                # by adding additional columns, so it is indeed possible to calculate the lower bound
                min_imp_ratio = n_declared_total/n_valid

                # If grouping is still needed, at least one additional design variable with minimum 2 options will be
                # added, so the lower bound can be multiplied by 2
                if grouping_needed:
                    min_imp_ratio *= 2

                if min_imp_ratio >= detect_imp_ratio:
                    raise DetectedHighImpRatio(cls, min_imp_ratio)

            # Set inactive flag
            if all_inactive:
                col_is_active[i_col] = False

            # Stop grouping if not needed anymore
            if not grouping_needed:
                design_vectors = design_vectors[:, :i_col+1]
                break

            row_mask_list = next_row_masks

        # Normalize design vectors
        if not normalize_within_group:
            design_vectors = cls.normalize_design_vectors(design_vectors)

        # Remove columns where there are no alternatives
        design_vectors = design_vectors[:, col_is_active[:design_vectors.shape[1]]]

        # Convert to other base
        if ordinal_base is not None:
            design_vectors = cls.convert_to_base(design_vectors, ordinal_base)
            has_alternatives = np.any(design_vectors > 0, axis=0)
            design_vectors = design_vectors[:, has_alternatives]

        return design_vectors

    @classmethod
    def convert_to_base(cls, values: np.ndarray, base: int = 2, n_declared_start=None) -> np.ndarray:
        """Convert the ordinal-encoded values to another base"""
        if base < 2 or base > 9:
            raise ValueError('Base should be between 2 and 9')
        if values.shape[1] == 0:
            return values

        detect_imp_ratio = cls._early_detect_high_imp_ratio
        n_valid = values.shape[0]
        if n_valid == 0:
            return np.zeros((0, 0), dtype=int)

        n_declared_total = n_declared_start if n_declared_start is not None else 1
        if detect_imp_ratio is not None and (n_declared_total/n_valid) >= detect_imp_ratio:
            raise DetectedHighImpRatio(cls, n_declared_total/n_valid)

        columns = []
        for i_col in range(values.shape[1]):
            # Get unique (active) values
            inactive_idx = []
            value_dict = defaultdict(list)
            for value_idx, value in enumerate(values[:, i_col]):
                if value == X_INACTIVE_VALUE:
                    inactive_idx.append(value_idx)
                    continue
                value_dict[value].append(value_idx)
            unique = sorted(list(value_dict.keys()))

            # Determine how many columns we need (i.e. how many digits does the largest number have)
            n_converted = len(np.base_repr(len(value_dict)-1))
            col_converted = np.zeros((values.shape[0], n_converted), dtype=int)
            col_converted[inactive_idx, :] = X_INACTIVE_VALUE

            # Convert and set values
            for i_value, value in enumerate(unique):
                base_converted = [
                    int(char) for char in (np.base_repr(i_value, base=base) if base != 2 else np.binary_repr(i_value))]
                col_converted[value_dict[value], -len(base_converted):] = base_converted

            # Early detect high imputation ratio
            if detect_imp_ratio is not None:
                n_declared_total *= np.prod(np.max(col_converted, axis=0)+1, dtype=float)
                min_imp_ratio = n_declared_total/n_valid
                if i_col < values.shape[1]-1:
                    min_imp_ratio *= 2

                if min_imp_ratio >= detect_imp_ratio:
                    raise DetectedHighImpRatio(cls, min_imp_ratio)

            columns.append(col_converted)
        return np.column_stack(columns)

    def _get_grouping_values(self, matrix: np.ndarray) -> np.ndarray:
        """Get a n_mat x n_dv matrix with values to sub-divide design variables by"""
        raise NotImplementedError

    def __repr__(self):
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError


class GroupByIndexEncoder(GroupedEncoder):
    """Grouping encoder that recursively separates the remaining matrices in n groups."""

    def __init__(self, *args, n_groups=2, **kwargs):
        self.n_groups = max(n_groups, 2)
        super().__init__(*args, **kwargs)

    def _get_grouping_values(self, matrix: np.ndarray) -> np.ndarray:
        n_groups = self.n_groups
        n_var = int(np.ceil(np.log2(matrix.shape[0])))
        design_vectors = np.array(list(itertools.product(*[range(n_groups) for _ in range(n_var)]))[:matrix.shape[0]])
        for i_var in reversed(range(1, n_var)):
            design_vectors[:, i_var:] += np.array([n_groups*design_vectors[:, i_var-1]]).T
        return design_vectors

    def __repr__(self):
        return f'{self.__class__.__name__}({self._imputer!r}, n_groups={self.n_groups})'

    def __str__(self):
        return f'Group By Index ({self.n_groups} groups) + {self._imputer!s}'
