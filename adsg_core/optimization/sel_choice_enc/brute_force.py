"""
MIT License

Copyright: (c) 2023, Deutsches Zentrum fuer Luft- und Raumfahrt e.V.
Contact: jasper.bussemaker@dlr.de

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import numpy as np
from typing import *
from cached_property import cached_property
from adsg_core.optimization.sel_choice_enc.encoder import *

__all__ = ['BruteForceSelectionChoiceEncoder']


class BruteForceSelectionChoiceEncoder(SelectionChoiceEncoder):
    """
    Reference implementation of the selection-choice encoder: derives all relevant information from
    exhaustively-generated design vectors and node statuses. Works correctly for small problems but quickly runs into
    memory and time constraints for larger problems.
    """

    def _get_design_variable_options(self) -> List[int]:
        """Get for each design variable the number of discrete options (at least 2)"""
        x_all, _ = self.all_design_vectors_and_statuses
        if x_all.shape[0] == 0 or x_all.shape[1] == 0:
            return []
        return list(np.max(x_all, axis=0)+1)

    def _design_variable_is_forced(self) -> List[bool]:
        """Get for each design variable whether it is forced or not"""
        if self.n_valid == 0:
            return []

        # Loop over each choice
        x_all, _ = self.all_design_vectors_and_statuses
        x_all_imputed = x_all.copy()
        x_all_imputed[x_all_imputed == X_INACTIVE_VALUE] = 0
        is_forced = [False for _ in range(x_all.shape[1])]
        for i_x in range(1, x_all.shape[1]):
            # Get unique values
            values, idx = np.unique(x_all_imputed[:, i_x], return_inverse=True)

            # Check if any of the left-side design variable values for each of the unique values overlaps
            all_left_side_dvs = set()
            for i_value in range(len(values)):
                dv_idx = np.where(idx == i_value)[0]
                left_side_dvs = {hash(tuple(dv)) for dv in x_all_imputed[dv_idx, :i_x]}
                if len(left_side_dvs & all_left_side_dvs) > 0:
                    break
                all_left_side_dvs |= left_side_dvs
            else:
                is_forced[i_x] = True

        return is_forced

    def _count_n_valid(self) -> int:
        """Get the number of valid design vectors"""
        return self.all_design_vectors_and_statuses[0].shape[0]

    def get_design_vector_indices(self, choice_opt_idx_map: Dict[int, int]) -> Set[int]:
        """
        Get design vectors indices conforming to a subset of choice_idx-opt_idx selections.
        Note: inactive design variables are matched by any opt_idx
        """
        x_all, _ = self.all_design_vectors_and_statuses
        return self._match_dv_indices(x_all, choice_opt_idx_map)

    @staticmethod
    def _match_dv_indices(x_all: np.ndarray, choice_opt_idx_map: Dict[int, int]) -> Set[int]:
        if x_all.shape[1] == 0:
            return {0} if x_all.shape[0] == 1 else set()

        matched_dv_idx = None
        for i_choice, i_opt in choice_opt_idx_map.items():
            # Match exact values and inactive values
            i_matches = np.where((x_all[:, i_choice] == i_opt) | (x_all[:, i_choice] == X_INACTIVE_VALUE))[0]

            if matched_dv_idx is None:
                matched_dv_idx = set(i_matches)
            else:
                matched_dv_idx &= set(i_matches)

        return matched_dv_idx if matched_dv_idx is not None else set()

    def get_design_vector_index(self, opt_idx: List[int], i_dv_excluded: Set[int] = None) -> Tuple[List[int], int]:
        """
        Get the design vector and associated index that most closely represents the given selection of choice-opt_idx
        values. Optionally a set of not-allowed design vector indices can be given.
        If none is found, a RuntimeError should be thrown.
        """

        # Check inputs
        if len(opt_idx) != self.n_choices:
            raise ValueError(f'Incorrect nr of options, expected {self.n_choices}, got {len(opt_idx)}')
        x_all, _ = self.all_design_vectors_and_statuses

        # If there are no choices, return the first possible design vector
        if len(opt_idx) == 0:
            return list(x_all[0, :]), 0

        # Initialize the exclusion set
        if i_dv_excluded is None:
            i_dv_excluded = set()
        if len(i_dv_excluded) == self.n_valid:
            raise RuntimeError('All design vectors are excluded!')

        # Check if the design vector has an exact match, ignoring forced choices
        opt_idx = np.array(opt_idx, dtype=int)
        opt_idx_not_forced = opt_idx[self._is_non_forced]
        i_dv = self._dv_map.get(tuple(opt_idx_not_forced))
        if i_dv is not None and i_dv not in i_dv_excluded:
            return list(x_all[i_dv, :]), i_dv

        # Otherwise, find the closest design vector
        x_all_non_forced: np.ndarray = x_all[:, self._is_non_forced]
        is_inactive: np.ndarray = x_all_non_forced == X_INACTIVE_VALUE

        idx_match: np.ndarray = x_all_non_forced == opt_idx_not_forced
        idx_match[is_inactive] = True
        nr_match = np.sum(idx_match, axis=1)
        for i_excl in i_dv_excluded:
            nr_match[i_excl] = -1

        highest_match_mask = nr_match == np.max(nr_match)
        i_matched = np.where(highest_match_mask)[0]
        if len(i_matched) == 1:
            i_dv = i_matched[0]
            return list(x_all[i_dv, :]), i_dv

        # If multiple design vectors have the same nr of matched design variables, find the closest
        diff: np.ndarray = x_all_non_forced[highest_match_mask, :]-opt_idx_not_forced
        diff = diff.astype(float)
        diff[is_inactive[highest_match_mask, :]] = .25
        dist = np.sqrt(np.nansum(diff**2, axis=1))
        i_min_dist = i_matched[np.argmin(dist)]
        return list(x_all[i_min_dist, :]), i_min_dist

    @cached_property
    def _is_non_forced(self):
        return ~np.array(self.design_variable_is_forced, dtype=bool)

    @cached_property
    def _dv_map(self):
        x_all, _ = self.all_design_vectors_and_statuses
        non_forced = self._is_non_forced
        return {tuple(xi): i for i, xi in enumerate(x_all[:, non_forced])}

    def get_node_existence_mask(self, node_ids: List[int], i_dv: int = None) -> np.ndarray:
        """
        Returns whether for each design vector the given node_ids exist or not.
        Optionally returns the status only for one specific design vector index.
        """
        node_idx_map = self.matrix_nodes_map
        i_nodes = [node_idx_map[node_id] for node_id in node_ids]

        _, all_nodes_status = self.all_design_vectors_and_statuses
        nodes_status: np.ndarray = all_nodes_status[:, i_nodes]
        if i_dv is not None:
            nodes_status = nodes_status[i_dv, :]

        # The nodes exist when their status is confirmed
        return nodes_status == Diag.CONFIRMED.value

    def generate_all_design_vectors_and_statuses(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate all valid design vectors and status arrays"""
        node_idx = self.matrix_nodes_map
        opt_node_map = self.choice_opt_map

        base_matrix = self.influence_matrix

        opt_invalid_status = {Diag.REMOVED.value, Diag.INFEASIBLE_OPTION.value}
        opt_idx_map = {node_idx[choice_node_id]: [node_idx[opt_node_id] for opt_node_id in opt_node_ids]
                       for choice_node_id, opt_node_ids in opt_node_map.items()}

        choice_constraint_map = {(node_idx[choice_id], node_idx[opt_id]):
                                 [(node_idx[c_id], node_idx[o_id]) for c_id, o_id in removed_map]
                                 for (choice_id, opt_id), removed_map in self.choice_constraint_map.items()}

        initial_status = Diag.INITIAL.value
        confirmed_status = Diag.CONFIRMED.value
        removed_status = Diag.REMOVED.value
        choice_made_status = Diag.CHOICE_MADE.value

        # Get choice node dependency
        choice_node_ids = self.choice_node_ids
        n_choice_nodes = len(choice_node_ids)
        i_choice_nodes = sorted([node_idx[choice_node_id] for choice_node_id in choice_node_ids])

        # Get choice-option influence map
        confirmation_influence_map = {}
        removal_influence_map = {}
        for opt_node_ids in opt_node_map.values():
            for opt_node_id in opt_node_ids:
                i_opt_ = node_idx[opt_node_id]
                confirmation_influence_map[i_opt_] = base_matrix[i_opt_, :] == OffDiag.CONFIRMATION.value
                removal_influence_map[i_opt_] = base_matrix[i_opt_, :] == OffDiag.REMOVAL.value

        def _take_next(des_vector: np.ndarray, status_array: np.ndarray, removed_choice_options: set) \
                -> Tuple[List[np.ndarray], List[np.ndarray]]:
            # Get next choice to make: starting from left, the first choice with confirmed status
            choice_make_idx = np.where(status_array[i_choice_nodes] == confirmed_status)[0]

            # Check if there are any choices left
            if len(choice_make_idx) == 0:
                # Return final status
                return [des_vector.copy()], [status_array.copy()]

            choice_node_idx = choice_make_idx[0]
            i_choice = i_choice_nodes[choice_node_idx]

            # Get option nodes
            i_opt_nodes = _get_opt_nodes(status_array, removed_choice_options, i_choice)
            if len(i_opt_nodes) == 0:
                # If there are no option nodes, this architecture is infeasible
                return [], []

            # Branch into the different options
            branched_des_vector_arrays = []
            branched_status_arrays = []
            for opt_idx, i_opt in i_opt_nodes:
                # Apply selected option
                branched_opt_idx, branched_status, branched_removed_choice_options = \
                    _apply_option(des_vector, status_array, removed_choice_options,
                                  i_choice, i_opt, opt_idx, choice_node_idx)
                if branched_status is None:
                    continue

                # Branch into subsequent choice-options
                next_opt_idx_arrays, next_status_arrays = \
                    _take_next(branched_opt_idx, branched_status, branched_removed_choice_options)
                branched_des_vector_arrays += next_opt_idx_arrays
                branched_status_arrays += next_status_arrays

            return branched_des_vector_arrays, branched_status_arrays

        def _get_opt_nodes(status_array: np.ndarray, removed_choice_options: set, i_node) -> List[Tuple[int, int]]:
            return [(opt_idx, i_opt) for opt_idx, i_opt in enumerate(opt_idx_map[i_node])
                    if status_array[i_opt] not in opt_invalid_status and (i_node, i_opt) not in removed_choice_options]

        def _apply_option(des_vector: np.ndarray, status_array: np.ndarray, removed_choice_options: set,
                          i_choice: int, i_opt: int, opt_idx: int, choice_node_idx: int) \
                -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[set]]:
            des_vector: np.ndarray = des_vector.copy()
            status_array: np.ndarray = status_array.copy()
            removed_choice_options = set(removed_choice_options)

            # Apply the choice
            _do_apply_choice(status_array, i_choice, i_opt)
            _do_apply_choice_constraints(removed_choice_options, i_choice, i_opt)
            des_vector[choice_node_idx] = opt_idx

            # Apply any remaining choices with only one option
            for other_choice_idx in range(n_choice_nodes):
                if other_choice_idx == choice_node_idx:
                    continue

                # Check if this decision is active
                i_choice_other = i_choice_nodes[other_choice_idx]
                if status_array[i_choice_other] != confirmed_status:
                    continue

                # Check if it is a decision with zero or one options left
                i_sel_choice = _get_opt_nodes(status_array, removed_choice_options, i_choice_other)
                if len(i_sel_choice) <= 1:
                    if len(i_sel_choice) == 0:
                        # Infeasible architecture: node without options detected
                        return None, None, None

                    opt_idx_apply, i_opt = i_sel_choice[0]
                    _do_apply_choice(status_array, i_choice_other, i_opt)
                    _do_apply_choice_constraints(removed_choice_options, i_choice_other, i_opt)
                    des_vector[other_choice_idx] = opt_idx_apply

            return des_vector, status_array, removed_choice_options

        def _do_apply_choice(status_array: np.ndarray, i_choice_apply, i_opt_apply):
            # Apply confirmation and removal influences
            initial_mask = status_array == initial_status
            for influence_map, diag_value in [(confirmation_influence_map, confirmed_status),
                                              (removal_influence_map, removed_status)]:
                influence_mask = influence_map[i_opt_apply] & initial_mask
                status_array[influence_mask] = diag_value

            # Mark choice as made and option node as confirmed
            status_array[i_choice_apply] = choice_made_status
            status_array[i_opt_apply] = confirmed_status

        def _do_apply_choice_constraints(removed_choice_options: set, i_choice_apply, i_opt_apply):
            for removed_key in choice_constraint_map.get((i_choice_apply, i_opt_apply), []):
                removed_choice_options.add(removed_key)

        # Generate matrix alternatives
        base_opt_idx_array = -np.ones((n_choice_nodes,), dtype=int)
        base_status_array = np.diag(base_matrix)
        des_vector_list, status_array_list = _take_next(base_opt_idx_array, base_status_array, set())

        if len(status_array_list) == 0:
            des_vector_matrix = np.zeros((0, n_choice_nodes), dtype=int)
            status_matrix = np.zeros((0, base_matrix.shape[0]), dtype=int)
        else:
            des_vector_matrix = np.array(des_vector_list, dtype=int)
            status_matrix = np.array(status_array_list, dtype=int)
        return des_vector_matrix, status_matrix
