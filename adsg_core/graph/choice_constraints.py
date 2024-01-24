"""
MIT License

Copyright: (c) 2024, Deutsches Zentrum fuer Luft- und Raumfahrt e.V.
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
import itertools
import numpy as np
from typing import *
from enum import Enum
from dataclasses import dataclass
from adsg_core.graph.adsg_nodes import *

__all__ = ['CDVNode', 'ChoiceConstraint', 'ChoiceConstraintType', 'get_constraint_removed_options',
           'get_constraint_pre_removed_options', 'get_valid_idx_combinations', 'count_n_combinations_max']

CDVNode = Union[ChoiceNode, DesignVariableNode]


class ChoiceConstraintType(Enum):
    LINKED = 1  # To make all choices have the same option index             --> AA, BB, CC
    PERMUTATION = 2  # To make the choices be permutations of option indices --> AB, BA, AC, CA, BC, CB
    UNORDERED = 3  # To have all option index combinations without ordering  --> AA, AB, AC, BB, BC, CC
    UNORDERED_NOREPL = 4  # Same but also without replacement                --> AB, AC, BC


@dataclass(frozen=True)
class ChoiceConstraint:
    type: ChoiceConstraintType
    nodes: List[CDVNode]
    options: List[list] = None

    def __hash__(self):
        return id(self)


def get_constraint_removed_options(choice_constraint: ChoiceConstraint, i_taken_choice: int, i_chosen_option: int) \
        -> List[Tuple[ChoiceNode, list]]:
    """
    Implements the option-removal logic, regardless of which choice is taken first.

    Consider two choices (D1, D2) with three options each (A, B, C).
    For a LINKED choice constraint: remove all options with different indices
    D1 = A --> remove D2 BC --> D2 = A; D1+D2 = AA
    D2 = B --> remove D1 AC --> D2 = B; D1+D2 = BB
    all D1+D2 options: AA, BB, CC

    For a PERMUTATION constraint: remove all options with the same index
    D1 = A --> remove D2 A --> D2 = B or C; D1+D2 = AB, AC
    D1 = B --> remove D2 B --> D2 = A or C; D1+D2 = BA, BC
    all D1+D2 options: AB, BA, AC, CA, BC, CB

    For an UNORDERED constraint: remove all subsequent options with lower indices (and vice-versa)
    D1 = B --> remove D2 A --> D2 = B or C; D1+D2 = BB, BC
    D2 = B --> remove D1 C --> D1 = A or B; D1+D2 = AB, BB
    all D1+D2 options: AA, AB, AC, BB, BC, CC

    For an UNORDERED_NOREPL constraint: remove all subsequent options with lower or same indices (and vice-versa)
    D1 = B --> remove D2 A,B --> D2 = C; D1+D2 = BC
    D2 = B --> remove D1 B,C --> D1 = A; D1+D2 = AB
    all D1+D2 options: AB, AC, BC --> observe that D1=C or D2=A lead to infeasible architectures
                                      (prevented using `get_constraint_pre_removed_options`)
    """
    if choice_constraint.options is None:
        return []

    removed_choice_opts = []
    for i, choice_node in enumerate(choice_constraint.nodes):
        if i == i_taken_choice:
            continue
        options = choice_constraint.options[i]
        enough_options = len(options)-1 >= i_chosen_option

        removed_opts = []
        if choice_constraint.type == ChoiceConstraintType.LINKED:
            # For linked constraints, we simply remove all options with different indices from the other choices
            # If a choice does not have as many options, we remove all but the last
            if enough_options:
                removed_opts = [opt for j, opt in enumerate(options) if j != i_chosen_option]
            else:
                removed_opts = options[:-1]

        elif choice_constraint.type == ChoiceConstraintType.PERMUTATION:
            # For permutation constraints, we remove all options with the same index from the other choices
            if enough_options:
                removed_opts = [options[i_chosen_option]]

        elif choice_constraint.type == ChoiceConstraintType.UNORDERED:
            # For unordered combination constraints, the goal is that subsequent choices cannot take options with
            # lower indices (and therefore preceding choices cannot have options with higher indices)
            if i < i_taken_choice:
                removed_opts = options[i_chosen_option+1:]  # Preceding cannot have higher-index options
            else:
                removed_opts = options[:i_chosen_option]  # Subsequent cannot have lower-index options

        elif choice_constraint.type == ChoiceConstraintType.UNORDERED_NOREPL:
            # Unordered non-replacing combination constraints behave almost the same as unordered, however here also
            # taking the same index (which would be replacement) is not allowed
            if i < i_taken_choice:
                removed_opts = options[i_chosen_option:]  # Preceding cannot have higher or the same index
            else:
                removed_opts = options[:i_chosen_option+1]  # Subsequent cannot have lower or the same index

        else:
            raise RuntimeError(f'Unsupported constraint type: {choice_constraint.type}')
        if len(removed_opts) > 0:
            removed_choice_opts.append((choice_node, removed_opts))
    return removed_choice_opts


def get_constraint_pre_removed_options(choice_constraint: ChoiceConstraint, permanent_nodes: Set[ADSGNode]) \
        -> List[Tuple[ChoiceNode, list]]:
    """
    Get options that can never be selected in order to prevent constraints that are impossible to satisfy.
    """
    if choice_constraint.options is None:
        return []

    # For permutations, if there are more choices that the max nr of options, there is no way to make a permutation
    if choice_constraint.type == ChoiceConstraintType.PERMUTATION:
        n_dec = len(choice_constraint.nodes)
        n_opt_max = max([len(options) for options in choice_constraint.options])

        # Remove all options if there are more choices than the max nr of options
        if n_dec > n_opt_max:
            return [(dec_node, choice_constraint.options[i]) for i, dec_node in enumerate(choice_constraint.nodes)]

    # For unordered non-replacing choices, remove options to ensure infeasible combinations never get selected
    # We can only apply this if all choices are permanent, otherwise we can't be sure
    if choice_constraint.type == ChoiceConstraintType.UNORDERED_NOREPL \
            and all(node in permanent_nodes for node in choice_constraint.nodes):
        pre_removed_opts = []
        n_dec = len(choice_constraint.nodes)
        for i_dec, dec_node in enumerate(choice_constraint.nodes):
            i_start = i_dec
            n_dec_after = n_dec-(i_dec+1)
            i_end = len(choice_constraint.options[i_dec]) - n_dec_after
            removed_options = [opt for i_opt, opt in enumerate(choice_constraint.options[i_dec])
                               if i_opt < i_start or i_opt >= i_end]
            pre_removed_opts.append((dec_node, removed_options))

        return pre_removed_opts

    return []


def get_valid_idx_combinations(idx_comb: np.ndarray, constraint_type: ChoiceConstraintType,
                               is_all_permanent=False) -> np.ndarray:
    """
    Given a matrix with n_comb x n_choices index combinations, return which combinations are valid.
    Inactive values (-1) are ignored.

    For example:
    [[0, 0],
     [0, 1],
     [1, 0],
     [1, 1]]

    for LINKED type, returns: [0, 3]
    for PERMUTATION type, returns: [1, 2]
    for UNORDERED type, returns: [0, 1, 3]
    for UNORDERED_NOREPL type: [1]

    The is_all_permanent defines whether all involved choices are permanent.
    """

    # If there is only one column, there is no constraint to be applied so all are valid
    if idx_comb.shape[1] <= 1:
        return np.arange(idx_comb.shape[0])

    valid_mask = np.ones((idx_comb.shape[0],), dtype=bool)

    def _iter_rows(is_valid):
        for i_row, row in enumerate(idx_comb):
            active_row = row[row != -1]
            if len(active_row) > 1:
                valid_mask[i_row] = is_valid(active_row)

    if constraint_type == ChoiceConstraintType.LINKED:
        _iter_rows(lambda v: np.all(v[1:] == v[0]))

    elif constraint_type == ChoiceConstraintType.PERMUTATION:
        for i, j in itertools.combinations(range(idx_comb.shape[1]), 2):
            valid_mask_i = (idx_comb[:, i] != idx_comb[:, j]) | (idx_comb[:, i] == -1) | (idx_comb[:, j] == -1)
            valid_mask &= valid_mask_i

    elif constraint_type == ChoiceConstraintType.UNORDERED or \
            (is_all_permanent and constraint_type == ChoiceConstraintType.UNORDERED_NOREPL):

        def _check_gte(row):
            for i_value in range(1, len(row)):
                if row[i_value] < row[i_value-1]:
                    return False
            return True

        _iter_rows(_check_gte)

    elif not is_all_permanent and constraint_type == ChoiceConstraintType.UNORDERED_NOREPL:

        def _check_gt(row):
            for i_value in range(1, len(row)):
                if row[i_value] <= row[i_value-1]:
                    return False
            return True

        _iter_rows(_check_gt)

    else:
        raise ValueError(f'Unknown choice constraint: {constraint_type}')

    return np.where(valid_mask)[0]


def count_n_combinations_max(choice_constraint: ChoiceConstraint, is_all_permanent=False) -> int:
    # Not implemented for continuous or connection choices
    if choice_constraint.options is None or not isinstance(choice_constraint.nodes[0], SelectionChoiceNode):
        return 0

    # Fixes counting for unordered non-replacing in case it is not known which nodes actually exist at a given time
    if choice_constraint.type == ChoiceConstraintType.UNORDERED_NOREPL and not is_all_permanent:
        is_all_permanent = True

    all_idx_comb = np.array(list(itertools.product(
        *[range(len(opt_nodes)) for opt_nodes in choice_constraint.options])))
    return len(get_valid_idx_combinations(all_idx_comb, choice_constraint.type, is_all_permanent=is_all_permanent))
