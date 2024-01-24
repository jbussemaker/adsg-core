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
import enum
import numpy as np
from typing import *
from cached_property import cached_property

__all__ = ['SelectionChoiceEncoder', 'X_INACTIVE_VALUE', 'Diag', 'OffDiag']

X_INACTIVE_VALUE = -1


class Diag(enum.Enum):
    INITIAL = 0
    CONFIRMED = 1
    REMOVED = 2
    INFEASIBLE_OPTION = 3  # The option would lead to an infeasible architecture
    CHOICE_MADE = 4  # The choice has been made (i.e. is not in the graph anymore)


class OffDiag(enum.Enum):
    NO_INTERACTION = 0
    CONFIRMATION = 1  # The option-node at i confirms the existence of the node at j if chosen
    REMOVAL = 2  # The option-node at i removes the node at j if chosen


class SelectionChoiceEncoder:
    """
    Base class for encoding an influence matrix into design variables, design vectors, and existence masks.
    """

    def __init__(self, influence_matrix: np.ndarray, node_ids: np.ndarray, choice_opt_map: Dict[int, List[int]],
                 choice_constraint_map: Dict[Tuple[int, int], List[Tuple[int, int]]], **_):
        """
        influence_matrix: n_nodes x n_nodes matrix with values on- and off-diagonal specified by Diag and OffDiag.
        node_ids: n_nodes vector specifying the node ids represented by each row/column
        choice_opt_map: mapping from choice node ids to option ids
        """
        self.influence_matrix = influence_matrix
        self.node_ids = node_ids
        self.choice_opt_map = choice_opt_map
        self.choice_constraint_map = choice_constraint_map

    @cached_property
    def matrix_nodes_map(self) -> Dict[int, int]:
        """Maps from node id to matrix row/column index"""
        return {node_id: i for i, node_id in enumerate(self.node_ids)}

    @cached_property
    def choice_node_ids(self) -> List[int]:
        """Node ids of choice nodes"""
        choice_opt_map = self.choice_opt_map
        return [node_id for node_id in self.node_ids if node_id in choice_opt_map]

    @property
    def n_choices(self):
        """The number of choices represented by this encoder"""
        return len(self.choice_opt_map)

    @cached_property
    def design_variable_options(self) -> List[int]:
        """Get for each design variable the number of discrete options"""
        return self._get_design_variable_options()

    @cached_property
    def n_design_space(self) -> int:
        """Number of design vectors possible to formulate from the design variables"""
        if self.n_valid <= 1:
            return self.n_valid
        return int(np.prod(self.design_variable_options, dtype=float))

    @cached_property
    def design_variable_is_forced(self) -> List[bool]:
        """Get for each design variable whether it is forced or not"""
        return self._design_variable_is_forced()

    @cached_property
    def n_valid(self) -> int:
        """Get the number of valid design vectors"""
        return self._count_n_valid()

    @cached_property
    def all_design_vectors_and_statuses(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get all valid design vectors and status arrays (use with caution!)"""
        return self.generate_all_design_vectors_and_statuses()

    #############################
    # IMPLEMENT FUNCTIONS BELOW #
    #############################

    def _get_design_variable_options(self) -> List[int]:
        """Get for each design variable the number of discrete options (at least 2)"""
        raise NotImplementedError

    def _design_variable_is_forced(self) -> List[bool]:
        """Get for each design variable whether it is forced or not"""
        raise NotImplementedError

    def _count_n_valid(self) -> int:
        """Get the number of valid design vectors"""
        raise NotImplementedError

    def generate_all_design_vectors_and_statuses(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate all valid design vectors and status arrays"""
        raise NotImplementedError

    def get_design_vector_indices(self, choice_opt_idx_map: Dict[int, int]) -> Set[int]:
        """
        Get design vectors indices conforming to a subset of choice_idx-opt_idx selections.
        Note: inactive design variables are matched by any opt_idx
        """
        raise NotImplementedError

    def get_design_vector_index(self, opt_idx: List[int], i_dv_excluded: Set[int] = None) -> Tuple[List[int], int]:
        """
        Get the design vector and associated index that most closely represents the given selection of choice-opt_idx
        values. Optionally a set of not-allowed design vector indices can be given.
        If none is found, a RuntimeError should be thrown.
        """
        raise NotImplementedError

    def get_node_existence_mask(self, node_ids: List[int], i_dv: int = None) -> np.ndarray:
        """
        Returns whether for each design vector the given node_ids exist or not.
        Optionally returns the status only for one specific design vector index.
        """
        raise NotImplementedError
