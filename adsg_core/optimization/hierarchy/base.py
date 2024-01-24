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
import numpy as np
from typing import *
from adsg_core.graph.adsg_nodes import *
from adsg_core.graph.adsg import ADSGType
from adsg_core.graph.influence_matrix import *
from cached_property import cached_property
from adsg_core.optimization.assign_enc.encoding import Encoder
from adsg_core.optimization.hierarchy.registry import SelChoiceEncoderType

__all__ = ['Diag', 'OffDiag', 'StatusScenarios', 'SelectionChoiceScenarios',
           'X_INACTIVE_VALUE', 'X_INACTIVE_IMPUTE', 'HierarchyAnalyzerBase']


class HierarchyAnalyzerBase:
    """
    Uses the influence matrix to find all possible combinations of selection-choice options, and determine which choices
    are active when.
    """

    def __init__(self, adsg: ADSGType, remove_duplicate_nodes=False):
        self._influence_matrix = InfluenceMatrix(adsg, remove_duplicate_nodes=remove_duplicate_nodes)
        self._feasibility_mask = None
        self._graph_cache = {}

    @property
    def influence_matrix(self):
        return self._influence_matrix

    @property
    def adsg(self) -> ADSGType:
        return self._influence_matrix.adsg

    @cached_property
    def selection_choice_nodes(self) -> List[SelectionChoiceNode]:
        return self._influence_matrix.selection_choice_nodes

    @property
    def n_choices(self) -> int:
        return len(self.selection_choice_nodes)

    @cached_property
    def selection_choice_option_nodes(self) -> Dict[SelectionChoiceNode, List[ADSGNode]]:
        return self._influence_matrix.selection_choice_option_nodes

    @cached_property
    def other_nodes(self) -> List[ADSGNode]:
        return self._influence_matrix.other_nodes

    @cached_property
    def _matrix_diagonal_nodes(self) -> List[ADSGNode]:
        return self._influence_matrix.matrix_diagonal_nodes

    @cached_property
    def _matrix_diagonal_nodes_idx(self) -> Dict[ADSGNode, int]:
        return self._influence_matrix.matrix_diagonal_nodes_idx

    @property
    def _influence_base_matrix(self) -> np.ndarray:
        """The base matrix of n_nodes x n_nodes specifying initial node status and node influence"""
        return self._influence_matrix.influence_matrix_no_choice_constraints

    @cached_property
    def n_opts(self) -> List[int]:
        return self._get_n_opts()

    @cached_property
    def selection_choice_is_forced(self) -> np.ndarray:
        """Returns for each selection choice whether it is forced (i.e. always applied as a one-option choice) or not;
        if true it would be unnecessary to represent it as a design variable"""
        return self._get_selection_choice_is_forced()

    @cached_property
    def n_design_space(self) -> int:
        if self.n_combinations <= 1:
            return self.n_combinations

        return int(np.prod(self.n_opts, dtype=float))

    @cached_property
    def imputation_ratio(self) -> float:
        if self.n_combinations == 0:
            return 1.
        return self.n_design_space/self.n_combinations

    @cached_property
    def information_index(self) -> float:
        return Encoder.calc_information_index(self.n_opts)

    @cached_property
    def n_combinations(self):
        return self._get_n_combinations()

    def get_nodes_existence(self, nodes: List[ADSGNode], i_comb: int = None) -> Optional[np.ndarray]:
        """Returns an n_comb x n_node matrix with flags specifying whether the associated node exists or not in each of
        the matrices"""
        return self._get_nodes_existence(nodes, i_comb=i_comb)

    def get_available_combinations_mask(self, fixed_comb_idx: Dict[int, int]) -> Optional[np.ndarray]:
        """Get a mask specifying which combinations are available (None means all available) for a given set of
        dv_idx->opt_idx fixed value pairs"""
        return self._get_available_combinations_mask(fixed_comb_idx)

    def get_graph(self, opt_idx: List[int], mask: np.ndarray = None, is_fixed: List[bool] = None, exclude: set = None) \
            -> Tuple[ADSGType, List[int], List[bool], Optional[int]]:
        """Creates a graph by making selection-choices given by the option indices. Returns the graph, a list of
        actually used option indices (imputation is applied), and a list of active choices."""

        # Initialize feasibility mask
        if self._feasibility_mask is None:
            self._feasibility_mask = np.ones((self.n_combinations,), dtype=bool)
        feasibility_mask = self._feasibility_mask

        def _get_graph() -> ADSGType:  # Same behavior as _assert_behavior!
            # Get associated option indices
            graph = self.adsg
            if len(choice_opt_idx) == 0:
                return graph.copy()

            # Check if graph has already been created
            graph_cache = self._graph_cache
            cache_key = tuple(choice_opt_idx)
            if cache_key in graph_cache:
                return graph_cache[cache_key]

            # Generate graphs
            taken_sel_opt = []
            while True:
                # Get next option-decision node
                choice_nodes = graph.get_ordered_next_choice_nodes()
                if len(choice_nodes) == 0:
                    break
                choice_node = choice_nodes[0]
                if not isinstance(choice_node, SelectionChoiceNode):
                    break

                # Get assigned option
                i_choice = i_sel_choice_nodes[choice_node]
                i_opt = choice_opt_idx[i_choice]
                if i_opt == X_INACTIVE_VALUE:
                    raise RuntimeError(f'Unexpected inactive choice: {i_choice} @ {choice_opt_idx}')
                taken_sel_opt.append((i_choice, i_opt))

                # Check if graph has already been created
                cache_key = tuple(taken_sel_opt)
                if cache_key in graph_cache:
                    graph = graph_cache[cache_key]
                    continue

                # Make choice
                option_node = sel_choice_opt_nodes[choice_node][i_opt]
                graph = graph.get_for_apply_selection_choice(choice_node, option_node)

                # Store in cache
                graph_cache[cache_key] = graph

            # Verify that indeed no selection_choices are left
            if len([node for node in graph.choice_nodes if isinstance(node, SelectionChoiceNode)]) > 0:
                raise RuntimeError(f'Selection-choice nodes left for dv: {opt_idx}')
            return graph

        sel_choice_nodes = self.selection_choice_nodes
        i_sel_choice_nodes = {node: i for i, node in enumerate(sel_choice_nodes)}
        sel_choice_opt_nodes = self.selection_choice_option_nodes

        # Get the associated or closest architecture index that is referred to by this list of option indices
        while True:
            # Find the combination index, excluding known infeasible graphs
            include_mask = feasibility_mask
            if mask is not None:
                include_mask &= mask
            i_comb, choice_opt_idx = self._get_comb_idx(opt_idx, include_mask=include_mask)
            if i_comb is None:
                raise RuntimeError('No more feasible architectures!')

            # Generate graph
            graph_instance = _get_graph()
            if graph_instance.feasible:
                break

            # Mark as infeasible and try again
            feasibility_mask[i_comb] = False

        # Return data associated to the architecture
        activeness = choice_opt_idx != X_INACTIVE_VALUE
        return graph_instance, list(choice_opt_idx), list(activeness), i_comb

    def get_opt_idx(self, opt_idx: List[int], mask: np.ndarray = None, is_fixed: List[bool] = None,
                    exclude: set = None) -> Tuple[List[int], List[bool], Optional[int]]:
        """Finds the closest combination indices that are feasible, without actually creating the graph and
        checking for feasibility"""

        if self._feasibility_mask is not None:
            include_mask = self._feasibility_mask
        else:
            include_mask = np.ones((self.n_combinations,), dtype=bool)
        if mask is not None:
            include_mask &= mask

        i_comb, sel_choice_idx = self._get_comb_idx(opt_idx, include_mask=include_mask)
        if i_comb is None:
            raise RuntimeError('No more feasible architectures!')

        activeness = sel_choice_idx != X_INACTIVE_VALUE
        return list(sel_choice_idx), list(activeness), i_comb

    def get_choice_option_indices(self) -> Optional[np.ndarray]:
        """Generate all combinations of selection-choice option indices: avoid, as this can lead to memory overflows"""
        return self._get_choice_option_indices()

    def get_existence_array(self) -> Optional[np.ndarray]:
        """Generate all combination of node existence statuses; avoid, as this can lead to memory overflow"""
        return self._get_existence_array()

    def get_encoder_type(self) -> SelChoiceEncoderType:
        raise NotImplementedError

    def _get_n_opts(self) -> List[int]:
        raise NotImplementedError

    def _get_selection_choice_is_forced(self) -> np.ndarray:
        raise NotImplementedError

    def _get_n_combinations(self) -> int:
        raise NotImplementedError

    def _get_nodes_existence(self, nodes: List[ADSGNode], i_comb: int = None) -> Optional[np.ndarray]:
        raise NotImplementedError

    def _get_available_combinations_mask(self, fixed_comb_idx: Dict[int, int]) -> Optional[np.ndarray]:
        raise NotImplementedError

    def _get_comb_idx(self, opt_idx: List[int], include_mask: np.ndarray = None) \
            -> Tuple[Optional[int], Optional[np.ndarray]]:
        raise NotImplementedError

    def _get_choice_option_indices(self) -> Optional[np.ndarray]:
        raise NotImplementedError

    def _get_existence_array(self) -> Optional[np.ndarray]:
        raise NotImplementedError

    def _assert_behavior(self):
        # Verify all combination of option indices are unique
        sel_choice_idx = self.get_choice_option_indices()
        if sel_choice_idx is None:
            return
        assert np.unique(sel_choice_idx, axis=0).shape == sel_choice_idx.shape
        sel_choice_nodes = self.selection_choice_nodes
        i_sel_choice_nodes = {node: i for i, node in enumerate(sel_choice_nodes)}
        sel_choice_opt_nodes = self.selection_choice_option_nodes

        # Loop over all combinations of decision-options
        diag_nodes = self._matrix_diagonal_nodes_idx
        existence_array = self.get_existence_array()
        if existence_array is None:
            return
        graphs = []
        for i_comb, i_sel_choice in enumerate(sel_choice_idx):
            try:
                # Loop while there are no selection-choices left to take
                graph = self.adsg
                made_choice_nodes = set()
                taken_opt_nodes = set()
                while True:  # Same behavior as _get_graph in get_graph!
                    # Get next choice node
                    choice_nodes = graph.get_ordered_next_choice_nodes()
                    if len(choice_nodes) == 0:
                        break
                    choice_node = choice_nodes[0]
                    if not isinstance(choice_node, SelectionChoiceNode):
                        break
                    made_choice_nodes.add(choice_node)

                    # Get assigned option
                    i_choice = i_sel_choice_nodes[choice_node]
                    i_opt = i_sel_choice[i_choice]
                    assert i_opt > X_INACTIVE_VALUE
                    opt_node = sel_choice_opt_nodes[choice_node][i_opt]

                    # Check if option node exists and take decision
                    opt_nodes = graph.get_option_nodes(choice_node)
                    assert opt_node in opt_nodes
                    taken_opt_nodes.add(opt_node)
                    graph = graph.get_for_apply_selection_choice(choice_node, opt_node)

                    for auto_taken_choice_node, _ in graph.get_taken_single_selection_choices():
                        made_choice_nodes.add(auto_taken_choice_node)

                # Verify there are no more selection-choices
                assert len([node for node in graph.choice_nodes if isinstance(node, SelectionChoiceNode)]) == 0

                # Check all node statuses
                node_status = existence_array[i_comb, :]
                graph_nodes = graph.graph.nodes
                for node, i_nd in diag_nodes.items():
                    status = node_status[i_nd]
                    try:
                        if status == Diag.CONFIRMED.value:
                            assert node in graph_nodes
                        elif status == Diag.CHOICE_MADE.value:
                            assert node in made_choice_nodes
                        else:
                            assert node not in graph_nodes
                    except AssertionError:
                        print(f'Asserting existence of {node!r}: {status} '
                              f'(in graph {node in graph_nodes}, choice made {node in made_choice_nodes})')
                        raise

            except AssertionError:
                print(f'Assertion failed for (idx {i_comb}): {list(i_sel_choice)!r}')
                raise

            graphs.append(graph)
            is_feasible = graph.feasible

            graph_, i_sel_choice_, is_active, i_comb_ = self.get_graph(i_sel_choice)
            assert np.all(i_sel_choice_ == i_sel_choice) == is_feasible
            assert np.all(is_active == (np.array(i_sel_choice_) != X_INACTIVE_VALUE))
            assert (i_comb_ == i_comb) == is_feasible
            if is_feasible:
                assert set(graph_.graph.nodes) == set(graph_nodes)

        self._assert_existence()
        return graphs

    def _assert_existence(self):
        existence_array_all = self.get_existence_array()
        if existence_array_all is None:
            return
        existence_array = existence_array_all == Diag.CONFIRMED.value
        if existence_array.shape[0] == 0:
            return

        for node, i in self._matrix_diagonal_nodes_idx.items():
            existence = self.get_nodes_existence([node])
            if existence is None:
                continue
            assert existence.shape == (existence_array.shape[0], 1)
            assert np.all(existence[:, 0] == existence_array[:, i])

            for i_comb in np.random.randint(0, existence_array.shape[0], 10):
                assert np.all(self.get_nodes_existence([node], i_comb=i_comb) == existence_array[i_comb, [i]])
