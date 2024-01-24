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
from adsg_core.graph.choice_constraints import *
from adsg_core.optimization.hierarchy.base import *
from adsg_core.optimization.hierarchy.registry import SelChoiceEncoderType

from adsg_core.optimization.sel_choice_enc.encoder import SelectionChoiceEncoder
from adsg_core.optimization.sel_choice_enc.brute_force import BruteForceSelectionChoiceEncoder
from adsg_core.optimization.sel_choice_enc.util import assert_behavior

__all__ = ['SelChoiceEncHierarchyAnalyzer']


class SelChoiceEncHierarchyAnalyzer(HierarchyAnalyzerBase):

    def __init__(self, adsg: ADSGType):
        super().__init__(adsg, remove_duplicate_nodes=True)

        node_idx_map = self.influence_matrix.matrix_diagonal_nodes_idx
        self._nodes = sorted(node_idx_map.keys(), key=lambda n: node_idx_map[n])
        self._node_id_map = {node: node_id for node_id, node in enumerate(self._nodes)}
        self._encoder = self._create_sel_choice_encoder()

    def get_encoder_type(self) -> SelChoiceEncoderType:
        return SelChoiceEncoderType.COMPLETE

    def _create_sel_choice_encoder(self) -> SelectionChoiceEncoder:
        influence_matrix = self.influence_matrix.influence_matrix_no_choice_constraints
        choice_constraint_map = self._get_choice_constraint_map()

        node_id_map = self._node_id_map
        node_ids = np.array([node_id_map[node] for node in self.influence_matrix.matrix_diagonal_nodes])

        choice_opt_map = {node_id_map[choice_node]: [node_id_map[opt_node] for opt_node in opt_nodes]
                          for choice_node, opt_nodes in self.influence_matrix.selection_choice_option_nodes.items()}

        return self._instantiate_encoder(influence_matrix, node_ids, choice_opt_map, choice_constraint_map)

    def _get_choice_constraint_map(self) -> Dict[Tuple[int, int], List[Tuple[int, int]]]:
        node_id_map = self._node_id_map
        choice_constraint_map = {}
        for choice_constraint in self.adsg.get_choice_constraints():
            for i_choice, choice_node in enumerate(choice_constraint.nodes):
                if choice_node not in node_id_map:
                    break
                choice_id = node_id_map[choice_node]
                if not isinstance(choice_node, SelectionChoiceNode):
                    break

                for i_opt, opt_node in enumerate(choice_constraint.options[i_choice]):
                    if opt_node not in node_id_map:
                        continue
                    opt_node_id = node_id_map[opt_node]

                    removed_map = []
                    for other_choice_node, removed_opt_nodes in \
                            get_constraint_removed_options(choice_constraint, i_choice, i_opt):

                        if other_choice_node not in node_id_map:
                            continue
                        other_choice_id = node_id_map[other_choice_node]

                        for removed_opt_node in removed_opt_nodes:
                            if removed_opt_node not in node_id_map:
                                continue
                            removed_opt_node_id = node_id_map[removed_opt_node]
                            removed_map.append((other_choice_id, removed_opt_node_id))

                    choice_constraint_map[choice_id, opt_node_id] = removed_map
        return choice_constraint_map

    def _instantiate_encoder(self, influence_matrix, node_ids, choice_opt_map, choice_constraint_map) \
            -> SelectionChoiceEncoder:
        return BruteForceSelectionChoiceEncoder(influence_matrix, node_ids, choice_opt_map, choice_constraint_map)

    def _get_n_opts(self) -> List[int]:
        return self._encoder.design_variable_options

    def _get_selection_choice_is_forced(self) -> np.ndarray:
        return np.array(self._encoder.design_variable_is_forced, dtype=bool)

    def _get_n_combinations(self) -> int:
        return self._encoder.n_valid

    def _get_nodes_existence(self, nodes: List[ADSGNode], i_comb: int = None) -> np.ndarray:
        node_ids = self._get_node_ids(nodes)
        return self._encoder.get_node_existence_mask(node_ids, i_dv=i_comb)

    def _get_available_combinations_mask(self, fixed_comb_idx: Dict[int, int]) -> Optional[np.ndarray]:
        if len(fixed_comb_idx) == 0:
            return

        dv_idx = self._encoder.get_design_vector_indices(fixed_comb_idx)
        dv_mask = np.zeros((self.n_combinations,), dtype=bool)
        dv_mask[list(dv_idx)] = True
        return dv_mask

    def _get_comb_idx(self, opt_idx: List[int], include_mask: np.ndarray = None) \
            -> Tuple[Optional[int], Optional[np.ndarray]]:

        i_dv_excluded = None
        if include_mask is not None:
            i_dv_excluded = set(np.where(~include_mask)[0])

        try:
            dv_matched, i_matched = self._encoder.get_design_vector_index(opt_idx, i_dv_excluded=i_dv_excluded)
            return i_matched, np.array(dv_matched)
        except RuntimeError:
            return None, None

    def _get_choice_option_indices(self) -> np.ndarray:
        return self._encoder.all_design_vectors_and_statuses[0]

    def _get_existence_array(self) -> np.ndarray:
        return self._encoder.all_design_vectors_and_statuses[1]

    def _get_node_ids(self, nodes: List[ADSGNode]) -> List[int]:
        node_id_map = self._node_id_map
        return [node_id_map[node] for node in nodes]

    def _assert_behavior(self):
        graph = super()._assert_behavior()
        assert_behavior(self._encoder)
        return graph
