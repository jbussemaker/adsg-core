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
from adsg_core.graph.choices import NoOptionError
from adsg_core.optimization.hierarchy.base import *
from adsg_core.optimization.hierarchy.registry import SelChoiceEncoderType

__all__ = ['FastHierarchyAnalyzer']


class FastHierarchyAnalyzer(HierarchyAnalyzerBase):
    """
    Hierarchy analyzer that ignores most of the choice interactions and thereby cannot provide some information,
    however should work quickly for any design space size.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._imputation_cache = {}

    def get_encoder_type(self) -> SelChoiceEncoderType:
        return SelChoiceEncoderType.FAST

    def _get_n_opts(self) -> List[int]:
        sel_choice_opt_nodes = self.selection_choice_option_nodes
        return [len(sel_choice_opt_nodes[node]) for node in self.selection_choice_nodes]

    def _get_selection_choice_is_forced(self) -> np.ndarray:
        is_forced = np.array([False for _ in range(len(self.selection_choice_option_nodes))], dtype=bool)

        # For linked choice constraints, set all but one to forced
        i_choice_nodes = {node: i for i, node in enumerate(self.selection_choice_nodes)}
        for choice_constraint in self.adsg.get_choice_constraints():
            if choice_constraint.type == ChoiceConstraintType.LINKED:
                i_choices = sorted([i_choice_nodes[node] for node in choice_constraint.nodes if node in i_choice_nodes])
                if len(i_choices) > 1:
                    for i_dep in i_choices[1:]:
                        is_forced[i_dep] = True

        return is_forced

    def _get_n_combinations(self) -> int:
        n_opts = list(self.n_opts)

        # Apply choice constraints
        permanent_nodes = self.influence_matrix.permanent_nodes_incl_choice_nodes
        i_choice_nodes = {node: i for i, node in enumerate(self.selection_choice_nodes)}
        for choice_constraint in self.adsg.get_choice_constraints():
            i_choices = sorted([i_choice_nodes[node] for node in choice_constraint.nodes if node in i_choice_nodes])
            if len(i_choices) <= 1:
                continue

            is_all_permanent = all([node in permanent_nodes for node in choice_constraint.nodes])
            n_opts[i_choices[0]] = count_n_combinations_max(choice_constraint, is_all_permanent=is_all_permanent)
            for i_other in i_choices[1:]:
                n_opts[i_other] = 1

        # Cartesian product as upper bound
        return int(np.prod(n_opts, dtype=float))

    def get_graph(self, opt_idx: List[int], mask: np.ndarray = None, is_fixed: List[bool] = None, exclude: set = None) \
            -> Tuple[ADSGType, List[int], List[bool], Optional[int]]:

        if exclude is None:
            exclude = set()
        if is_fixed is None:
            is_fixed = [False for _ in range(len(opt_idx))]

        graph_cache = self._graph_cache
        sel_choice_nodes = self.selection_choice_nodes
        i_sel_choice_nodes = {node: i for i, node in enumerate(sel_choice_nodes)}
        sel_choice_opt_nodes = self.selection_choice_option_nodes

        def _get_graph() -> Tuple[Tuple[int, ...], ADSGType]:
            graph = self.adsg
            if len(opt_idx_try) == 0:
                return graph.copy()

            # Generate graph
            taken_sel_opt = [X_INACTIVE_VALUE for _ in range(len(opt_idx_try))]
            while True:
                # Get next selection-choice node
                choice_nodes = graph.get_ordered_next_choice_nodes()
                if len(choice_nodes) == 0:
                    break
                choice_node = choice_nodes[0]
                if not isinstance(choice_node, SelectionChoiceNode):
                    break

                # Get assigned option
                i_choice = i_sel_choice_nodes[choice_node]
                i_opt = opt_idx_try[i_choice]
                if i_opt == X_INACTIVE_VALUE:
                    raise RuntimeError(f'Unexpected inactive choice: {i_choice} @ {choice_opt_idx}')
                taken_sel_opt[i_choice] = i_opt

                # Check if the graph has already been created
                cache_key = tuple(taken_sel_opt)
                if cache_key in graph_cache:
                    graph = graph_cache[cache_key]
                    continue

                # Make choice
                option_node = sel_choice_opt_nodes[choice_node][i_opt]
                graph_cache[cache_key] = graph = graph.get_for_apply_selection_choice(choice_node, option_node)

            # Verify that indeed no selection_choices are left
            if len([node for node in graph.choice_nodes if isinstance(node, SelectionChoiceNode)]) > 0:
                raise RuntimeError(f'Selection-choice nodes left for dv: {opt_idx}')
            return tuple(taken_sel_opt), graph

        # Iterate over current and neighboring design vectors
        tried = []
        opt_idx_imp = opt_idx
        graph_instance = None
        for opt_idx_try in self._iter_neighborhood(opt_idx, is_fixed):
            tried.append(opt_idx_try)
            if opt_idx_try in exclude:
                continue

            # Check cache
            if opt_idx_try in self._imputation_cache:
                outputs = self._imputation_cache[opt_idx_try]
                if tuple(outputs[1]) not in exclude:
                    return outputs

            # Try to get graph
            try:
                opt_idx_imp, graph_instance = _get_graph()

            except NoOptionError:  # Can happen if due to some choice constraint options are removed
                exclude.add(opt_idx_try)
                continue

            tried.append(opt_idx_imp)
            if opt_idx_imp in exclude:
                exclude.add(opt_idx_try)
                continue

            if graph_instance.feasible:
                break

            # Mark as infeasible
            exclude.add(opt_idx_try)
            exclude.add(opt_idx_imp)

        if not graph_instance.feasible:
            raise RuntimeError('No more feasible graphs!')

        # Update imputation cache
        choice_opt_idx = list(opt_idx_imp)
        activeness = list(np.array(choice_opt_idx) != X_INACTIVE_VALUE)
        outputs = (graph_instance, choice_opt_idx, activeness, None)
        for key in tried:
            self._imputation_cache[key] = outputs
        return outputs

    def _iter_neighborhood(self, opt_idx: List[int], is_fixed: List[bool]) -> Generator[Tuple[int, ...], None, None]:
        n_opts = self.n_opts

        def _iter_values(i_dv):
            i_current = opt_idx[i_dv]
            yield i_current
            if is_fixed[i_dv]:
                return

            for dist in range(1, n_opts[i_dv]):
                yielded = False
                pos_dir = i_current+dist
                if pos_dir < n_opts[i_dv]:
                    yield pos_dir
                    yielded = True

                neg_dir = i_current-dist
                if neg_dir >= 0:
                    yield neg_dir
                    yielded = True

                if not yielded:
                    break

        def _iter_neighbor_next(i_iter=0, prev_values=None):
            if prev_values is None:
                prev_values = []
            for value in _iter_values(i_dv=i_iter):
                next_values = prev_values+[value]
                if i_iter < len(opt_idx)-1:
                    yield from _iter_neighbor_next(i_iter=i_iter+1, prev_values=next_values)
                else:
                    yield tuple(next_values)

        yield from _iter_neighbor_next()

    def get_opt_idx(self, opt_idx: List[int], mask: np.ndarray = None, is_fixed: List[bool] = None,
                    exclude: set = None) -> Tuple[List[int], List[bool], Optional[int]]:
        _, choice_opt_idx, activeness, i_comb = self.get_graph(opt_idx, mask=mask, is_fixed=is_fixed)
        return choice_opt_idx, activeness, i_comb

    def _get_comb_idx(self, opt_idx: List[int], include_mask: np.ndarray = None) \
            -> Tuple[Optional[int], Optional[np.ndarray]]:
        raise RuntimeError

    def _get_nodes_existence(self, nodes: List[ADSGNode], i_comb: int = None) -> Optional[np.ndarray]:
        pass

    def _get_available_combinations_mask(self, fixed_comb_idx: Dict[int, int]) -> Optional[np.ndarray]:
        pass

    def _get_choice_option_indices(self) -> Optional[np.ndarray]:
        pass

    def _get_existence_array(self) -> Optional[np.ndarray]:
        pass
