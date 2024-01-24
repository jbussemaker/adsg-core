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
from dataclasses import dataclass
from collections import defaultdict
from adsg_core.graph.adsg_nodes import *
from adsg_core.graph.choice_constraints import *
from cached_property import cached_property
from adsg_core.optimization.hierarchy.base import *
from adsg_core.optimization.hierarchy.registry import SelChoiceEncoderType

__all__ = ['HierarchyAnalyzer', 'ApplyIterSpec']


@dataclass
class ApplyIterSpec:
    scenario: SelectionChoiceScenarios
    i_scenario: int
    i_usi: int  # unique scenario index
    i_comb: int
    n_every: int
    offsets: List[Tuple[int, int]]  # offset, n_apply
    n_total: int

    def __iter__(self):
        for i_start in range(0, self.n_total, self.n_every):
            for offset, n_apply in self.offsets:
                yield from range(i_start+offset, i_start+offset+n_apply)

    @cached_property
    def i_set(self):
        return set(iter(self))

    def __contains__(self, idx):
        if idx < 0 or idx >= self.n_total:
            return False
        idx_mod = idx % self.n_every
        for offset, n_apply in self.offsets:
            if offset <= idx_mod < offset+n_apply:
                return True
        return False


class HierarchyAnalyzer(HierarchyAnalyzerBase):
    """
    Hierarchy analyzer that analyzes the design space in such a way that the complete behavior is known:
    - Number of possible architectures
    - Possible design vectors --> imputation
    - Node existence for each architecture
    """

    def get_encoder_type(self) -> SelChoiceEncoderType:
        return SelChoiceEncoderType.COMPLETE

    @cached_property
    def __existence_array_base_brute_force(self):
        return self._create_existence_array_brute_force()

    @cached_property
    def _choice_option_indices_brute_force(self) -> np.ndarray:
        return self.__existence_array_base_brute_force[0]

    @cached_property
    def _existence_array_brute_force(self) -> np.ndarray:
        """All possible architectures specified in an n_comb x n_nodes array, only stores node status"""
        return self.__existence_array_base_brute_force[1]

    @cached_property
    def _choice_option_indices(self) -> np.ndarray:
        return self._generate_all_opt_idx_combinations()

    def _get_choice_option_indices(self) -> np.ndarray:
        """Generate all combinations of selection-choice option indices: avoid, as this can lead to memory overflows"""
        return self._choice_option_indices

    @cached_property
    def _existence_array(self) -> np.ndarray:
        """All possible architectures specified in an n_comb x n_nodes array, only stores node status"""
        return self._generate_existence_array()

    def _get_existence_array(self) -> np.ndarray:
        """Generate all combination of node existence statuses; avoid, as this can lead to memory overflow"""
        return self._existence_array

    def _get_n_combinations(self) -> int:
        indep_scenarios = self._indep_sel_choice_scenarios
        dep_scenarios = self._dep_sel_choice_scenarios

        # Get for each independent scenario combination the amount of dependent combinations
        indep_n_comb = {i: np.ones((indep_scenario.n_combinations_unique[0],))
                        for i, indep_scenario in indep_scenarios.items()}

        for (i_indep, i_comb), dep_input_activation in self._reduced_sel_choice_activation_map.items():
            for i_dep, i_input in dep_input_activation:
                n_dep_comb = dep_scenarios[i_dep].n_combinations[i_input]
                indep_n_comb[i_indep][i_comb] *= n_dep_comb

        # Get Cartesian product of independent scenarios
        return int(np.prod([np.sum(n_combs) for n_combs in indep_n_comb.values()]))

    @cached_property
    def _scenario_iter_spec(self):
        """Iterate over the design vector indices that different scenario combinations are applied at"""
        indep_scenarios = self._indep_sel_choice_scenarios
        dep_scenarios = self._dep_sel_choice_scenarios

        # Get amount of dependent combinations for each independent combination
        indep_n_comb = {i: [[(1, i, 0)] for _ in range(indep_scenario.n_combinations_unique[0])]
                        for i, indep_scenario in indep_scenarios.items()}

        for (i_indep, i_comb), dep_input_activation in self._reduced_sel_choice_activation_map.items():
            for i_dep, i_input in dep_input_activation:
                dep_scenario = dep_scenarios[i_dep]
                indep_n_comb[i_indep][i_comb].append((
                    dep_scenario.n_combinations[i_input],
                    i_dep,
                    dep_scenario.unique_scenario_idx[i_input],
                ))

        # Determine total nr of combinations for each independent scenario
        n_indep_comb = np.array([np.sum([np.prod([c[0] for c in comb]) for comb in combs])
                                 for combs in indep_n_comb.values()])
        n_repeat_every = np.cumprod(n_indep_comb[::-1])[::-1]
        n_repeated = np.cumprod([1]+list(n_indep_comb[::-1]))[:-1][::-1]
        n_total = int(np.prod(n_indep_comb))

        iter_specs = []
        for i, (i_indep, combs) in enumerate(indep_n_comb.items()):
            n_every = n_repeat_every[i]
            n_rep = n_repeated[i]

            # Loop through dependent scenarios for each independent scenario selected combination
            i_comb_start = 0
            dep_offsets: Dict[Any, list] = defaultdict(list)
            for i_comb, comb in enumerate(combs):

                # Determine nr of combinations for each activated dependent scenario
                n_dep_comb = np.array([c[0] for c in comb])
                n_dep_repeat_every = np.cumprod(n_dep_comb[::-1])[::-1]
                n_dep_repeated = np.cumprod([1]+list(n_dep_comb[::-1]))[:-1][::-1]
                n_comb = int(n_dep_repeated[0])
                if n_comb == 0:
                    continue

                n_apply = n_comb*n_rep
                iter_specs.append(ApplyIterSpec(
                    scenario=indep_scenarios[i_indep], i_scenario=i_indep,
                    i_usi=0, i_comb=i_comb, n_every=n_every,
                    offsets=[(i_comb_start, n_apply)],  # offset, n_apply
                    n_total=n_total,
                ))

                # Determine iteration offsets for each activated dependent scenario
                for j, (n, i_dep, usi) in enumerate(comb):
                    if j == 0:  # Independent scenario activation
                        continue

                    i_dep_comb_start = 0
                    n_dep_apply = n_dep_repeated[j]*n_rep
                    if n_dep_apply == 0:
                        continue
                    for i_comb_dep in range(n):
                        key = (i_dep, usi, i_comb_dep)
                        start_offset = i_comb_start+i_dep_comb_start
                        for i_rep_offset in range(0, n_comb, n_dep_repeat_every[j]):
                            rep_offset = i_rep_offset*n_rep
                            dep_offsets[key].append((start_offset+rep_offset, n_dep_apply))
                        i_dep_comb_start += n_dep_apply

                i_comb_start += n_apply

            for (i_dep, usi, i_comb_dep), offsets in dep_offsets.items():
                iter_specs.append(ApplyIterSpec(
                    scenario=dep_scenarios[i_dep], i_scenario=i_dep,
                    i_usi=usi, i_comb=i_comb_dep, n_every=n_every,
                    offsets=offsets, n_total=n_total,
                ))
        return iter_specs

    @cached_property
    def _iter_spec_confirming_node_idx(self):
        confirmed_status = Diag.CONFIRMED.value

        its_confirms = []
        for its in self._scenario_iter_spec:
            node_idx_confirmed = set()
            for i_choice, statuses in enumerate(its.scenario.applied_status_combinations[its.i_usi]):
                for i_confirmed in np.where(statuses[its.i_comb, 1:] == confirmed_status)[0]:
                    node_idx_confirmed.add(its.scenario.opt_node_idx[i_choice][i_confirmed])
            its_confirms.append(node_idx_confirmed)
        return its_confirms

    def _get_n_opts(self) -> List[int]:
        n_opts = np.zeros((self.n_choices,), dtype=int)
        for scenario in self._reduced_selection_choice_scenarios:
            n_opts_scenario = np.max([np.max(combs, axis=0) if combs.shape[0] > 0 else [0]
                                      for combs in scenario.opt_idx_combinations], axis=0)+1
            n_opts[scenario.choice_idx] = n_opts_scenario
        return list(n_opts)

    def _get_nodes_existence(self, nodes: List[ADSGNode], i_comb: int = None) -> np.ndarray:
        """Returns an n_comb x n_node matrix with flags specifying whether the associated node exists or not in each of
        the matrices"""
        iter_specs = self._scenario_iter_spec
        other_nodes = set(self.other_nodes)
        node_idx_map = self._matrix_diagonal_nodes_idx
        influence_matrix = self._influence_base_matrix
        its_confirms = self._iter_spec_confirming_node_idx

        confirmed_status = Diag.CONFIRMED.value
        exists = np.zeros((self.n_combinations if i_comb is None else 1, len(nodes)), dtype=bool)
        for i, node in enumerate(nodes):
            node_idx = node_idx_map[node]

            # Check if the node is a choice node
            if isinstance(node, SelectionChoiceNode):
                exists[:, i] = False
                continue

            # Check if the node is permanent
            if influence_matrix[node_idx, node_idx] == confirmed_status:
                exists[:, i] = True
                continue

            # Find triggering selection choice nodes for the node (if it is not in the selection choice nodes)
            if node in other_nodes:
                confirmed_by = set(np.where(influence_matrix[:, node_idx] == OffDiag.CONFIRMATION.value)[0])
                removed_by = set(np.where(influence_matrix[:, node_idx] == OffDiag.REMOVAL.value)[0])
            else:
                confirmed_by = {node_idx}
                removed_by = set()

            if len(confirmed_by) == 0:
                continue

            # Look for scenarios where the node is confirmed
            for node_idx_confirms in confirmed_by:
                for i_its, its in enumerate(iter_specs):
                    if node_idx_confirms in its_confirms[i_its]:

                        if i_comb is not None:
                            if i_comb in its:
                                exists[0, i] = True
                        else:
                            exists[list(its.i_set), i] = True

            # Look for scenarios where the node is removed
            for node_idx_removes in removed_by:
                for i_its, its in enumerate(iter_specs):
                    if node_idx_removes in its_confirms[i_its]:

                        if i_comb is not None:
                            if i_comb in its:
                                exists[0, i] = False
                        else:
                            exists[list(its.i_set), i] = False

        return exists[0, :] if i_comb is not None else exists

    def _get_available_combinations_mask(self, fixed_comb_idx: Dict[int, int]) -> Optional[np.ndarray]:
        """Get a mask specifying which combinations are available (None means all available) for a given set of
        dv_idx->opt_idx fixed value pairs"""
        iter_specs = self._scenario_iter_spec

        fixed_comb_set = None
        for i_choice, i_opt in fixed_comb_idx.items():

            fixed_comb_set_i = set()
            for its in iter_specs:
                i_ch = np.where(its.scenario.choice_idx == i_choice)[0]
                if len(i_ch) == 0:
                    continue

                # Check if this iter spec represents an opt_idx combination with the fixed value
                if its.scenario.opt_idx_combinations[its.i_usi][its.i_comb, i_ch[0]] == i_opt:
                    fixed_comb_set_i |= its.i_set

            fixed_comb_set = fixed_comb_set_i if fixed_comb_set is None else (fixed_comb_set & fixed_comb_set_i)

        if fixed_comb_set is not None:
            fixed_comb_mask = np.zeros((self.n_combinations,), dtype=bool)
            fixed_comb_mask[list(fixed_comb_set)] = True
            return fixed_comb_mask

    def _get_selection_choice_is_forced(self):
        """Returns for each selection choice whether it is forced (i.e. always applied as a one-option choice) or not;
        if true it would be unnecessary to represent it as a design variable"""
        # Check special case where there are no feasible architectures
        if self.n_combinations == 0:
            return np.ones((0,), dtype=bool)

        def _is_forced(choice_opt_idx):
            is_forced_ = np.zeros((choice_opt_idx.shape[1],), dtype=bool)
            imputed_choice_opt_idx = choice_opt_idx.copy()
            imputed_choice_opt_idx[imputed_choice_opt_idx == X_INACTIVE_VALUE] = X_INACTIVE_IMPUTE
            for i_dv in range(1, imputed_choice_opt_idx.shape[1]):
                # Get unique values
                values, idx = np.unique(imputed_choice_opt_idx[:, i_dv], return_inverse=True)

                # Check if any of the left-side design variable values for each of the unique values overlaps
                all_left_side_dvs = set()
                for i_value in range(len(values)):
                    dv_idx = np.where(idx == i_value)[0]
                    left_side_dvs = {hash(tuple(dv)) for dv in imputed_choice_opt_idx[dv_idx, :i_dv]}
                    if len(left_side_dvs & all_left_side_dvs) > 0:
                        break
                    all_left_side_dvs |= left_side_dvs
                else:
                    is_forced_[i_dv] = True
            return is_forced_

        # Determine scenario-by-scenario
        is_forced = np.zeros((self.n_choices,), dtype=bool)
        for scenario in self._reduced_selection_choice_scenarios:
            choice_idx = scenario.choice_idx
            i_sorted = np.argsort(choice_idx)
            i_reverse = np.empty((len(i_sorted),), dtype=int)
            i_reverse[i_sorted] = np.arange(len(i_sorted))

            is_forced_scenario = np.ones((len(choice_idx),), dtype=bool)
            for opt_idx_combs in scenario.opt_idx_combinations:
                # Forced status can only change if there are more than 1 combination
                if opt_idx_combs.shape[0] > 1:
                    # If there is only 1 DV, it is not forced
                    if opt_idx_combs.shape[1] == 1:
                        is_forced_scenario[0] = False
                        break

                    is_forced_scenario &= _is_forced(opt_idx_combs[:, i_sorted])
                    if np.all(~is_forced_scenario):
                        break

            is_forced[choice_idx] = is_forced_scenario[i_reverse]

        return is_forced

    def _get_comb_idx(self, opt_idx: List[int], include_mask: np.ndarray = None) \
            -> Tuple[Optional[int], Optional[np.ndarray]]:
        if len(opt_idx) != self.n_choices:
            raise ValueError(f'Incorrect nr of opt idx: {self.n_choices} != {len(opt_idx)}')

        # If we get a zero-length vector, there is only one combination
        if len(opt_idx) == 0:
            return 0, np.zeros((0,), dtype=int)

        # Initialize inclusion mask
        can_select_all = True
        if include_mask is None or np.all(include_mask):
            i_comb_possible = None
        else:
            i_comb_possible = set(np.where(include_mask)[0])
            if len(i_comb_possible) == 0:
                return None, None
            elif len(i_comb_possible) == len(include_mask):
                i_comb_possible = None
            else:
                can_select_all = False

        indep_scenarios = self._indep_sel_choice_scenarios
        dep_scenarios = self._dep_sel_choice_scenarios
        activation_map = self._reduced_sel_choice_activation_map
        iter_spec = self._scenario_iter_spec

        iter_spec_map = {(its.i_scenario, its.i_usi, its.i_comb): its for its in iter_spec}
        is_forced = self.selection_choice_is_forced
        is_non_forced = ~is_forced

        def _find_correct_opt_idx(i_sc, scenario: SelectionChoiceScenarios, i_usi, i_dv: np.ndarray) -> int:
            nonlocal i_comb_possible
            mod_opt_idx = sel_opt_idx[i_dv]
            non_forced_mask = is_non_forced[i_dv]

            # Find existing option-index combination
            opt_idx_combinations = scenario.opt_idx_combinations[i_usi]
            for ic, opt_idx_comb in enumerate(opt_idx_combinations):
                if np.all(opt_idx_comb[non_forced_mask] == mod_opt_idx[non_forced_mask]):
                    sel_opt_idx[i_dv] = opt_idx_comb

                    # Check if we can select this solution
                    its = iter_spec_map[i_sc, i_usi, ic]
                    ic_set = its.i_set
                    if i_comb_possible is None:
                        if len(ic_set) > 0:
                            i_comb_possible = set(ic_set)
                            return ic
                    else:
                        if len(ic_set & i_comb_possible) > 0:
                            i_comb_possible &= ic_set
                            return ic

            # Determine which of the combinations we can choose from
            i_sc_comb_avail = None
            if not can_select_all:
                i_sc_comb_avail = []
                for ic in range(opt_idx_combinations.shape[0]):
                    its = iter_spec_map[i_sc, i_usi, ic]
                    if i_comb_possible is None:
                        if len(its.i_set) > 0:
                            i_sc_comb_avail.append(ic)
                    else:
                        if len(its.i_set & i_comb_possible) > 0:
                            i_sc_comb_avail.append(ic)

                if len(i_sc_comb_avail) == 0:
                    raise RuntimeError('No more available!')
                opt_idx_combinations = opt_idx_combinations[i_sc_comb_avail, :]

            # Find the closest combination
            diff = opt_idx_combinations-mod_opt_idx
            diff[opt_idx_combinations == X_INACTIVE_VALUE] = 0
            diff[:, is_forced[i_dv]] = 0
            dist = np.sqrt(np.nansum(diff**2, axis=1))
            i_min_dist = np.argmin(dist)

            sel_opt_idx[i_dv] = opt_idx_combinations[i_min_dist, :]
            ic = int(i_min_dist)
            ic = i_sc_comb_avail[ic] if i_sc_comb_avail is not None else ic

            ic_set = iter_spec_map[i_sc, i_usi, ic].i_set
            if i_comb_possible is None:
                i_comb_possible = set(ic_set)
            else:
                i_comb_possible &= ic_set
            return ic

        # Parse option indices starting from independent scenarios
        sel_opt_idx = np.zeros((len(is_non_forced),), dtype=int)
        input_incl_forced = len(opt_idx) == len(is_non_forced)
        if input_incl_forced:
            sel_opt_idx[:] = opt_idx
        else:
            sel_opt_idx[is_non_forced] = opt_idx

        for i_indep, indep_scenario in indep_scenarios.items():
            i_indep_comb = _find_correct_opt_idx(i_indep, indep_scenario, 0, indep_scenario.choice_idx)

            # Parse option indices of dependent scenarios
            for i_dep, i_input in (activation_map.get((i_indep, i_indep_comb)) or []):
                dep_scenario = dep_scenarios[i_dep]
                unique_idx = dep_scenario.unique_scenario_idx[i_input]
                _find_correct_opt_idx(i_dep, dep_scenario, unique_idx, dep_scenario.choice_idx)

        if i_comb_possible is None or len(i_comb_possible) != 1:
            raise RuntimeError(f'Could not find unique option-index combination: {opt_idx}')

        i_sel_comb = list(i_comb_possible)[0]
        return_opt_idx = sel_opt_idx if input_incl_forced else sel_opt_idx[is_non_forced]
        return i_sel_comb, return_opt_idx

    @cached_property
    def _reduced_selection_choice_scenarios(self) -> List[SelectionChoiceScenarios]:
        """
        Merge coupled scenarios such that there is one base group with one or more groups only dependent on that base
        """
        # Determine which scenarios influence each other
        scenarios = self._influence_matrix.base_sel_choice_scenarios

        def _update_coupling_matrix():
            nonlocal coupling_matrix
            scenario_opt_idx_map = defaultdict(list)
            for i_scen, sc in enumerate(scenarios):
                for opt_indices in sc.opt_node_idx:
                    for opt_idx in opt_indices:
                        scenario_opt_idx_map[opt_idx].append(i_scen)

            coupling_matrix = np.zeros((len(scenarios), len(scenarios)), dtype=bool)
            for ii, scenario_ in enumerate(scenarios):
                for input_node_idx in scenario_.input_node_idx:
                    coupling_matrix[scenario_opt_idx_map[input_node_idx], ii] = True

        coupling_matrix = np.array([])
        _update_coupling_matrix()

        # Eliminate a special case of mutual incompatibility
        for i, j in itertools.combinations(range(len(scenarios)), 2):
            if coupling_matrix[i, j] and coupling_matrix[j, i]:
                replaced_scen = self._eliminate_feedback_incompatibility(scenarios[i], scenarios[j])
                if replaced_scen is not None:
                    scenarios = [replaced_scen if i_sc == i else scenario for i_sc, scenario in enumerate(scenarios)]
                    _update_coupling_matrix()

        # Merge choices constrained by a choice constraint
        base_status = np.diag(self._influence_base_matrix)
        for choice_constraint in self.adsg.get_choice_constraints():
            choice_nodes = set(choice_constraint.nodes)
            constrained_scenarios = [(i, scenario) for i, scenario in enumerate(scenarios)
                                     if scenario.choice_nodes[0] in choice_nodes]
            if len(constrained_scenarios) == 0:
                continue

            is_all_permanent = all(base_status[scenario.nodes_idx[0]] == Diag.CONFIRMED.value
                                   for _, scenario in constrained_scenarios)

            i_sc, merged_scenario = constrained_scenarios[0]
            if coupling_matrix[i_sc, i_sc]:
                merged_scenario = self._merge_scenarios(merged_scenario, merged_scenario)

            for i_sc, scenario in constrained_scenarios[1:]:
                # Merge with self first if self-referencing
                if coupling_matrix[i_sc, i_sc]:
                    scenario = self._merge_scenarios(scenario, scenario)

                merged_scenario = self._merge_scenarios(merged_scenario, scenario, constraint=choice_constraint.type,
                                                        is_all_permanent=is_all_permanent)

            i_replace = constrained_scenarios[0][0]
            i_other = set([i for i, _ in constrained_scenarios][1:])
            scenarios = [merged_scenario if k == i_replace else scenario
                         for k, scenario in enumerate(scenarios) if k not in i_other]
            _update_coupling_matrix()

        # Merge scenarios that reference the same option nodes
        while True:
            for i, scenario in enumerate(scenarios):
                if np.any(np.concatenate(scenario.opt_node_idx) != np.concatenate(scenario.unshared_opt_node_idx)):

                    # Check if already merged
                    shared_opt_idx = set(np.concatenate(scenario.opt_node_idx))
                    if len(shared_opt_idx - set(np.concatenate(scenario.unshared_opt_node_idx))) == 0:
                        continue

                    sharing_scenario = scenario
                    break
            else:
                break

            for j, scenario in enumerate(scenarios):
                if i == j:
                    continue
                if len(set(np.concatenate(scenario.unshared_opt_node_idx)) & shared_opt_idx) > 0:
                    break
            else:
                break

            # print(f'{datetime.datetime.now().isoformat()} Merging shared option nodes {i} -> {j}')
            merged_scenario = self._merge_scenarios(sharing_scenario, scenario)
            scenarios = [merged_scenario if k == j else scenario for k, scenario in enumerate(scenarios) if k != i]
            _update_coupling_matrix()

        def _merge_one_of(merging_candidates: List[Tuple[int, int]], insert_reverse=False):
            nonlocal scenarios

            n_combs_estimate = [scenarios[ii].n_total*scenarios[jj].n_total for ii, jj in merging_candidates]
            # print(f'Estimates: {merging_candidates} -> {n_combs_estimate}')
            i_merge = np.argmin(n_combs_estimate)
            ii, jj = merging_candidates[i_merge]

            # print(f'{datetime.datetime.now().isoformat()} Merging {ii} -> {jj} '
            #       f'({scenarios[ii].n_total} -> {scenarios[jj].n_total})')
            merged_scenario_ = self._merge_scenarios(scenarios[ii], scenarios[jj])
            # print(f'{datetime.datetime.now().isoformat()} DONE    {ii} -> {jj} ({merged_scenario_.n_total}, '
            #       f'{merged_scenario_.n_total*100/(scenarios[ii].n_total*scenarios[jj].n_total):.1f}%)')
            if insert_reverse:
                ii, jj = jj, ii
            if ii == jj:
                jj = -1
            scenarios = [merged_scenario_ if k == ii else scenario for k, scenario in enumerate(scenarios) if k != jj]
            _update_coupling_matrix()

        def _try_reduce_large_scenarios():
            nonlocal scenarios
            while True:
                # If one scenario has an excessively high number of combinations, try to reduce
                for i, scenario in enumerate(scenarios):
                    if i == 0 or scenario.n_total < 1000:
                        continue

                    # Only possible to reduce if there is no feedback before the large scenario
                    has_feedback = coupling_matrix[:, :i].copy()
                    has_feedback[np.triu_indices(i)] = False
                    if np.any(has_feedback):
                        continue

                    # Merge inputting scenarios
                    inputting_scenarios = np.where(coupling_matrix[:i, i])[0]
                    if len(inputting_scenarios) == 0:
                        return
                    if len(inputting_scenarios) == 1:
                        # print(f'Reducing large: merging {inputting_scenarios[0]} -> {i}')
                        _merge_one_of([(inputting_scenarios[0], i)])
                        break
                    else:
                        # print(f'Reducing large: merging inputs: {inputting_scenarios}')
                        _merge_one_of(list(itertools.combinations(inputting_scenarios, 2)))
                        break
                else:
                    break

        # Merge scenarios with self-loops
        while True:
            i_has_self_loop = np.where(np.diag(coupling_matrix))[0]
            if len(i_has_self_loop) == 0:
                break

            _merge_one_of([(i, i) for i in i_has_self_loop])

        # Merge coupling
        indep_scenarios = []
        while len(scenarios) > 1:
            _try_reduce_large_scenarios()

            # Merge feedback connection (starting closest to diagonal)
            could_be_merged = []
            insert_reverse = True
            for i_diag in range(1, coupling_matrix.shape[0]):
                i_feedback = np.where(np.diag(coupling_matrix, k=-i_diag))[0]
                if len(i_feedback) > 0:
                    for j_fb in i_feedback:
                        j = int(j_fb)
                        i = i_diag + j
                        could_be_merged.append((i, j))
                    break

            if len(could_be_merged) == 0:
                # Merge feedforward
                insert_reverse = False
                for i in range(1, coupling_matrix.shape[0]-1):
                    i_ff = np.where(np.diag(coupling_matrix[1:, 1:], k=i))[0]
                    if len(i_ff) > 0:
                        j = 1 + i + int(i_ff[0])
                        i = 1 + int(i_ff[0])
                        could_be_merged = [(i, j), (0, 1)]
                        break
                else:
                    break

            _merge_one_of(could_be_merged, insert_reverse=insert_reverse)

            # Separate independent scenarios
            is_indep = [np.all(~coupling_matrix[i, :]) and np.all(~coupling_matrix[:, i])
                        for i in range(len(scenarios))]
            if any(is_indep):
                indep_scenarios += [scenario for i, scenario in enumerate(scenarios) if is_indep[i]]
                scenarios = [sc for i, sc in enumerate(scenarios) if not is_indep[i]]
                _update_coupling_matrix()

        # Remove unused inputs for dependent scenarios
        if len(scenarios) > 1:
            # print(f'{datetime.datetime.now().isoformat()} Merge final (unused)')
            base_scenario = scenarios[0]
            dep_scenarios = [self._merge_scenarios(base_scenario, scenario, mod_target=True)
                             for scenario in scenarios[1:]]
            base_scenario = self._remove_zero_comb_downstream(base_scenario, dep_scenarios)
            scenarios = [base_scenario]+dep_scenarios

        # print(f'{datetime.datetime.now().isoformat()} Done')
        return indep_scenarios+scenarios

    @cached_property
    def _indep_sel_choice_scenarios(self) -> Dict[int, SelectionChoiceScenarios]:
        scenarios = self._reduced_selection_choice_scenarios
        return {i: scenario for i, scenario in enumerate(scenarios) if len(scenario.input_node_idx) == 0}

    @cached_property
    def _dep_sel_choice_scenarios(self) -> Dict[int, SelectionChoiceScenarios]:
        scenarios = self._reduced_selection_choice_scenarios
        return {i: scenario for i, scenario in enumerate(scenarios) if len(scenario.input_node_idx) > 0}

    @cached_property
    def _reduced_sel_choice_activation_map(self) -> Dict[Tuple[int, int], List[Tuple[int, int]]]:
        """
        Returns a map which dependent scenario inputs are activated by which independent opt_idx combinations:
        [i_scen_indep, i_comb] -> [(i_scen_dep, i_input), ...]
        """
        indep_scenarios = self._indep_sel_choice_scenarios
        dep_scenarios = self._dep_sel_choice_scenarios
        if len(dep_scenarios) == 0:
            return {}

        # Get the choice-option combinations for each input
        dep_scenario_activation_map: Dict[Any, list] = defaultdict(list)
        for i, dep_scenario in dep_scenarios.items():
            input_node_idx = dep_scenario.input_node_idx
            input_node_key = tuple(input_node_idx)
            for i_in, input_status in enumerate(dep_scenario.input_status_matrix):
                choice_opt_i = (input_node_key, tuple(input_status))
                dep_scenario_activation_map[choice_opt_i].append((i, i_in))

        # Determine how independent scenarios activate dependent scenario inputs
        sel_choice_activation_map: Dict[Any, list] = defaultdict(list)
        for i, indep_scenario in indep_scenarios.items():
            can_activate = set(np.concatenate(indep_scenario.opt_node_idx))
            for i_comb, node_idx in enumerate(indep_scenario.node_idx_combinations[0]):
                node_idx_i = set(node_idx)

                # Loop over scenario activations to find matches
                for (input_nodes, status), act_scenarios in dep_scenario_activation_map.items():
                    if self._matches_input_scenario(input_nodes, status, can_activate, node_idx_i):
                        sel_choice_activation_map[i, i_comb] += act_scenarios

        return sel_choice_activation_map

    @staticmethod
    def _matches_input_scenario(input_nodes, input_status, can_activate, node_idx_i):
        # Check if the independent scenario determines the activation of all input nodes
        if len(set(input_nodes) - can_activate) > 0:
            return False

        # Check if input node statuses match
        has_input_node = np.array([inp_node_idx in node_idx_i for inp_node_idx in input_nodes])
        return np.all(has_input_node == input_status)

    def _eliminate_feedback_incompatibility(self, source: SelectionChoiceScenarios, target: SelectionChoiceScenarios) \
            -> Optional[SelectionChoiceScenarios]:
        assert len(source.choice_idx) == 1
        assert len(target.choice_idx) == 1

        # Check if there are any removal influences between option nodes
        src_opt_idx = np.array(sorted(np.concatenate(source.opt_node_idx)))
        tgt_opt_idx = np.array(sorted(np.concatenate(target.opt_node_idx)))

        if len(set(src_opt_idx) & set(tgt_opt_idx)) > 0:
            return

        base_matrix = self._influence_base_matrix
        src_to_tgt_mat = base_matrix[np.ix_(src_opt_idx, tgt_opt_idx)]
        tgt_to_src_mat = base_matrix[np.ix_(tgt_opt_idx, src_opt_idx)]

        if not np.any(src_to_tgt_mat == OffDiag.REMOVAL.value) or np.any(src_to_tgt_mat != tgt_to_src_mat.T):
            return

        confirmed_status = Diag.CONFIRMED.value

        shared_inp_node_idx = sorted(set(source.input_node_idx) & set(target.input_node_idx))
        src_inp_node_idx_map = {value: key for key, value in enumerate(source.input_node_idx)}
        src_shared_i = np.array([src_inp_node_idx_map[node_idx] for node_idx in shared_inp_node_idx])
        tgt_inp_node_idx_map = {value: key for key, value in enumerate(target.input_node_idx)}
        tgt_shared_i = np.array([tgt_inp_node_idx_map[node_idx] for node_idx in shared_inp_node_idx])

        # Check if the target can ever be activated before the source
        src_node_idx = source.nodes_idx[0]
        if base_matrix[src_node_idx, src_node_idx] != confirmed_status:

            tgt_cpl_node_idx = sorted(set(target.input_node_idx) & set(np.concatenate(source.opt_node_idx)))
            tgt_coupled_inp_i = np.array([tgt_inp_node_idx_map[node_idx] for node_idx in tgt_cpl_node_idx], dtype=int)

            src_i_input_inactive = np.zeros((source.input_status_matrix.shape[0],), dtype=bool)
            for usi, status in enumerate(source.status_matrix):
                if status[0][0] != confirmed_status:
                    src_i_input_inactive[source.unique_scenario_idx == usi] = True

            tgt_i_input_active = np.zeros((target.input_status_matrix.shape[0],), dtype=bool)
            for usi, status in enumerate(target.status_matrix):
                if status[0][0] == confirmed_status:
                    tgt_i_input_active[target.unique_scenario_idx == usi] = True

            if len(tgt_coupled_inp_i) > 0:
                only_opt_idx_active = np.zeros((len(target.input_node_idx),), dtype=bool)
                only_opt_idx_active[tgt_coupled_inp_i] = True
                tgt_i_input_active[np.all(target.input_status_matrix == only_opt_idx_active, axis=1)] = False
            tgt_i_input_active = set(np.where(tgt_i_input_active)[0])

            if len(tgt_shared_i) == 0 and np.any(tgt_i_input_active):
                return
            for i_src_inactive in np.where(src_i_input_inactive)[0]:
                shared_status = source.input_status_matrix[i_src_inactive, src_shared_i]
                i_tgt_in = np.where(np.all(target.input_status_matrix[:, tgt_shared_i] == shared_status, axis=1))[0]
                if len(set(i_tgt_in) & tgt_i_input_active) > 0:
                    return

        # If the target can never be activated before the source, we can safely remove any feedback connections
        src_coupled_node_idx = sorted(set(source.input_node_idx) & set(np.concatenate(target.opt_node_idx)))
        src_coupled_inp_i = np.array([src_inp_node_idx_map[node_idx] for node_idx in src_coupled_node_idx], dtype=int)
        assert len(src_coupled_inp_i) > 0

        src_inp_keep = np.ones((source.input_status_matrix.shape[0],), dtype=bool)
        src_inp_keep[np.any(source.input_status_matrix[:, src_coupled_inp_i], axis=1)] = False

        keep_inp_nodes = np.ones((len(source.input_node_idx),), dtype=bool)
        keep_inp_nodes[src_coupled_inp_i] = False

        return self._get_for_kept_inputs(source, src_inp_keep, keep_inp_nodes=keep_inp_nodes)

    @classmethod
    def _merge_scenarios(cls, source: SelectionChoiceScenarios, target: SelectionChoiceScenarios, mod_target=False,
                         constraint: ChoiceConstraintType = None, is_all_permanent=False) -> SelectionChoiceScenarios:
        is_self_merge = False
        if len(source.choice_idx) == len(target.choice_idx) and np.all(source.choice_idx == target.choice_idx):
            is_self_merge = True
            all_choice_idx = np.arange(len(source.choice_idx))
        else:
            all_choice_idx = np.argsort(np.concatenate([source.choice_idx, target.choice_idx]))
        i_choice_src, i_choice_tgt = all_choice_idx[:len(source.choice_idx)], all_choice_idx[len(source.choice_idx):]

        def _merge_node_arrays(arr_self, arr_other, axis=0):
            if is_self_merge:
                return arr_self
            merged_array = np.concatenate([arr_self, arr_other], axis=axis)
            if axis == 0:
                merged_array[i_choice_src] = arr_self
                merged_array[i_choice_tgt] = arr_other
            else:
                merged_array[:, i_choice_src] = arr_self
                merged_array[:, i_choice_tgt] = arr_other
            return merged_array

        def _combine_into(src_list, tgt_list) -> list:
            combined_list = np.empty((len(all_choice_idx),), dtype=object)
            combined_list[i_choice_src] = src_list
            if not is_self_merge:
                combined_list[i_choice_tgt] = tgt_list
            return combined_list.tolist()

        choice_nodes: List[SelectionChoiceNode] = _combine_into(source.choice_nodes, target.choice_nodes)
        choice_idx = _merge_node_arrays(source.choice_idx, target.choice_idx)
        nodes_idx = _merge_node_arrays(source.nodes_idx, target.nodes_idx)
        n_opts = _merge_node_arrays(source.n_opts, target.n_opts)
        merged_opt_node_idx: List[np.ndarray] = _combine_into(source.opt_node_idx, target.opt_node_idx)
        merged_us_opt_node_idx: List[np.ndarray] = \
            _combine_into(source.unshared_opt_node_idx, target.unshared_opt_node_idx)

        # Get shared and coupled input nodes
        shared_inp_node_idx = sorted(set(source.input_node_idx) & set(target.input_node_idx))
        src_inp_node_idx_map = {value: key for key, value in enumerate(source.input_node_idx)}
        src_shared_i = np.array([src_inp_node_idx_map[node_idx] for node_idx in shared_inp_node_idx])
        tgt_inp_node_idx_map = {value: key for key, value in enumerate(target.input_node_idx)}
        tgt_shared_i = np.array([tgt_inp_node_idx_map[node_idx] for node_idx in shared_inp_node_idx])

        tgt_coupled_node_idx = sorted(set(target.input_node_idx) & set(np.concatenate(source.opt_node_idx)))
        tgt_coupled_node_idx_map = {node_idx: i for i, node_idx in enumerate(tgt_coupled_node_idx)}
        tgt_coupled_inp_i = np.array([tgt_inp_node_idx_map[node_idx] for node_idx in tgt_coupled_node_idx], dtype=int)
        src_coupled_node_idx = sorted(set(source.input_node_idx) & set(np.concatenate(target.opt_node_idx)))
        src_coupled_node_idx_map = {node_idx: i for i, node_idx in enumerate(src_coupled_node_idx)}
        src_coupled_inp_i = np.array([src_inp_node_idx_map[node_idx] for node_idx in src_coupled_node_idx], dtype=int)

        coupled_node_idx = set(tgt_coupled_node_idx) | set(src_coupled_node_idx)
        combined_inp_from_src = np.array([i for i, node_idx in enumerate(source.input_node_idx)
                                          if node_idx not in coupled_node_idx], dtype=int)
        combined_inp_from_tgt = np.array([i for i, node_idx in enumerate(target.input_node_idx)
                                          if node_idx not in coupled_node_idx and
                                          node_idx not in shared_inp_node_idx], dtype=int)
        n_comb_in = len(combined_inp_from_src)+len(combined_inp_from_tgt)

        input_node_idx = np.empty((n_comb_in,), dtype=int)
        input_node_idx[:len(combined_inp_from_src)] = source.input_node_idx[combined_inp_from_src]
        input_node_idx[len(combined_inp_from_src):] = target.input_node_idx[combined_inp_from_tgt]

        input_choice_idx = np.empty((n_comb_in,), dtype=int)
        input_choice_idx[:len(combined_inp_from_src)] = source.input_choice_idx[combined_inp_from_src]
        input_choice_idx[len(combined_inp_from_src):] = target.input_choice_idx[combined_inp_from_tgt]

        input_opt_idx = np.empty((n_comb_in,), dtype=int)
        input_opt_idx[:len(combined_inp_from_src)] = source.input_opt_idx[combined_inp_from_src]
        input_opt_idx[len(combined_inp_from_src):] = target.input_opt_idx[combined_inp_from_tgt]

        # If we are only modifying the target, there can be no source-to-target coupling and
        # the source should have only 1 input scenario
        if mod_target:
            assert len(src_coupled_node_idx) == 0

        def _argsort_combs(combs_):
            return np.lexsort(combs_.T[::-1, :])

        # Determine combined input scenarios
        src_comb_possible_cache = {}
        tgt_comb_possible_cache = {}
        combined_input_status_matrices = []
        combined_unique_idx_map = {}
        combined_unique_scenario_idx = []
        combined_status_matrices = []
        combined_opt_idx_combinations = []
        combined_node_idx_combinations = []
        combined_applied_status_combinations = []
        all_tgt_scenarios_ok = np.zeros((target.input_status_matrix.shape[0],), dtype=bool)
        for i_src_sc, src_input_status in enumerate(source.input_status_matrix):
            src_coupled_input_status = src_input_status[src_coupled_inp_i]
            src_usi = source.unique_scenario_idx[i_src_sc]
            src_status_matrix = source.status_matrix[src_usi]

            # Deselect target scenarios where shared inputs don't match
            tgt_scenarios_ok = np.ones((target.input_status_matrix.shape[0],), dtype=bool)
            if len(shared_inp_node_idx) > 0:
                if is_self_merge:
                    tgt_scenarios_ok[:] = False
                    tgt_scenarios_ok[i_src_sc] = True
                else:
                    i_check = np.arange(len(tgt_scenarios_ok))
                    for i_sh, i_tgt_sh in enumerate(tgt_shared_i):
                        i_not_ok = target.input_status_matrix[i_check, i_tgt_sh]
                        if src_input_status[src_shared_i[i_sh]]:
                            i_not_ok = ~i_not_ok
                        tgt_scenarios_ok[i_check[i_not_ok]] = False
                        i_check = i_check[~i_not_ok]

            for i_tgt_sc in np.where(tgt_scenarios_ok)[0]:
                tgt_input_status = target.input_status_matrix[i_tgt_sc, :]
                tgt_coupled_input_status = tgt_input_status[tgt_coupled_inp_i]
                tgt_usi = target.unique_scenario_idx[i_tgt_sc]
                tgt_status_matrix = target.status_matrix[tgt_usi]

                # Get combined input matrix
                combined_input_status_matrix = np.empty((n_comb_in,), dtype=bool)
                combined_input_status_matrix[:len(combined_inp_from_src)] = src_input_status[combined_inp_from_src]
                combined_input_status_matrix[len(combined_inp_from_src):] = tgt_input_status[combined_inp_from_tgt]

                # Get combined status matrix
                combined_status_matrix = _combine_into(src_status_matrix, tgt_status_matrix)

                # Check if scenario was already processed
                src_cp_key = hash((src_usi, tuple(tgt_coupled_input_status)))
                tgt_cp_key = hash((tgt_usi, tuple(src_coupled_input_status)))
                csm_key = hash(tuple(hash(tuple(mat)) for mat in combined_status_matrix) +
                               (src_cp_key, tgt_cp_key))
                if csm_key in combined_unique_idx_map:
                    # Invalid combination
                    if combined_unique_idx_map[csm_key] == -1:
                        tgt_scenarios_ok[i_tgt_sc] = False
                        continue

                    # Valid, already processed combination
                    combined_input_status_matrices.append(combined_input_status_matrix)
                    combined_unique_scenario_idx.append(combined_unique_idx_map[csm_key])
                    continue
                usi = len(combined_opt_idx_combinations)

                # Get possible source combinations given coupled source-to-target inputs
                if src_cp_key in src_comb_possible_cache:
                    src_comb_possible = src_comb_possible_cache[src_cp_key]
                else:
                    src_comb_possible = np.ones((len(source.node_idx_combinations[src_usi]),), dtype=bool)
                    if len(tgt_coupled_inp_i) > 0:
                        for i_comb_src, src_opt_node_indices in enumerate(source.node_idx_combinations[src_usi]):
                            tgt_coupled_inputs = np.zeros((len(tgt_coupled_inp_i),), dtype=bool)
                            for opt_node_idx in src_opt_node_indices:
                                if opt_node_idx != X_INACTIVE_VALUE and opt_node_idx in tgt_coupled_node_idx_map:
                                    tgt_coupled_inputs[tgt_coupled_node_idx_map[opt_node_idx]] = True

                            if any(tgt_coupled_input_status != tgt_coupled_inputs):
                                src_comb_possible[i_comb_src] = False

                    if not any(src_comb_possible) and len(src_comb_possible) > 0:
                        src_comb_possible = None
                    else:
                        src_comb_possible = np.where(src_comb_possible)[0]
                    src_comb_possible_cache[src_cp_key] = src_comb_possible

                # Get possible target combinations given coupled target-to-source inputs
                if tgt_cp_key in tgt_comb_possible_cache:
                    tgt_comb_possible = tgt_comb_possible_cache[tgt_cp_key]
                else:
                    tgt_comb_possible = np.ones((len(target.node_idx_combinations[tgt_usi]),), dtype=bool)
                    if len(src_coupled_inp_i) > 0:
                        for i_comb_tgt, tgt_opt_node_indices in enumerate(target.node_idx_combinations[tgt_usi]):
                            src_coupled_inputs = np.zeros((len(src_coupled_inp_i),), dtype=bool)
                            for opt_node_idx in tgt_opt_node_indices:
                                if opt_node_idx != X_INACTIVE_VALUE and opt_node_idx in src_coupled_node_idx_map:
                                    src_coupled_inputs[src_coupled_node_idx_map[opt_node_idx]] = True

                            if any(src_coupled_input_status != src_coupled_inputs):
                                tgt_comb_possible[i_comb_tgt] = False

                    if not any(tgt_comb_possible) and len(tgt_comb_possible) > 0:
                        tgt_comb_possible = None
                    else:
                        tgt_comb_possible = np.where(tgt_comb_possible)[0]
                    tgt_comb_possible_cache[tgt_cp_key] = tgt_comb_possible

                # If no combined combinations are possible, this combined scenario is invalid
                if src_comb_possible is None or tgt_comb_possible is None:
                    combined_unique_idx_map[csm_key] = -1
                    tgt_scenarios_ok[i_tgt_sc] = False
                    continue

                combined_unique_idx_map[csm_key] = usi
                if mod_target:
                    continue

                # Generate combined choice-option combinations
                if is_self_merge:
                    i_src_comb = src_comb_possible
                    i_tgt_comb = np.tile(tgt_comb_possible[:1], len(i_src_comb))
                else:
                    i_src_comb = np.repeat(src_comb_possible, len(tgt_comb_possible))
                    i_tgt_comb = np.tile(tgt_comb_possible, len(src_comb_possible))

                combined_opt_idx_comb = np.empty((len(i_src_comb), len(all_choice_idx)), dtype=int)
                combined_opt_idx_comb[:, i_choice_src] = source.opt_idx_combinations[src_usi][i_src_comb, :]
                if not is_self_merge:
                    combined_opt_idx_comb[:, i_choice_tgt] = target.opt_idx_combinations[tgt_usi][i_tgt_comb, :]

                i_comb_sorted = _argsort_combs(combined_opt_idx_comb)
                combined_opt_idx_comb = combined_opt_idx_comb[i_comb_sorted, :]
                if constraint is not None:  # Apply option-index constraint
                    i_valid = get_valid_idx_combinations(
                        combined_opt_idx_comb, constraint, is_all_permanent=is_all_permanent)
                    combined_opt_idx_comb = combined_opt_idx_comb[i_valid, :]
                    i_comb_sorted = i_comb_sorted[i_valid]

                combined_opt_idx_combinations.append(combined_opt_idx_comb)

                combined_node_idx_comb = np.empty((len(i_src_comb), len(all_choice_idx)), dtype=int)
                combined_node_idx_comb[:, i_choice_src] = source.node_idx_combinations[src_usi][i_src_comb, :]
                if not is_self_merge:
                    combined_node_idx_comb[:, i_choice_tgt] = target.node_idx_combinations[tgt_usi][i_tgt_comb, :]
                combined_node_idx_comb = combined_node_idx_comb[i_comb_sorted, :]
                combined_node_idx_combinations.append(combined_node_idx_comb)

                combined_applied_status_combinations.append(_combine_into(
                    [mat[i_src_comb, :][i_comb_sorted, :]
                     for i, mat in enumerate(source.applied_status_combinations[src_usi])],
                    [mat[i_tgt_comb, :][i_comb_sorted, :]
                     for i, mat in enumerate(target.applied_status_combinations[tgt_usi])],
                ))

                # Add new unique scenario
                combined_status_matrices.append(_combine_into(src_status_matrix, tgt_status_matrix))
                combined_input_status_matrices.append(combined_input_status_matrix)

                combined_unique_scenario_idx.append(usi)

            # Keep track of valid target scenarios if we want to modify only the target
            if mod_target:
                all_tgt_scenarios_ok |= tgt_scenarios_ok

        # Only modify the target scenarios given the source
        if mod_target:
            # Mark external input scenarios as valid
            external_node_idx = set(target.input_node_idx)-set(tgt_coupled_node_idx)
            for external_idx in external_node_idx:
                input_idx = np.where(target.input_node_idx == external_idx)[0][0]
                all_tgt_scenarios_ok[target.input_status_matrix[:, input_idx]] = True

            return cls._get_for_kept_inputs(target, all_tgt_scenarios_ok)

        # Merge input matrices (if there is coupling, one input matrix results in multiple scenarios)
        combined_input_status_matrices = np.array(combined_input_status_matrices)
        unique_ism, ism_map = np.unique(combined_input_status_matrices, axis=0, return_inverse=True)
        unique_scenario_idx = np.array(combined_unique_scenario_idx, dtype=int)

        dedup_input_status_matrices = []
        dedup_unique_scenario_idx = []
        unique_scenario_merge_map = {}
        retained_us = [False for _ in range(len(combined_status_matrices))]
        for i_inp, input_status_matrices in enumerate(unique_ism):
            input_idx = np.where(ism_map == i_inp)[0]
            used_usi = np.unique(unique_scenario_idx[input_idx])

            # Detect self-activation: if one of the referenced unique scenarios starts from all-inactive choices, then
            # there can't be another unique scenario where one of the choices is active
            is_all_inactive = [np.all([status[0] != Diag.CONFIRMED.value for status in combined_status_matrices[usi]])
                               for usi in used_usi]
            if any(is_all_inactive) and not all(is_all_inactive):
                used_usi = used_usi[is_all_inactive]

            # If there is no duplication, keep original scenario
            if len(used_usi) == 1:
                dedup_input_status_matrices.append(input_status_matrices)
                dedup_unique_scenario_idx.append(used_usi[0])
                retained_us[used_usi[0]] = True
                continue

            # Check if this specific set of unique scenario references has already been merged
            aus_key = tuple(used_usi)
            if aus_key in unique_scenario_merge_map:
                dedup_input_status_matrices.append(input_status_matrices)
                dedup_unique_scenario_idx.append(unique_scenario_merge_map[aus_key])
                continue
            merged_usi = len(combined_status_matrices)

            # Keep initial status matrix of first references unique scenario
            merged_status_matrix = combined_status_matrices[used_usi[0]]

            # Merge option combinations
            merged_opt_idx_comb = np.row_stack([combined_opt_idx_combinations[i] for i in used_usi])
            i_sorted = _argsort_combs(merged_opt_idx_comb)
            merged_opt_idx_comb = merged_opt_idx_comb[i_sorted, :]
            merged_node_idx_comb = np.row_stack([combined_node_idx_combinations[i] for i in used_usi])[i_sorted, :]
            merged_applied_status_comb = [np.row_stack([combined_applied_status_combinations[i][i_node]
                                                        for i in used_usi])[i_sorted, :]
                                          for i_node in range(len(choice_nodes))]

            # Add new unique scenario
            combined_status_matrices.append(merged_status_matrix)
            combined_opt_idx_combinations.append(merged_opt_idx_comb)
            combined_node_idx_combinations.append(merged_node_idx_comb)
            combined_applied_status_combinations.append(merged_applied_status_comb)

            dedup_input_status_matrices.append(input_status_matrices)
            dedup_unique_scenario_idx.append(merged_usi)
            retained_us.append(True)
            unique_scenario_merge_map[aus_key] = merged_usi

        # Remove unused unique scenarios
        combined_status_matrices = [value for i, value in enumerate(combined_status_matrices) if retained_us[i]]
        combined_opt_idx_combinations = [value for i, value in enumerate(combined_opt_idx_combinations)
                                         if retained_us[i]]
        combined_node_idx_combinations = [value for i, value in enumerate(combined_node_idx_combinations)
                                          if retained_us[i]]
        combined_applied_status_combinations = [value for i, value in enumerate(combined_applied_status_combinations)
                                                if retained_us[i]]

        new_unique_scenario_idx = np.zeros((len(dedup_input_status_matrices),), dtype=int)
        unique_scenario_idx = np.array(dedup_unique_scenario_idx, dtype=int)
        for i, i_old in enumerate(np.where(retained_us)[0]):
            new_unique_scenario_idx[unique_scenario_idx == i_old] = i
        unique_scenario_idx = new_unique_scenario_idx

        # Count combinations
        n_combinations_unique = np.array([combs.shape[0] for combs in combined_opt_idx_combinations])
        n_combinations = n_combinations_unique[unique_scenario_idx]

        return SelectionChoiceScenarios(
            choice_nodes=choice_nodes,
            choice_idx=choice_idx,
            nodes_idx=nodes_idx,
            n_opts=n_opts,
            opt_node_idx=merged_opt_node_idx,
            unshared_opt_node_idx=merged_us_opt_node_idx,
            input_node_idx=input_node_idx,
            input_choice_idx=input_choice_idx,
            input_opt_idx=input_opt_idx,
            input_status_matrix=np.array(dedup_input_status_matrices),
            unique_scenario_idx=unique_scenario_idx,
            n_combinations=n_combinations,
            n_total=int(np.sum(n_combinations)),
            status_matrix=combined_status_matrices,
            n_combinations_unique=n_combinations_unique,
            opt_idx_combinations=combined_opt_idx_combinations,
            node_idx_combinations=combined_node_idx_combinations,
            applied_status_combinations=combined_applied_status_combinations,
        )

    @staticmethod
    def _get_for_kept_inputs(scenario: SelectionChoiceScenarios, keep_mask: np.ndarray,
                             keep_inp_nodes: np.ndarray = None, keep_comb_idx: Dict[int, np.ndarray] = None) \
            -> SelectionChoiceScenarios:
        if np.all(keep_mask) and keep_comb_idx is None:
            return scenario

        input_status_matrix = scenario.input_status_matrix[keep_mask, :]
        unique_scenario_idx = scenario.unique_scenario_idx[keep_mask]
        kept_usi = np.unique(unique_scenario_idx)

        input_node_idx = scenario.input_node_idx
        input_choice_idx = scenario.input_choice_idx
        input_opt_idx = scenario.input_opt_idx
        if keep_inp_nodes is not None:
            input_node_idx = input_node_idx[keep_inp_nodes]
            input_choice_idx = input_choice_idx[keep_inp_nodes]
            input_opt_idx = input_opt_idx[keep_inp_nodes]
            input_status_matrix = input_status_matrix[:, keep_inp_nodes]

        status_matrix = [scenario.status_matrix[usi] for usi in kept_usi]
        opt_idx_combinations = [scenario.opt_idx_combinations[usi] for usi in kept_usi]
        node_idx_combinations = [scenario.node_idx_combinations[usi] for usi in kept_usi]
        applied_status_combinations = [scenario.applied_status_combinations[usi] for usi in kept_usi]

        if keep_comb_idx is not None:
            for i_usi, keep_comb_mask in keep_comb_idx.items():
                opt_idx_combinations[i_usi] = opt_idx_combinations[i_usi][keep_comb_mask, :]
                node_idx_combinations[i_usi] = node_idx_combinations[i_usi][keep_comb_mask, :]
                applied_status_combinations[i_usi] = \
                    [status_combs[keep_comb_mask, :] for status_combs in applied_status_combinations[i_usi]]

        new_unique_scenario_idx = np.zeros((len(unique_scenario_idx),), dtype=int)
        for i, i_old in enumerate(kept_usi):
            new_unique_scenario_idx[unique_scenario_idx == i_old] = i
        unique_scenario_idx = new_unique_scenario_idx

        n_combinations_unique = np.array([combs.shape[0] for combs in opt_idx_combinations])
        n_combinations = n_combinations_unique[unique_scenario_idx]

        return SelectionChoiceScenarios(
            choice_nodes=scenario.choice_nodes,
            choice_idx=scenario.choice_idx,
            nodes_idx=scenario.nodes_idx,
            n_opts=scenario.n_opts,
            opt_node_idx=scenario.opt_node_idx,
            unshared_opt_node_idx=scenario.unshared_opt_node_idx,
            input_node_idx=input_node_idx,
            input_choice_idx=input_choice_idx,
            input_opt_idx=input_opt_idx,
            input_status_matrix=input_status_matrix,
            unique_scenario_idx=unique_scenario_idx,
            n_combinations=n_combinations,
            n_total=int(np.sum(n_combinations)),
            status_matrix=status_matrix,
            n_combinations_unique=n_combinations_unique,
            opt_idx_combinations=opt_idx_combinations,
            node_idx_combinations=node_idx_combinations,
            applied_status_combinations=applied_status_combinations,
        )

    @classmethod
    def _remove_zero_comb_downstream(cls, source: SelectionChoiceScenarios, targets: List[SelectionChoiceScenarios]) \
            -> SelectionChoiceScenarios:

        # Check if there are any target scenarios with zero-length combinations
        if len(targets) == 0:
            return source
        for target in targets:
            if np.any(target.n_combinations_unique == 0):
                break
        else:
            return source

        # Get the choice-option combinations for each input
        target_scenario_empty_comb_map = []
        for target in targets:
            for i_in, input_status in enumerate(target.input_status_matrix):
                if target.n_combinations[i_in] == 0:
                    target_scenario_empty_comb_map.append((tuple(target.input_node_idx), tuple(input_status)))

        # Determine how independent scenarios activate dependent scenario inputs
        can_activate = set(np.concatenate(source.opt_node_idx))
        i_comb_keep = np.ones((source.node_idx_combinations[0].shape[0],), dtype=bool)
        all_kept = True
        for i_comb, node_idx in enumerate(source.node_idx_combinations[0]):
            node_idx_i = set(node_idx)

            # Loop over scenario activations to find matches
            for input_nodes, status in target_scenario_empty_comb_map:
                # Mark the combination for removal if it activates a zero-length scenario
                if cls._matches_input_scenario(input_nodes, status, can_activate, node_idx_i):
                    i_comb_keep[i_comb] = False
                    all_kept = False
                    break

        # Remove combinations from source scenario
        if all_kept:
            return source

        keep_mask = np.array([True])
        return cls._get_for_kept_inputs(source, keep_mask, keep_comb_idx={0: i_comb_keep})

    @staticmethod
    def _cartesian_product(src1: np.ndarray, src2: np.ndarray) -> np.ndarray:
        src1_select = np.repeat(np.arange(src1.shape[0]), src2.shape[0])
        src2_select = np.tile(np.arange(src2.shape[0]), src1.shape[0])
        return np.column_stack([src1[src1_select, :], src2[src2_select, :]])

    def _generate_all_opt_idx_combinations(self) -> np.ndarray:
        """Generate all possible choice-option-index combinations; avoid as this may lead to memory problems"""
        indep_scenarios = self._indep_sel_choice_scenarios
        dep_scenarios = self._dep_sel_choice_scenarios
        activation_map = self._reduced_sel_choice_activation_map

        if len(indep_scenarios) == 0:
            return np.zeros((1, 0), dtype=int)

        # Build vectors for independent scenarios
        merged_cartesian_product = None
        merged_all_choice_idx = []
        for i_indep, scenario in indep_scenarios.items():
            indep_opt_idx_combs_mat = []
            all_choice_idx = set()
            for i_comb, opt_idx_comb in enumerate(scenario.opt_idx_combinations[0]):
                choice_idx = scenario.choice_idx
                combs_product = np.array([opt_idx_comb])
                for i_dep, i_input in activation_map.get((i_indep, i_comb), []):
                    dep_scenario = dep_scenarios[i_dep]
                    unique_idx = dep_scenario.unique_scenario_idx[i_input]
                    dep_opt_idx_comb = dep_scenario.opt_idx_combinations[unique_idx]

                    combs_product = self._cartesian_product(combs_product, dep_opt_idx_comb)
                    choice_idx = np.concatenate([choice_idx, dep_scenario.choice_idx])

                combs_mat_i = np.empty((combs_product.shape[0], self.n_choices), dtype=int)
                combs_mat_i.fill(X_INACTIVE_VALUE)
                combs_mat_i[:, choice_idx] = combs_product
                indep_opt_idx_combs_mat.append(combs_mat_i)

                all_choice_idx |= set(choice_idx)

            all_choice_idx = np.array(sorted(all_choice_idx))
            if len(indep_opt_idx_combs_mat) == 0:
                indep_opt_idx_combs_mat = np.zeros((0, len(scenario.choice_idx)), dtype=int)
            else:
                indep_opt_idx_combs_mat = np.row_stack(indep_opt_idx_combs_mat)[:, all_choice_idx]

            # Get Cartesian product of independent scenarios
            if merged_cartesian_product is None:
                merged_cartesian_product = indep_opt_idx_combs_mat
            else:
                merged_cartesian_product = self._cartesian_product(merged_cartesian_product, indep_opt_idx_combs_mat)
            merged_all_choice_idx += list(all_choice_idx)

        sorted_cartesian_product = np.empty(merged_cartesian_product.shape, dtype=int)
        sorted_cartesian_product[:, merged_all_choice_idx] = merged_cartesian_product
        return sorted_cartesian_product

    def _generate_existence_array(self) -> np.ndarray:
        """Generate all possible node existence statuses; avoid as this may lead to memory problems"""
        indep_scenarios = self._indep_sel_choice_scenarios
        dep_scenarios = self._dep_sel_choice_scenarios
        activation_map = self._reduced_sel_choice_activation_map

        base_matrix = self._influence_base_matrix
        base_status = np.diag(base_matrix)
        shared_node_idx = defaultdict(list)

        if len(indep_scenarios) == 0:
            return np.array([base_status])

        def _process_choice_opt_node_idx(sc: SelectionChoiceScenarios):
            # Map shared node indices to unshared indices
            for i_node, unshared_opt_idx in enumerate(sc.unshared_opt_node_idx):
                for i_opt, opt_node_idx in enumerate(unshared_opt_idx):
                    if opt_node_idx != sc.opt_node_idx[i_node][i_opt]:
                        shared_node_idx[sc.opt_node_idx[i_node][i_opt]].append(opt_node_idx)

            return [node_idx for i_node, choice_node_idx in enumerate(sc.nodes_idx)
                    for node_idx in [choice_node_idx]+list(sc.unshared_opt_node_idx[i_node])]

        # Build status vectors for independent scenarios
        seen_node_idx = set()
        merged_cartesian_product = None
        merged_all_nodes_idx = []
        for i_indep, scenario in indep_scenarios.items():
            scen_nodes_idx = _process_choice_opt_node_idx(scenario)

            indep_status_combs_mat = []
            all_node_idx = set()
            for i_comb in range(scenario.applied_status_combinations[0][0].shape[0]):
                nodes_idx = scen_nodes_idx.copy()
                combs_product = np.column_stack([np.array([scenario.applied_status_combinations[0][i_node][i_comb]])
                                                 for i_node in range(len(scenario.nodes_idx))])

                # Apply dependent scenarios
                for i_dep, i_input in activation_map.get((i_indep, i_comb), []):
                    dep_scenario = dep_scenarios[i_dep]
                    unique_index = dep_scenario.unique_scenario_idx[i_input]
                    dep_status_combs = dep_scenario.applied_status_combinations[unique_index]

                    combs_product = self._cartesian_product(combs_product, np.column_stack(dep_status_combs))
                    nodes_idx += _process_choice_opt_node_idx(dep_scenario)

                combs_status_mat_i = np.tile(base_status, (combs_product.shape[0], 1))
                combs_status_mat_i[:, nodes_idx] = combs_product
                indep_status_combs_mat.append(combs_status_mat_i)

                all_node_idx |= set(nodes_idx)

            seen_node_idx |= all_node_idx
            all_node_idx = np.array(sorted(all_node_idx))
            if len(indep_status_combs_mat) == 0:
                indep_status_combs_mat = np.zeros((0, len(all_node_idx)), dtype=int)
            else:
                indep_status_combs_mat = np.row_stack(indep_status_combs_mat)[:, all_node_idx]

            # Get Cartesian product of independent scenarios
            if merged_cartesian_product is None:
                merged_cartesian_product = indep_status_combs_mat
            else:
                merged_cartesian_product = self._cartesian_product(merged_cartesian_product, indep_status_combs_mat)
            merged_all_nodes_idx += list(all_node_idx)

        # Merge shared node indices
        merged_status_array = np.tile(base_status, (merged_cartesian_product.shape[0], 1))
        merged_status_array[:, merged_all_nodes_idx] = merged_cartesian_product

        shared_nodes = set()
        for i_ref, i_shared in shared_node_idx.items():
            shared_idx = np.array([i_ref]+i_shared)
            shared_status = merged_status_array[:, shared_idx]

            is_removed = np.any(shared_status == Diag.REMOVED.value, axis=1)
            shared_status[is_removed, :] = Diag.REMOVED.value

            is_confirmed = np.any(shared_status == Diag.CONFIRMED.value, axis=1)
            is_confirmed[is_removed] = False
            shared_status[is_confirmed, :] = Diag.CONFIRMED.value
            merged_status_array[:, shared_idx] = shared_status

            shared_nodes |= set(i_shared)

        # Apply statuses of non-choice nodes
        other_node_idx = np.ones((len(base_status),), dtype=bool)
        other_node_idx[merged_all_nodes_idx] = False
        other_node_idx = np.where(other_node_idx)[0]
        other_status_array = merged_status_array[:, other_node_idx]

        confirmed_status = Diag.CONFIRMED.value
        removed_status = Diag.REMOVED.value
        is_initial_mask = base_status[other_node_idx] == Diag.INITIAL.value
        for i_node in merged_all_nodes_idx:
            node_is_applied = np.where(merged_status_array[:, i_node] == confirmed_status)[0]
            if len(node_is_applied) == 0:
                continue
            other_influence = base_matrix[i_node, other_node_idx]

            is_removed = is_initial_mask & (other_influence == OffDiag.REMOVAL.value)
            other_status_array[np.ix_(node_is_applied, is_removed)] = removed_status

            is_confirmed = np.where(is_initial_mask & (other_influence == OffDiag.CONFIRMATION.value))[0]
            apply = np.empty((len(node_is_applied), len(is_confirmed)), dtype=int)
            apply.fill(confirmed_status)
            apply[other_status_array[np.ix_(node_is_applied, is_confirmed)] == removed_status] = removed_status
            other_status_array[np.ix_(node_is_applied, is_confirmed)] = apply

        merged_status_array[:, other_node_idx] = other_status_array

        return merged_status_array

    def _create_existence_array_brute_force(self):
        diag_nodes = self._matrix_diagonal_nodes
        node_idx = self._matrix_diagonal_nodes_idx
        opt_node_map = self.selection_choice_option_nodes

        base_matrix = self._influence_base_matrix

        opt_invalid_status = {Diag.REMOVED.value, Diag.INFEASIBLE_OPTION.value}
        opt_idx_map = {dec_node: [node_idx[opt_node] for opt_node in opt_nodes]
                       for dec_node, opt_nodes in opt_node_map.items()}

        initial_status = Diag.INITIAL.value
        confirmed_status = Diag.CONFIRMED.value
        removed_status = Diag.REMOVED.value
        choice_made_status = Diag.CHOICE_MADE.value

        # Get decision node dependency
        choice_nodes = self.selection_choice_nodes
        n_choice_nodes = len(choice_nodes)
        i_choice_nodes = sorted([node_idx[dec_node] for dec_node in choice_nodes])

        # Get option-decision influence map
        confirmation_influence_map = {}
        removal_influence_map = {}
        for opt_nodes in opt_node_map.values():
            for opt_node in opt_nodes:
                i_opt_ = node_idx[opt_node]
                confirmation_influence_map[i_opt_] = base_matrix[i_opt_, :] == OffDiag.CONFIRMATION.value
                removal_influence_map[i_opt_] = base_matrix[i_opt_, :] == OffDiag.REMOVAL.value

        def _take_next(opt_idx_array: np.ndarray, status_array: np.ndarray) \
                -> Tuple[List[np.ndarray], List[np.ndarray]]:
            # Get next choice to make: starting from left, the first choice with confirmed status
            choice_make_idx = np.where(status_array[i_choice_nodes] == confirmed_status)[0]

            # Check if there are any choices left
            if len(choice_make_idx) == 0:
                # In the end, we're only really interested in the node statuses, i.e. the matrix diagonal
                return [opt_idx_array.copy()], [status_array.copy()]

            choice_node_idx = choice_make_idx[0]
            i_choice = i_choice_nodes[choice_node_idx]
            choice_node = choice_nodes[choice_node_idx]

            # Get option nodes
            i_opt_nodes = _get_opt_nodes(status_array, choice_node)
            if len(i_opt_nodes) == 0:
                # If there are no options left, this architecture is infeasible
                return [], []

            # Branch into the different options
            branched_opt_idx_arrays = []
            branched_status_arrays = []
            for opt_idx, i_opt in i_opt_nodes:
                branched_opt_idx, branched_status = \
                    _apply_option(opt_idx_array, status_array, i_choice, i_opt, opt_idx, choice_node_idx)
                if branched_status is None:
                    continue
                next_opt_idx_arrays, next_status_arrays = _take_next(branched_opt_idx, branched_status)
                branched_opt_idx_arrays += next_opt_idx_arrays
                branched_status_arrays += next_status_arrays

            return branched_opt_idx_arrays, branched_status_arrays

        def _get_opt_nodes(status_array: np.ndarray, choice_node: ADSGNode) -> List[Tuple[int, int]]:
            if not isinstance(choice_node, SelectionChoiceNode):
                return []
            return [(opt_idx, i_opt) for opt_idx, i_opt in enumerate(opt_idx_map[choice_node])
                    if status_array[i_opt] not in opt_invalid_status]

        def _apply_option(opt_idx_array: np.ndarray, status_array: np.ndarray, i_choice: int, i_opt: int, opt_idx: int,
                          choice_node_idx: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
            opt_idx_array: np.ndarray = opt_idx_array.copy()
            status_array: np.ndarray = status_array.copy()

            # Apply the choice
            _do_apply_sel_choice(status_array, i_choice, i_opt)
            opt_idx_array[choice_node_idx] = opt_idx

            # Apply any remaining choices with only one option
            for other_choice_idx in range(n_choice_nodes):
                if other_choice_idx == choice_node_idx:
                    continue

                # Check if this decision is active
                i_choice_other = i_choice_nodes[other_choice_idx]
                if status_array[i_choice_other] != confirmed_status:
                    continue

                # Check if it is a decision with zero or one options left
                i_sel_choice = _get_opt_nodes(status_array, diag_nodes[i_choice_other])
                if len(i_sel_choice) <= 1:
                    if len(i_sel_choice) == 0:
                        # Infeasible architecture: node without options detected
                        return None, None

                    opt_idx_apply, i_opt = i_sel_choice[0]
                    _do_apply_sel_choice(status_array, i_choice_other, i_opt)
                    opt_idx_array[other_choice_idx] = opt_idx_apply

            return opt_idx_array, status_array

        def _do_apply_sel_choice(status_array: np.ndarray, i_choice_apply, i_opt_apply):
            # Apply confirmation and removal influences
            initial_mask = status_array == initial_status
            for influence_map, diag_value in [(confirmation_influence_map, confirmed_status),
                                              (removal_influence_map, removed_status)]:
                influence_mask = influence_map[i_opt_apply] & initial_mask
                status_array[influence_mask] = diag_value

            # Mark choice as made
            status_array[i_choice_apply] = choice_made_status
            status_array[i_opt_apply] = confirmed_status

        # Generate matrix alternatives
        base_opt_idx_array = -np.ones((n_choice_nodes,), dtype=int)
        base_status_array = np.diag(base_matrix)
        opt_idx_array_list, status_array_list = _take_next(base_opt_idx_array, base_status_array)

        if len(status_array_list) == 0:
            opt_idx_matrix = np.zeros((0, n_choice_nodes), dtype=int)
            status_matrix = np.zeros((0, base_matrix.shape[0]), dtype=int)
        else:
            opt_idx_matrix = np.array(opt_idx_array_list, dtype=int)
            status_matrix = np.array(status_array_list, dtype=int)
        return opt_idx_matrix, status_matrix

    def _assert_behavior(self):
        graph = super()._assert_behavior()
        self._assert_iter_spec()
        return graph

    def _assert_iter_spec(self):
        sel_choice_idx = self.get_choice_option_indices()

        iter_choice_idx = np.empty(sel_choice_idx.shape, dtype=int)
        iter_choice_idx.fill(X_INACTIVE_VALUE)
        for iter_spec in self._scenario_iter_spec:
            assert iter_spec.n_total == sel_choice_idx.shape[0]
            scenario = iter_spec.scenario

            opt_idx_comb = scenario.opt_idx_combinations[iter_spec.i_usi][iter_spec.i_comb, :]
            idx_apply = np.array(list(iter_spec))
            assert len(idx_apply) > 0
            iter_choice_idx[np.ix_(idx_apply, scenario.choice_idx)] = opt_idx_comb

            idx_apply_set = set(idx_apply)
            for i in range(iter_spec.n_total):
                assert (i in iter_spec) == (i in idx_apply_set)

        assert np.all(sel_choice_idx == iter_choice_idx)

        for i, n_opts in enumerate(self.n_opts):
            for j in range(n_opts):
                value_mask = sel_choice_idx[:, i] == j
                assert np.all(value_mask == self.get_available_combinations_mask({i: j}))
