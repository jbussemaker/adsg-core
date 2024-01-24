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
import enum
import itertools
import numpy as np
from typing import *
from dataclasses import dataclass
from adsg_core.graph.adsg_nodes import *
from adsg_core.graph.incompatibility import *
from adsg_core.graph.choice_constraints import *
from adsg_core.graph.traversal import traverse_until_choice_nodes
from cached_property import cached_property
if False:
    from adsg_core.graph.adsg import ADSGType

__all__ = ['InfluenceMatrix', 'Diag', 'OffDiag', 'StatusScenarios', 'SelectionChoiceScenarios', 'X_INACTIVE_VALUE',
           'X_INACTIVE_IMPUTE']

X_INACTIVE_VALUE = -1
X_INACTIVE_IMPUTE = 0


class Diag(enum.Enum):
    INITIAL = 0  # Non-confirmed initially
    CONFIRMED = 1  # Confirmed
    REMOVED = 2  # Removed
    INFEASIBLE_OPTION = 3  # The option would lead to an infeasible architecture
    CHOICE_MADE = 4  # The decision has been taken (i.e. is not in the graph anymore)


class OffDiag(enum.Enum):
    NO_INTERACTION = 0
    CONFIRMATION = 1  # The option-node at i confirms the existence of the node at j if chosen
    REMOVAL = 2  # The option-node at i removes the node at j if chosen


@dataclass
class StatusScenarios:
    """
    Dataclass representing node status scenarios.
    A status scenario represents all possible combinations of node statuses (e.g. non-confirmed, confirmed, removed)
    given all possible input statuses. Input statuses are determined by non-0 coupling values in the influence matrix.
    """

    # SCENARIO INFO
    nodes: List[ADSGNode]  # Nodes this status scenario applies to
    nodes_idx: np.ndarray  # [n_nodes; int] Index of the nodes in the influence matrix

    # INPUT INFLUENCES
    n_scenarios: int  # Nr of input scenarios
    influence_input_node_idx: np.ndarray  # [n_inputs; int] Input node indices (in the influence matrix)
    # Input status matrices, True represents that the input node is selected in the graph, so its influences are applied
    input_status_matrix: np.ndarray  # [n_scenarios x n_inputs; bool]
    unique_scenario_idx: np.ndarray  # [n_scenarios; int] Mapping from input scenario to node scenario idx

    # NODE STATUSES
    n_unique_scenarios: int  # Nr of unique node status scenarios
    status_matrix: np.ndarray  # [n_unique_scenarios x n_nodes; Diag] Node status scenarios


@dataclass
class SelectionChoiceScenarios:
    """
    Dataclass representing status scenarios specific to a selection choice.
    """

    # CHOICE SCENARIO INFO
    choice_nodes: List[SelectionChoiceNode]  # Choice nodes represented by this object
    choice_idx: np.ndarray  # [n_nodes; int] Choice indices (in selection_choice_nodes list)
    nodes_idx: np.ndarray  # [n_nodes; int] Choice node indices (in the influence matrix)
    n_opts: np.ndarray  # [n_nodes; int] Nr of options for each choice
    opt_node_idx: List[np.ndarray]  # [n_nodes -> n_opts; int] Node idx of each option node (in influence matrix)
    unshared_opt_node_idx: List[np.ndarray]  # [n_nodes -> n_opts; int] Idem. but assuming no option nodes are shared

    # INPUT INFLUENCES
    input_node_idx: np.ndarray  # [n_inputs; int] Input option node indices (in the influence matrix)
    input_choice_idx: np.ndarray  # [n_inputs; int] Input choice indices (in selection_choice_nodes list)
    input_opt_idx: np.ndarray  # [n_inputs; int] Input option node indices (wrt choice)
    # Input status matrices, True represents that the input node is selected in the graph, so its influences are applied
    input_status_matrix: np.ndarray  # [n_scenarios x n_inputs; bool]
    unique_scenario_idx: np.ndarray  # [n_scenarios; int] Mapping from input scenario to choice activation scenario

    # OPTION SELECTION INFLUENCE
    n_combinations: np.ndarray  # [n_scenarios; int] Nr of option combinations for each input scenario
    n_total: int  # Total nr of combinations for all input scenarios (sum of n_combinations)
    n_combinations_unique: np.ndarray  # [n_unique_scenarios; int] Nr of option combs for choice activation scenarios
    # Status scenarios before choice selection, for each choice activation scenario, for each choice node, the initial
    # status matrix for 1+n_opts (choice node + option nodes); specifying choice activeness and option availability
    status_matrix: List[List[np.ndarray]]  # [n_unique_scenarios -> n_nodes -> 1+n_opts; Diag]
    # Option-index combinations, for each choice activation scenario, a matrix of n_combinations x n_nodes; specifying
    # in each row a possible option-selection combination, the values are selected option-indices wrt the choice
    opt_idx_combinations: List[np.ndarray]  # [n_unique_scenarios -> n_combs x n_nodes; int]
    # Idem. but values are selected node indices in the influence matrix
    node_idx_combinations: List[np.ndarray]  # [n_unique_scenarios -> n_combs x n_nodes; int]
    # Applied status combinations, representing node status after applying the option-index combination, for each choice
    # activation scenario, for each choice node
    applied_status_combinations: List[List[np.ndarray]]  # [n_unique_scenarios -> n_nodes -> n_combs x 1+n_opts; int]


class InfluenceMatrix:
    """
    Creates an influence matrix, specifying which option-decision node influence the existence of other nodes.
    The matrix is structured with the following elements on the diagonal:
    - For each option-decision (in order of descending importance)
      - The decision node
      - The option nodes (in order)
    - All other nodes
    """

    def __init__(self, adsg: 'ADSGType', remove_duplicate_nodes=False):
        if adsg.derivation_start_nodes is None or not adsg.feasible:
            raise ValueError('Provide a feasible graph with external functions!')
        self._remove_duplicate_nodes = remove_duplicate_nodes
        self._adsg = adsg

    @property
    def adsg(self) -> 'ADSGType':
        return self._adsg

    @cached_property
    def permanent_nodes(self) -> Set[ADSGNode]:
        adsg = self.adsg
        start_nodes = adsg.derivation_start_permanent_nodes
        permanent_nodes, _ = traverse_until_choice_nodes(adsg.graph, start_nodes)
        return permanent_nodes

    @cached_property
    def permanent_nodes_incl_choice_nodes(self) -> Set[ADSGNode]:
        adsg = self.adsg
        start_nodes = adsg.derivation_start_permanent_nodes
        permanent_nodes, init_decision_nodes = traverse_until_choice_nodes(adsg.graph, start_nodes)
        return permanent_nodes | init_decision_nodes

    @cached_property
    def choice_nodes(self) -> List[ChoiceNode]:
        adsg = self.adsg
        graph = adsg.graph

        decision_nodes = []
        start_nodes = adsg.derivation_start_nodes
        traversed = set()
        while True:
            # From the current starting nodes, get the next decision nodes
            non_dec_nodes, next_decision_nodes = traverse_until_choice_nodes(graph, start_nodes, traversed=traversed)
            if len(next_decision_nodes) == 0:
                break

            # Order the decision nodes
            decision_nodes += adsg.ordered_choice_nodes(next_decision_nodes)

            # Move to next starting nodes
            traversed |= set(non_dec_nodes) | set(next_decision_nodes)
            start_nodes = next_decision_nodes

        return decision_nodes

    @cached_property
    def selection_choice_nodes(self) -> List[SelectionChoiceNode]:
        return [choice_node for choice_node in self.choice_nodes if isinstance(choice_node, SelectionChoiceNode)]

    @cached_property
    def selection_choice_option_nodes(self) -> Dict[SelectionChoiceNode, List[ADSGNode]]:
        return {choice_node: self.adsg.get_option_nodes(choice_node) for choice_node in self.selection_choice_nodes}

    @cached_property
    def _sel_choice_influence(self) \
            -> Dict[SelectionChoiceNode, Dict[ADSGNode, Tuple[Set[ADSGNode], Set[ADSGNode], bool]]]:

        cache = {}

        def _get_confirmed_removed_nodes(sel_choice_node: SelectionChoiceNode, option_node: ADSGNode):
            # Confirmed nodes: nodes directly confirmed by this option node
            #  --> (i.e. nodes that will exist if this option node is chosen)
            # Removed nodes: nodes directly removed by this option node
            try:
                confirmed_edges, removed_nodes =\
                    graph.get_confirmed_edges_selection_choice(sel_choice_node, option_node, include_choice=True,
                                                               included_apply_edges=False, cache=cache)
                confirmed_nodes = {node for edge in confirmed_edges for node in edge[:2]}
                confirmed_nodes.add(option_node)
                is_feasible = True

            except IncompatibilityError:
                confirmed_nodes, removed_nodes, is_feasible = set(), set(), False

            return confirmed_nodes, removed_nodes, is_feasible

        graph = self.adsg
        return {node: {option: _get_confirmed_removed_nodes(node, option) for option in options}
                for node, options in self.selection_choice_option_nodes.items()}

    @cached_property
    def other_nodes(self) -> List[ADSGNode]:
        decision_nodes = set(self.selection_choice_nodes) | \
                         {node for opt_nodes in self.selection_choice_option_nodes.values() for node in opt_nodes}
        return [node for node in self.adsg.graph.nodes if node not in decision_nodes]

    @cached_property
    def matrix_diagonal_nodes(self) -> List[ADSGNode]:
        diagonal_nodes = []

        node_idx_map = {}
        sel_choice_opt_nodes = self.selection_choice_option_nodes
        for sel_choice_node in self.selection_choice_nodes:
            node_idx_map[sel_choice_node] = len(diagonal_nodes)
            diagonal_nodes.append(sel_choice_node)
            for opt_node in sel_choice_opt_nodes[sel_choice_node]:
                node_idx_map[opt_node] = len(diagonal_nodes)
                diagonal_nodes.append(opt_node)

        if self._remove_duplicate_nodes:
            keep_idx = sorted(node_idx_map.values())
            diagonal_nodes = [node for i, node in enumerate(diagonal_nodes) if i in keep_idx]

        diagonal_nodes += self.other_nodes
        return diagonal_nodes

    @cached_property
    def matrix_diagonal_nodes_idx(self) -> Dict[ADSGNode, int]:
        return {node: i for i, node in enumerate(self.matrix_diagonal_nodes)}

    @cached_property
    def choice_idx(self) -> np.ndarray:
        idx_map = self.matrix_diagonal_nodes_idx
        return np.array([idx_map[sel_choice_node] for sel_choice_node in self.choice_nodes])

    @cached_property
    def influence_matrix(self) -> np.ndarray:
        """The matrix of n_nodes x n_nodes specifying initial node status and node influence"""
        return self._create_influence_matrix()

    @cached_property
    def influence_matrix_no_choice_constraints(self) -> np.ndarray:
        """The matrix of n_nodes x n_nodes specifying initial node status and node influence"""
        return self._create_influence_matrix(apply_choice_constraints=False)

    @cached_property
    def init_status_array(self) -> np.ndarray:
        """Get the initial status array (before any decisions have been taken)"""
        matrix = self.influence_matrix
        i_diag = np.arange(matrix.shape[0])
        return matrix[i_diag, i_diag]

    def get_next_choice_nodes(self, status_array: np.ndarray) -> List[ChoiceNode]:
        if len(self.choice_idx) == 0:
            return []
        active_choice_node_idx = np.where(status_array[self.choice_idx] == Diag.CONFIRMED.value)[0]
        choice_nodes = self.choice_nodes
        return self.adsg.ordered_choice_nodes([choice_nodes[idx] for idx in active_choice_node_idx])

    def apply_selection_choice(self, status_array: np.ndarray, choice_node: SelectionChoiceNode,
                               option_node: Optional[ADSGNode]) -> np.ndarray:

        status_array = status_array.copy()
        i_choice_apply = self.matrix_diagonal_nodes_idx[choice_node]
        i_opt_apply = self.matrix_diagonal_nodes_idx[option_node] if option_node is not None else None

        # Mark which nodes are already confirmed
        not_confirmed_mask = status_array == Diag.INITIAL.value

        # Apply statuses for confirmation and removal influences
        if i_opt_apply is not None:
            matrix = self.influence_matrix
            confirmation_influence_map = matrix[i_opt_apply, :] == OffDiag.CONFIRMATION.value
            status_array[confirmation_influence_map & not_confirmed_mask] = Diag.CONFIRMED.value

            removal_influence_map = matrix[i_opt_apply, :] == OffDiag.REMOVAL.value
            status_array[removal_influence_map & not_confirmed_mask] = Diag.REMOVED.value

        # Mark choice as made
        status_array[i_choice_apply] = Diag.CHOICE_MADE.value
        if i_opt_apply is not None:
            status_array[i_opt_apply] = Diag.CONFIRMED.value

        return status_array

    def apply_choice(self, status_array: np.ndarray, decision_node: ChoiceNode, copy=True) -> np.ndarray:
        if copy:
            status_array = status_array.copy()

        i_choice_apply = self.matrix_diagonal_nodes_idx[decision_node]
        status_array[i_choice_apply] = Diag.CHOICE_MADE.value
        return status_array

    def _create_influence_matrix(self, apply_choice_constraints=True) -> np.ndarray:
        # Initialize matrix
        diag_nodes = self.matrix_diagonal_nodes
        node_idx = self.matrix_diagonal_nodes_idx
        matrix = np.zeros((len(diag_nodes), len(diag_nodes)), dtype=int)

        # Mark permanent nodes and initial decisions
        permanent_nodes = self.permanent_nodes_incl_choice_nodes
        for node in permanent_nodes:
            i_node = node_idx[node]
            matrix[i_node, i_node] = Diag.CONFIRMED.value

        # Loop over choices
        opt_node_map = self.selection_choice_option_nodes
        influence_map = self._sel_choice_influence
        for choice_node in self.selection_choice_nodes:

            # Determine hierarchical influence of all option nodes
            for opt_node in opt_node_map[choice_node]:
                i_opt = node_idx[opt_node]

                opt_confirmed_nodes, opt_removed_nodes, opt_is_feasible = influence_map[choice_node][opt_node]

                # Set feasibility flag
                if not opt_is_feasible:
                    matrix[i_opt, i_opt] = Diag.INFEASIBLE_OPTION.value
                    continue

                # Do not confirm permanent nodes
                opt_confirmed_nodes -= permanent_nodes

                # Set influence flags
                for nodes, flag in [(opt_confirmed_nodes, OffDiag.CONFIRMATION.value),
                                    (opt_removed_nodes, OffDiag.REMOVAL.value)]:
                    for node in nodes:
                        # Only if it is not referring to the option node
                        j_removed_node = node_idx[node]
                        if j_removed_node == i_opt:
                            continue
                        matrix[i_opt, j_removed_node] = flag

        # Apply choice constraints
        if apply_choice_constraints:
            other_nodes = self.other_nodes
            for choice_constraint in self.adsg.get_choice_constraints():
                for i_dec, choice_node in enumerate(choice_constraint.nodes):
                    if not isinstance(choice_node, SelectionChoiceNode):
                        break

                    for i_opt, opt_node in enumerate(choice_constraint.options[i_dec]):
                        if opt_node not in node_idx:
                            continue
                        i_src = node_idx[opt_node]
                        for _, removed_opt_nodes in get_constraint_removed_options(choice_constraint, i_dec, i_opt):
                            for removed_opt_node in removed_opt_nodes:
                                if removed_opt_node not in node_idx or removed_opt_node in other_nodes:
                                    continue
                                i_tgt = node_idx[removed_opt_node]
                                matrix[i_src, i_tgt] = OffDiag.REMOVAL.value

        # Clear feedback influences from non-choice nodes to choice nodes
        i_other = np.array(sorted([node_idx[node] for node in self.other_nodes]), dtype=int)
        i_choice_nodes = np.ones((len(diag_nodes),), dtype=bool)
        i_choice_nodes[i_other] = False
        i_choice_nodes = np.where(i_choice_nodes)[0]
        matrix[np.ix_(i_other, i_choice_nodes)] = OffDiag.NO_INTERACTION.value

        return matrix

    @cached_property
    def base_sel_choice_scenarios(self):
        """
        Get the selection choice scenarios without merging.
        """
        node_idx_map = self.matrix_diagonal_nodes_idx
        opt_node_map = self.selection_choice_option_nodes

        choice_idx_map = {node: i_choice for i_choice, node in enumerate(self.selection_choice_nodes)}
        opt_idx_map = {node_idx_map[opt_node]: (choice_idx_map[choice_node], i_opt)
                       for choice_node, opt_nodes in opt_node_map.items() for i_opt, opt_node in enumerate(opt_nodes)}

        # Loop over each choice node
        choice_scenarios = []
        for i_choice, sel_choice_node in enumerate(self.selection_choice_nodes):
            node_idx = node_idx_map[sel_choice_node]

            opt_nodes = opt_node_map[sel_choice_node]
            opt_nodes_idx = np.array([node_idx_map[opt_node] for opt_node in opt_nodes])
            unshared_opt_nodes_idx = opt_nodes_idx
            unshared_opt_nodes_idx = np.arange(len(unshared_opt_nodes_idx))+node_idx+1

            choice_opt_nodes: List[ADSGNode] = [sel_choice_node]
            choice_opt_nodes += opt_nodes

            # Get status scenarios for the choice node and option nodes
            status_scenario = self.get_status_scenarios(choice_opt_nodes)
            input_node_idx = status_scenario.influence_input_node_idx
            if len(input_node_idx) > 0:
                input_choice_idx, input_opt_idx = zip(*[opt_idx_map[node_idx] for node_idx in input_node_idx])
                input_choice_idx = np.array(input_choice_idx)
                input_opt_idx = np.array(input_opt_idx)
            else:
                input_choice_idx = np.array([], dtype=int)
                input_opt_idx = np.array([], dtype=int)
            input_status_matrix = status_scenario.input_status_matrix
            unique_scenario_idx = status_scenario.unique_scenario_idx

            # For each status scenario, determine whether the choice is active
            status_matrix = status_scenario.status_matrix
            choice_is_active = np.array(status_matrix[:, 0] == Diag.CONFIRMED.value)

            # Remove duplicate choice-inactive scenarios
            is_inactive = np.where(~choice_is_active)[0]
            if len(is_inactive) > 1:
                # Ignore removal interactions for choice-inactive scenarios
                status_matrix = status_matrix.copy()
                status_matrix_inactive = status_matrix[~choice_is_active, 1:]
                status_matrix_inactive[status_matrix_inactive == OffDiag.REMOVAL.value] = OffDiag.NO_INTERACTION.value
                status_matrix[~choice_is_active, 1:] = status_matrix_inactive

                # Get unique remaining scenarios
                status_matrix, selected_idx, selected_scenario_idx = \
                    np.unique(status_matrix, axis=0, return_inverse=True, return_index=True)
                choice_is_active = choice_is_active[selected_idx]
                unique_scenario_idx = selected_scenario_idx[unique_scenario_idx]

            # For each status scenario determine which options are available
            opt_is_available = ~((status_matrix[:, 1:] == Diag.REMOVED.value) |
                                 (status_matrix[:, 1:] == Diag.INFEASIBLE_OPTION.value))

            # Create option idx combinations
            opt_idx_combinations = []
            node_idx_combinations = []
            applied_status_combinations = []
            for i_stat, is_active in enumerate(choice_is_active):
                applied_status = status_matrix[[i_stat], :].copy()

                if not is_active:
                    opt_idx_combinations.append(np.array([[X_INACTIVE_VALUE]], dtype=int))
                    node_idx_combinations.append(np.array([[X_INACTIVE_VALUE]], dtype=int))
                    applied_status_combinations.append([applied_status])

                else:
                    opt_avail = np.where(opt_is_available[i_stat, :])[0]
                    opt_idx_combinations.append(np.array([opt_avail]).T)
                    node_idx_combinations.append(np.array([opt_nodes_idx[opt_avail]]).T)

                    # Mark choice as made
                    applied_status[0, 0] = Diag.CHOICE_MADE.value

                    # Mark chosen option nodes as selected (confirmed)
                    applied_status = np.repeat(applied_status, repeats=len(opt_avail), axis=0)
                    for i_app, applied_idx in enumerate(opt_avail):
                        applied_status[i_app, applied_idx+1] = Diag.CONFIRMED.value
                    applied_status_combinations.append([applied_status])

            n_combinations_unique = np.array([comb.shape[0] for comb in opt_idx_combinations])
            n_combinations = n_combinations_unique[unique_scenario_idx]

            choice_scenarios.append(SelectionChoiceScenarios(
                choice_nodes=[sel_choice_node],
                choice_idx=np.array([i_choice]),
                nodes_idx=np.array([node_idx]),
                n_opts=np.array([len(opt_nodes)]),
                opt_node_idx=[opt_nodes_idx],
                unshared_opt_node_idx=[unshared_opt_nodes_idx],
                input_node_idx=input_node_idx,
                input_choice_idx=input_choice_idx,
                input_opt_idx=input_opt_idx,
                input_status_matrix=input_status_matrix,
                unique_scenario_idx=unique_scenario_idx,
                n_combinations=n_combinations,
                n_total=int(np.sum(n_combinations)),
                status_matrix=[[row] for row in status_matrix],
                n_combinations_unique=n_combinations_unique,
                opt_idx_combinations=opt_idx_combinations,
                node_idx_combinations=node_idx_combinations,
                applied_status_combinations=applied_status_combinations,
            ))
        return choice_scenarios

    def get_status_scenarios(self, nodes: List[ADSGNode]) -> StatusScenarios:
        """
        Get independent status scenarios for given nodes. A status scenario is a set of status for the given nodes given
        some input influence (confirmation, removal) statuses.
        """
        node_idx_map = self.matrix_diagonal_nodes_idx
        nodes_idx = np.array([node_idx_map[node] for node in nodes])
        influence_matrix = self.influence_matrix_no_choice_constraints

        no_interaction = OffDiag.NO_INTERACTION.value
        confirms_interaction = OffDiag.CONFIRMATION.value
        removes_interaction = OffDiag.REMOVAL.value
        initial_status = Diag.INITIAL.value
        confirmed_status = Diag.CONFIRMED.value
        removed_status = Diag.REMOVED.value

        # Get input influences
        matrix_inputs = influence_matrix[:, nodes_idx].copy()
        matrix_inputs[nodes_idx, np.arange(len(nodes_idx))] = no_interaction
        influence_input_node_idx = np.where(np.any(matrix_inputs != no_interaction, axis=1))[0]
        is_internal = np.zeros((matrix_inputs.shape[0],), dtype=bool)
        is_internal[nodes_idx] = True
        permanent_nodes = np.diag(influence_matrix) == Diag.CONFIRMED.value

        # If there are no influences, there is one scenario: the starting scenario
        starting_status = np.diag(influence_matrix)[nodes_idx]
        if len(influence_input_node_idx) == 0:
            status_matrix = np.array([starting_status])
            input_status_matrix = np.zeros((1, 0), dtype=bool)

        else:
            # Get the Cartesian product of all input influence existence scenarios
            status_matrix = []
            input_status_matrix = []
            for input_exists in itertools.product(*[[True] if permanent_nodes[node_idx] else [False, True]
                                                    for node_idx in influence_input_node_idx]):

                input_exists_arr = np.array(input_exists)
                input_status_matrix.append(input_exists_arr)

                # Apply the input influences to modify the node statuses
                node_status = starting_status.copy()
                selected_inputs = influence_input_node_idx[input_exists_arr]
                applied_influences = matrix_inputs[selected_inputs, :]
                applied_is_internal = is_internal[selected_inputs]
                if applied_influences.shape[0] > 0:
                    is_initial_mask = starting_status == initial_status

                    # First apply internal confirming influences (relevant if selecting one option node also confirms
                    # another option node, e.g. for component instantiation)
                    is_confirmed = is_initial_mask & \
                        np.any(applied_influences[applied_is_internal] == confirms_interaction, axis=0)
                    node_status[is_confirmed] = confirmed_status

                    # Apply external influences
                    is_removed = is_initial_mask & ~is_confirmed & \
                        np.any(applied_influences == removes_interaction, axis=0)
                    node_status[is_removed] = removed_status

                    is_confirmed = is_initial_mask & ~is_removed & \
                        np.any(applied_influences == confirms_interaction, axis=0)
                    node_status[is_confirmed] = confirmed_status

                status_matrix.append(node_status)

            status_matrix = np.array(status_matrix)
            input_status_matrix = np.array(input_status_matrix)

        # Get unique scenarios
        status_matrix, unique_scenario_idx = np.unique(status_matrix, axis=0, return_inverse=True)

        return StatusScenarios(
            nodes=nodes,
            nodes_idx=nodes_idx,
            n_scenarios=len(unique_scenario_idx),
            n_unique_scenarios=status_matrix.shape[0],
            influence_input_node_idx=influence_input_node_idx,
            input_status_matrix=input_status_matrix,
            unique_scenario_idx=unique_scenario_idx,
            status_matrix=status_matrix,
        )
