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
import networkx as nx
from adsg_core.graph.export import *
from adsg_core.graph.traversal import *
from adsg_core.graph.adsg_nodes import *
from adsg_core.graph.graph_edges import *
from adsg_core.graph.choices import *
from adsg_core.graph.incompatibility import *
from adsg_core.graph.influence_matrix import *
from adsg_core.graph.choice_constraints import *

__all__ = ['ADSG', 'EdgeType', 'CDVNode', 'ChoiceConstraint', 'ChoiceConstraintType', 'ADSGType']


class ADSG:
    """
    The Architecture Design Space Graph (ADSG). Represent the architecture design space, its choices, and contains
    functions for manipulating the graph.

    If you actually want to use the ADSG directly, you can use the BasicADSG class in adsg_basic.
    """

    _taken_single_choices = []

    def __init__(self, _graph=None, _influence_matrix=None, _status_array=None, _choice_con_map=None,
                 _des_var_values=None, **_):
        self._graph = _graph or self._get_empty_graph()
        self._choice_constraints: List[ChoiceConstraint] = _choice_con_map or []
        self._influence_matrix: Optional[InfluenceMatrix] = _influence_matrix
        self._status_array: Optional[np.ndarray] = _status_array

        self._update_connector_grouping_degrees()
        self._des_var_values: Dict[DesignVariableNode, Union[float, int]] = _des_var_values or {}

    @staticmethod
    def _get_empty_graph():
        graph = nx.MultiDiGraph()
        # We need hashable edge attribute dicts to be able to use edges in sets
        graph.edge_attr_dict_factory = HashableDict
        return graph

    def __hash__(self):
        start_nodes = tuple(sorted(self.derivation_start_nodes, key=lambda n: n.name)) \
            if self.derivation_start_nodes else None

        g = self.graph
        node_hashes = tuple(sorted([hash(node) for node in g.nodes]))
        edge_hashes = tuple(sorted([hash(edge) for edge in g.edges]))

        constraints_hashes = tuple(hash(c) for c in self._choice_constraints)

        return hash((start_nodes, node_hashes, edge_hashes, constraints_hashes))

    def __eq__(self, other):
        return hash(self) == hash(other)

    def fingerprint(self):
        """Like a hash but supports pickled nodes (note: not all node properties are compared!)"""

        def _node_fingerprint(node: ADSGNode):
            return hash(node.str_context())

        def _edge_fingerprint(edge):
            return hash(tuple(_node_fingerprint(v) if i < 2 else v for i, v in enumerate(edge)))

        nodes_fingerprints = hash(tuple(sorted(_node_fingerprint(n) for n in self.graph.nodes)))
        edges_fingerprints = hash(tuple(sorted(_edge_fingerprint(e) for e in self.graph.edges)))

        start_fp = hash(tuple(sorted(_node_fingerprint(n) for n in (self.derivation_start_nodes or []))))
        constraint_fps = hash(tuple(hash((
            cc.type.name,
            tuple(_node_fingerprint(n) for n in cc.nodes),
            None if cc.options is None else tuple(_node_fingerprint(v) if isinstance(v, ADSGNode) else v
                                                  for opt_list in cc.options for v in opt_list),
        )) for cc in self._choice_constraints))

        return hash((start_fp, nodes_fingerprints, edges_fingerprints, constraint_fps))

    def is_same(self, other: 'ADSGType') -> bool:
        """Compare based on node export str and edge types (supports pickled nodes)"""

        if len(self.graph.nodes) != len(other.graph.nodes):
            return False
        if len(self.graph.edges) != len(other.graph.edges):
            return False

        return self.fingerprint() == other.fingerprint()

    """#####################################
    ### DERIVATION-START NODES FUNCTIONS ###
    #####################################"""

    @property
    def derivation_start_nodes(self) -> Optional[Set[ADSGNode]]:
        """Set of nodes (if set) where node and choice derivation starts"""
        return self._get_derivation_start_nodes()

    @property
    def derivation_start_permanent_nodes(self) -> Optional[Set[ADSGNode]]:
        derivation_start = self.derivation_start_nodes
        if derivation_start is None:
            return

        permanent_nodes = self._get_permanent_nodes()
        if permanent_nodes is not None:
            return derivation_start | permanent_nodes
        return derivation_start

    def _get_derivation_start_nodes(self) -> Optional[Set[ADSGNode]]:
        raise NotImplementedError

    def _get_permanent_nodes(self) -> Optional[Set[ADSGNode]]:
        pass

    """################################
    ### UNDERLYING GRAPHS FUNCTIONS ###
    ################################"""

    @property
    def graph(self):
        """The underlying networkx directed graph"""
        return self._graph

    def export_gml(self, path):
        """Export to GML (use e.g. Gephi to view)"""
        export_gml(self._get_graph_for_export(), path)

    def export_dot(self, path):
        """Export to DOT (use Graphviz / https://viz-js.com/ to view)"""
        export_dot(self._get_graph_for_export(), path, start_nodes=self.derivation_start_nodes)

    def export_drawio(self, path):
        """Export to draw.io"""
        export_drawio(self._get_graph_for_export(), path, start_nodes=self.derivation_start_nodes)

    def render(self):
        from adsg_core.render import render
        render(self)

    def render_all(self, idx=None):
        from adsg_core.render import render_all_instances
        render_all_instances(self, idx=idx)

    def _get_graph_for_export(self):
        graph = nx.MultiDiGraph()
        graph.add_edges_from(self._graph.edges(data=True))
        graph.add_nodes_from(self._graph.nodes(data=True))
        self._export_prepare(graph)
        return graph

    def _export_prepare(self, graph: nx.MultiDiGraph):
        pass

    """#########################
    ### NODE TYPE PROPERTIES ###
    #########################"""

    @property
    def choice_nodes(self):
        return get_choice_nodes(self._graph)

    def get_nodes_by_type(self, type_):
        return get_nodes_by_type(self._graph, type_)

    def get_nodes_by_subtype(self, type_):
        return get_nodes_by_subtype(self._graph, type_)

    """#########################################
    ### ADDITIONAL DESIGN VARIABLE FUNCTIONS ###
    #########################################"""

    @property
    def all_des_var_nodes(self) -> List[DesignVariableNode]:
        """Get all design variables nodes in the graph"""
        return self.ordered_choice_nodes(self.get_nodes_by_type(DesignVariableNode))

    @property
    def des_var_nodes(self) -> List[DesignVariableNode]:
        """Get all non-constrained (non-linked) design variable nodes in the graph"""
        seen_constraint_sets = set()
        mirror_con_dvs = {node: i for i, con in enumerate(self._choice_constraints) for node in con.nodes}

        # Only take the first-seen nodes out of each design constraint set
        des_var_nodes = []
        for node in self.all_des_var_nodes:
            i_set = mirror_con_dvs.get(node)
            if i_set is not None:
                if i_set in seen_constraint_sets:
                    continue
                seen_constraint_sets.add(i_set)

            des_var_nodes.append(node)
        return des_var_nodes

    def set_des_var_value(self, des_var_node: DesignVariableNode, value: Union[float, int]):
        """
        Set the value of a design variable node.
        Discrete values are defined as an option index (so not the value of an option itself).
        """
        value, bounds_fraction = des_var_node.correct_value(value)
        self._des_var_values[des_var_node] = value

        # Set values of linked design variables
        decision_constraint = self.is_constrained_choice(des_var_node)
        if decision_constraint is not None:
            if decision_constraint.type != ChoiceConstraintType.LINKED:
                raise RuntimeError(f'Unsupported constraint type for DV {des_var_node!r}: {decision_constraint.type}')
            for linked_des_var_node in decision_constraint.nodes:
                if linked_des_var_node == des_var_node:
                    continue

                # Check that design variables are of the same type
                if linked_des_var_node.is_discrete != des_var_node.is_discrete:
                    raise ValueError(f'Dependent des vars should be same type: discrete = {des_var_node.is_discrete}')

                # Map value of taken design variable onto domain of linked design variable
                if des_var_node.is_discrete:
                    dep_value = value
                else:
                    if linked_des_var_node.bounds is None:
                        raise ValueError(f'Design variable bounds not set: {linked_des_var_node!r}')
                    dep_lower, dep_upper = linked_des_var_node.bounds
                    dep_value = dep_lower + bounds_fraction * (dep_upper - dep_lower)

                self._des_var_values[linked_des_var_node] = dep_value

    def des_var_value(self, des_var_node: DesignVariableNode) -> Optional[Union[float, int]]:
        """
        Get the value of a design variable node. If not set, returns None.
        For discrete design variables, the index of the selected option is returned.
        """
        return self._des_var_values.get(des_var_node)

    @property
    def des_var_values(self) -> Dict[DesignVariableNode, Union[float, int]]:
        return self._des_var_values.copy()

    def reset_des_var_values(self):
        self._des_var_values = {}

    """################################
    ### CHOICE CONSTRAINT FUNCTIONS ###
    ################################"""

    def is_constrained_choice(self, choice_node: CDVNode) -> Optional[ChoiceConstraint]:
        for choice_con in self._choice_constraints:
            if choice_node in choice_con.nodes:
                return choice_con

    def get_choice_constraints(self) -> List[ChoiceConstraint]:
        return self._choice_constraints

    def constrain_choices(self, constraint: ChoiceConstraintType, choice_nodes: List[CDVNode],
                          remove_infeasible_choices=True) -> 'ADSG':
        """
        Create a new choice constraint (see `ChoiceConstraintType`).
        For design variable nodes, only the LINKED constraint is available.

        Following constraints are available (`ChoiceConstraintType`):

        - `LINKED`: To make all choices have the same option index             --> AA, BB, CC
        - `PERMUTATION`: To make the choices be permutations of option indices --> AB, BA, AC, CA, BC, CB
        - `UNORDERED`: To have all option index combinations without ordering  --> AA, AB, AC, BB, BC, CC
        - `UNORDERED_NOREPL`: Same but also without replacement                --> AB, AC, BC
        """
        if len(choice_nodes) == 0:
            return self
        choice_nodes = self.ordered_choice_nodes(choice_nodes)

        # Check if any nodes are previously constrained
        constrained_nodes = {node for choice_con in self._choice_constraints for node in choice_con.nodes}
        for node in choice_nodes:
            if node in constrained_nodes:
                raise RuntimeError(f'Node is already constrained: {node}')

        # Check constraint types
        node_type = type(choice_nodes[0])
        for node in choice_nodes:
            if type(node) != node_type:
                raise ValueError(f'Choice nodes should all be of type: {node_type}')
            if isinstance(node, ConnectionChoiceNode):
                raise RuntimeError(f'Constraints not implemented for connection choices!')
            if isinstance(node, DesignVariableNode) and constraint != ChoiceConstraintType.LINKED:
                raise ValueError(f'Only LINKED constraint can be applied to DV nodes: {node}')

        # Add the constraint
        node_options = []
        is_selection_choice = False
        for node in choice_nodes:
            if isinstance(node, SelectionChoiceNode):
                is_selection_choice = True
                node_options.append(self.get_option_nodes(node))
                continue

            # if isinstance(node, ConnectionChoiceNode):
            #     node_options.append(list(node.iter_conn_permutations(self)))
            #     continue

        if len(node_options) == 0:
            node_options = None
        elif constraint in [ChoiceConstraintType.UNORDERED, ChoiceConstraintType.UNORDERED_NOREPL]:
            n_opts = len(node_options[0])
            if any([len(opts) != n_opts for opts in node_options]):
                raise ValueError(f'All choices must have the same nr of options for constraint type: {constraint}')

        decision_constraint = ChoiceConstraint(constraint, choice_nodes, node_options)
        self._choice_constraints.append(decision_constraint)

        # Remove constrained selection-choices that would always lead to infeasible architectures
        if is_selection_choice:
            _, init_choice_nodes = traverse_until_choice_nodes(self.graph, self.derivation_start_permanent_nodes)
            permanent_nodes = {node for node in choice_nodes if node in init_choice_nodes}

            has_removed = False
            for dec_node, removed_opts in get_constraint_pre_removed_options(decision_constraint, permanent_nodes):
                removed_edges, removed_nodes = \
                    get_mod_apply_choice_constraint(self._graph, self.derivation_start_nodes, dec_node, removed_opts)
                self._graph.remove_edges_from(removed_edges)
                self._graph.remove_nodes_from(removed_nodes)

                if len(removed_opts) > 0:
                    has_removed = True
            if has_removed and remove_infeasible_choices:
                return self.resolve_single_selection_choices()

        return self

    def _get_removed_constrained_selection_choices(
            self, sel_choice_node: SelectionChoiceNode, option_node: Optional[ADSGNode]) \
            -> List[Tuple[SelectionChoiceNode, List[ADSGNode]]]:
        choice_constraint = self.is_constrained_choice(sel_choice_node)
        if choice_constraint is None or option_node is None:
            return []

        # Get index of decision and chosen option node
        for i_dec, dec_node in enumerate(choice_constraint.nodes):
            if dec_node == sel_choice_node:
                try:
                    i_opt = choice_constraint.options[i_dec].index(option_node)
                    break
                except ValueError:
                    raise ValueError(f'Node {option_node!r} not an option of choice: {sel_choice_node!r}')
        else:
            raise RuntimeError(f'Choice node not found in constraint: {sel_choice_node!r}')

        # Determine options of other choices to remove
        removed_sel_choice_opts: List[Tuple[SelectionChoiceNode, List[ADSGNode]]]
        removed_sel_choice_opts = get_constraint_removed_options(choice_constraint, i_dec, i_opt)
        return removed_sel_choice_opts

    """###########################
    ### CHOICE NODES FUNCTIONS ###
    ###########################"""

    def get_ordered_next_choice_nodes(self) -> List[ChoiceNode]:
        """Get the next (selection or connection) choice node(s) that are active and can be applied"""
        if self._influence_matrix is None:
            self.set_influence_matrix()
        if self._influence_matrix is None or self._status_array is None:
            return []

        return self._influence_matrix.get_next_choice_nodes(self._status_array)

    def ordered_choice_nodes(self, choice_nodes):
        return sorted(choice_nodes, key=self._choice_sort_key)

    def _choice_sort_key(self, choice_node: CDVNode) -> tuple:
        raise NotImplementedError

    def get_confirmed_graph(self):
        """Get a subset of the ADSG only containing confirmed nodes (i.e. everything up to the next active choices)"""
        start_nodes: Set[ADSGNode] = self.derivation_start_permanent_nodes
        if start_nodes is None:
            start_nodes = self._get_alternative_start_nodes()
        non_confirmed_nodes = get_non_confirmed_nodes(self._graph, start_nodes)
        return self.get_for_adjusted(removed_nodes=non_confirmed_nodes)

    def _get_alternative_start_nodes(self) -> Set[ADSGNode]:
        raise NotImplementedError

    def get_originating_node(self, choice_node: ChoiceNode) -> ADSGNode:
        return list(self._graph.predecessors(choice_node))[0]

    def get_option_nodes(self, choice_node: SelectionChoiceNode):
        """Get a list of option nodes available for a selection choice"""
        return sorted(self._graph.successors(choice_node),
                      key=lambda n: (str(n.decision_id or ''), n.option_id))

    def iter_possible_connection_edges(self, choice_node: ConnectionChoiceNode):
        """Iterate over possible sets of connection edges for this connection choice"""
        yield from choice_node.iter_conn_edges(self)

    def n_options(self, choice_node):
        if isinstance(choice_node, ConnectionChoiceNode):
            return len(list(choice_node.iter_conn_edges(self)))
        return len(self.get_option_nodes(choice_node))

    def _update_connector_grouping_degrees(self):
        node: ConnectorDegreeGroupingNode
        for node in self.get_nodes_by_type(ConnectorDegreeGroupingNode):
            node.update_deg(self._graph)

    """#####################
    ### CHOICE FUNCTIONS ###
    #####################"""

    def initialize_choices(self):
        """
        Initializes choice nodes and makes the ADSG ready for use in deriving architecture instances:
        - Removes nodes that are incompatible with any of the initially-confirmed nodes
        - Sets the influence matrix, needed for determining choice order
        - Resolves initially-active choices with 0 or 1 options
        """
        adsg = self
        if not adsg.derivation_start_nodes:
            raise RuntimeError('Start nodes not set!')

        # Resolve incompatibility constraints
        try:
            removed_nodes = get_mod_nodes_remove_incompatibilities(adsg.graph, adsg.derivation_start_nodes)
            if len(removed_nodes) > 0:
                adsg = adsg.get_for_adjusted(removed_nodes=removed_nodes)

        # In case of problems, just return the new graph: it will be infeasible
        except IncompatibilityError:
            pass

        adsg.set_influence_matrix()
        return adsg.resolve_single_selection_choices()

    def resolve_single_selection_choices(self) -> 'ADSG':
        """Removes selection-choices that have one or no options left"""
        graph = self
        taken_choices = []
        while True:
            for sel_choice_node in list(graph.get_ordered_next_choice_nodes()):
                if isinstance(sel_choice_node, SelectionChoiceNode) and sel_choice_node in graph.graph.nodes:
                    opt_nodes = graph.get_option_nodes(sel_choice_node)
                    if len(opt_nodes) <= 1:
                        option_node = opt_nodes[0] if len(opt_nodes) == 1 else None
                        taken_choices.append((sel_choice_node, option_node))
                        self.__class__._taken_single_choices = []

                        graph = graph.get_for_apply_selection_choice(sel_choice_node, option_node)
                        taken_choices += self.__class__._taken_single_choices
                        break
            else:
                break
        self.__class__._taken_single_choices = taken_choices
        return graph

    def set_influence_matrix(self):
        try:
            self._influence_matrix = InfluenceMatrix(self)
        except ValueError:
            pass
        if self._influence_matrix is not None:
            self._status_array = self._influence_matrix.init_status_array

    """###########################
    ### APPLY CHOICE FUNCTIONS ###
    ###########################"""

    def get_for_apply_selection_choice(self, choice_node, target_option_node):
        """Get a new ADSG where the selection choice has been applied with the provided target option node"""
        if self.derivation_start_nodes is None:
            raise RuntimeError('Start nodes not set!')

        removed_edges, removed_nodes, added_edges = \
            self.get_mod_apply_selection_choice(choice_node, target_option_node)

        if self._status_array is None or self._influence_matrix is None:
            raise RuntimeError('Influence matrix not set: check graph is feasible and has decisions!')
        status_array = self._influence_matrix.apply_selection_choice(
            self._status_array, choice_node, target_option_node)

        graph = self.get_for_adjusted(removed_edges=removed_edges, removed_nodes=removed_nodes,
                                      added_edges=added_edges, status_array=status_array)
        return graph.resolve_single_selection_choices()

    def get_taken_single_selection_choices(self) -> List[Tuple[SelectionChoiceNode, Optional[ADSGNode]]]:
        return self.__class__._taken_single_choices

    def get_mod_apply_selection_choice(self, choice_node: SelectionChoiceNode, target_option_node,
                                       only_added=False) -> tuple:
        choice_constraint_map = self._get_removed_constrained_selection_choices(choice_node, target_option_node)
        return get_mod_apply_selection_choice(
            self._graph, self.derivation_start_nodes, choice_node, target_option_node, choice_constraint_map,
            only_added=only_added)

    def get_confirmed_edges_selection_choice(self, choice_node, target_option_node, include_choice=False,
                                             included_apply_edges=True, cache=None) \
            -> Tuple[Set[EdgeTuple], Set[ADSGNode]]:

        _, _, added_edges = self.get_mod_apply_selection_choice(choice_node, target_option_node, only_added=True)
        confirmed_edges = get_confirmed_edges_for_node(
            self._graph, target_option_node, include_choice=include_choice, cache=cache)
        if included_apply_edges:
            confirmed_edges |= added_edges

        # Get nodes removed by incompatibility constraints
        confirmed_start_nodes = self.derivation_start_nodes | {target_option_node}
        removed_nodes = get_mod_nodes_remove_incompatibilities(self._graph, confirmed_start_nodes, cache=cache)

        return confirmed_edges, removed_nodes

    def get_for_apply_connection_choices(
            self, choice_node_edges: List[Tuple[ConnectionChoiceNode, Sequence[Tuple[ConnectorNode, ConnectorNode]]]],
            validate=True) -> 'ADSG':
        """Get a new ADSG where the connection choices have the provided connection edges applied"""
        if self.derivation_start_nodes is None:
            raise RuntimeError('Start nodes not set!')

        removed_edges = set()
        removed_nodes = set()
        added_edges = set()
        status_array = self._status_array.copy()

        for choice_node, edges in choice_node_edges:
            if validate and len(edges) > 0 and not choice_node.validate_conn_edges(self, edges):
                raise ValueError(f'Invalid edges for connection choice ({choice_node!r}): {edges!r}')

            removed_edges_i, removed_nodes_i, added_edges_i = self.get_mod_apply_connection_choice(choice_node, edges)
            removed_edges = removed_edges.union(removed_edges_i)
            removed_nodes = removed_nodes.union(removed_nodes_i)
            added_edges = added_edges.union(added_edges_i)

            status_array = self._influence_matrix.apply_choice(status_array, choice_node, copy=False)

        return self.get_for_adjusted(removed_edges=removed_edges, removed_nodes=removed_nodes,
                                     added_edges=added_edges, status_array=status_array)

    def get_for_apply_connection_choice(self, choice_node: ConnectionChoiceNode,
                                        edges: Sequence[Tuple[ConnectorNode, ConnectorNode]] = None, validate=True):
        """Get a new ADSG where the connection choice has the provided connection edges applied"""
        if edges is None:
            edges = tuple()
        return self.get_for_apply_connection_choices([(choice_node, edges)], validate=validate)

    def get_mod_apply_connection_choice(self, choice_node: ConnectionChoiceNode,
                                        edges: Sequence[Tuple[ConnectorNode, ConnectorNode]]) -> tuple:
        return get_mod_apply_connection_choice(self._graph, choice_node, edges)

    def get_confirmed_edges_connection_choice(self, choice_node, connection_option):
        _, _, added_edges = self.get_mod_apply_connection_choice(choice_node, connection_option)
        return added_edges

    """###########################
    ### GRAPH STATUS FUNCTIONS ###
    ###########################"""

    @property
    def feasible(self):
        """Whether the architecture is feasible or not (unconnectable connectors and/or unsolvable incompatibilities)"""
        try:
            self._check_unconnected_connectors()
        except ValueError:
            return False
        if self.has_confirmed_incompatibility_edges():
            return False
        return True

    @property
    def final(self):
        """Whether the architecture is final (no more choices to make) or not"""
        return len(self._graph.nodes) > 0 and len(self.choice_nodes) == 0

    def _check_unconnected_connectors(self):
        if len(get_unconnected_connectors(self._graph, self.derivation_start_nodes, stop_at_one=True)) > 0:
            raise ValueError('There are unconnectable ports')

    @property
    def unconnected_connectors(self):
        return get_unconnected_connectors(self._graph, self.derivation_start_nodes)

    """#############################
    ### GRAPH QUERYING FUNCTIONS ###
    #############################"""

    def has_conditional_existence(self, target_node, base_nodes=None):
        return has_conditional_existence(self._graph, self.derivation_start_nodes, target_node, base_nodes=base_nodes)

    """############################
    ### GRAPH COPYING FUNCTIONS ###
    ############################"""

    def copy(self):
        """Create a copy of the ADSG"""
        return self.get_for_adjusted()

    def get_for_adjusted(self, removed_edges=None, removed_nodes=None, added_edges=None,
                         status_array=None, inplace=False, **kwargs):
        graph = self._graph
        graph_copy = graph if inplace else self._get_empty_graph()

        if not inplace:
            graph_copy.add_edges_from(graph.edges(keys=True, data=True))
            graph_copy.add_nodes_from(graph.nodes)
        if removed_edges is not None:
            graph_copy.remove_edges_from(removed_edges)
        if removed_nodes is not None:
            graph_copy.remove_nodes_from(removed_nodes)
        if added_edges is not None:
            graph_copy.add_edges_from(added_edges)

        self._mod_graph_adjust_kwargs(kwargs)
        if inplace:
            if status_array is not None:
                self._status_array = status_array
            self._mod_graph_inplace(kwargs)
            return self

        dec_con_map_copy = self._choice_constraints.copy()
        return self.__class__(_graph=graph_copy, _influence_matrix=self._influence_matrix,
                              _status_array=status_array if status_array is not None else self._status_array,
                              _choice_con_map=dec_con_map_copy, _des_var_values=self._des_var_values, **kwargs)

    def _mod_graph_adjust_kwargs(self, kwargs):
        pass

    def _mod_graph_inplace(self, kwargs):
        pass

    def get_for_kept_edges(self, kept_edges, **kwargs):
        graph_copy = self._get_empty_graph()
        graph_copy.add_edges_from(kept_edges)

        self._mod_graph_adjust_kwargs(kwargs)
        return self.__class__(_graph=graph_copy, _influence_matrix=self._influence_matrix,
                              _status_array=self._status_array, _choice_con_map=self._choice_constraints,
                              _des_var_values=self._des_var_values, **kwargs)

    """#########################################
    ### INCOMPATIBILITY CONSTRAINT FUNCTIONS ###
    #########################################"""

    def add_incompatibility_constraint(self, nodes: List[ADSGNode], **attr):
        """
        An incompatibility constraint defines that two nodes cannot exist in an architecture at the same time. This
        means that when one of the nodes becomes confirmed by making a choice, the other node (and its derived nodes)
        will be removed from the graph. If both of the nodes are confirmed, it means that there is no way to have only
        one of them in an architecture, and that therefore the architecture is infeasible (this can also happen before
        any choices have been made!).

        Incompatibility constraints should be added before initializing the graph's choices.
        """
        add_incompatibility_constraint(self._graph, nodes, **attr)
        return self

    def get_confirmed_incompatibility_edges(self) -> Set[EdgeTuple]:
        """Returns incompatibility edges that link two confirmed nodes. There should be none in order to have a
        feasible architecture."""
        if self.derivation_start_nodes is None:
            return set()
        return get_confirmed_incompatibility_edges(self._graph, self.derivation_start_nodes)

    def has_confirmed_incompatibility_edges(self):
        return len(self.get_confirmed_incompatibility_edges()) > 0

    def get_incompatibility_deriving_nodes(self, target_node: ADSGNode, confirmed_nodes: Set[ADSGNode]):
        return get_incompatibility_deriving_nodes(self._graph, target_node, confirmed_nodes)


ADSGType = TypeVar('ADSGType', bound=ADSG)
