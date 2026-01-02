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
from typing import *
import networkx as nx
from collections import defaultdict
from adsg_core.graph.traversal import *
from adsg_core.graph.adsg_nodes import *
from adsg_core.graph.graph_edges import *
from adsg_core.graph.incompatibility import *

__all__ = ['apply_derive_target_nodes', 'apply_remove_unconnected', 'add_select_connector_nodes',
           'get_mod_apply_selection_choice', 'get_mod_apply_connection_choice', 'get_mod_apply_choice_constraint',
           'NoOptionError']


def _get_underlying_connector_nodes(graph: nx.MultiDiGraph, nodes: List[ConnectorNode]):
    underlying_nodes: List[ConnectorNode] = []
    conn_grouping_node_map: Dict[ConnectorNode, ConnectorDegreeGroupingNode] = {}

    for node in nodes:
        if isinstance(node, ConnectorDegreeGroupingNode):

            base_connector_nodes = node.get_base_connector_nodes(graph)
            for conn_node in base_connector_nodes:
                conn_grouping_node_map[conn_node] = node
            underlying_nodes += base_connector_nodes

        elif isinstance(node, ConnectorNode):
            underlying_nodes.append(node)

    return underlying_nodes, conn_grouping_node_map


def apply_derive_target_nodes(graph: nx.MultiDiGraph, connection_choice_node: ConnectionChoiceNode):
    """
    Add derivation edges so that the target nodes of a connection choices are included because of the source nodes.
    """

    source_nodes = [edge[0] for edge in iter_in_edges(graph, connection_choice_node, edge_type=EdgeType.CONNECTS)]

    target_nodes = [edge[1] for edge in iter_out_edges(graph, connection_choice_node, edge_type=EdgeType.CONNECTS)]
    base_target_nodes, _ = _get_underlying_connector_nodes(graph, target_nodes)

    # Add edges for deriving target connector nodes
    collector_node = CollectorNode()
    edges = [(src_node, collector_node) for src_node in source_nodes]
    edges += [(collector_node, tgt_node) for tgt_node in base_target_nodes]
    for edge in edges:
        add_edge(graph, edge[0], edge[1], edge_type=EdgeType.DERIVES)


def apply_remove_unconnected(graph: nx.MultiDiGraph, connection_choice_node: ConnectionChoiceNode) -> List[SelectionChoiceNode]:
    """
    Modify the graph such that unconnected connector nodes are removed, if they request so.
    """

    source_nodes = [edge[0] for edge in iter_in_edges(graph, connection_choice_node, edge_type=EdgeType.CONNECTS)]
    base_source_nodes, src_grouping_nodes = _get_underlying_connector_nodes(graph, source_nodes)

    target_nodes = [edge[1] for edge in iter_out_edges(graph, connection_choice_node, edge_type=EdgeType.CONNECTS)]
    base_target_nodes, tgt_grouping_nodes = _get_underlying_connector_nodes(graph, target_nodes)

    grouping_nodes = set(list(src_grouping_nodes.values()) + list(tgt_grouping_nodes.values()))

    # Check if there is a collector node between the source and target nodes
    # (added if the source nodes derive the target nodes)
    collector_node = None
    for src_node in source_nodes:
        for edge in iter_out_edges(graph, src_node, edge_type=EdgeType.DERIVES):
            if isinstance(edge[1], CollectorNode):
                collector_node = edge[1]
                break
        if collector_node is not None:
            break

    def _is_optionally_connectable(connector_node: ConnectorNode) -> bool:

        # If the connector is specified as a list, check if 0 is a possibility
        if connector_node.deg_list is not None:
            return 0 in connector_node.deg_list

        # Otherwise check if the lower bound is 0
        return connector_node.deg_min == 0

    def _set_must_connect(connector_nodes: List[ConnectorNode], grouping_node_map):
        group_nodes_must_connect: List[ConnectorDegreeGroupingNode] = []

        for connector_node in connector_nodes:

            # Check if we should set the connection to be required at the grouping node
            if connector_node in grouping_node_map:
                group_nodes_must_connect.append(grouping_node_map[connector_node])
                continue

            # Remove 0 from the list
            if connector_node.deg_list is not None:
                connector_node.deg_list.remove(0)

            # Or set the lower bound to 0 (if the upper bound isn't also 0)
            elif connector_node.deg_max != 0:
                connector_node.deg_min = 1

        for grouping_node_ in set(group_nodes_must_connect):
            grouping_node_.override_must_connect = True

    # Add logic to remove unconnected connector nodes:
    # For each optionally-connected node:
    # 1. Add a selection choice to select whether the node is included
    # 2. Set that, if the node is included, the connector requires at least 1 connection
    selection_choice_nodes = []

    # Apply for source nodes
    optionally_connectable_source_nodes = [src_node for src_node in base_source_nodes
                                           if src_node.remove_if_unconnected and _is_optionally_connectable(src_node)]

    if len(optionally_connectable_source_nodes) > 0:
        _set_must_connect(optionally_connectable_source_nodes, src_grouping_nodes)

        selection_choice_nodes += add_select_connector_nodes(
            graph, connection_choice_node, optionally_connectable_source_nodes, derive_collector_node=collector_node)

    # Apply for target nodes
    optionally_connectable_target_nodes = [tgt_node for tgt_node in base_target_nodes
                                           if tgt_node.remove_if_unconnected and _is_optionally_connectable(tgt_node)]

    if len(optionally_connectable_target_nodes) > 0:
        _set_must_connect(optionally_connectable_target_nodes, tgt_grouping_nodes)

        selection_choice_nodes += add_select_connector_nodes(
            graph, connection_choice_node, optionally_connectable_target_nodes)

    # Update the connection degrees of the connector grouping nodes
    for grouping_node in grouping_nodes:
        grouping_node.update_deg(graph)

    return selection_choice_nodes


def add_select_connector_nodes(graph: nx.MultiDiGraph, connector_choice_node: ConnectionChoiceNode,
                               connector_nodes: List[ConnectorNode], derive_collector_node: CollectorNode = None) \
        -> List[SelectionChoiceNode]:
    """Add selection choices for selecting whether to include connector nodes."""

    non_sel_node = NonSelectionNode()

    selection_choice_nodes = []
    for connector_node in connector_nodes:

        # Get edges deriving the node
        deriving_edges = list(iter_in_edges(graph, connector_node, edge_type=EdgeType.DERIVES))
        if len(deriving_edges) == 0:
            continue

        # Check if the connector node actually allows connections
        allows_conn = True
        if connector_node.deg_list is not None:
            if len(connector_node.deg_list) == 0 or connector_node.deg_list == [0]:
                allows_conn = False
        elif connector_node.deg_min == 0 and connector_node.deg_max == 0:
            allows_conn = False

        # Remove deriving edges
        for edge in deriving_edges:
            graph.remove_edge(edge[0], edge[1], key=edge[2])

        # Derive the collector node if needed
        if derive_collector_node is not None:
            for edge in deriving_edges:
                add_edge(graph, edge[0], derive_collector_node, edge_type=EdgeType.DERIVES)

        # If connections are not allowed, we are done because the node should never be included (derived)
        if not allows_conn:
            continue

        # Add selection choice node
        choice_id = f'CONN_{connector_choice_node.decision_id or "_"}_SEL_{connector_node.name or "_"}'
        choice_node = SelectionChoiceNode(choice_id)

        for edge in deriving_edges:
            add_edge(graph, edge[0], choice_node, edge_type=EdgeType.DERIVES)

        for i, option_node in enumerate([non_sel_node, connector_node]):
            if option_node.option_id is None:
                option_node.option_id = i

            add_edge(graph, choice_node, option_node, edge_type=EdgeType.DERIVES)

        selection_choice_nodes.append(choice_node)
    return selection_choice_nodes


def get_mod_apply_choice_constraint(graph: nx.MultiDiGraph, start_nodes: set, choice_node: ChoiceNode,
                                    removed_option_nodes: List[DSGNode]):
    removed_edges = set()
    removed_nodes = set()

    # Process decision-option constraints
    for edge in iter_out_edges(graph, choice_node):
        # Check if this target node is removed
        if edge[1] not in removed_option_nodes:
            continue

        derived_edges, derived_nodes = get_derived_edges_for_edge(
            graph, edge, start_nodes, removed_edges=removed_edges, removed_nodes=removed_nodes)
        removed_edges |= derived_edges
        removed_nodes |= derived_nodes

    return removed_edges, removed_nodes


class NoOptionError(ValueError):
    pass


def get_mod_apply_selection_choice(
        graph: nx.MultiDiGraph, start_nodes: set, choice_node: ChoiceNode, target_option_node: DSGNode = None,
        choice_con_map: List[Tuple[SelectionChoiceNode, List[DSGNode]]] = None, only_added=False) -> tuple:

    # Check outgoing edges
    choice_out_edges = set(iter_out_edges(graph, choice_node))
    option_nodes = {edge[1] for edge in choice_out_edges}

    # If there are no options, remove the decision node and mark the originating node as infeasible
    if len(option_nodes) == 0:
        removed_nodes = {choice_node}

        originating_nodes = list(graph.predecessors(choice_node))
        added_edges = {get_edge_for_type(
            list(start_nodes)[0], originating_node, EdgeType.INCOMPATIBILITY, choice_node=choice_node)
            for originating_node in originating_nodes}

        return set(), removed_nodes, added_edges

    if target_option_node not in option_nodes:
        raise NoOptionError(f'Node ({target_option_node!s}) is not an option of choice node: '
                            f'{choice_node!s} -> {option_nodes!s}')

    removed_edges = set()
    removed_nodes = set()
    in_edges = list(iter_in_edges(graph, choice_node))
    added_edges = {get_edge(in_edge[0], target_option_node) for in_edge in in_edges}

    if only_added:
        return set(), set(), added_edges

    # Process decision-option constraints
    if choice_con_map is not None:
        for constrained_dec_node, removed_options in choice_con_map:
            if constrained_dec_node not in graph.nodes:
                continue
            for constrained_out_edge in iter_out_edges(graph, constrained_dec_node):

                if constrained_out_edge[1] in removed_options:
                    choice_out_edges.add(constrained_out_edge)

    # Remove derived nodes
    for edge in choice_out_edges:
        # If this is the target edge, do not remove the derived nodes
        if edge[0] == choice_node and edge[1] == target_option_node:
            continue

        derived_edges, derived_nodes = get_derived_edges_for_edge(
            graph, edge, start_nodes, removed_edges=removed_edges, removed_nodes=removed_nodes)
        removed_edges |= derived_edges
        removed_nodes |= derived_nodes

    removed_nodes.add(choice_node)

    # Process incompatibility constraints
    confirmed_start_nodes = start_nodes | {target_option_node}
    try:
        removed_nodes |= get_mod_nodes_remove_incompatibilities(graph, confirmed_start_nodes, removed_edges)
    except IncompatibilityError as e:
        removed_nodes |= e.removed_nodes
        added_edges |= e.edges

    return removed_edges, removed_nodes, added_edges


def get_mod_apply_connection_choice(graph: nx.MultiDiGraph, choice_node: ConnectionChoiceNode,
                                    edges: Sequence[Tuple[ConnectorNode, ConnectorNode]]) -> tuple:

    in_nodes = {edge[0] for edge in iter_in_edges(graph, choice_node)}
    out_nodes = {edge[1] for edge in iter_out_edges(graph, choice_node)}

    for edge in edges:
        if edge[0] not in in_nodes or (edge[1] is not None and edge[1] not in out_nodes):
            raise ValueError('Node not part of connection choice')

    removed_nodes = {choice_node}

    # Create edges with correct keys
    added_edges = set()
    edge_key = defaultdict(int)
    for edge in edges:
        if edge[1] is not None:
            added_edges.add(get_edge(edge[0], edge[1], key=edge_key[edge], is_conn=True))
            edge_key[edge] += 1

    # Remove exclusion edges
    removed_edges = set(choice_node.get_excluded_edges(graph)) | set(choice_node.get_deriving_edges(graph))

    return removed_edges, removed_nodes, added_edges
