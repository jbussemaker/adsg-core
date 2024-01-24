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

__all__ = ['get_mod_apply_selection_choice', 'get_mod_apply_connection_choice', 'get_mod_apply_choice_constraint',
           'NoOptionError']


def get_mod_apply_choice_constraint(graph: nx.MultiDiGraph, start_nodes: set, choice_node: ChoiceNode,
                                    removed_option_nodes: List[ADSGNode]):
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
        graph: nx.MultiDiGraph, start_nodes: set, choice_node: ChoiceNode, target_option_node: ADSGNode = None,
        choice_con_map: List[Tuple[SelectionChoiceNode, List[ADSGNode]]] = None, only_added=False) -> tuple:

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
