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
from typing import *
import networkx as nx
from adsg_core.graph.traversal import *
from adsg_core.graph.adsg_nodes import *
from adsg_core.graph.graph_edges import *

__all__ = ['add_incompatibility_constraint', 'get_confirmed_incompatibility_edges', 'IncompatibilityError',
           'get_mod_nodes_remove_incompatibilities', 'get_incompatibility_deriving_nodes']


def add_incompatibility_constraint(graph: nx.MultiDiGraph, nodes: List[ADSGNode], **attr):
    # Loop over all pairs (both directions)
    for source_node, target_node in itertools.permutations(nodes, 2):
        add_edge(graph, source_node, target_node, edge_type=EdgeType.INCOMPATIBILITY, **attr)


def _get_canonical_edge(edge: EdgeTuple) -> EdgeTuple:
    source_node, target_node = sorted(edge[:2], key=lambda n: getattr(n, 'name', str(hash(n))))
    return source_node, target_node, 0, edge[-1]


def get_confirmed_incompatibility_edges(graph: nx.MultiDiGraph, start_nodes: Set[ADSGNode]) -> Set[EdgeTuple]:
    """Confirmed incompatibility edges represent incompatibility constraints between confirmed nodes, and thereby
    show that the confirmed architecture is infeasible (as there is an incompatibility)."""
    confirmed_nodes, _ = traverse_until_choice_nodes(graph, start_nodes)

    edges: Set[EdgeTuple] = set()
    for edge in iter_edges(graph):
        if get_edge_type(edge) != EdgeType.INCOMPATIBILITY:
            continue
        if edge[0] in confirmed_nodes or edge[1] in confirmed_nodes:
            edges.add(_get_canonical_edge(edge))
    return edges


class IncompatibilityError(RuntimeError):

    def __init__(self, msg: str, edges: Set[EdgeTuple], removed_nodes: Set[ADSGNode]):
        super(IncompatibilityError, self).__init__(msg)
        self.edges = {_get_canonical_edge(edge) for edge in edges}
        self.removed_nodes = removed_nodes

    def print_edges(self):
        return ', '.join([f'{edge[0]!s} <-> {edge[1]!s}' for edge in self.edges])


def get_mod_nodes_remove_incompatibilities(
        graph: nx.MultiDiGraph, start_nodes: Set[ADSGNode], removed_edges: Set[EdgeTuple] = None, cache=None) \
        -> Set[ADSGNode]:
    confirmed_nodes, _ = traverse_until_choice_nodes(graph, start_nodes)

    removed_nodes: Set[ADSGNode] = set()
    confirmed_incompatibility_edges: Set[EdgeTuple] = set()
    infeasible_incompatibility_edges: Set[EdgeTuple] = set()
    for edge in iter_edges(graph):
        if get_edge_type(edge) != EdgeType.INCOMPATIBILITY:
            continue

        # If both nodes are in the confirmed nodes, the graph is infeasible
        if edge[0] in confirmed_nodes and edge[1] in confirmed_nodes:
            infeasible_incompatibility_edges.add(edge)
            continue

        # If the source node is confirmed, we need to remove target (and derived) incompatible nodes
        if edge[0] in confirmed_nodes:
            confirmed_incompatibility_edges.add(edge)
            removed_nodes.add(edge[1])

    if len(infeasible_incompatibility_edges) > 0:
        raise IncompatibilityError('Could not resolve incompatibility constraints', infeasible_incompatibility_edges,
                                   removed_nodes)

    if len(confirmed_incompatibility_edges) == 0:
        return set()

    # Remove nodes derived from incompatible target nodes
    if removed_edges is None:
        removed_edges = set()
    for edge in confirmed_incompatibility_edges:
        # If this is the target edge, do not remove the derived nodes
        if edge[1] in confirmed_nodes:
            continue

        # Additionally remove nodes that are derived by the incompatible node
        derived_edges, derived_nodes = \
            get_derived_edges_for_edge(graph, edge, start_nodes, removed_edges=removed_edges,
                                       removed_nodes=removed_nodes, cache=cache)
        removed_edges |= derived_edges
        removed_nodes |= derived_nodes

        # Additionally remove nodes that are deriving the incompatible node
        deriving_nodes = get_incompatibility_deriving_nodes(graph, edge[1], confirmed_nodes, cache=cache)
        removed_nodes |= deriving_nodes

        # And the nodes that are derived by the deriving nodes
        for deriving_node in deriving_nodes:
            deriving_derived_edges, deriving_derived_nodes =\
                get_derived_edges_for_node(graph, deriving_node, start_nodes, removed_edges=removed_edges,
                                           removed_nodes=removed_nodes, traversed=removed_nodes, cache=cache)
            removed_edges |= deriving_derived_edges
            removed_nodes |= deriving_derived_nodes

        # If any of the deriving nodes are in the confirmed nodes, we have an infeasible graph
        if len(deriving_nodes & confirmed_nodes) > 0:
            removed_nodes -= confirmed_nodes
            raise IncompatibilityError('Incompatibility constraint derives from confirmed nodes', {edge}, removed_nodes)

    return removed_nodes


def get_incompatibility_deriving_nodes(
        graph: nx.MultiDiGraph, target_node: ADSGNode, confirmed_nodes: Set[ADSGNode],
        _deriving_nodes: Set[ADSGNode] = None, cache=None) -> Set[ADSGNode]:

    deriving_nodes = {target_node}
    if _deriving_nodes is not None:
        deriving_nodes |= _deriving_nodes

    # Get deriving nodes
    option_decision_nodes: Set[SelectionChoiceNode] = set()
    for edge in iter_in_edges_cached(graph, target_node, cache=cache):
        if get_edge_type(edge) != EdgeType.DERIVES:
            continue
        deriving_node = edge[0]
        if deriving_node in deriving_nodes:
            continue

        # If it is a confirmed node, do not continue searching (as the calling function will fail anyway)
        if deriving_node in confirmed_nodes:
            deriving_nodes.add(deriving_node)
            continue

        # If it is a decision node, do not continue searching for additional deriving nodes
        if isinstance(deriving_node, SelectionChoiceNode):
            option_decision_nodes.add(deriving_node)
            continue

        deriving_nodes.add(deriving_node)
        deriving_nodes |= get_incompatibility_deriving_nodes(
            graph, deriving_node, confirmed_nodes, _deriving_nodes=deriving_nodes, cache=cache)

    # Check option-decision nodes
    for option_decision_node in option_decision_nodes:
        option_nodes = {edge[1] for edge in iter_out_edges_cached(graph, option_decision_node, cache=cache)
                        if get_edge_type(edge) == EdgeType.DERIVES}

        # If all option nodes are deriving nodes, it means that the option-decision would be left without any options,
        # and therefore we also need to remove this decision and its deriving nodes
        if len(option_nodes.difference(deriving_nodes)) == 0:
            deriving_nodes |= get_incompatibility_deriving_nodes(graph, option_decision_node, confirmed_nodes,
                                                                 _deriving_nodes=deriving_nodes, cache=cache)

    return deriving_nodes
