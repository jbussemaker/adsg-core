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
from adsg_core.graph.adsg_nodes import *
from adsg_core.graph.graph_edges import *

__all__ = ['get_nodes_by_type', 'get_in_degree', 'get_out_degree', 'get_choice_nodes', 'get_derived_edges_for_edge',
           'get_derived_edges_for_node', 'get_deriving_in_edges', 'check_derives', 'has_conditional_existence',
           'get_confirmed_edges_for_node', 'get_unconnected_connectors', 'traverse_until_choice_nodes',
           'get_non_confirmed_nodes', 'get_nodes_by_subtype', 'iter_in_edges_cached', 'iter_out_edges_cached']


T = TypeVar('T')


def get_nodes_by_type(graph: nx.MultiDiGraph, type_: Type[T]) -> List[T]:
    return [node for node in graph.nodes if type(node) == type_]


def get_nodes_by_subtype(graph: nx.MultiDiGraph, type_: Type[T]) -> List[T]:
    return [node for node in graph.nodes if isinstance(node, type_)]


def get_in_degree(graph: nx.MultiDiGraph, node: ADSGNode, edge_type: EdgeType = None) -> int:
    if edge_type is None:
        return graph.in_degree(node)

    n_in = 0
    for edge in iter_in_edges(graph, node):
        if get_edge_type(edge) == edge_type:
            n_in += 1
    return n_in


def get_out_degree(graph: nx.MultiDiGraph, node: ADSGNode, edge_type: EdgeType = None) -> int:
    if edge_type is None:
        return graph.out_degree(node)

    n_out = 0
    for edge in iter_out_edges(graph, node):
        if get_edge_type(edge) == edge_type:
            n_out += 1
    return n_out


def get_choice_nodes(graph: nx.MultiDiGraph) -> List[ChoiceNode]:
    return get_nodes_by_type(graph, SelectionChoiceNode)+get_nodes_by_type(graph, ConnectionChoiceNode)


def get_derived_edges_for_edge(graph, edge, start_nodes, removed_edges=None, removed_nodes=None,
                               traversed: Set[ADSGNode] = None, cache=None):
    return get_derived_edges_for_node(graph, edge[0], start_nodes, removed_edges=removed_edges,
                                      removed_nodes=removed_nodes, derived_edges={edge}, traversed=traversed,
                                      cache=cache)


def get_derived_edges_for_node(
        graph: nx.MultiDiGraph,
        node: ADSGNode,
        start_nodes: Optional[Set[ADSGNode]],
        removed_edges: Set[Union[EdgeTuple, Tuple[ADSGNode, ADSGNode]]] = None,
        removed_nodes: Set[ADSGNode] = None,
        derived_edges: Set[EdgeTuple] = None,
        traversed: Set[ADSGNode] = None,
        cache=None,
) -> Tuple[Set[EdgeTuple], Set[ADSGNode]]:
    """
    Get the edges and nodes for a given node that are derived ONLY by this node.

    In general, from the starting node, all successor nodes/edges are derived if these nodes do not have additional
    in-edges not stemming from this origin node. In-edges that are in the removed_edges array are not taken into
    account. In addition, for connector nodes, in-edges stemming from other connector nodes are also not taken into
    account (because these do not serve as a "reason" for the connector node to exist).
    """

    derived_nodes = set()
    if removed_edges is None:
        removed_edges = set()
    if removed_nodes is None:
        removed_nodes = set()
    if traversed is None:
        traversed = {node}
    removed_derived_nodes = set(removed_nodes)

    # All out-edges are derived edges (if not given)
    if derived_edges is None:
        derived_edge_types = [EdgeType.DERIVES, EdgeType.CONNECTS]
        derived_edges = {edge for edge in iter_out_edges_cached(graph, node, cache=cache)
                         if get_edge_type(edge) in derived_edge_types}
    removed_derived_edges = removed_edges | derived_edges

    # For every out-edge, continue determining the derived edges
    for edge in list(derived_edges):
        target_node = edge[1]
        edge_type = get_edge_type(edge, default=EdgeType.DERIVES)

        # If we already checked this node, do not continue checking it
        if target_node in traversed:
            continue

        # If the target node is contained in the set of start nodes, regard it as an independent node
        if start_nodes is not None and target_node in start_nodes:
            continue

        # Get the derived edges and nodes of the target node
        target_derived_edges, target_derived_nodes = \
            get_derived_edges_for_node(graph, target_node, start_nodes, removed_edges=removed_derived_edges,
                                       removed_nodes=removed_derived_nodes | {target_node},
                                       traversed=traversed | {target_node}, cache=cache)

        # Get the in-edges deriving the target node, not taking the derived edges into account
        deriving_in_edges = get_deriving_in_edges(
            graph, target_node, removed_edges=removed_derived_edges, removed_nodes=removed_derived_nodes,
            edge_type=edge_type, cache=cache)

        # Remove any in-edge that is also derived by the target node
        deriving_in_edges -= target_derived_edges

        # If there are any remaining in-edges, it means that the target node could be derived by other means than
        # the origin node, so we do not consider it as a derived node
        if len(deriving_in_edges) > 0:
            continue

        # The target node is derived ONLY by the origin node, so we also consider its derived edges and nodes
        derived_edges |= target_derived_edges
        derived_nodes |= {target_node} | target_derived_nodes
        removed_derived_nodes |= {target_node} | target_derived_nodes

    return derived_edges, derived_nodes


def get_deriving_in_edges(graph: nx.MultiDiGraph, node: ADSGNode, removed_edges: Iterable[tuple] = None,
                          removed_nodes: Iterable[ADSGNode] = None, edge_type: EdgeType = None, cache=None) \
        -> Set[EdgeTuple]:
    """
    Returns in-edges of a node that are not in the removed_edges set or (indirectly) stemming from needed port
    nodes.
    """

    # Get non-removed in-edges
    if removed_edges is None:
        removed_edges = set()
    if removed_nodes is None:
        removed_nodes = set()

    # removing_nodes = frozenset({edge[0] for edge in removed_edges if edge[1] is node} | removed_nodes)

    # # Load cache if available
    # der_in_cache = cache_key = None
    # if cache is not None:
    #     cache['der_in_edges'] = der_in_cache = cache.get('der_in_edges', {})
    #
    #     # Load from cache if available
    #     cache_key = hash((node, edge_type, removing_nodes))
    #     if cache_key in der_in_cache:
    #         return der_in_cache[cache_key].copy()

    in_edges = set()
    deriving_edge_types = {EdgeType.DERIVES, edge_type}
    for edge in iter_in_edges_cached(graph, node, cache=cache):
        # Check if edge is in removed edges
        # if edge[0] in removing_nodes:
        if edge[0] in removed_nodes or edge in removed_edges or (edge[0], edge[1]) in removed_edges:
            continue

        # Check edge type
        if get_edge_type(edge, default=EdgeType.DERIVES) in deriving_edge_types:
            in_edges.add(edge)

    # if der_in_cache is not None:
    #     der_in_cache[cache_key] = in_edges.copy()

    return in_edges


def check_derives(graph: nx.MultiDiGraph, source_node: ADSGNode, target_node: ADSGNode, connects=False) -> bool:

    allowed_types = {EdgeType.DERIVES}
    if connects:
        allowed_types.add(EdgeType.CONNECTS)

    for edge in iter_in_edges(graph, target_node):
        edge_type = get_edge_type(edge)
        if edge_type not in allowed_types:
            continue

        if edge[0] == source_node:
            return True

        if check_derives(graph, source_node, edge[0], connects=connects):
            return True
    return False


def has_conditional_existence(graph: nx.MultiDiGraph, start_nodes: Set[ADSGNode], target_node: ADSGNode,
                              base_nodes: List[ADSGNode] = None) -> int:
    """
    Returns whether a node exists conditionally or not. Conditional existence means that due to some decision, there
    is a possibility that the node will not exist when taking this decision.

    There are three return values:
    - 0: Unconditional existence: target node will always exist
    - 1: Conditional existence: based on some intermediate decision, it can be possible that the target node does
                                not exist
    - 2: Unconditional non-existence: target node will never exist if base nodes exist (i.e. none of the base nodes
                                      derive the target node)
    """

    derives_cache = {}
    derives_traversed = {}

    # Function to check whether some node derives another node
    def _derives(base_node, check_node, start_check_node=None, removed_nodes=None, decision_cutoff=False):
        if base_node == check_node:
            return True
        removed_nodes_key = tuple(sorted([hash(n) for n in removed_nodes or []]))
        cache_key = (base_node, check_node, removed_nodes_key, decision_cutoff)
        if cache_key in derives_cache:
            return derives_cache[cache_key]

        if start_check_node is None:
            start_check_node = check_node
        traversed_key = (base_node, start_check_node, removed_nodes_key, decision_cutoff)
        if traversed_key not in derives_traversed:
            derives_traversed[traversed_key] = set()
        derives_traversed[traversed_key].add(check_node)

        # Loop over in-edges
        does_derive = False
        for edge in iter_in_edges(graph, check_node):
            if get_edge_type(edge) != EdgeType.DERIVES:
                continue

            derives_cache[edge[0], edge[1], removed_nodes_key, decision_cutoff] = True
            if edge[0] == base_node:
                return True

            # Prevent infinite recursion
            if edge[0] in derives_traversed[traversed_key]:
                continue

            # If the incoming node is removed, the link is broken
            if removed_nodes is not None and edge[0] in removed_nodes:
                continue

            # Check if we should continue at decision nodes
            if decision_cutoff and isinstance(edge[0], ChoiceNode):
                continue

            # Walk back
            if _derives(base_node, edge[0], start_check_node=start_check_node, removed_nodes=removed_nodes,
                        decision_cutoff=decision_cutoff):
                does_derive = True
                break

        derives_cache[cache_key] = does_derive
        return does_derive

    # Get nodes to check conditional existence from
    if base_nodes is None:
        base_nodes = start_nodes
    else:
        base_nodes = set(base_nodes)

        # If we are comparing to some custom target nodes, we do not have conditional existence if the target node
        # derives one of the base nodes
        for check_base_node in base_nodes:
            if _derives(target_node, check_base_node):
                return False

    if not base_nodes or len(base_nodes) == 0:
        return False

    traversed = {}
    in_walk_back = set()

    # Function to check whether some base node always derives some other node
    def _maybe_derived_from_start_node(base_node, node=None):
        # If node is not given, we want to check for the base node
        if node is None:
            node = base_node

        # If the base node is one of the ext func nodes, the node is indeed always derived from an ext func node
        if base_node in base_nodes:
            return False

        # Check cache (also to prevent infinite loops)
        if base_node in traversed:
            return traversed[base_node]

        in_walk_back.add(base_node)

        # If we are at a decision node, check whether all of the options derive the target node
        is_choice = isinstance(base_node, ChoiceNode)
        is_conditional = False
        if is_choice and node != base_node:
            # Loop over out edges (the decision-options)
            option_nodes = [edge[1] for edge in iter_out_edges(graph, base_node)
                            if get_edge_type(edge) == EdgeType.DERIVES]

            for option_node in option_nodes:
                removed_nodes = set(option_nodes)
                strictly_derives = _derives(option_node, node, decision_cutoff=True)
                if not strictly_derives:
                    removed_nodes -= {option_node}

                # If a specific option does not derive the target node, not all paths lead to the target node and
                # therefore the target node is also not always derived from an ext func node
                if not _derives(option_node, node, removed_nodes=removed_nodes):
                    is_conditional = True
                    break

        # Walk back
        derivation_status = 2
        for edge in iter_in_edges(graph, base_node):
            if get_edge_type(edge) != EdgeType.DERIVES:
                continue

            if edge[0] in in_walk_back:
                continue

            # If any of the previous nodes is always derived from an external function node, we are too
            maybe_derived_status = _maybe_derived_from_start_node(edge[0], node=node)
            if not maybe_derived_status:
                traversed[base_node] = is_conditional
                return is_conditional

            # If any of the previous node has conditional existence, we also have if we don't find an unconditional node
            if maybe_derived_status == 1:
                derivation_status = 1

        # If none of the previous nodes are derived from an external function node, we are neither
        traversed[base_node] = derivation_status
        return derivation_status

    return _maybe_derived_from_start_node(target_node)


def get_confirmed_edges_for_node(graph: nx.MultiDiGraph, node: ADSGNode, include_choice=False,
                                 _traversed=None, _traversed_to_update=None, _walked_hit=None,
                                 cache=None) -> Set[EdgeTuple]:

    # Initialize temp storage if needed
    # _traversed: store for each traversed node which edges were marked as confirmed
    # _traversed_to_update: node pairs (src, tgt) where the edges of the tgt should be the same as edges of the src
    # _walked_hit: temp cache to remember which nodes are part of a loop when encountering a previously-traversed node
    is_request_start = False
    if _traversed is None:
        _traversed = {node: set()}
        is_request_start = True
    if _traversed_to_update is None:
        _traversed_to_update = []
    if _walked_hit is None:
        _walked_hit = {}

    # Load cache if available
    conf_edges_cache = None
    if cache is not None:
        cache['conf_edge'] = conf_edges_cache = cache.get('conf_edge', {})

        # Load from cache if available
        if node in conf_edges_cache:
            return conf_edges_cache[node].copy()

    # Loop over outgoing edges
    confirmed_edges = set()
    for out_edge in iter_out_edges(graph, node):
        if get_edge_type(out_edge) == EdgeType.INCOMPATIBILITY:
            continue

        # Stop at choice nodes
        if isinstance(out_edge[1], ChoiceNode):
            # Mark edge as confirmed if we also want to return choice nodes
            if include_choice:
                confirmed_edges.add(out_edge)
            continue

        # Check if already walked
        if out_edge[1] in _traversed:
            # Remember where we obtained this hit
            if node not in _walked_hit:
                _walked_hit[node] = []
            _walked_hit[node].append(out_edge[1])

            # Mark the edge as confirmed
            confirmed_edges.add(out_edge)
            continue

        if out_edge[1] not in _traversed:
            _traversed[out_edge[1]] = set()

        # Recursively walk the graph
        confirmed_edges.add(out_edge)
        child_confirmed_edges = get_confirmed_edges_for_node(
            graph, out_edge[1], include_choice=include_choice,
            _traversed=_traversed, _traversed_to_update=_traversed_to_update, _walked_hit=_walked_hit, cache=cache)

        confirmed_edges |= child_confirmed_edges

        # If the target node lead to a (downstream) "walked hit"
        if out_edge[1] in _walked_hit:

            # Extend the update list to include the current edge target
            hit_nodes = _walked_hit[out_edge[1]]
            for tgt_nodes in hit_nodes:
                _traversed_to_update.append((out_edge[1], tgt_nodes))

            # Propagate "walked hit" nodes upstream
            _walked_hit[out_edge[1]] = hit_nodes = [nd for nd in hit_nodes if nd != node]
            if len(hit_nodes) > 0:
                if node not in _walked_hit:
                    _walked_hit[node] = []
                _walked_hit[node] += hit_nodes

    # Update the traversed edges for this node
    _traversed[node] = confirmed_edges

    # Update traversed edges for nodes part of a loop
    for tgt_node, src_node in _traversed_to_update:
        _traversed[tgt_node].update(_traversed[src_node])

    # Update cache only if this was the originally-requested start node
    if conf_edges_cache is not None and is_request_start:
        for start_node, edges in _traversed.items():
            if start_node not in conf_edges_cache:
                conf_edges_cache[start_node] = edges

    return confirmed_edges


def get_unconnected_connectors(graph: nx.MultiDiGraph, start_nodes: Set[ADSGNode], stop_at_one: bool = False)\
        -> List[ConnectorNode]:

    checked = {}
    unconnected_connectors = []
    for connector_node in get_nodes_by_subtype(graph, ConnectorNode):
        base_conn_node = connector_node
        for edge in iter_out_edges(graph, connector_node, edge_type=EdgeType.DERIVES):
            if isinstance(edge[1], ConnectorDegreeGroupingNode):
                base_conn_node = edge[1]
                break

        is_out_conn = True
        for edge in iter_in_edges(graph, base_conn_node):
            if get_edge_type(edge) == EdgeType.CONNECTS:
                is_out_conn = False
                break

        next_node = None
        for nested_edge in (iter_out_edges(graph, base_conn_node, edge_type=EdgeType.CONNECTS)
                            if is_out_conn else iter_in_edges(graph, base_conn_node, edge_type=EdgeType.CONNECTS)):
            next_node = nested_edge[1] if is_out_conn else nested_edge[0]
            break

        # next_node = self.next(connector_node, include_decisions=True)
        # if len(next_node) == 1 and isinstance(next_node[0], ConnectorDegreeGroupingNode):
        #     base_conn_node = next_node[0]
        #     next_node = self.next(next_node[0], include_decisions=True)

        if base_conn_node in checked:
            if checked[base_conn_node] and base_conn_node is not connector_node:
                unconnected_connectors.append(connector_node)
            continue

        # # If there is still a decision to be taken, in principle everything should be fine
        if next_node is not None and isinstance(next_node, ConnectionChoiceNode):
            conn_deg = get_out_degree(graph, next_node, edge_type=EdgeType.CONNECTS) \
                if is_out_conn else get_in_degree(graph, next_node, edge_type=EdgeType.CONNECTS)

            if not base_conn_node.is_valid(0) and conn_deg == 0 and \
                    not has_conditional_existence(graph, start_nodes, base_conn_node):
                unconnected_connectors.append(connector_node)
                if stop_at_one:
                    return unconnected_connectors
                checked[base_conn_node] = True
                continue

        else:
            conn_deg = get_out_degree(graph, base_conn_node, edge_type=EdgeType.CONNECTS) \
                if is_out_conn else get_in_degree(graph, base_conn_node, edge_type=EdgeType.CONNECTS)
            if not base_conn_node.is_valid(conn_deg):
                unconnected_connectors.append(connector_node)
                if stop_at_one:
                    return unconnected_connectors
                checked[base_conn_node] = True
                continue

        checked[base_conn_node] = False

    return unconnected_connectors


def traverse_until_choice_nodes(graph: nx.MultiDiGraph, start_nodes: Set[ADSGNode], traversed: set = None) \
        -> Tuple[Set[ADSGNode], Set[ChoiceNode]]:

    if traversed is None:
        traversed = set(start_nodes)

    # Get next step nodes
    next_nodes = set()
    for node in start_nodes:
        next_nodes |= {edge[1] for edge in iter_out_edges(graph, node)
                       if get_edge_type(edge) in [EdgeType.DERIVES, EdgeType.CONNECTS]}
    next_nodes -= traversed
    traversed |= next_nodes

    # Get choice nodes
    choice_nodes = set()
    non_decision_nodes = set()
    for node in next_nodes:
        if isinstance(node, ChoiceNode):
            choice_nodes.add(node)
        else:
            non_decision_nodes.add(node)

    # Recursively get choice nodes
    if len(non_decision_nodes) > 0:
        next_non_decision_nodes, next_decision_nodes = \
            traverse_until_choice_nodes(graph, non_decision_nodes, traversed=traversed)
        non_decision_nodes |= next_non_decision_nodes
        choice_nodes |= next_decision_nodes

    return (non_decision_nodes | start_nodes), choice_nodes


def get_non_confirmed_nodes(graph: nx.MultiDiGraph, start_nodes: Set[ADSGNode]) -> Set[ADSGNode]:

    # Get confirmed nodes
    confirmed_nodes, _ = traverse_until_choice_nodes(graph, set(start_nodes))

    # Get non-confirmed nodes
    non_confirmed_nodes = set(graph.nodes) - confirmed_nodes
    return non_confirmed_nodes


def iter_in_edges_cached(graph: nx.MultiDiGraph, node: 'ADSGNode', edge_type: EdgeType = None,
                         cache=None) -> Iterator['EdgeTuple']:

    if cache is None:
        yield from iter_in_edges(graph, node, edge_type=edge_type)
        return

    cache['iter_in'] = iter_in_cache = cache.get('iter_in', {})

    in_edges = iter_in_cache.get(node)
    if in_edges is not None:
        yield from in_edges
        return

    iter_in_cache[node] = in_edges = list(iter_in_edges(graph, node))
    yield from in_edges


def iter_out_edges_cached(graph: nx.MultiDiGraph, node: 'ADSGNode', edge_type: EdgeType = None,
                          cache=None) -> Iterator['EdgeTuple']:

    if cache is None:
        yield from iter_out_edges(graph, node, edge_type=edge_type)
        return

    cache['iter_out'] = iter_out_cache = cache.get('iter_out', {})

    out_edges = iter_out_cache.get(node)
    if out_edges is not None:
        yield from out_edges
        return

    iter_out_cache[node] = out_edges = list(iter_out_edges(graph, node))
    yield from out_edges
