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
from adsg_core.graph.adsg import *
from adsg_core.graph.traversal import *
from adsg_core.graph.adsg_nodes import *
from adsg_core.graph.graph_edges import *

__all__ = ['BasicADSG', 'EdgeType', 'CDVNode', 'ChoiceConstraint', 'ChoiceConstraintType', 'ADSGType', 'ConnNodes']

ConnNodes = List[Union[ConnectorNode, Tuple[ConnectorDegreeGroupingNode, List[ConnectorNode]]]]


class BasicADSG(ADSG):
    """
    Implementation of the ADSG with some helper functions:

    - add_edge and add_edges for adding nodes and edges to the graph
    - add_selection_choice for adding edges representing a selection choice
    - add_connection_choice for adding edges representing a connection choice
    - set_start_nodes for setting the nodes where derivation starts

    It is recommended to use these helper functions to build and initialize the graph:

    1. Add edges and choices (add_edge, add_edges, add_selection_choice, add_connection_choice)
    2. Set start nodes (set_start_nodes)
    3. Initialize choices (initialize_choices)
    """

    def __init__(self, *args, start_nodes=None, **kwargs):
        self._start_nodes = start_nodes
        super().__init__(*args, **kwargs)

    def _mod_graph_adjust_kwargs(self, kwargs):
        kwargs['start_nodes'] = self._start_nodes

    def _mod_graph_inplace(self, kwargs):
        if 'start_nodes' in kwargs:
            self._start_nodes = kwargs['start_nodes']

    def _get_derivation_start_nodes(self) -> Optional[Set[ADSGNode]]:
        return self._start_nodes

    def set_start_nodes(self, start_nodes: Set[ADSGNode] = None, initialize_choices=True):
        """
        Defines which nodes should be used for starting the existence derivations. If not explicitly defined, uses all
        nodes with no incoming edges.

        Nodes that cannot be derived from any of the starting nodes are removed from the graph.
        """
        if start_nodes is None:
            start_nodes = self._get_alternative_start_nodes()
        if len(start_nodes) == 0:
            raise ValueError('Provide at least one starting node!')
        self._start_nodes = start_nodes

        # Remove nodes that are not defined by any of the start nodes
        adsg = self
        removed_edges, removed_nodes = set(), set()
        for floating_node in self._get_floating_nodes():
            if floating_node in start_nodes:
                continue
            removed_nodes.add(floating_node)

            derived_edges, derived_nodes = get_derived_edges_for_node(
                adsg.graph, floating_node, start_nodes, removed_edges=removed_edges, removed_nodes=removed_nodes)
            removed_edges |= derived_edges
            removed_nodes |= derived_nodes

        if len(removed_edges) > 0 or len(removed_nodes) > 0:
            adsg = adsg.get_for_adjusted(removed_edges=removed_edges, removed_nodes=removed_nodes)

        if initialize_choices:
            return adsg.initialize_choices()
        return adsg

    def _get_alternative_start_nodes(self) -> Set[ADSGNode]:
        return self._get_floating_nodes()

    def _get_floating_nodes(self) -> Set[ADSGNode]:
        floating_nodes = set()
        for node in self._graph.nodes:
            if get_in_degree(self._graph, node) == 0:
                floating_nodes.add(node)
        return floating_nodes

    def _choice_sort_key(self, choice_node: CDVNode) -> tuple:
        type_idx = {SelectionChoiceNode: 0, ConnectionChoiceNode: 1, DesignVariableNode: 2}.get(type(choice_node), 3)
        return type_idx, choice_node.decision_id, getattr(choice_node, 'decision_sort_key', '')

    def add_edge(self, src: ADSGNode, tgt: ADSGNode, edge_type: EdgeType = EdgeType.DERIVES):
        """Add a directed edge between some source and target nodes"""
        self.add_edges([(src, tgt)], edge_type=edge_type)

    def add_edges(self, edges: List[Tuple[ADSGNode, ADSGNode]], edge_type: EdgeType = EdgeType.DERIVES):
        """Add multiple edges at a time"""
        for src, tgt in edges:
            add_edge(self._graph, src, tgt, edge_type=edge_type)

    def add_selection_choice(self, choice_id: str, originating_node: ADSGNode, option_nodes: List[ADSGNode]) \
            -> SelectionChoiceNode:
        """
        A selection choice is a choice between one or more mutually-exclusive option nodes. When choosing one of the
        options, the choice node is removed, and the originating node is connected to the chosen option node.
        """
        choice_node = SelectionChoiceNode(choice_id)

        edges = [(originating_node, choice_node)]
        for i, option_node in enumerate(option_nodes):
            if option_node.option_id is None:
                option_node.option_id = i

            edges.append((choice_node, option_node))

        self.add_edges(edges)
        return choice_node

    def add_connection_choice(self, choice_id: str, src_nodes: ConnNodes, tgt_nodes: ConnNodes,
                              exclude: List[Tuple[ConnectorNode, ConnectorNode]] = None) -> ConnectionChoiceNode:
        """
        A connection choice is a choice on how to connect one or more source nodes to one or more target nodes.
        Each connector node can place constraints on the accepted connection degree: one or more specific numbers,
        a range, or a lower limit. Additionally, repeated connections between the same nodes can be allowed or not.

        To model the non-influence of connection order, connector nodes can be grouped by a ConnectorDegreeGroupingNode,
        which as connection degree will have the sum of the connection degrees from underlying nodes, depending on
        whether they exist in an architecture or not.
        """
        graph = self._graph
        choice_node = ConnectionChoiceNode(choice_id)
        grouping_nodes: List[ConnectorDegreeGroupingNode] = []

        def _get_choice_nodes(conn_nodes: ConnNodes):
            if len(conn_nodes) == 0:
                raise ValueError('At least one source and target connector should be fined for connection choice!')

            choice_conn_nodes = []
            for conn_node in conn_nodes:
                if isinstance(conn_node, tuple):
                    if len(conn_node) != 2:
                        raise ValueError(f'Expecting a length-2 tuple: {conn_node!r}')

                    # Grouping nodes are represented by DERIVES edges from the underlying nodes to the grouping node
                    conn_grouping_node, underlying_conn_nodes = conn_node

                    if not isinstance(conn_grouping_node, ConnectorDegreeGroupingNode):
                        raise ValueError(f'Expecting a connector grouping node: {conn_grouping_node!r}')
                    choice_conn_nodes.append(conn_grouping_node)
                    grouping_nodes.append(conn_grouping_node)

                    for underlying_conn_node in underlying_conn_nodes:
                        if not isinstance(underlying_conn_node, ConnectorNode):
                            raise ValueError(f'Expecting a connector node: {underlying_conn_node!r}')

                        add_edge(graph, underlying_conn_node, conn_grouping_node, edge_type=EdgeType.DERIVES)

                else:
                    if not isinstance(conn_node, ConnectorNode):
                        raise ValueError(f'Expecting a connector node: {conn_node!r}')
                    choice_conn_nodes.append(conn_node)

            return choice_conn_nodes

        # Add CONNECTS edges from source to choice to target nodes
        for src_node in _get_choice_nodes(src_nodes):
            add_edge(graph, src_node, choice_node, edge_type=EdgeType.CONNECTS)
        for tgt_node in _get_choice_nodes(tgt_nodes):
            add_edge(graph, choice_node, tgt_node, edge_type=EdgeType.CONNECTS)

        # Add exclusion edges
        if exclude is not None:
            for src, tgt in exclude:
                add_edge(graph, src, tgt, edge_type=EdgeType.EXCLUDES)

        # Update connection degrees for grouping nodes
        for grouping_node in grouping_nodes:
            grouping_node.update_deg(graph)

        return choice_node

    def next(self, node, edge_type: EdgeType = None):
        """Iterate over outgoing nodes, optionally filtering by edge type"""
        for edge in iter_out_edges(self._graph, node, edge_type=edge_type):
            yield edge[1]

    def derived_nodes(self, node):
        """Get all nodes derived by this node"""
        cache = {}
        _, nodes = get_derived_edges_for_node(self._graph, node, start_nodes=None, cache=cache)
        return nodes

    def prev(self, node, edge_type: EdgeType = None):
        """Iterate over incoming nodes, optionally filtering by edge type"""
        for edge in iter_in_edges(self._graph, node, edge_type=edge_type):
            yield edge[0]
