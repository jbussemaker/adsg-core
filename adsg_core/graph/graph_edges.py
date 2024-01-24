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
from typing import *
import networkx as nx
if False:
    from adsg_core.graph.adsg_nodes import *

__all__ = ['HashableDict', 'add_edge', 'get_edge', 'get_edge_for_type', 'get_edge_type', 'EdgeType',
           'iter_edges', 'iter_in_edges', 'iter_out_edges', 'get_edge_data']


class EdgeType(enum.Enum):
    DERIVES = 1
    CONNECTS = 2
    EXCLUDES = 3
    INCOMPATIBILITY = 4


class HashableDict(dict):
    """
    Hashable dict, from: https://stackoverflow.com/a/1151686
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._key_cache = None

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self._key_cache = None

    def __delitem__(self, key):
        super().__delitem__(key)
        self._key_cache = None

    def __key(self):
        if self._key_cache is None:
            self._key_cache = hash(tuple((k, self[k]) for k in sorted(self)))
        return self._key_cache

    def __hash__(self):
        if self._key_cache is not None:
            return self._key_cache
        return self.__key()

    def __eq__(self, other):
        return self.__key() == other.__key()


def add_edge(graph: nx.MultiDiGraph, from_node, to_node, key=None, is_conn=False, edge_type=None, **attr):
    if edge_type is not None:
        data = get_edge_for_type(from_node, to_node, edge_type, key=key, **attr)[-1]
    else:
        data = get_edge(from_node, to_node, key=key, is_conn=is_conn, **attr)[-1]
    graph.add_edge(from_node, to_node, key=key, **data)


def get_edge(from_node, to_node, key=None, is_conn=False, **attr):
    edge_type = EdgeType.CONNECTS if is_conn else EdgeType.DERIVES
    return get_edge_for_type(from_node, to_node, edge_type, key=key, **attr)


def get_edge_for_type(from_node, to_node, edge_type, key=None, **attr):
    # We need a hashable data dict here to be able to use edges in sets
    data = HashableDict(**attr)
    data['type'] = edge_type
    if key is None:
        return from_node, to_node, data
    return from_node, to_node, key, data


def get_edge_type(edge: 'EdgeTuple', default: Optional[EdgeType] = None) -> EdgeType:
    edge_data = edge[-1]
    # Faster than isinstance check
    if edge_data is None or (edge_data.__class__ != dict and edge_data.__class__ != HashableDict):
        return default

    if 'type' not in edge_data:
        if default is None:
            raise ValueError('Edge data not provided')
        return default
    return edge_data['type']


def iter_edges(graph: nx.MultiDiGraph) -> Iterator['EdgeTuple']:
    for edge in graph.edges(keys=True, data=True):
        # if isinstance(edge[3], dict):
        #     edge = (edge[0], edge[1], edge[2], HashableDict(**edge[3]))
        yield edge


def iter_in_edges(graph: nx.MultiDiGraph, node: 'ADSGNode', edge_type: EdgeType = None) -> Iterator['EdgeTuple']:
    for edge in graph.in_edges(node, keys=True, data=True):
        if edge_type is None or get_edge_type(edge) == edge_type:
            # if isinstance(edge[3], dict):
            #     edge = (edge[0], edge[1], edge[2], HashableDict(**edge[3]))
            yield edge


def iter_out_edges(graph: nx.MultiDiGraph, node: 'ADSGNode', edge_type: EdgeType = None) -> Iterator['EdgeTuple']:
    for edge in graph.out_edges(node, keys=True, data=True):
        if edge_type is None or get_edge_type(edge) == edge_type:
            # if isinstance(edge[3], dict):
            #     edge = (edge[0], edge[1], edge[2], HashableDict(**edge[3]))
            yield edge


def get_edge_data(edge: 'EdgeTuple') -> HashableDict:
    # if isinstance(edge[3], dict):
    #     return HashableDict(**edge[3])
    return edge[-1]
