import warnings
import numpy as np
from typing import *
import networkx as nx
from adsg_core.graph.adsg_nodes import *
from adsg_core.graph.graph_edges import *

__all__ = ['export_gml', 'export_dot', 'export_drawio']


def export_gml(graph: nx.MultiDiGraph, path: str):
    nx.write_gml(graph, path, stringizer=str)


def export_dot(graph: nx.MultiDiGraph, path: str):
    graph_unique_node_ids = nx.MultiDiGraph()

    shape_map = {
        NodeExportShape.CIRCLE: 'ellipse',
        NodeExportShape.ROUNDED_RECT: 'rect',
        NodeExportShape.HEXAGON: 'hexagon',
    }
    node_map = {}

    def get_node(node: ADSGNode, node_id):
        if node_id not in node_map:
            color = node.get_export_color()
            graph_unique_node_ids.add_node(
                node_id, label='"' + str(node.get_export_title()) + '"',
                style='filled', fillcolor=color,
                shape=shape_map.get(node.get_export_shape(), 'ellipse'),
            )

        return node_id

    i = 0
    shown_incompatibilities = set()
    node_id_map = {}
    u: ADSGNode
    v: ADSGNode
    for u, v, k, d in graph.edges(keys=True, data=True):
        if u not in node_id_map:
            node_id_map[u] = i
            i += 1
        if v not in node_id_map:
            node_id_map[v] = i
            i += 1

        u_node = get_node(u, node_id_map[u])
        v_node = get_node(v, node_id_map[v])

        attr = d.copy()
        edge_type = get_edge_type((u, v, k, d))
        if edge_type == EdgeType.INCOMPATIBILITY:
            attr['color'] = 'red'
            attr['arrowhead'] = 'none'
            attr['constraint'] = 'false'

            if (u, v) in shown_incompatibilities:
                continue
            shown_incompatibilities |= {(u, v), (v, u)}

        elif edge_type == EdgeType.CONNECTS:
            attr['style'] = 'dashed'
        elif edge_type == EdgeType.EXCLUDES:
            attr['style'] = 'dashed'
            attr['color'] = 'red'

        edge_str = None
        if isinstance(u, ConnectorNode):
            if isinstance(v, ConnectorDegreeGroupingNode) or edge_type == EdgeType.CONNECTS:
                edge_str = u.get_full_deg_str()
        elif isinstance(v, ConnectorNode) and edge_type == EdgeType.CONNECTS:
            edge_str = v.get_full_deg_str()
        if edge_str is not None:
            attr['label'] = '"'+edge_str+'"'

        graph_unique_node_ids.add_edge(u_node, v_node, key=k, **attr)

    warnings.filterwarnings('ignore', message=r'.*write\_dot.*', category=PendingDeprecationWarning)
    nx.nx_pydot.write_dot(graph_unique_node_ids, path)


def export_drawio(graph: nx.MultiDiGraph, path: str, start_nodes: Set[ADSGNode] = None):
    from lxml.builder import E
    import lxml.etree as etree

    if start_nodes is None:
        start_nodes = set()

    shape_map = {
        NodeExportShape.CIRCLE: 'ellipse',
        NodeExportShape.ROUNDED_RECT: 'rounded=1',
        NodeExportShape.HEXAGON: 'shape=hexagon;perimeter=hexagonPerimeter2;fixedSize=1;size=10',
    }

    cells = [
        E.mxCell(id="0"), E.mxCell(id="1", parent="0"),
    ]

    cell_id = 2
    cell_id_map = {}
    height = 20
    geom = {'as': 'geometry'}
    for node in graph.nodes:
        if not isinstance(node, ADSGNode):
            continue

        style = [
            shape_map[node.get_export_shape()], f'fillColor={node.get_export_color()}',
            'whiteSpace=wrap', 'html=1',
        ]
        if node in start_nodes:
            style.append('strokeWidth=3')

        x = np.round(np.random.random()*50)*10
        y = np.round(np.random.random()*50)*10
        width = height
        if node.get_export_shape() == NodeExportShape.HEXAGON:
            width = 1.5*height

        cells.append(E.mxCell(
            E.mxGeometry(x=str(x), y=str(y), width=str(width), height=str(height), **geom),
            id=str(cell_id), value=node.get_export_title(), style=';'.join(style), vertex='1', parent='1',
        ))
        cell_id_map[node] = str(cell_id)
        cell_id += 1

    for edge in iter_edges(graph):
        src, tgt = edge[0], edge[1]
        if src not in cell_id_map or tgt not in cell_id_map:
            continue
        src_id, tgt_id = cell_id_map[src], cell_id_map[tgt]
        edge_type = get_edge_type(edge)

        style = [
            'rounded=1', 'orthogonalLoop=1', 'jettySize=auto', 'html=1',
        ]
        if edge_type in [EdgeType.CONNECTS, EdgeType.EXCLUDES]:
            style.append('dashed=1')
        if edge_type in [EdgeType.INCOMPATIBILITY, EdgeType.EXCLUDES]:
            style.append('strokeColor=#ff0000')
        if edge_type == EdgeType.INCOMPATIBILITY:
            style.append('endArrow=none;endFill=0')

        edge_str = None
        if isinstance(src, ConnectorNode):
            if isinstance(tgt, ConnectorDegreeGroupingNode) or edge_type == EdgeType.CONNECTS:
                edge_str = src.get_full_deg_str()
        elif isinstance(tgt, ConnectorNode) and edge_type == EdgeType.CONNECTS:
            edge_str = tgt.get_full_deg_str()

        cells.append(E.mxCell(
            E.mxGeometry(relative='1', **geom),
            id=str(cell_id), style=';'.join(style), edge='1', parent='1', source=src_id, target=tgt_id,
        ))
        edge_id = cell_id
        cell_id += 1

        if edge_str is not None:
            style = 'edgeLabel;html=1;align=center;verticalAlign=middle;resizable=0'
            cells.append(E.mxCell(
                E.mxGeometry(relative='1', **geom),
                id=str(cell_id), value=edge_str, style=style, parent=str(edge_id), vertex='1', connectable='0',
            ))
            cell_id += 1

    root = etree.ElementTree(E.mxGraphModel(
        E.root(*cells),
    ))
    with open(path, 'wb') as fp:
        fp.write(etree.tostring(root, encoding='utf-8', pretty_print=True))
