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
import warnings
import itertools
import numpy as np
from typing import *
import networkx as nx
from io import BytesIO, StringIO
from collections import defaultdict
from adsg_core.graph.adsg_nodes import *
from adsg_core.graph.graph_edges import *
from adsg_core.graph.choice_constraints import *

__all__ = ['export_gml', 'export_dot', 'export_drawio']

CHOICE_CONSTRAINT_COLOR = '#9C27B0'


def export_gml(graph: nx.MultiDiGraph, path: str = None):
    fp = BytesIO() if path is None else path
    nx.write_gml(graph, fp, stringizer=str)
    return fp.getvalue().decode('utf-8') if path is None else None


def export_dot(graph: nx.MultiDiGraph, path=None, start_nodes: Set[DSGNode] = None,
               choice_constraints: List[ChoiceConstraint] = None, legend_mode=False, return_dot=False):
    graph_export = nx.DiGraph()

    if start_nodes is None:
        start_nodes = set()

    shape_map = {
        NodeExportShape.CIRCLE: 'ellipse',
        NodeExportShape.ROUNDED_RECT: 'rect',
        NodeExportShape.HEXAGON: 'hexagon',
    }
    node_map = {}

    def get_node(node: DSGNode, node_id):
        dot_node_id = f'leg_{node_id}' if legend_mode else node_id
        if node_id not in node_map:
            label = str(node.get_export_title())
            style = ['filled']
            if node in start_nodes:
                style.append('bold')
                label = label.replace(':', ';')
                label = f'<<B>{label}</B>>'
            else:
                label = '"'+label+'"'

            color = node.get_export_color()
            graph_export.add_node(
                dot_node_id, label=label,
                style='"'+','.join(style)+'"', fillcolor=color,
                shape=shape_map.get(node.get_export_shape(), 'ellipse'),
                margin=0 if legend_mode else 0.05,
            )

        return dot_node_id

    i = 0
    node_id_map = {}

    # Hack to place the choice constraints lower in the graph
    if legend_mode:
        for choice_constraint in (choice_constraints or []):
            for node in choice_constraint.nodes:
                if node not in node_id_map:
                    node_id_map[node] = i
                    i += 1
                get_node(node, node_id_map[node])

    shown_incompatibilities = set()
    repeated_conn_edges = defaultdict(int)
    u: DSGNode
    v: DSGNode
    for u, v, k, d in graph.edges(keys=True, data=True):
        if u not in node_id_map:
            node_id_map[u] = i
            i += 1
        if v not in node_id_map:
            node_id_map[v] = i
            i += 1

        u_node = get_node(u, node_id_map[u])
        v_node = get_node(v, node_id_map[v])

        attr = {}
        for key, value in d.items():
            value = str(value)
            if ':' in value:
                value = f'\"{value}\"'
            attr[key] = value

        edge_str = None
        edge_type = get_edge_type((u, v, k, d))
        if edge_type == EdgeType.INCOMPATIBILITY:
            attr['color'] = 'red'
            attr['arrowhead'] = 'none'
            if not legend_mode:
                attr['constraint'] = 'false'

            if (u, v) in shown_incompatibilities:
                continue
            shown_incompatibilities |= {(u, v), (v, u)}

        elif edge_type == EdgeType.CONNECTS:
            attr['style'] = 'dashed'
        elif edge_type == EdgeType.EXCLUDES:
            attr['style'] = 'dashed'
            attr['color'] = 'red'
            if legend_mode:
                edge_str = 'connection\\nexcluded'

        if isinstance(u, ConnectorNode):
            if ((edge_type == EdgeType.DERIVES and isinstance(v, ConnectorDegreeGroupingNode)) or
                    (edge_type == EdgeType.CONNECTS and isinstance(v, ConnectionChoiceNode))):
                edge_str = u.get_full_deg_str()
        elif isinstance(u, ConnectionChoiceNode) and edge_type == EdgeType.CONNECTS and isinstance(v, ConnectorNode):
            edge_str = v.get_full_deg_str()
        if edge_str is not None:
            attr['label'] = '"'+edge_str+'"'

        if edge_type == EdgeType.CONNECTS:
            repeated_conn_edges[(u_node, v_node)] += 1

        graph_export.add_edge(u_node, v_node, **attr)

    # Set text of repeated connection edges
    for (u_node, v_node), count in repeated_conn_edges.items():
        if count > 1:
            nx.set_edge_attributes(graph_export, {(u_node, v_node): {'label': f'"{count}x"'}})

    # Add individual nodes
    for node_ in graph.nodes:
        if node_ not in node_id_map:
            node_id_map[node_] = i
            i += 1

            get_node(node_, node_id_map[node_])

    # Add choice constraints
    for choice_constraint in (choice_constraints or []):
        for u, v in itertools.combinations(choice_constraint.nodes, 2):
            if u not in node_id_map or v not in node_id_map:
                continue
            u_node = get_node(u, node_id_map[u])
            v_node = get_node(v, node_id_map[v])

            attr = {
                'color': CHOICE_CONSTRAINT_COLOR,
                'style': 'dotted',
                'penwidth': '3',
                'arrowhead': 'none',
                'constraint': 'false',
                'label': CCT_EXPORT_LABEL.get(choice_constraint.type, choice_constraint.type.name),
            }
            if legend_mode:
                del attr['constraint']
                attr['label'] = 'choice\\nconstraint'
            graph_export.add_edge(u_node, v_node, **attr)

    warnings.filterwarnings('ignore', message=r'.*write\_dot.*', category=PendingDeprecationWarning)

    # Define graph attributes (weird notation but works)
    graph_export.graph['graph'] = dict(
        rankdir='LR',  # Arrange left-to-right (vs vertical)
        dpi='60',
        fontsize='20',
        **(dict(
            ranksep=0,
            nodesep=.1,
        ) if legend_mode else {}),
    )
    if legend_mode:
        graph_export.graph['node'] = dict(
            height=0,
        )

    dot_graph = nx.nx_pydot.to_pydot(graph_export)
    if return_dot:
        return dot_graph

    dot_export = dot_graph.to_string()
    if path is not None:
        with open(path, 'wb') as fp:
            fp.write(dot_export.encode('utf-8'))
    return dot_export


def export_drawio(graph: nx.MultiDiGraph, path: str = None, start_nodes: Set[DSGNode] = None,
                  choice_constraints: List[ChoiceConstraint] = None):
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
        if not isinstance(node, DSGNode):
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

    def _add_edge(src_id_, tgt_id_, style_, edge_str_):
        nonlocal cell_id
        cells.append(E.mxCell(
            E.mxGeometry(relative='1', **geom),
            id=str(cell_id), style=';'.join(style_), edge='1', parent='1', source=src_id_, target=tgt_id_,
        ))
        edge_id = cell_id
        cell_id += 1

        if edge_str_ is not None:
            style__ = 'edgeLabel;html=1;align=center;verticalAlign=middle;resizable=0'
            cells.append(E.mxCell(
                E.mxGeometry(relative='1', **geom),
                id=str(cell_id), value=edge_str_, style=style__, parent=str(edge_id), vertex='1', connectable='0',
            ))
            cell_id += 1

    default_edge_style = [
        'rounded=1', 'orthogonalLoop=1', 'jettySize=auto', 'html=1',
    ]
    for edge in iter_edges(graph):
        src, tgt = edge[0], edge[1]
        if src not in cell_id_map or tgt not in cell_id_map:
            continue
        src_id, tgt_id = cell_id_map[src], cell_id_map[tgt]
        edge_type = get_edge_type(edge)

        style = default_edge_style.copy()
        if edge_type in [EdgeType.CONNECTS, EdgeType.EXCLUDES]:
            style.append('dashed=1')
        if edge_type in [EdgeType.INCOMPATIBILITY, EdgeType.EXCLUDES]:
            style.append('strokeColor=#ff0000')
        if edge_type == EdgeType.INCOMPATIBILITY:
            style.append('endArrow=none;endFill=0')

        edge_str = None
        if isinstance(src, ConnectorNode):
            if ((edge_type == EdgeType.DERIVES and isinstance(tgt, ConnectorDegreeGroupingNode)) or
                    (edge_type == EdgeType.CONNECTS and isinstance(tgt, ConnectionChoiceNode))):
                edge_str = src.get_full_deg_str()
        elif isinstance(src, ConnectionChoiceNode) and edge_type == EdgeType.CONNECTS and isinstance(tgt, ConnectorNode):
            edge_str = tgt.get_full_deg_str()

        _add_edge(src_id, tgt_id, style, edge_str)

    # Add choice constraints
    for choice_constraint in (choice_constraints or []):
        for src, tgt in itertools.combinations(choice_constraint.nodes, 2):
            if src not in cell_id_map or tgt not in cell_id_map:
                continue
            src_id, tgt_id = cell_id_map[src], cell_id_map[tgt]

            style = default_edge_style.copy()
            style += [
                'dashed=1',
                'strokeColor='+CHOICE_CONSTRAINT_COLOR,
                'endArrow=none;endFill=0',
            ]
            label = CCT_EXPORT_LABEL.get(choice_constraint.type, choice_constraint.type.name)
            _add_edge(src_id, tgt_id, style, label)

    root = etree.ElementTree(E.mxGraphModel(
        E.root(*cells),
    ))

    xml_contents = etree.tostring(root, encoding='utf-8', pretty_print=True)
    if path is None:
        return xml_contents
    with open(path, 'wb') as fp:
        fp.write(xml_contents)
