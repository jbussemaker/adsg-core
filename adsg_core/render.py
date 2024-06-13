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
import uuid
import json
import tempfile
import webbrowser
import numpy as np
from typing import *
from enum import Enum, auto
from adsg_core.graph.export import *
from adsg_core.graph.adsg_basic import *
from adsg_core.graph.adsg_nodes import *
from adsg_core.graph.graph_edges import *
from adsg_core.graph.choice_constraints import *
from adsg_core.optimization.graph_processor import GraphProcessor

__all__ = ['DSGRenderer', 'Legend']


class Legend(Enum):
    NODES = auto()
    EDGES = auto()
    CHOICES = auto()
    METRICS = auto()

    NODE = auto()
    NODE_START = auto()
    NODE_CONNECTOR = auto()

    EDGE_DERIVE = auto()
    EDGE_CONNECT = auto()
    EDGE_INCOMP = auto()
    EDGE_EXCLUDE = auto()

    CHOICE_SEL = auto()
    CHOICE_CONN = auto()

    METRICS_DV = auto()
    METRICS_OUTPUTS = auto()

    CHOICE_CONSTRAINT = auto()


class DSGRenderer:
    """Utility class for rendering and displaying DSGs"""

    def __init__(self, dsg: DSGType, title=None):
        self._dsg = dsg
        self._title = title if title is not None else 'DSG'

    def render(self, path=None, title=None):
        """
        Render the DSG and display it in a Jupyter notebook.
        """
        self._render(self._dsg, title=title or self._title, path=path)

    def render_all_instances(self, idx=None, title=None):
        from IPython.display import display, Markdown

        processor = GraphProcessor(self._dsg)
        x_all, _ = processor.get_all_discrete_x()

        n_total = x_all.shape[0]
        idx_all = np.arange(n_total)
        if idx is not None:
            x_all = x_all[idx, :]
            idx_all = idx_all[idx]
            display(Markdown(f'Rendering {x_all.shape[0]} of {n_total} instances'))
        else:
            display(Markdown(f'Rendering {n_total} instances'))

        if title is None:
            title = self._title
        for i, xi in enumerate(x_all):
            graph, _, _ = processor.get_graph(xi)
            self._render(graph, title=f'{title} [{idx_all[i]+1}/{x_all.shape[0]}]')

    @classmethod
    def _render(cls, dsg: DSGType, title, path=None):
        """
        Render the DSG and display it in a Jupyter notebook.
        """
        dot_contents = dsg.export_dot()
        cls._render_dot(dot_contents, title, path=path)

    @classmethod
    def _render_dot(cls, dot_contents, title, path=None):
        dot_html = cls._render_html(dot_contents)
        if cls._running_in_ipython():
            cls._display_ipython(dot_html)
        else:
            cls._display_browser(dot_html, title, path=path)

    @staticmethod
    def _render_html(dot):
        div_id = uuid.uuid4().hex
        return f"""<div id="{div_id}"></div>
<script type="text/javascript" src="https://cdn.jsdelivr.net/npm/@viz-js/viz/lib/viz-standalone.js"></script>
<script type="text/javascript">
(function() {{
  var dot = {json.dumps(dot)}; // Export of the dot graph notation
  // Create viz-js instance and render to SVG
  function doRender() {{
    Viz.instance().then(function(viz) {{ document.getElementById("{div_id}").appendChild(viz.renderSVGElement(dot)); }});
  }}
  // We may need to wait for loading to complete
  function checkRender() {{
    if (typeof Viz === "undefined") {{ setTimeout(checkRender, 200); }} else {{ doRender(); }} 
  }}
  checkRender();
}})()
</script>
"""

    @staticmethod
    def _running_in_ipython():
        try:
            from IPython.core.interactiveshell import InteractiveShell
            return InteractiveShell.initialized()
        except ModuleNotFoundError:
            return False

    @staticmethod
    def _display_ipython(dot_html):
        from IPython.display import display, HTML
        display(HTML(dot_html))

    @staticmethod
    def _display_browser(dot_html, title, path=None):
        full_html = f"""<!doctype html>
<html><head><title>{title}</title></head>
<body>{dot_html}</body></html>"""

        if path is None:
            with tempfile.NamedTemporaryFile('w', delete=False, suffix='.html') as fp:
                fp.write(full_html)
                url = f'file://{fp.name}'
        else:
            with open(path, 'w') as fp:
                fp.write(full_html)
                url = f'file://{path}'

        webbrowser.open(url)

    @classmethod
    def render_legend(cls, path=None, elements: Set[Union[str, Legend]] = None):
        dot_contents = cls._render_legend_dot(elements=elements).to_string()
        cls._render_dot(dot_contents, 'DSG Legend', path=path)

    @classmethod
    def _render_legend_dot(cls, elements: Set[Legend] = None):
        dsg = cls._get_legend_dsg(elements=elements)
        dot_graph = export_dot(dsg.graph, start_nodes=dsg.derivation_start_nodes,
                               choice_constraints=dsg.get_choice_constraints(), legend_mode=True, return_dot=True)

        dot_graph.obj_dict['attributes'].update(dict(
            label='DSG Legend',
            labelloc='t',
            fontsize='16pt',
            bgcolor='#f5f8fa',
        ))

        return dot_graph

    @staticmethod
    def _get_legend_dsg(elements: Set[Union[str, Legend]] = None):
        if elements is None:
            elements = {Legend.NODES, Legend.EDGES, Legend.CHOICES, Legend.METRICS, Legend.CHOICE_CONSTRAINT}
        else:
            elements = {el if isinstance(el, Legend) else Legend[el.upper()] for el in elements}
        if Legend.NODES in elements:
            elements |= {Legend.NODE, Legend.NODE_CONNECTOR, Legend.NODE_START}
        if Legend.EDGES in elements:
            elements |= {Legend.EDGE_DERIVE, Legend.EDGE_CONNECT, Legend.EDGE_INCOMP, Legend.EDGE_EXCLUDE}
        if Legend.CHOICES in elements:
            elements |= {Legend.CHOICE_SEL, Legend.CHOICE_CONN}
        if Legend.METRICS in elements:
            elements |= {Legend.METRICS_DV, Legend.METRICS_OUTPUTS}

        dsg = BasicDSG()
        dsg._start_nodes = set()

        if Legend.CHOICE_CONSTRAINT in elements:
            c1, c2 = SelectionChoiceNode('Choice 1'), SelectionChoiceNode('Choice 2')
            dsg.add_node(c1)
            dsg.add_node(c2)
            dsg.constrain_choices(ChoiceConstraintType.LINKED, [c1, c2])

        if Legend.NODE_CONNECTOR in elements:
            dsg.add_node(ConnectorNode('Connector'))
        if Legend.NODE in elements:
            dsg.add_node(NamedNode('Node'))

        # if Legend.CHOICE_CONSTRAINT in elements:
        #     for name, constraint in [
        #         ('Unordered\\nNon-replacing\\nConstraint', ChoiceConstraintType.UNORDERED_NOREPL),
        #         ('Unordered\\nConstraint', ChoiceConstraintType.UNORDERED),
        #         ('Permutation\\nConstraint', ChoiceConstraintType.PERMUTATION),
        #         ('Linked\\nConstraint', ChoiceConstraintType.LINKED),
        #     ]:
        #         src, opts = NamedNode(name), [NamedNode('Opt 1'), NamedNode('Opt 2')]
        #         dsg.constrain_choices(constraint, [
        #             dsg.add_selection_choice('Choice 1', src, opts),
        #             dsg.add_selection_choice('Choice 2', src, opts),
        #         ])

        if len(elements & {Legend.NODE_START, Legend.METRICS_DV, Legend.METRICS_OUTPUTS}) > 0:
            start_node = NamedNode('Start Node')
            dsg.add_node(start_node)
            dsg._start_nodes = {start_node}
            if Legend.METRICS_DV in elements:
                dsg.add_edge(start_node, DesignVariableNode('Continuous\\nDV', bounds=(0, 1)))
                dsg.add_edge(start_node, DesignVariableNode('Discrete\\nDV', options=['A', 'B', 'C']))
            if Legend.METRICS_OUTPUTS in elements:
                dsg.add_edge(start_node, MetricNode('Objective', direction=-1))
                dsg.add_edge(start_node, MetricNode('Constraint', direction=-1, ref=0))
                dsg.add_edge(start_node, MetricNode('Output Metric'))

        if Legend.CHOICE_CONN in elements:
            src_nodes = [
                ConnectorNode('Source 1', deg_list=[1]),
                ConnectorNode('Source 2', deg_list=[0, 1]),
                ConnectorNode('Source 2', deg_min=0, repeated_allowed=True),
            ]
            tgt_nodes = [
                ConnectorNode('Target 1', deg_min=1, repeated_allowed=True),
                ConnectorNode('Target 2', deg_min=0, deg_max=2),
            ]
            dsg.add_connection_choice(
                'Connection\\nChoice',
                src_nodes=src_nodes,
                tgt_nodes=tgt_nodes,
                exclude=[(src_nodes[0], tgt_nodes[0])] if Legend.EDGE_EXCLUDE in elements else [],
            )

        if Legend.CHOICE_SEL in elements:
            dsg.add_selection_choice('Selection\\nChoice', NamedNode('Source'),
                                      [NamedNode('Option 1'), NamedNode('Option 2'), NamedNode('Etc...')])

        if Legend.EDGE_INCOMP in elements:
            add_edge(dsg.graph, NamedNode('E'), NamedNode('F'), edge_type=EdgeType.INCOMPATIBILITY,
                     label='mutually\\nincompatible')
        if Legend.EDGE_CONNECT in elements:
            add_edge(dsg.graph, NamedNode('C'), NamedNode('D'), edge_type=EdgeType.CONNECTS, label='connects to')
        if Legend.EDGE_DERIVE in elements:
            add_edge(dsg.graph, NamedNode('A'), NamedNode('B'), edge_type=EdgeType.DERIVES, label='derives')

        return dsg
