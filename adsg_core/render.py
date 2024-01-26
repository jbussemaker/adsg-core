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
import io
import uuid
import json
from adsg_core.graph.adsg import *
from adsg_core.optimization.graph_processor import GraphProcessor


def render(adsg: ADSGType):
    """
    Render the ADSG and display it in a Jupyter notebook.
    """

    # Render ADSG to dot
    buffer = io.StringIO()
    adsg.export_dot(buffer)
    buffer.seek(0)
    dot_contents = buffer.read()

    # Wrap in HTML and display
    _display_ipython(_wrap_html(dot_contents))


def render_all_instances(adsg: ADSGType, idx=None):
    from IPython.display import display, Markdown

    processor = GraphProcessor(adsg)
    x_all, _ = processor.get_all_discrete_x()

    n_total = x_all.shape[0]
    if idx is not None:
        x_all = x_all[idx, :]
        display(Markdown(f'Rendering {x_all.shape[0]} of {n_total} instances'))
    else:
        display(Markdown(f'Rendering {n_total} instances'))

    for i, xi in enumerate(x_all):
        graph, _, _ = processor.get_graph(xi)
        render(graph)


def _wrap_html(dot):
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
  // We need to wait for loading to complete
  if (typeof Viz === "undefined") {{ setTimeout(doRender, 200); }} else {{ doRender(); }}
}})()
</script>
"""


def _display_ipython(html):
    from IPython.display import display, HTML
    display(HTML(html))
