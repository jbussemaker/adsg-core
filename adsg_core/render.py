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
import tempfile
import webbrowser
import numpy as np
from adsg_core.graph.adsg import *
from adsg_core.optimization.graph_processor import GraphProcessor

__all__ = ['ADSGRenderer']


class ADSGRenderer:
    """Utility class for rendering and displaying ADSGs"""

    def __init__(self, adsg: ADSGType, title=None):
        self._adsg = adsg
        self._title = title if title is not None else 'ADSG'

    def render(self, path=None, title=None):
        """
        Render the ADSG and display it in a Jupyter notebook.
        """
        self._render(self._adsg, title=title or self._title, path=path)

    def render_all_instances(self, idx=None, title=None):
        from IPython.display import display, Markdown

        processor = GraphProcessor(self._adsg)
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
    def _render(cls, adsg: ADSGType, title, path=None):
        """
        Render the ADSG and display it in a Jupyter notebook.
        """

        # Render ADSG to dot
        buffer = io.StringIO()
        adsg.export_dot(buffer)
        buffer.seek(0)
        dot_contents = buffer.read()

        # Wrap in HTML and display
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
