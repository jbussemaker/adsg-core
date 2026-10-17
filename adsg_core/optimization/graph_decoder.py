"""
MIT License

Copyright: (c) 2026, Deutsches Zentrum fuer Luft- und Raumfahrt e.V.
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

from __future__ import annotations

from dataclasses import dataclass, field
from operator import methodcaller
from typing import Dict, Protocol, Sequence, Tuple

import networkx as nx
import numpy as np

from adsg_core.graph.adsg import DSGType
from adsg_core.graph.adsg_nodes import DesignVariableNode
from adsg_core.graph.graph_edges import get_edge_type
from adsg_core.optimization.graph_processor import GraphProcessor

try:
    from sb_arch_opt.algo.arch_sbo.graph import (
        GraphDecoder,
        GraphRepresentation,
        NodeLabeling,
        SizingFeature,
    )

    HAS_SB_ARCH_OPT_GRAPH = True
except ImportError:
    HAS_SB_ARCH_OPT_GRAPH = False

    class GraphDecoder:
        pass


__all__ = [
    "HAS_SB_ARCH_OPT_GRAPH",
    "DSGSizingEncoder",
    "DesignVariableSizingEncoder",
    "DSGGraphDecoder",
]


def _check_dependency():
    if not HAS_SB_ARCH_OPT_GRAPH:
        raise ImportError("SBArchOpt graph support is not installed")


class DSGSizingEncoder(Protocol):
    """Provide extra sizing values for each decoded DSG.

    Declare one feature for each value returned by ``encode``. The decoder
    combines the values from its sizing encoders in the order they are given.
    """

    @property
    def features(self) -> Sequence["SizingFeature"]:
        """Sizing features in the same order as the values from ``encode``."""
        ...

    def encode(self, graph: DSGType, x_imputed: Sequence[float]) -> np.ndarray:
        """Extract sizing values for one design.

        :param graph: The DSG generated for this design.
        :param x_imputed: Its corrected design vector, including imputed values
            for inactive variables.
        :return: One-dimensional array with one value per declared feature.
        """
        ...


class DesignVariableSizingEncoder:
    """Encode DSG design-variable node values as sizing features."""

    def __init__(self, processor: GraphProcessor):
        self._variables = tuple(
            (index, variable)
            for index, variable in enumerate(processor.des_vars)
            if isinstance(variable.node, DesignVariableNode)
        )
        _check_dependency()
        self._features = tuple(
            SizingFeature(
                variable.name,
                "numeric" if not variable.is_discrete or variable.is_ordinal else "categorical",
            )
            for _, variable in self._variables
        )

    @property
    def features(self):
        return self._features

    def encode(self, graph, x_imputed):
        del graph
        return np.asarray(
            [self._normalize(x_imputed[index], variable) for index, variable in self._variables],
            dtype=float,
        )

    @staticmethod
    def _normalize(value, variable):
        if not variable.is_discrete:
            lower, upper = variable.bounds
            return (float(value) - lower) / (upper - lower)
        if variable.is_ordinal:
            return 0.0 if variable.n_opts == 1 else float(value) / (variable.n_opts - 1)
        return float(value)


@dataclass
class _DSGDecoderCache:
    raw_to_corrected: Dict[Tuple[float, ...], Tuple[float, ...]] = field(default_factory=dict)
    representations: Dict[Tuple[float, ...], "GraphRepresentation"] = field(default_factory=dict)


class DSGGraphDecoder(GraphDecoder):
    """Decode design vectors into neutral SBArchOpt graph representations."""

    @staticmethod
    def _constant_label(_node):
        return "node"

    def __init__(
        self,
        processor: GraphProcessor,
        sizing_encoders: Sequence[DSGSizingEncoder] = None,
        *,
        use_constant_labels: bool = True,
        use_dsg_type_labels: bool = True,
    ):
        _check_dependency()
        self.processor = processor
        self.use_constant_labels = use_constant_labels
        self.use_dsg_type_labels = use_dsg_type_labels
        self.constant_labeling = NodeLabeling("adsg_core.constant", self._constant_label)
        self.dsg_type_labeling = NodeLabeling("adsg_core.dsg", methodcaller("get_dsg_label"))
        self.sizing_encoders = tuple(
            sizing_encoders
            if sizing_encoders is not None
            else (DesignVariableSizingEncoder(processor),)
        )
        self._cache = _DSGDecoderCache()

    def __deepcopy__(self, memo):
        memo[id(self)] = self
        return self

    @property
    def labelings(self):
        return self.constant_labeling, self.dsg_type_labeling

    @property
    def selected_labelings(self):
        """Labelings selected by the decoder's boolean options."""
        selected = []
        if self.use_constant_labels:
            selected.append(self.constant_labeling)
        if self.use_dsg_type_labels:
            selected.append(self.dsg_type_labeling)
        return tuple(selected)

    @property
    def sizing_features(self):
        return tuple(
            feature
            for encoder in self.sizing_encoders
            for feature in encoder.features
        )

    def decode(self, x: np.ndarray):
        representations = []
        for row in np.asarray(x):
            raw_key = tuple(np.asarray(row, dtype=float))
            corrected_key = self._cache.raw_to_corrected.get(raw_key)
            graph = None
            if corrected_key is None:
                graph, x_imputed, _ = self.processor.get_graph(row, create=False)
                corrected_key = tuple(x_imputed)
                self._cache.raw_to_corrected[raw_key] = corrected_key

            representation = self._cache.representations.get(corrected_key)
            if representation is None:
                if graph is None:
                    graph, _, _ = self.processor.get_graph(corrected_key)
                representation = self._representation(graph, corrected_key)
                self._cache.representations[corrected_key] = representation
            representations.append(representation)
        return representations

    def _representation(self, dsg: DSGType, x_imputed):
        source_graph = dsg.graph
        source_nodes = tuple(source_graph.nodes)
        node_ids = {node: index for index, node in enumerate(source_nodes)}

        graph = nx.MultiDiGraph()
        graph.add_nodes_from(node_ids.values())
        for source, target, key, data in source_graph.edges(keys=True, data=True):
            graph.add_edge(
                node_ids[source],
                node_ids[target],
                kind=get_edge_type((source, target, key, data)).name.lower(),
            )

        node_labels = {
            labeling: {
                node_ids[node]: labeling(node)
                for node in source_nodes
            }
            for labeling in self.labelings
        }
        sizing_values = np.concatenate([
            encoder.encode(dsg, x_imputed)
            for encoder in self.sizing_encoders
        ]) if self.sizing_encoders else np.empty((0,), dtype=float)

        return GraphRepresentation(
            graph=graph,
            node_labels=node_labels,
            sizing_values=sizing_values,
        )
