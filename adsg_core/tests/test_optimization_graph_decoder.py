import copy
import pickle

import numpy as np
import pytest

from adsg_core.graph.adsg_basic import BasicDSG
from adsg_core.graph.adsg_nodes import ConnectorNode, DesignVariableNode, NamedNode
from adsg_core.graph.graph_edges import EdgeType, add_edge
from adsg_core.optimization.graph_decoder import (
    HAS_SB_ARCH_OPT_GRAPH,
    DSGGraphDecoder,
)
from adsg_core.optimization.graph_processor import GraphProcessor


pytestmark = pytest.mark.skipif(
    not HAS_SB_ARCH_OPT_GRAPH,
    reason="SBArchOpt graph support is not installed",
)


def test_dsg_graph_decoder():
    root = NamedNode("root")
    source = ConnectorNode("source", repeated_allowed=True)
    target = ConnectorNode("target", repeated_allowed=True)
    size = DesignVariableNode("size", bounds=(0.0, 10.0))
    material = DesignVariableNode("material", options=["a", "b"])

    dsg = BasicDSG()
    dsg.add_edges([(root, source), (root, target), (root, size), (root, material)])
    add_edge(dsg.graph, source, target, edge_type=EdgeType.CONNECTS)
    add_edge(dsg.graph, source, target, edge_type=EdgeType.CONNECTS)
    dsg = dsg.set_start_nodes({root})

    processor = GraphProcessor(dsg)
    decoder = DSGGraphDecoder(processor)
    values = {
        "size": 7.5,
        "material": 1,
    }
    x = np.asarray([[values[variable.name] for variable in processor.des_vars]])

    representation = decoder.decode(x)[0]

    assert all(isinstance(node, int) for node in representation.graph.nodes)
    assert set(decoder.labelings) == set(representation.node_labels)
    assert all(
        set(labels) == set(representation.graph.nodes)
        for labels in representation.node_labels.values()
    )
    assert [feature.name for feature in decoder.sizing_features] == [
        variable.name for variable in processor.des_vars
    ]
    sizing_by_name = dict(zip(
        [feature.name for feature in decoder.sizing_features],
        representation.sizing_values,
    ))
    assert sizing_by_name == {"material": 1.0, "size": 0.75}

    connects_edges = [
        edge
        for edge in representation.graph.edges(data=True)
        if edge[-1]["kind"] == "connects"
    ]
    assert len(connects_edges) == 2


def test_dsg_graph_decoder_exports_imputed_sizing_values():
    root = NamedNode("root")
    with_size = NamedNode("with-size")
    without_size = NamedNode("without-size")
    size = DesignVariableNode("size", bounds=(0.0, 10.0))

    dsg = BasicDSG()
    dsg.add_selection_choice("architecture", root, [with_size, without_size])
    dsg.add_edge(with_size, size)
    dsg = dsg.set_start_nodes({root})

    processor = GraphProcessor(dsg)
    decoder = DSGGraphDecoder(processor)

    def row(option, size_value=9.0):
        return [
            option if variable.name == "architecture" else size_value
            for variable in processor.des_vars
        ]

    with_size_representation, without_size_representation, equivalent_representation = decoder.decode(
        np.asarray([row(0), row(1), row(1, size_value=1.0)])
    )

    np.testing.assert_allclose(with_size_representation.sizing_values, [0.9])
    np.testing.assert_allclose(without_size_representation.sizing_values, [0.5])
    assert equivalent_representation is without_size_representation


def test_dsg_graph_decoder_composes_sizing_encoders():
    from sb_arch_opt.algo.arch_sbo.graph import GraphDecoder, SizingFeature

    class AttributeSizingEncoder:
        features = (SizingFeature("attribute", "categorical"),)

        def encode(self, graph, x_imputed):
            del graph, x_imputed
            return np.asarray([2.0])

    root = NamedNode("root")
    dsg = BasicDSG()
    dsg.add_node(root)
    dsg = dsg.set_start_nodes({root})

    decoder = DSGGraphDecoder(
        GraphProcessor(dsg),
        sizing_encoders=[AttributeSizingEncoder()],
    )
    representation = decoder.decode(np.empty((1, 0)))[0]

    assert GraphDecoder in DSGGraphDecoder.__bases__
    assert isinstance(decoder, GraphDecoder)
    assert decoder.sizing_features == (SizingFeature("attribute", "categorical"),)
    np.testing.assert_allclose(representation.sizing_values, [2.0])


def test_dsg_graph_decoder_copies_share_representations():
    root = NamedNode("root")
    dsg = BasicDSG()
    dsg.add_node(root)
    dsg = dsg.set_start_nodes({root})
    decoder = DSGGraphDecoder(GraphProcessor(dsg))
    copied = copy.deepcopy(decoder)

    assert copied is decoder
    representation = decoder.decode(np.empty((1, 0)))[0]

    assert copied.decode(np.empty((1, 0)))[0] is representation


def test_dsg_graph_decoder_labeling_flags():
    root = NamedNode("root")
    dsg = BasicDSG()
    dsg.add_node(root)
    dsg = dsg.set_start_nodes({root})
    decoder = DSGGraphDecoder(
        GraphProcessor(dsg),
        use_constant_labels=False,
    )

    assert decoder.constant_labeling in decoder.labelings
    assert decoder.dsg_type_labeling in decoder.labelings
    constant, dsg_type = pickle.loads(pickle.dumps(decoder.labelings))
    assert constant(root) == "node"
    assert dsg_type(root) == "named_node"
    assert [labeling.key for labeling in decoder.selected_labelings] == ["adsg_core.dsg"]
    decoder.use_constant_labels = True
    decoder.use_dsg_type_labels = False
    assert [labeling.key for labeling in decoder.selected_labelings] == ["adsg_core.constant"]

    representation = decoder.decode(np.empty((1, 0)))[0]
    assert set(representation.node_labels) == set(decoder.labelings)
