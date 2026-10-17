from adsg_core.graph.adsg_nodes import (
    ConnectorNode,
    DesignVariableNode,
    NamedNode,
    SelectionChoiceNode,
)


def test_generic_node_labelings():
    nodes_and_roles = [
        (NamedNode("node"), "named_node"),
        (ConnectorNode("port"), "connector"),
        (SelectionChoiceNode("choice"), "selection_choice"),
        (DesignVariableNode("size", bounds=(0.0, 1.0)), "design_variable"),
    ]

    for node, role in nodes_and_roles:
        assert node.get_dsg_label() == role
