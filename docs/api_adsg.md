# ADSG API Reference

::: adsg_core.graph.adsg_basic.BasicADSG
    handler: python
    options:
        inherited_members:
            - constrain_choices
            - add_incompatibility_constraint
            - get_ordered_next_choice_nodes
            - get_option_nodes
            - iter_possible_connection_edges
            - initialize_choices
            - get_for_apply_selection_choice
            - get_for_apply_connection_choice
            - des_var_nodes
            - set_des_var_value
            - des_var_value
            - get_nodes_by_subtype
            - derives
            - feasible
            - final
            - graph
            - export_dot
            - export_gml
            - export_drawio
            - render
            - render_all
            - copy

::: adsg_core.graph.adsg_nodes.ADSGNode
    handler: python

::: adsg_core.graph.adsg_nodes.NamedNode
    handler: python

::: adsg_core.graph.adsg_nodes.ConnectorNode
    handler: python
    options:
        members:
            - is_valid

::: adsg_core.graph.adsg_nodes.ConnectorDegreeGroupingNode
    handler: python

::: adsg_core.graph.adsg_nodes.DesignVariableNode
    handler: python

::: adsg_core.graph.adsg_nodes.MetricNode
    handler: python

::: adsg_core.graph.adsg_nodes.SelectionChoiceNode
    handler: python

::: adsg_core.graph.adsg_nodes.ConnectionChoiceNode
    handler: python
