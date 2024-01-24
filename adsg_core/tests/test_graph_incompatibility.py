import pytest
import itertools
from adsg_core.graph.traversal import *
from adsg_core.graph.adsg_basic import *
from adsg_core.graph.adsg_nodes import *
from adsg_core.graph.graph_edges import *
from adsg_core.optimization.graph_processor import *


def test_infeasible_incompatibility_constraint(n):
    adsg = BasicADSG()
    adsg.add_edges([(n[0], n[1]), (n[1], n[2]), (n[2], n[3])])
    adsg = adsg.set_start_nodes({n[0]}, initialize_choices=False)

    assert check_derives(adsg.graph, n[0], n[3])
    assert not check_derives(adsg.graph, n[3], n[0])
    assert adsg.feasible

    adsg.add_incompatibility_constraint([n[0], n[3]])
    assert check_derives(adsg.graph, n[0], n[3])
    assert not check_derives(adsg.graph, n[3], n[0])

    adsg_confirmed = adsg.get_confirmed_graph()
    assert n[0] in adsg_confirmed.graph.nodes
    assert n[3] in adsg_confirmed.graph.nodes

    confirmed_incompatibility_edges = adsg_confirmed.get_confirmed_incompatibility_edges()
    assert len(confirmed_incompatibility_edges) == 1
    edge = list(confirmed_incompatibility_edges)[0]
    assert edge[:2] == (n[0], n[3])

    assert adsg.has_confirmed_incompatibility_edges()
    assert not adsg.feasible

    adsg = adsg.initialize_choices()
    assert not adsg.feasible


def test_add_incompatibility_constraint_attr(n):
    adsg = BasicADSG()
    adsg.add_edges([(n[0], n[1]), (n[1], n[2])])
    adsg.add_incompatibility_constraint([n[0], n[1]], obj_ref=999)
    adsg = adsg.set_start_nodes({n[0]})

    found_edge = False
    for edge in iter_edges(adsg.graph):
        if get_edge_type(edge) == EdgeType.INCOMPATIBILITY:
            edge_data = get_edge_data(edge)
            assert 'obj_ref' in edge_data
            assert edge_data['obj_ref'] == 999
            found_edge = True
    assert found_edge


def test_incompatibility_constraint(n):
    adsg = BasicADSG()
    adsg.add_edges([
        (n[1], n[11]),
        (n[11], n[2]), (n[11], n[3]),
    ])
    adsg.add_selection_choice('C2', n[2], [n[21], n[22]])
    adsg.add_selection_choice('C3', n[3], [n[31], n[32]])
    adsg = adsg.set_start_nodes({n[1]}, initialize_choices=False)

    choice_adsg = adsg.copy().initialize_choices()
    assert len(choice_adsg.choice_nodes) == 2
    for i in range(2):
        choice_node = choice_adsg.get_ordered_next_choice_nodes()[0]
        assert isinstance(choice_node, SelectionChoiceNode)
        opt_nodes = choice_adsg.get_option_nodes(choice_node)
        assert len(opt_nodes) == 2
        adsg2 = choice_adsg.get_for_apply_selection_choice(choice_node, opt_nodes[i])

        choice_node = adsg2.get_ordered_next_choice_nodes()[0]
        assert isinstance(choice_node, SelectionChoiceNode)
        opt_nodes = adsg2.get_option_nodes(choice_node)
        assert len(opt_nodes) == 2
        for opt_node in opt_nodes:
            adsg3 = adsg2.get_for_apply_selection_choice(choice_node, opt_node)
            assert adsg3.feasible
            assert adsg3.final

    adsg.add_incompatibility_constraint([n[21], n[32]])

    adsg = adsg.initialize_choices()
    for i in range(2):
        choice_node = adsg.get_ordered_next_choice_nodes()[0]
        assert isinstance(choice_node, SelectionChoiceNode)
        opt_nodes = adsg.get_option_nodes(choice_node)
        assert len(opt_nodes) == 2
        adsg2 = adsg.get_for_apply_selection_choice(choice_node, opt_nodes[i])

        if i == 0:
            assert len(adsg2.get_taken_single_selection_choices()) == 1
            assert adsg2.get_ordered_next_choice_nodes() == []
            assert adsg2.final
            assert adsg2.feasible
            continue
        assert len(adsg2.get_taken_single_selection_choices()) == 0

        choice_node = adsg2.get_ordered_next_choice_nodes()[0]
        assert isinstance(choice_node, SelectionChoiceNode)
        opt_nodes = adsg2.get_option_nodes(choice_node)
        assert len(opt_nodes) == 2
        for opt_node in opt_nodes:
            adsg3 = adsg2.get_for_apply_selection_choice(choice_node, opt_node)
            assert len(adsg3.get_taken_single_selection_choices()) == 0
            assert adsg3.feasible
            assert adsg3.final


def test_infeasible_choice_incompatibility_constraint(n):
    adsg = BasicADSG()
    adsg.add_edges([
        (n[1], n[11]), (n[11], n[2]),
        (n[22], n[3]), (n[3], n[31]),
    ])
    adsg.add_selection_choice('C2', n[2], [n[21], n[22]])
    adsg.add_incompatibility_constraint([n[22], n[31]])

    adsg = adsg.set_start_nodes({n[1]})
    assert adsg.feasible
    assert len(adsg.choice_nodes) == 1

    choice_node = adsg.get_ordered_next_choice_nodes()[0]
    adsg2 = adsg.get_for_apply_selection_choice(choice_node, adsg.get_option_nodes(choice_node)[0])
    assert adsg2.feasible
    assert adsg2.final

    adsg3 = adsg.get_for_apply_selection_choice(choice_node, adsg.get_option_nodes(choice_node)[1])
    assert not adsg3.feasible
    assert adsg3.final


def test_strictly_mutually_exclusive_choice_options(n):
    adsg = BasicADSG()
    adsg.add_edges([
        (n[1], n[11]), (n[11], n[2]), (n[11], n[3]),
        (n[2], n[21]), (n[21], n[4]),
        (n[3], n[31]), (n[31], n[5]),
    ])
    adsg.add_selection_choice('C4', n[4], [n[41], n[42]])
    adsg.add_selection_choice('C5', n[5], [n[41], n[42]])

    adsg_free = adsg.set_start_nodes({n[1]}).copy()
    assert len(adsg_free.choice_nodes) == 2
    for i in range(2):
        choice_node = adsg_free.get_ordered_next_choice_nodes()[0]
        assert isinstance(choice_node, SelectionChoiceNode)
        opt_nodes = adsg_free.get_option_nodes(choice_node)
        assert len(opt_nodes) == 2
        adsg2 = adsg_free.get_for_apply_selection_choice(choice_node, opt_nodes[i])

        choice_node = adsg2.get_ordered_next_choice_nodes()[0]
        assert isinstance(choice_node, SelectionChoiceNode)
        opt_nodes = adsg2.get_option_nodes(choice_node)
        assert len(opt_nodes) == 2
        for opt_node in opt_nodes:
            adsg3 = adsg2.get_for_apply_selection_choice(choice_node, opt_node)
            assert adsg3.feasible
            assert adsg3.final

    adsg.add_incompatibility_constraint([n[41], n[42]])

    adsg = adsg.initialize_choices()
    assert len(adsg.choice_nodes) == 2
    for i in range(2):
        choice_nodes = adsg.get_ordered_next_choice_nodes()
        choice_node = choice_nodes[0]
        assert isinstance(choice_node, SelectionChoiceNode)
        opt_nodes = adsg.get_option_nodes(choice_node)
        assert len(opt_nodes) == 2
        adsg2 = adsg.get_for_apply_selection_choice(choice_node, opt_nodes[i])

        assert adsg2.get_taken_single_selection_choices()[0][0] == choice_nodes[1]
        assert adsg2.feasible
        assert adsg2.final


def test_downstream_incompatibility_constraint(n):
    adsg = BasicADSG()
    adsg.add_edges([
        (n[1], n[11]), (n[11], n[2]), (n[11], n[3]),
        (n[31], n[4]), (n[4], n[41]),
    ])
    adsg.add_selection_choice('C2', n[2], [n[21], n[22]])
    adsg.add_selection_choice('C3', n[3], [n[31], n[32]])

    adsg.add_incompatibility_constraint([n[21], n[41]])
    adsg = adsg.set_start_nodes({n[1]})

    choice_node = adsg.get_ordered_next_choice_nodes()[0]
    assert isinstance(choice_node, SelectionChoiceNode)
    opt_nodes = adsg.get_option_nodes(choice_node)
    assert opt_nodes[0] == n[21]

    adsg2 = adsg.get_for_apply_selection_choice(choice_node, opt_nodes[0])
    nodes = set(adsg2.graph.nodes)
    assert n[21] in nodes
    assert n[31] not in nodes
    assert n[41] not in nodes
    assert adsg2.final
    assert adsg2.feasible


def test_multi_choice_downstream_incompatibility_constraint(n):
    adsg = BasicADSG()
    adsg.add_edges([
        (n[1], n[11]), (n[11], n[2]), (n[11], n[3]),
        (n[32], n[4]),
        (n[41], n[5]), (n[42], n[5]), (n[5], n[45]),
    ])
    adsg.add_selection_choice('C2', n[2], [n[21], n[22]])
    adsg.add_selection_choice('C3', n[3], [n[31], n[32]])
    adsg.add_selection_choice('C4', n[4], [n[41], n[42]])

    adsg.add_incompatibility_constraint([n[21], n[45]])
    adsg = adsg.set_start_nodes({n[1]})

    choice_node = adsg.get_ordered_next_choice_nodes()[0]
    assert isinstance(choice_node, SelectionChoiceNode)
    opt_nodes = adsg.get_option_nodes(choice_node)
    assert opt_nodes[0] == n[21]

    adsg2 = adsg.get_for_apply_selection_choice(choice_node, opt_nodes[0])
    nodes = set(adsg2.graph.nodes)
    assert n[21] in nodes
    assert n[45] not in nodes
    assert n[41] not in nodes
    assert n[42] not in nodes
    assert n[32] not in nodes
    assert n[31] in nodes

    assert adsg2.final
    assert adsg2.feasible


def test_infeasible_multi_choice_downstream_incompatibility_constraint(n):
    adsg = BasicADSG()
    adsg.add_edges([
        (n[1], n[11]), (n[11], n[2]), (n[11], n[3]),
        (n[31], n[4]), (n[32], n[4]), (n[4], n[41]),
    ])
    adsg.add_selection_choice('C2', n[2], [n[21], n[22]])
    adsg.add_selection_choice('C3', n[3], [n[31], n[32]])

    adsg.add_incompatibility_constraint([n[21], n[41]])
    adsg = adsg.set_start_nodes({n[1]})

    choice_node = adsg.get_ordered_next_choice_nodes()[0]
    assert isinstance(choice_node, SelectionChoiceNode)
    opt_nodes = adsg.get_option_nodes(choice_node)
    assert opt_nodes[0] == n[21]

    adsg2 = adsg.get_for_apply_selection_choice(choice_node, opt_nodes[0])
    nodes = set(adsg2.graph.nodes)
    assert n[21] in nodes
    assert n[41] in nodes
    assert n[32] not in nodes
    assert n[31] not in nodes

    assert adsg2.final
    assert not adsg2.feasible


def test_multi_route_choice_remove_incompatibility_constraint(n):
    adsg = BasicADSG()
    adsg.add_edges([
        (n[1], n[11]), (n[11], n[2]),
        (n[22], n[3]),
        (n[31], n[4]), (n[32], n[5]),
        (n[41], n[6]), (n[42], n[6]),
        (n[5], n[45]), (n[45], n[7]),
        (n[6], n[46]), (n[46], n[7]),
        (n[7], n[47]),
    ])
    adsg.add_selection_choice('C2', n[2], [n[21], n[22]])
    adsg.add_selection_choice('C3', n[3], [n[31], n[32]])
    adsg.add_selection_choice('C4', n[4], [n[41], n[42]])

    adsg.add_incompatibility_constraint([n[21], n[47]])
    adsg = adsg.set_start_nodes({n[1]})

    choice_node = adsg.get_ordered_next_choice_nodes()[0]
    assert isinstance(choice_node, SelectionChoiceNode)
    opt_nodes = adsg.get_option_nodes(choice_node)
    assert opt_nodes[0] == n[21]

    adsg2 = adsg.get_for_apply_selection_choice(choice_node, opt_nodes[0])
    nodes = set(adsg2.graph.nodes)
    assert n[21] in nodes
    assert n[22] not in nodes
    assert n[41] not in nodes
    assert n[45] not in nodes
    assert n[47] not in nodes

    assert adsg2.final
    assert adsg2.feasible


def test_incompatibility_constraint_loop(n):
    adsg = BasicADSG()
    adsg.add_edges([
        (n[1], n[11]), (n[11], n[2]),
        (n[22], n[3]), (n[3], n[31]),
        (n[31], n[4]), (n[4], n[41]),
        (n[41], n[5]), (n[5], n[42]), (n[42], n[3]),
    ])
    adsg.add_selection_choice('C1', n[2], [n[21], n[22]])
    adsg.add_incompatibility_constraint([n[21], n[41]])
    adsg = adsg.set_start_nodes({n[1]})

    choice_node = adsg.get_ordered_next_choice_nodes()[0]
    assert isinstance(choice_node, SelectionChoiceNode)
    opt_nodes = adsg.get_option_nodes(choice_node)
    assert opt_nodes[0] == n[21]

    adsg2 = adsg.get_for_apply_selection_choice(choice_node, opt_nodes[0])
    nodes = set(adsg2.graph.nodes)
    assert n[21] in nodes
    assert n[22] not in nodes
    assert n[31] not in nodes
    assert n[41] not in nodes
    assert n[42] not in nodes

    assert adsg2.final
    assert adsg2.feasible


def test_incompatibility_constraint_linked_choice_strict(n):
    adsg = BasicADSG()
    adsg.add_edges([
        (n[1], n[11]), (n[11], n[2]), (n[11], n[3]),
        (n[2], n[21]), (n[21], n[4]),
        (n[3], n[31]), (n[31], n[5]),
    ])
    c1 = adsg.add_selection_choice('C4', n[4], [n[41], n[42]])
    c2 = adsg.add_selection_choice('C5', n[5], [n[41], n[42]])
    adsg = adsg.add_incompatibility_constraint([n[41], n[42]])
    adsg = adsg.set_start_nodes({n[1]})
    adsg = adsg.constrain_choices(ChoiceConstraintType.LINKED, [c1, c2])

    choice_node = adsg.get_ordered_next_choice_nodes()[0]
    assert isinstance(choice_node, SelectionChoiceNode)
    opt_nodes = adsg.get_option_nodes(choice_node)
    assert len(opt_nodes) == 2
    for opt_node in opt_nodes:
        adsg2 = adsg.get_for_apply_selection_choice(choice_node, opt_node)
        assert len(adsg2.get_taken_single_selection_choices()) == 1
        assert adsg2.feasible
        assert adsg2.final


def test_incompatibility_constraint_linked_choices_infeasible(n):
    adsg = BasicADSG()
    adsg.add_edges([
        (n[1], n[11]), (n[11], n[2]), (n[11], n[3]),
    ])
    c1 = adsg.add_selection_choice('C2', n[2], [n[21], n[22]])
    c2 = adsg.add_selection_choice('C3', n[3], [n[22], n[23]])
    adsg = adsg.add_incompatibility_constraint([n[21], n[22]])
    adsg = adsg.set_start_nodes({n[1]})

    adsg = adsg.constrain_choices(ChoiceConstraintType.LINKED, [c1, c2])

    choice_node = adsg.get_ordered_next_choice_nodes()[0]
    assert isinstance(choice_node, SelectionChoiceNode)
    opt_nodes = adsg.get_option_nodes(choice_node)
    assert len(opt_nodes) == 2
    for i, opt_node in enumerate(opt_nodes):
        adsg2 = adsg.get_for_apply_selection_choice(choice_node, opt_node)
        assert len(adsg2.get_taken_single_selection_choices()) == 1
        assert adsg2.feasible == (i == 1)
        assert adsg2.final


def test_two_incompatibility_constraints(n):
    adsg = BasicADSG()
    adsg.add_edges([
        (n[1], n[11]), (n[11], n[2]), (n[11], n[3]),
    ])
    adsg.add_selection_choice('C2', n[2], [n[21], n[22], n[23]])
    adsg.add_selection_choice('C3', n[3], [n[21], n[22], n[23]])
    adsg.add_incompatibility_constraint([n[21], n[22]])
    adsg.add_incompatibility_constraint([n[22], n[23]])
    adsg = adsg.set_start_nodes({n[1]})

    choice_node = adsg.get_ordered_next_choice_nodes()[0]
    assert isinstance(choice_node, SelectionChoiceNode)
    opt_nodes = adsg.get_option_nodes(choice_node)
    assert len(opt_nodes) == 3
    for i, opt_node in enumerate(opt_nodes):
        adsg2 = adsg.get_for_apply_selection_choice(choice_node, opt_node)

        if i == 1:
            assert adsg2.get_ordered_next_choice_nodes() == []
            assert adsg2.feasible
            assert adsg2.final
            continue

        choice_node2 = adsg2.get_ordered_next_choice_nodes()[0]
        opt_nodes2 = adsg2.get_option_nodes(choice_node2)
        assert len(opt_nodes2) == 2
        for opt_node2 in opt_nodes2:
            adsg3 = adsg2.get_for_apply_selection_choice(choice_node2, opt_node2)
            assert adsg3.feasible
            assert adsg3.final


def test_initially_infeasible_optimization(n):
    adsg = BasicADSG()
    adsg.add_edges([
        (n[1], n[2]), (n[2], n[3]),
    ])
    adsg.add_incompatibility_constraint([n[1], n[3]])
    adsg = adsg.set_start_nodes({n[1]})
    assert not adsg.feasible

    with pytest.raises(ValueError) as exception_info:
        GraphProcessor(adsg)
    assert 'not feasible' in str(exception_info.value)


def test_strictly_mutually_exclusive_options_optimization(n):
    adsg = BasicADSG()
    adsg.add_edges([
        (n[1], n[11]), (n[11], n[2]), (n[11], n[3]),
        (n[2], n[21]), (n[21], n[4]),
        (n[3], n[31]), (n[31], n[5]),
    ])
    adsg.add_selection_choice('C4', n[4], [n[41], n[42]])
    adsg.add_selection_choice('C5', n[5], [n[41], n[42]])
    adsg.add_incompatibility_constraint([n[41], n[42]])
    adsg = adsg.set_start_nodes({n[1]})

    processor = GraphProcessor(adsg)
    assert len(processor.des_vars) == 1

    i_used_values = set()
    for dv_values in itertools.product(*[range(des_var.n_opts) for des_var in processor.des_vars]):
        adsg2, i_used, _ = processor.get_graph(dv_values)
        assert adsg2.feasible
        assert adsg2.final
        i_used_values.add(tuple(i_used))
    assert len(i_used_values) == 2


def test_multi_incompatibility_derived_options(n):
    adsg = BasicADSG()
    adsg.add_edges([
        (n[1], n[11]), (n[11], n[2]), (n[11], n[3]),
        (n[21], n[4]), (n[22], n[4]), (n[31], n[4]), (n[32], n[4]),
    ])
    adsg.add_selection_choice('C2', n[2], [n[21], n[22]])
    adsg.add_selection_choice('C3', n[3], [n[31], n[32]])
    adsg.add_selection_choice('C4', n[4], [n[41], n[42]])

    adsg.add_incompatibility_constraint([n[21], n[41]])
    adsg.add_incompatibility_constraint([n[22], n[42]])
    adsg.add_incompatibility_constraint([n[31], n[42]])
    adsg.add_incompatibility_constraint([n[32], n[41]])

    adsg = adsg.set_start_nodes({n[1]})

    processor = GraphProcessor(adsg)
    assert len(processor.des_vars) == 1

    i_used_values = set()
    for dv_values in itertools.product(*[range(des_var.n_opts) for des_var in processor.des_vars]):
        adsg2, i_used, _ = processor.get_graph(dv_values)
        assert adsg2.feasible
        assert adsg2.final
        i_used_values.add(tuple(i_used))
    assert len(i_used_values) == 2


def test_choice_infeasible_incompatibility(n):
    adsg = BasicADSG()
    adsg.add_edges([
        (n[1], n[11]), (n[11], n[2]),
        (n[21], n[3]), (n[22], n[3]),
        (n[3], n[31]), (n[31], n[4]), (n[4], n[41]),
    ])
    c = adsg.add_selection_choice('C2', n[2], [n[21], n[22]])

    adsg.add_incompatibility_constraint([n[31], n[41]])
    adsg = adsg.set_start_nodes({n[1]})
    assert adsg.feasible

    for opt_node in adsg.get_option_nodes(c):
        adsg2 = adsg.get_for_apply_selection_choice(c, opt_node)
        assert adsg2.final
        assert not adsg2.feasible

    processor = GraphProcessor(adsg)
    with pytest.raises(RuntimeError):
        assert len(processor.des_vars) == 0
    processor._hierarchy_analyzer._assert_behavior()


def test_incompatibility_derived_nodes(n):
    adsg = BasicADSG()
    cn1 = ConnectorNode('CN1', deg_spec='?')
    cn2 = ConnectorNode('CN2', deg_spec='*')
    adsg.add_edges([
        (n[21], n[3]), (n[3], n[31]), (n[21], cn1),
        (n[4], n[41]), (n[41], cn2),
    ])
    c1 = adsg.add_selection_choice('C1', n[1], [n[11], n[12]])
    __ = adsg.add_selection_choice('C2', n[2], [n[21], n[22]])
    c3 = adsg.add_connection_choice('C3', [cn2], [cn1])

    adsg.add_incompatibility_constraint([n[11], n[31]])
    adsg = adsg.set_start_nodes({n[1], n[2], n[4]})
    assert adsg.feasible

    choice_nodes = adsg.get_ordered_next_choice_nodes()
    assert isinstance(choice_nodes[0], SelectionChoiceNode)
    assert choice_nodes[0] == c1
    opt_nodes = adsg.get_option_nodes(c1)
    assert len(opt_nodes) == 2
    assert opt_nodes[0] == n[11]
    adsg = adsg.get_for_apply_selection_choice(choice_nodes[0], opt_nodes[0])

    nodes = set(adsg.graph.nodes)
    assert n[21] not in nodes
    assert n[31] not in nodes
    assert len(adsg.get_nodes_by_type(ConnectorNode)) == 1

    choice_nodes = adsg.get_ordered_next_choice_nodes()
    assert len(choice_nodes) == 1
    assert isinstance(choice_nodes[0], ConnectionChoiceNode)
    assert choice_nodes[0] == c3

    conn_opts = list(c3.iter_conn_edges(adsg))
    assert len(conn_opts) == 1
    adsg = adsg.get_for_apply_connection_choice(c3, conn_opts[0])
    assert adsg.final
    assert adsg.feasible


def test_graph_processor_sub_choice_incompatibility(n):
    adsg = BasicADSG()
    adsg.add_edges([
        (n[11], n[2]), (n[12], n[2]), (n[12], n[3]),
        (n[2], n[21]), (n[21], n[4]),
    ])
    adsg.add_selection_choice('C1', n[1], [n[11], n[12]])
    adsg.add_selection_choice('C3', n[3], [n[31], n[32]])
    adsg.add_selection_choice('C4', n[4], [n[41], n[42]])
    adsg.add_incompatibility_constraint([n[12], n[42]])
    adsg = adsg.set_start_nodes({n[1]})

    processor = GraphProcessor(adsg)
    assert processor._hierarchy_analyzer.n_combinations == 4
    processor._hierarchy_analyzer._assert_behavior()


def test_circular_choices(n):
    adsg = BasicADSG()
    adsg.add_edges([
        (n[1], n[11]), (n[11], n[3]),
        (n[2], n[22]), (n[22], n[4]),
        (n[32], n[5]), (n[42], n[6]),
        (n[5], n[45]), (n[6], n[45]),
        (n[45], n[3]), (n[45], n[4]),
    ])
    adsg.add_selection_choice('C3', n[3], [n[31], n[32]])
    adsg.add_selection_choice('C4', n[4], [n[41], n[42]])
    adsg.add_incompatibility_constraint([n[31], n[42]])
    adsg.add_incompatibility_constraint([n[32], n[41]])
    adsg = adsg.set_start_nodes({n[1], n[2]})

    for i in range(2):
        choice_node = adsg.get_ordered_next_choice_nodes()[0]
        assert isinstance(choice_node, SelectionChoiceNode)
        opt_nodes = adsg.get_option_nodes(choice_node)
        adsg2 = adsg.get_for_apply_selection_choice(choice_node, opt_nodes[i])
        assert adsg2.final

        nodes = set(adsg2.graph.nodes)
        assert (n[31] in nodes) == (i == 0)
        assert (n[32] in nodes) == (i == 1)
        assert (n[41] in nodes) == (i == 0)
        assert (n[42] in nodes) == (i == 1)
        assert (n[45] in nodes) == (i == 1)

        assert adsg2.get_confirmed_graph().is_same(adsg2)
