import copy
import pytest
import itertools

from adsg_core.graph.adsg_basic import *
from adsg_core.graph.adsg_nodes import *
from adsg_core.optimization import *

from adsg_core.tests.test_optimization import assert_processor_get_all
from adsg_core.tests.test_optimization_hierarchy import get_hierarchy_analyzer


@pytest.fixture
def conn_sel_dsg(n):
    src_nodes = [
        ConnectorNode('A', deg_spec='+'),
        ConnectorNode('B', deg_list=[0, 1], repeated_allowed=True),
        ConnectorNode('C', deg_list=[0, 1], repeated_allowed=True),
    ]
    conn_group = ConnectorDegreeGroupingNode('G')
    tgt_nodes = [
        ConnectorNode('D', deg_min=0, repeated_allowed=True),
        ConnectorNode('E', deg_min=1, repeated_allowed=True),
        ConnectorNode('F', deg_list=[0, 2], repeated_allowed=True),
    ]
    tgt_conn_group = ConnectorDegreeGroupingNode('H')

    dsg = BasicDSG()
    dsg.add_edges([(n[0], node) for node in src_nodes])
    start_node = n[0]

    return src_nodes, conn_group, tgt_nodes, tgt_conn_group, dsg, start_node


def test_connection_choice_derives(conn_sel_dsg):
    src_nodes, conn_group, tgt_nodes, tgt_conn_group, dsg, start_node = conn_sel_dsg

    choice_node = dsg.add_connection_choice('A', src_nodes=[
        src_nodes[0],
        (conn_group, src_nodes[1:]),
    ], tgt_nodes=[
        tgt_nodes[0],
        (tgt_conn_group, tgt_nodes[1:]),
    ], derive_tgt_nodes=True)
    assert choice_node in dsg.graph.nodes
    assert conn_group in dsg.graph.nodes
    assert dsg.feasible
    assert not dsg.final

    dsg.export_dot('graph.dot')

    assert len(dsg.choice_nodes) == 1

    assert len(list(choice_node.iter_conn_edges(dsg))) == 15
    assert len(list(dsg.iter_possible_connection_edges(choice_node))) == 15

    dsg = dsg.set_start_nodes({start_node})
    assert dsg.feasible
    assert not dsg.final
    assert dsg.get_ordered_next_choice_nodes() == [choice_node]

    n_seen = 0
    for edges in dsg.iter_possible_connection_edges(choice_node):
        dsg2 = dsg.get_for_apply_connection_choice(choice_node, edges)
        assert dsg2.feasible
        assert dsg2.final

        assert tgt_nodes[0] in dsg2.graph.nodes

        n_seen += 1
    assert n_seen == 15


def test_connection_choice_derives_removes(conn_sel_dsg):
    src_nodes, conn_group, tgt_nodes, tgt_conn_group, dsg, start_node = conn_sel_dsg

    for node in src_nodes:
        node.remove_if_unconnected = True
    for node in tgt_nodes:
        node.remove_if_unconnected = True

    choice_node = dsg.add_connection_choice('A', src_nodes=[
        src_nodes[0],
        (conn_group, src_nodes[1:]),
    ], tgt_nodes=[
        tgt_nodes[0],
        (tgt_conn_group, tgt_nodes[1:]),
    ], derive_tgt_nodes=True)
    assert choice_node in dsg.graph.nodes
    assert conn_group in dsg.graph.nodes
    assert dsg.feasible
    assert not dsg.final

    dsg.export_dot('graph.dot')

    assert len(dsg.choice_nodes) == 5

    assert len(list(choice_node.iter_conn_edges(dsg))) == 11
    assert len(list(dsg.iter_possible_connection_edges(choice_node))) == 11

    dsg = dsg.set_start_nodes({start_node})
    assert dsg.feasible
    assert not dsg.final
    assert dsg.get_ordered_next_choice_nodes() != [choice_node]

    n_seen = 0
    for sel_opt_i in itertools.product(*[(0, 1) for _ in range(4)]):
        dsg2 = dsg
        i_sel = 0
        while True:
            next_choice_nodes = dsg2.get_ordered_next_choice_nodes()

            if isinstance(next_choice_nodes[0], SelectionChoiceNode):
                opt_nodes = dsg2.get_option_nodes(next_choice_nodes[0])
                dsg2 = dsg2.get_for_apply_selection_choice(next_choice_nodes[0], opt_nodes[sel_opt_i[i_sel]])
                i_sel += 1
                continue

            assert next_choice_nodes[0] == choice_node

            for edges in dsg2.iter_possible_connection_edges(choice_node):
                dsg3 = dsg2.get_for_apply_connection_choice(choice_node, edges)
                assert dsg3.feasible
                assert dsg3.final

                n_seen += 1

            break
    assert n_seen == 50


def test_connection_choice_derives_opt(conn_sel_dsg):
    src_nodes, conn_group, tgt_nodes, tgt_conn_group, dsg, start_node = conn_sel_dsg

    choice_node = dsg.add_connection_choice('A', src_nodes=[
        src_nodes[0],
        (conn_group, src_nodes[1:]),
    ], tgt_nodes=[
        tgt_nodes[0],
        (tgt_conn_group, tgt_nodes[1:]),
    ], derive_tgt_nodes=True)
    assert choice_node in dsg.graph.nodes
    assert conn_group in dsg.graph.nodes
    assert dsg.feasible
    assert not dsg.final

    dsg = dsg.set_start_nodes({start_node})

    an = get_hierarchy_analyzer(dsg)
    assert an.n_combinations == 1
    an._assert_behavior()

    processor = GraphProcessor(dsg)
    assert len(processor.des_vars) >= 1
    processor.print_stats()
    assert processor.get_n_valid_designs() == 15

    assert_processor_get_all(processor)


def test_connection_choice_derives_removes_opt(conn_sel_dsg):
    src_nodes, conn_group, tgt_nodes, tgt_conn_group, dsg, start_node = conn_sel_dsg

    for node in src_nodes:
        node.remove_if_unconnected = True
    for node in tgt_nodes:
        node.remove_if_unconnected = True

    choice_node = dsg.add_connection_choice('A', src_nodes=[
        src_nodes[0],
        (conn_group, src_nodes[1:]),
    ], tgt_nodes=[
        tgt_nodes[0],
        (tgt_conn_group, tgt_nodes[1:]),
    ], derive_tgt_nodes=True)
    assert choice_node in dsg.graph.nodes
    assert conn_group in dsg.graph.nodes
    assert dsg.feasible
    assert not dsg.final

    dsg = dsg.set_start_nodes({start_node})

    an = get_hierarchy_analyzer(dsg)
    assert an.n_combinations == 16
    an._assert_behavior()

    processor = GraphProcessor(dsg)
    assert len(processor.des_vars) >= 5
    processor.print_stats()
    assert processor.get_n_valid_designs() == 50

    assert_processor_get_all(processor)


def assert_n_valid(src_nodes_, tgt_nodes_, n_valid, n_sel_comb=None):
    if isinstance(n_valid, tuple):
        n_valid, n_valid_rem = n_valid
    else:
        n_valid_rem = n_valid

    for remove_unconnected in [False, True]:
        src_nodes = copy.deepcopy(src_nodes_)
        tgt_nodes = copy.deepcopy(tgt_nodes_)

        src_base_nodes = []
        for src_node in src_nodes:

            if isinstance(src_node, tuple):
                src_base_nodes += src_node[1]
                if remove_unconnected:
                    for underlying_src_node in src_node[1]:
                        underlying_src_node.remove_if_unconnected = True
            else:
                src_base_nodes.append(src_node)
                if remove_unconnected:
                    src_node.remove_if_unconnected = True

        if remove_unconnected:
            for tgt_node in tgt_nodes:
                if isinstance(tgt_node, tuple):
                    for underlying_tgt_node in tgt_node[1]:
                        underlying_tgt_node.remove_if_unconnected = True
                else:
                    tgt_node.remove_if_unconnected = True

        dsg = BasicDSG()

        start_node = NamedNode('start')
        dsg.add_edges([(start_node, node) for node in src_base_nodes])

        choice_node = dsg.add_connection_choice(
            'Conn', src_nodes=src_nodes, tgt_nodes=tgt_nodes, derive_tgt_nodes=True)

        dsg = dsg.set_start_nodes({start_node})
        assert dsg.feasible
        assert not dsg.final

        if remove_unconnected:
            dsg.export_dot('graph_derive_remove.dot')
            assert len(dsg.choice_nodes) >= 1
            assert dsg.get_ordered_next_choice_nodes() != [choice_node]
        else:
            dsg.export_dot('graph_derive.dot')
            assert len(dsg.choice_nodes) == 1
            assert dsg.get_ordered_next_choice_nodes() == [choice_node]

        an = get_hierarchy_analyzer(dsg)
        if remove_unconnected:
            if n_sel_comb is not None:
                assert an.n_combinations == n_sel_comb
        else:
            assert an.n_combinations == 1
        an._assert_behavior()

        processor = GraphProcessor(dsg)
        processor.print_stats()
        assert processor.get_n_valid_designs() == (n_valid_rem if remove_unconnected else n_valid)

        assert_processor_get_all(processor)


def test_conn_list():
    assert_n_valid([
        ConnectorNode('s1', deg_list=[0, 1]),
    ], [
        ConnectorNode('t1', deg_list=[0, 1]),
    ], n_valid=2, n_sel_comb=4)


def test_conn_list_src_non_opt():
    assert_n_valid([
        ConnectorNode('s1', deg_list=[0, 1]),
        ConnectorNode('s2', deg_list=[1]),
    ], [
        ConnectorNode('t1', deg_list=[0, 1, 2]),
    ], n_valid=2, n_sel_comb=4)


def test_conn_req_tgt():
    assert_n_valid([
        ConnectorNode('s1', deg_min=0, repeated_allowed=True),
    ], [
        ConnectorNode('t1', deg_list=[1, 2], repeated_allowed=True),
    ], n_valid=2, n_sel_comb=2)


def test_conn_opt_group():
    assert_n_valid([
        (ConnectorDegreeGroupingNode(), [
            ConnectorNode('s1', deg_list=[0, 1], repeated_allowed=True),
            ConnectorNode('s2', deg_list=[0, 1], repeated_allowed=True),
        ]),
    ], [
        ConnectorNode('t1', deg_min=0, repeated_allowed=True),
    ], n_valid=(3, 5), n_sel_comb=8)


def test_conn_sel_dsg():
    src_nodes = [
        ConnectorNode('A', deg_spec='+'),
        ConnectorNode('B', deg_list=[0, 1], repeated_allowed=True),
        ConnectorNode('C', deg_list=[0, 1], repeated_allowed=True),
    ]
    conn_group = ConnectorDegreeGroupingNode('G')
    tgt_nodes = [
        ConnectorNode('D', deg_min=0, repeated_allowed=True),
        ConnectorNode('E', deg_min=1, repeated_allowed=True),
        ConnectorNode('F', deg_list=[0, 2], repeated_allowed=True),
    ]
    tgt_conn_group = ConnectorDegreeGroupingNode('H')

    assert_n_valid([
        src_nodes[0],
        (conn_group, src_nodes[1:]),
    ], [
        tgt_nodes[0],
        (tgt_conn_group, tgt_nodes[1:]),
    ], n_valid=(15, 50), n_sel_comb=16)
