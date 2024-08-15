import math
import pytest
import webbrowser
import tempfile
import numpy as np
from adsg_core.graph.adsg import *
from adsg_core.graph.traversal import *
from adsg_core.graph.adsg_basic import *
from adsg_core.graph.adsg_nodes import *
from adsg_core.graph.graph_edges import *
from adsg_core.graph.influence_matrix import *


def test_graph():
    DSG()

    adsg = BasicDSG()
    assert hash(adsg)
    assert adsg == BasicDSG()


def test_add_edges():
    a, b = NamedNode('A'), NamedNode('B')

    adsg = BasicDSG()
    adsg.add_edge(a, b)
    assert a in adsg.graph.nodes
    assert b in adsg.graph.nodes
    assert (a, b) in adsg.graph.edges

    assert list(iter_in_edges(adsg.graph, a)) == []
    assert list(iter_out_edges(adsg.graph, a))[0][:2] == (a, b)
    assert list(iter_out_edges(adsg.graph, a, edge_type=EdgeType.CONNECTS)) == []

    assert list(adsg.prev(a)) == []
    assert list(adsg.next(a)) == [b]
    assert list(adsg.prev(b)) == [a]
    assert list(adsg.next(b)) == []
    assert list(adsg.derived_nodes(a)) == [b]
    assert adsg.derives(a, b)
    assert not adsg.derives(b, a)

    assert len(list(iter_in_edges(adsg.graph, b))) == 1

    assert get_in_degree(adsg.graph, a) == 0
    assert get_in_degree(adsg.graph, b) == 1
    assert get_in_degree(adsg.graph, b, edge_type=EdgeType.DERIVES) == 1
    assert get_in_degree(adsg.graph, b, edge_type=EdgeType.CONNECTS) == 0

    assert get_out_degree(adsg.graph, b) == 0
    assert get_out_degree(adsg.graph, a) == 1
    assert get_out_degree(adsg.graph, a, edge_type=EdgeType.DERIVES) == 1
    assert get_out_degree(adsg.graph, a, edge_type=EdgeType.CONNECTS) == 0

    c, d = NamedNode('C'), NamedNode('D')
    adsg.add_edges([(b, c), (c, d)], edge_type=EdgeType.CONNECTS)
    assert get_out_degree(adsg.graph, b, edge_type=EdgeType.DERIVES) == 0
    assert get_out_degree(adsg.graph, b, edge_type=EdgeType.CONNECTS) == 1
    assert (b, c) in adsg.graph.edges

    assert not adsg.derives(b, d)
    assert adsg.derives(b, d, connects=True)
    assert adsg.derives(c, d, connects=True)
    assert not adsg.derives(d, b, connects=True)

    adsg.export_gml()
    adsg.export_drawio()
    try:
        adsg.export_dot()
    except ModuleNotFoundError:
        pass

    with tempfile.TemporaryDirectory() as tmp_folder:
        adsg.export_gml(f'{tmp_folder}/graph.gml')
        adsg.export_drawio(f'{tmp_folder}/graph.drawio')

        try:
            adsg.export_dot(f'{tmp_folder}/graph.dot')
        except ModuleNotFoundError:
            pass
    # adsg.export_drawio('graph.drawio')

    webbrowser.open = lambda _: None
    adsg.render()
    adsg.render_legend()
    adsg.render_legend(elements=['EDGES'])


def test_copy_node():
    node = NamedNode('A')
    node_copy = node.copy_node()
    assert node != node_copy


def test_get_for_adjusted(n):
    for inplace in [False, True]:
        adsg = BasicDSG()
        adsg.add_edges([(n[0], n[1]), (n[1], n[2])])

        assert (n[1], n[2]) in adsg.graph.edges
        adsg = adsg.get_for_adjusted(removed_edges={(n[1], n[2])}, inplace=inplace)
        assert (n[1], n[2]) not in adsg.graph.edges
        assert n[2] in adsg.graph.nodes

        adsg.add_edge(n[1], n[2])
        adsg = adsg.get_for_adjusted(removed_edges={(n[1], n[2])}, removed_nodes={n[2]}, inplace=inplace)
        assert (n[1], n[2]) not in adsg.graph.edges
        assert n[2] not in adsg.graph.nodes

        adsg = adsg.get_for_adjusted(removed_nodes={n[1]}, added_edges={(n[1], n[2])}, inplace=inplace)
        assert (n[1], n[2]) in adsg.graph.edges
        assert n[1] in adsg.graph.nodes

        adsg = BasicDSG()
        adsg.add_edges([(n[0], n[1]), (n[1], n[2])])
        adsg = adsg.get_for_kept_edges({(n[1], n[2])})
        assert n[0] not in adsg.graph.nodes


def test_set_start_nodes():
    a, b, c, d, e, f = NamedNode('A'), NamedNode('B'), NamedNode('C'), NamedNode('D'), NamedNode('E'), NamedNode('F')

    adsg = BasicDSG()
    adsg.add_edges([
        (a, b), (b, c), (c, e), (c, f),
        (d, c),
    ])
    assert adsg.derivation_start_nodes is None
    assert adsg.feasible
    assert adsg.final

    assert adsg._get_floating_nodes() == {a, d}
    assert adsg._get_alternative_start_nodes() == {a, d}
    adsg2 = adsg.set_start_nodes()
    assert adsg2.graph == adsg.graph
    assert adsg2.derivation_start_nodes == {a, d}
    assert adsg2.feasible
    assert adsg2.final

    adsg3 = adsg.set_start_nodes({a, d})
    assert adsg3.graph == adsg.graph
    assert adsg3.derivation_start_nodes == {a, d}
    assert adsg3.feasible
    assert adsg3.final

    adsg4 = adsg.set_start_nodes({a})
    assert adsg4.derivation_start_nodes == {a}
    assert a in adsg4.graph.nodes
    assert c in adsg4.graph.nodes
    assert d not in adsg4.graph.nodes

    adsg5 = adsg.set_start_nodes({d})
    assert adsg5.derivation_start_nodes == {d}
    assert set(adsg5.graph.nodes) == {c, d, e, f}
    assert len(adsg5.graph.edges) == 3
    assert (d, c) in adsg5.graph.edges

    adsg6 = adsg.set_start_nodes({b})
    assert adsg6.derivation_start_nodes == {b}
    assert set(adsg6.graph.nodes) == {b, c, e, f}
    assert len(adsg6.graph.edges) == 3
    assert (b, c) in adsg6.graph.edges
    assert adsg6.feasible
    assert adsg6.final

    missing = NamedNode('missing')
    with pytest.raises(ValueError):
        adsg.set_start_nodes({a, missing})


def test_individual_nodes(n):
    adsg = BasicDSG()
    adsg.add_node(n[0])
    adsg.add_edge(n[1], n[2])

    adsg1 = adsg.set_start_nodes({n[1]})
    assert len(adsg1.graph.nodes) == 2

    adsg2 = adsg.set_start_nodes({n[0], n[1]})
    assert len(adsg2.graph.nodes) == 3


def test_individual_nodes_choice(n):
    adsg = BasicDSG()
    adsg.add_node(n[0])
    c = adsg.add_selection_choice('C', n[1], n[2:4])
    adsg = adsg.set_start_nodes({n[0], n[1]})

    adsg1 = adsg.get_for_apply_selection_choice(c, n[2])
    assert n[0] in adsg1.graph.nodes


def test_set_start_nodes_incompatibility(n):
    adsg = BasicDSG()
    adsg.add_edges([
        (n[0], n[1]),
        (n[2], n[3]), (n[2], n[4]),
    ])
    adsg.add_incompatibility_constraint([n[2], n[4]])

    adsg = adsg.set_start_nodes({n[0]})
    assert len(adsg.graph.nodes) == 2


def _strip(edges, keep_key=False):  # Strip edges of keys and data
    if keep_key:
        return {(edge[0], edge[1], 0 if len(edge) < 3 or isinstance(edge[2], dict) else edge[2])
                if type(edge) == tuple else edge for edge in edges}
    return {(edge[0], edge[1]) if type(edge) == tuple else edge for edge in edges}


def _strips(args, keep_key=False):
    return tuple(_strip(arg, keep_key=keep_key) if type(arg) == set else arg for arg in args)


def test_get_deriving_in_edges(n):
    adsg = BasicDSG()
    adsg.add_edges([
        (n[0], n[1]), (n[1], n[2]),
        (n[3], n[2]),
    ])
    adsg.add_edge(n[3], n[4], edge_type=EdgeType.CONNECTS)

    assert _strip(get_deriving_in_edges(adsg.graph, n[0])) == set()
    assert _strip(get_deriving_in_edges(adsg.graph, n[1])) == {(n[0], n[1])}
    assert _strip(get_deriving_in_edges(adsg.graph, n[1], edge_type=EdgeType.DERIVES)) == {(n[0], n[1])}
    assert _strip(get_deriving_in_edges(adsg.graph, n[1], edge_type=EdgeType.CONNECTS)) == {(n[0], n[1])}
    assert _strip(get_deriving_in_edges(adsg.graph, n[4], edge_type=EdgeType.DERIVES)) == set()
    assert _strip(get_deriving_in_edges(adsg.graph, n[4], edge_type=EdgeType.CONNECTS)) == {(n[3], n[4])}

    assert _strip(get_deriving_in_edges(adsg.graph, n[2])) == {(n[1], n[2]), (n[3], n[2])}
    assert _strip(get_deriving_in_edges(adsg.graph, n[2], removed_edges={(n[3], n[2])})) == {(n[1], n[2])}
    assert _strip(get_deriving_in_edges(adsg.graph, n[2], removed_nodes={n[3]})) == {(n[1], n[2])}


def test_get_derived_edges(n):
    adsg = BasicDSG()
    c1, c2 = ConnectorNode('A', deg_spec='*'), ConnectorNode('B', deg_spec='*')
    adsg.add_edges([
        (n[0], n[1]), (n[1], n[2]), (n[2], n[3]),
        (n[4], n[3]), (n[3], n[5]), (n[5], c1),
        (n[2], n[6]), (n[6], c2),
    ])
    choice_node = adsg.add_connection_choice('A', src_nodes=[c1], tgt_nodes=[c2])
    adsg = adsg.set_start_nodes({n[0], n[4]})

    assert _strips(get_derived_edges_for_node(adsg.graph, c1, adsg.derivation_start_nodes)) == ({
        (c1, choice_node), (choice_node, c2),
    }, {choice_node})
    assert _strips(get_derived_edges_for_node(adsg.graph, c2, adsg.derivation_start_nodes)) == (set(), set())

    assert _strips(get_derived_edges_for_node(adsg.graph, n[2], adsg.derivation_start_nodes)) == ({
        (n[2], n[3]),
        (n[2], n[6]), (n[6], c2),
    }, {n[6], c2})

    assert _strips(get_derived_edges_for_node(adsg.graph, n[0], adsg.derivation_start_nodes)) == ({
        (n[0], n[1]), (n[1], n[2]), (n[2], n[3]), (n[2], n[6]), (n[6], c2),
    }, {n[1], n[2], n[6], c2})
    assert _strips(get_derived_edges_for_node(
        adsg.graph, n[0], adsg.derivation_start_nodes, removed_edges={(n[4], n[3])})) == (
        {(n[0], n[1]), (n[1], n[2]), (n[2], n[3]), (n[2], n[6]), (n[6], c2),
         (n[3], n[5]), (n[5], c1), (c1, choice_node), (choice_node, c2)},
        {n[1], n[3], n[2], n[6], c2, choice_node, c1, n[5]})


def test_get_derived_edges_loop(n):
    adsg = BasicDSG()
    adsg.add_edges([
        (n[0], n[1]),
        (n[1], n[2]), (n[2], n[3]),
        (n[3], n[4]), (n[4], n[1]),
    ])
    adsg = adsg.set_start_nodes({n[0]})
    derived_edges, derived_nodes = get_derived_edges_for_node(adsg.graph, n[0], adsg.derivation_start_nodes)
    assert n[2] in derived_nodes
    assert n[4] in derived_nodes

    adsg2 = adsg.get_for_adjusted(removed_edges=derived_edges, removed_nodes=derived_nodes)
    assert len(adsg2.graph.nodes) == 1


def test_influence_matrix(n):
    adsg = BasicDSG()
    adsg.add_selection_choice('A', n[0], [n[1], n[2]])
    adsg.add_selection_choice('B', n[3], [n[4], n[5]])
    adsg.add_selection_choice('C', n[6], [n[7], n[8]])
    adsg.add_selection_choice('D', n[9], [n[10], n[13]])
    adsg.add_edges([
        (n[1], n[3]), (n[2], n[6]),
        (n[10], n[11]), (n[11], n[12]),
        (n[13], n[6]),
    ])
    adsg.set_start_nodes({n[0], n[9]})
    assert len(adsg.choice_nodes) == 4

    im = InfluenceMatrix(adsg)
    assert len(im.permanent_nodes) == 2
    assert len(im.selection_choice_nodes) == 4
    assert len(im.selection_choice_option_nodes) == 4
    assert im._sel_choice_influence
    assert len(im.matrix_diagonal_nodes) == len(adsg.graph.nodes)
    for node in im.matrix_diagonal_nodes:
        assert node in im.matrix_diagonal_nodes_idx

    matrix = im.influence_matrix
    assert matrix.shape == (len(adsg.graph.nodes), len(adsg.graph.nodes))
    assert sum(matrix.diagonal() == Diag.CONFIRMED.value) == len(im.permanent_nodes)+2

    # adsg.export_drawio('graph.drawio')
    # adsg.render(title='ASDG Test')


def test_get_confirmed_edges_for_node(n):
    adsg = BasicDSG()
    adsg.add_edges([(n[0], n[1]), (n[1], n[2])])
    assert _strip(get_confirmed_edges_for_node(adsg.graph, n[0])) == {(n[0], n[1]), (n[1], n[2])}

    choice_node = adsg.add_selection_choice('A', n[1], [n[3], n[4]])
    adsg.add_edges([(n[3], n[5])])
    adsg = adsg.set_start_nodes()

    assert _strip(get_confirmed_edges_for_node(adsg.graph, n[0])) == {(n[0], n[1]), (n[1], n[2])}

    with pytest.raises(ValueError):
        adsg.get_confirmed_edges_selection_choice(choice_node, n[2])
    assert _strip(adsg.get_confirmed_edges_selection_choice(choice_node, n[3])[0]) == {(n[1], n[3]), (n[3], n[5])}

    adsg2 = adsg.get_for_apply_selection_choice(choice_node, n[3])
    adsg3 = adsg.get_for_kept_edges(set(adsg.get_confirmed_graph().graph.edges) |
                                    adsg.get_confirmed_edges_selection_choice(choice_node, n[3])[0])
    assert adsg2 == adsg3


def test_selection_choice(n):
    adsg = BasicDSG()
    choice_node = adsg.add_selection_choice('A', n[0], n[1:4])
    adsg.add_edge(n[5], n[6])
    assert n[0] in adsg.graph.nodes
    assert n[5] in adsg.graph.nodes
    assert choice_node in adsg.graph.nodes

    assert adsg.feasible
    assert not adsg.final

    assert adsg.choice_nodes == [choice_node]
    option_nodes = adsg.get_option_nodes(choice_node)
    assert option_nodes == n[1:4]

    with pytest.raises(RuntimeError):
        adsg.get_for_apply_selection_choice(choice_node, n[1])
    with pytest.raises(RuntimeError):
        adsg.initialize_choices()

    adsg = adsg.set_start_nodes({n[0]})
    assert adsg.derivation_start_nodes == {n[0]}
    assert adsg._influence_matrix is not None
    assert n[5] not in adsg.graph.nodes

    with pytest.raises(ValueError):
        adsg.get_for_apply_selection_choice(choice_node, n[4])

    adsg2 = adsg.get_for_apply_selection_choice(choice_node, n[1])
    assert adsg2.feasible
    assert adsg2.final
    assert set(adsg2.graph.nodes) == {n[0], n[1]}
    assert (n[0], n[1]) in adsg2.graph.edges


def test_nested_selection(n):
    adsg = BasicDSG()
    choice_node = adsg.add_selection_choice('A', n[0], n[1:3])
    choice_node2 = adsg.add_selection_choice('B', n[2], n[3:5])
    assert choice_node2 in adsg.graph.nodes
    assert n[4] in adsg.graph.nodes

    adsg = adsg.set_start_nodes({n[0]})
    assert adsg.feasible
    assert not adsg.final

    assert adsg.get_ordered_next_choice_nodes() == [choice_node]
    assert adsg.get_option_nodes(choice_node) == n[1:3]
    adsg2 = adsg.get_for_apply_selection_choice(choice_node, n[1])
    assert adsg2.feasible
    assert adsg2.final
    assert n[4] not in adsg2.graph.nodes

    adsg3 = adsg.get_for_apply_selection_choice(choice_node, n[2])
    assert adsg3.feasible
    assert not adsg3.final
    assert n[4] in adsg3.graph.nodes

    assert adsg3.get_ordered_next_choice_nodes() == [choice_node2]
    assert adsg3.get_option_nodes(choice_node2) == n[3:5]
    adsg4 = adsg3.get_for_apply_selection_choice(choice_node2, n[4])
    assert adsg4.feasible
    assert adsg4.final
    assert n[4] in adsg4.graph.nodes


def test_looped_selection(n):
    adsg = BasicDSG()
    cn1 = adsg.add_selection_choice('A', n[0], [n[1], n[2]])
    cn2 = adsg.add_selection_choice('B', n[1], [n[0], n[3]])
    assert adsg.feasible
    assert not adsg.final

    assert adsg._get_alternative_start_nodes() == set()
    with pytest.raises(ValueError):
        adsg.set_start_nodes()
    assert adsg.get_ordered_next_choice_nodes() == []

    adsg = adsg.set_start_nodes({n[0]})
    assert adsg.get_ordered_next_choice_nodes() == [cn1]

    adsg2 = adsg.get_for_apply_selection_choice(cn1, n[1])
    assert adsg2.feasible
    assert not adsg2.final

    assert adsg2.get_ordered_next_choice_nodes() == [cn2]
    adsg3 = adsg2.get_for_apply_selection_choice(cn2, n[0])
    assert adsg3.final
    assert adsg3.feasible
    adsg3 = adsg2.get_for_apply_selection_choice(cn2, n[3])
    assert adsg3.final
    assert adsg3.feasible

    adsg4 = adsg.get_for_apply_selection_choice(cn1, n[2])
    assert adsg4.final
    assert adsg4.feasible

    adsg5 = adsg.set_start_nodes({n[0], n[1]})
    adsg6 = adsg5.get_for_apply_selection_choice(cn1, n[2])
    assert not adsg6.final


def test_options_choice_circular(n):
    adsg = BasicDSG()
    choice_node = adsg.add_selection_choice('A', n[0], [n[1], n[2]])
    adsg.add_edges([(n[1], n[0]), (n[2], n[0])])

    adsg = adsg.set_start_nodes({n[0]})
    assert adsg.feasible
    assert not adsg.final

    adsg2 = adsg.get_for_apply_selection_choice(choice_node, n[1])
    assert adsg2.final
    assert (n[0], n[1]) in adsg2.graph.edges
    assert (n[1], n[0]) in adsg2.graph.edges


def test_cross_choices(n):
    adsg = BasicDSG()
    c1 = adsg.add_selection_choice('A', n[0], [n[1], n[2]])
    c2 = adsg.add_selection_choice('B', n[3], [n[4], n[5]])
    adsg.add_edges([
        (n[6], n[0]), (n[7], n[3]),
        (n[2], n[8]), (n[5], n[8]),
        (n[8], n[0]), (n[8], n[3]),
    ])
    adsg = adsg.set_start_nodes({n[6], n[7]})
    assert len(adsg.choice_nodes) == 2

    for i in range(2):
        choice_node = adsg.get_ordered_next_choice_nodes()[0]
        assert choice_node == c1
        opt_nodes = adsg.get_option_nodes(choice_node)
        assert opt_nodes == [n[1], n[2]]
        adsg2 = adsg.get_for_apply_selection_choice(choice_node, opt_nodes[i])

        for j in range(2):
            choice_node = adsg2.get_ordered_next_choice_nodes()[0]
            assert choice_node == c2
            opt_nodes = adsg2.get_option_nodes(choice_node)
            assert opt_nodes == [n[4], n[5]]

            adsg3 = adsg2.get_for_apply_selection_choice(choice_node, opt_nodes[j])
            assert adsg3.final

            assert (n[1] in adsg3.graph.nodes) == (i == 0)
            assert (n[2] in adsg3.graph.nodes) == (i == 1)
            assert (n[4] in adsg3.graph.nodes) == (j == 0)
            assert (n[5] in adsg3.graph.nodes) == (j == 1)
            assert (n[8] in adsg3.graph.nodes) == (i == 1 or j == 1)


def test_graph_hash_eq(n):
    adsg = BasicDSG()
    adsg.add_edges([(n[0], n[1])])
    choice_node = adsg.add_selection_choice('A', n[1], [n[2], n[3]])
    adsg = adsg.set_start_nodes({n[0]})

    adsg2 = adsg.get_for_apply_selection_choice(choice_node, n[2])
    adsg3 = adsg.get_for_apply_selection_choice(choice_node, n[3])
    assert hash(adsg2) != hash(adsg3)
    assert adsg2 != adsg3

    assert adsg.get_for_apply_selection_choice(choice_node, n[2]) == adsg2


def test_get_ordered_next_choice_nodes(n):
    for _ in range(10):
        adsg = BasicDSG()
        c2 = adsg.add_selection_choice('B', n[0], [n[4], n[3]])
        c1 = adsg.add_selection_choice('A', n[0], [n[1], n[2]])
        adsg = adsg.set_start_nodes({n[0]})

        assert adsg.get_ordered_next_choice_nodes() == [c1, c2]
        assert adsg.get_option_nodes(c1) == [n[1], n[2]]
        assert adsg.get_option_nodes(c2) == [n[4], n[3]]


def test_get_confirmed_graph(n):
    adsg = BasicDSG()
    c1 = adsg.add_selection_choice('A', n[0], [n[1], n[2]])
    adsg.add_edge(n[1], n[3])
    c2 = adsg.add_selection_choice('B', n[3], [n[4], n[5]])

    assert set(adsg.get_confirmed_graph().graph.nodes) == {n[0]}

    adsg = adsg.set_start_nodes()
    assert adsg.derivation_start_nodes == {n[0]}
    assert set(adsg.get_confirmed_graph().graph.nodes) == {n[0]}

    assert adsg.get_ordered_next_choice_nodes() == [c1]
    adsg2 = adsg.get_for_apply_selection_choice(c1, n[1])
    assert set(adsg2.get_confirmed_graph().graph.nodes) == {n[0], n[1], n[3]}

    adsg3 = adsg2.get_for_apply_selection_choice(c2, n[4])
    assert set(adsg3.get_confirmed_graph().graph.nodes) == {n[0], n[1], n[3], n[4]}

    adsg2 = adsg.get_for_apply_selection_choice(c1, n[2])
    assert set(adsg2.get_confirmed_graph().graph.nodes) == {n[0], n[2]}


def test_has_conditional_existence(n):
    adsg = BasicDSG()
    adsg.add_edges([
        (n[0], n[1]), (n[1], n[2]),
        (n[3], n[5]), (n[4], n[5]),
        (n[8], n[9]),
    ])
    c1 = adsg.add_selection_choice('A', n[2], [n[3], n[4]])
    c2 = adsg.add_selection_choice('B', n[5], [n[6], n[7]])
    adsg = adsg.set_start_nodes({n[0], n[8]})

    # 0 = unconditional existence; 1 = conditional existence; 2 = unconditional non-existence
    assert adsg.has_conditional_existence(n[0]) == 0
    assert adsg.has_conditional_existence(n[1]) == 0
    assert adsg.has_conditional_existence(n[2]) == 0
    assert adsg.has_conditional_existence(c1) == 0

    assert adsg.has_conditional_existence(n[0], [n[1]]) == 0
    assert adsg.has_conditional_existence(n[0], [n[2]]) == 0
    assert adsg.has_conditional_existence(n[0], [c1]) == 0

    assert adsg.has_conditional_existence(n[3]) == 1
    assert adsg.has_conditional_existence(n[4]) == 1
    assert adsg.has_conditional_existence(n[5]) == 0

    assert adsg.has_conditional_existence(n[3], [n[4]]) == 2
    assert adsg.has_conditional_existence(n[3], [n[5]]) == 0
    assert adsg.has_conditional_existence(n[4], [n[4]]) == 0
    assert adsg.has_conditional_existence(n[4], [n[3]]) == 2
    assert adsg.has_conditional_existence(n[5]) == 0

    assert adsg.has_conditional_existence(c2) == 0
    assert adsg.has_conditional_existence(n[6]) == 1
    assert adsg.has_conditional_existence(n[7]) == 1

    assert adsg.has_conditional_existence(n[9]) == 0
    assert adsg.has_conditional_existence(n[9], [n[1]]) == 2


def test_infeasible_sel_no_opt(n):
    adsg = BasicDSG()
    adsg.add_selection_choice('A', n[0], [])
    assert not adsg.set_start_nodes().feasible


def test_single_option_choice(n):
    adsg = BasicDSG()
    sel_choice2 = adsg.add_selection_choice('B', n[0], [n[1]])
    sel_choice3 = adsg.add_selection_choice('C', n[0], [n[2], n[3]])
    sel_choice4 = adsg.add_selection_choice('D', n[2], [n[4], n[5]])
    sel_choice5 = adsg.add_selection_choice('E', n[5], [n[6]])
    assert adsg.feasible
    assert not adsg.final

    adsg = adsg.set_start_nodes()
    assert adsg.derivation_start_nodes == {n[0]}

    assert sel_choice2 not in adsg.graph.nodes
    assert sel_choice5 in adsg.graph.nodes

    assert adsg.get_ordered_next_choice_nodes() == [sel_choice3]
    adsg2 = adsg.get_for_apply_selection_choice(sel_choice3, n[2])
    assert adsg2.get_ordered_next_choice_nodes() == [sel_choice4]
    adsg3 = adsg2.get_for_apply_selection_choice(sel_choice4, n[5])

    assert sel_choice5 not in adsg3.graph.nodes
    assert n[6] in adsg3.graph.nodes
    assert adsg3.feasible
    assert adsg3.final


def test_connection_node_degree():
    assert ConnectorNode._parse_deg_spec([1, 2]) == ([1, 2], None, None)
    assert ConnectorNode._parse_deg_spec('2..5') == (None, 2, 5)

    assert ConnectorNode._parse_deg_spec('?') == (None, 0, 1)
    assert ConnectorNode._parse_deg_spec('+') == (None, 1, math.inf)
    assert ConnectorNode._parse_deg_spec('*') == (None, 0, math.inf)

    assert ConnectorNode._parse_deg_spec(7) == ([7], None, None)
    assert ConnectorNode._parse_deg_spec('8') == ([8], None, None)

    assert ConnectorNode._parse_deg_spec('opt') == (None, 0, 1)
    assert ConnectorNode._parse_deg_spec('req') == ([1], None, None)


def test_connection_grouping_node():
    node1 = ConnectorNode(deg_list=[1])
    node2 = ConnectorNode(deg_min=0, deg_max=1)
    node3 = ConnectorNode(deg_list=[0, 1])
    node4 = ConnectorNode(deg_min=1, deg_max=math.inf)

    assert ConnectorDegreeGroupingNode.get_combined_deg([node1, node1]) == ([2], None, None)
    assert ConnectorDegreeGroupingNode.get_combined_deg([node1, node2]) == ([1, 2], None, None)
    assert ConnectorDegreeGroupingNode.get_combined_deg([node1, node3]) == ([1, 2], None, None)
    assert ConnectorDegreeGroupingNode.get_combined_deg([node2, node3]) == ([0, 1, 2], None, None)

    assert ConnectorDegreeGroupingNode.get_combined_deg([node1, node4]) == (None, 2, math.inf)
    assert ConnectorDegreeGroupingNode.get_combined_deg([node2, node4]) == (None, 1, math.inf)


def test_connection_choice(n):
    src_nodes = [
        ConnectorNode('A', deg_spec='+'),
        ConnectorNode('B', deg_list=[0, 1], repeated_allowed=True),
        ConnectorNode('C', deg_list=[0, 1], repeated_allowed=True),
    ]
    conn_group = ConnectorDegreeGroupingNode('G')
    tgt_nodes = [
        ConnectorNode('D', deg_min=1, repeated_allowed=True),
    ]

    adsg = BasicDSG()
    adsg.add_edges([(n[0], node) for node in src_nodes])
    adsg.add_edges([(n[0], node) for node in tgt_nodes])

    with pytest.raises(ValueError):
        adsg.add_connection_choice('A', [], [])

    choice_node = adsg.add_connection_choice('A', src_nodes=[
        src_nodes[0],
        (conn_group, src_nodes[1:]),
    ], tgt_nodes=tgt_nodes)
    assert choice_node in adsg.graph.nodes
    assert conn_group in adsg.graph.nodes
    assert adsg.feasible
    assert not adsg.final

    assert len(list(choice_node.iter_conn_edges(adsg))) == 3
    assert len(list(adsg.iter_possible_connection_edges(choice_node))) == 3

    adsg = adsg.set_start_nodes({n[0]})
    assert adsg.feasible
    assert not adsg.final

    assert choice_node in adsg.graph.nodes
    assert conn_group in adsg.graph.nodes

    assert adsg.get_ordered_next_choice_nodes() == [choice_node]

    n_seen = 0
    for edges in adsg.iter_possible_connection_edges(choice_node):
        adsg2 = adsg.get_for_apply_connection_choice(choice_node, edges)
        assert adsg2.feasible
        assert adsg2.final

        try:
            adsg2.export_dot()
        except ModuleNotFoundError:
            pass

        n_seen += 1
    assert n_seen == 3

    with tempfile.TemporaryDirectory() as tmp_folder:
        adsg.export_drawio(f'{tmp_folder}/graph.drawio')


def test_connection_choice_excluded():
    src_nodes = [ConnectorNode('A', deg_spec='?'), ConnectorNode('B', deg_spec='?')]
    tgt_nodes = [ConnectorNode('C', deg_spec='?')]

    adsg = BasicDSG()
    choice_node = adsg.add_connection_choice('A', src_nodes, tgt_nodes, exclude=[(src_nodes[0], tgt_nodes[0])])
    adsg = adsg.set_start_nodes(set(src_nodes+tgt_nodes))
    assert len(list(adsg.iter_possible_connection_edges(choice_node))) == 2

    assert adsg.feasible
    assert not adsg.final

    with pytest.raises(ValueError):
        adsg.get_for_apply_connection_choice(choice_node, [(src_nodes[0], src_nodes[0])])
    with pytest.raises(ValueError):
        adsg.get_for_apply_connection_choice(choice_node, [(src_nodes[0], tgt_nodes[0])])

    adsg2 = adsg.get_for_apply_connection_choice(choice_node, list(adsg.iter_possible_connection_edges(choice_node))[0])
    assert adsg2.feasible
    assert adsg2.final

    # adsg.export_drawio('graph.drawio')


def test_unconnectable_connectors(n):
    c = [ConnectorNode('A', deg_spec='2..*'), ConnectorNode('B', deg_list=[1])]
    adsg = BasicDSG()
    adsg.add_edges([(n[0], c[0]), (n[0], c[1])])
    choice_node = adsg.add_connection_choice('C', [c[0]], [c[1]])

    adsg = adsg.set_start_nodes({n[0]})
    assert adsg.feasible
    assert not adsg.final

    assert len(list(adsg.iter_possible_connection_edges(choice_node))) == 0
    adsg2 = adsg.get_for_apply_connection_choice(choice_node)
    assert adsg2.final
    assert len(adsg2.unconnected_connectors) == 2
    assert not adsg2.feasible


def test_des_var_nodes(n):
    with pytest.raises(ValueError):
        DesignVariableNode('A', bounds=(1, 2, 3))
    with pytest.raises(ValueError):
        DesignVariableNode('A', bounds=(2, 1))
    with pytest.raises(ValueError):
        DesignVariableNode('A', bounds=(2, 1), options=[1, 2, 3])
    with pytest.raises(ValueError):
        DesignVariableNode('A', options=[])

    dv_node = DesignVariableNode('A', bounds=(0, 1))
    dis_dv_node = DesignVariableNode('B', options=[1, 2, 3])
    empty_dv_node = DesignVariableNode('C')

    assert not dv_node.is_discrete
    assert dis_dv_node.is_discrete

    adsg = BasicDSG()
    adsg.add_edges([
        (n[0], dv_node),
        (n[0], empty_dv_node),
    ])
    adsg = adsg.set_start_nodes({n[0]})

    assert adsg.feasible
    assert adsg.final

    with pytest.raises(ValueError):
        adsg.set_des_var_value(empty_dv_node, 0)

    for _ in range(2):
        assert adsg.des_var_value(dv_node) is None
        adsg.set_des_var_value(dv_node, .5)
        assert adsg.des_var_value(dv_node) == .5

        adsg.set_des_var_value(dv_node, -1)
        assert adsg.des_var_value(dv_node) == 0

        adsg.set_des_var_value(dv_node, 2)
        assert adsg.des_var_value(dv_node) == 1

        assert adsg.des_var_value(dis_dv_node) is None
        adsg.set_des_var_value(dis_dv_node, 0)
        assert adsg.des_var_value(dis_dv_node) == 0
        adsg.set_des_var_value(dis_dv_node, 3)
        assert adsg.des_var_value(dis_dv_node) == 2

        adsg.reset_des_var_values()


def test_get_confirmed_edges_cache(n):
    adsg = BasicDSG()
    adsg.add_edges([
        (n[0], n[1]), (n[1], n[2]),
        (n[0], n[5]),
        (n[2], n[3]), (n[3], n[4]), (n[4], n[2]),  # Loop
        (n[3], n[6]),
    ])
    adsg = adsg.set_start_nodes({n[0]})
    assert n[4] in adsg.graph.nodes

    loop_edges = {(n[2], n[3]), (n[3], n[4]), (n[4], n[2]), (n[3], n[6])}
    for start_node in n[:6]:
        for cache_start in [None]+n[:6]:
            cache = {}
            if cache_start is not None:
                get_confirmed_edges_for_node(adsg.graph, cache_start, cache=cache)

            confirmed_edges = get_confirmed_edges_for_node(adsg.graph, start_node, cache=cache)
            confirmed_edges_ = {edge[:2] for edge in confirmed_edges}

            if start_node == n[0]:
                assert confirmed_edges_ == {(n[0], n[1]), (n[1], n[2]), (n[0], n[5])} | loop_edges
            elif start_node == n[1]:
                assert confirmed_edges_ == {(n[1], n[2])} | loop_edges
            elif start_node in (n[5], n[6]):
                assert confirmed_edges_ == set()
            else:
                assert confirmed_edges_ == loop_edges

            confirmed_edges2 = get_confirmed_edges_for_node(adsg.graph, start_node, cache=cache)
            assert confirmed_edges2 == confirmed_edges


def test_circular_choices(n):
    adsg = BasicDSG()
    adsg.add_edges([
        (n[1], n[11]), (n[11], n[3]),
        (n[2], n[12]), (n[12], n[4]),
    ])
    c3 = adsg.add_selection_choice('C3', n[3], [n[31], n[32]])
    c4 = adsg.add_selection_choice('C4', n[4], [n[41], n[42]])
    adsg.add_edges([
        (n[32], n[5]), (n[5], n[15]),
        (n[42], n[6]), (n[6], n[15]),
        (n[15], n[3]), (n[15], n[4]),
    ])
    adsg = adsg.set_start_nodes({n[1], n[2]})
    assert len(adsg.choice_nodes) == 2

    for i in range(2):
        choice_node = adsg.get_ordered_next_choice_nodes()[0]
        assert isinstance(choice_node, SelectionChoiceNode)
        assert choice_node == c3
        opt_nodes = adsg.get_option_nodes(choice_node)

        adsg2 = adsg.get_for_apply_selection_choice(choice_node, opt_nodes[i])

        for j in range(2):
            choice_node = adsg2.get_ordered_next_choice_nodes()[0]
            assert isinstance(choice_node, SelectionChoiceNode)
            assert choice_node == c4
            opt_nodes = adsg2.get_option_nodes(choice_node)

            adsg3 = adsg2.get_for_apply_selection_choice(choice_node, opt_nodes[j])
            assert adsg3.final

            nodes = set(adsg3.graph.nodes)
            assert (n[31] in nodes) == (i == 0)
            assert (n[32] in nodes) == (i == 1)
            assert (n[41] in nodes) == (j == 0)
            assert (n[42] in nodes) == (j == 1)
            assert (n[15] in nodes) == (i == 1 or j == 1)


def test_sel_choice_scenarios(n):
    adsg = BasicDSG()
    c11 = adsg.add_selection_choice('C11', n[1], [n[11], n[12]])
    adsg.add_selection_choice('C22', n[2], [n[21], n[22]])
    adsg.add_selection_choice('C24', n[4], [n[21], n[22]])
    adsg.add_selection_choice('C33', n[3], [n[31], n[32]])
    adsg.add_selection_choice('C55', n[5], [n[35], n[36]])
    adsg.add_edges([
        (n[11], n[2]), (n[12], n[3]),
    ])

    adsg.add_incompatibility_constraint([n[21], n[31]])
    adsg.add_incompatibility_constraint([n[11], n[35]])
    adsg.add_incompatibility_constraint([n[12], n[36]])

    adsg = adsg.set_start_nodes({n[1], n[4], n[5]})
    assert len(adsg.choice_nodes) == 5

    im = InfluenceMatrix(adsg)
    scs = im.base_sel_choice_scenarios
    assert len(scs) == 5

    assert im.selection_choice_nodes[0] == c11
    assert scs[0].choice_nodes == [im.selection_choice_nodes[0]]
    assert scs[0].n_opts == 2
    assert len(scs[0].opt_idx_combinations) == 4
    assert np.all(scs[0].opt_idx_combinations[0] == np.array([[0, 1]]).T)


def test_early_start_node_def(n):
    adsg = BasicDSG()
    adsg.add_edges([
        (n[0], n[1]), (n[2], n[3]),
    ])
    adsg = adsg.set_start_nodes({n[0], n[2]})

    c1 = adsg.add_selection_choice('C1', n[0], n[4:6])
    c2 = adsg.add_selection_choice('C2', n[3], n[6:8])

    assert adsg.get_ordered_next_choice_nodes() == [c1, c2]
