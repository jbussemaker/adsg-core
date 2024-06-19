import pytest
from adsg_core.graph.adsg_basic import *
from adsg_core.graph.sup import *


@pytest.fixture
def s():
    return [SupNode(str(i)) for i in range(20)]


def test_sup_dsg():
    sup_dsg = SupDSG()
    b = SupNode('b', ref='test')
    sup_dsg.add_edge(SupNode('a'), b)
    sup_dsg.add_selection_choice('choice', b, [SupNode('c'), SupNode('d')])
    assert len(sup_dsg.graph.nodes) == 5
    assert len(sup_dsg.sup_nodes) == 4
    assert sup_dsg.get_by_ref('test') == b

    with pytest.raises(RuntimeError):
        sup_dsg.set_start_nodes()


def test_choice_option_mapping(n, s):
    adsg = BasicDSG()
    adsg.add_edge(n[0], n[1])
    adsg_choice = adsg.add_selection_choice('c', n[1], [n[2], n[3]])
    adsg.add_edge(n[2], n[3])
    adsg = adsg.set_start_nodes()
    assert adsg.derivation_start_nodes == {n[0]}

    sup_dsg = SupDSG()
    sup_choice = sup_dsg.add_selection_choice('c', s[0], [s[1], s[2]])

    node_map = {
        n[2]: s[1],
        n[3]: s[2],
    }
    mapping = SupSelChoiceOptionMapping(adsg_choice, node_map)
    assert str(mapping)
    assert repr(mapping)
    sup_dsg.add_mapping(sup_choice, adsg, mapping)
    sup_dsg = sup_dsg.set_start_nodes()

    with pytest.raises(RuntimeError):
        sup_dsg.resolve(adsg)

    opt_nodes = adsg.get_option_nodes(adsg_choice)
    assert opt_nodes == [n[2], n[3]]
    for opt_node in opt_nodes:
        adsg_final = adsg.get_for_apply_selection_choice(adsg_choice, opt_node)
        assert adsg_final.final

        sup_dsg_resolved = sup_dsg.resolve(adsg_final)
        assert sup_dsg_resolved.final

        assert node_map[opt_node] in sup_dsg_resolved.graph.nodes


def test_choice_option_mapping_errors(n, s):
    adsg = BasicDSG()
    adsg_choice = adsg.add_selection_choice('c', n[0], [n[1], n[2]])
    adsg = adsg.set_start_nodes()

    sup_dsg = SupDSG()
    sup_choice = sup_dsg.add_selection_choice('c', s[0], [s[1], s[2]])
    sup_dsg.add_selection_choice('d', s[2], [s[3], s[4]])

    with pytest.raises(RuntimeError):
        sup_dsg = sup_dsg.set_start_nodes()

    with pytest.raises(SupInitializationError):
        sup_dsg.add_mapping(sup_choice, adsg, SupSelChoiceOptionMapping(SelectionChoiceNode(), {}))
    with pytest.raises(SupInitializationError):
        sup_dsg.add_mapping(sup_choice, adsg, SupSelChoiceOptionMapping(adsg_choice, {}))
    with pytest.raises(SupInitializationError):
        sup_dsg.add_mapping(sup_choice, adsg, SupSelChoiceOptionMapping(adsg_choice, {n[1]: s[1]}))
    with pytest.raises(SupInitializationError):
        sup_dsg.add_mapping(sup_choice, adsg, SupSelChoiceOptionMapping(adsg_choice, {n[1]: s[1], n[2]: s[3]}))

    mapping = SupSelChoiceOptionMapping(adsg_choice, {n[1]: s[1], n[2]: s[1]})
    sup_dsg.add_mapping(sup_choice, adsg, mapping)
    sup_dsg.add_mapping(sup_choice, adsg, mapping)
    with pytest.raises(RuntimeError):
        sup_dsg = sup_dsg.set_start_nodes()
    sup_dsg._choice_mappings = sup_dsg._choice_mappings[:1]
    with pytest.raises(RuntimeError):
        sup_dsg = sup_dsg.set_start_nodes()


def test_inactive_choice_mapping(n, s):
    adsg = BasicDSG()
    adsg.add_selection_choice('a', n[0], [n[1], n[2]])
    adsg_choice = adsg.add_selection_choice('b', n[2], [n[3], n[4]])
    adsg = adsg.set_start_nodes()

    sup_dsg = SupDSG()
    sup_choice = sup_dsg.add_selection_choice('c', s[0], [s[1], s[2], s[3]])
    with pytest.raises(SupInitializationError):
        sup_dsg.add_mapping(sup_choice, adsg, SupSelChoiceOptionMapping(adsg_choice, {n[3]: s[1], n[4]: s[2]}))
    sup_dsg.add_mapping(sup_choice, adsg, SupSelChoiceOptionMapping(adsg_choice, {n[3]: s[1], n[4]: s[2], None: s[3]}))
    sup_dsg = sup_dsg.set_start_nodes()

    adsg_inactive = adsg.get_for_apply_selection_choice(adsg.get_ordered_next_choice_nodes()[0], n[1])
    assert adsg_inactive.final
    sup_dsg.resolve(adsg_inactive)


def test_existence_mapping(n, s):
    adsg = BasicDSG()
    adsg_choice = adsg.add_selection_choice('a', n[0], [n[1], n[2], n[3]])
    adsg.add_edges([(n[1], n[2]), (n[2], n[3])])
    adsg = adsg.set_start_nodes()

    sup_dsg = SupDSG()
    sup_choice = sup_dsg.add_selection_choice('b', s[0], [s[1], s[2], s[3]])

    with pytest.raises(SupInitializationError):
        sup_dsg.add_mapping(sup_choice, adsg, SupExistenceMapping({n[4]: s[1]}))
    with pytest.raises(SupInitializationError):
        sup_dsg.add_mapping(sup_choice, adsg, SupExistenceMapping({n[1]: s[1], n[2]: s[2], n[3]: s[3]}))
    with pytest.raises(SupInitializationError):
        sup_dsg.add_mapping(sup_choice, adsg, SupExistenceMapping({n[1]: s[1], n[2]: s[0], None: s[3]}))

    mapping = SupExistenceMapping({
        None: s[3],
        n[1]: s[1],
        n[2]: s[2],
    })
    assert str(mapping)
    assert repr(mapping)
    sup_dsg.add_mapping(sup_choice, adsg, mapping)
    sup_dsg = sup_dsg.set_start_nodes()

    expected = [(n[1], s[1]), (n[2], s[2]), (n[3], s[3])]
    for opt_node, sup_node in expected:
        adsg_final = adsg.get_for_apply_selection_choice(adsg_choice, opt_node)
        sup_dsg_resolved = sup_dsg.resolve(adsg_final)
        assert sup_node in sup_dsg_resolved.graph.nodes


def test_chained_mapping(n, s):
    adsg = BasicDSG()
    adsg_choice = adsg.add_selection_choice('c', n[0], [n[1], n[2]])
    adsg = adsg.set_start_nodes()

    sup_dsg1 = SupDSG()
    sup_choice = sup_dsg1.add_selection_choice('c', s[0], [s[1], s[2]])
    sup_dsg1.add_mapping(sup_choice, adsg, SupSelChoiceOptionMapping(adsg_choice, {
        n[1]: s[1],
        n[2]: s[2],
    }))
    sup_dsg1 = sup_dsg1.set_start_nodes()

    sup_dsg2 = SupDSG()
    sup_choice2 = sup_dsg2.add_selection_choice('c', s[3], [s[4], s[5]])
    sup_dsg2.add_mapping(sup_choice2, sup_dsg1, SupSelChoiceOptionMapping(sup_choice, {
        s[1]: s[4],
        s[2]: s[5],
    }))
    sup_dsg2 = sup_dsg2.set_start_nodes()

    for opt_node in adsg.get_option_nodes(adsg_choice):
        adsg_final = adsg.get_for_apply_selection_choice(adsg_choice, opt_node)
        sup_dsg1_resolved = sup_dsg1.resolve(adsg_final)
        sup_dsg2_resolved = sup_dsg2.resolve(sup_dsg1_resolved)

        if opt_node == n[1]:
            assert s[4] in sup_dsg2_resolved.graph.nodes
        elif opt_node == n[2]:
            assert s[5] in sup_dsg2_resolved.graph.nodes
