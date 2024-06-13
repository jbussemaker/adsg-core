import pytest
import itertools
from adsg_core.graph.adsg_basic import *
from adsg_core.graph.adsg_nodes import *


def test_linked_choice(n):
    adsg = BasicDSG()
    c1 = adsg.add_selection_choice('A', n[0], [n[1], n[2]])
    c2 = adsg.add_selection_choice('B', n[3], [n[4], n[5]])
    dv = DesignVariableNode('C', options=[1, 2])
    adsg.add_edge(n[6], dv)
    adsg = adsg.set_start_nodes({n[0], n[3], n[6]})
    assert len(adsg.choice_nodes) == 2

    with pytest.raises(ValueError):
        adsg.constrain_choices(ChoiceConstraintType.LINKED, [c1, dv])

    assert adsg.is_constrained_choice(c1) is None
    adsg = adsg.constrain_choices(ChoiceConstraintType.LINKED, [c1, c2])
    con = adsg.is_constrained_choice(c1)
    assert con
    assert adsg.is_constrained_choice(c2) == con
    assert con.type == ChoiceConstraintType.LINKED
    assert con.nodes == [c1, c2]
    assert con.options == [[n[1], n[2]], [n[4], n[5]]]

    assert adsg.ordered_choice_nodes(adsg.choice_nodes) == [c1, c2]
    assert adsg.get_ordered_next_choice_nodes() == [c1, c2]

    assert not adsg.final
    for choice_node in [c1, c2]:
        for opt_node in adsg.get_option_nodes(choice_node):
            adsg2 = adsg.get_for_apply_selection_choice(choice_node, opt_node)
            assert adsg2.get_taken_single_selection_choices()[0][0] == list({c1, c2}-{choice_node})[0]
            assert len(adsg2.choice_nodes) == 0
            assert adsg2.final

    assert adsg.fingerprint()
    assert hash(adsg)

    adsg.export_dot()
    adsg.export_drawio()
    # adsg.render()


def test_overlapping_constraint(n):
    adsg = BasicDSG()
    for i in range(3):
        adsg.add_selection_choice(str(i), n[i*3], [n[i*3+1], n[i*3+2]])
    adsg = adsg.set_start_nodes()
    assert len(adsg.choice_nodes) == 3

    adsg = adsg.constrain_choices(ChoiceConstraintType.LINKED, adsg.choice_nodes[:2])
    with pytest.raises(RuntimeError):
        adsg.constrain_choices(ChoiceConstraintType.LINKED, adsg.choice_nodes[1:])

    assert adsg.fingerprint()
    assert hash(adsg)


def test_dependent_connection_choice(n):
    adsg = BasicDSG()
    cn = [ConnectorNode(deg_spec='*') for _ in range(4)]
    c1 = adsg.add_connection_choice('A', [cn[0]], [cn[1]])
    c2 = adsg.add_connection_choice('B', [cn[0]], [cn[1]])
    adsg.add_edges([(n[0], cn_) for cn_ in cn])
    adsg = adsg.set_start_nodes()
    assert len(adsg.choice_nodes) == 2

    with pytest.raises(RuntimeError):
        adsg.constrain_choices(ChoiceConstraintType.PERMUTATION, [c1, c2])

    assert adsg.fingerprint()
    assert hash(adsg)


def test_linked_des_vars(n):
    adsg = BasicDSG()
    dv1 = DesignVariableNode('A', bounds=(0, 1))
    dv2 = DesignVariableNode('B', bounds=(.5, 2))
    adsg.add_edges([(n[0], dv1), (n[1], dv2)])
    adsg = adsg.set_start_nodes()

    adsg.set_des_var_value(dv1, .5)
    assert adsg.des_var_value(dv2) is None

    with pytest.raises(ValueError):
        adsg.constrain_choices(ChoiceConstraintType.PERMUTATION, [dv1, dv2])
    with pytest.raises(ValueError):
        adsg.constrain_choices(ChoiceConstraintType.UNORDERED, [dv1, dv2])
    adsg.constrain_choices(ChoiceConstraintType.LINKED, [dv1, dv2])
    assert adsg.fingerprint()
    assert hash(adsg)
    assert len(adsg.des_var_nodes) == 1
    assert len(adsg.all_des_var_nodes) == 2

    adsg.set_des_var_value(dv1, .5)
    assert adsg.des_var_value(dv2) == 1.25

    adsg.set_des_var_value(dv2, 2)
    assert adsg.des_var_value(dv1) == 1


def test_get_next_decision_dependent(n):
    adsg = BasicDSG()
    c1 = adsg.add_selection_choice('A', n[0], [n[1], n[2]])
    adsg.add_edges([(n[1], n[3]), (n[2], n[3])])
    c2 = adsg.add_selection_choice('B', n[3], [n[4], n[5]])
    adsg = adsg.set_start_nodes()
    assert len(adsg.choice_nodes) == 2

    adsg = adsg.constrain_choices(ChoiceConstraintType.LINKED, [c1, c2])
    assert adsg.fingerprint()
    assert hash(adsg)

    assert adsg.get_ordered_next_choice_nodes() == [c1]
    adsg = adsg.get_for_apply_selection_choice(c1, n[1])
    assert n[4] in adsg.get_confirmed_graph().graph.nodes
    assert adsg.final


def test_sel_choice_permutation_constraint(n):
    for n_sel_choice in [2, 3, 4]:
        adsg = BasicDSG()
        for i in range(n_sel_choice):
            adsg.add_selection_choice(str(i), n[i*4], n[i*4+1:i*4+4])
        adsg = adsg.set_start_nodes()
        assert len(adsg.derivation_start_nodes) == n_sel_choice

        choice_nodes = adsg.ordered_choice_nodes(adsg.choice_nodes)
        assert len(choice_nodes) == n_sel_choice
        init_opt_nodes = [adsg.get_option_nodes(cn) for cn in choice_nodes]

        adsg = adsg.constrain_choices(ChoiceConstraintType.PERMUTATION, choice_nodes)
        assert adsg.fingerprint()
        assert hash(adsg)

        if n_sel_choice == 4:
            assert not adsg.feasible
            assert adsg.final

            targets = set()
            seen_choice_nodes = set()
            for src, tgt, _, data in adsg.get_confirmed_incompatibility_edges():
                targets |= {src, tgt}
                assert 'choice_node' in data
                seen_choice_nodes.add(data['choice_node'])
            assert len(set(adsg.derivation_start_nodes) - targets) == 0
            assert seen_choice_nodes == set(choice_nodes)
            continue

        for choice_nodes_order in itertools.permutations(choice_nodes):
            i_selected_all = set()
            for i_opts in itertools.product(*[list(range(3-i)) for i in range(min(n_sel_choice, 3))]):
                adsg2 = adsg
                for i, choice_node in enumerate(choice_nodes_order):
                    assert choice_node in adsg2.get_ordered_next_choice_nodes()
                    opt_nodes = adsg2.get_option_nodes(choice_node)
                    assert len(opt_nodes) == 3-i

                    adsg2 = adsg2.get_for_apply_selection_choice(choice_node, opt_nodes[i_opts[i]])
                    if len(opt_nodes) == 2:
                        if n_sel_choice > 2:
                            assert len(adsg2.get_taken_single_selection_choices()) > 0
                        break
                    else:
                        assert len(adsg2.get_taken_single_selection_choices()) == 0

                assert adsg2.final
                assert adsg2.feasible == (n_sel_choice < 4)

                nodes = adsg2.graph.nodes
                i_selected = []
                for opt_nodes in init_opt_nodes:
                    for i_opt, opt_node in enumerate(opt_nodes):
                        if opt_node in nodes:
                            i_selected.append(i_opt)
                            break

                i_selected_all.add(tuple(i_selected))

            assert len(i_selected_all) == len(list(itertools.permutations('ABC', min(n_sel_choice, 3))))


def test_sel_choice_permutation_constraint_no_auto(n):
    n_sel_choice = 4
    adsg = BasicDSG()
    for i in range(n_sel_choice):
        adsg.add_selection_choice(str(i), n[i*4], n[i*4+1:i*4+4])
    adsg = adsg.set_start_nodes()
    assert len(adsg.derivation_start_nodes) == n_sel_choice

    choice_nodes = adsg.ordered_choice_nodes(adsg.choice_nodes)
    assert len(choice_nodes) == n_sel_choice

    adsg = adsg.constrain_choices(ChoiceConstraintType.PERMUTATION, choice_nodes, remove_infeasible_choices=False)
    assert adsg.fingerprint()
    assert hash(adsg)

    assert adsg.feasible
    assert len(adsg.choice_nodes) > 0

    adsg = adsg.resolve_single_selection_choices()
    assert not adsg.feasible
    assert adsg.final


def test_sel_choice_unordered_constraint(n):
    for n_sel_choice in [2, 3, 4]:
        adsg = BasicDSG()
        for i in range(n_sel_choice):
            adsg.add_selection_choice(str(i), n[i*4], n[i*4+1:i*4+4])
        adsg = adsg.set_start_nodes()
        assert len(adsg.derivation_start_nodes) == n_sel_choice

        choice_nodes = adsg.ordered_choice_nodes(adsg.choice_nodes)
        assert len(choice_nodes) == n_sel_choice
        init_opt_nodes = [adsg.get_option_nodes(cn) for cn in choice_nodes]

        adsg = adsg.constrain_choices(ChoiceConstraintType.UNORDERED, choice_nodes)
        assert adsg.fingerprint()
        assert hash(adsg)

        for choice_nodes_order in itertools.permutations(choice_nodes):
            i_selected_all = set()
            for i_opts in itertools.product(*[list(range(3)) for _ in range(n_sel_choice)]):
                adsg2 = adsg
                stop = False
                for i, choice_node in enumerate(choice_nodes_order):
                    if choice_node not in adsg2.get_ordered_next_choice_nodes():
                        continue
                    opt_nodes = adsg2.get_option_nodes(choice_node)
                    if i_opts[i] >= len(opt_nodes):
                        stop = True
                        break

                    adsg2 = adsg2.get_for_apply_selection_choice(choice_node, opt_nodes[i_opts[i]])
                if stop:
                    continue

                assert adsg2.final
                assert adsg2.feasible

                nodes = adsg2.graph.nodes
                i_selected = []
                for opt_nodes in init_opt_nodes:
                    for i_opt, opt_node in enumerate(opt_nodes):
                        if opt_node in nodes:
                            i_selected.append(i_opt)
                            break

                i_selected_all.add(tuple(i_selected))

            assert len(i_selected_all) == len(list(itertools.combinations_with_replacement('ABC', n_sel_choice)))


def test_unordered_n_opts(n):
    adsg = BasicDSG()
    adsg.add_selection_choice('A', n[0], n[1:3])
    adsg.add_selection_choice('B', n[3], n[4:7])
    adsg = adsg.set_start_nodes()

    with pytest.raises(ValueError):
        adsg.constrain_choices(ChoiceConstraintType.UNORDERED, adsg.choice_nodes)
    with pytest.raises(ValueError):
        adsg.constrain_choices(ChoiceConstraintType.UNORDERED_NOREPL, adsg.choice_nodes)


def test_sel_choice_unordered_non_replacing_constraint(n):
    for n_sel_choice in [2, 3, 4]:
        adsg = BasicDSG()
        for i in range(n_sel_choice):
            adsg.add_selection_choice(str(i), n[i*4], n[i*4+1:i*4+4])
        adsg = adsg.set_start_nodes()
        assert len(adsg.derivation_start_nodes) == n_sel_choice

        choice_nodes = adsg.ordered_choice_nodes(adsg.choice_nodes)
        assert len(choice_nodes) == n_sel_choice
        init_opt_nodes = [adsg.get_option_nodes(cn) for cn in choice_nodes]

        adsg = adsg.constrain_choices(ChoiceConstraintType.UNORDERED_NOREPL, choice_nodes)
        assert adsg.fingerprint()
        assert hash(adsg)
        if n_sel_choice >= 3:
            assert [node for node, _ in adsg.get_taken_single_selection_choices()] == choice_nodes
            assert adsg.feasible == (n_sel_choice < 4)
            assert adsg.final
            continue

        for choice_node in choice_nodes:
            assert len(adsg.get_option_nodes(choice_node)) == 3-(n_sel_choice-1)

        for choice_nodes_order in itertools.permutations(choice_nodes):
            i_selected_all = set()
            for i_opts in itertools.product(*[list(range(3-i)) for i in range(min(n_sel_choice, 3))]):
                adsg2 = adsg
                stop = False
                for i, choice_node in enumerate(choice_nodes_order):
                    if choice_node not in adsg2.get_ordered_next_choice_nodes():
                        continue
                    opt_nodes = adsg2.get_option_nodes(choice_node)
                    if i_opts[i] >= len(opt_nodes):
                        stop = True
                        break

                    adsg2 = adsg2.get_for_apply_selection_choice(choice_node, opt_nodes[i_opts[i]])
                if stop:
                    continue

                assert adsg2.final
                assert adsg2.feasible == (n_sel_choice < 4)

                nodes = adsg2.graph.nodes
                i_selected = []
                for opt_nodes in init_opt_nodes:
                    for i_opt, opt_node in enumerate(opt_nodes):
                        if opt_node in nodes:
                            i_selected.append(i_opt)
                            break

                i_selected_all.add(tuple(i_selected))

            assert len(i_selected_all) == len(list(itertools.combinations('ABC', n_sel_choice)))


def test_sel_choice_unordered_non_replacing_constraint_no_auto(n):
    for n_sel_choice in [3, 4]:
        adsg = BasicDSG()
        for i in range(n_sel_choice):
            adsg.add_selection_choice(str(i), n[i*4], n[i*4+1:i*4+4])
        adsg = adsg.set_start_nodes()
        assert len(adsg.derivation_start_nodes) == n_sel_choice

        choice_nodes = adsg.ordered_choice_nodes(adsg.choice_nodes)
        assert len(choice_nodes) == n_sel_choice

        adsg = adsg.constrain_choices(ChoiceConstraintType.UNORDERED_NOREPL, choice_nodes, remove_infeasible_choices=False)
        assert adsg.fingerprint()
        assert hash(adsg)
        if n_sel_choice >= 3:
            assert adsg.feasible
            assert len(adsg.choice_nodes) > 0

            adsg = adsg.resolve_single_selection_choices()
            assert [node for node, _ in adsg.get_taken_single_selection_choices()] == choice_nodes
            assert adsg.feasible == (n_sel_choice < 4)
            assert adsg.final
            continue
