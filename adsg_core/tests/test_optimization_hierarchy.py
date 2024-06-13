import itertools
import numpy as np
from adsg_core.graph.adsg_basic import *
from adsg_core.graph.adsg_nodes import *
from adsg_core.optimization.hierarchy import *
from adsg_core.optimization.graph_processor import *


def _get_hierarchy_analyzer(adsg) -> HierarchyAnalyzerBase:
    return HierarchyAnalyzer(adsg)
    # return SelChoiceEncHierarchyAnalyzer(adsg)


def test_merge_sel_choice_scenarios_indep(n):
    adsg = BasicDSG()
    adsg.add_selection_choice('C1', n[1], [n[11], n[12]])
    adsg.add_selection_choice('C2', n[2], [n[21], n[22]])
    adsg = adsg.set_start_nodes({n[1], n[2]})

    an = _get_hierarchy_analyzer(adsg)
    scs = an._influence_matrix.base_sel_choice_scenarios
    assert len(scs) == 2
    assert np.all(scs[0].choice_idx == [0])
    assert np.all(scs[0].n_combinations_unique == [2])
    assert np.all(scs[1].choice_idx == [1])
    assert np.all(scs[1].n_combinations_unique == [2])
    assert scs[0].input_status_matrix.shape == (1, 0)
    assert scs[0].unique_scenario_idx.shape == (1,)

    if isinstance(an, HierarchyAnalyzer):
        merged = an._merge_scenarios(scs[0], scs[1])
        assert np.all(merged.choice_idx == [0, 1])
        assert merged.input_status_matrix.shape == (1, 0)
        assert merged.unique_scenario_idx.shape == (1,)
        assert np.all(merged.n_combinations_unique == [4])

    assert an.n_combinations == 4
    assert np.all(~an.selection_choice_is_forced)
    assert an.n_opts == [2, 2]
    assert an.n_design_space == 4

    assert np.all(an.get_choice_option_indices() == np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ]))
    assert an.get_existence_array().shape[0] == 4
    an._assert_behavior()


def test_merge_sel_choice_scenarios_dep(n):
    adsg = BasicDSG()
    adsg.add_edges([
        (n[12], n[2]),
    ])
    adsg.add_selection_choice('C1', n[1], [n[11], n[12]])
    adsg.add_selection_choice('C2', n[2], [n[21], n[22]])
    adsg = adsg.set_start_nodes({n[1]})

    an = _get_hierarchy_analyzer(adsg)
    scs = an._influence_matrix.base_sel_choice_scenarios
    assert len(scs) == 2
    assert np.all(scs[0].choice_idx == [0])
    assert np.all(scs[0].n_combinations_unique == [2])
    assert scs[0].input_status_matrix.shape == (1, 0)
    assert scs[0].unique_scenario_idx.shape == (1,)
    assert np.all(scs[1].choice_idx == [1])
    assert np.all(scs[1].n_combinations_unique == [1, 2])
    assert scs[1].input_status_matrix.shape == (2, 1)
    assert scs[1].unique_scenario_idx.shape == (2,)

    if isinstance(an, HierarchyAnalyzer):
        merged = an._merge_scenarios(scs[0], scs[1])
        assert np.all(merged.choice_idx == [0, 1])
        assert merged.input_status_matrix.shape == (1, 0)
        assert merged.unique_scenario_idx.shape == (1,)
        assert np.all(merged.n_combinations_unique == [3])

    assert an.n_combinations == 3
    assert np.all(~an.selection_choice_is_forced)
    assert an.n_opts == [2, 2]
    assert an.n_design_space == 4

    assert np.all(an.get_choice_option_indices() == np.array([
        [0, -1],
        [1, 0],
        [1, 1],
    ]))
    assert an.get_existence_array().shape[0] == 3


def test_merge_sel_choice_scenarios_coupled(n):
    adsg = BasicDSG()
    adsg.add_selection_choice('C1', n[1], [n[11], n[12]])
    adsg.add_selection_choice('C2', n[2], [n[21], n[22]])
    adsg.add_incompatibility_constraint([n[11], n[21]])
    adsg = adsg.set_start_nodes({n[1], n[2]})

    an = _get_hierarchy_analyzer(adsg)
    scs = an._influence_matrix.base_sel_choice_scenarios
    assert len(scs) == 2
    assert np.all(scs[0].choice_idx == [0])
    assert np.all(scs[0].n_combinations_unique == [2, 1])
    assert scs[0].input_status_matrix.shape == (2, 1)
    assert scs[0].unique_scenario_idx.shape == (2,)
    assert np.all(scs[1].choice_idx == [1])
    assert np.all(scs[1].n_combinations_unique == [2, 1])
    assert scs[1].input_status_matrix.shape == (2, 1)
    assert scs[1].unique_scenario_idx.shape == (2,)

    if isinstance(an, HierarchyAnalyzer):
        merged = an._merge_scenarios(scs[0], scs[1])
        assert np.all(merged.choice_idx == [0, 1])
        assert merged.input_status_matrix.shape == (1, 0)
        assert merged.unique_scenario_idx.shape == (1,)
        assert np.all(merged.n_combinations_unique == [3])

    assert an.n_combinations == 3
    assert np.all(~an.selection_choice_is_forced)
    assert an.n_opts == [2, 2]
    assert an.n_design_space == 4

    assert np.all(an.get_choice_option_indices() == np.array([
        [0, 1],
        [1, 0],
        [1, 1],
    ]))
    assert an.get_existence_array().shape[0] == 3
    an._assert_behavior()


def test_merge_sel_choice_scenarios_coupled_cond_act(n):
    adsg = BasicDSG()
    adsg.add_selection_choice('C1', n[1], [n[11], n[12]])
    adsg.add_selection_choice('C2', n[2], [n[21], n[22]])
    adsg.add_selection_choice('C3', n[3], [n[31], n[32]])
    adsg.add_edges([
        (n[12], n[2]), (n[12], n[3]),
    ])
    adsg.add_incompatibility_constraint([n[21], n[31]])
    adsg = adsg.set_start_nodes({n[1]})

    an = _get_hierarchy_analyzer(adsg)
    assert an.n_combinations == 4
    assert np.all(~an.selection_choice_is_forced)
    assert an.n_opts == [2, 2, 2]

    assert np.all(an.get_choice_option_indices() == np.array([
        [0, -1, -1],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1],
    ]))
    an._assert_behavior()


def test_merge_sel_choice_scenarios_coupled_both(n):
    adsg = BasicDSG()
    adsg.add_selection_choice('C1', n[1], [n[11], n[12]])
    adsg.add_selection_choice('C2', n[2], [n[21], n[22]])
    adsg.add_incompatibility_constraint([n[11], n[21]])
    adsg.add_incompatibility_constraint([n[12], n[22]])
    adsg = adsg.set_start_nodes({n[1], n[2]})

    an = _get_hierarchy_analyzer(adsg)
    scs = an._influence_matrix.base_sel_choice_scenarios
    assert len(scs) == 2
    assert np.all(scs[0].choice_idx == [0])
    assert scs[0].input_status_matrix.shape == (4, 2)
    assert scs[0].unique_scenario_idx.shape == (4,)
    assert np.all(scs[0].n_combinations_unique == [2, 1, 1, 0])
    assert np.all(scs[1].choice_idx == [1])
    assert scs[1].input_status_matrix.shape == (4, 2)
    assert scs[1].unique_scenario_idx.shape == (4,)
    assert np.all(scs[1].n_combinations_unique == [2, 1, 1, 0])

    if isinstance(an, HierarchyAnalyzer):
        merged = an._merge_scenarios(scs[0], scs[1])
        assert np.all(merged.choice_idx == [0, 1])
        assert merged.input_status_matrix.shape == (1, 0)
        assert merged.unique_scenario_idx.shape == (1,)
        assert np.all(merged.n_combinations_unique == [2])

    assert an.n_combinations == 2
    assert np.all(an.selection_choice_is_forced == [False, True])
    assert an.n_opts == [2, 2]
    assert an.n_design_space == 4

    assert np.all(an.get_choice_option_indices() == np.array([
        [0, 1],
        [1, 0],
    ]))
    assert an.get_existence_array().shape[0] == 2
    an._assert_behavior()


def test_merge_sel_choice_scenarios_shared(n):
    adsg = BasicDSG()
    adsg.add_selection_choice('C1', n[1], [n[11], n[12]])
    adsg.add_selection_choice('C2', n[2], [n[11], n[12]])
    adsg = adsg.set_start_nodes({n[1], n[2]})

    an = _get_hierarchy_analyzer(adsg)
    scs = an._influence_matrix.base_sel_choice_scenarios
    assert len(scs) == 2
    assert np.all(scs[0].choice_idx == [0])
    assert scs[0].input_status_matrix.shape == (1, 0)
    assert scs[0].unique_scenario_idx.shape == (1,)
    assert np.all(scs[0].n_combinations_unique == [2])
    assert np.all(scs[1].choice_idx == [1])
    assert scs[1].input_status_matrix.shape == (1, 0)
    assert scs[1].unique_scenario_idx.shape == (1,)
    assert np.all(scs[1].n_combinations_unique == [2])

    if isinstance(an, HierarchyAnalyzer):
        merged = an._merge_scenarios(scs[0], scs[1])
        assert np.all(merged.choice_idx == [0, 1])
        assert merged.input_status_matrix.shape == (1, 0)
        assert merged.unique_scenario_idx.shape == (1,)
        assert np.all(merged.n_combinations_unique == [4])

    assert an.n_combinations == 4
    assert np.all(~an.selection_choice_is_forced)
    assert an.n_opts == [2, 2]
    assert an.n_design_space == 4

    assert np.all(an.get_choice_option_indices() == np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ]))
    assert an.get_existence_array().shape[0] == 4
    an._assert_behavior()


def test_merge_sel_choice_scenarios_shared_constrained(n):
    adsg = BasicDSG()
    adsg.add_selection_choice('C1', n[1], [n[11], n[12]])
    adsg.add_selection_choice('C2', n[2], [n[11], n[12]])
    adsg.add_incompatibility_constraint([n[11], n[12]])
    adsg = adsg.set_start_nodes({n[1], n[2]})

    an = _get_hierarchy_analyzer(adsg)
    scs = an._influence_matrix.base_sel_choice_scenarios
    assert len(scs) == 2
    assert np.all(scs[0].choice_idx == [0])
    assert scs[0].input_status_matrix.shape == (4, 2)
    assert scs[0].unique_scenario_idx.shape == (4,)
    assert np.all(scs[0].n_combinations_unique == [2, 1, 1, 0])
    assert np.all(scs[1].choice_idx == [1])
    assert scs[1].input_status_matrix.shape == (4, 2)
    assert scs[1].unique_scenario_idx.shape == (4,)
    assert np.all(scs[1].n_combinations_unique == [2, 1, 1, 0])

    if isinstance(an, HierarchyAnalyzer):
        merged = an._merge_scenarios(scs[0], scs[1])
        assert np.all(merged.choice_idx == [0, 1])
        assert merged.input_status_matrix.shape == (1, 0)
        assert merged.unique_scenario_idx.shape == (1,)
        assert np.all(merged.n_combinations_unique == [2])

    assert an.n_combinations == 2
    assert np.all(an.selection_choice_is_forced == [False, True])
    assert an.n_opts == [2, 2]
    assert an.n_design_space == 4

    assert np.all(an.get_choice_option_indices() == np.array([
        [0, 0],
        [1, 1],
    ]))
    assert an.get_existence_array().shape[0] == 2
    an._assert_behavior()


def test_sel_choice_scenarios_shared_dep(n):
    adsg = BasicDSG()
    adsg.add_selection_choice('C1', n[1], [n[11], n[12]])
    adsg.add_selection_choice('C2', n[2], [n[11], n[12]])
    adsg.add_selection_choice('C3', n[3], [n[31], n[32]])
    adsg.add_edge(n[12], n[3])
    adsg = adsg.set_start_nodes({n[1], n[2]})

    an = _get_hierarchy_analyzer(adsg)
    scs = an._influence_matrix.base_sel_choice_scenarios
    assert len(scs) == 3

    assert an.n_combinations == 7
    assert np.all(~an.selection_choice_is_forced)
    assert an.n_opts == [2, 2, 2]
    assert an.n_design_space == 8

    assert np.all(an.get_choice_option_indices() == np.array([
        [0, 0, -1],
        [0, 1, 0],
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1],
    ]))
    assert an.get_existence_array().shape[0] == 7
    an._assert_behavior()


def test_sel_choice_scenarios_partly_shared(n):
    adsg = BasicDSG()
    adsg.add_selection_choice('C1', n[1], [n[11], n[12]])
    adsg.add_selection_choice('C2', n[2], [n[11], n[21]])
    adsg = adsg.set_start_nodes({n[1], n[2]})

    an = _get_hierarchy_analyzer(adsg)
    assert an.n_combinations == 4
    an._assert_behavior()


def test_sel_choice_scenarios_partly_shared_constr(n):
    adsg = BasicDSG()
    adsg.add_selection_choice('C1', n[1], [n[11], n[12]])
    adsg.add_selection_choice('C2', n[2], [n[11], n[21]])
    adsg.add_incompatibility_constraint([n[11], n[21]])
    adsg = adsg.set_start_nodes({n[1], n[2]})

    an = _get_hierarchy_analyzer(adsg)
    assert an.n_combinations == 3
    an._assert_behavior()


def test_sel_choice_scenarios(n):
    adsg = BasicDSG()
    adsg.add_selection_choice('C1', n[1], [n[11], n[12]])
    adsg.add_selection_choice('C2', n[2], [n[21], n[22]])
    adsg.add_selection_choice('C3', n[3], [n[31], n[32]])
    adsg.add_selection_choice('C4', n[4], [n[21], n[22]])
    adsg.add_selection_choice('C5', n[5], [n[41], n[42]])
    adsg.add_edges([
        (n[11], n[2]), (n[12], n[3]),
    ])
    adsg.add_incompatibility_constraint([n[21], n[31]])
    adsg.add_incompatibility_constraint([n[11], n[41]])
    adsg.add_incompatibility_constraint([n[12], n[42]])
    adsg = adsg.set_start_nodes({n[1], n[4], n[5]})

    an = _get_hierarchy_analyzer(adsg)
    scs = an._influence_matrix.base_sel_choice_scenarios
    assert len(scs) == 5
    if isinstance(an, HierarchyAnalyzer):
        assert len(an._reduced_selection_choice_scenarios)
    assert an.n_combinations == 7
    assert np.all(an.selection_choice_is_forced == [False, False, True, False, False])
    assert an.n_opts == [2, 2, 2, 2, 2]
    assert an.n_design_space == 2**5
    assert an.get_choice_option_indices().shape[0] == 7
    assert an.get_existence_array().shape[0] == 7
    an._assert_behavior()


def test_hierarchy_analyzer(n):
    adsg = BasicDSG()
    adsg.add_edges([
        (n[11], n[4]), (n[12], n[3]),
        (n[21], n[5]), (n[22], n[3]),
        (n[5], n[45]),
    ])
    adsg.add_selection_choice('C1', n[1], [n[11], n[12]])
    adsg.add_selection_choice('C2', n[2], [n[21], n[22]])
    adsg.add_selection_choice('C3', n[3], [n[31], n[32]])
    adsg.add_selection_choice('C4', n[4], [n[41], n[42]])
    adsg.add_selection_choice('C5', n[45], [n[46], n[47]])
    adsg = adsg.set_start_nodes({n[1], n[2]})

    an = _get_hierarchy_analyzer(adsg)
    n_comb = 14
    assert an.n_combinations == n_comb

    dec_opt_idx = an.get_choice_option_indices()
    assert dec_opt_idx.shape == (n_comb, len(an.selection_choice_nodes))
    assert np.unique(dec_opt_idx, axis=0).shape == dec_opt_idx.shape

    low_nodes = n[1:6]
    assert len(low_nodes) == 5
    existence = an.get_nodes_existence(low_nodes)
    assert existence.shape == (n_comb, 5)
    assert np.all(existence[:, 0])
    assert np.all(existence[:, 1])
    assert not np.all(existence[:, 2])

    idx_map = an._matrix_diagonal_nodes_idx
    existence_check = an.get_existence_array()[:, [idx_map[node] for node in low_nodes]] == Diag.CONFIRMED.value
    assert np.all(existence_check == existence)

    high_nodes = [n[11], n[12], n[21], n[22], n[31], n[32], n[41], n[42], n[45]]
    existence = an.get_nodes_existence(high_nodes)
    assert existence.shape == (n_comb, 9)

    assert np.all(~an.selection_choice_is_forced)

    graphs = an._assert_behavior()
    assert len(graphs) == dec_opt_idx.shape[0]

    assert an._get_comb_idx([0, 0, 0, 0, 0])[0] == 0
    assert an._get_comb_idx([0, 0, 1, 0, 0])[0] == 0
    assert an._get_comb_idx([0, 0, 0, 0, 1])[0] == 1
    assert an._get_comb_idx([0, 0, 0, 0, 2])[0] == 1
    assert an._get_comb_idx([0, 1, 1, 0, 1])[0] == 6
    assert an._get_comb_idx([1, 1, 1, 0, 0])[0] == 13

    adsg, opt_idx, is_active, i_comb = an.get_graph([0, 0, 0, 0, 0])
    assert isinstance(adsg, BasicDSG)
    assert opt_idx == [0, 0, -1, 0, 0]
    assert is_active == [True, True, False, True, True]
    assert i_comb == 0

    opt_idx_, is_active_, i_comb_ = an.get_opt_idx([0, 0, 0, 0, 0])
    assert opt_idx_ == opt_idx
    assert is_active_ == is_active
    assert i_comb_ == i_comb

    adsg2, _, _, _ = an.get_graph([0, 0, 1, 0, 0])
    assert adsg2 is adsg

    assert adsg.graph.nodes == graphs[0].graph.nodes
    assert adsg.graph.nodes != graphs[1].graph.nodes


def test_dependent_choices(n):
    adsg = BasicDSG()
    adsg.add_selection_choice('C1', n[1], [n[11], n[12]])
    adsg.add_selection_choice('C2', n[2], [n[21], n[22]])
    adsg = adsg.set_start_nodes({n[1], n[2]})

    dec_nodes = adsg.get_ordered_next_choice_nodes()
    assert len(dec_nodes) == 2
    adsg = adsg.constrain_choices(ChoiceConstraintType.LINKED, dec_nodes)
    assert adsg.is_constrained_choice(dec_nodes[0])
    assert adsg.is_constrained_choice(dec_nodes[1])

    an = _get_hierarchy_analyzer(adsg)
    assert an.n_combinations == 2
    assert an.get_choice_option_indices().shape == (2, 2)

    assert np.all(an.selection_choice_is_forced == [False, True])

    graphs = an._assert_behavior()

    assert an._get_comb_idx([0, 0])[0] == 0
    assert an._get_comb_idx([0, 1])[0] == 0
    assert an._get_comb_idx([1, 0])[0] == 1
    assert an._get_comb_idx([1, 1])[0] == 1

    adsg, _, _, _ = an.get_graph([0, 0])
    assert adsg.graph.nodes == graphs[0].graph.nodes
    adsg, opt_idx, _, _ = an.get_graph([1, 0])
    assert opt_idx == [1, 1]
    assert adsg.graph.nodes == graphs[1].graph.nodes


def test_incompatibility(n):
    for both in [False, True]:
        adsg = BasicDSG()
        adsg.add_edges([
            (n[11], n[2]), (n[12], n[2]),
        ])
        adsg.add_selection_choice('C1', n[1], [n[11], n[12]])
        adsg.add_selection_choice('C2', n[2], [n[21], n[22]])
        adsg.add_incompatibility_constraint([n[11], n[21]])
        if both:
            adsg.add_incompatibility_constraint([n[12], n[22]])
        adsg = adsg.set_start_nodes({n[1]})
        test_nodes = [n[11], n[12], n[21], n[22]]

        an = _get_hierarchy_analyzer(adsg)
        assert an.n_combinations == (2 if both else 3)

        existence = an.get_nodes_existence(test_nodes)
        if both:
            assert np.all(existence == np.array([
                [True, False, False, True],
                [False, True, True, False],
            ]))
            assert np.all(an.selection_choice_is_forced == [False, True])

            assert an._get_comb_idx([0, 0])[0] == 0
            assert an._get_comb_idx([0, 1])[0] == 0
            assert an._get_comb_idx([1, 0])[0] == 1
            assert an._get_comb_idx([1, 1])[0] == 1
        else:
            assert np.all(existence == np.array([
                [True, False, False, True],
                [False, True, True, False],
                [False, True, False, True],
            ]))
            assert np.all(~an.selection_choice_is_forced)

            assert an._get_comb_idx([0, 0])[0] == 0
            assert an._get_comb_idx([0, 1])[0] == 0
            assert an._get_comb_idx([1, 0])[0] == 1
            assert an._get_comb_idx([1, 1])[0] == 2

        graphs = an._assert_behavior()
        for i_comb in range(an.n_combinations):
            opt_idx0 = list(an.get_choice_option_indices()[i_comb, :])
            adsg, opt_idx, is_active, i_comb2 = an.get_graph(opt_idx0)
            assert i_comb2 == i_comb
            assert opt_idx == opt_idx0
            assert np.all(np.array(opt_idx)[~np.array(is_active)] == -1)
            assert adsg.graph.nodes == graphs[i_comb].graph.nodes

            opt_idx_, is_active_, i_comb_ = an.get_opt_idx(opt_idx0)
            assert opt_idx_ == opt_idx
            assert is_active_ == is_active
            assert i_comb_ == i_comb


def test_incompatibility_no_opt_left(n):
    adsg = BasicDSG()
    adsg.add_edges([
        (n[21], n[3]),
    ])
    adsg.add_selection_choice('C1', n[1], [n[11], n[12]])
    adsg.add_selection_choice('C2', n[2], [n[21], n[22]])
    adsg.add_selection_choice('C3', n[3], [n[31], n[32]])
    adsg.add_incompatibility_constraint([n[11], n[32]])
    adsg.add_incompatibility_constraint([n[11], n[31]])
    adsg = adsg.set_start_nodes({n[1], n[2]})

    an = _get_hierarchy_analyzer(adsg)
    assert an.n_combinations == 4
    an._assert_behavior()


def test_no_sel_choice(n):
    adsg = BasicDSG()
    adsg.add_edges([
        (n[1], n[11]),
        (n[11], DesignVariableNode('DV', bounds=(0., 1.))),
    ])
    adsg = adsg.set_start_nodes({n[1]})

    an = _get_hierarchy_analyzer(adsg)
    assert len(an.selection_choice_nodes) == 0
    assert an.n_combinations == 1
    assert an.get_choice_option_indices().shape == (1, 0)
    an._assert_behavior()


def test_circular_choices(n):
    adsg = BasicDSG()
    adsg.add_edges([
        (n[1], n[11]), (n[11], n[2]), (n[13], n[3]),
        (n[4], n[21]), (n[21], n[5]), (n[23], n[6]),
        (n[3], n[31]), (n[6], n[31]),
        (n[31], n[2]), (n[31], n[5]),
    ])
    adsg.add_selection_choice('C1', n[2], [n[12], n[13]])
    adsg.add_selection_choice('C2', n[5], [n[22], n[23]])
    adsg = adsg.set_start_nodes({n[1], n[4]})

    an = _get_hierarchy_analyzer(adsg)
    assert len(an.selection_choice_nodes) == 2
    an._assert_behavior()


def test_reused_option_nodes(n):
    for with_incompatibility in [False, True]:
        adsg = BasicDSG()
        ref_nodes = [n[11], n[12]]
        adsg.add_selection_choice('C1', n[1], ref_nodes)
        adsg.add_selection_choice('C2', n[2], ref_nodes)
        adsg.add_selection_choice('C3', n[3], ref_nodes)
        if with_incompatibility:
            adsg.add_incompatibility_constraint(ref_nodes)
        adsg = adsg.set_start_nodes({n[1], n[2], n[3]})

        an = _get_hierarchy_analyzer(adsg)
        assert len(an.selection_choice_nodes) == 3

        if with_incompatibility:
            assert np.all(an.get_choice_option_indices() == np.array([
                [0, 0, 0],
                [1, 1, 1],
            ]))
            assert np.all(an.get_nodes_existence(ref_nodes) == np.array([
                [True, False],
                [False, True],
            ]))
            # assert an.get_unique_node_existence(comp_nodes).shape == (2, 2)
        else:
            assert np.all(an.get_choice_option_indices() == np.array([
                [0, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
                [0, 1, 1],
                [1, 0, 0],
                [1, 0, 1],
                [1, 1, 0],
                [1, 1, 1],
            ]))
            assert np.all(an.get_nodes_existence(ref_nodes) == np.array([
                [True, False],
                [True, True],
                [True, True],
                [True, True],
                [True, True],
                [True, True],
                [True, True],
                [False, True],
            ]))
            # assert an.get_unique_node_existence(comp_nodes).shape == (3, 2)
        an._assert_behavior()


def test_shared_self_activation(n):
    adsg = BasicDSG()
    adsg.add_edges([
        (n[11], n[3]), (n[3], n[31]), (n[31], n[2]),
    ])
    adsg.add_selection_choice('C1', n[1], [n[11], n[12]])
    adsg.add_selection_choice('C2', n[2], [n[11], n[12]])
    adsg = adsg.set_start_nodes({n[1]})

    an = _get_hierarchy_analyzer(adsg)
    assert an.n_combinations == 3
    assert an.get_choice_option_indices().shape[0] == 3
    an._assert_behavior()


def test_intermediate_external(n):
    adsg = BasicDSG()
    adsg.add_edges([
        (n[12], n[2]),
    ])
    adsg.add_selection_choice('C1', n[1], [n[11], n[12]])
    adsg.add_selection_choice('C2', n[2], [n[21], n[22]])
    adsg.add_selection_choice('C3', n[3], [n[31], n[32]])
    adsg = adsg.set_start_nodes({n[1], n[2], n[3]})

    an = _get_hierarchy_analyzer(adsg)
    assert an.n_combinations == 8
    an._assert_behavior()


def test_circular_sel_choice_confirmation(n):
    # graph = AdoreGraph([
    #     ComponentDef('C11', fulfills=['F1'], needs=['F2']),
    #     ComponentDef('C12', fulfills=['F1']),
    #
    #     ComponentDef('C21', fulfills=['F2'], needs=['F3']),
    #     ComponentDef('C22', fulfills=['F2'], needs=['F6']),
    #     ComponentDef('C31', fulfills=['F3'], needs=['F9']),  # IC1 [4]
    #     ComponentDef('C32', fulfills=['F3'], needs=['F4', 'F5']),  # IC2 [5], Multi-fulfillment
    #     ComponentDef('C33', fulfills=['F4'], needs=['F9']),
    #     ComponentDef('C41', fulfills=['F6'], needs=['F10']),  # IC2 [7]
    #     ComponentDef('C42', fulfills=['F6'], needs=['F7', 'F8']),  # IC1 [8], Multi-fulfillment
    #     ComponentDef('C43', fulfills=['F7'], needs=['F10']),
    #     ComponentDef('C44', fulfills=['F5', 'F8'], needs=['F3', 'F6']),  # Switch (PTU)
    #
    #     ComponentDef('C45', fulfills=['F9']),
    #     ComponentDef('C46', fulfills=['F10']),
    # ]).get_for_external_function_names({'F1'})
    adsg = BasicDSG()
    adsg.add_edges([
        (n[11], n[2]),
        (n[21], n[3]), (n[22], n[6]),
        (n[31], n[9]), (n[32], n[4]), (n[32], n[5]),
        (n[4], n[33]), (n[33], n[9]),
        (n[41], n[10]), (n[42], n[7]), (n[42], n[8]),
        (n[7], n[43]), (n[43], n[10]),
        (n[5], n[44]), (n[8], n[44]), (n[44], n[3]), (n[44], n[6]),
        (n[9], n[45]), (n[10], n[46]),
    ])

    adsg.add_selection_choice('C1', n[1], [n[11], n[12]])
    adsg.add_selection_choice('C2', n[2], [n[21], n[22]])
    adsg.add_selection_choice('C3', n[3], [n[31], n[32]])
    adsg.add_selection_choice('C4', n[6], [n[41], n[42]])

    adsg.add_incompatibility_constraint([n[31], n[42]])
    adsg.add_incompatibility_constraint([n[32], n[41]])
    adsg = adsg.set_start_nodes({n[1]})

    an = _get_hierarchy_analyzer(adsg)
    assert an.n_combinations == 5
    an._assert_behavior()
    for opt_idx in an.get_choice_option_indices():
        assert an.get_graph(opt_idx)

    assert np.all(an.selection_choice_is_forced == [False, False, False, True])
    gp = GraphProcessor(adsg)
    assert len(gp.des_vars) == 3
    assert [dv.n_opts for dv in gp.des_vars] == [2, 2, 2]
    seen_dv_imp = set()
    for dv in itertools.product(*[[0, 1] for _ in range(3)]):
        _, dv_imp, _ = gp.get_graph(dv)
        seen_dv_imp.add(tuple(dv_imp))
    assert len(seen_dv_imp) == an.n_combinations


def test_circular_sel_choice_confirmation_multi(n):
    adsg = BasicDSG()
    adsg.add_edges([
        (n[11], n[2]),
        (n[16], n[3]), (n[17], n[6]),
        (n[21], n[3]), (n[22], n[6]),
        (n[31], n[9]), (n[32], n[4]), (n[32], n[5]),
        (n[4], n[33]), (n[33], n[9]),
        (n[41], n[10]), (n[42], n[7]), (n[42], n[8]),
        (n[7], n[43]), (n[43], n[10]),
        (n[5], n[44]), (n[8], n[44]), (n[44], n[3]), (n[44], n[6]),
        (n[9], n[45]), (n[10], n[46]),
    ])

    adsg.add_selection_choice('C1', n[1], [n[11], n[12]])
    adsg.add_selection_choice('C2', n[15], [n[16], n[17]])
    adsg.add_selection_choice('C3', n[2], [n[21], n[22]])
    adsg.add_selection_choice('C4', n[3], [n[31], n[32]])
    adsg.add_selection_choice('C5', n[6], [n[41], n[42]])

    adsg.add_incompatibility_constraint([n[31], n[42]])
    adsg.add_incompatibility_constraint([n[32], n[41]])
    adsg = adsg.set_start_nodes({n[1], n[15]})

    an = _get_hierarchy_analyzer(adsg)
    assert an.n_combinations == 12
    an._assert_behavior()
    for opt_idx in an.get_choice_option_indices():
        assert an.get_graph(opt_idx)


def test_decision_constraint_permutation(n):
    adsg = BasicDSG()
    adsg.add_selection_choice('C1', n[1], [n[11], n[21]])
    adsg.add_selection_choice('C2', n[2], [n[11], n[21]])
    adsg = adsg.set_start_nodes({n[1], n[2]})
    adsg = adsg.constrain_choices(ChoiceConstraintType.PERMUTATION, adsg.choice_nodes)

    an = _get_hierarchy_analyzer(adsg)
    assert an.n_combinations == 2
    an._assert_behavior()


def test_decision_constraint_perm_n_inst(n):
    for n_comp in [2, 3, 4]:
        adsg = BasicDSG()
        for i in range(n_comp):
            i_node = 10*(i+1)
            adsg.add_edge(n[i], n[i_node])
            adsg.add_selection_choice(f'C{i}', n[i_node], n[i_node+1:i_node+4])

        adsg = adsg.set_start_nodes({n[i] for i in range(n_comp)})
        adsg = adsg.constrain_choices(ChoiceConstraintType.PERMUTATION, adsg.choice_nodes)

        assert adsg.feasible == (n_comp < 4)
        if n_comp < 4:
            an = _get_hierarchy_analyzer(adsg)
            assert an.n_combinations == len(list(itertools.permutations('ABC', n_comp)))
            an._assert_behavior()


def test_decision_constraint_unordered(n):
    for n_comp in [2, 3, 4]:
        adsg = BasicDSG()
        for i in range(n_comp):
            i_node = 10*(i+1)
            adsg.add_edge(n[i], n[i_node])
            adsg.add_selection_choice(f'C{i}', n[i_node], n[i_node+1:i_node+4])

        adsg = adsg.set_start_nodes({n[i] for i in range(n_comp)})
        adsg = adsg.constrain_choices(ChoiceConstraintType.UNORDERED, adsg.choice_nodes)

        an = _get_hierarchy_analyzer(adsg)
        assert an.n_combinations == len(list(itertools.combinations_with_replacement('ABC', n_comp)))
        an._assert_behavior()


def test_decision_constraint_unordered_non_replacing(n):
    for n_comp in [2, 3, 4]:
        adsg = BasicDSG()
        for i in range(n_comp):
            i_node = 10*(i+1)
            adsg.add_edge(n[i], n[i_node])
            adsg.add_selection_choice(f'C{i}', n[i_node], n[i_node+1:i_node+4])

        adsg = adsg.set_start_nodes({n[i] for i in range(n_comp)})
        adsg = adsg.constrain_choices(ChoiceConstraintType.UNORDERED_NOREPL, adsg.choice_nodes)

        assert adsg.feasible == (n_comp < 4)
        if n_comp < 4:
            an = _get_hierarchy_analyzer(adsg)
            assert an.n_combinations == len(list(itertools.combinations('ABC', n_comp)))
            an._assert_behavior()


def test_shared_constrained_choice(n):
    for type_ in ChoiceConstraintType:
        adsg = BasicDSG()
        adsg.add_selection_choice('C1', n[1], [n[11], n[12]])
        adsg.add_selection_choice('C2', n[2], [n[11], n[12]])
        adsg = adsg.set_start_nodes({n[1], n[2]})
        adsg = adsg.constrain_choices(type_, adsg.choice_nodes)

        an = _get_hierarchy_analyzer(adsg)
        if type_ in [ChoiceConstraintType.LINKED, ChoiceConstraintType.PERMUTATION]:
            assert an.n_combinations == 2
        elif type_ == ChoiceConstraintType.UNORDERED:
            assert an.n_combinations == 3
        elif type_ == ChoiceConstraintType.UNORDERED_NOREPL:
            assert an.n_combinations == 1
        an._assert_behavior()


def test_conditionally_active_constrained(n):
    for type_ in ChoiceConstraintType:
        adsg = BasicDSG()
        adsg.add_edge(n[1], n[11])
        adsg.add_selection_choice('C1', n[11], n[12:15])
        adsg.add_edges([
            (n[13], n[12]), (n[14], n[13]),
        ])
        c2 = adsg.add_selection_choice('C2', n[12], n[22:25])
        c3 = adsg.add_selection_choice('C3', n[13], n[32:35])
        c4 = adsg.add_selection_choice('C4', n[14], n[42:45])

        adsg = adsg.set_start_nodes({n[1]})
        adsg = adsg.constrain_choices(type_, [c2, c3, c4])

        an = _get_hierarchy_analyzer(adsg)
        if type_ == ChoiceConstraintType.LINKED:
            assert an.n_combinations == 9
        elif type_ == ChoiceConstraintType.PERMUTATION:
            assert an.n_combinations == 15
        elif type_ == ChoiceConstraintType.UNORDERED:
            assert an.n_combinations == 19
        elif type_ == ChoiceConstraintType.UNORDERED_NOREPL:
            assert an.n_combinations == 7
        an._assert_behavior()


def test_conditionally_active_constrained_fast(n):
    for type_ in ChoiceConstraintType:
        adsg = BasicDSG()
        adsg.add_edge(n[1], n[11])
        adsg.add_selection_choice('C1', n[11], n[12:15])
        adsg.add_edges([
            (n[13], n[12]), (n[14], n[13]),
        ])
        c2 = adsg.add_selection_choice('C2', n[12], n[22:25])
        c3 = adsg.add_selection_choice('C3', n[13], n[32:35])
        c4 = adsg.add_selection_choice('C4', n[14], n[42:45])

        adsg = adsg.set_start_nodes({n[1]})
        adsg = adsg.constrain_choices(type_, [c2, c3, c4])

        # Note: these are upper bounds!
        an = FastHierarchyAnalyzer(adsg)
        if type_ == ChoiceConstraintType.LINKED:
            assert an.n_combinations == 9
            assert np.all(an.selection_choice_is_forced == [False, False, True, True])
        elif type_ == ChoiceConstraintType.PERMUTATION:
            assert an.n_combinations == 18
            assert not np.any(an.selection_choice_is_forced)
        elif type_ == ChoiceConstraintType.UNORDERED:
            assert an.n_combinations == 30
            assert not np.any(an.selection_choice_is_forced)
        elif type_ == ChoiceConstraintType.UNORDERED_NOREPL:
            assert an.n_combinations == 30
            assert not np.any(an.selection_choice_is_forced)
        an._assert_behavior()


def test_forced_inactive_imputation(n):
    adsg = BasicDSG()
    adsg.add_edges([
        (n[11], n[2]), (n[12], n[3]),
        (n[31], n[2]),
    ])
    adsg.add_selection_choice('C1', n[1], [n[11], n[12]])
    adsg.add_selection_choice('C2', n[2], [n[21], n[22]])
    adsg.add_selection_choice('C3', n[3], [n[31], n[32]])
    adsg = adsg.set_start_nodes({n[1]})

    an = _get_hierarchy_analyzer(adsg)
    assert an.n_combinations == 5
    assert np.all(~an.selection_choice_is_forced)


def test_multi_dep(n):
    adsg = BasicDSG()
    adsg.add_edges([
        (n[13], n[4]), (n[4], n[11]),
        (n[13], n[5]), (n[5], n[12]),
        (n[3], n[31]),
    ])
    adsg.add_selection_choice('C1', n[1], n[11:14])
    adsg.add_selection_choice('C2', n[2], [n[21], n[22]])
    adsg.add_selection_choice('C3', n[11], n[14:17])
    adsg.add_selection_choice('C4', n[12], n[17:20])
    adsg.add_selection_choice('C5', n[31], n[32:35])
    adsg = adsg.set_start_nodes({n[1], n[2], n[3]})

    an = _get_hierarchy_analyzer(adsg)
    assert an.n_combinations == 90
    an._assert_behavior()
    an._assert_existence()


def test_dependent_activation_or_opt_sel(n):
    adsg = BasicDSG()
    adsg.add_selection_choice('C1', n[1], [n[2], n[3]])
    adsg.add_selection_choice('C2', n[2], [n[4], n[5]])
    adsg.add_edge(n[3], n[4])
    adsg = adsg.set_start_nodes({n[1]})

    an = _get_hierarchy_analyzer(adsg)
    assert an.n_combinations == 3
    an._assert_behavior()
    an._assert_existence()
