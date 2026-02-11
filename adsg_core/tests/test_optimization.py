import os
import math
import timeit
import random
import pickle
import pytest
import itertools
import numpy as np
from typing import *
from adsg_core.examples.gnc import *
from adsg_core.examples.apollo import *
from adsg_core.graph.graph_edges import *
from adsg_core.graph.adsg_basic import *
from adsg_core.graph.adsg_nodes import *
from adsg_core.optimization.evaluator import *
from adsg_core.optimization.dv_output_defs import *
from adsg_core.optimization.graph_processor import *


def _get_base_dsg(n):
    dsg = BasicDSG()

    cn1 = ConnectorNode('CN1', deg_spec='1..2')
    dsg.add_edges([
        (n[1], n[11]), (n[11], n[2]),
        (n[11], MetricNode('M1', direction=-1)), (n[11], MetricNode('M3', direction=-1, ref=0)),
        (n[11], cn1),
    ])

    dsg.add_selection_choice('C1', n[2], [n[21], n[31]], is_ordinal=True)
    cn2 = [ConnectorNode(f'CN2-{i}', deg_list=[0, 1]) for i in range(3)]
    dsg.add_edges([
        (cn2[1], cn2[0]), (cn2[2], cn2[1]),
    ])
    dsg.add_selection_choice('C2', n[21], cn2)

    cn3 = ConnectorNode('CN3', deg_list=[0, 1])
    dsg.add_edges([
        (n[31], n[3]), (n[31], cn3),
        (n[31], MetricNode('M2', direction=-1, ref=0)), (n[31], MetricNode('M4')),
    ])

    cn4 = ConnectorNode('CN4', deg_list=[0, 1])
    dsg.add_edges([
        (n[3], n[41]), (n[41], cn4),
        (n[41], DesignVariableNode('DV', bounds=(0., 10.))),
    ])

    dsg.add_connection_choice('C3', [cn1], cn2+[cn3, cn4])

    return dsg


@pytest.fixture
def base_adsg(n):
    return _get_base_dsg(n)


@pytest.fixture
def adsg_init(base_adsg: BasicDSG, n):
    return base_adsg.copy().set_start_nodes({n[1]})


def test_des_var():
    with pytest.raises(ValueError):
        DesVar('DV')

    with pytest.raises(ValueError):
        DesVar('DV', options=[])

    with pytest.raises(ValueError):
        DesVar('DV', bounds=(1.,))

    dv = DesVar('DV', options=[1, 2, 3])
    assert dv.name == 'DV'
    assert dv.is_discrete
    assert dv.options == [1, 2, 3]
    assert dv.n_opts == 3
    assert dv.bounds is None
    assert dv.node is None
    assert str(dv)
    assert repr(dv)

    assert GraphProcessor._get_inactive_value(dv) == 0

    for _ in range(1000):
        val = dv.rand()
        assert 0 <= val <= 2

    with pytest.raises(ValueError):
        DesVar('DV', bounds=(2., 1.))

    dv = DesVar('DV', bounds=(1., 2.))
    assert dv.name == 'DV'
    assert not dv.is_discrete
    assert dv.options is None
    assert dv.n_opts is None
    assert dv.bounds == (1., 2.)
    assert dv.node is None
    assert str(dv)
    assert repr(dv)

    assert GraphProcessor._get_inactive_value(dv) == 1.5

    for _ in range(1000):
        val = dv.rand()
        assert 1. <= val <= 2.

    des_var_node = DesignVariableNode('DV', bounds=(1., 2.))
    choice_node = ChoiceNode(decision_id='choice')

    dv = DesVar.from_des_var_node(des_var_node)
    assert dv.name == des_var_node.name
    assert not dv.is_discrete
    assert dv.bounds == des_var_node.bounds
    assert dv.node is des_var_node

    dv = DesVar.from_choice_node(choice_node, [1, 2, 3])
    assert dv.name == choice_node.decision_id
    assert dv.is_discrete
    assert dv.options == [1, 2, 3]
    assert dv.node is choice_node

    dv2 = DesVar.from_choice_node(choice_node, [1, 2, 3], existing_names={choice_node.decision_id})
    assert dv2.name == f'{choice_node.decision_id}_2'


def test_objective():
    obj = Objective('OBJ')
    assert obj.name == 'OBJ'
    assert obj.dir == Direction.MIN
    assert obj.sign == -1
    assert obj.node is None
    assert str(obj)
    assert repr(obj)

    metric_node = MetricNode('PR', direction=1, ref=5.)
    obj = Objective.from_metric_node(metric_node)
    assert obj.name == metric_node.name
    assert obj.dir == Direction.MAX
    assert obj.sign == 1
    assert obj.node is metric_node

    metric_node.dir = None
    with pytest.raises(ValueError):
        Objective.from_metric_node(metric_node)


def test_constraint():
    constr = Constraint('G')
    assert constr.name == 'G'
    assert constr.ref == 0.
    assert constr.dir == Direction.LTE
    assert constr.sign == -1
    assert str(constr)
    assert repr(constr)

    metric_node = MetricNode('PR', direction=1, ref=5.)
    constr = Constraint.from_metric_node(metric_node)
    assert constr.name == metric_node.name
    assert constr.ref == 5.
    assert constr.dir == Direction.GTE
    assert constr.sign == 1
    assert constr.node is metric_node

    metric_node.dir = None
    with pytest.raises(ValueError):
        Constraint.from_metric_node(metric_node)

    metric_node.dir = 1
    metric_node.ref = None
    with pytest.raises(ValueError):
        Constraint.from_metric_node(metric_node)


def test_metric_type():
    assert not MetricType.NONE
    assert MetricType.OBJECTIVE

    assert (MetricType.NONE | MetricType.NONE) == MetricType.NONE
    assert (MetricType.NONE | MetricType.OBJECTIVE) == MetricType.OBJECTIVE
    assert (MetricType.NONE | MetricType.CONSTRAINT) == MetricType.CONSTRAINT
    assert (MetricType.OBJECTIVE | MetricType.CONSTRAINT) == MetricType.OBJ_OR_CON

    assert MetricType.OBJECTIVE & MetricType.OBJECTIVE
    assert MetricType.OBJ_OR_CON & MetricType.OBJECTIVE
    assert not (MetricType.NONE & MetricType.OBJECTIVE)
    assert not (MetricType.CONSTRAINT & MetricType.OBJECTIVE)


def test_graph_processor_check_graph(base_adsg, adsg_init, n):
    with pytest.raises(ValueError):
        GraphProcessor._check_graph(base_adsg)

    adsg_no_choice = BasicDSG()
    adsg_no_choice.add_edges([(n[1], n[2]), (n[2], n[3])])
    adsg_no_choice = adsg_no_choice.set_start_nodes({n[1]})

    GraphProcessor._check_graph(adsg_no_choice)
    processor = GraphProcessor(adsg_no_choice)
    assert processor.encoder_type == SelChoiceEncoderType.COMPLETE
    assert len(processor.des_vars) == 0
    assert processor.dv_is_conditionally_active == []

    graph = GraphProcessor._check_graph(adsg_init)
    assert len(graph.choice_nodes) == 3

    processor = GraphProcessor(adsg_init)
    assert len(processor.graph.choice_nodes) == 3


def assert_processor_get_all(processor: GraphProcessor, get_all=True):
    assert processor.encoder_type == SelChoiceEncoderType.COMPLETE

    is_act_map = {}
    is_active_all = []
    for dv in itertools.product(*[list(range(dv.n_opts)) if dv.is_discrete else [dv.bounds[0]]
                                  for dv in processor.des_vars]):
        graph, dv_imp_no_create, is_act_no_create = processor.get_graph(dv, create=False)
        assert graph is None

        graph, dv_imp, is_active = processor.get_graph(dv)
        assert graph.final
        assert graph.feasible
        assert dv_imp == dv_imp_no_create
        assert is_act_no_create == is_active
        is_act_map[tuple(np.array(dv_imp).astype(float))] = np.array(is_active)
        is_active_all.append(is_active)

    # Check that there are no conditionally active variables that are reported as permanent
    conditionally_active = np.array(processor.dv_is_conditionally_active)
    if len(conditionally_active) > 0:
        assert not np.all(conditionally_active)
        is_cond_active = np.any(~np.row_stack(is_active_all), axis=0)
        assert not np.any(is_cond_active & ~conditionally_active)

    if get_all:
        x_all, is_active_all = processor.get_all_discrete_x()
        n_all = processor.get_n_valid_designs(with_fixed=True)
        assert x_all.shape == (n_all, len(processor.des_vars))
        assert is_active_all.shape == x_all.shape
        assert len(is_act_map) == n_all

        for i_dv, dv in enumerate(processor.des_vars):
            if not dv.is_discrete:
                x_all[is_active_all[:, i_dv], i_dv] = dv.bounds[0]
        for i, x in enumerate(x_all):
            assert np.all(is_act_map.get(tuple(x)) == is_active_all[i, :])

        x_all_no_fixed, is_act_all_no_fixed = processor.get_all_discrete_x(with_fixed=False)
        n_all_no_fixed = processor.get_n_valid_designs(with_fixed=False)
        assert x_all_no_fixed.shape == (n_all_no_fixed, len(processor.all_des_vars))
        assert x_all_no_fixed.shape == is_act_all_no_fixed.shape

    processor2: GraphProcessor = pickle.loads(pickle.dumps(processor))
    assert processor2.encoder_type == processor.encoder_type
    assert processor2._hierarchy_analyzer
    if get_all:
        assert processor.get_all_discrete_x()


def test_graph_processor_conditional_connection(n):
    adsg = BasicDSG()
    cn = [ConnectorNode(f'CN{i}', deg_spec='+') for i in range(4)]
    adsg.add_edges([
        (n[1], n[11]), (n[13], n[12]),
        (n[2], n[21]), (n[23], n[22]),
        (n[12], cn[0]), (n[13], cn[1]),
        (n[22], cn[2]), (n[23], cn[3]),
    ])
    adsg.add_selection_choice('C1', n[11], [n[12], n[13]])
    adsg.add_selection_choice('C2', n[21], [n[22], n[23]])
    adsg.add_connection_choice('C3', cn[:2], cn[2:])
    adsg = adsg.set_start_nodes({n[1], n[2]})

    processor = GraphProcessor(adsg)
    assert_processor_get_all(processor)


def test_graph_processor_conditional_connection2(n):
    adsg = BasicDSG()
    cn = [ConnectorNode(f'CN{i}', deg_spec='+') for i in range(4)]
    adsg.add_edges([
        (n[13], n[12]),
        (n[2], n[21]), (n[23], n[22]),
        (n[12], cn[0]), (n[13], cn[1]),
        (n[22], cn[2]), (n[23], cn[3]),
    ])
    adsg.add_selection_choice('C1', n[1], [n[11], n[15]])
    adsg.add_selection_choice('C2', n[11], [n[12], n[13]])
    adsg.add_selection_choice('C3', n[21], [n[22], n[23]])
    adsg.add_connection_choice('C4', cn[:2], cn[2:])
    adsg = adsg.set_start_nodes({n[1], n[2]})

    processor = GraphProcessor(adsg)
    assert_processor_get_all(processor)


def test_graph_processor_des_vars(adsg_init):
    adsg = adsg_init
    assert len(adsg.choice_nodes) == 3
    assert len(adsg.des_var_nodes) == 1

    processor = GraphProcessor(adsg)
    assert processor.encoder_type == SelChoiceEncoderType.COMPLETE
    processor._hierarchy_analyzer._assert_behavior()
    assert set(processor._choice_nodes) == set(adsg.choice_nodes)
    assert set(processor.design_variable_nodes) == set(adsg.des_var_nodes)
    des_vars, _, _, _ = processor._get_des_vars()
    assert len(des_vars) == 6

    # Option decision des vars
    assert des_vars[0].is_discrete
    assert des_vars[0].is_ordinal
    assert des_vars[0].node is processor._choice_nodes[0]
    assert des_vars[0].n_opts == 2
    assert des_vars[1].is_discrete
    assert not des_vars[1].is_ordinal
    assert des_vars[1].node is processor._choice_nodes[1]
    assert des_vars[1].n_opts == 3

    assert des_vars[2].is_discrete
    assert not des_vars[2].is_ordinal
    assert des_vars[2].node is processor._choice_nodes[2]
    assert des_vars[2].n_opts == 2
    assert des_vars[3].is_discrete
    assert not des_vars[3].is_ordinal
    assert des_vars[3].node is processor._choice_nodes[2]
    assert des_vars[3].n_opts == 2
    assert des_vars[4].is_discrete
    assert not des_vars[4].is_ordinal
    assert des_vars[4].node is processor._choice_nodes[2]
    assert des_vars[4].n_opts == 2

    assert not des_vars[5].is_discrete
    assert des_vars[5].is_ordinal
    assert des_vars[5].node is processor.design_variable_nodes[0]
    assert tuple(des_vars[5].bounds) == (0., 10.)

    assert len(processor.des_vars) == 6


def test_graph_processor_cont_des_vars(n):
    adsg = BasicDSG()
    adsg.add_edges([
        (n[1], n[11]), (n[14], n[13]), (n[13], n[12]),
    ])
    adsg.add_selection_choice('C1', n[11], n[12:15])
    for i, n_ in enumerate(n[12:15]):
        adsg.add_edge(n_, DesignVariableNode(f'DV{i}', bounds=(.5, 1.5)))
    adsg = adsg.set_start_nodes({n[1]})

    processor = GraphProcessor(adsg)
    assert processor.encoder_type == SelChoiceEncoderType.COMPLETE
    des_vars = processor.des_vars
    assert len(des_vars) == 4
    assert len(set([des_var.name for des_var in des_vars])) == 4
    assert all([not des_var.is_discrete for des_var in des_vars[1:]])


def test_graph_processor_int_des_vars(n):
    adsg = BasicDSG()
    adsg.add_edges([
        (n[1], n[11]), (n[14], n[13]), (n[13], n[12]),
    ])
    adsg.add_selection_choice('C1', n[11], n[12:15], is_ordinal=True)
    for i, n_ in enumerate(n[12:15]):
        adsg.add_edge(n_, DesignVariableNode(f'DV{i}', options=[1, 2, 3], is_ordinal=True))
    adsg = adsg.set_start_nodes({n[1]})

    processor = GraphProcessor(adsg)
    assert processor.encoder_type == SelChoiceEncoderType.COMPLETE
    des_vars = processor.des_vars
    assert len(des_vars) == 4
    assert len(set([des_var.name for des_var in des_vars])) == 4
    assert all([des_var.is_discrete for des_var in des_vars[1:]])
    assert all([des_var.is_ordinal for des_var in des_vars[1:]])


def test_graph_processor_repeated_des_vars(n):
    adsg = BasicDSG()
    adsg.add_edges([
        (n[1], n[11]), (n[11], n[2]), (n[11], n[3]),
    ])
    adsg.add_selection_choice('C1', n[2], [n[21], n[22]])
    adsg.add_selection_choice('C2', n[3], [n[31], n[32]])
    adsg = adsg.set_start_nodes({n[1]})

    processor = GraphProcessor(adsg)
    assert processor.encoder_type == SelChoiceEncoderType.COMPLETE
    des_vars = processor.des_vars
    assert len(des_vars) == 2
    assert len(set([des_var.name for des_var in des_vars])) == 2


@pytest.mark.skipif(int(os.getenv('RUN_SLOW_TESTS', 0)) != 1, reason='Set RUN_SLOW_TESTS=1 to run slow tests')
def test_graph_processor_fixed_des_vars(n):
    for encoder_type in SelChoiceEncoderType:
        is_fast = encoder_type == SelChoiceEncoderType.FAST

        for _ in range(2):
            adsg = _get_base_dsg(n).set_start_nodes({n[1]})
            processor = GraphProcessor(adsg, encoder_type=encoder_type)

            n_dv = 8 if is_fast else 6
            assert len(processor.des_vars) == n_dv
            assert len(processor.all_des_vars) == n_dv
            assert len(processor.fixed_values) == 0
            assert processor.encoder_type == encoder_type

            dvs = [0, 0, 1, 0, 0, 0, 0, 5.] if is_fast else [0, 0, 1, 0, 0, 5.]
            _, used_dvs_nc, is_active_nc = processor.get_graph(dvs, create=False)
            graph1, used_dvs, is_active = processor.get_graph(dvs)
            assert dvs == used_dvs
            assert len(is_active) == len(used_dvs)
            assert used_dvs == used_dvs_nc
            assert is_active == is_active_nc

            if not is_fast:
                assert_processor_get_all(processor)

            n_valid = {processor.get_n_valid_designs(with_fixed=True)}
            for i in range(n_dv):
                processor._hierarchy_analyzer._feasibility_mask = None

                if 2 <= i < n_dv-1:
                    with pytest.raises(RuntimeError):
                        processor.fix_des_var(processor.all_des_vars[i], dvs[i])
                    continue
                processor.fix_des_var(processor.all_des_vars[i], dvs[i])
                assert len(processor.des_vars) == n_dv-1
                assert len(processor.all_des_vars) == n_dv
                assert len(processor.fixed_values) == 1
                if i < len(processor._sel_choice_idx_map):
                    if not is_fast:
                        assert processor._comb_fixed_mask is not None
                assert processor.is_fixed(processor.all_des_vars[i])
                processor.print_stats()

                dvs_mod = [val for ii, val in enumerate(dvs) if ii != i]
                graph2, used_dvs2, is_active2 = processor.get_graph(dvs_mod)
                assert graph2 == graph1
                assert used_dvs2 == dvs_mod
                assert len(is_active2) == len(used_dvs2)

                if not is_fast:
                    assert_processor_get_all(processor)

                n_valid.add(processor.get_n_valid_designs(with_fixed=True))

                processor.fix_des_var(processor.all_des_vars[i], None)
                assert len(processor.des_vars) == n_dv
                assert len(processor.all_des_vars) == n_dv
                assert len(processor.fixed_values) == 0
                assert not processor.is_fixed(processor.all_des_vars[i])

                with pytest.raises(ValueError):
                    processor.get_graph(dvs_mod)
                with pytest.raises(ValueError):
                    processor.get_graph(dvs_mod, create=False)
                graph3, used_dvs3, _ = processor.get_graph(dvs)
                assert used_dvs3 == dvs
                assert graph3 == graph1

            assert len(n_valid) == (1 if is_fast else 3)


def test_graph_processor_existence_single(n):
    adsg = BasicDSG()
    cn1 = ConnectorNode('CN1', deg_list=[1])
    cn2 = ConnectorNode('CN2', deg_list=[0, 1])
    adsg.add_edges([
        (n[1], n[11]), (n[11], n[2]), (n[2], n[21]),
        (n[21], cn2),
    ])
    adsg.add_selection_choice('C1', n[11], [n[12], cn1])
    adsg.add_connection_choice('C2', [cn1], [cn2])
    adsg = adsg.set_start_nodes({n[1]})

    processor = GraphProcessor(adsg)
    existence = processor._hierarchy_analyzer.get_nodes_existence([cn1])
    assert existence.shape[0] == 2

    options = adsg.get_option_nodes(adsg.choice_nodes[0])
    assert len(options) == 2
    for i, option in enumerate(options):
        if option == cn1:
            assert existence[i, 0]
        else:
            assert not existence[i, 0]

    existence = processor._hierarchy_analyzer.get_nodes_existence([n[11]])
    assert np.all(existence[:, 0])


def test_graph_processor_existence_double(n):
    adsg = BasicDSG()
    cn1 = ConnectorNode('CN1', deg_list=[1])
    cn2 = ConnectorNode('CN2', deg_list=[0, 1])
    adsg.add_edges([
        (n[1], n[11]), (n[11], n[2]),
        (n[11], cn1),
    ])
    adsg.add_selection_choice('C1', n[2], [n[21], n[31]])
    adsg.add_selection_choice('C2', n[21], [n[22], cn2])
    adsg.add_connection_choice('C3', [cn1], [cn2])
    adsg = adsg.set_start_nodes({n[1]})

    assert len(adsg.choice_nodes) == 3
    processor = GraphProcessor(adsg)
    assert len(processor.selection_choice_nodes) == 2

    assert processor._hierarchy_analyzer.get_nodes_existence([cn2]).shape[0] == 3


def test_graph_processor_existence_multi(n):
    adsg = BasicDSG()
    cn1 = ConnectorNode('CN1', deg_list=[1])
    cn2 = ConnectorNode('CN2', deg_list=[0, 1])
    adsg.add_edges([
        (n[1], n[11]), (n[11], n[2]),
        (n[21], n[5]), (n[31], n[3]),
        (n[5], n[42]), (n[42], cn2),
    ])
    adsg.add_selection_choice('C1', n[2], [n[21], n[31]])
    adsg.add_selection_choice('C2', n[3], [n[41], n[42]])
    adsg.add_selection_choice('C3', n[41], [n[45], cn1])
    adsg.add_connection_choice('C4', [cn1], [cn2])
    adsg = adsg.set_start_nodes({n[1]})

    assert len(adsg.choice_nodes) == 4

    gp = GraphProcessor(adsg)
    provided_port_existence = gp._hierarchy_analyzer.get_nodes_existence([cn2])
    assert provided_port_existence.shape == (4, 1)
    assert np.sum(provided_port_existence[:, 0]) == 2

    needed_port_existence = gp._hierarchy_analyzer.get_nodes_existence([cn1])
    assert needed_port_existence.shape == (4, 1)
    assert np.sum(needed_port_existence[:, 0]) == 1


def test_graph_processor_derived_perm_des_vars(n):
    adsg = BasicDSG()
    cn1 = ConnectorNode('CN1', deg_list=[0, 1])
    cn2 = ConnectorNode('CN2', deg_list=[1])
    adsg.add_edges([
        (n[1], n[11]), (n[11], n[2]),
        (n[11], cn1), (n[21], cn2),
    ])
    adsg.add_selection_choice('C1', n[2], [n[21], n[31]])
    adsg.add_connection_choice('C2', [cn2], [cn1])
    adsg = adsg.set_start_nodes({n[1]})

    processor = GraphProcessor(adsg)
    assert len(processor.des_vars) == 1
    assert len(processor._conn_choice_data_map) == 1

    for _ in range(10):
        des_vars = [dv.rand() for dv in processor.des_vars]

        graph_nc, used_values_, is_active_ = processor.get_graph(des_vars, create=False)
        assert graph_nc is None

        graph, used_values, is_active = processor.get_graph(des_vars)
        assert graph.final
        assert graph.feasible
        assert used_values == des_vars
        assert len(is_active) == len(des_vars)
        assert all(is_active)
        assert used_values_ == used_values
        assert is_active_ == is_active


def test_graph_processor_get_metrics(adsg_init):
    for encoder_type in SelChoiceEncoderType:
        processor = GraphProcessor(adsg_init, encoder_type=encoder_type)
        assert processor.encoder_type == encoder_type

        permanent_nodes = processor.permanent_nodes
        assert len(permanent_nodes) == 6
        metric_nodes = processor.metric_nodes
        assert len(metric_nodes) == 4
        assert metric_nodes[0] in permanent_nodes
        assert metric_nodes[1] not in permanent_nodes
        assert metric_nodes[2] in permanent_nodes
        assert metric_nodes[3] not in permanent_nodes

        metrics = processor._get_metrics()
        assert metrics == [
            (metric_nodes[0], MetricType.OBJECTIVE),
            (metric_nodes[1], MetricType.CONSTRAINT),
            (metric_nodes[2], MetricType.OBJ_OR_CON),
            (metric_nodes[3], MetricType.NONE),
        ]

        with pytest.raises(RuntimeError):
            processor._categorize_metrics()

        processor._choose_metric_type = lambda o, c: o
        objs, constr = processor._categorize_metrics()
        assert len(objs) == 2
        assert [obj.node for obj in objs] == [metric_nodes[0], metric_nodes[2]]
        assert len(constr) == 1
        assert [con.node for con in constr] == [metric_nodes[1]]

        objs, constr = processor._categorized_metrics
        assert len(objs) == 2
        assert len(constr) == 1
        assert objs == processor.objectives
        assert constr == processor.constraints


def test_graph_processor_preferred_metric_type(adsg_init):
    processor = GraphProcessor(adsg_init)

    with pytest.raises(RuntimeError):
        processor._categorize_metrics()

    for metric_node in processor.metric_nodes:
        metric_node.type = MetricType.OBJECTIVE

    del processor.__dict__['_metrics']
    objs, constr = processor._categorize_metrics()
    assert len(objs) == 2
    assert len(constr) == 1


def test_graph_processor_get_graph(adsg_init):
    processor = GraphProcessor(adsg_init)
    assert processor.encoder_type == SelChoiceEncoderType.COMPLETE
    des_vars = processor.des_vars
    graph_cache = {}
    for i in range(6):
        for values in itertools.product(*[range(dv.n_opts) for dv in des_vars if dv.is_discrete]):
            values = list(values)+[sum(dv.bounds)*.6 for dv in des_vars if not dv.is_discrete]
            assert len(values) == len(des_vars)

            graph, used_values, is_active = processor.get_graph(values)
            assert isinstance(graph, BasicDSG)
            assert graph.final
            assert graph.feasible
            assert len(is_active) == len(used_values)

            dv_nodes = graph.des_var_nodes
            if len(dv_nodes) == 0:
                assert used_values[-1] == GraphProcessor._get_inactive_value(des_vars[-1])
                assert not is_active[-1]
            else:
                assert used_values[-1] == sum(des_vars[-1].bounds)*.6
                assert is_active[-1]

            if i == 0:
                graph_cache[tuple(values)] = graph
            else:
                cached_graph = graph_cache[tuple(values)]
                assert cached_graph.graph.nodes == graph.graph.nodes
                assert cached_graph.graph.edges == graph.graph.edges


def test_graph_processor_optional_connection(n):
    adsg = BasicDSG()
    cn1 = ConnectorNode('CN1', deg_list=[1])
    cn2 = ConnectorNode('CN2', deg_list=[0, 1])
    adsg.add_edges([
        (n[1], n[11]), (n[11], cn1),
        (n[11], n[2]), (n[2], n[21]),
    ])
    adsg.add_selection_choice('C1', n[21], [n[22], cn2])
    adsg.add_connection_choice('C2', [cn1], [cn2])
    adsg = adsg.set_start_nodes({n[1]})

    assert_processor_get_all(GraphProcessor(adsg))


def test_graph_processor_multi_level_optional_connection(n):
    adsg = BasicDSG()
    cn1 = ConnectorNode('CN1', deg_list=[1])
    cn2 = ConnectorNode('CN2', deg_list=[0, 1])
    adsg.add_edges([
        (n[1], n[11]), (n[11], n[2]), (n[2], n[21]),
    ])
    adsg.add_selection_choice('C1', n[11], [n[12], cn1])
    adsg.add_selection_choice('C2', n[21], [n[22], cn2])
    adsg.add_connection_choice('C3', [cn1], [cn2])
    adsg = adsg.set_start_nodes({n[1]})

    assert_processor_get_all(GraphProcessor(adsg))


class DummyADSGEvaluator(DSGEvaluator):

    def _evaluate(self, dsg: DSGType, metric_nodes: List[MetricNode]) -> Dict[MetricNode, float]:
        return {mn: random.random() for mn in metric_nodes}


def test_graph_evaluator(adsg_init):
    with pytest.raises(RuntimeError):
        assert DummyADSGEvaluator(adsg_init).objectives

    metric_nodes = sorted(adsg_init.get_nodes_by_type(MetricNode), key=lambda n: n.name)
    metric_nodes[2].type = MetricType.OBJECTIVE

    evaluator = DummyADSGEvaluator(adsg_init)
    assert len(evaluator.objectives) == 2
    assert len(evaluator.constraints) == 1

    for _ in range(10):
        des_var_values = [dv.rand() for dv in evaluator.des_vars]
        graph, _, _ = evaluator.get_graph(des_var_values)

        # assert len(graph.metric_values) == 0
        obj, con = evaluator.evaluate(graph)
        assert len(obj) == 2
        assert len(con) == 1

        assert len(graph.metric_values) > 0

    assert_processor_get_all(evaluator)


def test_fast_encoder(n):
    adsg = BasicDSG()
    cn1 = [ConnectorNode(f'CN{i}', deg_list=[1, 2], repeated_allowed=True) for i in range(2)]
    cn2 = ConnectorNode('CN3', deg_spec='*', repeated_allowed=True)
    adsg.add_edges([
        (n[13], n[12]), (n[23], n[22]),
        (n[12], cn1[0]), (n[13], cn1[1]),
        (n[2], n[31]), (n[31], cn2),
    ])
    adsg.add_selection_choice('C1', n[1], [n[11], n[21]])
    adsg.add_selection_choice('C2', n[11], [n[12], n[13]])
    adsg.add_selection_choice('C3', n[21], [n[22], n[23]])
    adsg.add_connection_choice('C4', cn1, [cn2])
    adsg = adsg.set_start_nodes({n[1], n[2]})

    cgp = GraphProcessor(adsg, encoder_type=SelChoiceEncoderType.COMPLETE)
    assert cgp.get_n_valid_designs() == 8
    assert cgp.get_n_design_space() == 32
    x_all, is_act_all = cgp.get_all_discrete_x()
    assert x_all.shape == (8, 5)

    fgp = GraphProcessor(adsg, encoder_type=SelChoiceEncoderType.FAST)
    assert fgp.encoder_type == SelChoiceEncoderType.FAST
    assert fgp.get_n_valid_designs() == 32
    assert fgp.get_n_design_space() == 32
    assert fgp.get_imputation_ratio() == 1.

    assert fgp.get_all_discrete_x() is None
    fgp.print_stats()

    x_all_seen = set()
    for i, xi in enumerate(x_all):
        graph, x_imp, is_act = fgp.get_graph(list(xi), create=False)
        assert graph is not None

        assert np.all(x_imp == xi)
        assert np.all(is_act == is_act_all[i, :])

        graph2, x_imp2, is_act2 = fgp.get_graph(list(xi), create=True)
        assert np.all(x_imp2 == x_imp)
        assert np.all(is_act2 == is_act)
        assert graph2 == graph

        x_all_seen.add(tuple(xi))

    for xi in itertools.product(*[range(dv.n_opts) for dv in fgp.des_vars]):
        graph, x_imp, _ = fgp.get_graph(list(xi))
        assert tuple(x_imp) in x_all_seen


def test_fast_encoder_infeasible_conn(n):
    adsg = BasicDSG()
    cn1 = [ConnectorNode(f'CN{i}', deg_list=[1, 2], repeated_allowed=True) for i in range(2)]
    cn2 = [ConnectorNode(f'CN{i+2}', deg_list=[1], repeated_allowed=True) for i in range(2)]
    cn3 = ConnectorNode('CN5', deg_min=2, repeated_allowed=True)
    adsg.add_edges([
        (n[13], n[12]), (n[23], n[22]),
        (n[12], cn1[0]), (n[13], cn1[1]),
        (n[22], cn2[0]), (n[23], cn2[1]),
        (n[2], n[31]), (n[31], cn3),
    ])
    adsg.add_selection_choice('C1', n[1], [n[11], n[21]])
    adsg.add_selection_choice('C2', n[11], [n[12], n[13]])
    adsg.add_selection_choice('C3', n[21], [n[22], n[23]])
    adsg.add_connection_choice('C4', cn1+cn2, [cn3])
    adsg = adsg.set_start_nodes({n[1], n[2]})

    cgp = GraphProcessor(adsg, encoder_type=SelChoiceEncoderType.COMPLETE)
    assert cgp.get_n_valid_designs() == 6
    assert len(cgp.des_vars) == 5
    x_all, is_act_all = cgp.get_all_discrete_x()

    fgp = GraphProcessor(adsg, encoder_type=SelChoiceEncoderType.FAST)
    assert fgp.encoder_type == SelChoiceEncoderType.FAST
    assert len(fgp.des_vars) == 5
    assert fgp.get_n_valid_designs() == 32
    fgp.print_stats()

    x_all_seen = set()
    for i, xi in enumerate(x_all):
        graph, x_imp, is_act = fgp.get_graph(list(xi), create=False)
        assert graph is not None

        assert np.all(x_imp == xi)
        assert np.all(is_act == is_act_all[i, :])

        graph2, x_imp2, is_act2 = fgp.get_graph(list(xi), create=True)
        assert np.all(x_imp2 == x_imp)
        assert np.all(is_act2 == is_act)
        assert graph2 == graph

        x_all_seen.add(tuple(xi))

    for xi in itertools.product(*[range(dv.n_opts) for dv in fgp.des_vars]):
        graph, x_imp, _ = fgp.get_graph(list(xi))
        assert tuple(x_imp) in x_all_seen


def test_fast_encoder_choice_constraint(n):
    for encoder_type in SelChoiceEncoderType:
        adsg = BasicDSG()
        c1 = adsg.add_selection_choice('C1', n[1], n[11:14])
        c2 = adsg.add_selection_choice('C2', n[2], n[21:24])
        adsg = adsg.set_start_nodes({n[1], n[2]})

        adsg = adsg.constrain_choices(ChoiceConstraintType.UNORDERED, [c1, c2])
        assert len(adsg.choice_nodes) == 2

        processor = GraphProcessor(adsg, encoder_type=encoder_type)
        assert len(processor.des_vars) == 2

        x_all_seen = set()
        for xi in itertools.product(*[range(dv.n_opts) for dv in processor.des_vars]):
            graph, x_imp, _ = processor.get_graph(list(xi))
            x_all_seen.add(tuple(x_imp))
        assert len(x_all_seen) == 6


def test_gnc_problem():
    gnc = GNCEvaluator()
    assert gnc.encoder_type == SelChoiceEncoderType.COMPLETE

    assert gnc.calc_mass(['A'], ['A']) == gnc.mass['S']['A'] + gnc.mass['C']['A']
    assert gnc.calc_failure_rate(['A'], ['A'], {(0, 0)}) == \
           -math.log10(gnc.failure_rate['S']['A']+gnc.failure_rate['C']['A'] +
                    (gnc.failure_rate['S']['A']*gnc.failure_rate['C']['A']))

    assert len(gnc.des_vars) == 17
    assert len(gnc.objectives) == 2
    assert len(gnc.constraints) == 0
    gnc.print_stats()

    assert len(GNCEvaluator(objective=0).objectives) == 1
    assert len(GNCEvaluator(objective=1).objectives) == 1

    for _ in range(10):
        des_var_values = gnc.get_random_design_vector()
        graph, des_var_values, _ = gnc.get_graph(des_var_values)
        assert graph.final
        assert graph.feasible

        obj, con = gnc.evaluate(graph)
        assert con == []
        assert all([obj > 0. for obj in obj])

    assert gnc.get_n_valid_designs() == 29857
    x_all, is_act_all = gnc.get_all_discrete_x()
    assert x_all.shape[0] == 29857
    assert x_all.shape == is_act_all.shape


@pytest.mark.skipif(int(os.getenv('RUN_SLOW_TESTS', 0)) != 1, reason='Set RUN_SLOW_TESTS=1 to run slow tests')
def test_gnc_problem_fast_encoder():
    gnc = GNCEvaluator()
    gnc._encoder_type = SelChoiceEncoderType.FAST
    assert len(gnc.des_vars) == 17
    assert gnc.encoder_type == SelChoiceEncoderType.FAST
    gnc.print_stats()

    assert gnc.get_n_valid_designs() > 29857
    assert gnc.get_all_discrete_x() is None

    for _ in range(10):
        des_var_values = gnc.get_random_design_vector()
        graph, des_var_values, _ = gnc.get_graph(des_var_values)
        assert graph.final
        assert graph.feasible

        obj, con = gnc.evaluate(graph)
        assert con == []
        assert all([obj > 0. for obj in obj])


def test_apollo_problem():
    apollo = ApolloEvaluator()
    assert apollo.encoder_type == SelChoiceEncoderType.COMPLETE

    assert len(apollo.des_vars) == 9
    assert [dv.n_opts for dv in apollo.des_vars] == [2, 2, 2, 2, 2, 3, 2, 2, 2]
    assert len(apollo.objectives) == 2
    assert len(apollo.constraints) == 0

    assert len(ApolloEvaluator(objective=0).objectives) == 1
    assert len(ApolloEvaluator(objective=1).objectives) == 1

    df_stats = apollo.get_statistics()
    assert df_stats['n_valid']['total-design-space'] == 108
    apollo.print_stats()

    apollo._hierarchy_analyzer._assert_behavior()
    for _ in range(100):
        des_var_values = apollo.get_random_design_vector()
        _, des_var_values_, is_active_ = apollo.get_graph(des_var_values, create=False)
        graph, des_var_values, is_active = apollo.get_graph(des_var_values)
        assert graph.final
        assert graph.feasible

        assert des_var_values == des_var_values_
        assert is_active == is_active_

        obj, con = apollo.evaluate(graph)
        assert con == []
        assert all([obj > 0. for obj in obj])

    assert_processor_get_all(apollo)


def test_apollo_problem_fast_encoder():
    apollo = ApolloEvaluator()
    apollo._encoder_type = SelChoiceEncoderType.FAST
    assert len(apollo.des_vars) == 9
    assert apollo.encoder_type == SelChoiceEncoderType.FAST
    apollo.print_stats()

    assert apollo.get_n_valid_designs() > 108
    assert apollo.get_all_discrete_x() is None

    apollo2 = ApolloEvaluator()

    n_wrong = 0
    for _ in range(100):
        des_var_values = apollo.get_random_design_vector()
        _, des_var_values_, is_active_ = apollo2.get_graph(des_var_values, create=False)
        graph, des_var_values, is_active = apollo.get_graph(des_var_values)
        assert graph.final
        assert graph.feasible

        if not (des_var_values == des_var_values_):
            n_wrong += 1
            continue
        assert des_var_values == des_var_values_
        assert is_active == is_active_

        obj, con = apollo.evaluate(graph)
        assert con == []
        assert all([obj > 0. for obj in obj])

    assert n_wrong < 5


def test_connection_encoder(n):
    adsg = BasicDSG()
    cn1 = [ConnectorNode('CN10', deg_list=[0, 1, 2]), ConnectorNode('CN11', deg_list=[0, 1])]
    cn2 = [ConnectorNode('CN20', deg_list=[0, 1]), ConnectorNode('CN21', deg_list=[1])]
    adsg.add_edges([
        (n[1], n[11]), (n[11], n[2]),
        (n[11], cn1[0]), (n[11], cn1[1]),
        (n[11], MetricNode('OBJ', direction=-1, type_=MetricType.OBJECTIVE)),
        (n[22], cn2[0]), (n[22], cn2[1]),
    ])
    adsg.add_selection_choice('C1', n[2], [n[21], n[22]])
    adsg.add_connection_choice('P1', [cn1[0]], [cn2[0]])
    adsg.add_connection_choice('P2', [cn1[1]], [cn2[1]])
    adsg = adsg.set_start_nodes({n[1]})

    processor = GraphProcessor(adsg)
    assert processor.encoder_type == SelChoiceEncoderType.COMPLETE
    assert len(processor.objectives) == 1
    assert len(processor.des_vars) == 2

    assert processor.des_vars[0].n_opts == 2
    assert processor.des_vars[1].n_opts == 2
    processor.print_stats()

    assert_processor_get_all(processor)


def test_single_option_connection_choice(n):
    dsg = BasicDSG()
    cn1 = ConnectorNode('CN1', deg_list=[1])
    cn2 = ConnectorNode('CN2', deg_list=[1])
    dsg.add_edges([(n[1], cn1), (n[2], cn2)])
    dsg.add_connection_choice('C', [cn1], [cn2])
    dsg = dsg.set_start_nodes({n[1], n[2]})

    processor = GraphProcessor(dsg)
    assert len(processor.des_vars) == 0
    processor.print_stats()

    assert_processor_get_all(processor)


def test_single_option_connection_choice2(n):
    dsg = BasicDSG()
    cn1 = ConnectorNode('CN1', deg_list=[1])
    cn2 = ConnectorNode('CN2', deg_min=1)
    dsg.add_edges([(n[1], cn1), (n[2], cn2)])
    dsg.add_connection_choice('C', [cn1], [cn2])
    dsg = dsg.set_start_nodes({n[1], n[2]})

    processor = GraphProcessor(dsg)
    assert len(processor.des_vars) == 0
    processor.print_stats()

    assert_processor_get_all(processor)


def test_port_group_edges(n):
    adsg = BasicDSG()
    cn1 = [ConnectorNode('CN1', deg_list=[0, 1], repeated_allowed=True) for _ in range(2)]
    cn2 = ConnectorNode('CN2', deg_list=[1, 2, 3], repeated_allowed=True)
    adsg.add_edges([
        (n[1], n[11]), (n[11], n[2]),
        (cn1[1], cn1[0]),
        (n[2], n[21]), (n[21], cn2),
    ])
    adsg.add_selection_choice('C1', n[11], cn1)
    adsg.add_connection_choice('C2', [(ConnectorDegreeGroupingNode('Grp'), cn1)], [cn2])
    adsg = adsg.set_start_nodes({n[1]})

    assert len(adsg.choice_nodes) == 2
    decision_node = adsg.get_ordered_next_choice_nodes()[0]
    assert isinstance(decision_node, SelectionChoiceNode)
    option_nodes = adsg.get_option_nodes(decision_node)
    assert len(option_nodes) == 2

    processor = GraphProcessor(adsg)
    assert len(processor.des_vars) == 2

    seen = set()
    for i in range(2):
        for j in range(2):
            dv = [i, j]
            adsg, used_dv, _ = processor.get_graph(dv)
            seen.add(tuple(used_dv))

            n_conn = len([edge for edge in iter_in_edges(adsg.graph, cn2) if get_edge_type(edge) == EdgeType.CONNECTS])
            assert n_conn == (2 if (i == 1 and j == 1) else 1)

    assert len(seen) == 3
    assert_processor_get_all(processor)


def test_graph_processor_none_metric(n):
    adsg = BasicDSG()
    adsg.add_edges([
        (n[1], n[11]), (n[11], n[2]),
        (n[2], MetricNode('M1', direction=-1)),
        (n[2], MetricNode('M2', direction=-1, type_=MetricType.NONE)),
    ])
    adsg.add_selection_choice('C1', n[2], [n[21], n[22]])
    adsg = adsg.set_start_nodes({n[1]})

    metric_nodes = sorted(adsg.get_nodes_by_type(MetricNode), key=lambda n: n.name)
    assert len(metric_nodes) == 2
    assert metric_nodes[0].type is None
    assert metric_nodes[1].type == MetricType.NONE

    processor = GraphProcessor(adsg)
    assert processor._metrics == [
        (metric_nodes[0], MetricType.OBJECTIVE),
        (metric_nodes[1], MetricType.NONE),
    ]
    assert len(processor.objectives) == 1
    assert len(processor.constraints) == 0
    assert_processor_get_all(processor)


def test_graph_processor_optional_conn(n):
    adsg = BasicDSG()
    cn = [ConnectorNode('CN', deg_list=[0, 1]) for _ in range(2)]
    adsg.add_edges([
        (n[11], n[2]), (n[12], n[2]), (n[12], n[3]),
        (n[2], n[21]), (n[5], n[22]),
        (n[21], cn[0]), (n[22], cn[1]),
    ])
    adsg.add_selection_choice('C1', n[1], [n[11], n[12]])
    adsg.add_selection_choice('C2', n[3], [n[31], n[32]])
    adsg.add_connection_choice('C3', [cn[0]], [cn[1]])
    adsg = adsg.set_start_nodes({n[1], n[5]})

    processor = GraphProcessor(adsg)
    assert len(processor.des_vars) == 3
    assert_processor_get_all(processor)


def test_graph_processor_derived_conn(n):
    adsg = BasicDSG()
    cn = [ConnectorNode('CN', deg_list=[1]) for _ in range(2)]
    adsg.add_edges([
        (n[11], n[2]), (n[12], n[2]), (n[12], n[3]),
        (n[2], n[21]), (n[5], n[22]),
        (n[21], cn[0]), (n[22], cn[1]),
    ])
    adsg.add_selection_choice('C1', n[1], [n[11], n[12]])
    adsg.add_selection_choice('C2', n[3], [n[31], n[32]])
    adsg.add_connection_choice('C3', [cn[0]], [cn[1]])
    adsg = adsg.set_start_nodes({n[1], n[5]})

    assert len(adsg.choice_nodes) == 3
    processor = GraphProcessor(adsg)
    assert len(processor.des_vars) == 2

    assert_processor_get_all(processor)


def test_graph_processor_sub_choice_incompatibility(n):
    adsg = BasicDSG()
    adsg.add_edges([
        (n[11], n[2]), (n[12], n[2]), (n[12], n[3]),
        (n[2], n[21]), (n[21], n[4]),
    ])
    adsg.add_selection_choice('C1', n[1], [n[11], n[12]])
    adsg.add_selection_choice('C2', n[3], [n[31], n[32]])
    adsg.add_selection_choice('C3', n[4], [n[41], n[42]])
    adsg.add_incompatibility_constraint([n[12], n[42]])
    adsg = adsg.set_start_nodes({n[1]})

    processor = GraphProcessor(adsg)
    assert len(processor.des_vars) == 3

    graph_inst, _, _ = processor.get_graph([0, 1, 1])
    assert graph_inst.final
    assert_processor_get_all(processor)


def test_graph_processor_dependent_choice(n):
    adsg = BasicDSG()
    adsg.add_edges([
        (n[1], n[11]), (n[2], n[12]),
    ])
    c1 = adsg.add_selection_choice('C1', n[11], [n[21], n[22]])
    adsg.add_selection_choice('C2', n[12], [n[23], n[24]])
    adsg = adsg.set_start_nodes({n[1], n[2]})

    assert len(adsg.choice_nodes) == 2
    adsg = adsg.constrain_choices(ChoiceConstraintType.LINKED, adsg.choice_nodes)

    option_nodes = adsg.get_option_nodes(c1)
    adsg2 = adsg.get_for_apply_selection_choice(c1, option_nodes[0])
    assert adsg2.final

    processor = GraphProcessor(adsg)
    assert len(processor.des_vars) == 1
    assert_processor_get_all(processor)


def test_many_arch(n):
    adsg = BasicDSG()
    start_nodes = set()
    i_use = 10
    for i in range(8):
        start_nodes.add(n[i])

        selectable_nodes = []
        for j in range(3):
            selectable_nodes.append(n[i_use])
            i_use += 1
        adsg.add_selection_choice(f'C{i}', n[i], selectable_nodes)

    adsg = adsg.set_start_nodes(start_nodes)
    gp2 = GraphProcessor(adsg)
    assert gp2.encoder_type == SelChoiceEncoderType.COMPLETE
    assert gp2.des_vars
    assert gp2._hierarchy_analyzer.n_combinations == 6561


def test_gp_derived_time():
    n = [NamedNode(str(i)) for i in range(200)]

    def _get_graph(n_opt_dec, n_opt=2):
        adsg = BasicDSG()
        base_node = [n[1]]
        next_base_node = []
        i = 2
        for _ in range(n_opt_dec):
            if len(base_node) == 0:
                base_node = next_base_node
                next_base_node = []

            func = base_node.pop(0)
            selection_targets = []
            for _ in range(n_opt):
                next_node = n[i]
                next_base_node.append(next_node)
                selection_targets.append(next_node)
                i += 1
            adsg.add_selection_choice(f'C{i}', func, selection_targets)

        return adsg.set_start_nodes({n[1]})

    def _check_graph(n_opt_dec, n_opt=2):
        graph = _get_graph(n_opt_dec, n_opt=n_opt)
        # graph.export_dot('graph.dot')
        s = timeit.default_timer()

        gp = GraphProcessor(graph)
        # gp.check_derived_dv = True
        gp._get_des_vars()

        t = timeit.default_timer()-s
        gp._hierarchy_analyzer._assert_behavior()
        print(f'n_opt_dec={n_opt_dec}, n_opt={n_opt}, n_arch={gp._hierarchy_analyzer.n_combinations}, '
              f'time = {t:.3f} s')

    print('')
    # _check_graph(n_opt_dec=18)  # 260k, ~5.7s (37ms)
    # _check_graph(n_opt_dec=12, n_opt=3)  # 530k, ~6.3s (43ms)
    _check_graph(n_opt_dec=10, n_opt=4)  # 1050k, ~10.7s (54ms)
    # _check_graph(n_opt_dec=8, n_opt=5)  # 390k, ~3.6s (61ms)
    # _check_graph(n_opt_dec=6, n_opt=10)  # 1000k, ~8.5s (159ms)
    # _check_graph(n_opt_dec=4, n_opt=20)  # 160k, ~1.6s (595ms)


def test_duplicate_assign_enc_patterns(n):
    adsg = BasicDSG()
    cn1 = ConnectorNode('CN1', deg_spec='*', repeated_allowed=True)
    cn2 = [ConnectorNode('CN2', deg_spec='*', repeated_allowed=True) for _ in range(2)]
    adsg.add_edges([
        (n[1], cn1), (cn2[1], cn2[0]),
    ])
    adsg.add_selection_choice('C1', n[2], cn2)
    adsg.add_connection_choice('C2', [cn1], [(ConnectorDegreeGroupingNode('Grp'), cn2)])
    adsg = adsg.set_start_nodes({n[1], n[2]})

    processor = GraphProcessor(adsg)
    assert processor.des_vars
    assert_processor_get_all(processor)


def test_simple_connector(n):
    adsg = BasicDSG()
    cn = [ConnectorNode('CN', deg_min=1, repeated_allowed=False) for _ in range(4)]
    adsg.add_connection_choice('C', cn[:2], [(ConnectorDegreeGroupingNode('Grp'), cn[2:])])
    start = NamedNode('S')
    adsg.add_edges([(start, cn_) for cn_ in cn])
    adsg = adsg.set_start_nodes({start})

    processor = GraphProcessor(adsg)
    assert len(processor.des_vars) == 0
    assert_processor_get_all(processor)


def test_async_start_node_def(n):
    cn = [ConnectorNode(f'CN{i}', deg_spec='+' if i < 5 else '*') for i in range(11)]

    adsg = BasicDSG()
    adsg.add_selection_choice('C1', n[1], [n[2], n[3]])
    adsg.add_edges([
        (n[0], n[1]),
        (n[2], n[10]), (n[2], n[11]),
        (n[3], n[10]), (n[3], n[11]), (n[3], n[30]),
    ])
    adsg.add_selection_choice('C2', n[10], cn[:2])
    adsg.add_selection_choice('C3', n[11], cn[2:4])
    adsg.add_selection_choice('C10', n[30], cn[4:5])

    adsg.add_connection_choice('C4', [cn[0], cn[2]], cn[5:7])
    adsg.add_connection_choice('C5', [cn[1], cn[3]], cn[7:10])
    adsg.add_connection_choice('C11', cn[4:5], cn[10:11])
    adsg.add_edges([
        (cn[5], n[12]), (cn[6], n[13]),
        (cn[7], n[14]), (cn[8], n[15]),
        (cn[9], n[31]), (cn[10], n[32]),
    ])

    adsg = adsg.set_start_nodes({n[0]} | set(cn[5:]))

    adsg.add_selection_choice('C6', n[12], n[16:18])
    adsg.add_selection_choice('C7', n[13], n[18:20])
    adsg.add_selection_choice('C8', n[14], n[20:22])
    adsg.add_selection_choice('C9', n[15], n[22:24])
    adsg.add_selection_choice('C12', n[31], n[33:35])
    adsg.add_selection_choice('C13', n[32], n[35:37])

    adsg.add_incompatibility_constraint([n[2], cn[0]])
    adsg.add_incompatibility_constraint([n[2], cn[2]])
    adsg.add_incompatibility_constraint([n[2], cn[4]])
    adsg.add_incompatibility_constraint([n[3], cn[1]])
    adsg.add_incompatibility_constraint([n[3], cn[3]])

    # adsg.render()

    processor = GraphProcessor(adsg)
    for _ in range(10):
        processor.get_graph(processor.get_random_design_vector())
    # _test_processor_get_all(processor)


def test_dv_ordering():
    adsg = BasicDSG()
    dv = [DesignVariableNode(f'x{i}', bounds=(0, 1)) for i in range(25)]
    start = NamedNode('S')
    adsg.add_edges([(start, dv_) for dv_ in dv])
    adsg = adsg.set_start_nodes({start})

    processor = GraphProcessor(adsg)
    assert len(processor.des_vars) == len(dv)
    x_names = [dv.name for dv in processor.des_vars]
    assert x_names == [dv_.name for dv_ in dv]
