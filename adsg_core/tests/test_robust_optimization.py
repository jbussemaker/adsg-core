import pytest
import numpy as np
import openturns as ot
from typing import *
from adsg_core.graph.adsg_basic import *
from adsg_core.graph.adsg_nodes import *
from adsg_core.optimization.stochastic_evaluator import *
from adsg_core.optimization.graph_processor import *
from sb_arch_opt.uncertainty import (MonteCarlo, PolynomialChaos, StochasticOutput,
                                     Mean, Margin, Quantile)


def _dsg_with_parameters(n, par_nodes):
    dsg = BasicDSG()
    dsg.add_edges([(n[0], par_node) for par_node in par_nodes])
    return dsg.set_start_nodes({n[0]})


def _evaluator(dsg, n_evaluations=5, seed=42) -> 'DSGStochasticEvaluator':
    class _Stochastic_Evaluator(DSGStochasticEvaluator):
        def _evaluate_sample(self, dsg_instance, metric_nodes):
            return {}

    return _Stochastic_Evaluator(dsg, uq_method=MonteCarlo(n_evaluations, seed=seed))


def _dsg_with_branch_parameters(n, common, only_a, only_b):
    dsg = BasicDSG()
    dsg.add_edges([(n[0], common), (n[1], only_a), (n[2], only_b)])
    dsg.add_selection_choice('C1', n[0], [n[1], n[2]])
    return dsg.set_start_nodes({n[0]})


class BeamStochasticEvaluator(DSGStochasticEvaluator):
    """Pick a material and a thickness for a beam under an uncertain load.

    material: steel (stiff, heavy) or alu; t: continuous thickness. The stiffness parameter is branch-local, so
    an instance never carries all parameters.
    """

    stiffness = {'steel': 210., 'alu': 70.}
    density = {'steel': 7.8, 'alu': 2.7}

    def __init__(self, uq_method=None, stress_ref=None, **kwargs):
        self.seen_loads = []

        self.par_load = InputParameterNode('load', ot.Normal(100., 20.))
        self.par_rho = InputParameterNode('rho_factor', 1.5)  # deterministic
        self.par_e = {'steel': InputParameterNode('E_steel', ot.Normal(210., 10.)),
                      'alu': InputParameterNode('E_alu', ot.Normal(70., 10.))}

        self.mass_node = MetricNode('mass', direction=-1, type_=MetricType.OBJECTIVE)
        self.deflection_node = MetricNode('deflection', direction=-1, type_=MetricType.OBJECTIVE)
        self.capacity_node = MetricNode('capacity', direction=1, type_=MetricType.OBJECTIVE)  # maximized

        self.stress_ref = stress_ref
        self.stress_node = None
        if stress_ref is not None:
            self.stress_node = MetricNode('stress', direction=-1, ref=stress_ref, type_=MetricType.CONSTRAINT)

        self.material_nodes = {}
        kwargs.setdefault('uq_method', uq_method if uq_method is not None else MonteCarlo(20, seed=42))
        super().__init__(self._build_dsg(), **kwargs)

    def _build_dsg(self):
        dsg = BasicDSG()
        beam = NamedNode('beam')
        dsg.add_edges([(beam, self.mass_node), (beam, self.deflection_node), (beam, self.capacity_node),
                       (beam, self.par_load), (beam, self.par_rho),
                       (beam, DesignVariableNode('t', bounds=(1., 5.)))])

        for name in ['steel', 'alu']:
            material_node = NamedNode(name)
            self.material_nodes[name] = material_node
            dsg.add_edge(material_node, self.par_e[name])

        if self.stress_node is not None:
            dsg.add_edge(beam, self.stress_node)

        dsg.add_selection_choice('material', beam, [self.material_nodes['steel'], self.material_nodes['alu']])
        return dsg.set_start_nodes({beam})

    def _material(self, dsg: DSGType) -> str:
        for name, material_node in self.material_nodes.items():
            if material_node in dsg.graph.nodes:
                return name
        raise RuntimeError('No material selected!')

    def _thickness(self, dsg: DSGType) -> float:
        for des_var_node, value in dsg.des_var_values.items():
            if des_var_node.name == 't':
                return value
        raise RuntimeError('No thickness!')

    def _evaluate_sample(self, dsg: DSGType, metric_nodes: List[MetricNode]) -> Dict[MetricNode, float]:
        material = self._material(dsg)
        thickness = self._thickness(dsg)
        load = dsg.inp_param_value(self.par_load)
        e_modulus = dsg.inp_param_value(self.par_e[material])
        rho_factor = dsg.inp_param_value(self.par_rho)
        self.seen_loads.append(load)

        values = {}

        for metric_node in metric_nodes:
            if metric_node.name == 'mass':
                values[metric_node] = self.density[material]*thickness*rho_factor
            elif metric_node.name == 'deflection':
                values[metric_node] = load / (e_modulus*thickness**3)
            elif metric_node.name == 'capacity':
                values[metric_node] = e_modulus*thickness**2 / load
            elif metric_node.name == 'stress':
                values[metric_node] = load / thickness**2

        return values


@pytest.fixture
def beam():
    return BeamStochasticEvaluator()


@pytest.fixture
def constrained_beam():
    return BeamStochasticEvaluator(stress_ref=60.)


def test_input_parameter_node(n):
    stochastic = InputParameterNode('E', ot.Normal(10., 2.))
    deterministic = InputParameterNode('rho', 1.225)

    assert stochastic.name == 'E'
    assert stochastic.idx is None
    assert stochastic.is_stochastic
    assert not deterministic.is_stochastic
    assert deterministic.value == 1.225

    assert str(stochastic) == 'INP[E]'
    assert stochastic.str_context() == 'INP[E]'

    # Only a distribution makes a parameter stochastic; any other value is a fixed number
    assert InputParameterNode('h', ot.Uniform(0., 1.)).is_stochastic
    assert not InputParameterNode('n_blades', 3).is_stochastic
    assert repr(stochastic)
    assert stochastic.get_export_color()
    assert 'E = ' in stochastic.get_export_title()

    # Nodes are identity-based, so two parameters with the same name are still different nodes
    assert InputParameterNode('E', ot.Normal(10., 2.)) != stochastic
    assert len({stochastic, InputParameterNode('E', ot.Normal(10., 2.))}) == 2


def test_set_get_input_parameter_value(n):
    par_a = InputParameterNode('A', ot.Normal(0., 1.))
    par_b = InputParameterNode('B', 2.5)
    dsg = _dsg_with_parameters(n, [par_a, par_b])

    assert dsg.feasible
    assert set(dsg.inp_param_nodes) == {par_a, par_b}
    assert dsg.inp_param_value(par_a) is None  # nothing assigned on the graph yet

    dist = ot.Normal(5., 1.)
    dsg.set_inp_param_value(par_a, dist)
    assert dsg.inp_param_value(par_a) is dist

    values = dsg.inp_param_values
    values[par_a] = 99.
    assert dsg.inp_param_value(par_a) is dist  # the mapping is a copy

    assert dsg.copy().inp_param_value(par_a) is dist  # and survives derivation

    dsg.reset_inp_param_values()
    assert dsg.inp_param_values == {}


def test_parameter_values_isolated_between_instances(n):
    par_a = InputParameterNode('A', ot.Normal(0., 1.))
    dsg = BasicDSG()
    dsg.add_edges([(n[0], par_a)])
    dsg.add_selection_choice('C1', n[0], [n[1], n[2]])
    processor = GraphProcessor(dsg.set_start_nodes({n[0]}))

    graph_a, _, _ = processor.get_graph([0])
    graph_b, _, _ = processor.get_graph([1])
    graph_a.set_inp_param_value(par_a, 1.)
    graph_b.set_inp_param_value(par_a, 2.)

    assert graph_a.inp_param_value(par_a) == 1.
    assert graph_b.inp_param_value(par_a) == 2.
    assert processor.graph.inp_param_value(par_a) is None  # template untouched


def test_parameters_are_not_design_variables(n):
    par_a = InputParameterNode('A', ot.Normal(0., 1.))
    dv_node = DesignVariableNode('DV', bounds=(0., 1.))

    dsg = BasicDSG()
    dsg.add_edges([(n[0], par_a), (n[0], dv_node)])
    dsg.add_selection_choice('C1', n[0], [n[1], n[2]])
    processor = GraphProcessor(dsg.set_start_nodes({n[0]}))

    assert [dv.name for dv in processor.des_vars] == ['C1', 'DV']
    assert all(dv.node is not par_a for dv in processor.des_vars)
    assert par_a not in processor.design_variable_nodes


def test_param_space_holds_only_stochastic_parameters(n):
    stochastic = InputParameterNode('u', ot.Normal(0., 1.))
    deterministic = InputParameterNode('rho', 1.225)
    processor = GraphProcessor(_dsg_with_parameters(n, [stochastic, deterministic]))

    assert [node.name for node in processor.inp_param_nodes] == ['rho', 'u']  # sorted by name
    assert processor.param_space.parameter_names == ['u']
    assert processor.param_space.n_parameters == 1
    assert processor.param_space.joint_dist.getDimension() == 1


def test_param_space_uses_the_distributions_from_the_graph(n):
    # The distribution stored on the node is what the parameter space hands to the UQ method
    normal = InputParameterNode('u', ot.Normal(10., 2.))
    uniform = InputParameterNode('h', ot.Uniform(4., 10.))
    fixed = InputParameterNode('n_blades', 3)  # not a distribution, so it never reaches the space
    processor = GraphProcessor(_dsg_with_parameters(n, [normal, uniform, fixed]))

    assert processor.param_space.parameter_names == ['h', 'u']
    assert [parameter.dist for parameter in processor.param_space._parameters] == [uniform.value, normal.value]

    marginals = {name: processor.param_space.joint_dist.getMarginal(i)
                 for i, name in enumerate(processor.param_space.parameter_names)}

    assert marginals['u'].getMean()[0] == pytest.approx(10.)
    assert marginals['u'].getStandardDeviation()[0] == pytest.approx(2.)
    assert marginals['h'].getRange().getLowerBound()[0] == pytest.approx(4.)
    assert marginals['h'].getRange().getUpperBound()[0] == pytest.approx(10.)


def test_param_realization_is_keyed_by_node(n):
    # The realization covers every input parameter node, stochastic or not: a deterministic one contributes its
    # own value, so the evaluation always finds a number for every parameter it can reach on the graph
    stochastic = InputParameterNode('u', ot.Normal(10., 2.))
    deterministic = InputParameterNode('rho', 1.225)
    evaluator = _evaluator(_dsg_with_parameters(n, [stochastic, deterministic]))

    samples = MonteCarlo(5, seed=42).get_samples(evaluator.param_space)
    param_nodes = [stochastic, deterministic]

    seen = []
    for i in range(5):
        realization = evaluator._param_realization(param_nodes, samples, i)

        assert realization[deterministic] == 1.225
        assert realization[stochastic] == pytest.approx(samples[i, 0])
        assert all(isinstance(value, float) for value in realization.values())
        seen.append(realization[stochastic])

    assert len(set(seen)) == 5  # a different realization each time


def test_param_realization_covers_branch_local_parameters(n):
    # A parameter that only exists in one branch still has a column in the space, so every instance's nodes
    # resolve against the same realization
    common = InputParameterNode('common', ot.Normal(0., 1.))
    only_a = InputParameterNode('only_a', ot.Normal(1., 1.))
    only_b = InputParameterNode('only_b', ot.Normal(2., 1.))
    evaluator = _evaluator(_dsg_with_branch_parameters(n, common, only_a, only_b))

    samples = MonteCarlo(5, seed=42).get_samples(evaluator.param_space)
    realization = evaluator._param_realization([common, only_a, only_b], samples, 0)

    assert set(realization) == {common, only_a, only_b}
    assert all(isinstance(value, float) for value in realization.values())

    # Only the nodes this instance carries are resolved, and they take the same columns of the same design
    instance_only = evaluator._param_realization([common, only_a], samples, 0)
    assert set(instance_only) == {common, only_a}
    assert instance_only[common] == realization[common]


def test_evaluate_restores_parameter_values(beam):
    dsg, _, _ = beam.get_graph([0, 3.])
    beam.evaluate(dsg)

    # The loop writes realizations onto the instance; afterwards the nodes carry their own value again
    for node in dsg.inp_param_nodes:
        assert dsg.inp_param_value(node) is node.value
    assert dsg.inp_param_value(beam.par_rho) == 1.5


def test_evaluate_stores_a_stochastic_output_per_metric(beam):
    dsg, _, _ = beam.get_graph([0, 3.])
    objective_values, constraint_values = beam.evaluate(dsg)

    assert constraint_values == []
    assert len(objective_values) == 3
    # The deflection scatters with the load, so it has a fitted distribution; the mass does not depend on any
    # stochastic parameter, so it is constant over the realizations and comes back as the value itself
    deflection, mass = dsg.metric_value(beam.deflection_node), dsg.metric_value(beam.mass_node)
    assert isinstance(deflection, StochasticOutput)
    assert isinstance(mass, float)

    # Values are physical: no sign conventions are applied by the evaluator
    assert mass == pytest.approx(7.8*3.*1.5)
    assert deflection.quantile(.01) > 0.
    assert deflection.std > 0.


def test_evaluate_pairs_each_metric_with_its_own_output(constrained_beam):
    # Regression: outputs were assigned by position over the instance's own metric_nodes, which is graph-ordered
    # (mass, deflection, capacity, stress) rather than objectives-by-name-then-constraints (capacity, deflection,
    # mass, stress). Every value below is checked against what the model computes, not against what evaluate()
    # returned: both come from the same mapping, so a mispairing would corrupt them consistently.
    dsg, _, _ = constrained_beam.get_graph([0, 3.])
    objective_values, constraint_values = constrained_beam.evaluate(dsg)
    by_name = {objective.name: value for objective, value in zip(constrained_beam.objectives, objective_values)}

    e_steel, load, thickness = 210., 100., 3.
    assert by_name['mass'] == pytest.approx(7.8*thickness*1.5)  # deterministic: a plain value
    assert by_name['capacity'].mean == pytest.approx(e_steel*thickness**2 / load, rel=.1)
    assert by_name['deflection'].mean == pytest.approx(load / (e_steel*thickness**3), rel=.1)
    assert constraint_values[0].mean == pytest.approx(load / thickness**2, rel=.1)

    assert dsg.metric_value(constrained_beam.mass_node) == pytest.approx(by_name['mass'])
    for node, name in [(constrained_beam.capacity_node, 'capacity'), (constrained_beam.deflection_node, 'deflection')]:
        assert dsg.metric_value(node).mean == pytest.approx(by_name[name].mean)
    assert dsg.metric_value(constrained_beam.stress_node).mean == pytest.approx(constraint_values[0].mean)


def test_evaluated_instances_keep_their_own_outputs(beam):
    steel, _, _ = beam.get_graph([0, 3.])
    alu, _, _ = beam.get_graph([1, 3.])
    beam.evaluate(steel)
    beam.evaluate(alu)

    means = [dsg.metric_value(beam.deflection_node).mean for dsg in (steel, alu)]
    assert means[0] != means[1]
    assert means[0] < means[1]  # steel is stiffer, so it deflects less


def test_export_handles_stochastic_outputs(beam):
    dsg, _, _ = beam.get_graph([0, 3.])
    beam.evaluate(dsg)
    dsg._get_graph_for_export()

    title = beam.deflection_node.get_export_title()
    assert 'mean =' in title and 'sigma =' in title


def test_problem_shape_and_parameter_space(constrained_beam):
    problem = constrained_beam.get_problem()

    assert problem.n_obj == len(constrained_beam.objectives) == 3
    assert problem.n_ieq_constr == len(constrained_beam.constraints) == 1
    assert problem.n_var == len(constrained_beam.des_vars)
    assert problem.param_space.parameter_names == constrained_beam.param_space.parameter_names
    assert problem.param_space.parameter_names == ['E_alu', 'E_steel', 'load']  # rho_factor is deterministic
    assert repr(problem)


def test_problem_applies_the_optimizer_conventions(constrained_beam):
    # Regression: obj_is_max and con_ref were computed but never applied, so a maximized objective was minimized
    problem = constrained_beam.get_problem()
    x = np.array([[0, 3.]])
    out = problem.evaluate(x, return_as_dictionary=True)

    i_capacity = [objective.name for objective in constrained_beam.objectives].index('capacity')
    i_mass = [objective.name for objective in constrained_beam.objectives].index('mass')
    capacity = out['f_stochastic'][0, i_capacity]

    # capacity is maximized, so it is stored negated; mass is minimized and stored as-is
    assert out['F'][0, i_capacity] < 0.
    assert capacity.mean > 0.
    assert out['F'][0, i_capacity] == pytest.approx(-Mean().scalarize(capacity))
    assert out['F'][0, i_mass] == pytest.approx(7.8*3.*1.5)

    # the constraint is 'stress <= 60', so g = stress - 60
    assert out['G'][0, 0] == pytest.approx(Mean().scalarize(out['g_stochastic'][0, 0]) - 60.)


def test_problem_evaluation_and_statistics(beam):
    problem = beam.get_problem()
    out = problem.evaluate(np.array([[0, 2.], [1, 4.]]), return_as_dictionary=True)

    assert out['F'].shape == (2, 3)
    assert np.all(np.isfinite(out['F']))

    assert out['f_stochastic'].shape == (2, 3)

    # Realizations really reach the model
    deflection = out['f_stochastic'][0, 1]
    assert isinstance(deflection, StochasticOutput)
    assert deflection.std > 0.
    assert out['F'][0, 1] == pytest.approx(Mean().scalarize(deflection))


def test_problem_scalars_take_effect():
    evaluator = BeamStochasticEvaluator(stress_ref=60.,
                                        obj_scalar=[Mean(), Margin(k=2.), Mean()],
                                        constr_scalar=[Quantile(q=.9)])
    problem = evaluator.get_problem()
    out = problem.evaluate(np.array([[0, 3.]]), return_as_dictionary=True)
    deflection, stress = out['f_stochastic'][0, 1], out['g_stochastic'][0, 0]

    # objectives are ordered by name: capacity, deflection, mass
    assert out['F'][0, 1] == pytest.approx(Margin(k=2.).scalarize(deflection))
    assert out['F'][0, 1] > deflection.mean

    # Regression: the constraint was scalarized with the objective's scalar
    assert out['G'][0, 0] == pytest.approx(Quantile(q=.9).scalarize(stress) - 60.)
    assert out['G'][0, 0] != pytest.approx(Mean().scalarize(stress) - 60.)


def test_problem_uses_common_random_numbers_and_one_graph_per_point(beam):
    problem = beam.get_problem()
    n_calls, original = [0], beam.get_graph

    def _counting(*args, **kwargs):
        n_calls[0] += 1
        return original(*args, **kwargs)

    beam.get_graph = _counting
    problem.evaluate(np.array([[0, 2.], [0, 2.]]), return_as_dictionary=True)

    assert n_calls[0] == 2  # the architecture does not depend on the realization
    first, second = beam.seen_loads[:20], beam.seen_loads[20:]
    assert first == second  # both identical design points saw the same realizations


@pytest.mark.parametrize('parallel_processes', [True, False])
def test_problem_evaluates_in_parallel(parallel_processes):
    x = np.array([[0, 2.], [1, 4.], [0, 3.], [1, 2.5]])
    f_serial = BeamStochasticEvaluator().get_problem().evaluate(x, return_as_dictionary=True)['F']

    problem = BeamStochasticEvaluator().get_problem(n_parallel=2, parallel_processes=parallel_processes)
    out = problem.evaluate(x, return_as_dictionary=True)

    assert problem.get_n_batch_evaluate() == 2
    assert np.all(np.isfinite(out['F']))
    assert out['F'] == pytest.approx(f_serial)  # workers see the same realizations as the serial loop


def test_problem_with_polynomial_chaos():
    evaluator = BeamStochasticEvaluator(uq_method=PolynomialChaos(40, seed=42, degree=2))
    out = evaluator.get_problem().evaluate(np.array([[0, 2.]]), return_as_dictionary=True)

    assert np.all(np.isfinite(out['F']))

    # Statistics come from the fitted expansion rather than from the 40 expensive evaluations, and the expansion
    # itself is carried on the output, so Sobol indices are available downstream
    chaos_result = out['f_stochastic'][0, 1].method_results
    assert isinstance(chaos_result, ot.FunctionalChaosResult)
    assert ot.FunctionalChaosSobolIndices(chaos_result).getSobolTotalIndex(0) >= 0.


def test_uav_example():
    from adsg_core.examples.robust_uav import RobustUAVStochasticEvaluator

    evaluator = RobustUAVStochasticEvaluator(MonteCarlo(25, seed=42), k=2.)
    assert set(evaluator.param_space.parameter_names) == {'bsfc', 'drag_factor', 'eta_bat', 'headwind'}
    assert not evaluator.par_payload.is_stochastic

    problem = evaluator.get_problem()
    assert problem.n_obj == 2

    x = np.array([evaluator.get_random_design_vector() for _ in range(4)], dtype=float)
    out = problem.evaluate(x, return_as_dictionary=True)

    assert out['F'].shape == (4, 2)
    assert np.all(np.isfinite(out['F']))

    # endurance is maximized, so it is stored negated while the graph keeps the physical value
    assert np.all(out['F'][:, 0] < 0.)
    endurance = out['f_stochastic'][0, 0]
    assert endurance.mean > 0.
    assert endurance.std > 0.

    # The mass of the UAV does not depend on any of the uncertain parameters, so it is a plain value
    mass = out['f_stochastic'][0, 1]
    assert isinstance(mass, float)
    assert out['F'][0, 1] == pytest.approx(mass)


def test_uav_example_statistics_helper():
    from adsg_core.examples.robust_uav import RobustUAVStochasticEvaluator

    evaluator = RobustUAVStochasticEvaluator(MonteCarlo(25, seed=1), k=2.)
    dsg, _, _ = evaluator.get_graph(evaluator.get_random_design_vector())
    statistics = evaluator.evaluate_statistics(dsg)

    assert statistics['endurance_mean'] > 0.
    assert statistics['endurance_std'] > 0.
    assert statistics['endurance_robust'] < statistics['endurance_mean']
    assert statistics['mass'] > 0.