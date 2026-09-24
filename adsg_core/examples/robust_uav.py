"""
MIT License

Copyright: (c) 2024, Deutsches Zentrum fuer Luft- und Raumfahrt e.V.
Contact: jasper.bussemaker@dlr.de

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import numpy as np
import openturns as ot
from typing import *
from adsg_core.graph.adsg_basic import *
from adsg_core.graph.adsg_nodes import *
from adsg_core.optimization.stochastic_evaluator import DSGStochasticEvaluator
from sb_arch_opt.uncertainty import UQMethod, Mean, Margin, PolynomialChaos
from sb_arch_opt.algo.pymoo_interface import plot

__all__ = ['RobustUAVStochasticEvaluator', 'UAVOptionNode', 'run_sbo']

GRAVITY = 9.81
RHO_SL = 1.225  # Sea-level air density [kg/m3]


class UAVOptionNode(NamedNode):
    """Custom node representing one option of an architectural decision"""

    def __init__(self, decision: str, value):
        self.decision = decision
        self.value = value
        super().__init__(f'{decision}:{value!s}')

    def get_export_title(self) -> str:
        return f'{self.decision} = {self.value}'


class RobustUAVStochasticEvaluator(DSGStochasticEvaluator):
    """
    Robust design of a multirotor UAV, as an example of architecture optimization under uncertainty.

    Design variables (13 total, 9 active in any one architecture):

    | Variable       | Type        | Active when          |
    |----------------|-------------|----------------------|
    | `mission`      | categorical | always               |
    | `airframe`     | categorical | always               |
    | `n_rotors`     | ordinal     | always               |
    | `powertrain`   | categorical | always               |
    | `n_cells`      | ordinal     | powertrain=electric  |
    | `cell_chem`    | categorical | powertrain=electric  |
    | `bat_frac`     | continuous  | powertrain=electric  |
    | `engine`       | categorical | powertrain=hybrid    |
    | `n_gen`        | ordinal     | powertrain=hybrid    |
    | `fuel_frac`    | continuous  | powertrain=hybrid    |
    | `rotor_radius` | continuous  | always               |
    | `cruise_speed` | continuous  | always               |
    | `cruise_alt`   | continuous  | always               |

    That gives 120 valid discrete architectures spanning a continuous design space on top.

    Uncertain parameters (few, and themselves hierarchical):

    - `payload`: payload mass [kg], always present
    - `headwind`: cruise headwind [m/s], always present
    - `drag_factor`: airframe drag scatter [-], always present
    - `eta_bat`: battery + motor efficiency [-], only in the electric architecture
    - `bsfc`: brake specific fuel consumption [kg/kWh], only in the hybrid architecture

    Metrics:

    - `endurance` [min]: maximized, stochastic, reduced with `Margin(k=-k)`, i.e. `mean - k*std` (the sign is
      negative because the scalar is applied to the physical samples of a *maximized* quantity)
    - `mass` [kg]: minimized, evaluated at the mean payload so it has no scatter of its own

    Ensure the optional dependencies are installed: `pip install sb-arch-opt[arch_sbo]`
    """

    # Mission -> payload scaling [-] and minimum useful cruise speed [m/s]
    mission_options = ['survey', 'delivery']
    mission_payload_factor = {'survey': 1., 'delivery': 1.8}

    # Airframe -> structural mass [kg] and drag scaling [-]
    airframe_options = ['carbon', 'aluminium']
    airframe_mass = {'carbon': 2.6, 'aluminium': 3.6}
    airframe_drag = {'carbon': 1., 'aluminium': 1.08}

    rotor_options = [4, 6, 8]

    # Battery: cells in series scale the usable specific energy, chemistry sets its base level [Wh/kg]
    cell_options = [3, 4, 6]
    cell_factor = {3: .92, 4: 1., 6: 1.06}
    chem_options = ['lipo', 'liion']
    chem_specific_energy = {'lipo': 180., 'liion': 240.}
    chem_mass_penalty = {'lipo': 0., 'liion': .4}  # Liion needs more protection circuitry [kg]

    # Engine -> installed mass [kg] and fuel consumption scaling [-]
    # The turbine is lighter but thirstier: a genuine categorical trade-off
    engine_options = ['piston', 'turbine']
    engine_mass = {'piston': 2.4, 'turbine': 1.6}
    engine_bsfc_factor = {'piston': 1., 'turbine': 1.35}

    # Generator count -> mass [kg] and combined engine+generator+motor chain efficiency [-]
    gen_options = [1, 2]
    gen_mass = {1: .8, 2: 1.7}
    gen_chain_efficiency = {1: .18, 2: .195}

    avionics_mass = .6  # [kg]
    rotor_mass_factor = 1.5  # Rotor + motor mass per rotor, per m of radius [kg/m]
    figure_of_merit = .72  # Rotor hover efficiency [-]
    energy_mass_ref = 8.  # Reference mass the energy fractions apply to [kg]

    def __init__(self, uq_method: UQMethod, k: float = 2., objective: int = None):
        """
        :param k: margin factor; the robust endurance is `mean - k*std`
        :param objective: 0 for endurance only, 1 for mass only, None for both
        """
        self.k = k
        self.uq_method = uq_method

        # Uncertain parameters: three always present, two conditional on the selected powertrain. The nodes are
        # identities only - the distributions are attached to the graph in get_dsg().
        self.par_payload = InputParameterNode('payload', 2.0)
        self.par_headwind = InputParameterNode('headwind', ot.Normal(10., 2.))
        self.par_drag = InputParameterNode('drag_factor', ot.Normal(1., .08))
        self.par_eta_bat = InputParameterNode('eta_bat', ot.Normal(.92, .03))
        self.par_bsfc = InputParameterNode('bsfc', ot.Normal(.42, .075))

        self.metric_node_map: Dict[str, MetricNode] = {}
        self.option_nodes: Dict[str, List[UAVOptionNode]] = {}

        obj_scalar = [Margin(k=self.k, direction=1), Mean()]

        super().__init__(self.get_dsg(objective=objective), uq_method=uq_method, obj_scalar=obj_scalar)

    def _add_choice(self, dsg: BasicDSG, decision: str, originating_node: DSGNode, values: list,
                    is_ordinal: bool = False):
        """Add a selection choice whose options are UAVOptionNodes carrying the decision value"""
        option_nodes = [UAVOptionNode(decision, value) for value in values]
        self.option_nodes[decision] = option_nodes
        dsg.add_selection_choice(decision, originating_node, option_nodes, is_ordinal=is_ordinal)
        return option_nodes

    def get_dsg(self, objective: int = None) -> DSGType:
        metric_nodes = []
        if objective is None or objective == 0:
            # Maximized, and the metric the uncertainty propagation feeds
            self.metric_node_map['endurance'] = MetricNode(
                'endurance', direction=1, type_=MetricType.OBJECTIVE)
            metric_nodes.append(self.metric_node_map['endurance'])

        if objective is None or objective == 1:
            # Minimized, evaluated at the mean payload so it carries no scatter
            self.metric_node_map['mass'] = MetricNode('mass', direction=-1, type_=MetricType.OBJECTIVE)
            metric_nodes.append(self.metric_node_map['mass'])

        if len(metric_nodes) == 0:
            raise ValueError('No objectives specified!')

        dsg = BasicDSG()

        uav = NamedNode('UAV')
        dsg.add_edges([(uav, mn) for mn in metric_nodes])

        # Always-present uncertain parameters and continuous design variables
        dsg.add_edges([
            (uav, self.par_payload),
            (uav, self.par_headwind),
            (uav, self.par_drag),
            (uav, DesignVariableNode('rotor_radius', bounds=(.12, .30))),
            (uav, DesignVariableNode('cruise_speed', bounds=(8., 26.))),
            (uav, DesignVariableNode('cruise_alt', bounds=(0., 2000.))),
        ])

        # ALWAYS-ACTIVE DISCRETE CHOICES
        mission = NamedNode('mission')
        airframe = NamedNode('airframe')
        rotors = NamedNode('rotors')
        dsg.add_edges([(uav, mission), (uav, airframe), (uav, rotors)])

        self._add_choice(dsg, 'mission', mission, self.mission_options)
        self._add_choice(dsg, 'airframe', airframe, self.airframe_options)
        self._add_choice(dsg, 'n_rotors', rotors, self.rotor_options, is_ordinal=True)

        # POWERTRAIN: categorical, and the root of the hierarchy
        powertrain = NamedNode('powertrain')
        dsg.add_edge(uav, powertrain)
        electric, hybrid = self._add_choice(dsg, 'powertrain', powertrain, ['electric', 'hybrid'])

        # ELECTRIC BRANCH
        self._add_choice(dsg, 'n_cells', electric, self.cell_options, is_ordinal=True)
        self._add_choice(dsg, 'cell_chem', electric, self.chem_options)
        dsg.add_edges([
            (electric, DesignVariableNode('bat_frac', bounds=(.20, .60))),
            (electric, self.par_eta_bat),
        ])

        # HYBRID BRANCH
        self._add_choice(dsg, 'engine', hybrid, self.engine_options)
        self._add_choice(dsg, 'n_gen', hybrid, self.gen_options, is_ordinal=True)
        dsg.add_edges([
            (hybrid, DesignVariableNode('fuel_frac', bounds=(.10, .40))),
            (hybrid, self.par_bsfc),
        ])

        dsg = dsg.set_start_nodes({uav})

        return dsg

    """###########################
    ### ARCHITECTURE ANALYSIS ###
    ###########################"""

    @staticmethod
    def _decision_values(dsg: DSGType) -> Dict[str, Any]:
        """Read the selected option of every decision taken in this architecture"""
        return {node.decision: node.value for node in dsg.get_nodes_by_type(UAVOptionNode)}

    @staticmethod
    def _des_var_values(dsg: DSGType) -> Dict[str, float]:
        """Read the values of the continuous design variables active in this architecture"""
        return {node.name: value for node, value in dsg.des_var_values.items()}

    def _mass(self, dsg: DSGType, payload: float) -> Tuple[float, float]:
        """Total mass [kg] and the energy-carrying mass (battery or fuel) [kg]"""
        decisions = self._decision_values(dsg)
        des_vars = self._des_var_values(dsg)

        structural = (self.airframe_mass[decisions['airframe']] + self.avionics_mass +
                      decisions['n_rotors'] * self.rotor_mass_factor * des_vars['rotor_radius'])

        if decisions['powertrain'] == 'electric':
            energy_mass = des_vars['bat_frac'] * self.energy_mass_ref
            installed = energy_mass + self.chem_mass_penalty[decisions['cell_chem']]
        else:
            energy_mass = des_vars['fuel_frac'] * self.energy_mass_ref
            installed = (energy_mass + self.engine_mass[decisions['engine']] +
                         self.gen_mass[decisions['n_gen']])

        effective_payload = payload * self.mission_payload_factor[decisions['mission']]
        return structural + installed + effective_payload, energy_mass

    def _power_required(self, dsg: DSGType, mass: float, headwind: float, drag_factor: float) -> float:
        """Cruise power [W] from simple momentum theory plus a forward-flight penalty"""
        decisions = self._decision_values(dsg)
        des_vars = self._des_var_values(dsg)

        # Air density falls off with altitude (ISA troposphere)
        rho = RHO_SL * (1. - 2.2558e-5*des_vars['cruise_alt'])**4.2559

        disc_area = decisions['n_rotors'] * np.pi * des_vars['rotor_radius']**2
        hover_power = (mass*GRAVITY)**1.5 / (np.sqrt(2*rho*disc_area) * self.figure_of_merit)

        airspeed = des_vars['cruise_speed'] + headwind
        drag = drag_factor * self.airframe_drag[decisions['airframe']]
        return hover_power * (1. + .0016*drag*airspeed**2)

    def _endurance(self, dsg: DSGType, sample: Dict[InputParameterNode, float]) -> float:
        """
        Endurance [min] for one realization of the uncertain parameters.

        `sample` maps every parameter node present in this architecture to its value for this realization, so the
        branch-local parameters (`eta_bat`, `bsfc`) are only read where they exist.
        """
        decisions = self._decision_values(dsg)
        mass, energy_mass = self._mass(dsg, sample[self.par_payload])
        power = self._power_required(dsg, mass, sample[self.par_headwind], sample[self.par_drag])

        if decisions['powertrain'] == 'electric':
            # Stored energy [Wh] delivered at the sampled efficiency
            specific_energy = (self.chem_specific_energy[decisions['cell_chem']] *
                               self.cell_factor[decisions['n_cells']])
            energy = specific_energy * energy_mass * sample[self.par_eta_bat]
            return 60. * energy / power

        # Hybrid: shaft power follows from the chain efficiency, fuel burn from the sampled BSFC [kg/kWh]
        shaft_power = power / self.gen_chain_efficiency[decisions['n_gen']]
        bsfc = sample[self.par_bsfc] * self.engine_bsfc_factor[decisions['engine']]
        fuel_flow = bsfc * (shaft_power/1000.)  # [kg/h]
        return 60. * energy_mass / fuel_flow

    """#####################
    ### ROBUST EVALUATION ###
    #####################"""

    def _evaluate_sample(self, dsg: DSGType, metric_nodes: List[MetricNode]) -> Dict[MetricNode, float]:
        """Evaluate one architecture for ONE realization of the uncertain parameters"""
        results = {}
        parameters = dsg.inp_param_values
        for metric_node in metric_nodes:
            if metric_node.name == 'endurance':
                results[metric_node] = self._endurance(dsg, parameters)

            elif metric_node.name == 'mass':
                results[metric_node], _ = self._mass(dsg, parameters[self.par_payload])

        return results

    def evaluate_statistics(self, dsg: DSGType) -> Dict[str, float]:
        """Convenience helper returning the raw statistics of one architecture, using this evaluator's UQ method"""
        objective_values, _ = self.evaluate(dsg)

        by_name = {objective.name: output for objective, output in zip(self.objectives, objective_values)}
        endurance, mass = by_name['endurance'], by_name['mass']
        return {
            'endurance_mean': endurance.mean,
            'endurance_std': endurance.std,
            'endurance_robust': endurance.mean - self.k*endurance.std,
            'mass': mass,
        }


def run_sbo(uq: UQMethod, n_infill: int = 20, init_size: int = 40, k: float = 2., objective: int = None,
            seed: int = None, verbose: bool = True):
    """
    Optimize the robust UAV problem with SBArchOpt's Surrogate-Based Optimization (SBO).
    """
    from pymoo.optimize import minimize
    from sb_arch_opt.algo.arch_sbo import get_arch_sbo_gp

    if seed is not None:
        np.random.seed(seed)

    evaluator = RobustUAVStochasticEvaluator(uq, k=k, objective=objective)

    problem = evaluator.get_problem(n_parallel=4)
    problem.print_stats()

    sbo = get_arch_sbo_gp(problem, init_size=init_size)
    result = minimize(problem, sbo, termination=('n_eval', init_size+n_infill), verbose=verbose,
                      seed=seed, progress=False)

    plot(result.opt.get('F'),
         labels=f'SBO ({n_infill+init_size} evaluations)')

    # Print results
    opt = result.opt
    print('Best f:', opt.get('F')[0])
    print('Best dist:', list(opt.get('f_stochastic')[0]))
    print('Best x:', list(opt.get('X')[0]))

    if verbose:
        print('\nOptimum found:')
        x_opt = np.atleast_2d(result.X)
        f_opt = np.atleast_2d(result.F)
        for x_i, f_i in zip(x_opt, f_opt):
            # Objectives are stored for minimization, so flip the maximized ones back
            values = [-f if obj.dir.value > 0 else f for f, obj in zip(f_i, evaluator.objectives)]
            described = ', '.join(f'{obj.name}={value:.2f}' for obj, value in zip(evaluator.objectives, values))
            print(f'  x={np.array2string(x_i, precision=3)} --> {described}')

    return evaluator, problem, result


if __name__ == '__main__':
    uq = PolynomialChaos(n_evaluations=70, seed=42, degree=3, n_metamodel_samples=1000)
    evaluator = RobustUAVStochasticEvaluator(uq, k=2, objective=None)
    x = evaluator.get_random_design_vector()
    dsg, _, _ = evaluator.get_graph(x)
    result = evaluator.evaluate(dsg)
    print(result)
    dsg_all = evaluator.get_dsg()
    dsg_all.render()
    dsg.render()

    run_sbo(uq, n_infill=20, init_size=40, k=3, objective=None, seed=42, verbose=True)