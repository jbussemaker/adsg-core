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
import math
from typing import *
from adsg_core.graph.adsg_basic import *
from adsg_core.graph.adsg_nodes import *
from adsg_core.optimization.evaluator import ADSGEvaluator

__all__ = ['ApolloEvaluator']


class ApolloDecisionVar(NamedNode):

    def __init__(self, decision: str, value):
        self.decision = decision
        self.value = value
        super().__init__(f'{decision}:{value!s}')

    def get_export_title(self) -> str:
        return f'{self.decision} = {self.value}'


class ApolloEvaluator(ADSGEvaluator):
    """
    Class representing the Apollo system architecture design problem, as presented by:
    Simmons 2008: A Framework for Decision Support in Systems Architecting

    The problem involves selection of the Apollo moon mission and vehicle architectures:
    - Mass (minimization) and success probability (maximization) are optimized
    - Mission architecture is selected:
      - Earth Orbit Rendezvous (yes/no)
      - Earth launch (orbit/direct)
      - Lunar Orbit Rendezvous (yes/no)
      - Moon arrival (orbit/direct)
      - Moon departure (orbit/direct)
    - Vehicle architecture is selected:
      - Command module crew (2/3)
      - Lunar module crew (0/1/2/3)
      - Service module fuel (cryogenic/storable)
      - Lunar module fuel (NA/cryogenic/storable)
    """

    risk_table = {
        'EOR': {False: .98, True: .95},
        'earthLaunch': {'orbit': .99, 'direct': .9},
        'LOR': {False: 1., True: .95},
        'moonArrival': {'orbit': .99, 'direct': .95},
        'moonDeparture': {'orbit': .9, 'direct': .9},
        'cmCrew': {2: 1., 3: 1.},
        'lmCrew': {0: 1., 1: .9, 2: 1., 3: 1.},
        'smFuel': {'cryogenic': .95, 'storable': 1.},
        'lmFuel': {'NA': 1., 'cryogenic': .9025, 'storable': 1.},
    }

    def __init__(self, objective: int = None):
        super().__init__(self.get_adsg(objective=objective))

    @staticmethod
    def get_adsg(objective: int = None):
        metric_nodes = []
        if objective is None or objective == 0:
            metric_nodes.append(MetricNode('mass', direction=-1, type_=MetricType.OBJECTIVE))
        if objective is None or objective == 1:
            metric_nodes.append(MetricNode('success', direction=1, type_=MetricType.OBJECTIVE))
        if len(metric_nodes) == 0:
            raise ValueError('No objectives specified!')

        adsg = BasicADSG()
        start_nodes = set()

        # EARTH LAUNCH DECISION #
        # If we choose EOR (Earth orbit rendezvous), we have to launch to orbit
        # We represent this constraint by first adding the EOR yes/no decision,
        # and then adding launch decisions after the no-EOR node
        earth_launch = NamedNode('earthLaunch')
        adsg.add_edges([(earth_launch, mn) for mn in metric_nodes])
        start_nodes.add(earth_launch)

        eor_yes = ApolloDecisionVar('EOR', True)
        eor_no = ApolloDecisionVar('EOR', False)
        earth_launch_orbit = ApolloDecisionVar('earthLaunch', 'orbit')
        earth_launch_direct = ApolloDecisionVar('earthLaunch', 'direct')

        adsg.add_selection_choice('EOR', earth_launch, [eor_no, eor_yes])
        adsg.add_selection_choice('earthLaunch', eor_no, [earth_launch_orbit, earth_launch_direct])
        adsg.add_edge(eor_yes, earth_launch_orbit)

        # MOON ARRIVAL/DEPARTURE DECISIONS #
        # If we choose LOR, we have to arrive and depart by orbit
        # We represent this constraint by first adding the LOR yes/no decision, and then adding arrival/departure
        # decisions after the no-LOR node
        moon_arr_dep = NamedNode('moonArrDep')
        start_nodes.add(moon_arr_dep)

        lor_yes = ApolloDecisionVar('LOR', True)
        lor_no = ApolloDecisionVar('LOR', False)
        moon_arr_orbit = ApolloDecisionVar('moonArrival', 'orbit')
        moon_arr_direct = ApolloDecisionVar('moonArrival', 'direct')
        moon_dep_orbit = ApolloDecisionVar('moonDeparture', 'orbit')
        moon_dep_direct = ApolloDecisionVar('moonDeparture', 'direct')

        adsg.add_selection_choice('moonLOR', moon_arr_dep, [lor_no, lor_yes])
        adsg.add_selection_choice('moonArr', lor_no, [moon_arr_orbit, moon_arr_direct])
        adsg.add_selection_choice('moonDep', lor_no, [moon_dep_orbit, moon_dep_direct])
        adsg.add_edges([
            (lor_yes, moon_arr_orbit),
            (lor_yes, moon_dep_orbit),
        ])

        # CREW DECISION #
        # Select either 2 or 3 crew in the command module (aka the mission)
        crew = NamedNode('crew')
        start_nodes.add(crew)

        crew_2 = ApolloDecisionVar('cmCrew', 2)
        crew_3 = ApolloDecisionVar('cmCrew', 3)
        adsg.add_selection_choice('crew', crew, [crew_2, crew_3])

        # SERVICE MODULE FUEL DECISION #
        # Select fuel type for the service module (connected to the command module)
        service_fuel = NamedNode('smFuel')
        start_nodes.add(service_fuel)

        sm_fuel_cryogenic = ApolloDecisionVar('smFuel', 'cryogenic')
        sm_fuel_storable = ApolloDecisionVar('smFuel', 'storable')
        adsg.add_selection_choice('smFuel', service_fuel, [sm_fuel_cryogenic, sm_fuel_storable])

        # LUNAR MODULE DECISIONS #
        # Whether there is a lunar module is determined by whether there is a LOR
        # If yes, we select the nr of crew members and the fuel type
        # If no, we select 0 crew members and "NA" fuel type
        adsg.add_edge(lor_no, ApolloDecisionVar('lmCrew', 0))
        lm_crew = [ApolloDecisionVar('lmCrew', n) for n in [1, 2, 3]]
        adsg.add_selection_choice('lmCrew', lor_yes, lm_crew)

        adsg.add_edge(lor_no, ApolloDecisionVar('lmFuel', 'NA'))
        adsg.add_selection_choice('lmFuel', lor_yes, [
            ApolloDecisionVar('lmFuel', 'cryogenic'), ApolloDecisionVar('lmFuel', 'storable')])

        # Constrain that if we select 2 crew members, we cannot select 3 lunar module crew members
        adsg.add_incompatibility_constraint([crew_2, lm_crew[-1]])

        # Set start nodes
        adsg = adsg.set_start_nodes(start_nodes)

        return adsg

    def _evaluate(self, adsg: ADSGType, metric_nodes: List[MetricNode]) -> Dict[MetricNode, float]:
        # Get decision values from selected ApolloDecisionVar nodes
        decision_values = {
            'EOR': False,
            'earthLaunch': 'direct',
            'LOR': False,
            'moonArrival': 'direct',
            'moonDeparture': 'direct',
            'lmCrew': 0,
            'lmFuel': 'NA',
        }
        dv_node: ApolloDecisionVar
        for dv_node in adsg.get_nodes_by_type(ApolloDecisionVar):
            decision_values[dv_node.decision] = dv_node.value

        # Check that no constraint is violated
        self.is_possible(decision_values)

        # Calculate metrics: the outputs of this function should be a dict mapping requested metric nodes to values
        results = {}
        for metric_node in metric_nodes:
            if metric_node.name == 'mass':
                results[metric_node] = self.eval_mass(decision_values)
            elif metric_node.name == 'success':
                results[metric_node] = self.eval_success_probability(decision_values)

        return results

    @staticmethod
    def is_possible(dv):  # Table 4-3 (p.104)

        if dv['EOR'] and dv['earthLaunch'] != 'orbit':  # EORconstraint
            raise RuntimeError('EOR constraint violated!')

        if dv['LOR'] and (dv['moonArrival'] != 'orbit' or dv['moonDeparture'] != 'orbit'):  # LORconstraint, moonLeaving
            raise RuntimeError('LOR constraint violated!')

        if dv['cmCrew'] < dv['lmCrew']:  # lmcmcrew
            raise RuntimeError('Crew constraint violated!')

        if dv['LOR'] == (dv['lmCrew'] == 0):  # lmexists
            raise RuntimeError('Lunar module constraint violated!')

        if dv['LOR'] == (dv['lmFuel'] == 'NA'):  # lmFuelConstraint
            raise RuntimeError('Fuel constraint violated!')

    @classmethod
    def eval_success_probability(cls, dv):
        prob = 1.
        for key, value in dv.items():

            # Special case for cryogenic service module fuel
            if key == 'smFuel' and value == 'cryogenic':
                n_burn = 2. if dv['LOR'] else 4.
                prob *= math.pow(cls.risk_table[key][value], n_burn)
                continue

            try:
                prob *= cls.risk_table[key][value]
            except KeyError:
                raise KeyError('Risk table key not found: %s, %r' % (key, value))
        return prob

    @staticmethod
    def eval_mass(dv):
        """
        Based on the Jython script provided in Appendix A of:
        Simmons2008 A Framework for Decision Support in Systems Architecting

        :param dv:
        :rtype: float
        """

        d_veod = 11100
        d_vmoe = 3840
        d_vdom = 6798
        d_vmoa = 7468
        dv_mod = 3661
        g = 9.8 * 100 / 30
        i_sp1 = 315
        i_sp2 = 425
        f = 0.08

        def cm_mass(cm_crew):
            if cm_crew == 2:
                return 8000
            if cm_crew == 3:
                return 11000
            return 0

        def lm_mass(lm_crew):
            if lm_crew == 1:
                return 3000
            if lm_crew == 2:
                return 4000
            if lm_crew == 3:
                return 5000
            return 0

        def sm_4_mass(m_cm, i_sp):
            return stage_mass(dv_mod, i_sp, m_cm)

        def lm_2_mass(lor, m_lm, i_sp):
            if not lor:
                return 0

            return stage_mass(d_vmoa, i_sp, m_lm)

        def lm_1_mass(lor, m_lm2, m_lm, i_sp):
            if not lor:
                return 0

            m_payload = m_lm2 + m_lm
            return stage_mass(d_vdom, i_sp, m_payload)

        def sm_3_mass(lor, m_sm4, m_cm, i_sp):
            if lor:
                return 0

            m_payload = m_sm4 + m_cm
            return stage_mass(d_vmoa, i_sp, m_payload)

        def sm_2_mass(lor, m_sm3, m_sm4, m_cm, i_sp):
            if lor:
                return 0

            m_payload = m_sm3 + m_sm4 + m_cm
            return stage_mass(d_vdom, i_sp, m_payload)

        def sm_1_mass(m_lm_total, m_sm2, m_sm3, m_sm4, m_cm, i_sp):
            m_payload = m_lm_total + m_sm2 + m_sm3 + m_sm4 + m_cm
            return stage_mass(d_vmoe, i_sp, m_payload)

        def s_4_mass(m_payload):
            return stage_mass(d_veod, i_sp2, m_payload)

        def r_calc(delta_v, i_sp):
            x = (delta_v / (i_sp * g))
            ratio = math.exp(-x)
            return ratio

        def rt2p(delta_v, i_sp):
            r = r_calc(delta_v, i_sp)
            return 1 / (r - f + (f * r))

        def stage_mass(delta_v, i_sp, m_payload):
            total_payload_ratio = rt2p(delta_v, i_sp)
            return total_payload_ratio*m_payload - m_payload

        def calc_i_sp(fuel_type):
            return i_sp2 if fuel_type == 'cryogenic' else i_sp1

        def im_leo(lor, cm_crew, lm_crew, sm_fuel_type, lm_fuel_type):
            m_cm = cm_mass(cm_crew)
            m_sm4 = sm_4_mass(m_cm, calc_i_sp(sm_fuel_type))

            m_lm = lm_mass(lm_crew)
            m_lm2 = lm_2_mass(lor, m_lm, calc_i_sp(lm_fuel_type))
            m_lm1 = lm_1_mass(lor, m_lm2, m_lm, calc_i_sp(lm_fuel_type))
            m_lm_total = sum([m_lm1, m_lm2, m_lm])

            m_sm3 = sm_3_mass(lor, m_sm4, m_cm, calc_i_sp(sm_fuel_type))
            m_sm2 = sm_2_mass(lor, m_sm3, m_sm4, m_cm, calc_i_sp(sm_fuel_type))
            m_sm1 = sm_1_mass(m_lm_total, m_sm2, m_sm3, m_sm4, m_cm, calc_i_sp(sm_fuel_type))

            m_total = sum([m_sm1, m_sm2, m_sm3, m_sm4])

            return sum([
                s_4_mass(sum([m_total, m_lm_total, m_cm])),
                m_total,
                m_lm_total,
                m_cm,
            ])

        return im_leo(dv['LOR'], dv['cmCrew'], dv['lmCrew'], dv['smFuel'], dv['lmFuel'])


if __name__ == '__main__':
    # After export, visualize contents using https://viz-js.com/
    ApolloEvaluator.get_adsg().export_dot('apollo.dot')

    evaluator = ApolloEvaluator()
    for _ in range(10):
        graph, _, _ = evaluator.get_graph(evaluator.get_random_design_vector())
        evaluator.evaluate(graph)
