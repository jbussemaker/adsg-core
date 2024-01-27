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
import itertools
from typing import *
from adsg_core.graph.adsg_basic import *
from adsg_core.graph.adsg_nodes import *
from adsg_core.optimization.evaluator import ADSGEvaluator

__all__ = ['GNCEvaluator']


class GNCInstanceNode(NamedNode):
    """Custom node that represents the instance of some object"""

    def __init__(self, base_name: str, idx: int):
        self.idx = idx
        super().__init__(base_name)

    def get_export_title(self) -> str:
        return f'{self.name}[{self.idx}]'

    def __str__(self):
        return f'[{self.name} {self.idx}]'


class GNCTypeNode(NamedNode):
    """Custom node that represents the type of some object"""

    def __init__(self, name: str, type_: str):
        self.type = type_
        super().__init__(name)

    def get_export_title(self) -> str:
        return f'{self.name}: {self.type}'

    def __str__(self):
        return f'[{self.type}]'


class GNCEvaluator(ADSGEvaluator):
    """
    Class representing the GN&C (Guidance, Navigation and Control) problem presented in (Chapter 15):
    Crawley 2015: System Architecture - Strategy and Product Development for Complex Systems

    Component mass and probabilities are estimated, as they are not given in the text.

    The problem consists of the connection of a set of sensors to a set of flight computers:
    - System-level mass (minimization) and failure rate (maximization) are optimized
    - The number of sensors and computers can be chosen (1, 2, or 3) --> more reduces failure rate but increases mass
    - The types of sensors and computers can be chosen (A, B, or C) --> each has a different mass/failure rate trade-off
    - Connections between sensors and computers are established
      - Any sensor can connect to any computer
      - Each sensor and computer need at least one connection (otherwise their existence is useless)
    - Sensor/computer type selections are constrained to be non-repeatable
      - E.g. sensor types can be AA, AB, AC, BB, BC, CC (no permutations of these)
      - This is to prevent including isomorphic architectures, architectures where sensors and/or computers, including
        their connections, are permutations of each other and therefor have the same objective values
    """

    mass = {
        'C': {'A': 5., 'B': 10., 'C': 15.},
        'S': {'A': 1., 'B': 2., 'C': 6.},
    }
    failure_rate = {
        'C': {'A': .004, 'B': .0025, 'C': .001},
        'S': {'A': .001, 'B': .0005, 'C': .0001},
    }

    def __init__(self, objective: int = None):
        super().__init__(self.get_adsg(objective=objective))

    @staticmethod
    def get_adsg(objective: int = None):
        metric_nodes = []
        if objective is None or objective == 0:
            metric_nodes.append(MetricNode('mass', direction=-1, type_=MetricType.OBJECTIVE))
        if objective is None or objective == 1:
            metric_nodes.append(MetricNode('failureRate', direction=1, type_=MetricType.OBJECTIVE))
        if len(metric_nodes) == 0:
            raise ValueError('No objectives specified!')

        adsg = BasicADSG()

        # Add top-level node and metrics
        gnc = NamedNode('GNC')
        adsg.add_edges([(gnc, mn) for mn in metric_nodes])

        # Add sensor choices
        sensor = NamedNode('Sensor')
        adsg.add_edge(gnc, sensor)

        sensor_inst_nodes = [GNCInstanceNode('Inst', i) for i in range(3)]
        adsg.add_selection_choice('S', sensor, sensor_inst_nodes)

        sensor_type_nodes = [GNCTypeNode('Type', type_) for type_ in ['A', 'B', 'C']]

        sensor_type_choices = []
        sensor_conn_nodes = []
        for i, si_node in enumerate(sensor_inst_nodes):
            # Type selection
            sensor_type_choices.append(adsg.add_selection_choice(f'ST{i}', si_node, sensor_type_nodes))

            # Connection node, require at least one connection
            conn_node = ConnectorNode(f'SC{i}', deg_spec='+')
            sensor_conn_nodes.append(conn_node)
            adsg.add_edge(si_node, conn_node)

            if i > 0:  # Select previous instances
                adsg.add_edge(si_node, sensor_inst_nodes[i-1])

        # Add computer choices
        computer = NamedNode('Comp')
        adsg.add_edge(gnc, computer)

        comp_inst_nodes = [GNCInstanceNode('Inst', i) for i in range(3)]
        adsg.add_selection_choice('C', computer, comp_inst_nodes)

        comp_type_nodes = [GNCTypeNode('Type', type_) for type_ in ['A', 'B', 'C']]

        comp_type_choices = []
        comp_conn_nodes = []
        for i, ci_node in enumerate(comp_inst_nodes):
            # Type selection
            comp_type_choices.append(adsg.add_selection_choice(f'CT{i}', ci_node, comp_type_nodes))

            # Connection node, require at least one connection
            conn_node = ConnectorNode(f'CC{i}', deg_spec='+')
            comp_conn_nodes.append(conn_node)
            adsg.add_edge(ci_node, conn_node)

            if i > 0:  # Select previous instances
                adsg.add_edge(ci_node, comp_inst_nodes[i-1])

        # Add connection choice
        adsg.add_connection_choice('Conn', sensor_conn_nodes, comp_conn_nodes)

        # Set start node
        adsg = adsg.set_start_nodes({gnc})

        # Constraint sensor/computer type selections
        adsg = adsg.constrain_choices(ChoiceConstraintType.UNORDERED, comp_type_choices)
        adsg = adsg.constrain_choices(ChoiceConstraintType.UNORDERED, sensor_type_choices)

        return adsg

    def _evaluate(self, adsg: ADSGType, metric_nodes: List[MetricNode]) -> Dict[MetricNode, float]:
        # Analyze the architecture
        sensors, computers, conns = self._analyze_arch(adsg)

        # Calculate metrics: the outputs of this function should be a dict mapping requested metric nodes to values
        results = {}
        for metric_node in metric_nodes:
            if metric_node.name == 'mass':
                results[metric_node] = self.calc_mass(sensors, computers)
            elif metric_node.name == 'failureRate':
                results[metric_node] = self.calc_failure_rate(sensors, computers, conns)

        return results

    @staticmethod
    def _analyze_arch(adsg: BasicADSG) -> Tuple[list, list, list]:

        def _analyze_object(object_root_node):
            obj_types = []
            connector_nodes = {}

            # Find object instance nodes
            inst_nodes = []
            for obj_inst_node in adsg.derived_nodes(object_root_node):
                if isinstance(obj_inst_node, GNCInstanceNode):
                    inst_nodes.append(obj_inst_node)

            # Loop over sorted instance nodes
            for i, obj_inst_node in enumerate(sorted(inst_nodes, key=lambda n: n.idx)):
                for next_node in adsg.next(obj_inst_node):
                    # Record selected type
                    if isinstance(next_node, GNCTypeNode):
                        obj_types.append(next_node.type)

                    # Record connector nodes
                    elif isinstance(next_node, ConnectorNode):
                        connector_nodes[next_node] = i

            return obj_types, connector_nodes

        # Analyze object types
        sensors, sensor_conns = [], {}
        computers, comp_conns = [], {}
        for node in adsg.graph.nodes:
            if isinstance(node, NamedNode):
                if node.name == 'Sensor':
                    sensors, sensor_conns = _analyze_object(node)
                elif node.name == 'Comp':
                    computers, comp_conns = _analyze_object(node)

        # Get object connections
        conns = []
        for src_node, src_idx in sensor_conns.items():
            for tgt_node in adsg.next(src_node, edge_type=EdgeType.CONNECTS):
                tgt_idx = comp_conns[tgt_node]
                conns.append((src_idx, tgt_idx))

        return sensors, computers, conns

    @classmethod
    def calc_mass(cls, sensors, computers):
        sensor_mass = sum([cls.mass['S'][type_] for type_ in sensors])
        computer_mass = sum([cls.mass['C'][type_] for type_ in computers])
        return sensor_mass+computer_mass

    @classmethod
    def calc_failure_rate(cls, sensors, computers, conns):

        def system_fails(sensor_failed, computer_failed):
            # Find remaining connections for the failed sensors and computers
            remaining_conns = [conn for conn in conns if not sensor_failed[conn[0]] and not computer_failed[conn[1]]]

            # If there are no remaining connections, the system has failed
            return len(remaining_conns) == 0

        # Get item failure rates
        rate = cls.failure_rate
        failure_rates = [rate['S'][type_] for type_ in sensors]+[rate['C'][type_] for type_ in computers]

        # Loop over number of failures
        failure_rate = 0.
        n_s = len(sensors)
        n_c = len(computers)
        n_obj = n_s+n_c
        for n_failed in range(n_obj):
            base_failures = [i < n_failed+1 for i in range(n_obj)]

            # Iterate over possible failure permutations
            for failure_scheme in set(itertools.permutations(base_failures)):
                s_failed = failure_scheme[:n_s]
                c_failed = failure_scheme[n_s:]

                # Check if system fails
                if system_fails(s_failed, c_failed):
                    # Determine probability of this state happening
                    prob = 1.
                    for i, f_rate in enumerate(failure_rates):
                        if failure_scheme[i]:
                            prob *= f_rate

                    failure_rate += prob

        return -math.log10(failure_rate)


if __name__ == '__main__':
    # After export, visualize contents using https://viz-js.com/
    GNCEvaluator.get_adsg().export_dot('gnc.dot')

    evaluator = GNCEvaluator()
    for _ in range(10):
        graph, _, _ = evaluator.get_graph(evaluator.get_random_design_vector())
        evaluator.evaluate(graph)
