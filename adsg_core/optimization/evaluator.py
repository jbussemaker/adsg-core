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
from adsg_core.graph.adsg import ADSGType
from adsg_core.graph.adsg_nodes import MetricNode
from adsg_core.optimization.dv_output_defs import *
from adsg_core.optimization.graph_processor import *

__all__ = ['ADSGEvaluator']


class ADSGEvaluator(GraphProcessor):
    """
    Base class for implementing an evaluator that directly evaluates ADSG instances.
    Override _evaluate to implement the evaluation.

    Extends `GraphProcessor`, so all its functions are also available.
    """

    def get_problem(self, n_parallel=None, parallel_processes=True):
        """Get an SBArchOpt problem instance."""
        from adsg_core.optimization.problem import ADSGArchOptProblem
        return ADSGArchOptProblem(self, n_parallel=n_parallel, parallel_processes=parallel_processes)

    def _choose_metric_type(self, objective: Objective, constraint: Constraint) -> Union[Objective, Constraint]:
        raise RuntimeError(f'Metric {objective.name} can either be an objective or a constraint! '
                           f'Specify the metric type using node.type = MetricType.x')

    def evaluate(self, adsg: ADSGType) -> Tuple[List[float], List[float]]:
        """
        Evaluate an ADSG instance. Returns a list of objective values and a list of constraint values.
        """

        # Evaluate the ADSG instance
        metric_nodes = adsg.get_nodes_by_type(MetricNode)
        value_map = self._evaluate(adsg, metric_nodes)

        # Associate values to objectives
        objective_values = [value_map.get(objective.node, math.nan) for objective in self.objectives]

        # Associate values to constraints: if the constraint does not exist in this ADSG instance, set value to ref
        constraint_values = [value_map.get(constraint.node, math.nan)
                             if constraint.node in metric_nodes else constraint.ref
                             for constraint in self.constraints]

        return objective_values, constraint_values

    def _evaluate(self, adsg: ADSGType, metric_nodes: List[MetricNode]) -> Dict[MetricNode, float]:
        """
        Implement this function to provide ADSG evaluation.
        Should return a mapping from metric node to float (NaN is allowed).
        """
        raise NotImplementedError
