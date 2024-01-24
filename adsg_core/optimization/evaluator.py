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
    """

    def get_problem(self, n_parallel=None, parallel_processes=True):
        from adsg_core.optimization.problem import ADSGArchOptProblem
        return ADSGArchOptProblem(self, n_parallel=n_parallel, parallel_processes=parallel_processes)

    def _choose_metric_type(self, objective: Objective, constraint: Constraint) -> Union[Objective, Constraint]:
        raise RuntimeError(f'Metric {objective.name} can either be an objective or a constraint! '
                           f'Specify the metric type using node.type = MetricType.x')

    def evaluate(self, adsg: ADSGType):
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
        Evaluate an ADSG instance for the provided metric nodes.
        Returns a mapping from metric node to float (NaN is allowed).
        """
        raise NotImplementedError
