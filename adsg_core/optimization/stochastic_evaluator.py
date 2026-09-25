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
import numpy as np
from adsg_core.graph.adsg import DSGType
from adsg_core.graph.adsg_nodes import MetricNode, InputParameterNode
from adsg_core.optimization.evaluator import DSGEvaluator
from sb_arch_opt.uncertainty import EvaluationOutput, Scalarization, UQMethod

__all__ = ['DSGStochasticEvaluator', 'StochasticADSGEvaluator']


class DSGStochasticEvaluator(DSGEvaluator):
    """
    Base class for implementing an evaluator for stochastic problem that directly evaluates DSG instances.
    Override _evaluate_sample to implement the evaluation.

    Inherits `DSGEvaluator` and 'GraphProcessor', so all their functions are also available.
    """

    def __init__(self,
                 *args,
                 uq_method: UQMethod,
                 obj_scalar: Optional[List[Scalarization]] = None,
                 constr_scalar: Optional[List[Scalarization]] = None,
                 **kwargs):

        self.uq_method = uq_method
        self.obj_scalar = obj_scalar
        self.constr_scalar = constr_scalar

        super().__init__(*args, **kwargs)

    def get_problem(self, n_parallel=None, parallel_processes=True):

        from adsg_core.optimization.stochastic_problem import DSGStochasticArchOptProblem
        return DSGStochasticArchOptProblem(self,
                                           self.param_space,
                                           self.uq_method,
                                           self.obj_scalar,
                                           self.constr_scalar,
                                           n_parallel=n_parallel, parallel_processes=parallel_processes)

    def _set_param_realizations(self, dsg, param_nodes: List[InputParameterNode], samples: np.ndarray, i_realization: int):
        realized = {param.ref: param.sample for param in self.param_space.param_realization(samples, i_realization)}
        for node in param_nodes:
            # Stochastic parameters get their realization, deterministic ones keep their fixed value
            dsg.set_inp_param_value(node, realized.get(node, node.value))

    def _evaluate(self, dsg: DSGType, metric_nodes: List[MetricNode]) -> Dict[MetricNode, EvaluationOutput]:
        """
        Implement _evaluate_sample with evaluation code for each realized sample stored on DSG.
        """
        # Sample the stochastic parameters
        stochastic_samples = self.uq_method.get_samples(self.param_space)
        param_nodes = dsg.inp_param_nodes

        n_s = stochastic_samples.shape[0]
        n_obj = len(self.objectives)
        n_constr = len(self.constraints)

        f_s = np.zeros((n_s, n_obj)) * np.nan
        g_s = np.zeros((n_s, n_constr)) * np.nan

        for i in range(n_s):
            # Set parameter realizations on the graph
            self._set_param_realizations(dsg, param_nodes, stochastic_samples, i)

            # Evaluate architecture for a realized sample
            value_map = self._evaluate_sample(dsg, metric_nodes)

            f_s[i, :] = [value_map.get(objective.node, math.nan) for objective in self.objectives]
            g_s[i, :] = [value_map.get(constraint.node, math.nan) if constraint.node in metric_nodes else constraint.ref for constraint in self.constraints]

        # Apply UQ method to process the results and return StochasticResult for this DSG instance
        outputs = self.uq_method.process_results(np.concatenate([f_s, g_s], axis=1), param_space = self.param_space)

        metric_map = {}

        # After UQ reset the parameter node to store its distribution
        for parameter_node in param_nodes:
            dsg.set_inp_param_value(parameter_node, parameter_node.value)

        # Return metric map
        for i, objective in enumerate(self.objectives):
            if objective.node in metric_nodes:
                metric_map[objective.node] = outputs[i]
        for i, constraint in enumerate(self.constraints):
            if constraint.node in metric_nodes and outputs[n_obj+i] is not None:
                metric_map[constraint.node] = outputs[n_obj+i]

        return metric_map

    def _evaluate_sample(self, dsg: DSGType, metric_nodes: List[MetricNode]) -> Dict[MetricNode, float]:
        """
        Implement this function to provide DSG evaluation for ONE realization of the uncertain parameters.
        Should return a mapping from metric node to float (NaN is allowed).
        """
        raise NotImplementedError


StochasticADSGEvaluator = DSGStochasticEvaluator