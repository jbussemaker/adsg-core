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
import logging
import math
from typing import *
import warnings

import numpy as np

from adsg_core import DSGType, DSGEvaluator
from adsg_core.graph.adsg_nodes import MetricNode
from adsg_core.optimization.graph_processor import *

__all__ = ['StochasticDSGEvaluator', 'StochasticADSGEvaluator', 'HAS_SB_ARCH_OPT', 'check_dependency']
try:
    from sb_arch_opt.stochastic_problem import StochasticArchOptProblem
    from sb_arch_opt.uncertainty import Scalarization, StochasticResults, UQMethod, StochasticParameterSpace, StochasticOutput
    from sb_arch_opt.sampling import TrailRepairWarning

    warnings.simplefilter("ignore", category=TrailRepairWarning)

    HAS_SB_ARCH_OPT = True

except ImportError:
    HAS_SB_ARCH_OPT = False

    class Scalarization:
        pass

    class StochasticResults:
        pass

    class UQMethod:
        pass

    class StochasticParameterSpace:
        pass

    class StochasticOutput:
        pass

log = logging.getLogger('adsg.opt')


def check_dependency():
    if not HAS_SB_ARCH_OPT:
        raise ImportError('Looks like SBArchOpt is not installed! Run: pip install sb-arch-opt')

class StochasticDSGEvaluator(DSGEvaluator):
    """
    Base class for implementing an evaluator for stochastic problem that directly evaluates DSG instances.
    Override _evaluate_sample to implement the evaluation.

    Inherits `DSGEvaluator` and 'GraphProcessor', so all their functions are also available.
    """

    def __init__(self,
                 *args,
                 uq_method: UQMethod,
                 obj_scalar: List[Scalarization] = None,
                 constr_scalar: List[Scalarization] = None,
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


    def _evaluate(self, dsg: DSGType, metric_nodes: List[MetricNode]) -> Dict[MetricNode, StochasticOutput]:
        """
        Implement this function to provide stochastic DSG evaluation .
        Override this function if external UQ tool is linked.

        Implement _evaluate_sample with evaluation code for each realized sample stored on DSG.
        """
        # Sample the stochastic parameters
        stochastic_samples = self.uq_method.get_samples(self.param_space)
        parameter_nodes = dsg.input_parameter_nodes

        n_s = stochastic_samples.shape[0]
        n_obj = len(self.objectives)
        n_constr = len(self.constraints)

        f_s = np.zeros((n_s, n_obj)) * np.nan
        g_s = np.zeros((n_s, n_constr)) * np.nan

        for i in range(n_s):
            # Create a dictionary that associates parameters with its realization
            sample_values = self.param_realization(stochastic_samples, i)

            if sample_values is None:
                raise ValueError(f"No sample values available for realization {i}")

            # Set parameter realization or use its deterministic value on the DSG instance
            for parameter in parameter_nodes:
                value = sample_values[parameter]
                dsg.set_input_parameter_value(parameter, value)

            # Evaluate architecture for a realized sample
            value_map = self._evaluate_sample(dsg, metric_nodes)

            f_s[i, :] = [value_map.get(objective.node, math.nan) for objective in self.objectives]
            g_s[i, :] = [value_map.get(constraint.node, math.nan) if constraint.node in metric_nodes else constraint.ref for constraint in self.constraints]

        # Apply UQ method to process the results and return StochasticResult for this DSG instance
        result = self.uq_method.process_results(np.concatenate([f_s, g_s], axis=1), param_space = self.param_space)

        metric_map = {}

        # After UQ reset the parameter node to store its distribution
        for parameter_node in parameter_nodes:
            dsg.set_input_parameter_value(parameter_node, parameter_node.value)

        # Return metric map
        for i, objective in enumerate(self.objectives):
            if objective.node in metric_nodes:
                metric_map[objective.node] = result.outputs[i]
        for i, constraint in enumerate(self.constraints):
            if constraint.node in metric_nodes:
                metric_map[constraint.node] = result.outputs[n_obj+i]

        return metric_map


    def _evaluate_sample(self, dsg: DSGType, metric_nodes: List[MetricNode]) -> Dict[MetricNode, float]:
        """
        Implement this function to provide DSG evaluation for ONE realization of the uncertain parameters.
        Should return a mapping from metric node to float (NaN is allowed).
        """
        raise NotImplementedError

StochasticADSGEvaluator = StochasticDSGEvaluator