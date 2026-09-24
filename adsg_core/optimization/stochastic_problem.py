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
import numpy as np
from typing import *
from concurrent.futures import wait, ProcessPoolExecutor, ThreadPoolExecutor

from adsg_core.optimization.stochastic_evaluator import  DSGStochasticEvaluator
from adsg_core.optimization.problem import DSGDesignSpace
from adsg_core.uncertainty import HAS_SB_ARCH_OPT, check_dependency, Scalarization, StochasticArchOptProblem, StochasticParameterSpace, UQMethod

__all__ = ['check_dependency', 'DSGStochasticArchOptProblem', 'HAS_SB_ARCH_OPT', 'ADSGStochasticArchOptProblem']

log = logging.getLogger('adsg.opt')


class DSGStochasticArchOptProblem(StochasticArchOptProblem):
    """
    [SBArchOpt](https://sbarchopt.readthedocs.io/) wrapper for a DSG stochastic optimization problem. Note that under the
    hood, SBArchOpt uses [pymoo](https://pymoo.org/).
    The connection is made between the `StochasticArchOptProblem` class (which specifies all information needed to optimize an
    architecture optimization problem), and the `DSGStochasticEvaluator` class, which contains all information for
    running a stochastic DSG architecture optimization problem.

    Parallel processing is possible by setting `n_parallel` to a number higher than 1.
    By default, assumes parallel processing is done within the thread and therefore starts a multiprocessing pool to
    run the parallel evaluations.

    Ensure SBArchOpt is installed: `pip install sb-arch-opt`

    Example usage:

    ```python
    from pymoo.optimize import minimize
    from sb_arch_opt.algo.pymoo_interface import get_nsga2

    evaluator = ...  # Instance of DSGStochasticEvaluator

    algorithm = get_nsga2(pop_size=100)
    problem = DSGStochasticArchOptProblem(evaluator, uq_method)

    result = minimize(problem, algorithm, termination=('n_eval', 500))
    ```
    """

    def __init__(self, evaluator: DSGStochasticEvaluator,
                 param_space: StochasticParameterSpace,
                 uq_method: UQMethod,
                 obj_scalar: Optional[List[Scalarization]] = None,
                 constr_scalar: Optional[List[Scalarization]] = None,
                 n_parallel=None, parallel_processes=True):
        check_dependency()

        self.evaluator = evaluator
        self.n_parallel = n_parallel
        self.parallel_processes = parallel_processes

        n_obj = len(evaluator.objectives)
        n_constr = len(evaluator.constraints)

        design_space = DSGDesignSpace(evaluator)

        super().__init__(design_space, param_space=param_space, uq_method=uq_method, n_obj=n_obj, n_ieq_constr=n_constr,
                         obj_scalar=obj_scalar, ieq_constr_scalar=constr_scalar)

        self.obj_is_max = [obj.dir.value > 0 for obj in evaluator.objectives]
        self.con_ref = [(con.dir.value > 0, con.ref) for con in evaluator.constraints]

    def _arch_evaluate(self, x: np.ndarray, is_active_out: np.ndarray, f_out: np.ndarray, g_out: np.ndarray, h_out: np.ndarray, *args,
                       f_stoch_out: np.ndarray=None, g_stoch_out: np.ndarray=None, h_stoch_out: np.ndarray=None, **kwargs):
        """
        Overrides parent _arch_evaluate class to integrate it with DSGStochasticEvaluator, but maintains the same functionality.
        """
        # Correct integer design variables
        self.design_space.round_x_discrete(x)

        # Generate architectures
        is_discrete_mask = self.is_discrete_mask
        dsg_instances = []
        for i, xi in enumerate(x):
            x_arch = [int(val) if is_discrete_mask[j] else float(val) for j, val in enumerate(xi)]
            dsg_instance, x_imputed, is_active_arch = self.evaluator.get_graph(x_arch)
            dsg_instances.append(dsg_instance)
            x[i, :] = x_imputed
            is_active_out[i, :] = is_active_arch

        # Evaluate architectures for each DSG instance
        if self.n_parallel is not None and self.n_parallel > 1:
            executor_class = ProcessPoolExecutor if self.parallel_processes else ThreadPoolExecutor
            with executor_class(max_workers=self.n_parallel) as executor:
                futures = [executor.submit(self.evaluator.evaluate, dsg) for dsg in dsg_instances]

                wait(futures)
                results = [fut.result() for fut in futures]

        else:
            results = [self.evaluator.evaluate(dsg) for dsg in dsg_instances]

        # Process results
        for i, (obj_outputs, con_outputs) in enumerate(results):

            for j, obj_output in enumerate(obj_outputs):
                obj_scalar = self.obj_scalar[j]
                val = obj_scalar.scalarize(obj_output) if not isinstance(obj_output, float) else obj_output

                f_stoch_out[i, j] = obj_output
                f_out[i, j] = -val if self.obj_is_max[j] else val

            for j, con_output in enumerate(con_outputs):
                con_scalar = self.ieq_constr_scalar[j]
                val = con_scalar.scalarize(con_output) if not isinstance(con_output, float) else con_output

                g_stoch_out[i, j] = con_output
                g_out[i, j] = (val-self.con_ref[j][1])*(-1 if self.con_ref[j][0] else 1)

    def _print_extra_stats(self):
        super()._print_extra_stats()
        self.get_discrete_rates(show=True)
        self.evaluator.print_stats()

    def get_n_batch_evaluate(self) -> Optional[int]:
        return self.n_parallel

    def __repr__(self):
        return f'{self.__class__.__name__}({self.evaluator!r})'


ADSGStochasticArchOptProblem = DSGStochasticArchOptProblem  # Backward compatibility
