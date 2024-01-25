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
import warnings
import functools
import numpy as np
from typing import *
from concurrent.futures import wait, ProcessPoolExecutor, ThreadPoolExecutor
from adsg_core.optimization.evaluator import ADSGEvaluator
from adsg_core.optimization.dv_output_defs import DesVar
from adsg_core.optimization.graph_processor import GraphProcessor
from adsg_core.optimization.assign_enc.time_limiter import run_timeout

try:
    from sb_arch_opt.problem import ArchOptProblemBase
    from sb_arch_opt.design_space import ArchDesignSpace
    from pymoo.core.variable import Variable, Real, Integer, Choice

    from sb_arch_opt.sampling import TrailRepairWarning
    warnings.simplefilter("ignore", category=TrailRepairWarning)

    HAS_SB_ARCH_OPT = True

except ImportError:
    HAS_SB_ARCH_OPT = False

    class ArchDesignSpace:
        pass

    class ArchOptProblemBase:
        pass

__all__ = ['check_dependency', 'ADSGDesignSpace', 'ADSGArchOptProblem', 'HAS_SB_ARCH_OPT']

log = logging.getLogger('adsg.opt')


def check_dependency():
    if not HAS_SB_ARCH_OPT:
        raise ImportError('Looks like SBArchOpt is not installed! Run: pip install sb-arch-opt')


class ADSGDesignSpace(ArchDesignSpace):
    """
    SBArchOpt design space implementation for an ADSG design problem.
    """
    x_all_cutoff = 500000
    x_all_timeout = 10  # sec

    def __init__(self, processor: GraphProcessor):
        if len(processor.des_vars) == 0:
            raise ValueError('No design variables in optimization problem!')

        self._processor = processor
        super().__init__()

    @property
    def processor(self) -> GraphProcessor:
        return self._processor

    def is_explicit(self) -> bool:
        return False

    def _get_variables(self) -> List['Variable']:
        """Returns the list of design variables (pymoo classes)"""
        des_vars = []
        for i, des_var in enumerate(self._processor.des_vars):
            if des_var.is_discrete:
                # Define as categorical is explicitly said so and if there are more than two options (gives slightly
                # better optimizer performance in some cases)
                if self._is_categorical(i, des_var) and len(des_var.options) > 2:
                    des_vars.append(Choice(options=[ii for ii in range(len(des_var.options))]))
                else:
                    des_vars.append(Integer(bounds=(0, len(des_var.options)-1)))

            else:
                des_vars.append(Real(bounds=tuple(des_var.bounds)))

        return des_vars

    def _is_categorical(self, idx, des_var: DesVar) -> bool:
        """Whether a given discrete design variable should be represented as a categorical variable"""
        # We do not have a better way to determine this currently
        return True

    def _is_conditionally_active(self) -> Optional[List[bool]]:
        """Returns for each design variable whether it is conditionally active (i.e. may become inactive)"""
        return self._processor.dv_is_conditionally_active

    def _correct_x(self, x: np.ndarray, is_active: np.ndarray):
        """
        Fill the activeness matrix (n x nx) and if needed correct design vectors (n x nx) that are partially inactive.
        Imputation of inactive variables is handled automatically.
        """
        is_discrete_mask = self.is_discrete_mask
        for i, xi in enumerate(x):
            x_arch = [int(val) if is_discrete_mask[j] else float(val) for j, val in enumerate(xi)]
            _, x_imputed, is_active_arch = self._processor.get_graph(x_arch, create=False)
            x[i, :] = x_imputed
            is_active[i, :] = is_active_arch

    def _quick_sample_discrete_x(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """Sample n discrete design vectors (also return is_active) without generating all design vectors first"""
        return self._quick_random_sample_discrete_x(n)

    @functools.lru_cache()
    def _get_n_valid_discrete(self) -> Optional[int]:
        """
        Return the number of valid discrete design points (ignoring continuous dimensions); enables calculation of
        the imputation ratio.
        Valid discrete design points are discrete design points where value constraints are satisfied and where
        inactive design variables are imputed/canonical (compare _get_n_correct_discrete).
        """
        return self._processor.get_n_valid_designs(with_fixed=True)

    @functools.lru_cache()
    def _get_n_active_cont_mean(self) -> Optional[float]:
        """
        Get the mean number of active continuous dimensions, as seen over all valid discrete design vectors.
        """
        _, _, _, _, n_dim_cont_mean, _ = \
            self._processor.get_additional_dv_stats(with_fixed=True)
        return n_dim_cont_mean

    def _get_n_correct_discrete(self) -> Optional[int]:
        """
        Return the number of correct discrete design points (ignoring continuous dimensions); enables calculation of
        the correction ratio.
        Correct discrete design points are discrete design points where value constraints are satisfied, where however
        inactive design variables can have any value (compare _get_n_valid_discrete).
        """

    def _get_n_active_cont_mean_correct(self) -> Optional[float]:
        """
        Get the mean number of active continuous dimensions, as seen over all valid discrete design vectors.
        """

    def _gen_all_discrete_x(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Generate all possible discrete design vectors (if available). Returns design vectors and activeness
        information."""

        if self._get_n_valid_discrete() > self.x_all_cutoff:
            log.debug(f'Not generating all design vectors because '
                      f'{self._get_n_valid_discrete()} (n_x_discrete) > {self.x_all_cutoff}')
            return

        try:
            log.debug('Generating all design vectors')
            res = run_timeout(
                self.x_all_timeout,
                lambda: self._processor.get_all_discrete_x(with_fixed=True))
            if res is None:
                return
            x, is_active = res
            log.debug(f'Generated {x.shape[0]} design vectors')

            is_cond_act = np.array(self._is_conditionally_active(), dtype=bool)
            any_inactive, not_cond_active = np.any(~is_active, axis=0), ~is_cond_act
            if np.any(any_inactive & not_cond_active):
                i_where, = np.where(any_inactive & not_cond_active)
                raise RuntimeError(f'Inconsistent conditionality vector for i_dv: {i_where}')

            return x, is_active
        except (MemoryError, TimeoutError) as e:
            log.debug(f'Could not generate all design vectors: {e.__class__.__name__}')


class ADSGArchOptProblem(ArchOptProblemBase):
    """
    [SBArchOpt](https://sbarchopt.readthedocs.io/) wrapper for an ADSG optimization problem. Note that under the
    hood, SBArchOpt uses [pymoo](https://pymoo.org/).
    The connection is made between the `ArchOptProblemBase` class (which specifies all information needed to optimize an
    architecture optimization problem), and the `ADSGEvaluator` class, which contains all information for
    running an ADSG architecture optimization problem.

    Parallel processing is possible by setting `n_parallel` to a number higher than 1.
    By default, assumes parallel processing is done within the thread and therefore starts a multiprocessing pool to
    run the parallel evaluations.

    Ensure SBArchOpt is installed: `pip install sb-arch-opt`

    Example usage:

    ```python
    from pymoo.optimize import minimize
    from sb_arch_opt.algo.pymoo_interface import get_nsga2

    evaluator = ...  # Instance of ADSGEvaluator

    algorithm = get_nsga2(pop_size=100)
    problem = ADSGArchOptProblem(evaluator)

    result = minimize(problem, algorithm, termination=('n_eval', 500))
    ```
    """

    def __init__(self, evaluator: ADSGEvaluator, n_parallel=None, parallel_processes=True):
        check_dependency()

        self.evaluator = evaluator
        self.n_parallel = n_parallel
        self.parallel_processes = parallel_processes

        n_objs = len(evaluator.objectives)
        n_constr = len(evaluator.constraints)

        design_space = ADSGDesignSpace(evaluator)
        super().__init__(design_space, n_obj=n_objs, n_ieq_constr=n_constr)

        self.obj_is_max = [obj.dir.value > 0 for obj in evaluator.objectives]
        self.con_ref = [(con.dir > 0, con.ref) for con in evaluator.constraints]

    def _arch_evaluate(self, x: np.ndarray, is_active_out: np.ndarray, f_out: np.ndarray, g_out: np.ndarray,
                       h_out: np.ndarray, *args, **kwargs):
        # Correct integer design variables
        self.design_space.round_x_discrete(x)

        # Generate architectures
        is_discrete_mask = self.is_discrete_mask
        adsg_instances = []
        for i, xi in enumerate(x):
            x_arch = [int(val) if is_discrete_mask[j] else float(val) for j, val in enumerate(xi)]
            adsg_instance, x_imputed, is_active_arch = self.evaluator.get_graph(x_arch)
            adsg_instances.append(adsg_instance)
            x[i, :] = x_imputed
            is_active_out[i, :] = is_active_arch

        # Evaluate architectures
        if self.n_parallel is not None and self.n_parallel > 1:
            executor_class = ProcessPoolExecutor if self.parallel_processes else ThreadPoolExecutor
            with executor_class(max_workers=self.n_parallel) as executor:
                futures = [executor.submit(self.evaluator.evaluate, adsg) for adsg in adsg_instances]

                wait(futures)
                results = [fut.result() for fut in futures]

        else:
            results = [self.evaluator.evaluate(adsg) for adsg in adsg_instances]

        # Process results
        for i, (obj_values, con_values) in enumerate(results):
            # Correct directions of objectives to represent minimization
            f_out[i, :] = [-val if self.obj_is_max[j] else val for j, val in enumerate(obj_values)]

            # Correct directions and offset constraints to represent g(x) <= 0
            g_out[i, :] = [(val-self.con_ref[j][1])*(-1 if self.con_ref[j][0] else 1)
                           for j, val in enumerate(con_values)]

    def _print_extra_stats(self):
        self.get_discrete_rates(show=True)
        self.evaluator.print_stats()

    def get_n_batch_evaluate(self) -> Optional[int]:
        return self.n_parallel

    def __repr__(self):
        return f'{self.__class__.__name__}({self.evaluator!r})'
