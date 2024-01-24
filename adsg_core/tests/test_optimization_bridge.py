import os
import pytest
import numpy as np
from adsg_core.examples.gnc import *
from adsg_core.examples.apollo import *
from adsg_core.optimization.problem import *

if HAS_SB_ARCH_OPT:
    from sb_arch_opt.algo.pymoo_interface import get_nsga2
    from pymoo.optimize import minimize

check_dependency = lambda: pytest.mark.skipif(not HAS_SB_ARCH_OPT, reason='SBArchOpt not installed')


@check_dependency()
def test_gnc():
    gnc = GNCEvaluator()
    problem = gnc.get_problem()

    assert np.all(problem.is_conditionally_active == gnc.dv_is_conditionally_active)
    assert problem.get_n_valid_discrete() == 29857

    x_all, is_act_all = problem.all_discrete_x
    assert x_all.shape == (29857, len(gnc.des_vars))

    x_try = []
    x_imp = []
    for _ in range(10):
        dv = gnc.get_random_design_vector()
        _, dv_imp, _ = gnc.get_graph(dv)
        x_try.append(dv)
        x_imp.append(dv_imp)
    x_try, x_imp = np.array(x_try), np.array(x_imp)

    x_imp_, _ = problem.correct_x(x_try)
    assert np.all(x_imp_ == x_imp)

    problem.evaluate(x_try)

    nsga2 = get_nsga2(pop_size=100)
    result = minimize(problem, nsga2, termination=('n_gen', 10), verbose=True)
    assert len(result.F) > 10


@pytest.mark.skipif(int(os.getenv('RUN_SLOW_TESTS', 0)) != 1, reason='Set RUN_SLOW_TESTS=1 to run slow tests')
@check_dependency()
def test_gnc_parallel():
    gnc = GNCEvaluator()
    problem = gnc.get_problem(n_parallel=10, parallel_processes=False)

    nsga2 = get_nsga2(pop_size=100)
    result = minimize(problem, nsga2, termination=('n_gen', 10), verbose=True)
    assert len(result.F) > 10


@pytest.mark.skipif(int(os.getenv('RUN_SLOW_TESTS', 0)) != 1, reason='Set RUN_SLOW_TESTS=1 to run slow tests')
@check_dependency()
def test_apollo():
    problem = ApolloEvaluator().get_problem()

    x_all, is_act_all = problem.all_discrete_x
    assert x_all.shape == (108, problem.n_var)

    x_imp_, _ = problem.correct_x(x_all)
    assert np.all(x_imp_ == x_all)

    problem.evaluate(x_all)

    nsga2 = get_nsga2(pop_size=100)
    result = minimize(problem, nsga2, termination=('n_gen', 2), verbose=True)
    assert len(result.F) > 1
