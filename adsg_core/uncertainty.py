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
import warnings
from typing import *

__all__ = ['HAS_SB_ARCH_OPT', 'check_dependency', 'EvaluationOutput', 'Scalarization', 'StochasticArchOptProblem',
           'StochasticOutput', 'StochasticParameter', 'StochasticParameterSpace', 'UQMethod']

try:
    from sb_arch_opt.stochastic_problem import StochasticArchOptProblem
    from sb_arch_opt.uncertainty import (Scalarization, StochasticOutput, StochasticParameter, StochasticParameterSpace,
                                         UQMethod, MonteCarlo, PolynomialChaos, Mean, Margin)
    from sb_arch_opt.sampling import TrailRepairWarning

    warnings.simplefilter('ignore', category=TrailRepairWarning)

    HAS_SB_ARCH_OPT = True

    EvaluationOutput = Union[StochasticOutput, float]
    """Output either distribution or numeric value."""

except ImportError:
    HAS_SB_ARCH_OPT = False


    class StochasticArchOptProblem:
        pass


    class Scalarization:
        pass


    class StochasticOutput:
        pass


    class StochasticParameter:
        pass


    class StochasticParameterSpace:
        pass


    class UQMethod:
        pass


    class MonteCarlo:
        pass


    class PolynomialChaos:
        pass


    class Mean:
        pass


    class Margin:
        pass


    EvaluationOutput = Union[StochasticOutput, float]
    """Output either distribution or numeric value."""


def check_dependency():
    if not HAS_SB_ARCH_OPT:
        raise ImportError('Looks like SBArchOpt is not installed! Run: pip install sb-arch-opt[uncertainty]')