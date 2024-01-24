"""
MIT License

Copyright: (c) 2023, Deutsches Zentrum fuer Luft- und Raumfahrt e.V.
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
import os
import glob
import pytest
import numpy as np
from adsg_core.optimization.sel_choice_enc.util import *
from adsg_core.optimization.sel_choice_enc.brute_force import BruteForceSelectionChoiceEncoder


def _encoder_factory(**kwargs):
    return BruteForceSelectionChoiceEncoder(**kwargs)


@pytest.mark.skipif(int(os.getenv('RUN_SLOW_TESTS', 0)) != 1, reason='Set RUN_SLOW_TESTS=1 to run slow tests')
def test_all(case_data_path, other_data_path):
    for file in glob.glob(case_data_path+'/case_*.pkl'):
        print(f'Testing: {os.path.basename(file)}')
        validate_results(_encoder_factory, file)

    for file in glob.glob(other_data_path+'/*.pkl'):
        print(f'Testing: {os.path.basename(file)}')
        validate_results(_encoder_factory, file)


def test_case_1(case_data_path):
    encoder, results = create_from_results(_encoder_factory, case_data_path+'/case_1.pkl')

    assert encoder.n_valid == 2
    assert np.all(encoder.all_design_vectors_and_statuses[0] == results['x_all'])
