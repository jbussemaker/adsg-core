import os
import pytest
import testbook

_docs_path = f'{os.path.dirname(__file__)}/../../docs'
_t = 60*20


@pytest.mark.skipif(int(os.getenv('RUN_SLOW_TESTS', 0)) != 1, reason='Set RUN_SLOW_TESTS=1 to run slow tests')
@testbook.testbook(f'{_docs_path}/guide.ipynb', execute=True, timeout=_t)
def test_tutorial(tb):
    pass


@pytest.mark.skipif(int(os.getenv('RUN_SLOW_TESTS', 0)) != 1, reason='Set RUN_SLOW_TESTS=1 to run slow tests')
@testbook.testbook(f'{_docs_path}/example_apollo.ipynb', execute=True, timeout=_t)
def test_tutorial(tb):
    pass


@pytest.mark.skipif(int(os.getenv('RUN_SLOW_TESTS', 0)) != 1, reason='Set RUN_SLOW_TESTS=1 to run slow tests')
@testbook.testbook(f'{_docs_path}/example_gnc.ipynb', execute=True, timeout=_t)
def test_tutorial(tb):
    pass
