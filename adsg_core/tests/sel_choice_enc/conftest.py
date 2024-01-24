import os
import pytest


@pytest.fixture
def case_data_path():
    return os.path.dirname(__file__)+'/case_data'


@pytest.fixture
def other_data_path():
    return os.path.dirname(__file__)+'/other_data'
