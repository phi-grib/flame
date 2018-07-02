
import pytest
import pathlib
from flame import manage


def test_manage_new_model():
    manage.set_model_repository('/home/testmodels')
    manage.action_new('TESTMODEL')
    home_dirs = list(pathlib.Path('/home/testmodels').iterdir())

    case = pathlib.Path('/home/testmodels/TESTMODEL')
    assert case in home_dirs
