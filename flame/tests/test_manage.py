
import pytest
import pathlib
from flame import manage


def test_manage_new_model():
    manage.set_model_repository('/home')
    manage.action_new('TESTMODEL')
    home_dirs = list(pathlib.Path('/home').iterdir())

    case = pathlib.Path('/home/TESTMODEL')
    assert case in home_dirs
