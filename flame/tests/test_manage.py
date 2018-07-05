
import pytest
import pathlib
import os
from flame import manage

MODEL_REPOSITORY = '/home/biel/Documents/TESTS'
MODEL_NAME = 'TESTMODEL'
SDF_FILE_NAME = 'minicaco.sdf'


def test_manage_new_model():
    manage.set_model_repository(MODEL_REPOSITORY)
    manage.action_new(MODEL_NAME)
    home_dirs = list(pathlib.Path(MODEL_REPOSITORY).iterdir())

    case = pathlib.Path(os.path.join(MODEL_REPOSITORY, MODEL_NAME))
    assert case in home_dirs
