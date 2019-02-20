import pytest
import pathlib
import os
from flame import manage

from repo_config import MODEL_REPOSITORY

MODEL_NAME = "TESTMODEL"


def test_manage_new_model():
    manage.set_model_repository(MODEL_REPOSITORY)
    manage.action_new(MODEL_NAME)
    home_dirs = list(pathlib.Path(MODEL_REPOSITORY).iterdir())

    case = pathlib.Path(os.path.join(MODEL_REPOSITORY, MODEL_NAME))
    assert case in home_dirs
