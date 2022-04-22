# import pytest
import pathlib
import os
from flame import manage
from flame.util import utils
from repo_config import MODEL_REPOSITORY

MODEL_NAME = "TESTMODEL"

def test_manage_new_model():
    utils.set_repositories(MODEL_REPOSITORY)
    manage.action_new(MODEL_NAME)
    home_dirs = list(pathlib.Path(os.path.join(MODEL_REPOSITORY,'models')).iterdir())

    case = pathlib.Path(os.path.join(os.path.join(MODEL_REPOSITORY,'models'), MODEL_NAME))
    assert case in home_dirs
