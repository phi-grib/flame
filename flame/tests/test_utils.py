import pytest
import pathlib
import os
import sys

from flame.util import utils
from flame import manage

from repo_config import MODEL_REPOSITORY

MODEL_NAME = "TESTMODEL"


def test_read_congiguration():
    assert isinstance(utils._read_configuration(), dict)


# Must rethink the following test

# def test_default_model_repository_path():
#     utils.set_model_repository()
#     model_path = utils.model_repository_path()
#     realpath = os.path.join(pathlib.Path(__file__).resolve().parents[1],
#                             'models/')
#     assert model_path == realpath


def test_custom_model_repository_path():
    utils.set_model_repository(".")
    model_path = utils.model_repository_path()
    realpath = str(pathlib.Path(".").resolve())
    assert model_path == realpath


def test_module_path_sys_append():
    """
    Tests if model directory is in sys.path to use importlib.module_import()
    for child model classes.
    """
    models_dir = MODEL_REPOSITORY
    manage.set_model_repository(models_dir)
    manage.action_new(MODEL_NAME)
    utils.module_path(MODEL_NAME, 0)
    assert sys.path[0] == models_dir


def test_module_path_module_name():
    """
    Tests if importlib.module_import() works
    """
    models_dir = MODEL_REPOSITORY
    manage.set_model_repository(models_dir)
    manage.action_new(MODEL_NAME)
    module_name = utils.module_path(MODEL_NAME, 0)
    assert module_name == (MODEL_NAME + ".dev")
