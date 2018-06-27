
import pytest
import pathlib
import os

from flame.util import utils


def test_read_congiguration():
    assert isinstance(utils._read_configuration(), dict)


def test_default_model_repository_path():
    utils.set_model_repository()
    model_path = utils.model_repository_path()
    realpath = os.path.join(pathlib.Path(__file__).resolve().parents[1],
                            'models/')
    assert model_path == realpath


def test_custom_model_repository_path():
    utils.set_model_repository('.')
    model_path = utils.model_repository_path()
    realpath = pathlib.Path('.').resolve()
    assert model_path == realpath
