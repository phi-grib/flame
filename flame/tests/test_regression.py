import pytest

import io
import os
from pathlib import Path
import json

import numpy as np

from flame import manage
from flame import build
from flame import predict

# paths configs
from repo_config import MODEL_REPOSITORY

MODEL_NAME = "REGR"
current = Path(__file__).parent.resolve()
SDF_FILE_NAME = str(current / "data" / "minicaco.sdf")
FIXED_RESULTS = current / "data" / "regression_res.json"


@pytest.fixture
def make_model():
    manage.set_model_repository(MODEL_REPOSITORY)
    return manage.action_new(MODEL_NAME)


@pytest.fixture
def build_model():
    builder = build.Build(MODEL_NAME)
    builder.param.setVal("tune", False)
    builder.param.setVal("conformal", False)
    return builder.run(SDF_FILE_NAME)


@pytest.fixture
def fixed_results():

    with open(FIXED_RESULTS) as f:
        results = json.load(f)
    return np.array(results["Y_pred"])


def test_regression(make_model, build_model, fixed_results):
    """test predict comparing results"""

    make_status, message = make_model
    assert (make_status is True) or (message == f"Endpoint {MODEL_NAME} already exists")

    build_status, _ = build_model
    assert build_status is True

    predictor = predict.Predict(MODEL_NAME, 0, label='temp')
    predictor.param.setVal("conformal", False)
    predictor.param.setVal("output_format", "JSON")

    _, results = predictor.run(SDF_FILE_NAME)
    result_values = results.getVal("values")

    assert all(np.isclose(fixed_results, result_values, rtol=1e-4))
