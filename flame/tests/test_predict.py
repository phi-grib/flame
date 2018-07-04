import pytest

import io
import json

import numpy as np

from flame import manage
from flame import build
from flame import predict

MODEL_REPOSITORY = '/home/testmodels'
MODEL_NAME = 'FULLMODEL'
SDF_FILE_NAME = 'minicaco.sdf'


@pytest.fixture
def make_model():
    manage.set_model_repository(MODEL_REPOSITORY)
    return manage.action_new(MODEL_NAME)


@pytest.fixture
def build_model():
    builder = build.Build(MODEL_NAME)
    builder.parameters['tune'] = False
    return builder.run(SDF_FILE_NAME)


@pytest.fixture
def fixed_results():
    with open('results_file.json') as f:
        results = json.load(f)
    return np.array(results['values'])


@pytest.fixture
def predict_model():
    predictor = predict.Predict(MODEL_NAME, 0)
    _, results_str = predictor.run(SDF_FILE_NAME)
    prediction_results = json.load(io.StringIO(results_str))
    return np.array(prediction_results['values'])


def test_new_build_predict(make_model,
                           build_model,
                           fixed_results,
                           predict_model):
    """
    test until comparing results. Caution with 'rtol'
    """

    make_status, _ = make_model
    assert make_status is True

    build_status, _ = build_model
    assert build_status is True

    results = predict_model

    assert all(np.isclose(fixed_results, results, rtol=1.e-2))
