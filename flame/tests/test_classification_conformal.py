import pytest
from pathlib import Path
import json
import numpy as np

from flame import manage
from flame import build
from flame import predict
from flame.util import utils

from repo_config import MODEL_REPOSITORY

MODEL_NAME = "CLASSIFCONF"
SDF = Path("data/classification.sdf")
current = Path(__file__).parent.resolve()
SDF_FILE_NAME = str(current / SDF)
FIXED_RESULTS = current / "data/classif_res_conf.json"


@pytest.fixture
def make_model():
    utils.set_repositories(MODEL_REPOSITORY)
    return manage.action_new(MODEL_NAME)


@pytest.fixture
def build_model():
    builder = build.Build(MODEL_NAME)
    builder.param.setVal("tune", False)
    builder.param.setVal("quantitative", False)
    builder.param.setVal("mol_batch", "objects")
    return builder.run(SDF_FILE_NAME)


@pytest.fixture
def fixed_results():

    with open(FIXED_RESULTS) as f:
        results = json.load(f)
    return np.array(results["c0"])


def test_classification_conformal(make_model, build_model, fixed_results):
    """test predict comparing results"""

    make_status, message = make_model
    assert (make_status is True) or (message == f"Endpoint {MODEL_NAME} already exists")

    build_status, _ = build_model
    assert build_status is True

    predictor = predict.Predict(MODEL_NAME, 0, label='temp')
    predictor.param.setVal("quantitative", False)
    predictor.param.setVal("output_format", "ghost")

    _, results = predictor.run(SDF_FILE_NAME)
    result_values = results.getVal("c0")

    assert all(np.isclose(fixed_results, result_values, rtol=1e-4))
