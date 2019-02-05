import pytest

import io
import os
import json
from pathlib import Path

from flame.util import utils
from flame import manage

from repo_config import MODEL_REPOSITORY

MODEL_NAME = 'TESTPARAMS'

# @pytest.fixture
# def make_model():
manage.set_model_repository(MODEL_REPOSITORY)
manage.action_new(MODEL_NAME)

_, params = utils.get_parameters(MODEL_NAME, 0)

# delete model info to sanitize params
del params['endpoint']
del params['version']
del params['model_path']
del params['md5']
del params['param_format']


for k, v in params.items():
    if v['options']:
        # option is single value not listed
        if not isinstance(v['options'], list):
            if not v['options'] == v['value']:
                print(f'bad keys {k}')
                continue
        # values are a list and all must be in options
        if isinstance(v['value'], list):
            if not all(val in v['options'] for val in v['value']):
                print(f'bad keys {k}')
                continue
        if v['value'] not in v['options']:
            print(f'bad keys {k}')
            continue