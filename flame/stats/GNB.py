
# -*- coding: utf-8 -*-

# Description    Flame Parent Model Class
##
# Authors:       Jose Carlos Gómez (josecarlos.gomez@upf.edu)
#                Manuel Pastor (manuel.pastor@upf.edu)
##
# Copyright 2018 Manuel Pastor
##
# This file is part of Flame
##
# Flame is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation version 3.
##
# Flame is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
##
# You should have received a copy of the GNU General Public License
# along with Flame.  If not, see <http://www.gnu.org/licenses/>.


from sklearn.naive_bayes import GaussianNB
from flame.stats.base_model import BaseEstimator
from flame.stats.base_model import getCrossVal
from flame.stats.scale import scale, center
from flame.stats.model_validation import CF_QuanVal

from nonconformist.base import ClassifierAdapter, RegressorAdapter
from nonconformist.acp import AggregatedCp
from nonconformist.acp import BootstrapSampler
from nonconformist.icp import IcpClassifier, IcpRegressor
from nonconformist.nc import ClassifierNc, MarginErrFunc, RegressorNc
from flame.util import get_logger

LOG = get_logger(__name__)


class GNB(BaseEstimator):
    """
        This class inherits from BaseEstimator and wraps SKLEARN
        GaussianNB estimator

        ...
        
        Attributes
        ----------

        estimator_parameters : dict
            parameter values
        name : string
            name of the estimator
        
        
        Methods
        -------
        
        build(X)
            Instance the estimator optimizing it
            if tune=true.
        
    """

    def __init__(self, X, Y, parameters):
        # Initialize parent class
        try:
            super(GNB, self).__init__(X, Y, parameters)
            LOG.debug('Initializing BaseEstimator parent class')
        except Exception as e:
            LOG.error(f'Error initializing BaseEstimator parent'
                f'class with exception {e}')
        self.estimator_parameters = self.param.getDict('GNB_parameters')
        if self.param.getVal('quantitative'):
            raise Exception("GNB only applies to qualitative data")
        else:
            self.name = "GNB-Classifier"

    def build(self):
        '''Build a new qualitative GNB model with the X and Y numpy matrices'''

        # Make a copy of data matrices
        X = self.X.copy()
        Y = self.Y.copy()

        results = []
        results.append(('nobj', 'number of objects', self.nobj))
        results.append(('nvarx', 'number of predictor variables', self.nvarx))

        LOG.info('Building GaussianNB model')
        # Build estimator
        self.estimator = GaussianNB(**self.estimator_parameters)
        results.append(('model', 'model type', 'GNB qualitative'))
        # If conformal, then create aggregated conformal classifier
        if self.param.getVal('conformal'):
            self.conformal_pred = AggregatedCp(
                IcpClassifier(ClassifierNc(ClassifierAdapter(
                self.estimator), MarginErrFunc())), BootstrapSampler())
            # Fit estimator to the data
            self.conformal_pred.fit(X, Y)
            results.append(
                ('model', 'model type', 'conformal GNB qualitative'))

        self.estimator.fit(X, Y)
        return True, results
