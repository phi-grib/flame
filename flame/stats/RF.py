#! -*- coding: utf-8 -*-

# Description    Flame Parent Model Class
##
# Authors:       Jose Carlos Gómez (josecarlos.gomez@upf.edu)
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

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

from nonconformist.base import ClassifierAdapter, RegressorAdapter
from nonconformist.acp import AggregatedCp
from nonconformist.acp import BootstrapSampler
from nonconformist.icp import IcpClassifier, IcpRegressor
from nonconformist.nc import ClassifierNc, MarginErrFunc, RegressorNc
from sklearn.neighbors import KNeighborsRegressor
from nonconformist.nc import AbsErrorErrFunc, RegressorNormalizer

from flame.stats.base_model import BaseEstimator
from flame.stats.model_validation import getCrossVal
from flame.stats.scale import scale, center
from flame.stats.model_validation import CF_QuanVal
from flame.util import get_logger

LOG = get_logger(__name__)


class RF(BaseEstimator):
    def __init__(self, X, Y, parameters):
        # Initialize parent class
        try:
            super(RF, self).__init__(X, Y, parameters)
            LOG.debug('Initialize BaseEstimator parent class')
        except Exception as e:
            LOG.error(f'Error initializing BaseEstimator parent'
                    f'class with exception: {e}')
            raise e
        self.estimator_parameters = parameters['RF_parameters']
        self.tune_parameters = parameters['RF_optimize']

        if self.parameters['quantitative']:
            self.name = "RF-R"
            self.tune_parameters.pop("class_weight")
        else:
            self.name = "RF-C"

    def build(self):
        '''Build a new RF model with the X and Y numpy matrices '''

        # Make a copy of the matrices
        X = self.X.copy()
        Y = self.Y.copy()

        results = []
        results.append(('nobj', 'number of objects', self.nobj))
        results.append(('nvarx', 'number of predictor variables', self.nvarx))

        if self.parameters['tune']:
                if self.parameters['quantitative']:
                    self.optimize(X, Y, RandomForestRegressor(),
                                self.tune_parameters)
                    results.append(
                        ('model', 'model type', 'RF quantitative (optimized)'))
                else:
                    self.optimize(X, Y, RandomForestClassifier(),
                                self.tune_parameters)
                    results.append(
                        ('model', 'model type', 'RF qualitative (optimized)'))
        else:
            if self.parameters['quantitative']:
                LOG.info("Building Quantitative RF model")
                self.estimator_parameters.pop('class_weight', None)

                self.estimator = RandomForestRegressor(
                    **self.estimator_parameters)
                results.append(('model', 'model type', 'RF quantitative'))

            else:
                LOG.info("Building Qualitative RF model")
                self.estimator = RandomForestClassifier(
                    **self.estimator_parameters)
                results.append(('model', 'model type', 'RF qualitative'))
        # Create the conformal estimator
        if self.parameters['conformal']:
            # Conformal regressor
            if self.parameters['quantitative']:
                underlying_model = RegressorAdapter(self.estimator)
                normalizing_model = RegressorAdapter(
                    KNeighborsRegressor(n_neighbors=5))
                normalizing_model = RegressorAdapter(self.estimator)
                normalizer = RegressorNormalizer(
                    underlying_model, normalizing_model, AbsErrorErrFunc())
                nc = RegressorNc(underlying_model,
                                 AbsErrorErrFunc(), normalizer)
                # self.conformal_pred = AggregatedCp(IcpRegressor
                # (RegressorNc(RegressorAdapter(self.estimator))),
                #                                   BootstrapSampler())

                self.conformal_pred = AggregatedCp(IcpRegressor(nc),
                                                   BootstrapSampler())
                self.conformal_pred.fit(X, Y)
                # overrides non-conformal
                results.append(
                    ('model', 'model type', 'conformal RF quantitative'))
            # Conformal classifier
            else:
                self.conformal_pred = AggregatedCp(
                                        IcpClassifier(
                                            ClassifierNc(
                                                ClassifierAdapter(self.estimator),
                                                    MarginErrFunc())),
                                         BootstrapSampler())
                # Fit estimator to the data
                self.conformal_pred.fit(X, Y)
                results.append(
                    ('model', 'model type', 'conformal RF qualitative'))

        self.estimator.fit(X, Y)

        return True, results



#### Overriding of parent methods

    # def CF_quantitative_validation(self):
    #     ''' performs validation for conformal quantitative models '''

      

    # def CF_qualitative_validation(self):
    #     ''' performs validation for conformal qualitative models '''


    # def quantitativeValidation(self):
    #     ''' performs validation for quantitative models '''

    # def qualitativeValidation(self):
    #     ''' performs validation for qualitative models '''


    # def validate(self):
    #     ''' Validates the model and computes suitable model quality scoring values'''


    # def optimize(self, X, Y, estimator, tune_parameters):
    #     ''' optimizes a model using a grid search over a range of values for diverse parameters'''


    # def regularProject(self, Xb, results):
    #     ''' projects a collection of query objects in a regular model, for obtaining predictions '''


    # def conformalProject(self, Xb, results):
    #     ''' projects a collection of query objects in a conformal model, for obtaining predictions '''


    # def project(self, Xb, results):
    #     ''' Uses the X matrix provided as argument to predict Y'''