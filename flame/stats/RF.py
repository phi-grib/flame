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

from copy import copy

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

from nonconformist.base import ClassifierAdapter, RegressorAdapter
from nonconformist.acp import AggregatedCp
from nonconformist.acp import BootstrapSampler
from nonconformist.icp import IcpClassifier, IcpRegressor
from nonconformist.nc import ClassifierNc, MarginErrFunc, RegressorNc
from nonconformist.nc import AbsErrorErrFunc, RegressorNormalizer

from flame.stats.base_model import BaseEstimator
from flame.util import get_logger
LOG = get_logger(__name__)


class RF(BaseEstimator):
    """
        This class inherits from BaseEstimator and wraps SKLEARN
        RandomForestClassifier or RandomForestRegressor estimator

        ...
        
        Attributes
        ----------

        estimator_parameters : dict
            parameter values
        name : string
            name of the estimator
        tune_parameters: dict
            Hyperparameter optimization settings
        
        Methods
        -------

        build(X)
            Instance the estimator optimizing it
            if tune=true.

    """
    def __init__(self, X, Y, parameters, conveyor):
        # Initialize parent class
        try:
            BaseEstimator.__init__(self, X, Y, parameters, conveyor)
            LOG.debug('Initialize BaseEstimator parent class')
        except Exception as e:
            self.conveyor.setError(f'Error initializing BaseEstimator parent class with exception: {e}')
            LOG.error(f'Error initializing BaseEstimator parent class with exception: {e}')
            return

        # Load estimator parameters        
        self.estimator_parameters = self.param.getDict('RF_parameters')

        # Load tune parameters
        self.tune_parameters = self.param.getDict('RF_optimize')

        if self.param.getVal('quantitative'):
            self.name = "RF-R"
            self.tune_parameters.pop("class_weight")
        else:
            self.name = "RF-C"

    def build(self):
        '''Build a new RF model with the X and Y numpy matrices '''

        # Make a copy of data matrices
        X = self.X.copy()
        Y = self.Y.copy()

        results = []
        results.append(('nobj', 'number of objects', self.nobj))
        results.append(('nvarx', 'number of predictor variables', self.nvarx))

        # If tune then call gridsearch to optimize the estimator
        if self.param.getVal('tune'):

            LOG.info("Optimizing RF estimator")

            try:
                # Check type of model
                if self.param.getVal('quantitative'):
                    self.optimize(X, Y, RandomForestRegressor(
                                    **self.estimator_parameters),
                                    self.tune_parameters)
                    results.append(('model','model type','RF quantitative (optimized)'))
                else:
                    self.optimize(X, Y, RandomForestClassifier(
                                    **self.estimator_parameters),
                                    self.tune_parameters)
                    results.append(('model','model type','RF qualitative (optimized)'))

            except Exception as e:
                return False, f'Exception optimizing RF estimator with exception {e}'
            
        else:
            try:
                if self.param.getVal('quantitative'):

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

                self.estimator.fit(X, Y)
                self.estimator_temp = copy(self.estimator)

            except Exception as e:
                return False, f'Exception building RF estimator with exception {e}'


        if not self.param.getVal('conformal'):
            return True, results

        # Create the conformal estimator
        try:
            # Conformal regressor
            if self.param.getVal('quantitative'):

                LOG.info("Building conformal Quantitative RF model")

                underlying_model = RegressorAdapter(self.estimator_temp)
                #normalizing_model = RegressorAdapter(
                    #KNeighborsRegressor(n_neighbors=5))
                normalizing_model = RegressorAdapter(self.estimator_temp)
                normalizer = RegressorNormalizer(
                                underlying_model,
                                normalizing_model,
                                AbsErrorErrFunc())
                nc = RegressorNc(underlying_model,
                                    AbsErrorErrFunc(),
                                    normalizer)

                # self.conformal_pred = AggregatedCp(IcpRegressor
                # (RegressorNc(RegressorAdapter(self.estimator))),
                #                                   BootstrapSampler())

                self.estimator = AggregatedCp(IcpRegressor(nc),
                                                BootstrapSampler())

                self.estimator.fit(X, Y)
                results.append(('model', 'model type', 'conformal RF quantitative'))

            # Conformal classifier
            else:

                LOG.info("Building conformal Qualitative RF model")

                self.estimator = AggregatedCp(
                                    IcpClassifier(
                                        ClassifierNc(
                                            ClassifierAdapter(self.estimator_temp),
                                            MarginErrFunc()
                                        )
                                    ),
                                    BootstrapSampler())

                # Fit estimator to the data
                self.estimator.fit(X, Y)
                results.append(('model', 'model type', 'conformal RF qualitative'))

        except Exception as e:
            return False, f'Exception building conformal RF estimator with exception {e}'

        return True, results



## Overriding of parent methods

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