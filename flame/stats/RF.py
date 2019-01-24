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
from copy import copy
from flame.stats.base_model import BaseEstimator
from flame.stats.model_validation import getCrossVal
from flame.stats.scale import scale, center
from flame.stats.model_validation import CF_QuanVal
from flame.util import get_logger
#from flame.parameters import Parameters

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
    def __init__(self, X, Y, parameters):
        # Initialize parent class
        try:
            super(RF, self).__init__(X, Y, parameters)
            LOG.debug('Initialize BaseEstimator parent class')
        except Exception as e:
            LOG.error(f'Error initializing BaseEstimator parent'
                    f'class with exception: {e}')
            raise e
        # Load estimator parameters

        # alternative approach based in the use of a Parameters class for storing 
        # estimator and optimizer parameters
        # self.estimator_parametes = Parameters()
        # self.estimator_parametes.loadDict(self.param.getVal('RF_parameters'))
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
                try:
                    # Check type of model
                    if self.param.getVal('quantitative'):
                        self.optimize(X, Y, RandomForestRegressor(),
                                    self.tune_parameters)
                        results.append(
                            ('model', 'model type', 
                            'RF quantitative (optimized)'))
                    else:
                        self.optimize(X, Y, RandomForestClassifier(),
                                      self.tune_parameters)
                        results.append(
                            ('model', 'model type', 
                             'RF qualitative (optimized)'))
                    LOG.debug('RF estimator optimized')
                except Exception as e:
                    LOG.error(f'Exception optimizing RF' 
                              f'estimator with exception {e}')

        else:
            try:
                LOG.info("Building  RF model")
                if self.param.getVal('quantitative'):
                    self.estimator_parameters.pop('class_weight', None)
                    self.estimator = RandomForestRegressor(
                        **self.estimator_parameters)
                    results.append(('model', 'model type', 'RF quantitative'))
                else:
                    LOG.info("Building Qualitative RF model")
                    self.estimator = RandomForestClassifier(
                        **self.estimator_parameters)
                    results.append(('model', 'model type', 'RF qualitative'))
            except Exception as e:
                LOG.error(f'Exception building RF' 
                          f'estimator with exception {e}')
        self.estimator_temp = copy(self.estimator.fit(X, Y))
        # Create the conformal estimator
        if self.param.getVal('conformal'):
            try:
                LOG.info("Building aggregated conformal RF model")
                if self.param.getVal('quantitative'):
                    underlying_model = RegressorAdapter(self.estimator_temp)
                    normalizing_model = RegressorAdapter(
                        KNeighborsRegressor(n_neighbors=5))
                    # normalizing_model = RegressorAdapter(estimator)
                    normalizer = RegressorNormalizer(
                                    underlying_model,
                                    normalizing_model,
                                    AbsErrorErrFunc())
                    nc = RegressorNc(underlying_model,
                                    AbsErrorErrFunc(), normalizer)
                    # self.conformal_pred = AggregatedCp(IcpRegressor
                    # (RegressorNc(RegressorAdapter(self.estimator))),
                    #                                   BootstrapSampler())

                    self.estimator = AggregatedCp(IcpRegressor(nc),
                                                    BootstrapSampler())
                    self.estimator.fit(X, Y)
                    # overrides non-conformal
                    results.append(
                        ('model', 'model type', 'conformal RF quantitative'))
                # Conformal classifier
                else:
                    self.estimator = AggregatedCp(
                                        IcpClassifier(
                                            ClassifierNc(
                                                ClassifierAdapter(
                                                    self.estimator_temp),
                                                MarginErrFunc())),
                                        BootstrapSampler())
                    # Fit estimator to the data
                    self.estimator.fit(X, Y)
                    results.append(
                        ('model', 'model type', 'conformal RF qualitative'))
            except Exception as e:
                LOG.error(f'Exception building aggregated conformal Random'
                          f'Forest estimator with exception {e}')

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