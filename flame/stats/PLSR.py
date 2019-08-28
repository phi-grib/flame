
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

from copy import copy
from flame.stats.base_model import BaseEstimator

from nonconformist.base import ClassifierAdapter, RegressorAdapter
from nonconformist.acp import AggregatedCp
from nonconformist.acp import BootstrapSampler
from nonconformist.icp import IcpClassifier, IcpRegressor
from nonconformist.nc import ClassifierNc, MarginErrFunc, RegressorNc
from nonconformist.nc import AbsErrorErrFunc, RegressorNormalizer

import numpy as np

from sklearn.cross_decomposition import PLSCanonical, PLSRegression, CCA
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor

from flame.util import get_logger
LOG = get_logger(__name__)


class PLS_r(PLSRegression):
    """
        This class inherits from PLSRegression overwritting predict
        method so the output format is compatible with gridsearchcv
        and keep consistency with other estimators. 

        ...
        
        Attributes
        ----------

        threshold : float
            number between 0 and 1 taken as cutoff for class assignation
        
        Methods
        -------

        predict(X)
            Overwrites parent class predict by replacing the continuous 
            value of PLSR regression by a class (0 or 1) according 
            to the established threshold
        
    """
    def predict(self, X, copy=True):
        # Apply ravel to the list of predictions so a vector is 
        # returned instead of a matrix
        results = super(PLS_r, self).predict(X, copy=True).ravel()
        return results


class PLSR(BaseEstimator):
    """
        This class inherits from BaseEstimator and wraps SKLEARN
        PLSR estimator

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

        optimize( X, Y, estimator, tune_parameters)
            Gridsearch specially designed for PLSR.
            Optimizes number of variables
        
    """
    def __init__(self, X, Y, parameters, conveyor):
        # Initialize parent class
        try:
            BaseEstimator.__init__(self,X, Y, parameters, conveyor)
            LOG.debug('Initialize BaseEstimator parent class')
        except Exception as e:
            LOG.error(f'Error initializing BaseEstimator parent class with exception: {e}')
            self.conveyor.setError(f'Error initializing BaseEstimator parent class with exception: {e}')
            return

        self.estimator_parameters = self.param.getDict('PLSR_parameters')

        self.name = "PLSR"

        # Check if the model is quantitative
        if not self.param.getVal('quantitative'):
            LOG.error('PLSR only applies to quantitative data')
            self.conveyor.setError('PLSR only applies to quantitative data')
            return

    def build(self):

        # Make a copy of data matrices
        X = self.X.copy()
        Y = self.Y.copy()

        results = []
        results.append(('nobj', 'number of objects', self.nobj))
        results.append(('nvarx', 'number of predictor variables', self.nvarx))

        if self.param.getVal('tune'):

            # Optimize estimator using sklearn-gridsearch
            if self.estimator_parameters['optimize'] == 'auto':
                try:

                    LOG.info('Optimizing PLSR using SK-LearnGridSearch')

                    # Remove optimize key from parameter dictionary
                    # to avoid sklearn estimator error (unexpected keyword)
                    self.estimator_parameters.pop("optimize")   

                    super(PLSR, self).optimize(X, Y, PLS_r(
                        **self.estimator_parameters), 
                        self.param.getDict('PLSR_optimize'))

                except Exception as e:
                    LOG.error(f'Error performing SK-LearnGridSearch'
                              f' on PLSR estimator with exception {e}')
                    return False, f'Error performing SK-LearnGridSearch on PLSR estimator with exception {e}'

            # Optimize using flame implementation (recommended)
            elif self.estimator_parameters['optimize'] == 'manual':

                LOG.info('Optimizing PLSR using manual method')

                # Remove optimize key from parameter dictionary
                # to avoid sklearn estimator error (unexpected keyword)
                self.estimator_parameters.pop("optimize")   

                success, message = self.optimize(X, Y, PLS_r(
                    **self.estimator_parameters), 
                    self.param.getDict('PLSR_optimize'))

                if not success:
                    return False, message

            else: 
                LOG.error('Type of tune not recognized, check the input')
                return False, 'Type of tune not recognized, check the input'    

            results.append(('model', 'model type', 'PLSR quantitative (optimized)'))

        else:
            LOG.info('Building Quantitative PLSR with no optimization')
            try:
                # Remove optimize key from parameters to avoid error
                self.estimator_parameters.pop("optimize") 

                # as the sklearn estimator does not have this key
                self.estimator = PLS_r(**self.estimator_parameters)
            except Exception as e:
                LOG.error(f'Error at PLS_r instantiation with '
                          f'exception {e}')
                return False, f'Error at PLS_da instantiation with exception {e}'

            results.append(('model', 'model type', 'PLSR quantitative'))
        
        # Fit estimator to the data
        self.estimator.fit(X, Y)

        if not self.param.getVal('conformal'):
            return True, results

        self.estimator_temp = copy(self.estimator)
        try:
            
            LOG.info('Building PLSR aggregated conformal predictor')

            underlying_model = RegressorAdapter(self.estimator_temp)
            # normalizing_model = RegressorAdapter(
            #     KNeighborsRegressor(n_neighbors=1))
            normalizing_model = RegressorAdapter(self.estimator_temp)
            normalizer = RegressorNormalizer(underlying_model, normalizing_model, AbsErrorErrFunc())

            nc = RegressorNc(underlying_model, AbsErrorErrFunc(), normalizer)
            self.estimator = AggregatedCp(IcpRegressor(nc), BootstrapSampler())

        except Exception as e:
            LOG.error(f'Error building aggregated PLSR conformal'
                        f' regressor with exception: {e}')
            return False, f'Error building aggregated PLSR conformal regressor with exception: {e}'

            # self.conformal_pred = AggregatedCp(IcpRegressor(
            # RegressorNc(RegressorAdapter(self.estimator))),
            #                                    BootstrapSampler())

        # Fit conformal estimator to the data
        self.estimator.fit(X, Y)

        # overrides non-conformal
        results.append(('model', 'model type', 'conformal PLSR quantitative'))

        return True, results

    def optimize(self, X, Y, estimator, tune_parameters):
        ''' optimizes a model using a grid search over a 
        range of values for diverse parameters'''
 
        # Max number of latent variables
        latent_variables = tune_parameters['n_components']
        
        # Best r2
        r2 = 0
        estimator0 = None
        
        # List to add r2 for each number of latent variables
        list_latent = []
        try:
            for n_comp in latent_variables:
                r2_0 = 0
                estimator.set_params(**{"n_components": n_comp})
                y_pred = cross_val_predict(estimator, X, Y,
                                             cv=self.cv, n_jobs=1)
                r2_0 = r2_score(Y, y_pred)

                # print (n_comp, r2_0)

                # Update estimator0 to best current estimator
                if r2_0 >= r2 or estimator0 == None:
                    r2 = r2_0
                    estimator0 = copy(estimator)

                list_latent.append([n_comp, r2_0])
        except Exception as e:
            return False, f'Error optimizing PLSR with exception {e}'

        self.estimator = estimator0

        # LOG.debug('Number of latent variables, r2')
        # for lv in list_latent:
        #     LOG.debug(f'Number of latent variables: {lv[0]} \nr2: {lv[1]}\n')

        self.estimator.fit(X, Y)
        LOG.info(f'Estimator best parameters: {self.estimator.get_params()}')
        return True, 'OK'

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