
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


# To ignore warnings comming from data precision in Cross-validation
# Study more in deep

from copy import copy
from flame.stats.base_model import BaseEstimator

from nonconformist.base import ClassifierAdapter, RegressorAdapter
from nonconformist.acp import AggregatedCp
from nonconformist.acp import BootstrapSampler
from nonconformist.icp import IcpClassifier, IcpRegressor
from nonconformist.nc import ClassifierNc, MarginErrFunc, RegressorNc

import numpy as np

from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn.metrics import matthews_corrcoef as mcc 
from sklearn.metrics import f1_score as f1
from sklearn.cross_decomposition import PLSCanonical
from sklearn.cross_decomposition import PLSRegression, CCA

import warnings
warnings.filterwarnings('ignore')

from flame.util import get_logger
LOG = get_logger(__name__)


class PLS_da(PLSRegression):
    """
        This class inherits from PLSRegression overwritting predict
        method so continuous value is replaced by a class assignation
        according to a given threshold

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

    def __init__(self, n_components=2, scale=False, max_iter=500,
                 tol=1e-6, copy=True, threshold=None):
        # Initialize parent class
        try:
            super(PLS_da, self).__init__(n_components=n_components,
                                        scale=scale, max_iter=max_iter,
                                         tol=tol, copy=copy)
        except Exception as e:
            LOG.error(f'Error initializing PLSRegression parent class with exception: {e}')
            # self.conveyor.setError(f'Error initializing PLSRegression parent class with exception: {e}')
            return 

        # Cut-off for class assignation
        self.threshold = threshold

    # Overwrites parent class predict
    def predict(self, X, copy=True):

        threshold = self.threshold
        if threshold is None:
            return super(PLS_da, self).predict(X, copy=True).ravel()

        results = super(PLS_da, self).predict(X, copy=True).ravel()
        results[results < threshold] = 0
        results[results >= threshold] = 1
        results = results.astype(dtype=float)
        return results


class PLSDA(BaseEstimator):
    """
        This class inherits from BaseEstimator and wraps the 
        class PLS_da able to return class assignment.

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
            Gridsearch specially designed for PLS-DA.
            Optimizes cutoff and number of latent variables
        
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

        # Load estimator parameters
        self.estimator_parameters = self.param.getDict('PLSDA_parameters')
        
        # Scale is hard-coded to False for making use of external scalers        
        self.estimator_parameters['scale'] = False

        self.name = "PLSDA"

        if self.param.getVal('quantitative'):
            LOG.error('PLSDA only applies to qualitative data')
            self.conveyor.setError('PLSDA only applies to qualitative data')
            return 

        if self.param.getVal('conformal'):
            LOG.error('Conformal prediction no implemented in PLSDA yet')
            self.conveyor.setError('Conformal prediction no implemented in PLSDA yet')
            return 

    def build(self):
        '''Build a new PLSDA model with the X and Y numpy matrices '''

        # Make a copy of data matrices
        X = self.X.copy()
        Y = self.Y.copy()

        results = []
        results.append(('nobj', 'number of objects', self.nobj))
        results.append(('nvarx', 'number of predictor variables', self.nvarx))
        results.append(('model', 'model type', 'PLSDA'))


        if self.param.getVal('tune'):

            opt_param = self.param.getDict('PLSDA_optimize')

            # workaround to solve problem with tolerance values not recognized as float
            if 'tol' in opt_param:
                opt_param['tol'] = [np.float(i) for i in opt_param['tol']]

            # Optimize estimator using sklearn-gridsearch
            if self.estimator_parameters['optimize'] == 'auto':
                try:

                    LOG.info('Optimizing PLSDA using SK-LearnGridSearch')
                    
                    super(PLSDA, self).optimize(X, Y, 
                                            PLS_da(n_components=2,
                                            scale=False, max_iter=500,
                                            tol=0.000006, copy=True,
                                            threshold=0.5), 
                                            opt_param)

                except Exception as e:
                    LOG.error(f'Error performing SK-LearnGridSearch'
                              f' on PLSDA estimator with exception {e}')
                    return False, f'Error performing SK-LearnGridSearch on PLSDA estimator with exception {e}'

            # Optimize using flame implementation (recommended)
            elif self.estimator_parameters['optimize'] == 'manual':
                
                LOG.info('Optimizing PLSDA using manual method')
                
                success, message = self.optimize(X, Y,
                                        PLS_da(n_components=2, 
                                        scale=False, max_iter=500,
                                        tol=1e-6, copy=True, 
                                        threshold=None),
                                        opt_param)
                if not success:
                    return False, message

            else:

                LOG.error('Type of tune not recognized, check the input')
                
                return False, 'Type of tune not recognized, check the input'

            # results.append(('model', 'model type', 'PLSDA qualitative (optimized)'))

        else:
            LOG.info('Building Qualitative PLSDA with no optimization')
            try:
                # Remove optimize key from parameters to avoid error
                self.estimator_parameters.pop("optimize")   

                # as the sklearn estimator does not have this key
                self.estimator = PLS_da(**self.estimator_parameters)
            except Exception as e:
                LOG.error(f'Error at PLS_da instantiation with '
                          f'exception {e}')
                return False, f'Error at PLS_da instantiation with exception {e}'

            # results.append(('model', 'model type', 'PLSDA qualitative'))

        # Fit estimator to the data
        self.estimator.fit(X, Y)

        return True, results

    def optimize(self, X, Y, estimator, tune_parameters):
        ''' optimizes a model using a grid search over a 
        range of values for diverse parameters'''

        # Max number of latent variables
        latent_variables = tune_parameters["n_components"]

        # Mathew correlation coefficient of best threshold
        mcc_final = 0
        estimator0 = ""

        # List to add the best threshold and Matthews correlation
        # coefficient for each number of latent variables
        list_latent = []
        try:
            for n_comp in latent_variables:
                mcc0 = 0
                estimator.set_params(**{"n_components": n_comp})
                y_pred = cross_val_predict(estimator, X, Y, cv=self.cv, n_jobs=-1)
                estimator1 = ""
                threshold_1 = 0
                # Get optimum threshold
                for threshold in range(0, 100, 5):
                    threshold = threshold / 100
                    y_pred2 = copy(y_pred)
                    y_pred2[y_pred2 < threshold] = 0
                    y_pred2[y_pred2 >= threshold] = 1
                    mcc1 = mcc(Y, y_pred2)
                    # Update threshold value with current best value
                    if mcc1 >= mcc0:
                        mcc0 = mcc1
                        estimator1 = copy(estimator)
                        estimator1.set_params(**{'threshold': threshold})
                        threshold_1 = (threshold)
                # Assign class estimator the best current estimator
                if mcc0 >= mcc_final:
                    mcc_final = mcc0
                    estimator0 = copy(estimator1)
                    self.estimator = estimator0

                list_latent.append([n_comp, threshold_1, mcc0])
        except Exception as e:
            LOG.error(f'Error optimizing PLS-DA with exception {e}')
            return False, f'Error optimizing PLS-DA with exception {e}'

        # LOG.debug('Number of latent variables, Best cutoff, and its Matthews '
        # 'correlation coefficient')
        # for lv in list_latent:
        #     LOG.debug(f'Number of latent variables: '
        #     f'{lv[0]} \nBest cutoff: {lv[1]} \nMCC: {lv[2]}\n')
                  
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