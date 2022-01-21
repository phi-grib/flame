
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
import numpy as np

from sklearn.cross_decomposition import PLSCanonical, PLSRegression, CCA
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import r2_score

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

        # Load estimator parameters
        self.estimator_parameters = self.param.getDict('PLSR_parameters')

        # Scale is hard-coded to False for making use of external scalers        
        self.estimator_parameters['scale'] = False

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
        results.append(('model', 'model type', 'PLSR'))

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

        else:
            LOG.info('Building Quantitative PLSR with no optimization')
            try:
                # Remove optimize key from parameters to avoid error
                self.estimator_parameters.pop("optimize") 

                # as the sklearn estimator does not have this key
                self.estimator = PLS_r(**self.estimator_parameters)
            except Exception as e:
                LOG.error(f'Error at PLS_r with exception {e}')
                return False, f'Error at PLS_r with exception {e}'

        # Fit estimator to the data
        self.regularBuild(X, Y)
        
        # The model coefficients can be easily extracted and stored for building 
        # confidential models. These coefficients can be used to predict the properties of
        # new compounds, simply by multiplying the X ( X @ coef ), as shown below
        # coef = self.estimator.coef_
        # print (coef)
        # yp = self.estimator.predict(X)
        # yp2 = X @ coef 
        # yp2 += np.mean(Y)
        # yp2 = np.reshape(yp2, (self.nobj))
        # print (yp, yp2)

        if not self.param.getVal('conformal'):
            return True, results

        self.estimator_temp = self.estimator
        success, error = self.conformalBuild(X, Y)
        if success:
            return True, results
        else:
            return False, error

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
                # y_pred = cross_val_predict(estimator, X, Y, cv=self.cv, n_jobs=-1)
                y_pred = cross_val_predict(estimator, X, Y, cv=self.cv, n_jobs=self.cross_jobs)
                
                r2_0 = r2_score(Y, y_pred)

                # Update estimator0 to best current estimator
                if r2_0 >= r2 or estimator0 == None:
                    r2 = r2_0
                    estimator0 = copy(estimator)

                list_latent.append([n_comp, r2_0])
        except Exception as e:
            return False, f'Error optimizing PLSR with exception {e}'

        self.estimator = estimator0

        self.estimator.fit(X, Y)
        LOG.info(f'Estimator best parameters: {self.estimator.get_params()}')
        return True, 'OK'
