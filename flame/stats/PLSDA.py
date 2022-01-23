
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

import copy
from flame.stats.base_model import BaseEstimator

import numpy as np

from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import matthews_corrcoef as mcc 
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import StratifiedKFold

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
            self.conveyor.setError(f'Error initializing PLSRegression parent class with exception: {e}')
            return 

        # Cut-off for class assignation
        self.threshold = threshold
        self.estimator_set = None

    # Overwrites parent class predict
    def predict(self, X, copy=True):

        threshold = self.threshold
        if threshold is None:
            return super(PLS_da, self).predict(X, copy).ravel()

        results = super(PLS_da, self).predict(X, copy).ravel()
        results[results < threshold] = 0
        results[results >= threshold] = 1
        results = results.astype(dtype=float)
        return results

    def fit (self, X, Y):

        super(PLS_da, self).fit(X,Y)

        if self.estimator_set != None:
            return

        param = self.get_params()

        nobj, nvarx = np.shape (X)

        splits = min (10, nobj)

        strtfdKFold = StratifiedKFold(n_splits=splits)
        kfold = strtfdKFold.split(X, Y)
        
        self.estimator_set = []
        for k, (train, test) in enumerate(kfold):
            estimatori = PLS_da (**param)
            super(PLS_da, estimatori).fit(X[train], Y[train])
            self.estimator_set.append(estimatori)

    def predict_proba(self, X):
        nobj, nvarx = np.shape (X)

        proba = np.zeros((nobj,2), dtype=int)
        for iestimator in self.estimator_set:
            results = iestimator.predict(X)
            for i in range (nobj):
                if results [i] < self.threshold:
                    proba[i,0]+=1
                else:
                    proba[i,1]+=1

        return proba / len(self.estimator_set)

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

        # 'PLS_da' object has no attribute 'predict_proba', required for conformal models
        # if self.param.getVal('conformal'):
        #     self.conveyor.setError('Conformal prediction no implemented in PLSDA yet')
        #     return 



    def build(self):
        '''Build a new PLSDA model with the X and Y numpy matrices '''

        # Make a copy of data matrices
        X = self.X.copy()
        Y = self.Y.copy()

        results = []
        results.append(('nobj', 'number of objects', self.nobj))
        results.append(('nvarx', 'number of predictor variables', self.nvarx))
        results.append(('model', 'model type', 'PLSDA'))

        if 'optimize' in self.estimator_parameters:
            self.estimator_parameters.pop("optimize") 

        if self.param.getVal('tune'):

            opt_param = self.param.getDict('PLSDA_optimize')

            # workaround to solve problem with tolerance values not recognized as float
            if 'tol' in opt_param:
                opt_param['tol'] = [0.000006]
    

            # Optimize estimator using sklearn-gridsearch
            LOG.info('Optimizing PLSDA using SK-LearnGridSearch')
            try:
                self.estimator = PLS_da (**self.estimator_parameters)
                super(PLSDA, self).optimize(X, Y, self.estimator, opt_param)

            except Exception as e:
                return False, f'Error performing SK-LearnGridSearch on PLSDA estimator with exception {e}'

                
            # results.append(('model', 'model type', 'PLSDA qualitative (optimized)'))

        else:
            LOG.info('Building Qualitative PLSDA with no optimization')
            try:
                self.estimator = PLS_da(**self.estimator_parameters)
            except Exception as e:
                return False, f'Error at PLS_da with exception {e}'

            # results.append(('model', 'model type', 'PLSDA qualitative'))

        # Fit estimator to the data
        self.regularBuild (X, Y)

        if not self.param.getVal('conformal'):
            return True, results

        self.estimator_temp = self.estimator
        
        # approach predict proba by building a set of 10 estimators by CV
        # PROBA


        
        success, error = self.conformalBuild(X, Y)
        
        if success:
            return True, results
        else:
            return False, error
