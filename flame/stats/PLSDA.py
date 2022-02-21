
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


import numpy as np
from flame.stats.base_model import BaseEstimator

from sklearn.cross_decomposition import PLSRegression

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
    
    def inject (self, cmodel):
        self.exo_coef = np.array(cmodel['coef']) 
        self.exo_y = cmodel['ymean']

    def __init__(self, n_components=2, scale=False, max_iter=500,
                 tol=1e-6, copy=True, threshold=None, conformal=False):
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
        self.conformal = conformal

    def predict(self, X, copy=True):
        threshold = self.threshold
        if threshold is None:
            threshold = 0.5

        # if this is a confidential model, use the injected coefficients
        if hasattr(self,'exo_coef'):
            nobj = np.shape(X)[0]
            yp = X @ self.exo_coef 
            yp += self.exo_y
            yp = np.reshape(yp, nobj)
        else:
            yp = super(PLS_da, self).predict(X, copy).ravel()
        yp[yp < threshold] = 0
        yp[yp >= threshold] = 1
        return yp.astype(dtype=float)

    def fit (self, X, Y):
        super(PLS_da, self).fit(X,Y)

        if self.conformal is False:
            return

        if self.estimator_set != None:
            return

        # build a set of 10 additional cv estimators
        # that will be used to compute prediction probabilities 
        param = self.get_params()

        np.random.seed(46)

        splits = min (10, Y.size)

        self.estimator_set = []

        n_selected = 0

        while n_selected < splits:
            train = np.random.choice(Y.size,int(Y.size*0.7), replace=False)

            # make sure that Y train contains compounds from both class 0 and 1
            if len(np.unique(Y[train])) > 1: 
                n_selected += 1
                estimatori = PLS_da (**param)
                
                # we fit a PLSregressor, not a PLS_da! please note 
                super(PLS_da, estimatori).fit(X[train], Y[train])
                self.estimator_set.append(estimatori)

        # strtfdKFold = StratifiedKFold(n_splits=splits)
        # kfold = strtfdKFold.split(X, Y)
        # for (train, test) in kfold:
        #     estimatori = PLS_da (**param)
        #     super(PLS_da, estimatori).fit(X[train], Y[train])
        #     self.estimator_set.append(estimatori)

    def predict_proba(self, X):
        nobj = np.shape (X)[0]

        proba = np.zeros((nobj,2), dtype=int)
        for iestimator in self.estimator_set:
            results = iestimator.predict(X)
            for i in range (nobj):
                if results [i] < self.threshold:
                    proba[i,0]+=1 # increment class 0
                else:
                    proba[i,1]+=1 # increment class 1

        return proba / float(len(self.estimator_set)) 

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

        # Solves back-comptibility issue
        if 'optimize' in self.estimator_parameters:
            self.estimator_parameters.pop("optimize") 
        
        # Scale is hard-coded to False for making use of external scalers        
        self.estimator_parameters['scale'] = False

        self.name = "PLSDA"

        if 'threshold' in self.estimator_parameters:
            self.threshold = self.estimator_parameters['threshold']
        else:
            self.threshold = 0.5

        if self.param.getVal('quantitative'):
            self.conveyor.setError('PLSDA only applies to qualitative data')
            return 

        # For confidential models, create an empty estimator
        if self.param.getVal('confidential'):                
            self.estimator = PLS_da(**self.estimator_parameters)

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

        # in case of conformal models, compute an estimator set to compute prediction probabilities
        self.estimator_parameters['conformal'] = self.param.getVal('conformal')

        # if 'optimize' in self.estimator_parameters:
        #     self.estimator_parameters.pop("optimize") 
            
        if self.param.getVal('tune'):

            opt_param = self.param.getDict('PLSDA_optimize')

            # workaround to solve problem with tolerance values not recognized as float
            if 'tol' in opt_param:
                opt_param['tol'] = [0.000006]
    
            # Optimize estimator using sklearn-gridsearch
            LOG.info('Optimizing PLSDA estimator')
            try:
                self.estimator = PLS_da (**self.estimator_parameters)
                super(PLSDA, self).optimize(X, Y, self.estimator, opt_param)

            except Exception as e:
                return False, f'Error performing SK-LearnGridSearch on PLSDA estimator with exception {e}'

        else:
            LOG.info('Building Qualitative PLSDA with no optimization')
            try:
                self.estimator = PLS_da(**self.estimator_parameters)
            
            except Exception as e:
                return False, f'Error at PLS_da with exception {e}'

        # Fit estimator to the data
        self.regularBuild (X, Y)

        if not self.param.getVal('conformal'):
            return True, results

        # from sklearn.calibration import calibration_curve
        # prepro = self.estimator.predict_proba(X)[:, 1]
        # binned_true_p, binned_predict_p = calibration_curve(Y, prepro, n_bins=10)
        # with open('pproba.txt','w') as f:
        #     for i, j in zip (binned_true_p, binned_predict_p):
        #         f.write (f'{i} \t {j}\n')

        self.estimator_temp = self.estimator
        success, error = self.conformalBuild(X, Y)
        
        if success:
            return True, results

        return False, error
