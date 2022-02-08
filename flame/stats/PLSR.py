
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

from flame.stats.base_model import BaseEstimator
from sklearn.cross_decomposition import PLSRegression
import numpy as np
from flame.util import utils
import os.path
import yaml

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

    def cpredict (self, X, model_file_name):

        nobj, nvar = np.shape(X)

        with open(model_file_name, 'r') as f:
            cmodel = yaml.safe_load (f)

        yp = X @ np.array(cmodel['coef']) 
        yp += cmodel['ymean']
        return np.reshape(yp, nobj)

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
        
    """
    def __init__(self, X, Y, parameters, conveyor):
        # Initialize parent class
        try:
            BaseEstimator.__init__(self,X, Y, parameters, conveyor)
            LOG.debug('Initialize BaseEstimator parent class')
        except Exception as e:
            self.conveyor.setError(f'Error initializing BaseEstimator parent class with exception: {e}')
            return

        # Load estimator parameters
        self.estimator_parameters = self.param.getDict('PLSR_parameters')
        
        # Solves back-comptibility issue
        if 'optimize' in self.estimator_parameters:
            self.estimator_parameters.pop('optimize')

        # Scale is hard-coded to False for making use of external scalers        
        self.estimator_parameters['scale'] = False

        self.name = "PLSR"

        # Check if the model is quantitative
        if not self.param.getVal('quantitative'):
            self.conveyor.setError('PLSR only applies to quantitative data')
            return
        
        # For confidential models, create an empty estimator
        if self.param.getVal('confidential'):
            self.estimator = PLS_r(**self.estimator_parameters)

    def build(self):

        # Make a copy of data matrices
        X = self.X.copy()
        Y = self.Y.copy()

        results = []
        results.append(('nobj', 'number of objects', self.nobj))
        results.append(('nvarx', 'number of predictor variables', self.nvarx))
        results.append(('model', 'model type', 'PLSR'))

        if self.param.getVal('tune'):

            LOG.info('Optimizing PLSR estimator')
            try:
                self.estimator = PLS_r(**self.estimator_parameters)
                self.optimize(X, Y, self.estimator, self.param.getDict('PLSR_optimize'))

            except Exception as e:
                return False, f'Exception optimizing PLS estimator with exception {e}'

        else:
            LOG.info('Building Quantitative PLSR with no optimization')
            try:
                self.estimator = PLS_r(**self.estimator_parameters)

            except Exception as e:
                return False, f'Error at PLS_r with exception {e}'

        # Fit estimator to the data
        self.regularBuild(X, Y)
        
        if self.param.getVal ('confidential'):

            nobj, nvar = np.shape(X)

            cmodel = {}
            cmodel['nobj'] = nobj
            cmodel['nvarx'] = nvar
            cmodel['modelID'] = self.conveyor.getMeta('modelID')
            cmodel['quantitative'] = True
            cmodel['model'] = 'PLSR'
            cmodel['conformal'] = self.param.getVal('conformal')
            cmodel['conformal_confidence'] = self.param.getVal('conformal_confidence')
            cmodel['coef'] = self.estimator.coef_.tolist()
            cmodel['ymean'] = np.mean(Y).tolist()

            model_file_path = utils.model_path(self.param.getVal('endpoint'), 0)
            model_file_name = os.path.join (model_file_path, 'confidential_model.yaml')
            with open(model_file_name, 'w') as f:
                yaml.dump (cmodel, f)

            return True, results

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

    # def predict (self, X, copy=True):

    #     if self.param.getVal('confidential'):
    #         return self.cpredict (X)

    #     return (PLS_r).predict (X,copy)