
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
import numpy as np

from flame.stats.base_model import BaseEstimator
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

    def __init__(self, X, Y, parameters, conveyor):
        # Initialize parent class
        try:
            BaseEstimator.__init__(self, X, Y, parameters, conveyor)
            LOG.debug('Initializing BaseEstimator parent class')
        except Exception as e:
            self.conveyor.setError(f'Error initializing BaseEstimator parent class with exception: {e}')
            LOG.error(f'Error initializing BaseEstimator parent class with exception: {e}')
            return

        # Load estimator parameters                
        GNB_parameters = self.param.getDict('GNB_parameters')
        priors = [0.0, 0.0]
        if 'prior_negative' in GNB_parameters and GNB_parameters['prior_negative'] != None:
            priors[0] = GNB_parameters['prior_negative']
        if 'prior_positive' in GNB_parameters and GNB_parameters['prior_positive'] != None:
            priors[1] = GNB_parameters['prior_positive'] 
        
        self.estimator_parameters = {}
        if GNB_parameters['var_smoothing'] is not None:
            self.estimator_parameters['var_smoothing'] = GNB_parameters['var_smoothing']

        if priors[0]!=0.0 and priors[1]!=0.0:
            if priors[0]+priors[1] != 1.0:
                LOG.error(f'GNB: the sum of the priors should be 1. priors set to {priors[0], priors[1]} ')
                priors[1] = 1.0 - priors[0]
            self.estimator_parameters['priors'] = priors
            # self.param.setInnerVal('GNB_parameters','priors',priors)

        if self.param.getVal('quantitative'):
            self.conveyor.setError('GNB only applies to qualitative data')
            LOG.error('GNB only applies to qualitative data')
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
        results.append(('model', 'model type', 'GNB'))

        # Build estimator
        LOG.info('Building GaussianNB model')
        try:
            self.estimator = GaussianNB(**self.estimator_parameters)

            self.regularBuild (X, Y)
        except Exception as e:
            return False, f'Exception building GNB estimator with exception {e}'

        if not self.param.getVal('conformal'):
            return True, results

        self.estimator_temp = self.estimator

        success, error = self.conformalBuild(X, Y)
        if success:
            return True, results
        else:
            return False, error
        