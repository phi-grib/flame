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

        # patch to solve bug in the object_type definition of max_depth
        # should not appear from versions older than 20/01/2021
        if 'max_depth' in self.estimator_parameters:
            v = self.estimator_parameters['max_depth']
            if v is not None:
                try:
                    self.estimator_parameters['max_depth']=int(v)
                except:
                    self.estimator_parameters['max_depth']=None

        # Load tune parameters
        self.tune_parameters = self.param.getDict('RF_optimize')

        if self.param.getVal('quantitative'):
            self.name = "RF-R"
            self.tune_parameters.pop("class_weight")
            self.estimator_parameters.pop("class_weight")
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
        results.append(('model','model type','RF'))

        conformal = self.param.getVal('conformal')
        # If tune then call gridsearch to optimize the estimator
        if self.param.getVal('tune'):

            LOG.info("Optimizing RF estimator")
            
            try:
                # Check type of model
                if self.param.getVal('quantitative'):
                    self.estimator = RandomForestRegressor(
                                        **self.estimator_parameters)
                    self.optimize(X, Y, self.estimator, self.tune_parameters)
                else:
                    self.estimator = RandomForestClassifier(
                                        **self.estimator_parameters)
                    self.optimize(X, Y, self.estimator, self.tune_parameters)

            except Exception as e:
                return False, f'Exception optimizing RF estimator with exception {e}'
            
        else:
            try:
                if self.param.getVal('quantitative'):
                    LOG.info("Building Quantitative RF model")

                    self.estimator = RandomForestRegressor(
                                        **self.estimator_parameters)

                else:
                    LOG.info("Building Qualitative RF model")

                    self.estimator = RandomForestClassifier(
                                        **self.estimator_parameters)

                self.estimator.fit(X, Y)

            except Exception as e:
                return False, f'Exception building RF estimator with exception {e}'


        if not conformal:
            return True, results

        self.estimator_temp = self.estimator
        success, error = self.conformalBuild(X, Y)
        if success:
            return True, results
        else:
            return False, error


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