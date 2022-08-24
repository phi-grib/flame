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

from flame.stats.base_model import BaseEstimator
from flame.util import get_logger
LOG = get_logger(__name__)

class XGBOOST(BaseEstimator):
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

        try:
            import xgboost as xgb
            xgb.set_config(verbosity=0)
        except:
            LOG.error ('XGboost not found, please revise your environment')

        # Load estimator parameters        
        self.estimator_parameters = self.param.getDict('XGBOOST_parameters')

        # Load tune parameters
        self.tune_parameters = self.param.getDict('XGBOOST_optimize')

        if self.param.getVal('quantitative'):
            self.estimator_parameters['objective'] = 'reg:squarederror'
            
            self.name = "XGB-Regressor"
        else:
            self.estimator_parameters['objective'] = 'binary:logistic'
            self.estimator_parameters['eval_metric'] = 'logloss'
            self.estimator_parameters['use_label_encoder'] = False

            self.name = "XGB-Classifier"

        # Missing value must be defined. Otherwyse it returns 'nan' which cannot be
        # converted to JSON and produces trouble in different points
        self.estimator_parameters['missing'] = -99.99999

    def build(self):
        '''Build a new XGBOOST model with the X and Y numpy matrices '''

        try:
            from xgboost.sklearn import XGBClassifier
            from xgboost.sklearn import XGBRegressor
        except Exception as e:
            return False,  'XGboost not found, please revise your environment'

        # Make a copy of data matrices
        X = self.X.copy()
        Y = self.Y.copy()

        results = []
        results.append(('nobj', 'number of objects', self.nobj))
        results.append(('nvarx', 'number of predictor variables', self.nvarx))
        results.append(('model','model type','XGBOOST'))


        # If tune then call gridsearch to optimize the estimator
        if self.param.getVal('tune'):

            LOG.info("Optimizing XGBOOST estimator")
            
            try:
                # Check type of model
                if self.param.getVal('quantitative'):
                    self.estimator = XGBRegressor(**self.estimator_parameters)
                    self.optimize(X, Y, self.estimator, self.tune_parameters)
                else:
                    self.estimator = XGBClassifier(**self.estimator_parameters)
                    params = self.estimator.get_params()
                    params['num_class'] = 2
                    self.optimize(X, Y, self.estimator, self.tune_parameters)

            except Exception as e:
                return False, f'Exception optimizing XGBOOST estimator with exception {e}'
            
        else:
            try:
                if self.param.getVal('quantitative'):
                    LOG.info("Building Quantitative XGBOOST model")

                    self.estimator = XGBRegressor(**self.estimator_parameters)
                else:
                    LOG.info("Building Qualitative XGBOOST model")

                    self.estimator = XGBClassifier(**self.estimator_parameters)

                # self.estimator.fit(X, Y)
                # self.feature_importances = self.estimator.feature_importances_
                self.regularBuild(X,Y)

            except Exception as e:
                return False, f'Exception building XGBOOST estimator with exception {e}'

        if not self.param.getVal('conformal'):
            return True, results

        self.estimator_temp = self.estimator
        success, error = self.conformalBuild(X, Y)
        if success:
            return True, results
        else:
            return False, error
