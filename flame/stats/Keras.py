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

from copy import copy

from nonconformist.base import ClassifierAdapter, RegressorAdapter
from nonconformist.acp import AggregatedCp
from nonconformist.acp import BootstrapSampler
from nonconformist.icp import IcpClassifier, IcpRegressor
from nonconformist.nc import ClassifierNc, MarginErrFunc, RegressorNc
from nonconformist.nc import AbsErrorErrFunc, RegressorNormalizer
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from keras.models import Sequential
from keras.layers import Dense
from sklearn.base import clone
import keras

import numpy as np

from flame.stats.base_model import BaseEstimator
from flame.util import get_logger
LOG = get_logger(__name__)


class Keras_nn(BaseEstimator):
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
        self.estimator_parameters = self.param.getDict('Keras_parameters')

        # Load tune parameters
        self.tune_parameters = self.param.getDict('Keras_optimize')

        if self.param.getVal('quantitative'):
            self.name = "Keras-Regressor"
        else:
            self.name = "Keras-Classifier"

    def build(self):
        '''Build a new DL model with the X and Y numpy matrices '''


        try:
            from keras.wrappers.scikit_learn import KerasClassifier
            from keras.wrappers.scikit_learn import KerasRegressor       
        except Exception as e:
            return False,  'Keras not found, please revise your environment'

        # Make a copy of data matrices
        X = self.X.copy()
        Y = self.Y.copy()

        results = []
        results.append(('nobj', 'number of objects', self.nobj))
        results.append(('nvarx', 'number of predictor variables', self.nvarx))

        # If tune then call gridsearch to optimize the estimator
        if self.param.getVal('tune'):

            LOG.info("Optimizing Keras estimator")
            
            try:
                # Check type of model
                if self.param.getVal('quantitative'):
                    self.estimator = KerasRegressor(
                                        **self.estimator_parameters)
                    self.optimize(X, Y, self.estimator, self.tune_parameters)
                    results.append(('model','model type','KERAS quantitative (optimized)'))
                else:
                    self.estimator = KerasClassifier(
                                        **self.estimator_parameters)
                    #params = self.estimator.get_params()
                    #params['num_class'] = 2
                    self.optimize(X, Y, self.estimator,
                                  self.tune_parameters)
                    results.append(('model','model type','KERAS qualitative (optimized)'))

            except Exception as e:
                return False, f'Exception optimizing KERAS estimator with exception {e}'
            
        else:
            try:
                if self.param.getVal('quantitative'):

                    LOG.info("Building Quantitative KERAS mode")
                    self.estimator = KerasRegressor(build_fn=self.create_model, 
                    **self.estimator_parameters, verbose=0)
                    results.append(('model', 'model type', 'Keras quantitative'))
                else:
                    print(self.estimator_parameters)
                    LOG.info("Building Qualitative Keras model")
                    self.estimator = KerasClassifier(build_fn=self.create_model, dim=self.X.shape[1],
                     **self.estimator_parameters, verbose=0)
                    results.append(('model', 'model type', 'Keras qualitative'))

                self.estimator.fit(X, Y)
                print(self.estimator)

            except Exception as e:
                raise e
                return False, f'Exception building Keras estimator with exception {e}'

        self.estimator_temp = clone(self.estimator)

        if not self.param.getVal('conformal'):
            return True, results
        # Create the conformal estimator
        try:
            # Conformal regressor
            if self.param.getVal('quantitative'):

                LOG.info("Building conformal Quantitative Keras model")

                underlying_model = RegressorAdapter(self.estimator_temp)
                normalizing_model = RegressorAdapter(
                    KNeighborsRegressor(n_neighbors=15))
                # normalizing_model = RegressorAdapter(self.estimator_temp)
                normalizer = RegressorNormalizer(
                                underlying_model,
                                normalizing_model,
                                AbsErrorErrFunc())
                nc = RegressorNc(underlying_model,
                                    AbsErrorErrFunc(),
                                    normalizer)

                # self.conformal_pred = AggregatedCp(IcpRegressor
                # (RegressorNc(RegressorAdapter(self.estimator))),
                #                                   BootstrapSampler())

                self.estimator = AggregatedCp(IcpRegressor(nc),
                                                BootstrapSampler())

                self.estimator.fit(X, Y)
                results.append(('model', 'model type', 'conformal Keras quantitative'))

            # Conformal classifier
            else:

                LOG.info("Building conformal Qualitative Keras model")

                self.estimator = AggregatedCp(
                                    IcpClassifier(
                                        ClassifierNc(
                                            ClassifierAdapter(self.estimator_temp),
                                            MarginErrFunc()
                                        )
                                    ),
                                    BootstrapSampler())

                # Fit estimator to the data
                print('build finished')
                self.estimator.fit(X, Y)
                results.append(('model', 'model type', 'conformal Keras qualitative'))

        except Exception as e:
            raise e
            return False, f'Exception building conformal Keras estimator with exception {e}'

        return True, []

# Function to create model, required for KerasClassifier
    def create_model(self, dim=568):
        # create model
        model = Sequential()
        model.add(Dense(50, input_dim=dim, activation='relu'))
        model.add(Dense(20, activation='sigmoid'))
        model.add(Dense(1, activation='sigmoid'))
        # Compile model
        
        #if self.param.getVal('quantitative'):
            #loss = 'mean_squared_error'
        #else:
        loss = 'binary_crossentropy'
        optimizer = keras.optimizers.Adam(lr=0.1)
        model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
        # model.compile(
        #             optimizer=keras.optimizers.Adam(
        #                 hp.Choice('learning_rate',
        #                 values=[1e-2, 1e-3, 1e-4])),
        #             loss='sparse_categorical_crossentropy',
        #             metrics=['accuracy'])
        
        return model

    # Overrides regular project to single class prediction
    def regularProject(self, Xb):
        ''' projects a collection of query objects in a regular model,
         for obtaining predictions '''
        Yp = self.estimator.predict(Xb)
        Yp = np.asarray([x[0] for x in Yp])

        # if conveyor contains experimental values for any of the objects replace the
        # predictions with the experimental results
        exp = self.conveyor.getVal('experim')
        if exp is not None:
            if len(exp) == len(Yp):
                for i in range (len(Yp)):
                    if not np.isnan(exp[i]):
                        # print (exp[i], Yp[i])
                        Yp[i] = exp[i]
                    else:
                    # if exp is nan, substitute it with a number which can be recognized
                    # to facilitate handling and do not replace Yp
                        exp[i]= float ('-99999')

        self.conveyor.addVal(Yp, 'values', 'Prediction',
                        'result', 'objs',
                        'Results of the prediction', 'main')
    # Overrides base_model validation
    # def validate(self,):
    #     results ['quality'] = []
    #     return True, None
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


    def optimize(self, X, Y, estimator, tune_parameters):
        # TODO Perhaps not suitable
        ''' optimizes a model using a grid search over a range of values for diverse parameters'''
        LOG.info('Keras model parameter optimization not implemented. skipping....')
    # def regularProject(self, Xb, results):
    #     ''' projects a collection of query objects in a regular model, for obtaining predictions '''


    # def conformalProject(self, Xb, results):
    #     ''' projects a collection of query objects in a conformal model, for obtaining predictions '''


    # def project(self, Xb, results):
    #     ''' Uses the X matrix provided as argument to predict Y'''