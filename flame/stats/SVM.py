
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
from flame.stats.base_model import getCrossVal
from flame.stats.scale import scale, center
from flame.stats.model_validation import CF_QuanVal

from sklearn import svm

from nonconformist.base import ClassifierAdapter, RegressorAdapter
from nonconformist.acp import AggregatedCp
from nonconformist.acp import BootstrapSampler
from nonconformist.icp import IcpClassifier, IcpRegressor
from nonconformist.nc import ClassifierNc, MarginErrFunc, RegressorNc
from sklearn.neighbors import KNeighborsRegressor
from nonconformist.nc import AbsErrorErrFunc, RegressorNormalizer
from flame.util import get_logger

LOG = get_logger(__name__)

class SVM(BaseEstimator):
    """
        This class inherits from BaseEstimator and wraps SKLEARN
        SVClassifier or SVRegressor estimator

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
    def __init__(self, X, Y, parameters):
        # Initialize parent class
        try:
            super(SVM, self).__init__(X, Y, parameters)
            LOG.debug('Initialize BaseEstimator parent class')
        except Exception as e:
            LOG.error(f'Error initializing BaseEstimator parent'
                    f'class with exception: {e}')
            raise e
        # Load estimator parameters
        self.estimator_parameters = self.param.getVal('SVM_parameters')
        # Load tune parameters
        self.tune_parameters = self.param.getVal('SVM_optimize')
        if self.param.getVal('quantitative'):
            self.name = "SVM-R"
            # Remove parameters of SVC class and set the name
            self.estimator_parameters.pop("class_weight", None)
            self.estimator_parameters.pop("probability", None)
            self.estimator_parameters.pop("decision_function_shape", None)
            self.estimator_parameters.pop("random_state", None)
            self.tune_parameters.pop("class_weight", None)
            self.tune_parameters.pop("random_state", None)

        else:
            # Remove parameters of SVR class and set the name
            self.estimator_parameters.pop("epsilon", None)
            self.name = "SVM-C"

    def build(self):
        '''Build a new SVM model with the X and Y numpy matrices'''

        # Make a copy of data matrices
        X = self.X.copy()
        Y = self.Y.copy()

        results = []
        results.append(('nobj', 'number of objects', self.nobj))
        results.append(('nvarx', 'number of predictor variables', self.nvarx))
       
        # If tune then call gridsearch to optimize the estimator
        if self.param.getVal('tune'):
            try:
                # Check type of model
                if self.param.getVal('quantitative'):
                    self.optimize(X, Y, svm.SVR(), self.tune_parameters)
                    results.append(
                        ('model', 'model type', 'SVM quantitative (optimized)'))

                else:
                    self.optimize(X, Y, svm.SVC(probability=True),
                                self.tune_parameters)
                    results.append(
                        ('model', 'model type', 'SVM qualitative (optimized)'))
                LOG.debug('SVM estimator optimized')
            except Exception as e:
                LOG.error(f'Exception optimizing SVM' 
                        f'estimator with exception {e}')            
        else:
            try:
                LOG.info("Building  SVM model")
                if self.param.getVal('quantitative'):
                    LOG.info("Building Quantitative SVM-R model")
                    self.estimator = svm.SVR(**self.estimator_parameters)
                    results.append(('model', 'model type', 'SVM quantitative'))
                else:
                    self.estimator = svm.SVC(**self.estimator_parameters)
                    results.append(('model', 'model type', 'SVM qualitative'))
            except Exception as e:
                LOG.error(f'Exception building SVM' 
                        f'estimator with exception {e}')

        if self.param.getVal('conformal'):
            try:
                LOG.info("Building aggregated conformal SVM model")
                if self.param.getVal('quantitative'):
                    underlying_model = RegressorAdapter(self.estimator)
                    normalizing_model = RegressorAdapter(
                        KNeighborsRegressor(n_neighbors=5))
                    normalizing_model = RegressorAdapter(self.estimator)
                    normalizer = RegressorNormalizer(
                        underlying_model, normalizing_model, AbsErrorErrFunc())
                    nc = RegressorNc(underlying_model,
                                    AbsErrorErrFunc(), normalizer)
                    # self.conformal_pred = AggregatedCp(IcpRegressor(
                    # RegressorNc(RegressorAdapter(self.estimator))),
                    #                                   BootstrapSampler())

                    self.conformal_pred = AggregatedCp(IcpRegressor(nc),
                                                    BootstrapSampler())
                    self.conformal_pred.fit(X, Y)
                    # overrides non-conformal
                    results.append(
                        ('model', 'model type', 'conformal SVM quantitative'))

                else:
                    self.conformal_pred = AggregatedCp(
                        IcpClassifier(
                            ClassifierNc(
                                ClassifierAdapter(self.estimator),
                                    MarginErrFunc())), 
                        BootstrapSampler())
                    self.conformal_pred.fit(X, Y)
                    # overrides non-conformal
                    results.append(
                        ('model', 'model type', 'conformal SVM qualitative'))
            except Exception as e:
                LOG.error(f'Exception building aggregated conformal SVM ' 
                        f'estimator with exception {e}')
        # Fit estimator to the data
        self.estimator.fit(X, Y)

        return True, results


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