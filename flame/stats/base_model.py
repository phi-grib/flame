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

from flame.util import utils
from flame.stats.imbalance import *  
from flame.stats.model_validation import *
from flame.stats.scale import center, scale
from flame.stats.feature_selection import *
import pickle
import numpy as np
import os
import copy
import time
import glob
import gc
from scipy import stats
import matplotlib.pyplot as plt
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, matthews_corrcoef as mcc
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler 
# nonconformist imports
from nonconformist.base import ClassifierAdapter, RegressorAdapter
from nonconformist.icp import IcpClassifier, IcpRegressor
from nonconformist.nc import MarginErrFunc
from nonconformist.nc import ClassifierNc, RegressorNc
from nonconformist.nc import AbsErrorErrFunc, SignErrorErrFunc, RegressorNormalizer
from nonconformist.acp import AggregatedCp
from nonconformist.acp import BootstrapSampler, CrossSampler, RandomSubSampler
from nonconformist.acp import BootstrapConformalClassifier
from nonconformist.acp import CrossConformalClassifier

from nonconformist.evaluation import class_mean_errors, class_one_c
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from nonconformist.base import ClassifierAdapter
from nonconformist.icp import IcpClassifier
from nonconformist.nc import ClassifierNc, MarginErrFunc
from nonconformist.evaluation import cross_val_score as conformal_cross_val_score
from nonconformist.evaluation import ClassIcpCvHelper, RegIcpCvHelper
from nonconformist.evaluation import class_avg_c, class_mean_errors
from nonconformist.evaluation import reg_mean_errors, reg_median_size
from nonconformist.evaluation import reg_mean_size
from nonconformist.evaluation import class_mean_errors

from flame.util import utils, get_logger, supress_log
LOG = get_logger(__name__)


class BaseEstimator:
    """
    Estimator parent class, contains all attributes methods shared
     by different algorithms.Particular implementation of these 
     methods are overwritten by child classes      
        
        Attributes
        ----------

        parameters : dict
            parameter values
        X_original : numpy.darray
            original X matrix
        Y_original : numpy.darray
            original X matrix
        variable_mask: numpy.darray
            variable mask from feature selection
        X : numpy.darray
            current X matrix
        Y : numpy.darray
            current Y vector/matrix
        scaler: sklearn scaler
            scaler object to scale prediction 
            instances
        nobj : int
            number of objects
        nvarx : int
            number of variables
        
        
        Methods
        -------

        build(X)
            Instance the estimator optimizing it
            if tune=true.
        run_feature_selection()
            Performs feature selection
        CF_quantitative_validation(self)
            Performs conformal quantitative validation
        CF_qualitative_validation(self)
            Performs conformal qualitative validation
        quantitativeValidation(self)
            Performs quantitative validation
        qualitativeValidation
            Performs qualitative validation
        validate(self)
            Checks type of validation and calls it
        optimize(self, X, Y, estimator, tune_parameters)
            Performs GridSearchCV to optimize estimator
            hyperparameters
        regularProject(self, Xb, results)
            Returns prediction/s for unknown instance/s
        conformalProject(self, Xb, results)
            Returns conformal prediction/s for unknown instance/s
        project(self, Xb, results)
            Checks type of projection and calls it

    """
    def __init__(self, X, Y, parameters):
        """Initializes the estimator.
        Actions
        -------
            - Attribute assignment
            - Scaling
            - Sampling
            - Feature selection
        """

        self.param = parameters
        self.scaler = None
        self.variable_mask = None

        if X is not None:
            self.X_original = X
            self.Y_original = Y
            self.variable_mask = []
            self.X = X
            self.Y = Y
            self.nobj, self.nvarx = np.shape(X)

            # Get cross-validator
            # Consider to include a Random Seed for cross-validator
            if self.param.getVal('ModelValidationCV'):
                try:
                    self.cv = getCrossVal(
                                    self.param.getVal('ModelValidationCV'),
                                    46,
                                    self.param.getVal('ModelValidationN'),
                                    self.param.getVal('ModelValidationP'))
                    LOG.debug('Cross-validator retrieved')
                    LOG.info(f'cv is: {self.cv}')
                except Exception as e:
                    LOG.error(f'Error retrieving cross-validator with'
                            f'exception: {e}')
                    raise e

            # Perform subsampling on the majority class. Consider to move.
            # Only for qualitative endpoints.
            if self.param.getVal("imbalance") is not None and \
            not self.param.getVal("quantitative"):
                try:
                    self.X, self.Y = run_imbalance(
                        self.param.getVal('imbalance'), self.X, self.Y, 46)
                    # This is necessary to avoid inconsistences in methods
                    # using self.nobj
                    LOG.info(f'{self.param.getVal("imbalance")}'
                                f'sampling method performed')
                    LOG.info(f'Number of objects after sampling: {self.nobj}')
                except Exception as e:
                    LOG.error(f'Unable to perform sampling '
                                f'method with exception: {e}')
                    raise e

            # Run scaling.
            if self.param.getVal('modelAutoscaling'):
                try:
                    # self.X, self.mux = center(self.X)
                    # self.X, self.wgx = scale(self.X, 
                    #                     self.param.getVal('modelAutoscaling'))
                    # MinMaxScaler is used between range 1-0 so 
                    # there is no negative values.
                    scaler = MinMaxScaler(copy=True, feature_range=(0,1))
                    # The scaler is saved so it can be used later
                    # to prediction instances.
                    self.scaler = scaler.fit(self.X)
                    # Scale the data.
                    self.X = scaler.transform(self.X)
                    LOG.info('Data scaling performed')
                except Exception as e:
                    LOG.error(f'Unable to perform scaling'
                    ' with exception : {e}')
                    raise e

            #### Alternative way to make all values positives (sum the minimum 
            #### of each column to the column)
                # list_min = np.min(self.X, axis=0)
                # newX = copy.copy(self.X)
                # for i in range(len(self.X[0])):
                #     newX[:, i] = np.array(self.X[:, i] -list_min[i])

            # Run feature selection. Move to a instance method.
            if self.param.getVal("feature_selection"):
                self.run_feature_selection()
        
            # Set the new number of instances/variables
            # if sampling/feature selection performed
            self.nobj, self.nvarx = np.shape(X)

            # Check X and Y integrity.
            if (self.nobj == 0) or (self.nvarx == 0):
                LOG.error('No objects/variables in the matrix')
                raise Exception('No objects/variables in the matrix')
            if len(Y) == 0:
                self.failed = True
                LOG.error('No activity values')
                raise ValueError("No activity values (Y)")

    def run_feature_selection(self):
        """Compute the number of variables to be retained.
        """
        # When auto, the 10% top informative variables are retained.
        if self.param.getVal("feature_number") == "auto":
            # Use 10% of the total number of objects:
            # The number of variables is greater than the 10% of the objects
            # And the number of objects is greater than 100
            if self.nvarx > (self.nobj * 0.1) and not self.nobj < 100:
                self.n_features = int(self.nobj * 0.1)
            # If number of objects is smaller than 100 then n_features
            # is set to 10
            elif self.nobj < 100:
                self.n_features = 10
            # In any other circunstance set number of variables to 10 
            else:
                self.n_features = self.nvarx
        # Manual selection of number of variables
        else:
            self.n_features = int(self.param.getVal("feature_number"))

        # Apply variable selection.
        try:
            # Apply the variable selection algorithm obtaining
            # the variable mask.
            self.variable_mask = selectkBest(self.X, self.Y, 
                                self.n_features, 
                                self.param.getVal('quantitative'))
            
            # The scaler has to be fitted to the reduced matrix
            # in order to be applied in prediction.
            self.X = self.scaler.inverse_transform(self.X)
            self.X = self.X[:, self.variable_mask]
            self.scaler = self.scaler.fit(self.X)
            self.X = self.scaler.transform(self.X)
            # self.mux = self.mux.reshape(1, -1)[:, self.variable_mask]
            # self.wgx = self.wgx.reshape(1, -1)[:, self.variable_mask]
            LOG.info(f'Variable selection applied, number of final variables:'
                        f'{self.n_features}')
        except Exception as e:
            LOG.error(f'Error performing feature selection'
                        f' with exception: {e}')
            raise e 

    # Validation methods section
    def CF_quantitative_validation(self):
        ''' Performs internal  validation for conformal quantitative models '''

        # Make a copy of original matrices.
        X = self.X.copy()
        Y = self.Y.copy()

        info = []
        kf = KFold(n_splits=5, shuffle=True)
        # Copy Y vector to use it as template to assign predictions
        Y_pred = copy.copy(Y).tolist()
        try:
            for train_index, test_index in kf.split(X):
                # Generate training and test sets
                X_train, X_test = X[train_index], X[test_index]
                Y_train, Y_test = Y[train_index], Y[test_index]
                # Generate training a test sets
                # Create the aggregated conformal regressor.
                conformal_pred = AggregatedCp(IcpRegressor(
                                    RegressorNc(RegressorAdapter(
                                        self.estimator_temp))),
                                            BootstrapSampler())
                # Fit conformal regressor to the data
                conformal_pred.fit(X_train, Y_train)

                # Perform prediction on test set
                prediction = conformal_pred.predict(
                    X_test, self.param.getVal('conformalSignificance'))
                # Assign the prediction its original index
                for index, el in enumerate(test_index):
                    Y_pred[el] = prediction[index]

        except Exception as e:
            LOG.error(f'Quantitative conformal validation'
                        f' failed with exception: {e}')
            raise e

        Y_pred = np.asarray(Y_pred)
        # Add the n validation interval means
        interval_mean = np.mean(np.abs((Y_pred[:, 0]) - 
                        (Y_pred[:, 1])))
        # Get boolean mask of instances
        #  within the applicability domain.
        inside_interval = ((Y_pred[:, 0].reshape(-1, 1)
                                < Y) & 
                                (Y_pred[:, 1].reshape(-1, 1) 
                                > Y))
        # Compute the accuracy (number of instances within the AD).
        accuracy = np.sum(inside_interval)/len(Y)

        # Cut into two decimals.
        self.conformal_interval_medians = (np.mean(Y_pred, axis=1))
        self.conformal_accuracy = float("{0:.2f}".format(accuracy))
        self.conformal_mean_interval = float("{0:.2f}".format(interval_mean))

        #Add quality metrics to results.
        info.append(('Conformal_mean_interval',
                        'Conformal mean interval', 
                        self.conformal_mean_interval))
        info.append(
            ('Conformal_accuracy', 'Conformal accuracy', 
            self.conformal_accuracy))
        info.append(
            ('Conformal_interval_medians',
             'Conformal interval medians', 
            self.conformal_interval_medians))
        info.append(
            ('Conformal_prediction_ranges',
             'Conformal prediction ranges', 
             Y_pred))

        results = {}
        results ['quality'] = info
        return True, results

    def CF_qualitative_validation(self):
        ''' performs validation for conformal qualitative models '''

        # Make a copy of original matrices.
        X = self.X.copy()
        Y = self.Y.copy()

        # Total number of class 0 correct predictions.
        c0_correct_all = 0
        # Total number of class 0 incorrect predictions.
        c0_incorrect_all = 0
        # Total number of class 1 correct predictions.
        c1_correct_all = 0
        # Total number of class 1 incorrect predictions
        c1_incorrect_all = 0
        # Total number of instances out of the applicability domain.
        not_predicted_all = 0

        info = []

        kf = KFold(n_splits=5, shuffle=True, random_state=46)
        # Copy Y vector to use it as template to assign predictions
        Y_pred = copy.copy(Y).tolist()
        try:
            for train_index, test_index in kf.split(X):
                # Generate training and test sets
                X_train, X_test = X[train_index], X[test_index]
                Y_train, Y_test = Y[train_index], Y[test_index]
                # Create the aggregated conformal classifier.
                conformal_pred = AggregatedCp(IcpClassifier(
                                            ClassifierNc(ClassifierAdapter(
                                                            self.estimator_temp),
                                                MarginErrFunc())),
                                            BootstrapSampler())
                # Fit the conformal classifier to the data
                conformal_pred.fit(X_train, Y_train)
                # Perform prediction on test set
                prediction = conformal_pred.predict(
                            X_test, self.param.getVal('conformalSignificance'))
                # Assign the prediction the correct index. 
                for index, el in enumerate(test_index):
                    Y_pred[el] = prediction[index]
            # Iterate over the prediction and check the result
            for i in range(len(Y_pred)):
                real = float(Y[i])
                predicted = Y_pred[i]
                if predicted[0] != predicted[1]:
                    if real == 0 and predicted[0] == True:
                        c0_correct_all += 1
                    if real == 0 and predicted[1] == True:
                        c0_incorrect_all += 1
                    if real == 1 and predicted[1] == True:
                        c1_correct_all += 1
                    if real == 1 and predicted[0] == True:
                        c1_incorrect_all += 1
                else:
                    not_predicted_all += 1

        except Exception as e:
            LOG.error(f'Qualitative conformal validation'
                        f' failed with exception: {e}')
            raise e
        # Get the mean confusion matrix.
        self.TN = c0_correct_all
        self.FP = c0_incorrect_all
        self.TP = c1_correct_all
        self.FN = c1_incorrect_all
        not_predicted_all = not_predicted_all

        info.append(('TP', 'True positives in cross-validation', self.TP))
        info.append(('TN', 'True negatives in cross-validation', self.TN))
        info.append(('FP', 'False positives in cross-validation', self.FP))
        info.append(('FN', 'False negatives in cross-validation', self.FN))
        
        # Compute sensitivity and specificity
        try:
            self.sensitivity = (self.TP / (self.TP + self.FN))
            self.specificity = (self.TN / (self.TN + self.FP))
        except Exception as e:
            LOG.error(f'Failed to compute sens/spe with'
                        f'exception {e}')
        # Compute Matthews Correlation Coefficient
        self.mcc = (((self.TP * self.TN) - (self.FP * self.FN)) /
                    np.sqrt((self.TP + self.FP) * (self.TP + self.FN) *
                            (self.TN + self.FP) * (self.TN + self.FN)))
        info.append(
            ('Sensitivity', 'Sensitivity in cross-validation', 
                self.sensitivity))
        info.append(
            ('Specificity', 'Specificity in cross-validation', 
                self.specificity))
        info.append(
            ('MCC', 'Matthews Correlation Coefficient in cross-validation',
                 self.mcc))

        # Compute coverage (% of compounds inside the applicability domain)
        self.conformal_coverage = (self.TN + self.FP + self.TP +
                                    self.FN) / ((self.TN + self.FP +
                                     self.TP + self.FN) +
                                      not_predicted_all)
        # Compute accuracy (% of correct predictions)
        self.conformal_accuracy = float(
            self.TN + self.TP) / float(self.FP + self.FN + self.TN + self.TP)

        info.append(
            ('Conformal_coverage', 'Conformal coverage',
                 self.conformal_coverage))
        info.append(
            ('Conformal_accuracy', 'Conformal accuracy', 
                self.conformal_accuracy))

        results = {}
        results ['quality'] = info
        #results ['classes'] = prediction
        return True, results

    def quantitativeValidation(self):
        ''' performs validation for quantitative models '''

        # Make a copy of the original matrices
        X = self.X.copy()
        Y = self.Y.copy()

        # Get predicted Y
        Yp = self.estimator.predict(X)
        # Compute  mean of predicted Y
        Ym = np.mean(Y)
        info = []

        # Compute Goodness of the fit metric (adjusted Y)
        try:
            SSY0 = np.sum(np.square(Ym-Y))
            SSY = np.sum(np.square(Yp-Y))

            self.scoringR = np.mean(
                mean_squared_error(Y, Yp)) 
            self.SDEC = np.sqrt(SSY/self.nobj)
            self.R2 = 1.00 - (SSY/SSY0)

            info.append(('scoringR', 'Scoring P', self.scoringR))
            info.append(('R2', 'Determination coefficient', self.R2))
            info.append(
                ('SDEC', 'Standard Deviation Error of the Calculations', 
                    self.SDEC))
            LOG.debug(f'Goodness of the fit calculated: {self.scoringR}')
        except Exception as e:
            LOG.error(f'Error computing goodness of the fit'
                f'with exception {e}')
            raise e

        # Compute Cross-validation quality metrics
        try:
            # Get predicted Y
            y_pred = cross_val_predict(copy.copy(self.estimator),
                            copy.copy(X), copy.copy(Y),
                                cv=self.cv,
                                    n_jobs=1)
            SSY0_out = np.sum(np.square(Ym - Y))
            SSY_out = np.sum(np.square(Y - y_pred))
            self.scoringP = mean_squared_error(Y, y_pred)
            self.SDEP = np.sqrt(SSY_out/(self.nobj))
            self.Q2 = 1.00 - (SSY_out/SSY0_out)

            info.append(('scoringP', 'Scoring P', self.scoringP))
            info.append(
                ('Q2', 'Determination coefficient in cross-validation',
                     self.Q2))
            info.append(
                ('SDEP', 'Standard Deviation Error of the Predictions',
                     self.SDEP))

            # newy.append (
            #     ('Y_adj', 'Recalculated Y values', Yp) )          
            # newy.append (
            #     ('Y_pred', 'Predicted Y values (after cross-validation)', y_pred) )  
            LOG.debug(f'Squared-Q calculated: {self.scoringP}')
        except Exception as e:
            LOG.error(f'Error cross-validating the estimator'
                        f' with exception {e}')
            raise e
              
        results = {}
        results ['quality'] = info
        results ['Y_adj'] = Yp
        results ['Y_pred'] = y_pred
        return True, results

    def qualitativeValidation(self):
        ''' performs validation for qualitative models '''

        # Make a copy of the original matrices
        X = self.X.copy()
        Y = self.Y.copy()

        # Get predicted classes.
        Yp = self.estimator.predict(X)

        if len(Yp) != len(Y):
            raise Exception('Lenght of experimental and predicted Y'
                            'do not match')

        info = []

        # Get confusion matrix for predicted Y
        try:
            self.TNpred, self.FPpred,\
            self.FNpred, self.TPpred = confusion_matrix(Y, Yp,
                                                     labels=[0, 1]).ravel()
            self.sensitivityPred = (self.TPpred / (self.TPpred + self.FNpred))
            self.specificityPred = (self.TNpred / (self.TNpred + self.FPpred))
            self.mccp = mcc(Y, Yp)

            info.append(('TPpred', 'True positives', self.TPpred))
            info.append(('TNpred', 'True negatives', self.TNpred))
            info.append(('FPpred', 'False positives', self.FPpred))
            info.append(('FNpred', 'False negatives', self.FNpred))
            info.append(('SensitivityPed', 'Sensitivity in fitting', 
                    self.sensitivityPred))
            info.append(
                ('SpecificityPred', 'Specificity in fitting', 
                    self.specificityPred))
            info.append(('MCCpred', 'Matthews Correlation Coefficient', 
                    self.mccp))
            LOG.debug('Computed class prediction for estimator instances')
        except Exception as e:
            LOG.error(f'Error computing class prediction of Yexp'
                f'with exception {e}')
            raise e

        # Get cross-validated Y 
        try:
            y_pred = cross_val_predict(self.estimator, X, Y,
                    cv=self.cv,
                             n_jobs=1)

            # Get confusion matrix
            self.TN, self.FP, self.FN, self.TP = confusion_matrix(
                Y, y_pred, labels=[0, 1]).ravel()
            self.sensitivity = (self.TP / (self.TP + self.FN))
            self.specificity = (self.TN / (self.TN + self.FP))
            self.mcc = mcc(Y, y_pred)

            info.append(('TP', 'True positives in cross-validation',
             self.TP))
            info.append(('TN', 'True negatives in cross-validation',
             self.TN))
            info.append(('FP', 'False positives in cross-validation',
             self.FP))
            info.append(('FN', 'False negatives in cross-validation',
             self.FN))

            info.append(
                ('Sensitivity', 'Sensitivity in cross-validation',
                 self.sensitivity))
            info.append(
                ('Specificity', 'Specificity in cross-validation',
                 self.specificity))
            info.append(
                ('MCC', 'Matthews Correlation Coefficient in cross-validation',
                 self.mcc))
            info.append (
                ('Y_adj', 'Adjusted Y values', Y) ) 
            info.append (
                ('Y_adj', 'Adjusted Y values', Yp) )          
            info.append (
                ('Y_pred', 'Predicted Y values after cross-validation',
                 y_pred))
            LOG.debug(f'Qualitative crossvalidation performed')
        except Exception as e:
            LOG.error(f'Error computing crossvalidated Y'
                f' with exception {e}')
            raise e

        results = {}
        results ['quality'] = info
        results ['Y_adj'] = Yp
        results ['Y_pred'] = y_pred
        return True, results

    def validate(self):
        ''' Validates the model and computes suitable
         model quality scoring values'''
        # Check estimator integrity
        if self.X is None or self.estimator is None:
            return False, 'no estimator'

        if not self.param.getVal('conformal'):
            if self.param.getVal('quantitative'):
                success, results = self.quantitativeValidation()
            else:
                success, results = self.qualitativeValidation()
        else:
            if self.param.getVal('quantitative'):
                success, results = self.CF_quantitative_validation()
            else:
                success, results = self.CF_qualitative_validation()

        return success, results

    def optimize(self, X, Y, estimator, tune_parameters):
        ''' optimizes a model using a grid search over a 
        range of values for diverse parameters'''
        
        LOG.info('Computing best hyperparameter values')
        metric = ""
        # Select the metric according to the type of model
        if self.param.getVal('quantitative'):
            metric = 'r2'
        else:
            metric = make_scorer(mcc)

        tune_parameters = [tune_parameters]
        # Count computation time
        LOG.debug("Hyperparameter optimization ")
        start = time.time()
        # Consider crossval number to be a parameter, not
        # constant.
        try:
            tclf = GridSearchCV(estimator, tune_parameters,
                                scoring=metric, cv=3, n_jobs=4)
            tclf.fit(X, Y)
            self.estimator = copy.copy(tclf.best_estimator_)
        except Exception as e:
            LOG.error(f'Error optimizing hyperparameters with'
            f'exception {e}')
            raise e
        end = time.time()
        LOG.info(f'best parameters: , {tclf.best_params_}')
        LOG.debug(f'Best estimator found in {end-start} seconds')
        # Remove garbage in memory
        del(tclf)
        gc.collect()

    # Projection section

    def regularProject(self, Xb, results):
        ''' projects a collection of query objects in a regular model,
         for obtaining predictions '''

        Yp = self.estimator.predict(Xb)

        utils.add_result(results, Yp, 'values', 'Prediction',
                         'result', 'objs',
                          'Results of the prediction', 'main')

    def conformalProject(self, Xb, results):
        ''' projects a collection of query objects in a conformal model,
         for obtaining predictions '''

        prediction = self.estimator.predict(
            Xb, significance=self.param.getVal('conformalSignificance'))

        if self.param.getVal('quantitative'):
            mean1 = np.mean(prediction, axis=1)
            lower_limit = prediction[:, 0]
            upper_limit = prediction[:, 1]
            utils.add_result(results, mean1, 'values', 'Prediction',
                             'result', 'objs',
                              'Results of the prediction', 'main')
            utils.add_result(results, lower_limit, 'lower_limit',
                             'Lower limit', 'confidence', 'objs',
                              'Lower limit of the conformal prediction')
            utils.add_result(results, upper_limit, 'upper_limit',
                             'Upper limit', 'confidence', 'objs',
                              'Upper limit of the conformal prediction')
        else:
            # For the moment is returning a dictionary with class
            #  predictions
            # / c0 / c1 / c2 /
            # /True/True/False/
            for i in range(len(prediction[0])):
                class_key = 'c' + str(i)
                class_label = 'Class ' + str(i)
                class_list = prediction[:, i].tolist()
                utils.add_result(results, class_list, 
                                class_key, class_label,
                                'result', 'objs', 
                                'Conformal class assignment',
                                 'main')

    def project(self, Xb, results):
        ''' Uses the X matrix provided as argument to predict Y'''

        if self.estimator == None:
            results['error'] = 'failed to load classifier'
            return
        # Apply variable mask to prediction vector/matrix
        if self.param.getVal("feature_selection"):
            Xb = Xb[:, self.variable_mask]
        # Scale prediction vector/matrix
        if self.param.getVal('modelAutoscaling'):
            # Xb = Xb-self.mux
            # Xb = Xb*self.wgx
            Xb = self.scaler.transform(Xb)
        # Select the type of projection
        if not self.param.getVal('conformal'):
            self.regularProject(Xb, results)
        else:
            self.conformalProject(Xb, results)
    
    def save_model(self):
        ''' This function saves estimator and scaler in a pickle file '''

        # This dictionary contain all the objects which will be needed
        # for prediction
        dict_estimator = {'estimator' : self.estimator,\
                            'scaler' : self.scaler,\
                            'variable_mask' : self.variable_mask,\
                            'version' : 1}

        model_pkl_path = os.path.join(self.param.getVal('model_path'),
                                      'estimator.pkl')
        
        with open(model_pkl_path, 'wb') as handle:
            pickle.dump(dict_estimator, handle, 
                        protocol=pickle.HIGHEST_PROTOCOL)
        LOG.debug('Model saved as:{}'.format(model_pkl_path))

        return


    def load_model(self):
        ''' This function loads estimator and scaler in a pickle file '''

        model_file = os.path.join(self.param.getVal('model_path'),'estimator.pkl')
        LOG.debug(f'Loading model from pickle file, path: {model_file}')
        try:
            with open(model_file, "rb") as input_file:
                dict_estimator = pickle.load(input_file)
        except FileNotFoundError:
            LOG.error(f'No valid model estimator found at: {model_file}')
            raise FileNotFoundError

        # Load model
        self.estimator = dict_estimator['estimator']
        self.version = dict_estimator['version']
        if self.estimator is None:
            LOG.error('Estimator is None.'
            'Probably model building was not successful')
            raise Exception('Loaded estimator is None.'
                            'Probably model building was not successful')
        
        # Load rest of info in an extensible way
        # This allows to add new variables keeping
        # Retro-compatibility
        if 'scaler' in dict_estimator.keys():
            self.scaler = dict_estimator['scaler']
        if 'variable_mask' in dict_estimator.keys():
            self.variable_mask = dict_estimator['variable_mask']

        # Check consistency between parameter file and pickle info
        if self.param.getVal('modelAutoscaling') and \
            self.scaler is None:
            raise Exception('Inconsistency error. Autoscaling is True'
            ' in parameter file but no Scaler loaded')
        if self.param.getVal('feature_selection') and \
            self.variable_mask is None:
            raise Exception('Inconsistency error. Feature is True'
            ' in parameter file but no variable mask loaded')
        return