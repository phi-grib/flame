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
    def __init__(self, X, Y, parameters, conveyor=None):
        """Initializes the estimator.
        Actions
        -------
            - Attribute assignment
        """

        self.param = parameters
        self.conveyor = None
        if conveyor != None:
            self.conveyor = conveyor

        if X is None:
            return

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
        

    # Validation methods section
    def CF_quantitative_validation(self):
        ''' Performs internal  validation for conformal quantitative models '''

        # Make a copy of original matrices.
        X = self.X.copy()
        Y = self.Y.copy()

        info = []
        kf = KFold(n_splits=self.param.getVal('ModelValidationN')
                   , shuffle=True, random_state=46)
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
        inside_interval = ((Y_pred[:, 0].reshape(1, -1)
                                < Y) & 
                                (Y_pred[:, 1].reshape(1, -1) 
                                > Y)).reshape(1, -1)
        # Compute the accuracy (number of instances within the AD).
        accuracy = np.sum(inside_interval/len(Y))

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
        
        # Compute sensitivity, specificity and MCC
        try:
            self.sensitivity = (self.TP / (self.TP + self.FN))
        except Exception as e:
            LOG.error(f'Failed to compute sensibility with'
                        f'exception {e}')
            self.sensitivity = '-'
        try:
            self.specificity = (self.TN / (self.TN + self.FP))
        except Exception as e:
            LOG.error(f'Failed to compute specificity with'
                        f'exception {e}')
            self.specificity = '-'
        try:
            # Compute Matthews Correlation Coefficient
            self.mcc = (((self.TP * self.TN) - (self.FP * self.FN)) /
                        np.sqrt((self.TP + self.FP) * (self.TP + self.FN) *
                         (self.TN + self.FP) * (self.TN + self.FN)))
        except Exception as e:
            LOG.error(f'Failed to compute Mathews Correlation Coefficient'
                        f'exception {e}')
            self.mcc = '-'

        info.append(
            ('Sensitivity', 'Sensitivity in cross-validation', 
                self.sensitivity))
        info.append(
            ('Specificity', 'Specificity in cross-validation', 
                self.specificity))
        info.append(
            ('MCC', 'Matthews Correlation Coefficient in cross-validation',
                 self.mcc))
        try:
            # Compute coverage (% of compounds inside the applicability domain)
            self.conformal_coverage = (self.TN + self.FP + self.TP +
                                        self.FN) / ((self.TN + self.FP +
                                        self.TP + self.FN) +
                                        not_predicted_all)
        except Exception as e:
            LOG.error(f'Failed to compute conformal coverage with'
                        f'exception {e}')
            self.conformal_coverage = '-'
        
        try:
            # Compute accuracy (% of correct predictions)
            self.conformal_accuracy = (float(self.TN + self.TP) /
                                        float(self.FP + self.FN + 
                                            self.TN + self.TP))
        except Exception as e:
            LOG.error(f'Failed to compute conformal accuracy with'
                        f'exception {e}')
            self.conformal_accuracy = '-'
                                                    
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
            if SSY0 == 0.0:
                self.R2 = 0.0
            else:
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
            if SSY0_out == 0.0:
                self.Q2 = 0.0
            else:
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
            info.append(('SensitivityPred', 'Sensitivity in fitting', 
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
                             n_jobs=-1)
        except Exception as e:
            LOG.error(f'Cross-validation failed with exception' 
                        f'exception {e}')
            raise e
        # Get confusion matrix
        try:
            self.TN, self.FP, self.FN, self.TP = confusion_matrix(
                Y, y_pred, labels=[0, 1]).ravel()
        except Exception as e:
            LOG.error(f'Failed to compute confusion matrix with'
                        f'exception {e}')
            raise e
        try:
            self.sensitivity = (self.TP / (self.TP + self.FN))
        except Exception as e:
            LOG.error(f'Failed to compute sensibility with'
                        f'exception {e}')
            self.sensitivity = '-'
        try:
            self.specificity = (self.TN / (self.TN + self.FP))
        except Exception as e:
            LOG.error(f'Failed to compute specificity with'
                        f'exception {e}')
            self.specificity = '-'
        try:
            # Compute Matthews Correlation Coefficient
            self.mcc = (((self.TP * self.TN) - (self.FP * self.FN)) /
                        np.sqrt((self.TP + self.FP) * (self.TP + self.FN) *
                         (self.TN + self.FP) * (self.TN + self.FN)))
        except Exception as e:
            LOG.error(f'Failed to compute Mathews Correlation Coefficient'
                        f'exception {e}')
            self.mcc = '-'


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

        # the default value is represented as 'default' in the YAML parameter file to
        # avoid problems with empty values and must be replaced by None here        
        for key, value in tune_parameters.items():
            if 'default' in value:
                tune_parameters[key] = [None if i == 'default' else i for i in value ]

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

    def regularProject(self, Xb):
        ''' projects a collection of query objects in a regular model,
         for obtaining predictions '''

        Yp = self.estimator.predict(Xb)

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

    def conformalProject(self, Xb):
        ''' projects a collection of query objects in a conformal model,
         for obtaining predictions '''

        if not 'nonconformist' in str(type(self.estimator)):
            self.conveyor.setError('Inconsistence error: non-conformal classifier found. Rebuild the model')
            return

        prediction = self.estimator.predict(
            Xb, significance=self.param.getVal('conformalSignificance'))

        if self.param.getVal('quantitative'):
            Yp = np.mean(prediction, axis=1)
            
            lower_limit = prediction[:, 0]
            upper_limit = prediction[:, 1]

            # if conveyor contains experimental values for any of the objects replace the
            # predictions with the experimental results
            exp = self.conveyor.getVal('experim')
            if exp is not None:
                if len(exp) == len(Yp):
                    for i in range (len(Yp)):
                        if not np.isnan(exp[i]):
                            # print (exp[i], Yp[i])
                            Yp[i] = exp[i]
                            lower_limit[i] = exp[i]
                            upper_limit[i] = exp[i]
                        # if exp is nan, substitute it with a number which can be recognized
                        # to facilitate handling and do not replace Yp
                        else:
                            exp[i]= float ('-99999')

            self.conveyor.addVal(Yp, 'values', 'Prediction',
                    'result', 'objs',
                    'Results of the prediction', 'main')

            self.conveyor.addVal(lower_limit, 'lower_limit',
                             'Lower limit', 'confidence', 'objs',
                              'Lower limit of the conformal prediction')

            self.conveyor.addVal(upper_limit, 'upper_limit',
                             'Upper limit', 'confidence', 'objs',
                              'Upper limit of the conformal prediction')
        else:
            # Returns a dictionary with class
            # predictions
            # / c0 / c1 
            # /True/False
            # This is also converted to a binary 1/0 results
            for i in range(len(prediction[0])):
                class_key = 'c' + str(i)
                class_label = 'Class ' + str(i)
                class_list = prediction[:, i].tolist()
                self.conveyor.addVal(class_list, 
                                class_key, 
                                class_label,
                                'confidence', 'objs', 
                                'Conformal class assignment')

            # the use of np.zeros defaults to 0 (negative)

            nobj, nvary = np.shape(prediction)

            Yp = np.zeros(nobj, dtype = np.float64)
            for j in range (nobj):
                p0 = prediction[j,0]
                if p0==prediction[j,1]:  # if both are equal results are unconclusive
                    Yp[j]=-1
                elif p0 == 0: # if do not belong to class 0 is must be class 1 (positive)
                    Yp[j]=1

            # if conveyor contains experimental values for any of the objects replace the
            # predictions with the experimental results
            # TODO: this section is incomplete and experimental. The replacement of cualitative
            # variables without checking can produce wrong results
            exp = self.conveyor.getVal('experim')
            if exp is not None:
                if len(exp) == len(Yp):
                    for i in range (len(Yp)):
                        if not np.isnan(exp[i]):
                            # print (exp[i], Yp[i])
                            # Yp is copied without any checking, BEWARE!!! 
                            Yp[i] = exp[i]
                        # if exp is nan, substitute it with a number which can be recognized
                        # to facilitate handling and do not replace Yp
                        else:
                            exp[i]= float ('-99999')

                #TODO: moddify the classes, by getting the classes, overwritting the values and
                # saving again

            self.conveyor.addVal(Yp, 'values', 'Prediction',
                    'result', 'objs',
                    'Results of the prediction', 'main')

    def project(self, Xb):
        ''' Uses the X matrix provided as argument to predict Y'''

        if self.estimator == None:
            self.conveyor.setError('failed to load classifier')
            return
        # Apply variable mask to prediction vector/matrix
        # if self.param.getVal("feature_selection"):
        #     Xb = Xb[:, self.variable_mask]
        # Scale prediction vector/matrix
        # if self.param.getVal('modelAutoscaling'):
            # Xb = Xb-self.mux
            # Xb = Xb*self.wgx
            # Xb = self.scaler.transform(Xb)
        # Select the type of projection
        if not self.param.getVal('conformal'):
            self.regularProject(Xb)
        else:
            self.conformalProject(Xb)
    
    def save_model(self):
        ''' This function saves estimator and scaler in a pickle file '''

        # This dictionary contain all the objects which will be needed
        # for prediction
        dict_estimator = {'estimator' : self.estimator,\
                            'version' : 1}

        model_pkl_path = os.path.join(self.param.getVal('model_path'),
                                      'estimator.pkl')
        with open(model_pkl_path, 'wb') as handle:
            pickle.dump(dict_estimator, handle, 
                        protocol=pickle.HIGHEST_PROTOCOL)
        LOG.debug('Model saved as:{}'.format(model_pkl_path))


        # Add estimator parameters to Conveyor
        params = dict()
        if  self.param.getVal('conformal'):
            params = self.estimator_temp.get_params()
        else:
            params = self.estimator.get_params()

        self.conveyor.addVal(params, 'estimator_parameters',
                            'estimator parameters', 'method', 'single',
                            'Hyperparameter values for the algorithm')
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
        self.version = dict_estimator['version']

        # check if the pickle was created with a compatible version
        # currently 1
        if self.version is not 1:
            raise Exception ('Incompatible model version')

        self.estimator = dict_estimator['estimator']
        if self.estimator is None:
            raise Exception('Loaded estimator is None.'
                            'Probably model building was not successful')
    
        return