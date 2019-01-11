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

# TODO
# Raise errors to child class and handling from there

class BaseEstimator:
    """Estimator parent class, contains all attributes methods shared
     by different algorithms.Particular implementation of these 
     methods are overwritten by child classes"""

    def __init__(self, X, Y, parameters):
        """Initializes the estimator.
        Actions
        -------
            - Attribute assignment
            - Scaling
            - Sampling
            - Feature selection
        """

        self.failed = False
        self.parameters = parameters
        self.X_original = X
        self.Y_original = Y
        self.variable_mask = []
        self.X = X
        self.Y = Y
        self.nobj, self.nvarx = np.shape(X)

        # Assign model attributes.
        # self.quantitative = self.parameters['quantitative']
        # self.autoscale = self.parameters['modelAutoscaling']
        # self.cv = self.parameters['ModelValidationCV']
        # self.n = self.parameters['ModelValidationN']
        # self.p = self.parameters['ModelValidationP']
        # self.conformal = self.parameters['conformal']
        #self.feature_selection = self.parameters["feature_selection"]
        #self.conformalSignificance = self.parameters['conformalSignificance']

        # Check X and Y integrity.
        if (self.nobj == 0) or (self.nvarx == 0):
            LOG.error('No objects/variables in the matrix')
            raise Exception('No objects/variables in the matrix')
        if len(Y) == 0:
            self.failed = True
            LOG.error('No activity values')
            raise ValueError("No activity values (Y)")


        # Get cross-validator 
        # Consider to include a Random Seed for cross-validator
        if self.parameters['ModelValidationCV']:
            try:
                self.cv = getCrossVal(self.parameters['ModelValidationCV'],
                                46,
                                self.parameters['ModelValidationN'],
                                self.parameters['ModelValidationP'])
                LOG.debug('Cross-validator retrieved')
                LOG.info(f'cv is: {self.cv}')
            except Exception as e:
                LOG.error(f'Error retrieving cross-validator with'
                           f'exception: {e}')
                raise e

        # Perform subsampling on the majoritary class. Consider to move.
        # Only for qualititive endpoints.
        if self.parameters["imbalance"] is not None and \
         not self.parameters["quantitative"]:
            try:
                self.X, self.Y = run_imbalance(
                    self.parameters['imbalance'], self.X, self.Y, 46)
                # This is necessary to avoid inconsistences in methods
                # using self.nobj
                self.nobj, self.nvarx = np.shape(self.X)
                LOG.info(f'{self.parameters["imbalance"]}'
                             f'sampling method performed')
                LOG.info(f'Number of objects after sampling: {self.nobj}')
            except Exception as e:
                LOG.error(f'Unable to perform sampling'
                            f'method with exception: {e}')
                raise e

        # Run scaling.
        if self.parameters['modelAutoscaling']:
            try:
                self.X, self.mux = center(self.X)
                self.X, self.wgx = scale(self.X, 
                                    self.parameters['modelAutoscaling'])
                # MinMaxScaler is used between range 1-0 so 
                # there is no negative values.
                scaler =  MinMaxScaler(copy=True, feature_range=(0,1))
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
        if self.parameters["feature_selection"]:
            self.run_feature_selection()


    def run_feature_selection(self):
        """Compute the number of variables to be retained.
        """
        # When auto, the 10% top informative variables are retained.
        if self.parameters["feature_number"] == "auto":
            # Use 10% of the total number of objects:
            # -The number of variables is greater than the 10% of the objects
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
            self.n_features = int(self.parameters["feature_number"])

        # Apply variable selection.
        try:
            # Apply the variable selection algorithm obtaining
            # the variable mask.
            self.variable_mask = selectkBest(self.X, self.Y, 
                                self.n_features, 
                                self.parameters['quantitative'])
            
            # The scaler has to be fitted to the reduced matrix
            # in order to be applied in prediction.
            self.X = self.scaler.inverse_transform(self.X)
            self.X = self.X[:, self.variable_mask]
            self.scaler = self.scaler.fit(self.X)
            self.X = self.scaler.transform(self.X)
            self.mux = self.mux.reshape(1, -1)[:, self.variable_mask]
            self.wgx = self.wgx.reshape(1, -1)[:, self.variable_mask]
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

        # Number of external validations for the aggregated conformal estimator.
        seeds = [5, 7, 35]
        # Interval means for each aggregated  conformal estimator (out of 3)
        interval_means = []
        # Accuracies for each aggregated conformal estimator (out of 3)
        accuracies = []
        results = []
        try:
            for i in range(len(seeds)):
                # Generate training a test sets
                X_train, X_test, Y_train, Y_test = train_test_split(
                        X, Y, test_size=0.25,
                        random_state=i, shuffle=False)
                # Create the aggregated conformal regressor.
                conformal_pred = AggregatedCp(IcpRegressor(
                                    RegressorNc(RegressorAdapter(
                                        self.estimator))),
                                            BootstrapSampler())
                # Fit conformal regressor to the data
                conformal_pred.fit(X_train, Y_train)

                # Perform prediction on test set
                prediction = conformal_pred.predict(
                    X_test, self.parameters['conformalSignificance'])
                # Add the n validation interval means
                interval_means.append(
                    np.mean(np.abs(prediction[:, 0]) - np.abs(prediction[:, 1])))
                Y_test = Y_test.reshape(-1, 1)
                # Get boolean mask of instances within the applicability domain.
                inside_interval = ((prediction[:, 0].reshape(-1, 1) < Y_test) & 
                                     (prediction[:, 1].reshape(-1, 1) > Y_test))
                # Compute the accuracy (number of instances within the AD).
                accuracy = np.sum(inside_interval)/len(Y_test)
                # Add validation result to the list of accuracies.
                accuracies.append(accuracy)
        except Exception as e:
            LOG.error(f'Quantitative conformal validation'
                        f' failed with exception: {e}')
            raise e

        # Compute mean interval_means and accuracy.
        interval_means = np.mean(interval_means)
        accuracies = np.mean(accuracies)
        # Cut into two decimals.
        self.conformal_accuracy = float("{0:.2f}".format(accuracies))
        self.conformal_mean_interval = float("{0:.2f}".format(interval_means))
        #Add quality metrics to results.

        results.append(('Conformal_mean_interval',
                        'Conformal mean interval', self.conformal_mean_interval))
        results.append(
            ('Conformal_accuracy', 'Conformal accuracy', self.conformal_accuracy))

        return True, (results,)

    def CF_qualitative_validation(self):
        ''' performs validation for conformal qualitative models '''

        # Make a copy of original matrices.
        X = self.X.copy()
        Y = self.Y.copy()

        # Number of external validations for the 
        # aggregated conformal estimator.
        seeds = [5, 7, 35]
        # Total number of class 0 correct predictions.
        c0_correct_all = []
        # Total number of class 0 incorrect predictions.
        c0_incorrect_all = []
        # Total number of class 1 correct predictions.
        c1_correct_all = []
        # Total number of class 1 incorrect predictions
        c1_incorrect_all = []
        # Total number of instances out of the applicability domain.
        not_predicted_all = []

        results = []
        # Iterate over the seeds.
        try:
            for i in range(len(seeds)):
                # Generate training and test sets
                X_train, X_test,\
                Y_train, Y_test = train_test_split(X, Y,
                                                    test_size=0.25,
                                                    random_state=i, 
                                                    shuffle=True)
                # Create the aggregated conformal classifier.
                conformal_pred = AggregatedCp(IcpClassifier(
                                            ClassifierNc(ClassifierAdapter(
                                                            self.estimator),
                                                MarginErrFunc())),
                                            BootstrapSampler())
                # Fit the conformal classifier to the data
                conformal_pred.fit(X_train, Y_train)
                # Perform prediction on test set
                prediction = conformal_pred.predict(
                            X_test, self.parameters['conformalSignificance'])
                
                c0_correct = 0
                c1_correct = 0
                not_predicted = 0
                c0_incorrect = 0
                c1_incorrect = 0

                # Iterate over the prediction and check the result
                for i in range(len(Y_test)):
                    real = float(Y_test[i])
                    predicted = prediction[i]
                    if predicted[0] != predicted[1]:
                        if real == 0 and predicted[0] == True:
                            c0_correct += 1
                        if real == 0 and predicted[1] == True:
                            c0_incorrect += 1
                        if real == 1 and predicted[1] == True:
                            c1_correct += 1
                        if real == 1 and predicted[0] == True:
                            c1_incorrect += 1
                    else:
                        not_predicted += 1
                # Add the results to the lists.
                c0_correct_all.append(c0_correct)
                c0_incorrect_all.append(c0_incorrect)
                c1_correct_all.append(c1_correct)
                c1_incorrect_all.append(c1_incorrect)
                not_predicted_all.append(not_predicted)
        except Exception as e:
            LOG.error(f'Qualitative conformal validation'
                        f' failed with exception: {e}')
            raise e
        # Get the mean confusion matrix.
        self.TN = np.int(np.mean(c0_correct_all))
        self.FP = np.int(np.mean(c0_incorrect_all))
        self.TP = np.int(np.mean(c1_correct_all))
        self.FN = np.int(np.mean(c1_incorrect_all))
        not_predicted_all = np.int(np.mean(not_predicted_all))

        results.append(('TP', 'True positives in cross-validation', self.TP))
        results.append(('TN', 'True negatives in cross-validation', self.TN))
        results.append(('FP', 'False positives in cross-validation', self.FP))
        results.append(('FN', 'False negatives in cross-validation', self.FN))
        
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
        results.append(
            ('Sensitivity', 'Sensitivity in cross-validation', 
                self.sensitivity))
        results.append(
            ('Specificity', 'Specificity in cross-validation', 
                self.specificity))
        results.append(
            ('MCC', 'Matthews Correlation Coefficient in cross-validation',
                 self.mcc))

        # Compute coverage (% of compouds inside the applicability domain)
        self.conformal_coverage = (self.TN + self.FP + self.TP +
                                    self.FN) / ((self.TN + self.FP +
                                     self.TP + self.FN) +
                                      not_predicted_all)
        # Compute accuracy (% of correct predictions)
        self.conformal_accuracy = float(
            self.TN + self.TP) / float(self.FP + self.FN + self.TN + self.TP)

        results.append(
            ('Conformal_coverage', 'Conformal coverage',
                 self.conformal_coverage))
        results.append(
            ('Conformal_accuracy', 'Conformal accuracy', 
                self.conformal_accuracy))

        return True, (results,)

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
              
        return True, (info, Yp, y_pred)

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
                    cv=self.parameters['ModelValidationCV'],
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
                f'with exception {e}')
            raise e

        return True, (info, Yp, y_pred)

    def validate(self):
        ''' Validates the model and computes suitable
         model quality scoring values'''
        # Check estimator integrity
        if self.X is None or self.estimator is None:
            return False, 'no estimator'

        if not self.parameters['conformal']:
            if self.parameters['quantitative']:
                success, results = self.quantitativeValidation()
            else:
                success, results = self.qualitativeValidation()
        else:
            if self.parameters['quantitative']:
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
        if self.parameters['quantitative']:
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
        LOG.debug(f'best parameters: , {tclf.best_params_}')
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

        prediction = self.conformal_pred.predict(
            Xb, significance=self.parameters['conformalSignificance'])

        if self.parameters['quantitative']:
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
        if self.parameters["feature_selection"]:
            print ("feature selection")
            Xb = Xb[:, self.variable_mask]
        # Scale prediction vector/matrix
        if self.parameters['modelAutoscaling']:
            Xb = Xb-self.mux
            Xb = Xb*self.wgx
            Xb = self.scaler.transform(Xb)
        # Select the type of projection
        if not self.parameters['conformal']:
            self.regularProject(Xb, results)
        else:
            self.conformalProject(Xb, results)
