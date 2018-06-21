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

import numpy as np
import os
import copy
import time
import glob
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


class BaseEstimator:
    def __init__(self, X, Y, parameters):

        self.failed = False
        self.parameters = parameters

        self.X = X
        self.Y = Y
        self.nobj, self.nvarx = np.shape(X)

        if self.parameters["imbalance"] is not None and not self.parameters["quantitative"]:
            self.X, self.Y = run_imbalance(
                self.parameters['imbalance'], self.X, self.Y, 46)

        if (self.nobj == 0) or (self.nvarx == 0):
            self.failed = True
            return

        if (np.shape(Y) == 0):
            self.failed = True
            return

        self.quantitative = self.parameters['quantitative']
        self.autoscale = self.parameters['modelAutoscaling']
        self.cv = self.parameters['ModelValidationCV']
        self.n = self.parameters['ModelValidationN']
        self.p = self.parameters['ModelValidationP']
        self.learning_curve = parameters['ModelValidationLC']
        self.conformal = self.parameters['conformal']
        self.conformalSignificance = self.parameters['conformalSignificance']

    # Validation methods section

    def CF_quantitative_validation(self):
        ''' performs validation for conformal quantitative models '''

        X = self.X.copy()
        Y = self.Y.copy()

        seeds = [5, 7, 35]
        interval_means = []
        accuracies = []
        results = []
        for i in range(len(seeds)):
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25,
                                                                random_state=i, shuffle=False)
            conformal_pred = AggregatedCp(IcpRegressor(RegressorNc(RegressorAdapter(self.estimator))),
                                          BootstrapSampler())
            conformal_pred.fit(X_train, Y_train)
            prediction = conformal_pred.predict(
                X_test, self.conformalSignificance)

            interval_means.append(
                np.mean(np.abs(prediction[:, 0]) - np.abs(prediction[:, 1])))
            Y_test = Y_test.reshape(-1, 1)
            inside_interval = (
                prediction[:, 0].reshape(-1, 1) < Y_test) & (prediction[:, 1].reshape(-1, 1) > Y_test)
            accuracy = np.sum(inside_interval)/len(Y_test)
            accuracies.append(accuracy)
        interval_means = np.mean(interval_means)
        accuracies = np.mean(accuracies)
        self.conformal_accuracy = float("{0:.2f}".format(accuracies))
        self.conformal_mean_interval = float("{0:.2f}".format(interval_means))

        results.append(('Conformal_mean_interval',
                        'Conformal mean interval', self.conformal_mean_interval))
        results.append(
            ('Conformal_accuracy', 'Conformal accuracy', self.conformal_accuracy))

        return True, results

    def CF_qualitative_validation(self):
        ''' performs validation for conformal qualitative models '''

        X = self.X.copy()
        Y = self.Y.copy()

        seeds = [5, 7, 35]
        average_class_errors = []
        c0_correct_all = []
        c0_incorrect_all = []
        c1_correct_all = []
        c1_incorrect_all = []
        not_predicted_all = []

        results = []

        for i in range(len(seeds)):
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25,
                                                                random_state=i, shuffle=True)
            conformal_pred = AggregatedCp(IcpClassifier(ClassifierNc(ClassifierAdapter(self.estimator),
                                                                     MarginErrFunc())), BootstrapSampler())
            conformal_pred.fit(X_train, Y_train)
            prediction = conformal_pred.predict(
                X_test, self.conformalSignificance)
            c0_correct = 0
            c1_correct = 0
            not_predicted = 0
            c0_incorrect = 0
            c1_incorrect = 0
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
            c0_correct_all.append(c0_correct)
            c0_incorrect_all.append(c0_incorrect)
            c1_correct_all.append(c1_correct)
            c1_incorrect_all.append(c1_incorrect)
            not_predicted_all.append(not_predicted)

        self.TN = np.int(np.mean(c0_correct_all))
        self.FP = np.int(np.mean(c0_incorrect_all))
        self.TP = np.int(np.mean(c1_correct_all))
        self.FN = np.int(np.mean(c1_incorrect_all))
        not_predicted_all = np.int(np.mean(not_predicted_all))

        results.append(('TP', 'True positives in cross-validation', self.TP))
        results.append(('TN', 'True negatives in cross-validation', self.TN))
        results.append(('FP', 'False positives in cross-validation', self.FP))
        results.append(('FN', 'False negatives in cross-validation', self.FN))

        self.sensitivity = (self.TP / (self.TP + self.FN))
        self.specificity = (self.TN / (self.TN + self.FP))

        self.mcc = (((self.TP * self.TN) - (self.FP * self.FN)) /
                    np.sqrt((self.TP + self.FP) * (self.TP + self.FN) *
                            (self.TN + self.FP) * (self.TN + self.FN)))

        results.append(
            ('Sensitivity', 'Sensitivity in cross-validation', self.sensitivity))
        results.append(
            ('Specificity', 'Specificity in cross-validation', self.specificity))
        results.append(
            ('MCC', 'Mattews Correlation Coefficient in cross-validation', self.mcc))

        self.conformal_coverage = (self.TN + self.FP + self.TP + self.FN) / (
            (self.TN + self.FP + self.TP + self.FN) + not_predicted_all)

        self.conformal_accuracy = float(
            self.TN + self.TP) / float(self.FP + self.FN + self.TN + self.TP)

        results.append(
            ('Conformal_coverage', 'Conformal coverage', self.conformal_coverage))
        results.append(
            ('Conformal_accuracy', 'Conformal accuracy', self.conformal_accuracy))

        return True, results

    def quantitativeValidation(self):
        ''' performs validation for quantitative models '''

        X = self.X.copy()
        Y = self.Y.copy()

        nobj = self.nobj

        if self.autoscale:
            X = X - self.mux
            X = X * self.wgx

        Yp = self.estimator.predict(X)
        Ym = np.mean(Y)
        results = []

        # Goodness of the fitt

        SSY0 = np.sum(np.square(Ym-Y))
        SSY = np.sum(np.square(Yp-Y))

        self.scoringR = np.mean(
            mean_squared_error(Y, Yp))  # Mean Squared Error
        self.SDEC = np.sqrt(SSY/nobj)
        self.R2 = 1.00 - (SSY/SSY0)

        results.append(('scoringR', 'Scoring P', self.scoringR))
        results.append(('R2', 'Determination coefficient', self.R2))
        results.append(
            ('SDEC', 'Standard Deviation Error of the Calculations', self.SDEC))

        # Cross-validation

        y_pred = cross_val_predict(copy.copy(self.estimator), copy.copy(X),
                                   copy.copy(Y), cv=self.cv, n_jobs=1)
        SSY0_out = np.sum(np.square(Ym - Y))
        SSY_out = np.sum(np.square(Y - y_pred))
        self.scoringP = mean_squared_error(Y, y_pred)
        self.SDEP = np.sqrt(SSY_out/(nobj))
        self.Q2 = 1.00 - (SSY_out/SSY0_out)

        results.append(('scoringP', 'Scoring P', self.scoringP))
        results.append(
            ('Q2', 'Determination coefficient in cross-validation', self.Q2))
        results.append(
            ('SDEP', 'Standard Deviation Error of the Predictions', self.SDEP))

        return True, results

    def qualitativeValidation(self):
        ''' performs validation for qualitative models '''

        X = self.X.copy()
        Y = self.Y.copy()

        if self.autoscale:
            X = X - self.mux
            X = X * self.wgx

        Yp = self.estimator.predict(X)

        if len(Yp) != len(Y):
            return False, 'lenght of prediction do not match'

        results = []

        # Goodness of the fit

        self.TNpred, self.FPpred, self.FNpred, self.TPpred = confusion_matrix(
            Y, Yp, labels=[0, 1]).ravel()
        self.sensitivityPred = (self.TPpred / (self.TPpred + self.FNpred))
        self.specificityPred = (self.TNpred / (self.TNpred + self.FPpred))
        self.mccp = mcc(Y, Yp)

        results.append(('TPpred', 'True positives', self.TPpred))
        results.append(('TNpred', 'True negatives', self.TNpred))
        results.append(('FPpred', 'False positives', self.FPpred))
        results.append(('FNpred', 'False negatives', self.FNpred))
        results.append(
            ('SensitivityPed', 'Sensitivity in fitting', self.sensitivityPred))
        results.append(
            ('SpecificityPred', 'Specificity in fitting', self.specificityPred))
        results.append(
            ('MCCpred', 'Mattews Correlation Coefficient', self.mccp))

        # Cross validation

        y_pred = cross_val_predict(self.estimator, X, Y, cv=self.cv, n_jobs=1)
        self.TN, self.FP, self.FN, self.TP = confusion_matrix(
            Y, y_pred, labels=[0, 1]).ravel()
        self.sensitivity = (self.TP / (self.TP + self.FN))
        self.specificity = (self.TN / (self.TN + self.FP))
        self.mcc = mcc(Y, y_pred)

        results.append(('TP', 'True positives in cross-validation', self.TP))
        results.append(('TN', 'True negatives in cross-validation', self.TN))
        results.append(('FP', 'False positives in cross-validation', self.FP))
        results.append(('FN', 'False negatives in cross-validation', self.FN))

        results.append(
            ('Sensitivity', 'Sensitivity in cross-validation', self.sensitivity))
        results.append(
            ('Specificity', 'Specificity in cross-validation', self.specificity))
        results.append(
            ('MCC', 'Mattews Correlation Coefficient in cross-validation', self.mcc))

        return True, results

    def validate(self):
        ''' Validates the model and computes suitable model quality scoring values'''

        if self.X is None or self.estimator is None:
            return False, 'no estimator'

        if not self.conformal:
            if self.quantitative:
                success, results = self.quantitativeValidation()
            else:
                success, results = self.qualitativeValidation()
        else:
            if self.quantitative:
                success, results = self.CF_quantitative_validation()
            else:
                success, results = self.CF_qualitative_validation()

        return success, results

        # Move this to an external module ****

        # if self.learning_curve:
        #     print ('Building Learning Curves')
        #     title = "Learning Curves (RF)"
        #     cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
        #     estimator = self.clf
        #     plot = plot_learning_curve(estimator, title, self.X, self.Y, (0.0, 1.01), cv=cv)
        #     plot.savefig(self.vpath+"/" + self.name + "-learning_curves.png", format='png')
        #     plot.savefig("./" + self.name + "-learning_curves.png", format='png')

        # return (Yp)

    def optimize(self, X, Y, estimator, tune_parameters):
        ''' optimizes a model using a grid search over a range of values for diverse parameters'''

        metric = ""
        if self.quantitative:
            metric = 'r2'
        else:
            # metric = make_scorer(mcc)
            # metric = make_scorer(f1_score)
            metric = "f1"
        # if self.name == 'PLSR':  # Remember problems optimizing PLSR
        #     metric = 'neg_mean_squared_error'
        #     Y = np.asarray(pd.get_dummies(Y)).tolist() # Move this to a new PLS-DA ***
        #     Y = np.asarray(Y)

        tune_parameters = [tune_parameters]

        start = time.time()
        print("tune_parameters")
        print("metric: " + str(metric))
        tclf = GridSearchCV(estimator, tune_parameters,
                            scoring=metric, cv=self.cv, n_jobs=-1)
        # n_splits=10, shuffle=False,
        #   random_state=42), n_jobs= -1)
        tclf.fit(X, Y)
        self.estimator = tclf.best_estimator_
        print("best parameters: ", tclf.best_params_)
        end = time.time()
        print("found in: ", end-start, " seconds")
        # print self.estimator.get_params()

    # Projection section

    def regularProject(self, Xb, results):
        ''' projects a collection of query objects in a regular model, for obtaining predictions '''

        Yp = self.estimator.predict(Xb)

        utils.add_result(results, Yp, 'values', 'Prediction',
                         'result', 'objs', 'Results of the prediction', 'main')

    def conformalProject(self, Xb, results):
        ''' projects a collection of query objects in a conformal model, for obtaining predictions '''

        prediction = self.conformal_pred.predict(
            Xb, significance=self.conformalSignificance)

        if self.quantitative:
            mean1 = np.mean(prediction, axis=1)
            # predictionSize = abs(abs(prediction[0][0]) - abs(prediction[0][1]))
            lower_limit = prediction[:, 0]
            upper_limit = prediction[:, 1]
            utils.add_result(results, mean1, 'values', 'Prediction',
                             'result', 'objs', 'Results of the prediction', 'main')
            utils.add_result(results, lower_limit, 'lower_limit', 'Lower limit',
                             'confidence', 'objs', 'Lower limit of the conformal prediction')
            utils.add_result(results, upper_limit, 'upper_limit', 'Upper limit',
                             'confidence', 'objs', 'Upper limit of the conformal prediction')
        else:
            # For the moment is returning a dictionary with class predictions
            # / c0 / c1 / c2 /
            # /True/True/False/

            for i in range(len(prediction[0])):
                class_key = 'c' + str(i)
                class_label = 'Class ' + str(i)
                class_list = prediction[:, i].tolist()
                utils.add_result(results, class_list, class_key, class_label,
                                 'result', 'objs', 'Conformal class assignment', 'main')

    def project(self, Xb, results):
        ''' Uses the X matrix provided as argument to predict Y'''

        if self.estimator == None:
            results['error'] = 'failed to load classifier'
            return

        if self.autoscale:
            Xb = Xb-self.mux
            Xb = Xb*self.wgx

        if not self.conformal:
            self.regularProject(Xb, results)
        else:
            self.conformalProject(Xb, results)
