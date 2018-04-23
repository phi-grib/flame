#! -*- coding: utf-8 -*-

##    Description    Flame Parent Model Class
##
##    Authors:       Jose Carlos Gómez (josecarlos.gomez@upf.edu)
##
##    Copyright 2018 Manuel Pastor
##
##    This file is part of Flame
##
##    Flame is free software: you can redistribute it and/or modify
##    it under the terms of the GNU General Public License as published by
##    the Free Software Foundation version 3.
##
##    Flame is distributed in the hope that it will be useful,
##    but WITHOUT ANY WARRANTY; without even the implied warranty of
##    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
##    GNU General Public License for more details.
##
##    You should have received a copy of the GNU General Public License
##    along with Flame.  If not, see <http://www.gnu.org/licenses/>.


import numpy as np
import os
import copy
import glob
from scipy import stats
import matplotlib.pyplot as plt
import warnings
##warnings.filterwarnings("ignore", category=UserWarning)
#warnings.filterwarnings("ignore", category=DeprecationWarning)

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
from stats.model_validation import *
from stats.scale import center, scale

## nonconformist imports

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

import util.utils as utils

class BaseEstimator:
    def __init__(self, X, Y, quantitative=False, autoscale=False,
                 cv='loo', n=2, p=1, lc=True,
                 conformalSignificance=0.05, vpath='',
                 estimator_parameters={}, conformal=False):

        self.name = ""
        self.X = X
        self.Y = Y
        self.nobj, self.nvarx = np.shape(X)
        self.quantitative = quantitative
        self.autoscale = autoscale
        self.learning_curve = lc
        self.cv = cv
        self.n = n
        self.p = p
        self.scoring_function = None

        self.mux = None
        self.wgx = None

        # Cross-val
        self.TP = 0
        self.TN = 0
        self.FP = 0
        self.FN = 0
        self.sensitivity = 0.00
        self.specificity = 0.00
        self.mcc = 0
        self.SDEP = 0.00
        self.Q2 = 0.00
        self.scoringP = 0.00

        # Goodness of the fit restults
        self.TPpred = 0
        self.TNpred = 0
        self.FPpred = 0
        self.FNpred = 0
        self.sensitivityPred = 0.00
        self.specificityPred = 0.00
        self.SDEC = 0.00    # SD error of the calculations
        self.R2 = 0.00    # determination coefficient
        self.scoringR = 0.00
        self.mccp = 0

        self.estimator = None
        self.estimator_parameters = estimator_parameters
        self.conformal = conformal
        self.conformalSignificance = conformalSignificance
        self.conformal_pred = None
        self.conformal_mean_interval = 0.00
        self.conformal_accuracy = 0.00
        self.conformal_coverage = 0.00


        self.failed = False


    def printQuantitativeValidationResults(self):

        print ("Recalculated results")
        print ('rec R2:%5.3f SDEC:%5.3f  mean_squared_error:%5.3f' %
                (self.R2,self.SDEC, self.scoringR))
        print (str(self.cv)+" cross-validation results")
        print ('pred R2:%5.3f Q2:%5.3f SDEP:%5.3f mean_squared_error:%5.3f' % \
                (self.R2,self.Q2,self.SDEP, self.scoringP))
    

    def printQualitativeValidationResults(self):
        print ("Recalculated results")
        print ("rec  TP:%d TN:%d FP:%d FN:%d spec:%5.3f sens:%5.3f MCC:%5.3f" % \
                (self.TPpred, self.TNpred, self.FPpred, self.FNpred, self.specificityPred,
                 self.sensitivityPred, self.mccp))

        print (str(self.cv)+" cross-validation results")
        print ("pred  TP:%d TN:%d FP:%d FN:%d spec:%5.3f sens:%5.3f MCC:%5.3f" % \
                (self.TP, self.TN, self.FP, self.FN, self.specificity, self.sensitivity, self.mcc))


    def CF_quantitative_validation(self):

        X = self.X.copy()
        Y = self.Y.copy()

        seeds = [5, 7, 35]
        interval_means = []
        accuracies = []
        for i in range(len(seeds)):
            X_train, X_test, Y_train, Y_test =  train_test_split(X, Y, test_size= 0.25,
                                                     random_state=i, shuffle=False)
            conformal_pred = AggregatedCp(IcpRegressor(RegressorNc(RegressorAdapter(self.estimator))),
            BootstrapSampler())
            conformal_pred.fit(X_train, Y_train)
            prediction = conformal_pred.predict(X_test, self.conformalSignificance)

            interval_means.append(np.mean(np.abs(prediction[:, 0]) - np.abs(prediction[:, 1])))
            Y_test = Y_test.reshape(-1, 1)
            inside_interval = (prediction[:, 0].reshape(-1,1) < Y_test ) & (prediction[:, 1].reshape(-1,1) > Y_test )
            accuracy =  np.sum(inside_interval)/len(Y_test)
            accuracies.append(accuracy)
        interval_means = np.mean(interval_means)
        accuracies = np.mean(accuracies)
        self.conformal_accuracy = float("{0:.2f}".format(accuracies))
        self.conformal_mean_interval = float("{0:.2f}".format(interval_means))
        return True, 'ok'

    def CF_qualitative_validation(self):
        
        X = self.X.copy()
        Y = self.Y.copy()


        seeds = [5, 7, 35]
        average_class_errors = []
        c0_correct_all = []
        c0_incorrect_all = []
        c1_correct_all = []
        c1_incorrect_all = []
        not_predicted_all = []

        for i in range(len(seeds)):
            X_train, X_test, Y_train, Y_test =  train_test_split(X, Y, test_size= 0.25,
                                                     random_state=i, shuffle=True)
            conformal_pred = AggregatedCp(IcpClassifier(ClassifierNc(ClassifierAdapter(self.estimator),
                MarginErrFunc())), BootstrapSampler())
            conformal_pred.fit(X_train, Y_train)
            prediction = conformal_pred.predict(X_test, self.conformalSignificance)
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
                    if real == 1 and predicted[1]== True:
                        c1_correct += 1
                    if real == 1 and predicted[0]== True:
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

        self.sensitivity = (self.TP / (self.TP + self.FN))
        self.specificity = (self.TN / (self.TN + self.FP))

        self.mcc = (((self.TP * self.TN) - (self.FP * self.FN)) / 
                    np.sqrt((self.TP + self.FP) * (self.TP + self.FN) * 
                    (self.TN + self.FP) * (self.TN + self.FN)))
        self.mcc = float("{0:.2f}".format(self.mcc))
        self.sensitivity = float("{0:.2f}".format(self.sensitivity))
        self.specificity = float("{0:.2f}".format(self.specificity))
        self.conformal_coverage = (self.TN + self.FP + self.TP + self.FN) / ((self.TN + self.FP + self.TP + self.FN) + not_predicted_all)
        self.conformal_coverage = float("{0:.2f}".format(self.conformal_coverage))
        self.conformal_accuracy = float(self.TN + self.TP)/ float(self.FP + self.FN + self.TN + self.TP)
        self.conformal_accuracy = float("{0:.2f}".format(self.conformal_accuracy))
        return True, 'ok'



    def quantitativeValidation(self):
        
        X = self.X.copy()
        Y = self.Y.copy()

        nobj = len(X)

        if self.autoscale:
            X = X - self.mux
            X = X * self.wgx

        Yp = self.estimator.predict(X)
        Ym = np.mean(Y)

        ## Goodness of the fit

        SSY0 = np.sum (np.square(Ym-Y))
        SSY  = np.sum (np.square(Yp-Y))

        self.scoringR = np.mean(mean_squared_error(Y, Yp)) # Mean Squared Error
        self.SDEC = np.sqrt(SSY/nobj)
        self.R2   = 1.00 - (SSY/SSY0)

        ## Cross-validation

        y_pred = cross_val_predict(copy.copy(self.estimator), copy.copy(X), 
                                             copy.copy(Y), cv=self.cv, n_jobs=-1)
        SSY0_out = np.sum(np.square(Ym - Y))
        SSY_out = np.sum(np.square(Y - y_pred))
        self.scoringP = mean_squared_error(Y, y_pred)
        self.SDEP = np.sqrt(SSY_out/(nobj))
        self.Q2   = 1.00 - (SSY_out/SSY0_out)

        return True, 'ok'

    def qualitativeValidation(self):
        
        X = self.X.copy()
        Y = self.Y.copy()

        if self.autoscale:
            X = X - self.mux
            X = X * self.wgx

        Yp = self.estimator.predict(X)

        if len(Yp) != len(Y):
            return False, 'lenght of prediction do not match'

        ## Goodness of the fit

        print ('Goodness of the fit validation')
        self.TPpred, self.FPpred, self.FNpred, self.TNpred = confusion_matrix(Y,Yp).ravel()
        self.sensitivityPred = (self.TPpred / (self.TPpred + self.FNpred))
        self.specificityPred = (self.TNpred / (self.TNpred + self.FPpred))
        self.mccp  = mcc(Y, Yp)

        ## Cross validation

        print ('Cross validating')
        y_pred = cross_val_predict(self.estimator, X, Y, cv=self.cv, n_jobs=-1)
        self.TP, self.FP, self.FN, self.TN = confusion_matrix(Y,y_pred).ravel()
        self.sensitivity = (self.TP / (self.TP + self.FN))
        self.specificity = (self.TN / (self.TN + self.FP))
        self.mcc  = mcc(Y, y_pred)

        return True, 'ok'


    def validate(self):
        """ Validates the models and completes suitable scoring values"""
        nobj, nvarx = np.shape(self.X)
        if self.X is None or self.estimator is None:
            return False, 'no estimator'
        if not self.conformal:
            if self.quantitative:
                success, results = self.quantitativeValidation()
                if success :
                    self.printQuantitativeValidationResults()
            else:
                success, results = self.qualitativeValidation()
                if success :
                    self.printQualitativeValidationResults()
        else:
            if self.quantitative:
                success, results =  self.CF_quantitative_validation()
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


    def regularProject(self, Xb, results):
        
        Yp =  self.estimator.predict(Xb)

        utils.add_result (results, Yp, 'values', 'Prediction', 'result', 'objs', 'Results of the prediction', 'main')


    def conformalProject(self, Xb, results):

        prediction = self.conformal_pred.predict(Xb, significance=self.conformalSignificance)
        
        if self.quantitative:
            mean1 = np.mean(prediction, axis=1)
            #predictionSize = abs(abs(prediction[0][0]) - abs(prediction[0][1]))
            lower_limit = prediction[:, 0]
            upper_limit = prediction[:, 1]
            utils.add_result (results, mean1, 'values', 'Prediction', 'result', 'objs', 'Results of the prediction', 'main')
            utils.add_result (results, lower_limit, 'lower_limit', 'Lower limit', 'confidence', 'objs', 'Lower limit of the conformal prediction' )
            utils.add_result (results, upper_limit, 'upper_limit', 'Upper limit', 'confidence', 'objs', 'Upper limit of the conformal prediction' )

        else:

            ## For the moment is returning a dictionary with class predictions 
            # / c0 / c1 / c2 /
            # /True/True/False/
            
            for i in range(len(prediction[0])):
                class_key = 'c' + str(i)
                class_label = 'Class ' + str(i) 
                class_list = prediction[:, i].tolist()
                utils.add_result (results, class_list, class_key, class_label, 'result', 'objs', 'Conformal class assignment', 'main')
               

    def project (self, Xb, results):
        """ Uses the X matrix provided as argument to predict Y"""
        
        if self.estimator == None:
            results['error']='failed to load classifier'
            return
        
        if self.autoscale:
            Xb = Xb-self.mux
            Xb = Xb*self.wgx

        if not self.conformal:
            self.regularProject(Xb, results)
        else:
            self.conformalProject(Xb, results)
        

    def conformal_calibration(self,):

        X = copy.copy(self.X)
        Y = copy.copy(self.Y)
        if self.autoscale:
            X, self.mux = center(X)
            X, self.wgx = scale(X, self.autoscale)
        if not self.quantitative:
            self.conformal_pred = CF_QualCal(X, Y, copy.copy(self.estimator))
        else:
            self.conformal_pred = CF_QuanCal(X, Y, copy.copy(self.estimator))


    def optimize (self, X, Y, estimator, tune_parameters ):
        metric = ""
        if self.quantitative:
            metric = 'r2'
        else:
            #metric = make_scorer(mcc)
            #metric = make_scorer(f1_score)
            metric = "f1"
        # if self.name == 'PLSR':  # Remember problems optimizing PLSR
        #     metric = 'neg_mean_squared_error'
        #     Y = np.asarray(pd.get_dummies(Y)).tolist() # Move this to a new PLS-DA ***
        #     Y = np.asarray(Y)


        tune_parameters = [tune_parameters]
        print ("tune_parameters")
        print ("metric: " + str(metric))
        tclf = GridSearchCV(estimator, tune_parameters, scoring=metric, cv=self.cv)
        #n_splits=10, shuffle=False,
            #   random_state=42), n_jobs= -1)
        tclf.fit(X, Y)
        self.estimator = tclf.best_estimator_
        print (tclf.best_params_)
        #print self.estimator.get_params() 


    def getResults (self):

        results = {}
        
        # Goodness of the fit results
        
        results ['TPpred'] = self.TPpred 
        results ['TNpred'] = self.TNpred 
        results ['FPpred'] = self.FPpred 
        results ['FNpred'] = self.FNpred 

        #self.sensitivityPred = 0.00
        #self.specificityPred = 0.00
        
        results ['SDEC' ] = self.SDEC     # SD error of the calculations
        results ['R2'] = self.R2     # determination coefficient
        
        #self.scoringR = 0.00
        #self.mccp = 0

        # Cross-val

        if not self.quantitative:
            results ['TP'] = self.TP 
            results ['TN'] = self.TN 
            results ['FP'] = self.FP 
            results ['FN'] = self.FN 
            results ['mcc'] =self.mcc
            if self.conformal:
                results ['Conformal_accuracy'] = self.conformal_accuracy
                results ['Conformal_significance'] = self.conformalSignificance
        else:
            results ['SDEC' ] = self.SDEC     # SD error of the calculations
            results ['R2'] = self.R2     # determination coefficient
            if self.conformal:
                results ['Conformal_accuracy'] = self.conformal_accuracy
                results ["Conformal_mean_interval"] = self.conformal_mean_interval
                results ['Conformal_coverage'] = self.conformal_coverage
                results ['Conformal_significance'] = self.conformalSignificance


        #self.sensitivity = 0.00
        #self.specificity = 0.00
        #self.mcc = 0
        
        results ['SDEP'] = self.SDEP 
        results ['Q2'] = self.Q2 

        #self.scoringP = 0.00

        return results





