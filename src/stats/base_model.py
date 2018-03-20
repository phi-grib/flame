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

from stats.model_validation import *
from stats.scale import center, scale

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
        self.meanConformalInterval = 0.00

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


    def quantitativeValidation(self):
        
        nobj, nvarx = np.shape(self.X)
        if self.X is None or self.estimator is None:
            return False, 'no estimator'

        X = self.X.copy()
        Y = self.Y.copy()

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
        
        nobj, nvarx = np.shape(self.X)
        if self.X is None or self.estimator is None:
            return False, 'no estimator'

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
        
        if self.quantitative:
            success, results = self.quantitativeValidation()
            if success :
                self.printQuantitativeValidationResults()
        else:
            success, results = self.qualitativeValidation()
            if success :
                self.printQualitativeValidationResults()
        
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


    def regularProject(self, Xb):
        Yp = self.estimator.predict(Xb)
        return Yp


    def conformalProject(self, Xb):
        Yp = self.regularProject(Xb)
        prediction = conformal_pred_pred(Xb, self.conformal_pred,
                     self.conformalSignificance)

        if self.quantitative:
            predictionSize = abs(abs(prediction[0][0]) - abs(prediction[0][1]))
            prediction = prediction.tolist() + [predictionSize, self.meanConformalInterval]
            return ([Yp, prediction])         
        else:
            return ([Yp, prediction])
               

    def project (self, Xb):
        """ Uses the X matrix provided as argument to predict Y"""
        
        results = None
        if self.estimator == None:
            print ('failed to load clasifier')
            return
        if self.autoscale:
            Xb = Xb-self.mux
            Xb = Xb*self.wgx

        if self.conformal:
            results = self.conformalProject(Xb)
        else:
            results = self.regularProject(Xb)
        return results


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


    def getResults (self, results):

        # Goodness of the fit restults
        
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

        results ['TP'] = self.TP 
        results ['TN'] = self.TN 
        results ['FP'] = self.FP 
        results ['FN'] = self.FN 

        #self.sensitivity = 0.00
        #self.specificity = 0.00
        #self.mcc = 0
        
        results ['SDEP'] = self.SDEP 
        results ['Q2'] = self.Q2 

        #self.scoringP = 0.00

        return results





