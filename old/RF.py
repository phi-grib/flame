# -*- coding: utf-8 -*-

##    Description    RF model classifier and regressor
##
##    Authors:       Manuel Pastor (manuel.pastor@upf.edu)
##
##    Copyright 2017 Manuel Pastor
##
##    This file is part of eTOXlab.
##
##    eTOXlab is free software: you can redistribute it and/or modify
##    it under the terms of the GNU General Public License as published by
##    the Free Software Foundation version 3.
##
##    eTOXlab is distributed in the hope that it will be useful,
##    but WITHOUT ANY WARRANTY; without even the implied warranty of
##    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
##    GNU General Public License for more details.
##
##    You should have received a copy of the GNU General Public License
##    along with eTOXlab.  If not, see <http://www.gnu.org/licenses/>.


import numpy as np
import sys
import os
import glob

from scipy import stats
from scipy.stats import t
import matplotlib.pyplot as plt
from collections import OrderedDict

import warnings
##warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib
from sklearn.model_selection import LeaveOneOut #JC
from sklearn.model_selection import cross_val_score #JC
from sklearn.model_selection import ShuffleSplit  # JC
from sklearn.grid_search import GridSearchCV  # JC
from sklearn.metrics import mean_squared_error
from sklearn.metrics import f1_score

from model_validation import *  #JC

from scale import center, scale
from qualit import *
from utils import updateProgress


class RF:

    def __init__ (self):

        self.X = None
        self.Y = None

        self.nobj = 0
        self.nvarx = 0

        self.quantitative = False
        self.autoscale = False
        self.estimators = 0
        self.features = ''
        self.random = False
        self.class_weight = False
        self.learning_curve = True
        self.cv = None
        self.n = 2
        self.p = 1
        self.scoring_function = None

        self.mux = None
        self.wgx = None

        self.TP = 0
        self.TN = 0
        self.FP = 0
        self.FN = 0

        self.TPpred = 0
        self.TNpred = 0
        self.FPpred = 0
        self.FNpred = 0

        self.SDEC = 0.00    # SD error of the calculations
        self.R2   = 0.00    # determination coefficient
        self.scoringR = 0.00
        self.SDEP = 0.00
        self.Q2 = 0.00
        self.scoringP = 0.00
        self.OOBe = 0.00

        self.clf = None

        self.vpath= None


    def saveModel(self,filename):
        """Saves the model to a binary file in numpy file and another in pkl format

        """

        f = file(filename,'wb')

        np.save(f,self.nobj)
        np.save(f,self.nvarx)

        np.save(f,self.quantitative)
        np.save(f,self.autoscale)
        np.save(f,self.estimators)
        np.save(f,self.features)
        np.save(f,self.random)
        np.save(f,self.class_weight)
        np.save(f,self.learning_curve)
        np.save(f,self.cv)
        np.save(f,self.n)
        np.save(f,self.p)



        np.save(f,self.mux)
        np.save(f,self.wgx)

        np.save(f,self.TP)
        np.save(f,self.TN)
        np.save(f,self.FP)
        np.save(f,self.FN)

        np.save(f,self.TPpred)
        np.save(f,self.TNpred)
        np.save(f,self.FPpred)
        np.save(f,self.FNpred)

        np.save(f,self.SDEC)
        np.save(f,self.R2)
        np.save(f, self.scoringR)
        np.save(f, self.Q2)
        np.save(f, self.SDEP)
        np.save(f, self.scoringP)

        np.save(f,self.OOBe)

        np.save(f,self.vpath)

        f.close()

        # the classifier cannot be saved with numpy
        joblib.dump(self.clf, os.path.dirname(filename)+'/clasifier.pkl')


    def loadModel(self,filename):
        """Loads the model from two files, one in numpy and another in pkl format
        """

        f = file(filename,'rb')

        self.nobj = np.load(f)
        self.nvarx = np.load(f)

        self.quantitative = np.load(f)
        self.autoscale = np.load(f)
        self.estimators = np.load(f)
        self.features = np.load(f)
        self.random = np.load(f)
        self.class_weight = np.load(f)
        self.learning_curve = np.load(f)
        self.cv = np.load(f)
        self.n = np.load(f)
        self.p = np.load(f)

        self.mux = np.load(f)
        self.wgx = np.load(f)

        self.TP = np.load(f)
        self.TN = np.load(f)
        self.FP = np.load(f)
        self.FN = np.load(f)

        self.TPpred = np.load(f)
        self.TNpred = np.load(f)
        self.FPpred = np.load(f)
        self.FNpred = np.load(f)

        self.SDEC = np.load(f)
        self.R2 = np.load(f)
        self.scoringR = np.load(f)
        self.Q2 = np.load(f)
        self.SDEP = np.load(f)
        self.scoringP = np.load(f)

        self.OOBe = np.load(f)

        self.vpath = np.load(f)

        f.close()

        # the classifier cannot be loaded with numpy
        self.clf = joblib.load(os.path.dirname(filename)+'/clasifier.pkl')


    def build (self, X, Y, quantitative=False, autoscale=False,
               nestimators=0, features='', random=False, tune=False, class_weight="balanced",
               cv='loo', n=2, p=1, lc=True, vpath = ''):
        """Build a new RF model with the X and Y numpy matrices

        """

        nobj, nvarx= np.shape(X)

        self.nobj  = nobj
        self.nvarx = nvarx

        self.quantitative = quantitative
        self.autoscale = autoscale
        self.estimators = nestimators
        self.features = features
        self.random = random
        self.class_weight = class_weight
        self.learning_curve = lc
        self.n = n
        self.p = p
        self.cv = cv

        self.X = X.copy()
        self.Y = Y.copy()

        self.vpath = vpath

        #print self.vpath
        if autoscale:
            self.X, self.mux = center(self.X)
            self.X, self.wgx = scale(self.X, autoscale)

        if random :
            RANDOM_STATE = None
        else:
            RANDOM_STATE = 1226 # no reason to pick this number

        if self.cv:
            self.cv = getCrossVal(self.cv, RANDOM_STATE, self.n, self.p)
            
        if tune :
            self.estimators, self.features = self.optimize (self.X, self.Y)

            if self.features=='none':
                self.features = None

        #print self.estimators

        if self.quantitative:
            print "Building Quantitative RF model"
            self.clf = RandomForestRegressor(n_estimators = int(self.estimators),
                                            warm_start=False,
                                            max_features=self.features,
                                            oob_score=True,
                                            random_state=RANDOM_STATE)
        else:
            print "Building Qualitative RF_model"
            self.clf = RandomForestClassifier(n_estimators = int(self.estimators),
                                            warm_start=False,
                                            max_features=self.features,
                                            oob_score=True,
                                            random_state=RANDOM_STATE,
                                            class_weight=self.class_weight)

        self.clf.fit(self.X, self.Y)
        
        print 'Building Learning Curves'
        if self.learning_curve:
            title = "Learning Curves (RF)"
            # SVC is more expensive so we do a lower number of CV iterations:
            cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
            estimator = self.clf
            plot = plot_learning_curve(estimator, title, self.X, self.Y, (0.0, 1.01), cv=cv)
            plot.savefig(self.vpath+"/RF-learning_curves.png", format='png')
            plot.savefig("./RF-learning_curves.png", format='png')


        # Regenerate the X and Y, since they might have been centered/scaled
        self.X = X.copy()
        self.Y = Y.copy()


    def validate (self):
        """ Validates the models and completes suitable scoring values

        """

##        valRF = open("valRF.txt", "w")
##        valRF.write("Experimental\tRecalculated\tPredicted\n")
        if self.X == None or self.clf == None:
            return

        X = self.X.copy()
        Y = self.Y.copy()
        if self.autoscale:
            X = X-self.mux
            X = X*self.wgx

        Yp = self.clf.predict(X)
        Ym = np.mean(Y)

        ######################################################################3
        ### quantitative
        
        if self.quantitative:
            # OOB_errors = []
            # Recalculated predictions
            SSY0 = np.sum (np.square(Ym-Y))
            SSY  = np.sum (np.square(Yp-Y))

            NMSErec = np.mean(mean_squared_error(Y, Yp)) # Mean Squared Error
            self.scoringR = NMSErec
            self.SDEC = np.sqrt(SSY/self.nobj)
            self.R2   = 1.00 - (SSY/SSY0)
            self.OOBe = 1.00 - self.clf.oob_score_

            print "Recalculated results"
            print 'rec R2:%5.3f SDEC:%5.3f OOB_error:%5.3f neg_mean_squared_error:%5.3f' % \
                  (self.R2,self.SDEC,self.OOBe, self.scoringR)

            
            scoring = 'neg_mean_squared_error'

            y_pred = cross_val_predict(self.clf, X, Y, cv=self.cv)
            NMSEcv = np.mean(cross_val_score(self.clf, X, Y, cv=self.cv, scoring=scoring)) # Mean Squared Error

 
            SSY0_out = np.sum(np.square(Ym - Y))
            SSY_out = np.sum(np.square(Y - y_pred))
            self.scoringP = NMSEcv
            self.SDEP = np.sqrt(SSY_out/(self.nobj))
            self.Q2   = 1.00 - (SSY_out/SSY0_out)
            # OOBe_loo  = 1.00 - np.mean(OOB_errors)

            print str(self.cv)+" cross-validation results"
            print 'pred R2:%5.3f Q2:%5.3f SDEP:%5.3f neg_mean_squared_error:%5.3f' % \
                  (self.R2,self.Q2,self.SDEP, self.scoringP)


            # Automated cross-validation loo scikitlearn

            clf = RandomForestRegressor(n_estimators = int(self.estimators),
                                    warm_start=False,
                                    max_features=self.features,
                                    oob_score=True,
                                    random_state=1226)

            # GRAPHS

            pngfiles = glob.glob (self.vpath+'/*.png')
            for i in pngfiles:
##                print i
                os.remove(i)
            try:
                fig1=plt.figure()
                plt.xlabel('experimental y')
                plt.ylabel('recalculated\n',fontsize=14)
                plt.title('R2: %4.2f  /  SDEC: %4.2f \n' % (self.R2,self.SDEC), fontsize=14)
                plt.plot(Y,Yp,"ro")
                fig1.savefig(self.vpath+"/RF-recalculated.png", format='png')
                fig1.savefig("./RF-recalculated.png", format='png')
            except:
                print "Error creating Recalculated vs Experimental RF model graph"

            try:
                fig1=plt.figure()
                plt.xlabel('experimental y')
                plt.ylabel('predicted\n',fontsize=14)
                plt.title('Q2: %4.2f  /  SDEP: %4.2f \n' % (self.Q2,self.SDEP), fontsize=14)
                plt.plot(Y, y_pred,"ro")
                fig1.savefig(self.vpath+"/RF-predicted.png", format='png')
                fig1.savefig("./RF-predicted.png", format='png')
            except:
                print "Error creating Predicted vs Experimental RF model graph"


           # File with experimental, recalculated and cv predictions values.
##            for i in range(len(Y)):
##               valRF.write(str(Y[i]) + "\t" + str(Yp[i]) + "\t" + str(y_pred[i]) + "\n")

        ######################################################################3
        ### qualitative
        else:

            # I think this is not needed.... by the characteristics of RF it allways shows perfect performance
            if len(Yp) != len(Y):
                return

            TP=TN=FP=FN=0

            for i in range(len(Y)):

                if Y[i] == 1.0:
                    if Yp[i] == 1.0:
                        TP+=1
                    else:
                        FN+=1
                else:
                    if Yp[i] == 1.0:
                        FP+=1
                    else:
                        TN+=1

            if TP+TN+FP+FN == 0:
                #print 'no objects'
                return

            self.TP = TP
            self.TN = TN
            self.FP = FP
            self.FN = FN

            sens = sensitivity (TP, FN)
            spec = specificity (TN, FP)
            mcc  = MCC (TP, TN, FP, FN)
            f1   = f1_score(Y, Yp, pos_label=1, average='binary')

            self.OOBe = 1.00 - self.clf.oob_score_

            print "Recalculated results"
            print "rec  TP:%d TN:%d FP:%d FN:%d spec:%5.3f sens:%5.3f MCC:%5.3f OOB_error:%5.3f f1_score:%5.3f" % \
                  (TP, TN, FP, FN, spec, sens, mcc, self.OOBe, f1 )

            # Leave-one-out Cross validation
            print 'Cross validating RF....'
            scoring = 'f1'

            y_pred = cross_val_predict(self.clf, X, Y, cv=self.cv)
            
            #Y_score = np.mean(cross_val_score(self.clf, X, Y, cv=self.cv, scoring=scoring))

            TPo=TNo=FPo=FNo = 0
            
            for i in range(len(Y)):

                if Y[i] == 1.0:
                    if y_pred[i] == 1.0:
                        TPo+=1
                    else:
                        FNo+=1
                else:
                    if y_pred[i] == 1.0:
                        FPo+=1
                    else:
                        TNo+=1

            if TPo+TNo+FPo+FNo == 0:
                return

            self.TPpred = TPo
            self.TNpred = TNo
            self.FPpred = FPo
            self.FNpred = FNo

            sens_cv = sensitivity (TPo, FNo)
            spec_cv = specificity (TNo, FPo)
            mcc_cv  = MCC (TPo, TNo, FPo, FNo)
            f1_cv = f1_score(Y, y_pred, pos_label=1, average='binary')

            print str(self.cv)+" cross-validation results"
            print "pred  TP:%d TN:%d FP:%d FN:%d spec:%5.3f sens:%5.3f MCC:%5.3f f1_score:%5.3f" % \
                  (TPo, TNo, FPo, FNo, spec_cv, sens_cv, mcc_cv, f1_cv )

            # Create Graphs

            pngfiles = glob.glob (self.vpath+'/*.png')
            for i in pngfiles:
                os.remove(i)
                
            # Predicted confusion matrix graph
            try:
                FourfoldDisplay(TPo,TNo,FPo,FNo, 'RFC Predicted', 'RF_predicted_confusion_matrix.png' , self.vpath)
            except:
                print "Failed to generate RF predicted validation graph"

            # Recalculated confusion matrix graph
            try:
                FourfoldDisplay(TP,TN,FP,FN, 'RFC Recalculated', 'RF_recalculated_confusion_matrix.png' , self.vpath)
            except:
                print "Failed to generate RF recalculated validation graph"

        return (Yp)


    def project (self, Xb):
        """ Uses the X matrix provided as argument to predict Y
        """

        if self.clf == None:
            print 'failed to load clasifier'
            return

        if self.autoscale:
            Xb = Xb-self.mux
            Xb = Xb*self.wgx

        Xb = Xb.reshape(1,-1) # required by sklean, to avoid deprecation warning
        Yp = self.clf.predict(Xb)

        return (Yp)


    def optimize (self, X, Y ):
        """ Optimizes the number of trees (estimators) and max features used (features)
            and returns the best values, acording to the OOB criteria

            The results are shown in a diagnostic plot

            To avoid including many trees to produce tiny improvements, increments of OOB error
            below 0.01 are considered irrelevant
        """

        RANDOM_STATE = 1226
        errors = {}
        features = ['sqrt','log2','none']

        if self.quantitative:
            tclf = {'sqrt': RandomForestRegressor(warm_start=False, oob_score=True,
                        max_features="sqrt",random_state=RANDOM_STATE),
                    'log2': RandomForestRegressor(warm_start=False, oob_score=True,
                        max_features="log2",random_state=RANDOM_STATE),
                    'none': RandomForestRegressor(warm_start=False, oob_score=True,
                        max_features=None  ,random_state=RANDOM_STATE) }
        else:
            tclf = {'sqrt': RandomForestClassifier(warm_start=False, oob_score=True,
                        max_features="sqrt",random_state=RANDOM_STATE,
                        class_weight=self.class_weight),
                    'log2': RandomForestClassifier(warm_start=False, oob_score=True,
                        max_features="log2",random_state=RANDOM_STATE,
                        class_weight=self.class_weight),
                    'none': RandomForestClassifier(warm_start=False, oob_score=True,
                        max_features=None  ,random_state=RANDOM_STATE,
                        class_weight=self.class_weight) }

        # Range of `n_estimators` values to explore.
        min_estimators = 15
        max_estimators = 700
        stp_estimators = 100

        num_steps = int((max_estimators-min_estimators)/stp_estimators)

        print 'optimizing RF....'
        updateProgress (0.0)

        optValue = 1.0e10
        j = 0
        for fi in features:
            errors[fi] = []
            count = 0
            for i in range(min_estimators, max_estimators + 1,stp_estimators):
                clf = tclf[fi]
                clf.set_params(n_estimators=i)
                clf.fit(X,Y)
                oob_error = 1 - clf.oob_score_
                errors[fi].append((i,oob_error))
                if oob_error < optValue:
                    if np.abs(oob_error - optValue) > 0.01:
                        optValue = oob_error
                        optEstimators = i
                        optFeatures = fi

                updateProgress (float(count+(j*num_steps))/float(len(features)*num_steps))
                count = count+1
            j=j+1

        for ie in errors:
            xs, ys = zip (*errors[ie])
            plt.plot(xs, ys, label=ie)

        plt.xlim(min_estimators, max_estimators)
        plt.xlabel("n_estimators (Trees)")
        plt.ylabel("OOB error rate")
        plt.legend(loc="upper right")
        plt.show()

        plt.savefig(self.vpath+"/rf-OOB-parameter-tuning.png")
        plt.savefig("./rf-OOB-parameter-tuning.png")

        print 'optimum features:', optFeatures, 'optimum estimators:', optEstimators, 'best OOB:', optValue

        return (optEstimators, optFeatures)

