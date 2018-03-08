

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
from sklearn.model_selection import LeaveOneOut  # JC
from sklearn.model_selection import cross_val_score  # JC
from sklearn.model_selection import ShuffleSplit  # JC
from sklearn.model_selection import KFold  # JC
from sklearn.model_selection import GridSearchCV  # JC
from sklearn.metrics import mean_squared_error, matthews_corrcoef as mcc
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer

#from sklearn.model_selection import *  # JC
from model_validation import *

from scale import center, scale


class BaseEstimator(object):
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
        self.mcc = 0
        self.SDEP = 0.00
        self.Q2 = 0.00
        self.scoringP = 0.00
        self.OOBe = 0.00
        
        # Goodness of the fit restults
        
        self.TPpred = 0
        self.TNpred = 0
        self.FPpred = 0
        self.FNpred = 0
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

    def validate(self):
        """ Validates the models and completes suitable scoring values


        """

        if ("PLSR" in self.name): pass ## Override for PLS_DA
         
        nobj, nvarx = np.shape(self.X)
        if self.X is None or self.estimator is None:
            print ("no estimator")
            return

        X = self.X.copy()
        Y = self.Y.copy()

        if self.autoscale:
            X = X-self.mux
            X = X*self.wgx

        Yp = self.estimator.predict(X)
        Ym = np.mean(Y)

        # Quantitative
        if self.quantitative:
            SSY0 = np.sum (np.square(Ym-Y))
            SSY  = np.sum (np.square(Yp-Y))
            NMSErec = np.mean(mean_squared_error(Y, Yp)) # Mean Squared Error
            self.scoringR = NMSErec
            self.SDEC = np.sqrt(SSY/nobj)
            self.R2   = 1.00 - (SSY/SSY0)

            if "RF" in self.name:
                self.OOBe = 1.00 - self.estimator.oob_score_
                print ("Recalculated results")
                print ('rec R2:%5.3f SDEC:%5.3f OOB_error:%5.3f neg_mean_squared_error:%5.3f' %
                      (self.R2,self.SDEC,self.OOBe, self.scoringR))

            else:
                print ("Recalculated results")
                print ('rec R2:%5.3f SDEC:%5.3f  neg_mean_squared_error:%5.3f' %
                      (self.R2,self.SDEC, self.scoringR))



            scoring = 'neg_mean_squared_error'

            y_pred = cross_val_predict(copy.copy(self.estimator), copy.copy(X), copy.copy(Y), cv=self.cv, n_jobs=-1)

            NMSEcv = np.mean(cross_val_score(self.estimator, X, Y, cv=self.cv, scoring=scoring, n_jobs=-1)) # Mean Squared Error
            SSY0_out = np.sum(np.square(Ym - Y))
            SSY_out = np.sum(np.square(Y - y_pred))
            self.scoringP = NMSEcv
            self.SDEP = np.sqrt(SSY_out/(nobj))
            self.Q2   = 1.00 - (SSY_out/SSY0_out)
            # OOBe_loo  = 1.00 - np.mean(OOB_errors)

            print (str(self.cv)+" cross-validation results")
            print ('pred R2:%5.3f Q2:%5.3f SDEP:%5.3f neg_mean_squared_error:%5.3f' % \
                  (self.R2,self.Q2,self.SDEP, self.scoringP))

            # GRAPHS

            # pngfiles = glob.glob (self.vpath+'/*.png')
            # for i in pngfiles:
            #     os.remove(i)
            # try:
            #     fig1=plt.figure()
            #     plt.xlabel('experimental y')
            #     plt.ylabel('recalculated\n',fontsize=14)
            #     plt.title('R2: %4.2f  /  SDEC: %4.2f \n' % (self.R2,self.SDEC), fontsize=14)
            #     plt.plot(Y,Yp,"ro")
            #     fig1.savefig(self.vpath+"/" + self.name + "-recalculated.png", format='png')
            #     fig1.savefig("./" + self.name + "-recalculated.png", format='png')
            # except:
            #     print "Error creating Recalculated vs Experimental" + self.name + " model graph"

            # try:
            #     fig1=plt.figure()
            #     plt.xlabel('experimental y')

            #     plt.ylabel('predicted\n',fontsize=14)
            #     plt.title('Q2: %4.2f  /  SDEP: %4.2f \n' % (self.Q2,self.SDEP), fontsize=14)
            #     plt.plot(Y, y_pred,"ro")
            #     fig1.savefig(self.vpath+"/" + self.name + "-predicted.png", format='png')
            #     fig1.savefig("./" + self.name + "-predicted.png", format='png')
            # except:
            #     print "Error creating Predicted vs Experimental " + self.name + " model graph"

            #if self.conformal:
            #    self.meanConformalInterval = CF_QuanVal(X, Y, copy.copy(self.estimator), self.conformalSignificance)


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
                print ('no objects')
                return

            self.TPpred = TP
            self.TNpred = TN
            self.FPpred = FP
            self.FNpred = FN

            sens = sensitivity (TP, FN)
            spec = specificity (TN, FP)
            self.mccp  = MCC (TP, TN, FP, FN)
            print ("MCCaa")
            f1   = f1_score(Y, Yp, pos_label=1, average='binary')
            if "RF" in self.name:
                self.OOBe = 1.00 - self.estimator.oob_score_
                print ("Recalculated results")
                print ("rec  TP:%d TN:%d FP:%d FN:%d spec:%5.3f sens:%5.3f MCC:%5.3f OOB_error:%5.3f f1_score:%5.3f" % \
                      (TP, TN, FP, FN, spec, sens, self.mcc, self.OOBe, f1 ))
            else:
                print ("Recalculated results")
                print ("rec  TP:%d TN:%d FP:%d FN:%d spec:%5.3f sens:%5.3f MCC:%5.3f f1_score:%5.3f" % \
                      (TP, TN, FP, FN, spec, sens, self.mcc, f1 ))

            # Leave-one-out Cross validation
            print ('Cross validating')
            scoring = 'f1'

            y_pred = cross_val_predict(self.estimator, X, Y, cv=self.cv, n_jobs=-1)


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

            self.TP = TPo
            self.TN = TNo
            self.FP = FPo
            self.FN = FNo

            sens_cv = sensitivity (TPo, FNo)
            spec_cv = specificity (TNo, FPo)
            self.mcc  = MCC (TPo, TNo, FPo, FNo)
            f1_cv = f1_score(Y, y_pred, pos_label=1, average='binary')




            print (str(self.cv)+" cross-validation results")
            print ("pred  TP:%d TN:%d FP:%d FN:%d spec:%5.3f sens:%5.3f MCC:%5.3f f1_score:%5.3f" % \
                  (TPo, TNo, FPo, FNo, spec_cv, sens_cv, self.mccp, f1_cv ))

            # Create Graphs

            # pngfiles = glob.glob (self.vpath+'/*.png')
            # for i in pngfiles:
            #     os.remove(i)

            # # Predicted confusion matrix graph
            # try:
            #     FourfoldDisplay(TPo,TNo,FPo,FNo, self.name +'-Predicted', self.name +'_predicted_confusion_matrix.png' , self.vpath)
            # except:
            #     print "Failed to generate RF predicted validation graph"

            # # Recalculated confusion matrix graph
            # try:
            #     FourfoldDisplay(TP,TN,FP,FN, self.name + ' Recalculated',  self.name +'_recalculated_confusion_matrix.png' , self.vpath)
            # except:
            #     print "Failed to generate RF recalculated validation graph"

        #if self.conformal:
        #    self.conformal_pred = CF_QualVal(X, Y, copy.copy(self.estimator), self.conformalSignificance)


        # Move this to an external module ****
        if self.learning_curve:
            print ('Building Learning Curves')
            title = "Learning Curves (RF)"
            cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
            estimator = self.clf
            plot = plot_learning_curve(estimator, title, self.X, self.Y, (0.0, 1.01), cv=cv)
            plot.savefig(self.vpath+"/" + self.name + "-learning_curves.png", format='png')
            plot.savefig("./" + self.name + "-learning_curves.png", format='png')

        return (Yp)


    def project (self, Xb):
        """ Uses the X matrix provided as argument to predict Y
        """

#        if self.autoscale:
#            Xb = Xb-self.mux
#            Xb = Xb*self.wgx

         # Xb = Xb.reshape(1,-1) # required by sklean, to avoid deprecation warning
        Yp = self.estimator.predict(Xb)

        if self.estimator == None:
            print ('failed to load clasifier')
            return
        if self.conformal:
            predictionSize = 0

            # self.conformal_calibration()
            prediction = conformal_pred_pred(Xb, self.conformal_pred, self.conformalSignificance)

            if self.quantitative:
                predictionSize = abs(abs(prediction[0][0]) - abs(prediction[0][1]))
                prediction = prediction.tolist() + [predictionSize, self.meanConformalInterval]
                return ([Yp, prediction])
            else:
                return ([Yp, prediction])
        else:
            return Yp

    def conformal_calibration(self,):

        X = copy.copy(self.X)
        Y = copy.copy(self.Y)
        if self.autoscale:
            X, self.mux = center(X)
            X, self.wgx = scale(X, self.autoscale)
        if not self.quantitative:
            # self.features is a numpy array -> convert to string

            self.conformal_pred = CF_QualCal(X, Y, copy.copy(self.estimator))
        else:
            # self.features is a numpy array -> convert to string
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




