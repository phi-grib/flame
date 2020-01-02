#! -*- coding: utf-8 -*-

# Description    Set of combination models
##
# Authors:       Manuel Pastor (manuel.pastor@upf.edu)
##
# Copyright 2019 Manuel Pastor
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

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef
from flame.stats.base_model import BaseEstimator
from flame.util import get_logger

LOG = get_logger(__name__)

class Combo (BaseEstimator):
    """
       Generic class for combining results of multiple models
    """
    def __init__(self, X, Y, parameters, conveyor):
        # Initialize parent class
        try:
            BaseEstimator.__init__(self, X, Y, parameters, conveyor)
            LOG.debug('Initialize BaseEstimator parent class')
        except Exception as e:
            LOG.error(f'Error initializing BaseEstimator parent'
                    f'class with exception: {e}')
            raise e

        self.method_name = ''


    def build(self):
        '''nothing to build, just return a some model information '''

        results = []
        results.append(('nobj', 'number of objects', self.nobj))
        results.append(('nvarx', 'number of predictor variables', self.nvarx))
        results.append(('model', 'model type', 'combination:'+self.method_name))

        return True, results

    def predict(self, X):
        ''' the method used to combine the results, this is a dummy method '''
        return 0

    def project(self, Xb):
        '''return the median of the input parameters'''

        Yp = self.predict(Xb)

        self.conveyor.addVal(Yp, 'values', 'Prediction',
                        'result', 'objs',
                        'Results of the prediction', 'main')

    def validate(self):
        ''' validate the model and return a set of results. This version does not performs CV '''

         # Make a copy of the original matrices
        X = self.X.copy()
        Y = self.Y.copy()

        # Get predicted Y
        Yp = self.predict(X)

        info = []

        if self.param.getVal('quantitative'):
            # Compute  mean of predicted Y
            Ym = np.mean(Y)

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
                
        else:
            # Get confusion matrix for predicted Y
            try:
                self.TNpred, self.FPpred,\
                self.FNpred, self.TPpred = confusion_matrix(Y, Yp,
                                                        labels=[0, 1]).ravel()
                self.sensitivityPred = (self.TPpred / (self.TPpred + self.FNpred))
                self.specificityPred = (self.TNpred / (self.TNpred + self.FPpred))
                self.mccp = matthews_corrcoef(Y, Yp)

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

        info.append (('Y_adj', 'Adjusted Y values', Yp) )          

        results = {}
        results ['quality'] = info
        results ['Y_adj'] = Yp

        return True, results

    def save_model(self):
        return

    def load_model(self):
        return


class median (Combo):
    """
       Simple median calculator used to combine the results of multiple models
    """
    def __init__(self, X, Y, parameters, conveyor):
        Combo.__init__(self, X, Y, parameters, conveyor)
        self.method_name = 'median'

    def predict(self, X):

       # obtain dimensions of X matrix
        self.nobj, self.nvarx = np.shape(X)

        CI_names = self.conveyor.getVal('ensemble_confidence_names')

        if  CI_names is not None and len(CI_names)==(2 * self.nvarx):

            # get values
            CI_vals = self.conveyor.getVal('ensemble_confidence')

            # assume that the CI represent 95% CI and normal distribution        
            z = 1.96 
    
            # compute weighted average 
            w = np.zeros(self.nvarx, dtype = np.float64 )
            xmedian = []
            cilow = []
            ciupp = []

            for j in range (self.nobj):
                pred = []
                for i in range (self.nvarx):
                    r = CI_vals[j,i*2+1] - CI_vals[j,i*2]
                    sd = r/(z*2)
                    w[i] = 1.0/np.square(sd)

                    # create a tupla with prediction ID, value and weight
                    pred.append ( (i, X[j,i], w[i]) )
                    
                wcenter = np.sum(w)/2.00

                # sort pred
                sorted_pred = sorted(pred, key=lambda tup: tup[1])

                # fpr even number of predictions
                if self.nvarx % 2 == 0:

                    if self.nvarx == 2:
                        selectedA = 0
                        selectedB = 1
                    else:
                        acc_w = 0.00
                        selectedA = sorted_pred[0][0]
                        for i,ipred in enumerate(sorted_pred):
                            selectedB = ipred[0]
                            # accumulate weights
                            acc_w += ipred[2]
                            if acc_w > wcenter:
                                break
                        selectedA = sorted_pred[i-1][0]

                    print ('even',j, selectedA, selectedB)

                    xmedian.append(np.mean((X[j,selectedA], X[j,selectedB])))
                    cilow.append(np.mean((CI_vals[j,selectedA*2], CI_vals[j,selectedB*2])))
                    ciupp.append(np.mean((CI_vals[j,(selectedA*2)+1], CI_vals[j,(selectedB*2)+1])))

                # for odd number of predictions
                else:
                    acc_w = 0.00
                    for ipred in sorted_pred:
                        selected = ipred[0]    
                        # accumulate weights
                        acc_w += ipred[2]
                        if acc_w >= wcenter:
                            break

                    print ('odd',j, selected)

                    xmedian.append(X[j,selected])
                    cilow.append(CI_vals[j,selected*2])
                    ciupp.append(CI_vals[j,(selected*2)+1])

            self.conveyor.addVal(np.array(cilow), 
                        'lower_limit', 
                        'Lower limit', 
                        'confidence',
                        'objs',
                        'Lower limit of the conformal prediction'
                    )

            self.conveyor.addVal(np.array(ciupp), 
                        'upper_limit', 
                        'Upper limit', 
                        'confidence',
                        'objs',
                        'Upper limit of the conformal prediction'
                    )

            #print (xmedian, cilow, ciupp)

            return np.array(xmedian)

        else:

            return np.median (X,1)

class mean (Combo):
    """
       Simple mean calculator used to combine the results of multiple models
    """
    def __init__(self, X, Y, parameters, conveyor):
        Combo.__init__(self, X, Y, parameters, conveyor)
        self.method_name = 'mean'

    def predict(self, X):

        # obtain dimensions of X matrix
        self.nobj, self.nvarx = np.shape(X)

        CI_names = self.conveyor.getVal('ensemble_confidence_names')

        if  CI_names is not None and len(CI_names)==(2 * self.nvarx):

            # compute weigthed mean and CI for the estimator
            # as described here
            #
            #   https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Weighted_sample_variance
            #

            # get values
            CI_vals = self.conveyor.getVal('ensemble_confidence')

            # assume that the CI represent 95% CI and normal distribution        
            z = 1.96 
    
            # compute weighted average 
            xmean = []
            cilow = []
            ciupp = []

            for j in range (self.nobj):
                w = np.zeros(self.nvarx, dtype = np.float64 )
                for i in range (self.nvarx):
                    r = CI_vals[j,i*2+1] - CI_vals[j,i*2]
                    sd = r/(z*2)
                    w[i] = 1.0/np.square(sd)

                ws = np.sum(w)
                s = 1.0/np.sqrt(ws)

                xm = 0.0
                for i in range (self.nvarx):
                    xm += X[j,i]*w[i]
                xmean.append(xm/ws) 

                cilow.append(xm-z*s)
                ciupp.append(xm+z*s)

            self.conveyor.addVal(np.array(cilow), 
                        'lower_limit', 
                        'Lower limit', 
                        'confidence',
                        'objs',
                        'Lower limit of the conformal prediction'
                    )

            self.conveyor.addVal(np.array(ciupp), 
                        'upper_limit', 
                        'Upper limit', 
                        'confidence',
                        'objs',
                        'Upper limit of the conformal prediction'
                    )
            
            return np.array(xmean)
        else:
            return np.mean (X,1)

class majority (Combo):
    """
       Simple majority voting calculator used to combine the results of multiple models
    """
    def __init__(self, X, Y, parameters, conveyor):
        Combo.__init__(self, X, Y, parameters, conveyor)
        self.method_name = 'majority voting'

        # majority is not compatible with conformal because the prediction results
        # are not stored as c0, c1 but as value, ensemble_c0, ensemble_c1
        if self.param.getVal('conformal'):
            self.param.setVal('conformal', False)

    def predict(self, X):

        # obtain dimensions of X matrix
        self.nobj, self.nvarx = np.shape(X)

        # check if the underlying models are conformal
        confidence = self.conveyor.getVal('ensemble_confidence')

        # when not all models are conformal use a simple approach
        if confidence is None or len(confidence[0]) != (2 * self.nvarx):
            return np.round(np.mean (X,1))
        
        # print (confidence)

        # if all models are conformal, simply add the classes
        # and return 0 if majority is class 0, 1 if majority is class 1
        # and -1 if there is a tie
        yp = np.zeros(self.nobj, dtype=np.float64)
        c0 = np.zeros(self.nobj, dtype=np.float64)
        c1 = np.zeros(self.nobj, dtype=np.float64)

        for i,iobj in enumerate(confidence):
            for j in range(self.nvarx):
                c0[i] += iobj[j*2]
                c1[i] += iobj[(j*2)+1] 
            if c1[i] > c0[i]:
                yp[i] = 1
            elif c0[i] == c1[i]:
                yp[i] = -1

        # add the sum of classes for evaluating the result
        self.conveyor.addVal(c0, 
                    'ensemble_c0', 
                    'Ensemble Class 0', 
                    'confidence',
                    'objs',
                    'Conformal class assignment'
                )

        self.conveyor.addVal(c1, 
                    'ensemble_c1', 
                    'Ensemble Class 1', 
                    'confidence',
                    'objs',
                    'Conformal class assignment'
                )
        
        return yp

class matrix (Combo):
    """
       Lockup matrix
    """
    def __init__(self, X, Y, parameters, conveyor):
        Combo.__init__(self, X, Y, parameters, conveyor)
        self.method_name = 'matrix'



    def predict(self, X):

        #TODO: load this from a file
        #load info for each input value [name] (loop + vnum + vzero + vstep)
        mmatrix = {}
        mmatrix ['CACO2'] = [0,3,-7.0,1.0]
        mmatrix ['CACO3'] = [1,3,-7.0,1.0]
        
        #load input matrix 
        vmatrix = [ 10.0,12.0,15.0,
                    11.0,20.0,25.0,
                    12.0,18.0,30.0]
        ##################################

        # obtain dimensions of X matrix
        self.nobj, self.nvarx = np.shape(X)
        ymatrix = []

        var_names = self.conveyor.getVal('var_nam')
        
        vloop = []
        vsize = []
        vzero = []
        vstep = []
        for i in var_names:
            vname = i.split(':')[1]
            if not vname in mmatrix:
                return False
            
            vloop.append(mmatrix[vname][0]) # inner loop is 0, then 1, 2 etc...
            vsize.append(mmatrix[vname][1]) # number of bins in the matrix for this variable
            vzero.append(mmatrix[vname][2]) # origin (left side) of the first bin 
            vstep.append(mmatrix[vname][3]) # width of each bin, must the identical for all bins

        # compute offset for each variable in sequential order
        # this means computing the factor for which each
        # variable value must be multiplied in order to identify
        # the position the linear vector representing 

        offset = []
        for i in range(self.nvarx):
            ioffset = 1.0
            for j in range(self.nvarx):
                if vloop[j]<vloop[i]:
                    ioffset*=vsize[j]
            offset.append(int(ioffset))

        print ('offset:', offset)

        for j in range (self.nobj):

            ############## move all this to a function with 
            # input  = the vector of variables X[j] 
            # output = the Y value
            # ############################################## 
            index = []
            for i in range (self.nvarx):
                # value of the variable to be binned
                valuei = X[j,i]

                # max value in the matrix
                valuemax = vzero[i]+(vsize[i]*vstep[i])

                # any value smaller than vzero plus vstep will 
                # fall in the first cell (0) 
                vrule = vzero[i]+vstep[i]
                
                irule = 0
                while vrule < valuemax:
                    if valuei < vrule:
                        index.append(irule)
                        break
                    else:
                        vrule += vstep[i]
                        irule += 1
                        if vrule >= valuemax:
                            index.append(irule)
                            break

            matrix_index = 0
            for i in range(self.nvarx):
                matrix_index += (index[i]*offset[i])

            ymatrix.append (vmatrix[matrix_index]) 
            # ############################################## 

            print ('matrix index, array index and value:', index, matrix_index, vmatrix[matrix_index])


        CI_names = self.conveyor.getVal('ensemble_confidence_names')
        if  CI_names is not None and len(CI_names)==(2 * self.nvarx):

            # compute weigthed mean and CI for the estimator
            # as described here
            #
            #   https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Weighted_sample_variance
            #

            # get values
            CI_vals = self.conveyor.getVal('ensemble_confidence')

            # assume that the CI represent 95% CI and normal distribution        
            z = 1.96 
    
            # # compute weighted average 
            # xmean = []
            cilow = []
            ciupp = []

            for j in range (self.nobj):
                # w = np.zeros(self.nvarx, dtype = np.float64 )

                # y1000 = []
                #for r in range (1000):

                    # (to increase efficiency, do this out of the 1000 loop and store values)
                    # for i in range (self.nvarx):
                    #     r = CI_vals[j,i*2+1] - CI_vals[j,i*2]
                    #     sd = r/(z*2)


                        # compute a random number with mean 0 and sd 
                        # add this to the X[j,i]
                      
                    # transform X values into lookup indexes
                    # get value
                    # append value to list of 1000 y's

                # obtain percentile 5 and 95 from list of 1000 y's

                cilow.append(0.000)
                ciupp.append(10.000)

            self.conveyor.addVal(np.array(cilow), 
                        'lower_limit', 
                        'Lower limit', 
                        'confidence',
                        'objs',
                        'Lower limit of the conformal prediction'
                    )

            self.conveyor.addVal(np.array(ciupp), 
                        'upper_limit', 
                        'Upper limit', 
                        'confidence',
                        'objs',
                        'Upper limit of the conformal prediction'
                    )

        return np.mean(X,1)
            