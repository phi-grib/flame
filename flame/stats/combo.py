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
import copy
import yaml
import os
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef
from scipy import stats
from flame.stats.base_model import BaseEstimator
from flame.util import get_logger

LOG = get_logger(__name__)
SIMULATION_SIZE = 500
CONFIDENCE = 0.95

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
                if SSY0 == 0.00:
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

       TODO: 
       - implementing qualitative input and/or output
       - use conformal settings to decide to run or not the simulations to compute CI

    """
    def __init__(self, X, Y, parameters, conveyor):
        Combo.__init__(self, X, Y, parameters, conveyor)
        self.model_path = parameters.getVal('model_path')
        self.method_name = 'matrix'


    def lookup (self, x, vmatrix):
        """Uses the array x of quantitative values to lookup in the matrix of values vmatrix the
         corresponding value 
         
         The binning of the cells of vmatrix are defined by the following class variables which 
         define, for each matrix dimension: 
          - self.vzero: starting value
          - self.vsize: number of cells
          - self.vstep: width of bins 
         
         Once the variables are transformed into vmatrix indexes, a single matrix_index is computed
         since the n-dimensional vmatrix is stored as a deconvoluted monodimensional array
        """

        # transform the values of the vectors into vmatrix indexes
        # note that:
        #   values < self.vzero are set to 0
        #   values > self.vzero + self.vsize*self.vstep are set to self.vsize

        index = []
        for i in range (self.nvarx):
            cellmax = self.vzero[i]
            step = self.vstep[i]

            # if values grow, find the first j producing matrix 
            # value bigger than the x 
            if step > 0.0: 
                for j in range (int(self.vsize[i])):
                    cellmax += step
                    if x[i] < cellmax:
                        break
                        # if values shrink, find the first j producing matrix 
            # value lower than the x 
            else:          
                for j in range (int(self.vsize[i])):
                    cellmax += step
                    if x[i] > cellmax:
                        break
            index.append (j)

        # compute the index in the deconvoluted monodimensional vector where
        # the values of vmatrix are stored
        matrix_index = 0
        for i in range(self.nvarx):
            matrix_index += (index[i]*self.offset[i])

        return vmatrix[matrix_index]

    def load_data(self):
        ''' read the matrix, stored as a 1D or 2D table of floats, separted by ',' 
            and the metaiformation 
        '''
        #load input matrix metadata
        mmatrix_path = os.path.join(self.model_path,'mmatrix.yaml')
        with open(mmatrix_path, 'r') as f:
            mmatrix = yaml.safe_load(f)
        
        #load input matrix 
        vmatrix_path = os.path.join(self.model_path,'vmatrix.txt')
        with open(vmatrix_path) as f:
            vmatrix = np.loadtxt(f, delimiter=',')
            if len(np.shape(vmatrix))>1:
                vmatrix = vmatrix.flatten()

        return mmatrix, vmatrix

    def preprocess (self, X):
        ''' transform to customize input values, before looking into the table '''
        return X

    def postprocess (self, varray):
        ''' transform to customize output values, after they were extracted from the table 
            input is an array of np.arrays
            For simple predictions, it only contains a single value
            For simulations, it contains the low, up and mean values of the CI
        '''
        return varray

    def predict(self, X):
        ''' return a prediction obtained by looking up a table of preprocessed values
            The input X values are converted to the matrix indexes
            When all the X values have an associated error, run a simulation to estimate the
            output error 
        '''
        # obtain dimensions of X matrix
        self.nobj, self.nvarx = np.shape(X)

        # apply custom modifications to the input values
        X = self.preprocess (X)

        # load matrix and matrix metadata        
        mmatrix, vmatrix = self.load_data()

        # this is the array of predicted Y values 
        yarray = []

        # assign metainformation to every variable
        var_names = self.conveyor.getVal('var_nam')
        
        self.vloop = []
        self.vsize = []
        self.vzero = []
        self.vstep = []
        for i in var_names:
            vname = i.split(':')[1]
            if not vname in mmatrix:
                return False
            
            self.vloop.append(mmatrix[vname][0]) # inner loop is 0, then 1, 2 etc...
            self.vsize.append(mmatrix[vname][1]) # number of bins in the matrix for this variable
            self.vzero.append(mmatrix[vname][2]) # origin (left side) of the first bin 
            self.vstep.append(mmatrix[vname][3]) # width of each bin, must the identical for all bins

        # check that the size of the vmatrix corresponds with the vsize
        mlen = 1
        for i in self.vsize:
            mlen *= i
        if int(mlen) != len(vmatrix):
            raise Exception ('vmatrix size does not match metadata')

        # compute offset for each variable in sequential order
        # this means computing the factor for which each
        # variable value must be multiplied in order to identify
        # the position the linear vector representing 

        self.offset = []
        for i in range(self.nvarx):
            ioffset = 1.0
            for j in range(self.nvarx):
                if self.vloop[j]<self.vloop[i]:
                    ioffset*=self.vsize[j]
            self.offset.append(int(ioffset))

        # if all the original methods contain CI run a simulation to compute the CI for the 
        # output values and return the mean, the 5% percentil and 95% percentil of the values obtained 
        CI_names = self.conveyor.getVal('ensemble_confidence_names')
        if  CI_names is not None and len(CI_names)==(2 * self.nvarx):

            # get CI values
            CI_vals = self.conveyor.getVal('ensemble_confidence')

            # we read the conformal significance from the top model
            # ideally we must read this from every bottom model and store a list of
            # conformal significances at the conveyor to derive individual z for
            # each variable
            # TODO: implement this
            conformal_significance = self.param.getVal('conformalSignificance') 
            conformal_confidence_left  = conformal_significance /2.0
            conformal_confidence_right = 1.0 - conformal_confidence_left

            confidence_left  = (1.0 - CONFIDENCE)/2.0
            confidence_right = 1.0 - confidence_left
            print ("confidences: ", CONFIDENCE, confidence_left, confidence_right)

            z = stats.norm.ppf (conformal_confidence_right)
    
            cilow = []
            ciupp = []
            cimean = []

            # make sure the random numbers are reproducible
            np.random.seed(2324)

            for j in range (self.nobj):
                ymulti = []
                for m in range (SIMULATION_SIZE):
                    x = copy.copy(X[j])
                    for i in range (self.nvarx):
                        #ci range is the width of the CI
                        cirange = CI_vals[j,i*2+1] - CI_vals[j,i*2]

                        # we asume that the CI were estimated as +/- 1.96 * SE
                        sd = cirange/(z*2)

                        # now we add normal random noise, with mean 0 and SD = sd
                        x[i]+=np.random.normal(0.0,sd)

                    ymulti.append (self.lookup (x,vmatrix))

                ymulti_array = np.array(ymulti)

                # obtain percentiles to estimate the left and right part of the CI 
                cilow.append (np.percentile(ymulti_array,confidence_left*100 ,interpolation='linear'))
                ciupp.append (np.percentile(ymulti_array,confidence_right*100 ,interpolation='linear'))
                cimean.append(np.percentile(ymulti_array,50,interpolation='linear'))
                # cimean.append(np.median(ymulti_array))

            cival = [cilow, ciupp, cimean]
            cival = self.postprocess (cival)

            self.conveyor.addVal(cival[0], 
                        'lower_limit', 
                        'Lower limit', 
                        'confidence',
                        'objs',
                        'Lower limit of the conformal prediction'
                    )

            self.conveyor.addVal(cival[1], 
                        'upper_limit', 
                        'Upper limit', 
                        'confidence',
                        'objs',
                        'Upper limit of the conformal prediction'
                    )
        
            for i in range (len(cival[0])):
                print (f'{cival[0][i]:.2f} - {cival[2][i]:.2f} - {cival[1][i]:.2f}')

            yarray = np.array(cival[2])

        else:

            # For each object look up in the vmatrix, by transforming the input X variables
            # into indexes and then extracting the corresponding values
            for j in range (self.nobj):
                yarray.append (self.lookup (X[j],vmatrix))

            sval = [np.array(yarray)]
            yarray = self.postprocess(sval) 

        return yarray
            