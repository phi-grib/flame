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

        conveyor.addVal(Yp, 'values', 'Prediction',
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

            #TODO: compute a weighted median and associated CI

            # the concept could be based in 
            # order the values and CI according to the value
            # weigth the prediction by 1/var
            # compute the center using the weights selecting the value having a balanced weight in both sides
            # assign the CI of the value or the mean of the adjacent values
            #   
            #    https://en.wikipedia.org/wiki/Weighted_median
            #
            # conveyor.addVal(np.array(cilow), 
            #             'lower_limit', 
            #             'Lower limit', 
            #             'confidence',
            #             'objs',
            #             'Lower limit of the conformal prediction'
            #         )

            # conveyor.addVal(np.array(ciupp), 
            #             'upper_limit', 
            #             'Upper limit', 
            #             'confidence',
            #             'objs',
            #             'Upper limit of the conformal prediction'
            #         )

            # provisional
            return np.median (X,1)

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
            w = np.zeros(self.nvarx, dtype = np.float64 )
            xmean = []
            cilow = []
            ciupp = []

            for j in range (self.nobj):
                for i in range (self.nvarx):
                    r = CI_vals[j,i*2+1] - CI_vals[j,i*2]
                    sd = r/(z*2)
                    w[i] = 1.0/(sd*sd)

                ws = np.sum(w)
                s = 1.0/np.sqrt(ws)

                xm = 0.0
                for i in range (self.nvarx):
                    xm += X[j,i]*w[i]
                xmean.append(xm/ws) 

                cilow.append(xm-z*s)
                ciupp.append(xm+z*s)

            conveyor.addVal(np.array(cilow), 
                        'lower_limit', 
                        'Lower limit', 
                        'confidence',
                        'objs',
                        'Lower limit of the conformal prediction'
                    )

            conveyor.addVal(np.array(ciupp), 
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

    def predict(self, X):

        # check if the undelying models use conformal methods and then handle classes
        # for example, for three models
        # 
        # clear majority -> majority
        # 10 10 10        10
        # 10 10 01        10
        # 01 01 10        01
        # 01 01 01        01
        #
        # minority of uncertain -> ignored
        # 11 10 10        10
        # 00 10 10        10
        # 11 01 01        01
        # 00 01 01        01
        #   
        # ties -> uncertain
        # 11 10 01        11
        # 00 10 01        11
        #
        # majority of uncertain -> uncertain
        # 11 11 01        11
        # 11 00 01        11
        # 11 11 10        11
        # 11 00 01        11

        # proposed algorithm:
        # 1. If majority of results is uncertain -> uncertain (use 11 or 00 as the most frequent result)
        # 2. Remove uncertains
        # 3. It result is a tie -> uncertain (11)
        # 4. Compute majority


        return np.round(np.mean (X,1))

