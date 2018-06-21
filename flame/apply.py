#! -*- coding: utf-8 -*-

# Description    Flame Apply class
##
# Authors:       Manuel Pastor (manuel.pastor@upf.edu), Jose Carlos Gómez (josecarlos.gomez@upf.edu)
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

import numpy as np
import pickle
import os
from flame.util import utils

from sklearn.metrics import mean_squared_error, matthews_corrcoef as mcc
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer
from sklearn.metrics import confusion_matrix

class Apply:

    def __init__(self, parameters, results):

        self.parameters = parameters
        self.results = results

        self.results['origin'] = 'apply'

    def external_validation(self):
        ''' when experimental values are available for the predicted compounds, apply external validation '''

        if not 'ymatrix' in self.results:
            return

        ext_val_results  = []

        if not self.parameters["quantitative"]:
            Ye = np.asarray(self.results["ymatrix"])
            Yp = np.asarray(self.results["values"])

            if Ye.size == 0:
                raise ValueError("Experimental activity vector is empty")
            if Yp.size == 0:
                raise ValueError("Predicted activity vector is empty")

            # the use of labels is compulsory to inform the confusion matrix that
            # it must return a 2x2 confussion matrix. Otherwise it will fail when
            # a single class is represented (all TP, for example) 
            TN, FP, FN, TP = confusion_matrix(Ye, Yp, labels = [0,1]).ravel()

            # protect to avoid warnings in special cases (div by zero)
            MCC = mcc(Ye, Yp)

            if (TP+FN) > 0:
                sensitivity = (TP / (TP + FN))
            else:
                sensitivity = 0.0

            if (TN+FP) > 0:
                specificity = (TN / (TN + FP))
            else:
                specificity = 0.0

            ext_val_results.append (('TP','True positives in external-validation', float(TP)))
            ext_val_results.append (('TN','True negatives in external-validation', float(TN)))
            ext_val_results.append (('FP','False positives in external-validation', float(FP)))
            ext_val_results.append (('FN','False negatives in external-validation', float(FN)))

            ext_val_results.append (('Sensitivity','Sensitivity in external-validation', float(sensitivity)))
            ext_val_results.append (('Specificity','Specificity in external-validation', float(specificity)))
            ext_val_results.append (('MCC', 'Mattews Correlation Coefficient in external-validation', float(MCC )))

        else:
            Ye = np.asarray(self.results["ymatrix"])
            Yp = np.asarray(self.results["values"])
            Ym = np.mean(Ye)
            nobj = len(Yp)

            SSY0_out = np.sum(np.square(Ym - Ye))
            SSY_out = np.sum(np.square(Ye - Yp))
            scoringP = mean_squared_error(Ye, Yp)
            SDEP = np.sqrt(SSY_out/(nobj))
            Q2 = 1.00 - (SSY_out/SSY0_out)

            ext_val_results.append (('scoringP','Scoring P', scoringP))
            ext_val_results.append (('Q2','Determination coefficient in cross-validation', Q2))
            ext_val_results.append (('SDEP','Standard Deviation Error of the Predictions', SDEP))

        utils.add_result(self.results, ext_val_results, 'external-validation', 'external validation', 'method', 'single', 'External validation results')
            

    def run_internal(self):
        ''' 

        Runs prediction tasks using internally defined methods

        Most of these methods can be found at the stats folder

        '''

        # assume X matrix is present in 'xmatrix0
        X = self.results["xmatrix"]

        # retrieve data and dimensions from results
        try:
            nobj, nvarx = np.shape(X)
        except:
            self.results['error'] = 'Failed to generate MD'
            return

        if (nobj == 0) or (nvarx == 0):
            self.results['error'] = 'Failed to extract activity or to generate MD'
            return

        try:
            model_file = os.path.join(self.parameters['model_path'],'model.pkl')
            with open(model_file, "rb") as input_file:
                estimator = pickle.load(input_file)
        except:
            self.results['error'] = 'No valid model estimator found'
            return

        estimator.project(X, self.results)

        # if len(self.results["ymatrix"]) > 0:
        #     # print (len(self.results["ymatrix"]))
        #     # print (self.parameters["conformal"])

        if not self.parameters["conformal"]:
            self.external_validation()
            
        # TODO: implement this for every prediction
        # zero_array = np.zeros(nobj, dtype=np.float64)

        # if not 'CI' in self.results:
        #     self.results['CI'] = zero_array
        # if not 'RI' in self.results:
        #     self.results['RI'] = zero_array

        # utils.add_result (self.results, zero_array, 'CI', 'CI (95%)',
        # 'confidence', 'objs', 'Approximate 95% Confidence Interval')

        # utils.add_result (self.results, zero_array, 'RI', 'RI (prob)',
        # 'confidence', 'objs', 'Reliability Index, from 0 (good) to 6 (bad)')

        return

    def run_R(self):
        ''' Runs prediction tasks using an importer KNIME workflow '''
        self.results['error'] = 'R toolkit is not supported in this version'
        return

    def run_KNIME(self):
        ''' Runs prediction tasks using R code '''
        self.results['error'] = 'KNIME toolkit is not supported in this version'
        return

    def run_custom(self):
        ''' Template to be overriden in apply_child.py

            Input: must be already present in self.results
            Output: add prediction results to self.results using the utils.add_result() method 

        '''

        self.results['error'] = 'custom prediction must be defined in the model apply_chlid class'
        return

    def run(self):
        ''' 

        Runs prediction tasks using the information present in self.results. 

        Depending on the modelingToolkit defined in self.parameters this task will use internal methods
        or make use if imported code in R/KNIME

        The custom option allows advanced uses to write their own function 'run_custom' method in 
        the model apply_child.py

        '''

        if self.parameters['modelingToolkit'] == 'internal':
            self.run_internal()
        elif self.parameters['modelingToolkit'] == 'R':
            self.run_R()
        elif self.parameters['modelingToolkit'] == 'KNIME':
            self.run_KNIME()
        elif self.parameters['modelingToolkit'] == 'custom':
            self.run_custom()
        else:
            self.results['error'] = 'Unknown prediction toolkit to run ', self.parameters['modelingToolkit']

        return self.results
