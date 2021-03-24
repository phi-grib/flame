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

from flame.stats.RF import RF
from flame.stats.SVM import SVM
from flame.stats.GNB import GNB
from flame.stats.PLSR import PLSR
from flame.stats.PLSDA import PLSDA
from flame.stats.combo import median, mean, majority, logicalOR, matrix

from sklearn.metrics import mean_squared_error, matthews_corrcoef as mcc
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer
from sklearn.metrics import confusion_matrix

from flame.graph.graph import projectPredictions
from flame.graph.graph import projectManifoldPredictions
from flame.graph.graph import projecttsnePredictions
from flame.graph.graph import projectIsomapPredictions

from flame.util import utils, get_logger
LOG = get_logger(__name__)


class Apply:

    def __init__(self, parameters, conveyor):

        self.param = parameters
        self.conveyor = conveyor




    # def external_validation(self):
    #     ''' when experimental values are available for the predicted compounds,
    #     run external validation '''

    #     ext_val_results = []
        
    #     # Ye are the y values present in the input file
    #     Ye = np.asarray(self.conveyor.getVal("ymatrix"))

    #     # For qualitative models, make sure the Y is qualitative as well
    #     if not self.param.getVal("quantitative"):
    #         qy, message = utils.qualitative_Y(Ye)
    #         if not qy:
    #             self.conveyor.setWarning(f'No qualitative activity suitable for external validation "{message}". Skipping.')
    #             LOG.warning(f'No qualitative activity suitable for external validation "{message}". Skipping.')
    #             return

    #     # there are four variants of external validation, depending if the method
    #     # if conformal or non-conformal and the model is qualitative and quantitative

    #     if not self.param.getVal("conformal"):

    #         # non-conformal 
    #         if not self.param.getVal("quantitative"):
                
    #             # non-conformal & qualitative
    #             Yp = np.asarray(self.conveyor.getVal("values"))

    #             if Ye.size == 0:
    #                 raise ValueError("Experimental activity vector is empty")
    #             if Yp.size == 0:
    #                 raise ValueError("Predicted activity vector is empty")

    #             # the use of labels is compulsory to inform the confusion matrix that
    #             # it must return a 2x2 confussion matrix. Otherwise it will fail when
    #             # a single class is represented (all TP, for example)
    #             TN, FP, FN, TP = confusion_matrix(
    #                 Ye, Yp, labels=[0, 1]).ravel()

    #             # protect to avoid warnings in special cases (div by zero)
    #             MCC = mcc(Ye, Yp)

    #             if (TP+FN) > 0:
    #                 sensitivity = (TP / (TP + FN))
    #             else:
    #                 sensitivity = 0.0

    #             if (TN+FP) > 0:
    #                 specificity = (TN / (TN + FP))
    #             else:
    #                 specificity = 0.0

    #             ext_val_results.append(('TP','True positives in external-validation', float(TP)))
    #             ext_val_results.append(('TN','True negatives in external-validation', float(TN)))
    #             ext_val_results.append(('FP','False positives in external-validation', float(FP)))
    #             ext_val_results.append(('FN','False negatives in external-validation', float(FN)))
    #             ext_val_results.append(('Sensitivity', 'Sensitivity in external-validation', float(sensitivity)))
    #             ext_val_results.append(('Specificity', 'Specificity in external-validation', float(specificity)))
    #             ext_val_results.append(('MCC', 'Mattews Correlation Coefficient in external-validation', float(MCC)))

    #         else:

    #             # non-conformal & quantitative
    #             Yp = np.asarray(self.conveyor.getVal("values"))

    #             if Ye.size == 0:
    #                 raise ValueError("Experimental activity vector is empty")
    #             if Yp.size == 0:
    #                 raise ValueError("Predicted activity vector is empty")

    #             Ym = np.mean(Ye)
    #             nobj = len(Yp)

    #             SSY0_out = np.sum(np.square(Ym - Ye))
    #             SSY_out = np.sum(np.square(Ye - Yp))
    #             scoringP = mean_squared_error(Ye, Yp)
    #             SDEP = np.sqrt(SSY_out / (nobj))
    #             if SSY0_out == 0:
    #                 Q2 = 0.0
    #             else:
    #                 Q2 = 1.00 - (SSY_out / SSY0_out)

    #             ext_val_results.append(
    #                 ('scoringP', 'Scoring P', scoringP))
    #             ext_val_results.append(
    #                 ('Q2', 'Determination coefficient in cross-validation', Q2))
    #             ext_val_results.append(
    #                 ('SDEP', 'Standard Deviation Error of the Predictions', SDEP))

    #         self.conveyor.addVal(
    #                          ext_val_results,
    #                          'external-validation',
    #                          'external validation',
    #                          'method',
    #                          'single',
    #                          'External validation results')

    #     else:
    #         # conformal external validation

    #         if not self.param.getVal("quantitative"):
                
    #             # conformal & qualitative
    #             Yp = np.concatenate((np.asarray(self.conveyor.getVal('c0')).reshape(
    #                 -1, 1), np.asarray(self.conveyor.getVal('c1')).reshape(-1, 1)), axis=1)

    #             if Ye.size == 0:
    #                 raise ValueError("Experimental activity vector is empty")
    #             if Yp.size == 0:
    #                 raise ValueError("Predicted activity vector is empty")

    #             c0_correct = 0
    #             c1_correct = 0
    #             not_predicted = 0
    #             c0_incorrect = 0
    #             c1_incorrect = 0

    #             Ye1 = []
    #             Yp1 = []
    #             for i in range(len(Ye)):
    #                 real = float(Ye[i])
    #                 predicted = Yp[i]
    #                 if predicted[0] != predicted[1]:
    #                     Ye1.append(real)
    #                     if predicted[0]:
    #                         Yp1.append(0)
    #                     else:
    #                         Yp1.append(1)

    #                     if real == 0 and predicted[0] == True:
    #                         c0_correct += 1
    #                     if real == 0 and predicted[1] == True:
    #                         c0_incorrect += 1
    #                     if real == 1 and predicted[1] == True:
    #                         c1_correct += 1
    #                     if real == 1 and predicted[0] == True:
    #                         c1_incorrect += 1
    #                 else:
    #                     not_predicted += 1
    #             MCC = mcc(Ye1, Yp1)
    #             TN = c0_correct
    #             FP = c0_incorrect
    #             TP = c1_correct
    #             FN = c1_incorrect
    #             coverage = float((len(Yp) - not_predicted) / len(Yp))

    #             try:
    #                 # Compute accuracy (% of correct predictions)
    #                 conformal_accuracy = (float(TN + TP) /  float(FP + FN + TN + TP))
    #             except Exception as e:
    #                 LOG.error(f'Failed to compute conformal accuracy with'
    #                             f'exception {e}')
    #                 conformal_accuracy = '-'
                                                            
    #             if (TP+FN) > 0:
    #                 sensitivity = (TP / (TP + FN))
    #             else:
    #                 sensitivity = 0.0
    #             if (TN+FP) > 0:
    #                 specificity = (TN / (TN + FP))
    #             else:
    #                 specificity = 0.0

    #             ext_val_results.append(('TP','True positives in external-validation', float(TP)))
    #             ext_val_results.append(('TN','True negatives in external-validation', float(TN)))
    #             ext_val_results.append(('FP','False positives in external-validation', float(FP)))
    #             ext_val_results.append(('FN', 'False negatives in external-validation', float(FN)))
    #             ext_val_results.append(('Sensitivity', 'Sensitivity in external-validation', float(sensitivity)))
    #             ext_val_results.append(('Specificity', 'Specificity in external-validation', float(specificity)))
    #             ext_val_results.append(('MCC', 'Mattews Correlation Coefficient in external-validation', float(MCC)))
    #             ext_val_results.append(('Conformal_coverage', 'Conformal coverage in external-validation', float(coverage)))
    #             ext_val_results.append(('Conformal_accuracy', 'Conformal accuracy in external-validation', float(conformal_accuracy)))

    #             self.conveyor.addVal( ext_val_results,
    #                                 'external-validation',
    #                                 'external validation',
    #                                 'method',
    #                                 'single',
    #                                 'External validation results')
    #         else:

    #             # conformal & quantitative

    #             Yp_lower = np.asarray(self.conveyor.getVal('lower_limit'))
    #             Yp_upper = np.asarray(self.conveyor.getVal('upper_limit'))

    #             mean_interval = np.mean(np.abs(Yp_lower) - np.abs(Yp_upper))
    #             interval_means = (Yp_lower + Yp_upper) / 2

    #             inside_interval = (Yp_lower.reshape(-1, 1) <
    #                                Ye) & (Yp_upper.reshape(-1, 1) > Ye)
    #             accuracy = len(inside_interval)/len(Ye)
    #             conformal_accuracy = float("{0:.2f}".format(accuracy))
    #             conformal_mean_interval = float(
    #                 "{0:.2f}".format(mean_interval))
    #             ext_val_results.append(('Conformal_mean_interval',
    #                                     'Conformal mean interval',
    #                                     conformal_mean_interval))
    #             ext_val_results.append(('Conformal_accuracy',
    #                                     'Conformal accuracy',
    #                                     conformal_accuracy))
    #             # Compute classic Cross-validation quality metrics using inteval mean
    #             try:
    #                 nobj = len(Ye)
    #                 Ym = np.mean(Ye)
    #                 SSY0_out = np.sum(np.square(Ym - Ye))
    #                 SSY_out = np.sum(np.square(Ye - interval_means))
    #                 scoringP = mean_squared_error(Ye, interval_means)
    #                 SDEP = np.sqrt(SSY_out/(nobj))
    #                 if SSY0_out == 0.0:
    #                     Q2 = 0.0
    #                 else:
    #                     Q2 = 1.00 - (SSY_out/SSY0_out)

    #                 ext_val_results.append(('scoringP', 'Scoring P', scoringP))
    #                 ext_val_results.append(('Q2', 'Determination coefficient in cross-validation',  Q2))
    #                 ext_val_results.append(('SDEP', 'Standard Deviation Error of the Predictions',  SDEP))

    #             except Exception as e:
    #                 LOG.error(f'Error in external validation with exception {e}')
    #                 raise e

    #             self.conveyor.addVal( ext_val_results,
    #                                  'external-validation',
    #                                  'external validation',
    #                                  'method',
    #                                  'single',
    #                                  'External validation results')

    def preprocess(self, X):
        ''' This function loads the scaler and variable mask from a pickle file 
        and apply them to the X matrix passed as an argument'''

        prepro_file = os.path.join(self.param.getVal('model_path'), 'preprocessing.pkl')

        LOG.debug(f'Loading model from pickle file, path: {prepro_file}')
        try:
            with open(prepro_file, "rb") as input_file:
                dict_prepro = pickle.load(input_file)
        except FileNotFoundError:
            return False, f'No valid preprocessing tools found at: {prepro_file}'

        # Load version
        self.version = dict_prepro['version']

        # check if the pickle was created with a compatible version
        # currently 1
        if self.version is not 1:
            return False, 'Incompatible preprocessing version'   
    
        # Load rest of info in an extensible way
        # This allows to add new variables keeping
        # Retro-compatibility
        self.variable_mask = None
        if 'variable_mask' in dict_prepro.keys():
            self.variable_mask = dict_prepro['variable_mask']

        if self.param.getVal('feature_selection') and self.variable_mask is None:
            return False, 'Inconsistency error. Feature is True in parameter file but no variable mask loaded'

        # apply variable_mask
        if self.param.getVal("feature_selection"):
            X = X[:, self.variable_mask]

        if self.param.getVal('modelAutoscaling') is None:
            return True, X

        # Load rest of info in an extensible way
        # This allows to add new variables keeping
        # Retro-compatibility
        self.scaler = None
        if 'scaler' in dict_prepro.keys():
            self.scaler = dict_prepro['scaler']

        # Check consistency between parameter file and pickle info
        non_scale_list = ['majority','logicalOR','matrix']
        if self.scaler is None:
            # methods like majority and matrix are forced to avoid scaling 
            if self.param.getVal('model') in non_scale_list:   
                return True, X
            else:
                return False, 'Inconsistency error. Scaling method defined but no Scaler loaded'
        
        return True, self.scaler.transform(X)

    def run_internal(self): 
        ''' 

        Runs prediction tasks using internally defined methods

        Most of these methods can be found at the stats folder

        '''

        # expand with new methods here:
        registered_methods = [('RF', RF),
                              ('SVM', SVM),
                              ('GNB', GNB),
                              ('PLSR', PLSR),
                              ('PLSDA', PLSDA),
                              ('median', median),
                              ('mean', mean),
                              ('majority', majority),
                              ('logicalOR', logicalOR),
                              ('matrix', matrix)]

        if self.param.getVal('model') == 'XGBOOST':
            from flame.stats.XGboost import XGBOOST
            registered_methods.append( ('XGBOOST', XGBOOST))

        # assume X matrix is present in 'xmatrix'
        X = self.conveyor.getVal("xmatrix")

        # use in single mol predictions
        if X.ndim < 2:  # if flat array
            X = X.reshape(1, -1)  # to 1 row matrix

        # retrieve data and dimensions from results
        nobj, nvarx = np.shape(X)

        # check that the dimensions of the X matrix are acceptable
        if (nobj == 0):
            LOG.error('No object found')
            self.conveyor.setError('No object found')
            return

        if (nvarx == 0):
            LOG.error('Failed to generate MDs')
            self.conveyor.setError('Failed to generate MDs')
            return
            
        # Load scaler and variable mask and preprocess the data
        success, result = self.preprocess(X)
        if not success:
            self.conveyor.setError(result)
            return          

        X = result

        # instantiate an appropriate child of base_model
        model = None
        for imethod in registered_methods:
            if imethod[0] == self.param.getVal('model'):

                # we instantiate the subtype of base_model, 
                # passing 
                # - model parameters (param) 
                # - already obtained results (conveyor)

                model = imethod[1](None, None, self.param, self.conveyor)
                LOG.debug(f'Recognized learner: {self.param.getVal("model")}')
                break

        if not model:
            self.conveyor.setError('modeling method not recognized')
            LOG.error(f'Modeling method {self.param.getVal("model")} '
                      'not recognized')
            return
        
        if self.conveyor.getError():
            return

        # try to load model previously built
        LOG.debug(f'Loading model from pickle file')
        success, results = model.load_model()
        if not success:
            self.conveyor.setError(f'Failed to load model estimator, with error "{result}"')
            return 

        # project the X matrix into the model and save predictions in self.conveyor
        model.project(X)
        
        # if this prediction is only generating input for an ensemble model skip validation
        # and projection on the chemical space
        if 'ghost' in self.param.getVal('output_format'):
            return

        # if the input file contains activity values use them to run external validation 
        if self.conveyor.isKey('ymatrix'):
            model.external_validation()

        # project on the chemical space built using the training series    
        # projectPredictions (X, self.param, self.conveyor)
        projectManifoldPredictions (X,self.param,self.conveyor)
        # projecttsnePredictions (X,self.param,self.conveyor)
        # projectIsomapPredictions (X,self.param,self.conveyor)


        return

    def run_R(self):
        ''' Runs prediction tasks using an importer KNIME workflow '''
        self.conveyor.setError('R toolkit is not supported in this version')
        return

    def run_KNIME(self):
        ''' Runs prediction tasks using R code '''
        self.conveyor.setError('KNIME toolkit is not supported in this version')
        return

    def run_custom(self):
        ''' Template to be overriden in apply_child.py

            Input: must be already present in self.results
            Output: add prediction results to self.results using the utils.add_result() method 

        '''
        self.conveyor.setError('custom prediction must be defined in the model apply_chlid class')
        return

    def run(self):
        ''' 

        Runs prediction tasks using the information present in self.results. 

        Depending on the modelingToolkit defined in self.param this task will use internal methods
        or make use if imported code in R/KNIME

        The custom option allows advanced uses to write their own function 'run_custom' method in 
        the model apply_child.py

        '''

        toolkit = self.param.getVal('modelingToolkit')

        if toolkit == 'internal':
            self.run_internal()
        elif toolkit == 'R':
            self.run_R()
        elif toolkit == 'KNIME':
            self.run_KNIME()
        elif toolkit == 'custom':
            self.run_custom()
        else:
            self.conveyor.setError('Unknown prediction toolkit to run ')
        return 
