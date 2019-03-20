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

from sklearn.metrics import mean_squared_error, matthews_corrcoef as mcc
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer
from sklearn.metrics import confusion_matrix
from flame.util import utils, get_logger
LOG = get_logger(__name__)


class Apply:

    def __init__(self, parameters, conveyor):

        self.param = parameters
        self.conveyor = conveyor
        self.conveyor.setOrigin('apply')


    def external_validation(self):
        ''' when experimental values are available for the predicted compounds,
        run external validation '''

        ext_val_results = []
        
        # Ye are the y values present in the input file
        Ye = np.asarray(self.conveyor.getVal("ymatrix"))

        # For qualitative models, make sure the Y is qualitative as well
        if not self.param.getVal("quantitative"):
            qy, message = utils.qualitative_Y(Ye)
            if not qy:
                self.conveyor.setWarning(f'No qualitative activity suitable for external validation "{message}". Skipping.')
                LOG.warning(f'No qualitative activity suitable for external validation "{message}". Skipping.')
                return

        # there are four variants of external validation, depending if the method
        # if conformal or non-conformal and the model is qualitative and quantitative

        if not self.param.getVal("conformal"):

            # non-conformal 
            if not self.param.getVal("quantitative"):
                
                # non-conformal & qualitative
                Yp = np.asarray(self.conveyor.getVal("values"))

                if Ye.size == 0:
                    raise ValueError("Experimental activity vector is empty")
                if Yp.size == 0:
                    raise ValueError("Predicted activity vector is empty")

                # the use of labels is compulsory to inform the confusion matrix that
                # it must return a 2x2 confussion matrix. Otherwise it will fail when
                # a single class is represented (all TP, for example)
                TN, FP, FN, TP = confusion_matrix(
                    Ye, Yp, labels=[0, 1]).ravel()

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

                ext_val_results.append(('TP_ex',
                                        'True positives in external-validation',
                                        float(TP)))
                ext_val_results.append(('TN_ex',
                                        'True negatives in external-validation',
                                        float(TN)))
                ext_val_results.append(('FP_ex',
                                        'False positives in external-validation',
                                        float(FP)))
                ext_val_results.append(('FN_ex',
                                        'False negatives in external-validation',
                                        float(FN)))
                ext_val_results.append(('Sensitivity_ex',
                                        'Sensitivity in external-validation',
                                        float(sensitivity)))
                ext_val_results.append(('Specificity_ex',
                                        'Specificity in external-validation',
                                        float(specificity)))
                ext_val_results.append(('MCC_ex',
                                        'Mattews Correlation Coefficient in external-validation',
                                        float(MCC)))

            else:

                # non-conformal & quantitative
                Yp = np.asarray(self.conveyor.getVal("values"))

                if Ye.size == 0:
                    raise ValueError("Experimental activity vector is empty")
                if Yp.size == 0:
                    raise ValueError("Predicted activity vector is empty")

                Ym = np.mean(Ye)
                nobj = len(Yp)

                SSY0_out = np.sum(np.square(Ym - Ye))
                SSY_out = np.sum(np.square(Ye - Yp))
                scoringP = mean_squared_error(Ye, Yp)
                SDEP = np.sqrt(SSY_out / (nobj))
                Q2 = 1.00 - (SSY_out / SSY0_out)

                ext_val_results.append(
                    ('scoringP_ex', 'Scoring P', scoringP))
                ext_val_results.append(
                    ('Q2_ex', 'Determination coefficient in cross-validation', Q2))
                ext_val_results.append(
                    ('SDEP_ex', 'Standard Deviation Error of the Predictions', SDEP))

            self.conveyor.addVal(
                             ext_val_results,
                             'external-validation',
                             'external validation',
                             'method',
                             'single',
                             'External validation results')

        else:
            # conformal external validation

            if not self.param.getVal("quantitative"):
                
                # conformal & qualitative
                Yp = np.concatenate((np.asarray(self.conveyor.getVal('c0')).reshape(
                    -1, 1), np.asarray(self.conveyor.getVal('c1')).reshape(-1, 1)), axis=1)

                if Ye.size == 0:
                    raise ValueError("Experimental activity vector is empty")
                if Yp.size == 0:
                    raise ValueError("Predicted activity vector is empty")

                c0_correct = 0
                c1_correct = 0
                not_predicted = 0
                c0_incorrect = 0
                c1_incorrect = 0

                Ye1 = []
                Yp1 = []
                for i in range(len(Ye)):
                    real = float(Ye[i])
                    predicted = Yp[i]
                    if predicted[0] != predicted[1]:
                        Ye1.append(real)
                        if predicted[0]:
                            Yp1.append(0)
                        else:
                            Yp1.append(1)

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
                MCC = mcc(Ye1, Yp1)
                TN = c0_correct
                FP = c0_incorrect
                TP = c1_correct
                FN = c1_incorrect
                coverage = float((len(Yp) - not_predicted) / len(Yp))

                if (TP+FN) > 0:
                    sensitivity = (TP / (TP + FN))
                else:
                    sensitivity = 0.0
                if (TN+FP) > 0:
                    specificity = (TN / (TN + FP))
                else:
                    specificity = 0.0
                ext_val_results.append(('TP',
                                        'True positives in external-validation',
                                        float(TP)))
                ext_val_results.append(('TN',
                                        'True negatives in external-validation',
                                        float(TN)))
                ext_val_results.append(('FP',
                                        'False positives in external-validation',
                                        float(FP)))
                ext_val_results.append(('FN',
                                        'False negatives in external-validation',
                                        float(FN)))
                ext_val_results.append(('Sensitivity',
                                        'Sensitivity in external-validation',
                                        float(sensitivity)))
                ext_val_results.append(('Specificity',
                                        'Specificity in external-validation',
                                        float(specificity)))
                ext_val_results.append(('MCC',
                                        'Mattews Correlation Coefficient in external-validation',
                                        float(MCC)))
                ext_val_results.append(('Conformal_coverage',
                                        'Conformal coverage in external-validation',
                                        float(coverage)))
                self.conveyor.addVal(
                                 ext_val_results,
                                 'external-validation',
                                 'external validation',
                                 'method',
                                 'single',
                                 'External validation results')
            else:

                # conformal & quantitative
                Yp_lower = self.conveyor.getVal('lower_limit')
                Yp_upper = self.conveyor.getVal('upper_limit')

                mean_interval = np.mean(np.abs(Yp_lower) - np.abs(Yp_upper))
                inside_interval = (Yp_lower.reshape(-1, 1) <
                                   Ye) & (Yp_upper.reshape(-1, 1) > Ye)
                accuracy = len(inside_interval)/len(Ye)
                conformal_accuracy = float("{0:.2f}".format(accuracy))
                conformal_mean_interval = float(
                    "{0:.2f}".format(mean_interval))

                ext_val_results.append(('Conformal_mean_interval',
                                        'Conformal mean interval',
                                        conformal_mean_interval))
                ext_val_results.append(('Conformal_accuracy',
                                        'Conformal accuracy',
                                        conformal_accuracy))

                self.conveyor.addVal(
                                 ext_val_results,
                                 'external-validation',
                                 'external validation',
                                 'method',
                                 'single',
                                 'External validation results')

    def load_prepro(self):
        ''' This function loads estimator and scaler in a pickle file '''

        prepro_file = os.path.join(self.param.getVal('model_path'),
                                    'preprocessing.pkl')
        LOG.debug(f'Loading model from pickle file, path: {prepro_file}')
        try:
            with open(prepro_file, "rb") as input_file:
                dict_prepro = pickle.load(input_file)
        except FileNotFoundError:
            LOG.error(f'No valid preprocessing tools'
                     f'found at: {prepro_file}')
            raise FileNotFoundError

        # Load model
        self.version = dict_prepro['version']
        # check if the pickle was created with a compatible version
        # currently 1
        if self.version is not 1:
            raise Exception ('Incompatible preprocessing version')        
    
        # Load rest of info in an extensible way
        # This allows to add new variables keeping
        # Retro-compatibility
        if 'scaler' in dict_prepro.keys():
            self.scaler = dict_prepro['scaler']

        if 'variable_mask' in dict_prepro.keys():
            self.variable_mask = dict_prepro['variable_mask']

        # Check consistency between parameter file and pickle info
        if self.param.getVal('modelAutoscaling') and \
                            self.scaler is None:
            raise Exception('Inconsistency error. Autoscaling is True'
                            ' in parameter file but no Scaler loaded')

        if self.param.getVal('feature_selection') and \
                            self.variable_mask is None:
            raise Exception('Inconsistency error. Feature is True'
                        ' in parameter file but no variable mask loaded')

        return
     
    def preprocess(self, X):
        self.load_prepro()
        if self.param.getVal("feature_selection"):
            X = X[:, self.variable_mask]
        if self.param.getVal('modelAutoscaling'):
            X = self.scaler.transform(X)
        return True, X

    def run_internal(self): 
        ''' 

        Runs prediction tasks using internally defined methods

        Most of these methods can be found at the stats folder

        '''

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
        X = self.preprocess(X)[1]


        # TODO: Load scaler and variable mask and preprocess the data

        # preprocess
        # success, message = self.preprocess()
        # if not success:
        #     self.conveyor.setError(message)
        #     return

        # Load model 

        # expand with new methods here:
        registered_methods = [('RF', RF),
                              ('SVM', SVM),
                              ('GNB', GNB),
                              ('PLSR', PLSR),
                              ('PLSDA', PLSDA), ]

        # instantiate an appropriate child of base_model
        model = None
        for imethod in registered_methods:
            if imethod[0] == self.param.getVal('model'):
                model = imethod[1](None, None, self.param)
                LOG.debug('Recognized learner: '
                          f"{self.param.getVal('model')}")
                break

        if not model:
            self.conveyor.setError('modeling method not recognized')
            LOG.error(f'Modeling method {self.param.getVal("model")}'
                      'not recognized')
            return
        try:
            model.load_model()
            LOG.debug(f'Loading model from pickle file')
        except Exception as e:
            #LOG.error(f'No valid model estimator found with exception "{e}"')
            self.conveyor.setError(f'No valid model estimator found with exception "{e}"')
            return False, f'Exception ocurred when loading model: {e}'

        # project the X matrix into the model and save predictions in self.conveyor
        model.project(X, self.conveyor)

        # if the input file contains activity values use them to run external validation 
        if self.conveyor.isKey('ymatrix'):
            self.external_validation()

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
