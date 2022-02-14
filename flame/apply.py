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
# import pickle
# import yaml
# import os
import time

from flame.stats.RF import RF
from flame.stats.SVM import SVM
from flame.stats.GNB import GNB
from flame.stats.PLSR import PLSR
from flame.stats.PLSDA import PLSDA
from flame.stats.combo import median, mean, majority, logicalOR, matrix, external_model

from sklearn.metrics import mean_squared_error, matthews_corrcoef as mcc
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer
from sklearn.metrics import confusion_matrix
from flame.graph.graph import projectReduced

from flame.util import utils, get_logger
LOG = get_logger(__name__)


class Apply:

    def __init__(self, parameters, conveyor):

        self.param = parameters
        self.conveyor = conveyor

        # registered method should be defined in the constructor, so the overrided
        # childs can enrich the methods
        self.registered_methods = [('RF', RF),
                        ('SVM', SVM),
                        ('GNB', GNB),
                        ('PLSR', PLSR),
                        ('PLSDA', PLSDA),
                        ('median', median),
                        ('mean', mean),
                        ('majority', majority),
                        ('logicalOR', logicalOR),
                        ('matrix', matrix),
                        ('external_model', external_model)]


    # def cpreprocess (self, X):
    #     ''' preprocesing for confidential models'''

    #     path = utils.model_path(self.param.getVal('endpoint'), 0)
    #     cmeta = os.path.join(path, 'confidential_preprocess.yaml')
    #     with open(cmeta, 'r') as f:
    #         cmodel = yaml.safe_load (f)
        
    #     X = X.astype(float)
    #     X -= np.array(cmodel['xmean'])
        
    #     if self.param.getVal('modelAutoscaling') == 'StandardScaler':
    #         X *= np.array(cmodel['wg']) 

    #     return True, X

    # def preprocess(self, X):
    #     ''' This function loads the scaler and variable mask from a pickle file 
    #     and apply them to the X matrix passed as an argument'''

    #     prepro_file = os.path.join(self.param.getVal('model_path'), 'preprocessing.pkl')

    #     LOG.debug(f'Loading model from pickle file, path: {prepro_file}')
    #     try:
    #         with open(prepro_file, "rb") as input_file:
    #             dict_prepro = pickle.load(input_file)
    #     except FileNotFoundError:
    #         return False, f'No valid preprocessing tools found at: {prepro_file}'

    #     # Load version
    #     self.version = dict_prepro['version']

    #     # check if the pickle was created with a compatible version
    #     # currently 1
    #     if self.version is not 1:
    #         return False, 'Incompatible preprocessing version'   
    
    #     # Load rest of info in an extensible way
    #     # This allows to add new variables keeping
    #     # Retro-compatibility
    #     self.variable_mask = None
    #     if 'variable_mask' in dict_prepro.keys():
    #         self.variable_mask = dict_prepro['variable_mask']

    #     if self.param.getVal('feature_selection') and self.variable_mask is None:
    #         return False, 'Inconsistency error. Feature is True in parameter file but no variable mask loaded'

    #     # apply variable_mask
    #     if self.param.getVal("feature_selection"):
    #         X = X[:, self.variable_mask]

    #     if self.param.getVal('modelAutoscaling') is None:
    #         return True, X

    #     # Load rest of info in an extensible way
    #     # This allows to add new variables keeping
    #     # Retro-compatibility
    #     self.scaler = None
    #     if 'scaler' in dict_prepro.keys():
    #         self.scaler = dict_prepro['scaler']

    #     # Check consistency between parameter file and pickle info
    #     non_scale_list = ['majority','logicalOR','matrix']
    #     if self.scaler is None:
    #         # methods like majority and matrix are forced to avoid scaling 
    #         if self.param.getVal('model') in non_scale_list:   
    #             return True, X
    #         else:
    #             return False, 'Inconsistency error. Scaling method defined but no Scaler loaded'
        
    #     return True, self.scaler.transform(X)

    def run_internal(self): 
        ''' 

        Runs prediction tasks using internally defined methods

        Most of these methods can be found at the stats folder

        '''

        if self.param.getVal('model') == 'XGBOOST':
            from flame.stats.XGboost import XGBOOST
            self.registered_methods.append( ('XGBOOST', XGBOOST))

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
            
        # TODO: support confidential preprocessing
        # Load scaler and variable mask and preprocess the data
        # if self.param.getVal('confidential'):
        #     success, result = self.cpreprocess(X)
        #     if not success:
        #         self.conveyor.setError(result)
        #         return          
            
        #     X = result

        # instantiate an appropriate child of base_model
        model = None
        for imethod in self.registered_methods:
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

        if not self.param.getVal('confidential'):
            # try to load model previously built
            start = time.time()
            LOG.debug('Loading model from pickle file...')
            success, results = model.load_model()
            end = time.time()
            LOG.debug(f'Model loaded with message "{results}" in {(end-start):.2f} seconds')

            if not success:
                self.conveyor.setError(f'Failed to load model estimator, with error "{results}"')
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

        projectReduced(X,self.param,self.conveyor)

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
