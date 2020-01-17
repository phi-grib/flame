#! -*- coding: utf-8 -*-

# Description    Flame Learn class
#
# Authors: Manuel Pastor (manuel.pastor@upf.edu),
#          Jose Carlos Gómez (josecarlos.gomez@upf.edu)
#
# Copyright 2018 Manuel Pastor
#
# This file is part of Flame
#
# Flame is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation version 3.
#
# Flame is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Flame.  If not, see <http://www.gnu.org/licenses/>.

import os
import pickle
import numpy as np

from flame.stats.RF import RF
from flame.stats.SVM import SVM
from flame.stats.GNB import GNB
from flame.stats.PLSR import PLSR
from flame.stats.PLSDA import PLSDA
from flame.stats.XGboost import XGBOOST
from flame.stats.combo import median, mean, majority, matrix
from sklearn.preprocessing import MinMaxScaler 
from sklearn.preprocessing import StandardScaler 
from sklearn.preprocessing import RobustScaler


from flame.stats.imbalance import *  
from flame.stats import feature_selection
from flame.util import utils, get_logger
LOG = get_logger(__name__)


class Learn:

    def __init__(self, parameters, conveyor):

        self.param = parameters
        self.conveyor = conveyor
        self.conveyor.setOrigin('learn')

        self.X = self.conveyor.getVal('xmatrix')
        self.Y = self.conveyor.getVal('ymatrix')

        # Preprocessing variables
        self.scaler = None
        self.variable_mask = None

        # expand with new methods here:
        self.registered_methods = [('RF', RF),
                              ('XGBOOST', XGBOOST),
                              ('SVM', SVM),
                              ('GNB', GNB),
                              ('PLSR', PLSR),
                              ('PLSDA', PLSDA), 
                              ('median', median),
                              ('mean', mean),
                              ('majority', majority),
                              ('matrix', matrix)]

    def run_custom(self):
        '''
        Build a model using custom code to be defined in the learn child
        classes.
        '''

        self.conveyor.setError ('Not implemented')


    def preprocess(self):
        ''' 
        This function scales the X matrix and selects features 
        The scaler and the variable mask are saved in a pickl file 
        '''

        # Perform subsampling on the majority class. Consider to move.
        # Only for qualitative endpoints.
        if self.param.getVal("imbalance") is not None and \
                        not self.param.getVal("quantitative"):
            try:
                self.X, self.Y = run_imbalance(
                    self.param.getVal('imbalance'), self.X, self.Y, 46)
                # This is necessary to avoid inconsistences in methods
                # using self.nobj
                LOG.info(f'{self.param.getVal("imbalance")}'
                            f' performed')
                LOG.info(f'Number of objects after sampling: {self.X.shape[0]}')
            except Exception as e:
                return False, (f'Unable to perform sampling'
                               f' method with exception: {e}')

        # Run scaling.
        self.scaler = None

        # update if other fingerprints are added
        isFingerprint = (self.param.getVal('computeMD_method') == ['morganFP'])

        if self.param.getVal('modelAutoscaling') and \
                        not isFingerprint:
            try:
                scaler = None
                if self.param.getVal('modelAutoscaling') == 'StandardScaler':
                    scaler = StandardScaler()
                    LOG.info('Data scaled using StandarScaler')

                elif self.param.getVal('modelAutoscaling') == 'MinMaxScaler':
                    scaler = MinMaxScaler(copy=True, feature_range=(0,1))
                    LOG.info('Data scaled using MinMaxScaler')

                elif self.param.getVal('modelAutoscaling') == 'RobustScaler':
                    scaler = RobustScaler()
                    LOG.info('Data scaled using RobustScaler')

                else:
                    return False, 'Scaler not recognized'

                if scaler is not None:
                    # The scaler is saved so it can be used later
                    # to prediction instances.
                    self.scaler = scaler.fit(self.X)

                    # Scale the data.
                    self.X = scaler.transform(self.X)

            except Exception as e:
                return False, f'Unable to perform scaling with exception: {e}'
          
        # Run feature selection. Move to a instance method.
        if self.param.getVal("feature_selection"):
            # TODO: implement feature selection with other scalers
            self.variable_mask, self.scaler = \
                                feature_selection.run_feature_selection(
                                            self.X, self.Y, self.scaler,
                                            self.param)
            self.X = self.X[:, self.variable_mask]

        # Set the new number of instances/variables
        # if sampling/feature selection performed
        self.nobj, self.nvarx = np.shape(self.X)

        # Check X and Y integrity.
        if (self.nobj == 0) or (self.nvarx == 0):
            return False, 'No objects/variables in the matrix'

        if len(self.Y) == 0:
            self.failed = True
            return False, 'No activity values'

        # This dictionary contain all the objects which will be needed
        # for prediction
        prepro = {'scaler':self.scaler,\
                  'variable_mask':self.variable_mask,\
                  'version':1}

        prepro_pkl_path = os.path.join(self.param.getVal('model_path'),
                                      'preprocessing.pkl')
        
        with open(prepro_pkl_path, 'wb') as handle:
            pickle.dump(prepro, handle, 
                        protocol=pickle.HIGHEST_PROTOCOL)

        LOG.debug('Model saved as:{}'.format(prepro_pkl_path))
        return True, 'OK'


    def run_internal(self):
        '''
        Builds a model using the internally defined machine learning tools.

        All input parameters are extracted from self.param.

        The main output is an instance of basemodel saved in
        the model folder as a pickle (model.pkl) and used for prediction.

        The results of building and validation are added to results,
        but also saved to the model folder as a pickle (info.pkl)
        for being displayed in manage tools.
        '''

        # check suitability of Y matrix
        if not self.param.getVal('quantitative') :
            success, yresult  = utils.qualitative_Y(self.Y)
            if not success:
                self.conveyor.setError(yresult)
                return

        # pre-process data
        success, message = self.preprocess()
        if not success:
            self.conveyor.setError(message)
            return



        # instantiate an appropriate child of base_model
        model = None
        for imethod in self.registered_methods:
            if imethod[0] == self.param.getVal('model'):

                # we instantiate the subtype of base_model, 
                # passing 
                # - preteated X and Y matrices for model building
                # - model parameters (param) 
                # - already obtained results (conveyor)

                model = imethod[1](self.X, self.Y, self.param, self.conveyor)
                LOG.debug('Recognized learner: '
                          f"{self.param.getVal('model')}")
                break

        if not model:
            self.conveyor.setError(f'Modeling method {self.param.getVal("model")}'
                                    'not recognized')
            LOG.error(f'Modeling method {self.param.getVal("model")}'
                       'not recognized')
            return
            
        if self.conveyor.getError():
            return

        # build model
        LOG.info('Starting model building')
        success, model_building_results = model.build()
        if not success:
            self.conveyor.setError(model_building_results)
            return

        self.conveyor.addVal(
                    model_building_results,
                    'model_build_info',
                    'model building information',
                    'method',
                    'single',
                    'Information about the model')

        # validate model
        LOG.info('Starting model validation')
        success, model_validation_results = model.validate()
        if not success:
            self.conveyor.setError(model_validation_results)
            return

        # model_validation_results is a dictionary which contains model_validation_info and 
        # (optionally) Y_adj and Y_pred, depending on the model type    
        
        self.conveyor.addVal(
            model_validation_results['quality'],
            'model_valid_info',
            'model validation information',
            'method',
            'single',
            'Information about the model validation')

        # non-conformal qualitative and quantitative models
        if 'Y_adj' in model_validation_results:
            self.conveyor.addVal(
                model_validation_results['Y_adj'],
                'Y_adj',
                'Y fitted',
                'result',
                'objs',
                'Y values of the training series fitted by the model')
        
        if 'Y_pred' in model_validation_results:
            self.conveyor.addVal(
                model_validation_results['Y_pred'],
                'Y_pred',
                'Y predicted',
                'result',
                'objs',
                'Y values of the training series predicted by the model')

        # conformal qualitative models produce a list of tuples, indicating
        # if the object is predicted to belong to class 0 and 1
        if 'classes' in model_validation_results:
            for i in range(len(model_validation_results['classes'][0])):
                class_key = 'c' + str(i)
                class_label = 'Class ' + str(i)
                class_list = model_validation_results['classes'][:, i].tolist()
                self.conveyor.addVal( class_list, 
                                class_key, class_label,
                                'result', 'objs', 
                                'Conformal class assignment',
                                'main')

        # conformal quantitataive models produce a list of tuples, indicating
        # the minumum and maximum value

        # TODO: compute AD (when applicable)

        LOG.info('Model finished successfully')

        # save model
        try:
            model.save_model()

        except Exception as e:
            LOG.error(f'Error saving model with exception {e}')
            return False, 'An error ocurred saving the model'

        return

    def run(self):
        '''
        Builds the model using the appropriate toolkit (internal or custom).
        '''

        toolkit = self.param.getVal('modelingToolkit')

        if toolkit == 'internal':
            LOG.info('Building model using internal toolkit : Sci-kit learn')
            self.run_internal()

        elif toolkit == 'custom':
            LOG.info('Building model using custom toolkit')
            self.run_custom()
        else:
            LOG.error("Modeling toolkit is not yet supported")
            self.conveyor.setError( 'modeling Toolkit ' + \
                toolkit+' is not supported yet')

        return 
