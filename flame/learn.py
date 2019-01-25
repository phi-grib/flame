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
from flame.util import utils, get_logger
LOG = get_logger(__name__)


class Learn:

    def __init__(self, parameters, results):

        self.param = parameters

        self.X = results['xmatrix']
        self.Y = results['ymatrix']
        # TODO: make use of other results items

        self.results = results
        self.results['origin'] = 'learn'

    def run_custom(self):
        '''
        Build a model using custom code to be defined in the learn child
        classes.
        '''

        self.results['error'] = 'not implemented'
        return

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
                model = imethod[1](self.X, self.Y, self.param)
                LOG.debug('Recognized learner: '
                          f"{self.param.getVal('model')}")
                break

        if not model:
            self.results['error'] = 'modeling method not recognised'
            LOG.error(f'Modeling method {self.param.getVal("model")}'
                      'not recognized')
            return

        # build model
        LOG.info('Starting model building')
        success, model_building_results = model.build()
        if not success:
            self.results['error'] = model_building_results
            return

        utils.add_result(self.results,
                    model_building_results,
                    'model_build_info',
                    'model building information',
                    'method',
                    'single',
                    'Information about the model')
        # self.results['model_build'] = results

        # validate model
        LOG.info('Starting model validation')
        success, model_validation_results = model.validate()
        if not success:
            self.results['error'] = model_validation_results
            return

        # model_validation_results is a dictionary which contains model_validation_info and 
        # (optionally) Y_adj and Y_pred, depending on the model type    
        
        utils.add_result(self.results,
            model_validation_results['quality'],
            'model_valid_info',
            'model validation information',
            'method',
            'single',
            'Information about the model validation')

        # non-conformal qualitative and quantitative models
        if 'Y_adj' in model_validation_results:
            utils.add_result(self.results,
                model_validation_results['Y_adj'],
                'Y_adj',
                'Y fitted',
                'result',
                'objs',
                'Y values of the training series fitted by the model')
        
        if 'Y_pred' in model_validation_results:
            utils.add_result(self.results,
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
                utils.add_result(self.results, class_list, 
                                class_key, class_label,
                                'result', 'objs', 
                                'Conformal class assignment',
                                    'main')

        # conformal quantitataive models produce a list of tuples, indicating
        # the minumum and maximum value

        # if 'interval' in model_validation_results:
            # mean1 = np.mean(model_validation_results['classes'], axis=1)
            # lower_limit = model_validation_results['classes'][:, 0]
            # upper_limit = model_validation_results['classes'][:, 1]
            # utils.add_result(results, mean1, 'values', 'Prediction',
            #                  'result', 'objs',
            #                   'Results of the prediction', 'main')
            # utils.add_result(results, lower_limit, 'lower_limit',
            #                  'Lower limit', 'confidence', 'objs',
            #                   'Lower limit of the conformal prediction')
            # utils.add_result(results, upper_limit, 'upper_limit',
            #                  'Upper limit', 'confidence', 'objs',
            #                   'Upper limit of the conformal prediction')

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
            self.results['error'] = 'modeling Toolkit ' + \
                toolkit+' is not supported yet'

        return self.results
