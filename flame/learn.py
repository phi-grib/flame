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

        # instanciate an appropriate child of base_model
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
            self.results['error'] = model_buidling_results
            return

        utils.add_result(self.results,
                    model_building_results,
                    'model_build_info',
                    'model buidling information',
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

        # model_validation_results is a tuple which contains model_validation_info and 
        # (optionally) Y_adj and Y_pred, depending on the model type    
        
        utils.add_result(self.results,
            model_validation_results[0],
            'model_valid_info',
            'model validation information',
            'method',
            'single',
            'Information about the model validation')

        if len(model_validation_results)>1:
            utils.add_result(self.results,
                model_validation_results[1],
                'Y_adj',
                'Y fitted',
                'result',
                'objs',
                'Y values of the training series fitted by the model')
        
        if len(model_validation_results)>2:
            utils.add_result(self.results,
                model_validation_results[2],
                'Y_pred',
                'Y predicted',
                'result',
                'objs',
                'Y values of the training series predicted by the model')

        # TODO: compute AD (when applicable)

        LOG.info('Model finished succesfully')

        # save model
        model_pkl_path = os.path.join(self.param.getVal('model_path'),
                                      'model.pkl')
        with open(model_pkl_path, 'wb') as handle:
            pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
        LOG.debug('Model saved as:{}'.format(model_pkl_path))

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
