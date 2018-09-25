#! -*- coding: utf-8 -*-

# Description    Flame Learn class
##
# Authors:       Manuel Pastor (manuel.pastor@upf.edu), Jose Carlos Gómez (josecarlos.gomez@upf.edu)
##
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

import os
import pickle
import numpy as np

from flame.stats.RF import RF
from flame.stats.SVM import SVM
from flame.stats.GNB import GNB
from flame.stats.PLSR import PLSR
from flame.stats.PLSDA import PLSDA


class Learn:

    def __init__(self, parameters, results):

        self.parameters = parameters

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

        All input parameters are extracted from self.parameters.

        The main output is an instance of basemodel saved in
        the model folder as a pickle (model.pkl) and used for prediction.

        The results of building and validation are added to results,
        but also saved to the model folder as a pickle (info.pkl)
        for being displayed in manage tools.
        '''

        registered_methods = [('RF', RF),
                              ('SVM', SVM),
                              ('GNB', GNB),
                              ('PLSR', PLSR),
                              ('PLSDA', PLSDA), ]  # expand with new methods here

        # instanciate an appropriate child of base_model
        model = None
        for imethod in registered_methods:
            if imethod[0] == self.parameters['model']:
                model = imethod[1](self.X, self.Y, self.parameters)
                break

        if not model:
            self.results['error'] = 'modeling method not recognised'
            return

        # build model
        success, results = model.build()
        if not results:
            self.results['error'] = results
            return
        self.results['model_build'] = results

        # validate model
        success, results = model.validate()
        if not success:
            self.results['error'] = results
            return
        self.results['model_validate'] = results

        # TODO: compute AD (when applicable)

        # save model
        with open(os.path.join(self.parameters['model_path'], 'model.pkl'), 'wb') as handle:
            pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # save model info for informative purposes
        with open(os.path.join(self.parameters['model_path'], 'info.pkl'), 'wb') as handle:
            pickle.dump(self.results['model_build'], handle)
            pickle.dump(self.results['model_validate'], handle)

        return

    def run(self):
        '''
        Builds the model using the appropriate toolkit (internal or custom).
        '''

        toolkit = self.parameters['modelingToolkit']

        if toolkit == 'internal':
            self.run_internal()
        elif toolkit == 'custom':
            self.run_custom()
        else:
            self.results['error'] = 'modeling Toolkit ' + \
                toolkit+' is not supported yet'

        return self.results
