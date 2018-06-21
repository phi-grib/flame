#! -*- coding: utf-8 -*-

# Description    Flame Build class
##
# Authors:       Manuel Pastor (manuel.pastor@upf.edu)
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
# along with Flame. If not, see <http://www.gnu.org/licenses/>.

import os
import importlib

from flame.util import utils
from flame.control import Control


class Build:

    def __init__(self, model, output_format=None):

        self.model = model

        # instance Control object
        self.control = Control(self.model, 0)
        self.parameters = self.control.get_parameters()

        # set parameter overriding value in
        if output_format is not None:
            self.parameters['output_format'] = output_format

        return

    def get_model_set(self):
        '''
        Returns a Boolean indicating if the model uses external
        input sources and a list with these sources 
        '''
        return self.control.get_model_set()

    def set_single_CPU(self):
        ''' Forces the use of a single CPU '''
        self.parameters['numCPUs'] = 1

    def run(self, input_source):
        ''' Executes a default predicton workflow '''

        results = {}

        # path to endpoint
        epd = utils.model_path(self.model, 0)
        if not os.path.isdir(epd):
            results['error'] = 'unable to find model: '+self.model

        if 'error' not in results:
            # uses the child classes within the 'model' folder, to allow customization of
            # the processing applied to each model
            modpath = utils.module_path(self.model, 0)

            idata_child = importlib.import_module(modpath+".idata_child")
            learn_child = importlib.import_module(modpath+".learn_child")
            odata_child = importlib.import_module(modpath+".odata_child")

            # run idata object, in charge of generate model data from local copy of input
            idata = idata_child.IdataChild(self.parameters, input_source)
            results = idata.run()

        if 'error' not in results:
            # run learn object, in charge of generate a prediction from idata
            learn = learn_child.LearnChild(self.parameters, results)
            results = learn.run()

        # run odata object, in charge of formatting the prediction results
        odata = odata_child.OdataChild(self.parameters, results)
        return odata.run()
