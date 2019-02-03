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
import sys
import importlib

from flame.util import utils, get_logger
from flame.parameters import Parameters
from flame.conveyor import Conveyor

LOG = get_logger(__name__)

class Build:

    def __init__(self, model, param_file=None, output_format=None):
        LOG.debug('Starting build...')
        self.model = model
        self.param = Parameters()
        self.conveyor = Conveyor()

        # load parameters
        if param_file is not None:
            # use the param_file to update existing parameters at the model
            # directory and save changes to make them persistent
            success = self.param.delta(model, 0, param_file)
        else:
            # load parameter file at the model directory
            success = self.param.loadYaml(model, 0)

        # being unable to load parameters is a critical error
        if not success:
            LOG.critical('Unable to load model parameters. Aborting...')
            sys.exit(1)

        # add additional output formats included in the constructor 
        # this is requiered to add JSON format as output when the object is
        # instantiated from a web service call, requiring this output   
        if output_format is not None:
            if output_format not in self.param.getVal('output_format'):
                self.param.appVal('output_format',output_format)
 

    def get_model_set(self):
        ''' Returns a Boolean indicating if the model uses external input
            sources and a list with these sources '''
        return self.param.getModelSet()

    def set_single_CPU(self) -> None:
        ''' Forces the use of a single CPU '''
        LOG.debug('parameter "numCPUs" forced to be 1')
        self.param.setVal('numCPUs',1)

    def run(self, input_source):
        ''' Executes a default predicton workflow '''

        # path to endpoint
        epd = utils.model_path(self.model, 0)
        if not os.path.isdir(epd):
            self.conveyor.setError(f'Unable to find model {self.model}')
            #LOG.error(f'Unable to find model {self.model}')

        # import ichild classes
        if not self.conveyor.getError():
            # uses the child classes within the 'model' folder,
            # to allow customization of  the processing applied to each model
            modpath = utils.module_path(self.model, 0)

            idata_child = importlib.import_module(modpath+".idata_child")
            learn_child = importlib.import_module(modpath+".learn_child")
            odata_child = importlib.import_module(modpath+".odata_child")

            LOG.debug('child modules imported: '
                      f' {idata_child.__name__},'
                      f' {learn_child.__name__},'
                      f' {odata_child.__name__}')

            # instantiate idata (gets data for modeling) and run it
            idata = idata_child.IdataChild(self.param, self.conveyor, input_source)
            results = idata.run() 
            LOG.debug(f'idata child {idata_child.__name__} completed `run()`')

        # check there is a suitable X and Y
        if not self.conveyor.getError():
            if 'xmatrix' not in results:
                self.conveyor.addError(f'Failed to compute MDs')

            if 'ymatrix' not in results:
                self.conveyor.addError(f'No activity data (Y) found in training series')

        # instantiate lear (build a model from idata) and run it
        if not self.conveyor.getError():
            learn = learn_child.LearnChild(self.param, results)
            results = learn.run()
            LOG.debug(f'learn child {learn_child.__name__} completed `run()`')

        # run odata object, in charge of formatting the prediction results
        # note that if any of the above steps failed, an error has been inserted in the
        # conveyor and odata will take case of showing an error message
        odata = odata_child.OdataChild(self.param, results)
        LOG.info('Building completed')

        return odata.run()