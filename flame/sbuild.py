#! -*- coding: utf-8 -*-

# Description    Flame Build class
#
# Authors:       Manuel Pastor (manuel.pastor@upf.edu)
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
# along with Flame. If not, see <http://www.gnu.org/licenses/>.

import os
import sys
import importlib

from flame.util import utils, get_logger
from flame.parameters import Parameters
from flame.conveyor import Conveyor
from flame.idata import Idata
from flame.slearn import Slearn
from flame.odata import Odata

LOG = get_logger(__name__)

class Sbuild:

    def __init__(self, space, param_file=None, param_string=None, output_format=None):
        LOG.debug('Starting sbuild...')
        self.space = space
        self.param = Parameters()
        self.conveyor = Conveyor()
        self.conveyor.setOrigin('slearn')

        # load parameters
        if param_file is not None:
            # use the param_file to update existing parameters at the space
            # directory and save changes to make them persistent
            success, message = self.param.delta(space, 0, param_file, iformat='YAML', isSpace=True)

        elif param_string is not None:
            success, message = self.param.delta(space, 0, param_string, iformat='JSONS', isSpace=True)

        else:
            # load parameter file at the space directory
            success, message = self.param.loadYaml(space, 0, isSpace=True)

        # being unable to load parameters is a critical error
        if not success:
            LOG.critical(f'Unable to load space parameters. {message}. Aborting...')
            sys.exit(1)

        # add additional output formats included in the constructor 
        # this is requiered to add JSON format as output when the object is
        # instantiated from a web service call, requiring this output   
        if output_format is not None:
            if output_format not in self.param.getVal('output_format'):
                self.param.appVal('output_format',output_format)
 
    def set_single_CPU(self) -> None:
        ''' Forces the use of a single CPU '''
        LOG.debug('parameter "numCPUs" forced to be 1')
        self.param.setVal('numCPUs',1)

    def run(self, input_source):
        ''' Executes a default chemical space building workflow '''

        # path to endpoint
        epd = utils.space_path(self.space, 0)
        if not os.path.isdir(epd):
            self.conveyor.setError(f'Unable to find space {self.space}')
            #LOG.error(f'Unable to find space {self.space}')

        # import ichild classes
        if not self.conveyor.getError():
            # uses the child classes within the 'space' folder,
            # to allow customization of  the processing applied to each space
            modpath = utils.smodule_path(self.space, 0)

            idata_child = importlib.import_module(modpath+".idata_child")
            slearn_child = importlib.import_module(modpath+".slearn_child")
            odata_child = importlib.import_module(modpath+".odata_child")

            # run idata object, in charge of generate space data from input
            try:
                idata = idata_child.IdataChild(self.param, self.conveyor, input_source)
            except:
                LOG.warning ('Idata child architecture mismatch, defaulting to Idata parent')
                idata = Idata(self.param, self.conveyor, input_source)
                
            idata.run() 
            LOG.debug(f'idata child {type(idata).__name__} completed `run()`')

        if not self.conveyor.getError():
            # check there is a suitable X and Y
            if not self.conveyor.isKey ('xmatrix'):
                self.conveyor.setError(f'Failed to compute MDs')

        if not self.conveyor.getError():
            # instantiate learn (build a space from idata) and run it
            try:
                slearn = slearn_child.SlearnChild(self.param, self.conveyor)
            except:
                LOG.warning ('Slearn child architecture mismatch, defaulting to Learn parent')
                slearn = Slearn(self.param, self.conveyor)

            slearn.run()
            LOG.debug(f'slearn child {type(slearn).__name__} completed `run()`')

        # run odata object, in charge of formatting the prediction results
        # note that if any of the above steps failed, an error has been inserted in the
        # conveyor and odata will take case of showing an error message
        try:
            odata = odata_child.OdataChild(self.param, self.conveyor)
        except:
            LOG.warning ('Odata child architecture mismatch, defaulting to Odata parent')
            odata = Odata(self.param, self.conveyor)
    
        return odata.run()