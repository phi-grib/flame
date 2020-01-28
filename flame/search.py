#! -*- coding: utf-8 -*-

# Description    Flame Predict class
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
import yaml
import importlib

from flame.util import utils, get_logger
from flame.parameters import Parameters
from flame.conveyor import Conveyor
from flame.idata import Idata
from flame.sapply import Sapply
from flame.odata import Odata

LOG = get_logger(__name__)

class Search:

    def __init__(self, space, version, output_format=None, label=None):
        LOG.debug('Starting predict...')
        self.space = space
        self.version = version
        self.label = label
        self.param = Parameters()
        self.conveyor = Conveyor()

        self.conveyor.addVal(label, 'prediction_label', 'prediction label',
            'method', 'single',
            'Label used to identify the prediction')

        if not self.param.loadYaml(space, version, isSpace=True):
            LOG.critical('Unable to load space parameters. Aborting...')
            sys.exit()

        # add additional output formats included in the constructor 
        # this is requiered to add JSON format as output when the object is
        # instantiated from a web service call, requiring this output   
        if output_format != None:
            if output_format not in self.param.getVal('output_format'):
                self.param.appVal('output_format',output_format)
 
        return

    def set_single_CPU(self) -> None:
        ''' Forces the use of a single CPU '''
        LOG.debug('parameter "numCPUs" forced to be 1')
        self.param.setVal('numCPUs',1)

    def run(self, input_source, runtime_param=None, metric=None, numsel=None, cutoff=None):
        ''' Executes a default predicton workflow '''

        # path to endpoint
        epd = utils.space_path(self.space, self.version)
        if not os.path.isdir(epd):
            self.conveyor.setError(f'Unable to find space {self.space}, version {self.version}')
            #LOG.error(f'Unable to find space {self.space}')

        print (runtime_param)
        if runtime_param is not None:
            try:
                with open(runtime_param, 'r') as pfile:
                    rtparam = yaml.safe_load(pfile)
                    try:
                        cutoff = rtparam['similarity_cutoff_distance']
                        numsel = rtparam['similarity_cutoff_num']
                        metric = rtparam['similarity_metric']
                    except:
                        LOG.error('wrong format in the runtime similarity parameters')
                        self.conveyor.setError('wrong format in the runtime similarity parameters')
            except:
                LOG.error('runtime similarity parameter file not found')
                self.conveyor.setError('runtime similarity parameter file not found')

        if not self.conveyor.getError():
            # uses the child classes within the 'space' folder,
            # to allow customization of
            # the processing applied to each space
            modpath = utils.smodule_path(self.space, self.version)

            idata_child = importlib.import_module(modpath+".idata_child")
            sapply_child = importlib.import_module(modpath+".sapply_child")
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
            # make sure there is X data
            if not self.conveyor.isKey('xmatrix'):
                LOG.debug(f'Failed to compute MDs')
                self.conveyor.setError(f'Failed to compute MDs')

        if not self.conveyor.getError():
            # run apply object, in charge of generate a prediction from idata
            try:
                sapply = sapply_child.SapplyChild(self.param, self.conveyor)
            except:
                LOG.warning ('Sapply child architecture mismatch, defaulting to Sapply parent')
                sapply = Sapply(self.param, self.conveyor)

            sapply.run(cutoff, numsel, metric)
            LOG.debug(f'sapply child {type(sapply).__name__} completed `run()`')

        # run odata object, in charge of formatting the prediction results
        # note that if any of the above steps failed, an error has been inserted in the
        # conveyor and odata will take case of showing an error message
        try:
            odata = odata_child.OdataChild(self.param, self.conveyor, self.label)
        except:
            LOG.warning ('Odata child architecture mismatch, defaulting to Odata parent')
            odata = Odata(self.param, self.conveyor, self.label)

        return odata.run()
