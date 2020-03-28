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
import importlib

from flame.util import utils, get_logger
from flame.parameters import Parameters
from flame.conveyor import Conveyor
from flame.idata import Idata
from flame.apply import Apply
from flame.odata import Odata

LOG = get_logger(__name__)

class Predict:

    def __init__(self, model, version=0, output_format=None, label=None):
        LOG.debug('Starting predict...')
        self.model = model
        self.version = version
        self.param = Parameters()
        self.conveyor = Conveyor()

        self.conveyor.setOrigin('apply')

        self.conveyor.addVal(label, 'prediction_label', 'prediction label',
                    'method', 'single',
                    'Label used to identify the prediction')

        success, results = self.param.loadYaml(model, version)
        if not success:
            LOG.critical(f'Unable to load model parameters. {results}. Aborting...')
            sys.exit()

        # add additional output formats included in the constructor 
        # this is requiered to add JSON format as output when the object is
        # instantiated from a web service call, requiring this output   
        if output_format != None:
            if output_format not in self.param.getVal('output_format'):
                self.param.appVal('output_format',output_format)
 
        return

    def get_ensemble(self):
        ''' Returns a Boolean indicating if the model uses external input
            sources and a list with these sources '''
        return self.param.getEnsemble()

    def set_single_CPU(self) -> None:
        ''' Forces the use of a single CPU '''
        LOG.debug('parameter "numCPUs" forced to be 1')
        self.param.setVal('numCPUs',1)

    def run(self, input_source):
        ''' Executes a default predicton workflow '''

        # path to endpoint
        endpoint = utils.model_path(self.model, self.version)
        if not os.path.isdir(endpoint):
            self.conveyor.setError(f'Unable to find model {self.model}, version {self.version}')
            #LOG.error(f'Unable to find model {self.model}')

        if not self.conveyor.getError():
            # uses the child classes within the 'model' folder,
            # to allow customization of
            # the processing applied to each model
            modpath = utils.module_path(self.model, self.version)

            idata_child = importlib.import_module(modpath+".idata_child")
            apply_child = importlib.import_module(modpath+".apply_child")
            odata_child = importlib.import_module(modpath+".odata_child")

            # run idata object, in charge of generate model data from input
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

        if self.param.getVal('output_similar') is True:

            from flame.sapply import Sapply

            metric = self.param.getVal('similarity_metric')
            numsel = self.param.getVal('similarity_cutoff_num')
            cutoff = self.param.getVal('similarity_cutoff_distance')
            
            # sapply = Sapply(self.param, self.conveyor)

            sapply_child = importlib.import_module(modpath+".sapply_child")

            # run apply object, in charge of generate a prediction from idata
            try:
                sapply = sapply_child.SapplyChild(self.param, self.conveyor)
            except:
                LOG.warning ('Sapply child architecture mismatch, defaulting to Sapply parent')
                sapply = Sapply(self.param, self.conveyor)

            sapply.run(cutoff, numsel, metric)
            LOG.debug(f'sapply child {type(sapply).__name__} completed `run()`')

        if not self.conveyor.getError():
            # run apply object, in charge of generate a prediction from idata
            try:
                apply = apply_child.ApplyChild(self.param, self.conveyor)
            except:
                LOG.warning ('Apply child architecture mismatch, defaulting to Apply parent')
                apply = Apply(self.param, self.conveyor)

            apply.run()
            LOG.debug(f'apply child {type(apply).__name__} completed `run()`')

        # run odata object, in charge of formatting the prediction results
        # note that if any of the above steps failed, an error has been inserted in the
        # conveyor and odata will take case of showing an error message
        try:
            odata = odata_child.OdataChild(self.param, self.conveyor)
        except:
            LOG.warning ('Odata child architecture mismatch, defaulting to Odata parent')
            odata = Odata(self.param, self.conveyor)

        return odata.run()
