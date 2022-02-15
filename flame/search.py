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
import pickle
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
        LOG.debug('Starting search...')
        self.space = space
        self.version = version
        self.label = label
        self.param = Parameters()
        self.conveyor = Conveyor()

        # identify the workflow type
        self.conveyor.setOrigin('sapply')

        # load modelID
        path = utils.space_path(space, version)
        meta = os.path.join(path,'space-meta.pkl')
        try:
            with open(meta, 'rb') as handle:
                modelID = pickle.load(handle)
        except:
            LOG.critical(f'Unable to load modelID from {meta}. Aborting...')
            sys.exit()

        self.conveyor.addMeta('modelID', modelID)
        LOG.debug (f'Loaded space with modelID: {modelID}')

        # assign prediction (search) label
        self.conveyor.addVal(label, 'prediction_label', 'prediction label',
            'method', 'single',
            'Label used to identify the prediction')

        success, results = self.param.loadYaml(space, version, isSpace=True)
        if not success:
            LOG.critical(f'Unable to load space parameters. {results}. Aborting...')
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

    def getVal (self, idict, ikey):
        if not ikey in idict:
            return None
        return idict[ikey]

    # def run(self, input_source, runtime_param=None, metric=None, numsel=None, cutoff=None):
    def run(self, param_dict):
        ''' Executes a default predicton workflow '''

        metric = None
        numsel = None
        cutoff = None
        
        # path to endpoint
        epd = utils.space_path(self.space, self.version)
        if not os.path.isdir(epd):
            LOG.error(f'Unable to find space {self.space}')
            self.conveyor.setError(f'Unable to find space {self.space}, version {self.version}')

        if self.getVal(param_dict,'smarts') is not None:
            input_source = param_dict['smarts']
            self.param.setVal('input_type', 'smarts')

        elif self.getVal(param_dict,'infile') is not None:
            input_source = param_dict['infile']

        else:
            LOG.error(f'Unable to find input_file')
            self.conveyor.setError('wrong format in the runtime similarity parameters')

        if 'runtime_param' in param_dict:
            runtime_param = self.getVal(param_dict, 'runtime_param')
            if runtime_param is not None:
                LOG.info (f'runtime parameters: {str(runtime_param)}')
                try:
                    with open(runtime_param, 'r') as pfile:
                        rtparam = yaml.safe_load(pfile)
                        try:
                            metric = rtparam['similarity_metric']
                            numsel = rtparam['similarity_cutoff_num']
                            cutoff = rtparam['similarity_cutoff_distance']
                        except:
                            LOG.error('wrong format in the runtime similarity parameters')
                            self.conveyor.setError('wrong format in the runtime similarity parameters')
                except:
                    LOG.error('runtime similarity parameter file not found')
                    self.conveyor.setError('runtime similarity parameter file not found')
        else:
            try:
                metric = param_dict['metric']
                numsel = param_dict['numsel']
                cutoff = param_dict['cutoff']
            except:
                LOG.error('wrong format in the runtime similarity parameters')
                self.conveyor.setError('wrong format in the runtime similarity parameters')

        md = self.param.getVal('computeMD_method')
        if utils.isFingerprint(md) and len(md) > 1:
            LOG.warning(f'When using fingerprints, only a single type of MD can be used to build spaces. Selecting {md[0]}')
            self.conveyor.setWarning(f'When using fingerprints, only a single type of MD can be used to build spaces. Selecting {md[0]}')
            self.param.setVal('computeMD_method',[md[0]])

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
            success, results = idata.preprocess_apply()
            if not success:
                self.conveyor.setError(results)
                
        if not self.conveyor.getError():

            # make sure there is X data
            if not self.conveyor.isKey('xmatrix'):
                if not self.conveyor.isKey ('SMARTS'):
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
            odata = odata_child.OdataChild(self.param, self.conveyor)
        except:
            LOG.warning ('Odata child architecture mismatch, defaulting to Odata parent')
            odata = Odata(self.param, self.conveyor)

        return odata.run()
