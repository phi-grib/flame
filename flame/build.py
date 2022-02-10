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
from flame.learn import Learn
from flame.odata import Odata

LOG = get_logger(__name__)

class Build:

    def __init__(self, model, param_file=None, param_string=None, output_format=None):
        LOG.debug('Starting build...')
        self.model = model
        self.param = Parameters()
        
        self.conveyor = Conveyor()

        # identify the workflow type
        self.conveyor.setOrigin('learn')

        # generate a unique modelID
        self.conveyor.addMeta('modelID',utils.id_generator())
        LOG.debug(f'Generated new model with modelID: {self.conveyor.getMeta("modelID")}')

        # load parameters
        if param_file is not None:
            # use the param_file to update existing parameters at the model
            # directory and save changes to make them persistent
            success, message = self.param.delta(model, 0, param_file, iformat='YAML')

        elif param_string is not None:
            success, message = self.param.delta(model, 0, param_string, iformat='JSONS')

        else:
            # load parameter file at the model directory
            success, message = self.param.loadYaml(model, 0)

        # being unable to load parameters is a critical error
        if not success:
            LOG.critical(f'Unable to load model parameters. {message}. Aborting...')
            sys.exit(1)

        # add additional output formats included in the constructor 
        # this is requiered to add JSON format as output when the object is
        # instantiated from a web service call, requiring this output   
        if output_format is not None:
            if output_format not in self.param.getVal('output_format'):
                self.param.appVal('output_format',output_format)

        if self.param.getVal('confidential'):
            self.confidentialAuditParam()
 
    def confidentialAuditParam (self):
        import yaml

        original_method = self.param.getVal('model')
        if self.param.getVal ('quantitative'):
            if original_method != 'PLSR':
                self.param.setVal('model', 'PLSR')
                LOG.info (f'CONFIDENTIALITY AUDIT: the model was set to PLSR, '
                f'the original method {original_method} was not suitable to build confidential models')
        else:
            if original_method != 'PLSDA':
                self.param.setVal('model', 'PLSDA')
                LOG.info (f'CONFIDENTIALITY AUDIT: the model was set to PLSDA, '
                f'the original method {original_method} was not suitable to build confidential models')
        
        parameters_file_path = utils.model_path(self.model, 0)
        parameters_file_name = os.path.join (parameters_file_path,
                                            'parameters.yaml')
        with open(parameters_file_name, 'w') as pfile:
            yaml.dump (self.param.p, pfile)

    def get_ensemble(self):
        ''' Returns a Boolean indicating if the model uses external input
            sources and a list with these sources '''
        return self.param.getEnsemble()

    def extend_modelID (self, ensembleID):
        modelID = self.conveyor.getMeta('modelID')
        modelID = f'{modelID}-{ensembleID}'
        self.conveyor.addMeta('modelID', modelID)
        LOG.debug (f'modelID re-defined as {self.conveyor.getVal("modelID")}')

    def set_single_CPU(self) -> None:
        ''' Forces the use of a single CPU '''
        LOG.debug('parameter "numCPUs" forced to be 1')
        self.param.setVal('numCPUs',1)

    def run(self, input_source):
        ''' Executes a default predicton workflow '''

        # path to endpoint
        epd = utils.model_path(self.model, 0)
        # if not os.path.isdir(epd):
        #     self.conveyor.setError(f'Unable to find model {self.model}')
        #     #LOG.error(f'Unable to find model {self.model}')

        # import ichild classes
        # if not self.conveyor.getError():
        # uses the child classes within the 'model' folder,
        # to allow customization of  the processing applied to each model
        modpath = utils.module_path(self.model, 0)

        idata_child = importlib.import_module(modpath+".idata_child")
        learn_child = importlib.import_module(modpath+".learn_child")
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
            # check there is a suitable X and Y
            if not self.conveyor.isKey ('xmatrix'):
                self.conveyor.setError(f'Failed to compute MDs')

            if not self.conveyor.isKey ('ymatrix'):
                self.conveyor.setError(f'No activity data (Y) found in training series')
    
            # run optional chemical space building for supporting "closest" training series object
            # if self.param.getVal('buildSimilarity'):
            if self.param.getVal('output_similar') is True:

                from flame.slearn import Slearn

                slearn_child = importlib.import_module(modpath+".slearn_child")
                
                if not self.conveyor.getError():
                    # instantiate learn (build a space from idata) and run it
                    try:
                        slearn = slearn_child.SlearnChild(self.param, self.conveyor)
                    except:
                        LOG.warning ('Slearn child architecture mismatch, defaulting to Learn parent')
                        slearn = Slearn(self.param, self.conveyor)

                    slearn.run()
                    LOG.debug(f'slearn child {type(slearn).__name__} completed `run()`')

        if not self.conveyor.getError():

            # instantiate learn (build a model from idata) and run it
            try:
                learn = learn_child.LearnChild(self.param, self.conveyor)
            except:
                LOG.warning ('Learn child architecture mismatch, defaulting to Learn parent')
                learn = Learn(self.param, self.conveyor)
            learn.run()

            LOG.debug(f'learn child {type(learn).__name__} completed `run()`')

        # run odata object, in charge of formatting the prediction results
        # note that if any of the above steps failed, an error has been inserted in the
        # conveyor and odata will take case of showing an error message
        try:
            odata = odata_child.OdataChild(self.param, self.conveyor)
        except:
            LOG.warning ('Odata child architecture mismatch, defaulting to Odata parent')
            odata = Odata(self.param, self.conveyor)

        return odata.run()