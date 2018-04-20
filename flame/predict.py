#! -*- coding: utf-8 -*-

##    Description    Flame Predict class
##
##    Authors:       Manuel Pastor (manuel.pastor@upf.edu)
##
##    Copyright 2018 Manuel Pastor
##
##    This file is part of Flame
##
##    Flame is free software: you can redistribute it and/or modify
##    it under the terms of the GNU General Public License as published by
##    the Free Software Foundation version 3.
##
##    Flame is distributed in the hope that it will be useful,
##    but WITHOUT ANY WARRANTY; without even the implied warranty of
##    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
##    GNU General Public License for more details.
##
##    You should have received a copy of the GNU General Public License
##    along with Flame. If not, see <http://www.gnu.org/licenses/>.

import os
import importlib

import util.utils as utils
from control import Control

class Predict:

    def __init__ (self, model, version, out_format='JSON'):

        self.model = model
        self.version = version
        self.out_format = out_format

        # instance Control object
        self.control = Control(model, version)
        self.parameters = self.control.get_parameters()

        return


    def get_model_set(self):
        ''' Returns a Boolean indicating if the model uses external input sources and a list with these sources '''
        return self.control.get_model_set()


    def set_single_CPU(self):
        ''' Forces the use of a single CPU '''
        self.parameters['numCPUs']=1


    def run (self, input_source):
        ''' Executes a default predicton workflow '''

        # path to endpoint
        epd = utils.model_path(self.model, self.version)
        if not os.path.isdir(epd):
            return False, 'unable to find model: '+self.model+' version: '+str(self.version)

        #uses the child classes within the 'model' folder, to allow customization of
        #the processing applied to each model
        modpath = utils.module_path(self.model, self.version)

        idata_child = importlib.import_module (modpath+".idata_child")
        apply_child = importlib.import_module (modpath+".apply_child")
        odata_child = importlib.import_module (modpath+".odata_child")
        
        # run idata object, in charge of generate model data from input
        idata = idata_child.IdataChild (self.parameters, input_source)
        success, results = idata.run ()

        if not success:
            return success, results

        # run apply object, in charge of generate a prediction from idata
        apply = apply_child.ApplyChild (self.parameters, results)
        success, results = apply.run ()
        
        if not success:
            return success, results

        # run odata object, in charge of formatting the prediction results
        odata = odata_child.OdataChild (self.parameters, results, self.out_format)
        success, results = odata.run ()

        return success, results

