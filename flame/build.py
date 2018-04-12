#! -*- coding: utf-8 -*-

##    Description    Flame Build class
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
import shutil

import util.utils as utils
from control import Control

class Build:

    def __init__ (self, ifile, model):

        self.ifile = ifile
        self.model = model
        self.lfile = None

        return

    def run (self):
        ''' Executes a default predicton workflow '''

        # path to endpoint
        epd = utils.model_path(self.model, 0)
        if not os.path.isdir(epd):
            return False, 'unable to find model: '+self.model
        
        self.lfile = epd+'/'+os.path.basename(self.ifile)
        shutil.copy (self.ifile,self.lfile)

        #uses the child classes within the 'model' folder, to allow customization of
        #the processing applied to each model
        modpath = utils.module_path(self.model, 0)
     
        idata_child = importlib.import_module (modpath+".idata_child")
        learn_child = importlib.import_module (modpath+".learn_child")
        odata_child = importlib.import_module (modpath+".odata_child")
        
        # instance Control object
        control = Control(self.model,0)
        parameters = control.get_parameters()

        # run idata object, in charge of generate model data from local copy of input
        idata = idata_child.IdataChild (parameters, self.lfile)
        success, results = idata.run ()
        
        if not success:
            return success, results

        # run learn object, in charge of generate a prediction from idata
        learn = learn_child.LearnChild (parameters, results)
        success, results = learn.run ()
        
        if not success:
            return success, results

        # run odata object, in charge of formatting the prediction results
        odata = odata_child.OdataChild (parameters, results)
        success, results = odata.run ()

        return success, results

