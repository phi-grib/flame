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
import sys
import shutil

class Build:

    def __init__ (self, ifile, model):

        self.ifile = ifile
        self.model = model
        self.lfile = None

        return

    def run (self):
        ''' Executes a default predicton workflow '''

        # identify path to endpoint
        wkd = os.path.dirname(os.path.abspath(__file__))
        epd = wkd+'/models/'+self.model+'/dev'

        # copy the input file to the model development directory of the endpoint
        self.lfile = epd+'/'+os.path.basename(self.ifile)
        shutil.copy (self.ifile,self.lfile)

        success = True
        results = ''

        #uses the child classes within the 'model' folder, to allow customization of
        #the processing applied to each model
        if not os.path.isdir(epd):
            return False, 'unable to find model: '+self.model

        try:
            sys.path.append(epd)
            from control_child import ControlChild
            from idata_child import IdataChild
            from learn_child import LearnChild
            from odata_child import OdataChild
        except:
            raise
            #success = False
            #results = 'Error loading model classes:', sys.exc_info()[0]

        if not success:
            return success, results
        
        # instance Control object
        control = ControlChild()

        # run idata object, in charge of generate model data from local copy of input
        idata = IdataChild (control, self.lfile)
        success, results = idata.run ()
        
        if not success:
            return success, results

        # run learn object, in charge of generate a prediction from idata
        learn = LearnChild (control, results)
        success, results = learn.run ()
        
        if not success:
            return success, results

        # run odata object, in charge of formatting the prediction results
        odata = OdataChild (control, results)
        success, results = odata.run ()

        return success, results

