#! -*- coding: utf-8 -*-

##    Description    Flame flPredict class
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
##    along with Flame.  If not, see <http://www.gnu.org/licenses/>.

import os
import sys

class Predict:

    def __init__ (self, ifile, model):

        self.ifile = ifile
        self.model = model

        return

    def run (self):    

        success = True
        results = 'Error'

        # use self.iModel to load path for main classes
        try:
            epd = './'+self.model
            if os.path.isdir(epd):
                sys.path.append(epd)
                from control_child import ControlChild
                from idata_child import IdataChild
                from apply_child import ApplyChild
                from odata_child import OdataChild
            else:
                print ('unable to find specified model '+self.model)
                success = False    

        except:
            print ('unable to load main classes')
            success = False

        if success:
            control = ControlChild()

            idata = IdataChild (control, self.ifile)
            success, results = idata.run ()

        if success :
            apply = ApplyChild (control, results)
            success, results = apply.run ()

        if success : 
            odata = OdataChild (control, results)
            success, results = odata.run ()

        return success, results

