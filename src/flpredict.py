#! -*- coding: utf-8 -*-

##    Description    Flame flPredict class
##
##    Authors:       Manuel Pastor (manuel.pastor@upf.edu)
##
##    Copyright 2018 Manuel Pastor
##
##    This file is part of Flame
##
##    eTOXlab is free software: you can redistribute it and/or modify
##    it under the terms of the GNU General Public License as published by
##    the Free Software Foundation version 3.
##
##    eTOXlab is distributed in the hope that it will be useful,
##    but WITHOUT ANY WARRANTY; without even the implied warranty of
##    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
##    GNU General Public License for more details.
##
##    You should have received a copy of the GNU General Public License
##    along with Flame.  If not, see <http://www.gnu.org/licenses/>.

import os
import sys

class flPredict:

    def __init__ (self, iFile, iModel):

        self.iFile = iFile
        self.iModel = iModel

        return

    def run (self):    

        success = True
        results = 'Error'

        # use self.iModel to load path for main classes
        try:
            epd = './'+self.iModel
            if os.path.isdir(epd):
                sys.path.append(epd)
                from iflcontrol import iflControl
                from iflinput import iflInput
                from iflapply import iflApply
                from ifloutput import iflOutput
            else:
                print ('unable to find specified model '+self.iModel)
                success = False    

        except:
            print ('unable to load main classes')
            success = False


        if success:
            flcontrol = iflControl()

            flinput = iflInput (flcontrol, self.iFile)
            success, results = flinput.run ()

        if success :
            flapply = iflApply (flcontrol, results)
            success, results = flapply.run ()

        if success : 
            floutput = iflOutput (flcontrol, results)
            success, results = floutput.run ()

        return success, results

