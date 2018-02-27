#! -*- coding: utf-8 -*-

##    Description    Flame Learn class
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

class Learn:

    def __init__ (self, ifile):

        self.ifile = ifile

    def run (self):

        # retrieve data and dimensions from results
        
        # get dimensions 
        success = True
        results = self.ifile + '_i'

        return success, results

