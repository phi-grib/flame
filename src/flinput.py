#! -*- coding: utf-8 -*-

##    Description    Flame flInput class
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

class flInput:

    def __init__ (self, iflcontrol, ifile):

        self.iflcontrol = iflcontrol
        self.ifile = ifile

    def run (self):

        # open file

        # depending on control flags, process as molecule or as or data

        # as molecule:

        #   test and obtain dimensions
        #   normalize chemical
        #   ionize

        #   generate MD

        # as data

        #   test and obtain dimensions
        #   normalize data

        # save and stamp

        # runner class? will split in chunks and run every chunck in a thread, then reassembling the results
        # at the end
        # the same class will take care of situations where the loop execution fails and fallback to compound 
        # per compound mode

        success = True
        results = self.ifile + '_i'


        return success, results

