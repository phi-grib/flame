#! -*- coding: utf-8 -*-

##    Description    Flame flInput class
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
import hashlib
from rdkit import Chem

class flInput:

    def __init__ (self, iflcontrol, ifile):

        self.control = iflcontrol
        self.ifile = ifile

    def countmol (self, ifile):
        # estimate number of molecules inside the SDFile

        # RdKit version
        try:
            suppl = Chem.SDMolSupplier(ifile)
        except:
            return False, 'unable to open molfile'
        
        nmol = len(suppl)

        # Trivial version
        # nmol = 0
        # try:
        #     with open (ifile,'r') as f:
        #         for line in f:
        #             if line.startswith('$$$$'):
        #                 nmol+=1
        # except:
        #     return False, "error opening"+ifile
        
        if nmol == 0:
            return False, "no molecule found in file"+ifile
        
        return True, nmol

    def standardize (self, ifile):

        return True, "debug dummy"

    def ionize (self, ifile):

        return True, "debug dummy"

    def computeMD (self, ifile):

        return True, "debug dummy"

    def save (self, results):

        print (self.control.md5stamp())
        # pickle results + stamp in ifile.pickle
        # return True

        return True

    def run (self):

        # check for presence of pickle file
        # if true, extract MD5 stamp, compute control MD5 stamp and if both are coincident extract results and exit

        # open file
        if not os.path.isfile (self.ifile):
            return False, "input error: file not found"

        tfile = self.ifile
        
        if (self.control.input_type == 'molecule'):

            # count number of molecules
            success, results = self.countmol (tfile)
            if not success:
                return False, "input error: no molecule recognized: "+str(results)
            else:
                nobj = results

                # debug
                print (nobj)

            # normalize chemical
            if self.control.normalize_method != None:
                success, results = self.standardize (tfile)
                if not success:
                    return False, "input error: chemical standardization failed: "+str(results)
                else:
                    tfile = results

            # ionize molecules
            if self.control.ionize_method != None:
                success, results = self.ionize (tfile)
                if not success:
                    return False, "input error: molecule ionization error at position: "+str(results)
                else:
                    tfile = results

            # compute MD
            success, results = self.computeMD (tfile)
            if not success:
                return False, "input error: failed computing MD: "+str(results)        

        elif (self.control.input_type == 'data'):

            #   test and obtain dimensions
            #   normalize data

            print ("data")

        else:

            print ("unknown input format")


        # save and stamp
        success = self.save (results)

        # runner class? will split in chunks and run every chunck in a thread, then reassembling the results
        # at the end
        # the same class will take care of situations where the loop execution fails and fallback to compound 
        # per compound mode

        results = self.ifile + '_i'

        return success, results

