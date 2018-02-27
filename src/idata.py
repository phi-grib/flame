#! -*- coding: utf-8 -*-

##    Description    Flame Idata class
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
import hashlib
from rdkit import Chem
import multiprocessing as mp
from sdfileutils import splitSDFile
from sdfileutils import nummols
from standardiser import standardise
import numpy as np

class Idata:

    def __init__ (self, control, ifile):

        self.control = control      # control object defining the processing
        self.ifile = ifile          # input file

    def countmol (self, ifile):
        ''' 
    
        Counts number of molecules inside an SDFile given as argument. 
        Additionaly splits it in chunks for multiprocessing.

        '''

        nobj = []
        temp_files = []
        
        success, results = nummols (ifile)
        if not success :
            return False, 'unable to open molfile'
        else :
            nmol = int(results)

        if nmol == 0:
            return False, "no molecule found in file: "+ifile

        if self.control.numCPUs > 1 :

            success, results = splitSDFile(ifile, nmol, self.control.numCPUs)

            if success : 
                nobj = results[0]
                temp_files = results[1]
            else:
                return False, "error splitting: "+ifile

        else :
            nobj.append(nmol)
            temp_files.append(ifile)

        return True, (nobj, temp_files)

    def extractAnotations (self, ifile):
        ''' 
        
        Extracts from an SDFile molecule names, biological anotations and experimental values. 
        Returns three lists of values.
        
        '''
        # if self.SDFileActivity:
        #     if m.HasProp(self.SDFileActivity):
        #         activity = m.GetProp(self.SDFileActivity)
        #         fo.write('>  <'+self.SDFileActivity+'>\n'+activity+'\n')

        # if self.SDFileExperimental:
        #     if m.HasProp(self.SDFileExperimental):
        #         exp = m.GetProp(self.SDFileExperimental)
        #         fo.write('>  <'+self.SDFileExperimental+'>\n'+exp+'\n')

        # for prop in self.SDFileMetadata:
        #     if m.HasProp(prop):
        #         exp = m.GetProp(prop)
        #         fo.write('>  <'+prop+'>\n'+exp+'\n')

        # fo.write('\n$$$$')
        # fo.close()

        # TODO: make it more flexible and extract other info
          
        results = [None, None, None]

        return True, results

    def standardize (self, ifile, clean=False):
        '''
        Applies a structure normalization protocol provided by Francis Atkinson (EBI)

        https://github.com/flatkinson/standardiser
        
        Returns a tuple containing the result of the method and (if True) the name of the 
        output molecule and an error message otherwyse

        '''

        filename, fileext = os.path.splitext(ifile)
        
        ofile = filename + '_std' + fileext

        try:
            suppl=Chem.SDMolSupplier(ifile)
        except:
            return False, 'Error at processing input file for standardizing structures'

        with open (ofile,'w') as fo:
            for m in suppl:
                try:
                    parent = standardise.run (Chem.MolToMolBlock(m))
                except standardise.StandardiseException as e:
                    if e.name == "no_non_salt":
                        parent = Chem.MolToMolBlock(m)
                    else:
                        return False, e.name

                fo.write(parent)
                fo.write('$$$$\n')

        if clean:
            removefile (moli)

        return True, ofile

    def ionize (self, ifile):
        ''' Adjust the ionization status of the molecular strcuture, using a given pH.'''

        return True, ifile

    def convert3D (self, ifile):
        ''' Assigns 3D structures to the molecular structures provided as input.'''

        return True, ifile

    def computeMD (self, ifile):
        ''' Uses the molecular structures for computing an array of values (int or float) '''

        # return a numpy array with as many rows and nobj        
         
        success, results = nummols (ifile)
        if not success :
            return False, 'unable to open molfile'
        else :
            nmol = int(results)

        xmatrix = np.zeros ((nmol,10),dtype=np.float64)

        return True, xmatrix

    def consolidate (self, results, nobj):
        ''' Mix the results obtained by multiple CPUs into a single result file '''

        return True, "debug dummy consolidate"

    def save (self, results):
        ''' 

        Saves the results in serialized form, together with the MD5 stamp of the control class

        '''

        print (self.control.md5stamp())
        # pickle results + stamp in ifile.pickle
        # return True

        return True

    def workflow (self, ifile):
        ''' 
        
        Executes in sequence methods required to generate MD, starting from a single molecular file

        input : ifile, a molecular file in SDFile format
        output: results is a numpy bidimensional array containing MD 
            
        '''

        tfile = ifile
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

        # generate a 3D structure
        if self.control.convert3D_method != None:
            success, results = self.convert3D (tfile)
            if not success:
                return False, "input error: 3D conversion error at position: "+str(results)
            else:
                tfile = results

        # compute MD
        success, results = self.computeMD (tfile)
        if not success:
            return False, "input error: failed computing MD: "+str(results)

        return success, results

    def run (self):
        ''' 
        
        Process input file to obtain metadata (size, type, number of objects, name of objects, etc.) as well
        as for generating MD
            
        The results are saved in a MD5 stamped pickle, to avoid recomputing model input from the same input
        file
            
        This methods supports multiprocessing, splitting original files in a chunck per CPU
        
        '''

        # TODO: check for presence of pickle file
        # if true, extract MD5 stamp, compute control MD5 stamp and if both are coincident extract results and exit

        # open file
        if not os.path.isfile (self.ifile):
            return False, "input error: file not found"
        
        # processing for molecular input (for now an SDFile)
        if (self.control.input_type == 'molecule'):

            # extract useful information from file
            success, results = self.extractAnotations (self.ifile)
            if not success:
                return False, "input error: annotation extraction failed: "+str(results)
            else:
                self.obj_nam = results[0]
                self.obj_bio = results[1]
                self.obj_exp = results[2]

            # count number of molecules and split in chuncks for multiprocessing if necessary
            success, results = self.countmol (self.ifile)
            if not success:
                return False, "input error: no molecule recognized: "+str(results)
            else:
                nobj   = results[0]  # list with nobj of each piece
                tfiles = results[1]  # list with filename of pieces

            print (nobj, tfiles)

            # execute the workflow in 1 or n CPUs
            if len(tfiles) > 1 :
                print ('multi CPU')
                pool = mp.Pool(len(tfiles))
                results = pool.map(self.workflow, tfiles)

                # check the results and make sure there are no missing objects
                # reassemble results for parallel computing results
                success, results = self.consolidate(results,nobj) 
            else:
                print ('single CPU')
                success, results = self.workflow (tfiles[0])

            if not success:
                return False, str(results)

        # processing for non-molecular input
        elif (self.control.input_type == 'data'):

            #   test and obtain dimensions
            #   normalize data

            print ("data")

        else:

            print ("unknown input format")

        # save and stamp
        success = self.save (results)

        # nonsense, only for debugging purposes in development
        #results = self.ifile + '_i'

        return success, results
