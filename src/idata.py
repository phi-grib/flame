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
import sdfileutils as sdfu
from standardiser import standardise
import numpy as np

class Idata:

    def __init__ (self, control, ifile):

        self.control = control      # control object defining the processing
        self.ifile = ifile          # input file

    def extractAnotations (self, ifile):
        """         
        Extracts molecule names, biological anotations and experimental values from an SDFile . 
        Returns a tupple with three lists:
            - Molecule names
            - Molecule activity values
            - Molecule activity type (i.e. IC50)        
        """

        suppl = Chem.SDMolSupplier(ifile)
        molcount = 0
        obj_nam = []
        obj_bio = []
        obj_exp = []

        for i in range(len(suppl)):
            mol = suppl[i]
            molname = sdfu.getName(mol, count=i, field=self.control.SDFile_name, suppl= suppl)

            activity_num = None
            exp = None

            if mol.HasProp(self.control.SDFile_activity):
                activity_str = mol.GetProp(self.control.SDFile_activity)
                try:
                    activity_num = float (activity_str)
                except:
                    activity_num = None            

            if mol.HasProp(self.control.SDFile_experimental):
                exp = mol.GetProp(self.control.SDFile_experimental)

            obj_nam.append(molname)
            obj_bio.append(activity_num)
            obj_exp.append(exp)

        result = (obj_nam, obj_bio, obj_exp)

        return result

    def normalize (self, ifile, clean=False):
        """
        Generates a simplified SDFile with MolBlock and an internal ID for further processing

        Also, when defined in control, applies chemical standardization protocols, like the 
        one provided by Francis Atkinson (EBI), accessible from:

            https://github.com/flatkinson/standardiser
        
        Returns a tuple containing the result of the method and (if True) the name of the 
        output molecule and an error message otherwyse

        WARNING: if clean is set to True it will remove the original file
        """

        try:
            suppl=Chem.SDMolSupplier(ifile)
        except:
            success = False
            result = 'Error at processing input file for standardizing structures'
        else:
            success = True
            filename, fileext = os.path.splitext(ifile)
            ofile = filename + '_std' + fileext
            with open (ofile,'w') as fo:
                mcount = 0
                for m in suppl:

                    # if standardize
                    if self.control.chemstand_method == 'standardize':
                        try:
                            success, parent, error = standardise.run (Chem.MolToMolBlock(m))
                        except standardise.StandardiseException as e:
                            if e.name == "no_non_salt":
                                parent = Chem.MolToMolBlock(m)
                            else:
                                return False, e.name

                    # in any case, write parent plus internal ID (flameID)
                    fo.write(parent)

                    flameID = 'fl%0.10d' % mcount
                    fo.write('>  <flameID>\n'+flameID+'\n\n')

                    mcount += 1

                    # terminator
                    fo.write('$$$$\n')

            if clean:
                try:
                    os.remove (ifile)
                except OSError:
                    pass

            result = ofile

        return success, result

    def ionize (self, ifile):
        """ Adjust the ionization status of the molecular strcuture, using a given pH.
        """

        return True, ifile

    def convert3D (self, ifile):
        """ Assigns 3D structures to the molecular structures provided as input.
        """

        return True, ifile

    def computeMD (self, ifile):
        """ Uses the molecular structures for computing an array of values (int or float) 
        """

        # return a numpy array with as many rows and nobj        
         
        nmol = sdfu.count_mols (ifile)

        xmatrix = np.zeros ((nmol,5),dtype=np.float64)
        result = xmatrix

        return True, result

    def consolidate (self, results, nobj):
        """ Mix the results obtained by multiple CPUs into a single result file 
        """

        first = True
        nresults = None

        for iresults in results:
            if iresults[0] == False :
                success = False
                break
            
            if type (iresults[1]).__module__ == np.__name__:

                if first:
                    nresults = iresults [1]
                    first = False
                else:
                    nresults = np.vstack ((nresults, iresults[1]))

                print ('merge arrays')
            
            else :
                print ('unknown')

        if success:
            result = nresults
        else:
            result = 'Error in consolidation'

        return True, result

    def save (self, results):
        """ 
        Saves the results in serialized form, together with the MD5 stamp of the control class
        """

        print (self.control.md5stamp())
        # pickle results + stamp in ifile.pickle
        # return True

        return True

    def workflow (self, ifile):
        """         
        Executes in sequence methods required to generate MD, starting from a single molecular file

        input : ifile, a molecular file in SDFile format
        output: results is a numpy bidimensional array containing MD       
        """

        # tfile is the name of the temporary molecular file and will change in the workflow
        tfile = ifile  

        # normalize chemical  
        success, results = self.normalize (tfile)
        if not success:
            result = 'Input error: chemical standardization failed: '+str(results)
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
        """         
        Process input file to obtain metadata (size, type, number of objects, name of objects, etc.) as well
        as for generating MD
            
        The results are saved in a MD5 stamped pickle, to avoid recomputing model input from the same input
        file
        
        This methods supports multiprocessing, splitting original files in a chunck per CPU        
        """

        # TODO: check for presence of pickle file
        # if true, extract MD5 stamp, compute control MD5 stamp and if both are coincident extract results and exit
        
        # processing for molecular input (for now an SDFile)
        if (self.control.input_type == 'molecule'):

            # extract useful information from file
            results = self.extractAnotations (self.ifile)
            self.obj_nam = results[0]
            self.obj_bio = results[1]
            self.obj_exp = results[2]

            # print (self.obj_nam)
            # print (self.obj_bio)
            # print (self.obj_exp)

            # Execute the workflow in 1 or n CPUs
            if self.control.numCPUs > 1:
                # Count number of molecules and split in chuncks 
                # for multiprocessing 
                split_files_sizes, split_files = sdfu.split_SDFile (self.ifile, self.control.numCPUs)
                pool = mp.Pool(self.control.numCPUs)
                results = pool.map(self.workflow, split_files)

                # Check the results and make sure there are 
                # no missing objects.
                # Reassemble results for parallel computing results
                success, results = self.consolidate(results, split_files_sizes) 
            else:
                success, results = self.workflow (self.ifile)

        # processing for non-molecular input
        elif (self.control.input_type == 'data'):

            #   test and obtain dimensions
            #   normalize data

            print ("data")

        else:

            print ("unknown input format")

        # save and stamp
        success = self.save (results)

        # results is a tuple with:
        # [0] X numpy
        # [1] Y numpy
        # [2] flameID       this is important for retrieving structure
        # [2] objnames      for presenting results
        # [3] expinfo       for prediction quality assessment      

        return success, (results, self.obj_bio, None, self.obj_nam, self.obj_exp)
