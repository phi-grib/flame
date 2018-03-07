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
import pickle

import numpy as np
from rdkit import Chem

import multiprocessing as mp

import sdfileutils as sdfu
from standardiser import standardise

import _compute_md as computeMD
import _convert_3d as convert3D

import utils as utils

class Idata:

    def __init__ (self, control, ifile):

        self.control = control      # control object defining the processing
        self.ifile = ifile          # input file

    def extractAnotations (self, ifile):
        """         
        Extracts molecule names, biological anotations and experimental values from an SDFile.

        Returns a tupple with three lists:
        [0] Molecule names
        [1] Molecule activity values (as np.array(dtype=np.float64))
        [2] Molecule activity type (i.e. IC50) (as np.array(dtype=np.float64))     
        
        """

        suppl = Chem.SDMolSupplier(ifile)
        obj_nam = []
        obj_bio = []
        obj_exp = []

        for i, mol in enumerate(suppl):

            # Do not even try to process molecules not recognised by RDKit. 
            # They will be removed at the normalization step
            if mol is None:
                continue
                
            name = sdfu.getName(mol, count=i, field=self.control.SDFile_name, suppl= suppl)

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

            obj_nam.append(name)
            obj_bio.append(activity_num)
            obj_exp.append(exp)

        result = (obj_nam, np.array(obj_bio, dtype=np.float64), np.array(obj_exp, dtype=np.float64))

        return result

    def normalize (self, ifile, method, clean=False):
        """
        Generates a simplified SDFile with MolBlock and an internal ID for further processing

        Also, when defined in control, applies chemical standardization protocols, like the 
        one provided by Francis Atkinson (EBI), accessible from:

            https://github.com/flatkinson/standardiser
        
        Returns a tuple containing the result of the method and (if True) the name of the 
        output molecule and an error message otherwyse

        WARNING: if clean is set to True it will remove the original file
        """

        if not method :
            return True, ifile

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
                merror = 0
                for m in suppl:

                    # molecule not recognised by RDKit
                    if m is None:
                        print ("ERROR: unable to process molecule #"+str(merror))
                        merror+=1
                        continue

                    # if standardize
                    if method == 'standardize':
                        try:
                            parent = standardise.run (Chem.MolToMolBlock(m))
                        except standardise.StandardiseException as e:
                            if e.name == "no_non_salt":
                                parent = Chem.MolToMolBlock(m)
                            else:
                                return False, e.name
                        except:
                            return False, "Unknown standardiser error"

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

    def ionize (self, ifile, method):
        """ Adjust the ionization status of the molecular strcuture, using a given pH.
        """

        if not method :
            return True, ifile

        success = False
        results = 'not ionized'

        # methods here

        results = 'ionization method not recognised'

        return success, results

    def convert3D (self, ifile, method):
        """ Assigns 3D structures to the molecular structures provided as input.
        """

        if not method :
            return True, ifile
            
        success = False
        results = 'not converted to 3D'

        if 'ETKDG' in method :
            success, results  = convert3D._ETKDG(ifile)
            
        return success, results

    def computeMD_custom (self, ifile):
        """ 
        
        Empty method for computing molecular descriptors

        ifile is a molecular file in SDFile format

        returns a boolean anda a tupla of two elements:
        [0] xmatrix (nparray np.float64)
        [1] list of variable names (str)

        example:    return True, (xmatrix, md_nam)

        """
        
        return False, 'not implemented'

    def computeMD (self, ifile, method):
        """ Uses the molecular structures for computing an array of values (int or float) 
        """

        # any call to computeMD_[whatever] must return a numpy array with a value for
        # each molecule in ifile       
        
        results_all = []

        if 'RDKit_properties' in method :
            success, results  = computeMD._RDKit_properties(ifile)
            if success :
                results_all.append(results)
        
        if 'RDKit_md' in method :
            success, results  = computeMD._RDKit_descriptors(ifile)
            if success :
                results_all.append(results)
        
        if 'custom' in method :
            success, results  = self.computeMD_custom(ifile)
            if success :
                results_all.append(results)
        
        if len(results_all) == 0:
            success = False
            results = 'undefined MD'

        # TODO: check that the number of objects is the same for all the pieces

        return success, results

    def consolidate (self, results, nobj):
        """ Mix the results obtained by multiple CPUs into a single result file 
        """

        success = True
        first = True
        nresults = None
        nnames = []

        for iresults in results:
            #iresults [0] = success
            #iresults [1] = (xmatrix, var_name)

            if iresults[0] == False :
                success = False
                results = iresults [1]
                break
            
            internal = iresults [1]

            if type (internal[0]).__module__ == np.__name__:

                if first:
                    nresults = internal[0]
                    nnames = internal[1]
                    first = False
                else:
                    nresults = np.vstack ((nresults, internal[0]))
                    nnames.append(internal[1])

                #print ('merge arrays')
            
            else :
                success = False
                results = 'unknown results type in consolidate'
                break

        if success:
            results = (nresults, nnames)

        return success, results

    def save (self, results):
        """ 
        Saves the results in serialized form, together with the MD5 signature of the control class and the input file
        """

        #md5_control = self.control.md5stamp()
        md5_control = utils.md5stamp (self.control)
        md5_input = utils.md5sum(self.ifile)  # run md5 in self.ifile

        try:
            with open ('./data.pkl', 'wb') as fo:
                pickle.dump (md5_control, fo)
                pickle.dump (md5_input, fo)
                pickle.dump (results[0],fo)
                pickle.dump (results[1],fo)
                pickle.dump (results[2],fo)
                pickle.dump (results[3],fo)
                pickle.dump (results[4],fo)
        except :
            return False

        return True

    def load (self):
        """ 
        Loads the results in serialized form, together with the MD5 signature of the control class and the input file
        """

        try:
            with open ('./data.pkl', 'rb') as fi:
                md5_con = pickle.load(fi)
                if md5_con != utils.md5stamp(self.control):
                    return False, 'md5 control'

                md5_fil = pickle.load(fi)
                if md5_fil != utils.md5sum(self.ifile):
                    return False, 'md5 input file'

                xmatrix = pickle.load(fi)
                ymatrix = pickle.load(fi)
                experim = pickle.load(fi)
                obj_nam = pickle.load(fi)
                var_nam = pickle.load(fi)
        except :
            return False, 'unable to open pickl file'
    
        results = (xmatrix, ymatrix, experim, obj_nam, var_nam)

        print ('recycling!')

        return True, results

    def workflow (self, ifile):
        """         
        Executes in sequence methods required to generate MD, starting from a single molecular file

        input : ifile, a molecular file in SDFile format
        output: results is a numpy bidimensional array containing MD       
        """

        # normalize chemical  
        success, results = self.normalize (ifile, self.control.normalize_method)
        if not success :
            return success, results

        # ionize molecules
        success, results = self.ionize (results, self.control.ionize_method)
        if not success :
            return success, results
        
        # generate a 3D structure
        success, results = self.convert3D (results, self.control.convert3D_method)
        if not success :
            return success, results
        
        # compute MD
        success, results = self.computeMD (results, self.control.computeMD_method)

        return success, results

    def run (self):
        """         
        Process input file to obtain metadata (size, type, number of objects, name of objects, etc.) as well
        as for generating MD
            
        The results are saved in a MD5 stamped pickle, to avoid recomputing model input from the same input
        file
        
        This methods supports multiprocessing, splitting original files in a chunck per CPU        
        """

        # check for the presence of a valid pickle file
        success, results = self.load()

        if success:
            return success, results

        # processing for molecular input (for now an SDFile)
        if (self.control.input_type == 'molecule'):

            # trick to avoid RDKit dumping warnings to the console
            if not self.control.verbose_error:
                stderr_fileno = sys.stderr.fileno()       # saves current syserr
                stderr_save = os.dup(stderr_fileno)
                stderr_fd = open('errorRDKit.log', 'w')   # open a specific RDKit log file
                os.dup2(stderr_fd.fileno(), stderr_fileno)

            # extract useful information from file

            results = self.extractAnotations (self.ifile)
            obj_nam = results[0]
            ymatrix = results[1]
            experim = results[2]

            # print (self.obj_nam)
            # print (self.obj_bio)
            # print (self.obj_exp)

            # Execute the workflow in 1 or n CPUs
            if self.control.numCPUs > 1:
                # Count number of molecules and split in chuncks 
                # for multiprocessing 
                success, results = sdfu.split_SDFile(self.ifile, self.control.numCPUs)

                if not success : 
                    return False, "error splitting: "+self.ifile

                split_files_names = results[0]
                split_files_sizes = results[1]

                # print (split_files_names, split_files_sizes)

                pool = mp.Pool(self.control.numCPUs)
                results = pool.map(self.workflow, split_files_names)

                # Check the results and make sure there are 
                # no missing objects.
                # Reassemble results for parallel computing results
                success, results = self.consolidate(results, split_files_sizes) 
            else:
                success, results = self.workflow (self.ifile)

            if not success:
                return False, results

            if not self.control.verbose_error:
                stderr_fd.close()                     # close the RDKit log
                os.dup2(stderr_save, stderr_fileno)   # restore old syserr

        # processing for non-molecular input
        elif (self.control.input_type == 'data'):

            # TODO: import csv
            # test and obtain dimensions
            
            print ("data")

        else:
            return False, 'unknown input data format'

        # extract x matrix and variable names from results
        xmatrix = results [0]
        var_nam = None
        if len(results) > 1:
            var_nam = results [1]
        
        # results is a tuple with:
        # [0] xmatrix
        # [1] ymatrix 
        # [2] experim      for prediction quality assessment   
        # [3] obj_nam      for presenting results
        # [4] var_nam        
        results = (xmatrix, ymatrix, experim, obj_nam, var_nam)
        
        # save and stamp
        success = self.save (results)

        return success, results
