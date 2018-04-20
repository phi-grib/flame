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
import json

import numpy as np
from rdkit import Chem

import multiprocessing as mp

from standardiser import standardise

import chem.sdfileutils as sdfu
import chem.compute_md as computeMD
import chem.convert_3d as convert3D

import util.utils as utils

class Idata:

    def __init__ (self, parameters, input_source):

        self.parameters = parameters      # control object defining the processing

        if ('ext_input' in parameters) and (parameters['ext_input']):
            self.idata = input_source
            self.ifile = None
            self.dest_path = '.' ## TODO: define an appropriate path 

        else:
            self.idate = None
            self.ifile = input_source          
            self.dest_path = os.path.dirname(self.ifile)     # path where any temp file must be written

        if self.dest_path == '':
            self.dest_path = '.'

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
        obj_sml = []

        for i, mol in enumerate(suppl):

            # Do not even try to process molecules not recognised by RDKit. 
            # They will be removed at the normalization step
            if mol is None:
                continue
                
            name = sdfu.getName(mol, count=i, field=self.parameters['SDFile_name'], suppl= suppl)

            activity_num = None
            exp = None

            if mol.HasProp(self.parameters['SDFile_activity']):
                activity_str = mol.GetProp(self.parameters['SDFile_activity'])
                try:
                    activity_num = float (activity_str)
                except:
                    activity_num = None            

            if mol.HasProp(self.parameters['SDFile_experimental']):
                exp = mol.GetProp(self.parameters['SDFile_experimental'])

            ## generate a SMILES
            sml = Chem.MolToSmiles(mol)

            obj_nam.append(name)
            obj_bio.append(activity_num)
            obj_exp.append(exp)
            obj_sml.append(sml)

        anotation_results = {
            'obj_nam': obj_nam,
            'SMILES': obj_sml,
            'ymatrix': np.array(obj_bio, dtype=np.float64),
            'experim': np.array(obj_exp, dtype=np.float64)
        }
        
        return anotation_results 

    def normalize (self, ifile, method):
        """

        Generates a simplified SDFile with MolBlock and an internal ID for further processing

        Also, when defined in control, applies chemical standardization protocols, like the 
        one provided by Francis Atkinson (EBI), accessible from:

            https://github.com/flatkinson/standardiser
        
        Returns a tuple containing the result of the method and (if True) the name of the 
        output molecule and an error message otherwyse

        """

        if not method :
            return True, ifile

        try:
            suppl=Chem.SDMolSupplier(ifile)
        except:
            return False, 'Error at processing input file for standardizing structures'

        success = True
        filename, fileext = os.path.splitext(ifile)
        ofile = filename + '_std' + fileext

        with open (ofile,'w') as fo:
            mcount = 0
            # merror = 0
            for m in suppl:

                # molecule not recognised by RDKit
                if m is None:
                    # print ("ERROR: unable to process molecule #"+str(merror))
                    # merror+=1
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

        return success, ofile

    def ionize (self, ifile, method):
        """ 
        Adjust the ionization status of the molecular structure, using a given pH.
        """

        if not method :
            return True, ifile

        success = False
        results = 'not ionized'

        # methods here

        results = 'ionization method not recognised'

        return success, results

    def convert3D (self, ifile, method):
        """ 
        Assigns 3D structures to the molecular structures provided as input.
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
        """ 
        Uses the molecular structures for computing an array of values (int or float) 
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
        """ 
        Mix the results obtained by multiple CPUs into a single result file 
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
                    first_nobj, first_nvar = np.shape(nresults)
                    first = False
                else:
                    nobj, nvar = np.shape(internal[0])
                    if nvar != first_nvar :
                        return False, "inconsistent number of variables"

                    nresults = np.vstack ((nresults, internal[0]))
                    nnames.append(internal[1])

                #print ('merge arrays')
            
            else :
                return False, "unknown results type in consolidate"

        if success:
            results = (nresults, nnames)

        return success, results

    def save (self, results):
        """ 
        Saves the results in serialized form, together with the MD5 signature of the control class and the input file
        """

        if 'ext_input' in self.parameters and self.parameters['ext_input']:
            return True

        md5_parameters = self.parameters['md5']
        md5_input = utils.md5sum(self.ifile)  # run md5 in self.ifile

        try:
            with open (self.dest_path+'/data.pkl', 'wb') as fo:

                pickle.dump (md5_parameters, fo)
                pickle.dump (md5_input, fo)
                pickle.dump (results["xmatrix"],fo)
                pickle.dump (results["ymatrix"],fo)
                pickle.dump (results["experim"],fo)
                pickle.dump (results["obj_nam"],fo)
                pickle.dump (results["SMILES"],fo)
                pickle.dump (results["var_nam"],fo)
        except :
            return False

        return True

    def load (self):
        """ 
        Loads the results in serialized form, together with the MD5 signature of the control class and the input file
        """

        if 'ext_input' in self.parameters and self.parameters['ext_input']:
            return False, 'model depends on external data sources'

        try:
            with open (self.dest_path+'/data.pkl', 'rb') as fi:
                md5_parameters = pickle.load(fi)
                if md5_parameters != self.parameters['md5']:
                    return False, 'md5 parameters'

                md5_input = pickle.load(fi)
                if md5_input != utils.md5sum(self.ifile):
                    return False, 'md5 input file'

                results = {}
                results["xmatrix"] = pickle.load(fi)
                results["ymatrix"] = pickle.load(fi)
                results["experim"] = pickle.load(fi)
                results["obj_nam"] = pickle.load(fi)
                results["SMILES"] = pickle.load(fi)
                results["var_nam"] = pickle.load(fi)
        except :
            return False, 'unable to open pickl file'
    
        #results = (xmatrix, ymatrix, experim, obj_nam, var_nam)

        print ('recycling!')

        return True, results

    def workflow (self, ifile):
        """      

        Executes in sequence methods required to generate MD, starting from a single molecular file

        input : ifile, a molecular file in SDFile format
        output: results is a numpy bidimensional array containing MD     

        """

        # TODO: implement control of object size, in case any of the steps removes molecules
        # this would produce missmatch problems with object names and Y values and must be
        # avoided, even at the cost of repeating the computation molecule-by-molecule

        # normalize chemical  
        success, results = self.normalize (ifile, self.parameters['normalize_method'])
        if not success :
            return success, results

        # ionize molecules
        success, results = self.ionize (results, self.parameters['ionize_method'])
        if not success :
            return success, results
        
        # generate a 3D structure
        success, results = self.convert3D (results, self.parameters['convert3D_method'])
        if not success :
            return success, results
        
        # compute MD
        success, results = self.computeMD (results, self.parameters['computeMD_method'])

        return success, results

    def _run_molecule (self):
        """
        version of Run for molecular input

        """

        # trick to avoid RDKit dumping warnings to the console
        if not self.parameters['verbose_error']:
            stderr_fileno = sys.stderr.fileno()       # saves current syserr
            stderr_save = os.dup(stderr_fileno)
            stderr_fd = open('errorRDKit.log', 'w')   # open a specific RDKit log file
            os.dup2(stderr_fd.fileno(), stderr_fileno)

        # extract useful information from file

        workflow_results = self.extractAnotations (self.ifile)

        # obj_nam = results[0]
        # obj_sml = results[1]
        # ymatrix = results[2]
        # experim = results[3]

        nobj = len(workflow_results['obj_nam'])

        ncpu = self.parameters['numCPUs']

        # do not run multiprocess for small series, the overheads slow the overall computation time
        if nobj < 10 :
            ncpu = 1

        # never run a single molecule in two CPUs!
        if ncpu > nobj:
            ncpu = nobj

        # Execute the workflow in 1 or n CPUs
        if ncpu > 1:
            # Count number of molecules and split in chuncks 
            # for multiprocessing 
            success, results = sdfu.split_SDFile(self.ifile, ncpu)

            if not success : 
                return False, "error splitting: "+self.ifile

            split_files_names = results[0]
            split_files_sizes = results[1]

            # print (split_files_names, split_files_sizes)

            pool = mp.Pool(ncpu)
            results = pool.map(self.workflow, split_files_names)

            # Check the results and make sure there are 
            # no missing objects.
            # Reassemble results for parallel computing results
            success, results = self.consolidate(results, split_files_sizes) 
        else:
            success, results = self.workflow (self.ifile)

        if not success:
            return False, results

        if not self.parameters['verbose_error']:
            stderr_fd.close()                     # close the RDKit log
            os.dup2(stderr_save, stderr_fileno)   # restore old syserr

        workflow_results['xmatrix'] = results[0]

        if len(results)>1 :
            workflow_results['var_nam'] = results[1]

        # return success, results
        return success, workflow_results

    def _run_data (self):
        """
        version of Run for data input (CSV tabular format)
        """

        success = False
        results = 'not implemented'


        return success, results


    def _run_ext_data (self):
        """
        version of Run for inter-process input (calling another model to obtain input)
        """

        # idata is a list of JSON from 1-n sources
        # the data usable for input must be listed in the ['meta']['main'] key

        # use first JSON to load common info like obj_nam, etc         
        results = json.loads(self.idata[0])

        # identify usable data imported from element 0. This will be deleted latter
        original_main = results ['meta']['main']

        # new, consolidated, usable data will be added as 'xmatrix' 
        results['meta']['main']= ['xmatrix']

        # extract usable data from every source and add to 'combo' np.array
        combo = None
        var_nam = []
        for ijson in self.idata:
            idict = json.loads(ijson)
            main_keys = idict['meta']['main']
            for j in main_keys:
                ## TODO: consider adding a prefix (e.g. 'source_1')
                var_nam.append(j)
                if combo is None:  # for first element just copy
                    combo = np.array(idict[j], dtype=np.float64)
                else: # append laterally
                    combo = np.c_[combo, np.array(idict[j], dtype=np.float64)]
                
        results['xmatrix'] = combo
        results['var_nam'] = var_nam

        # del original usable data in element 0
        for key in original_main:
            del results[key]
        
        return True, results


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

        input_type = self.parameters['input_type'] 
        # processing for molecular input (for now an SDFile)
        if (input_type== 'molecule'):
            success, results = self._run_molecule()
        elif (input_type == 'data'):
            success, results = self._run_data()
        elif (input_type == 'ext_data'):
            success, results = self._run_ext_data()
        else:
            return False, 'unknown input data format'

        # save in a pickle file stamped with MD5 hash of file and control
        if success:
            success = self.save (results)

        return success, results
