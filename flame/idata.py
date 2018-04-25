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
import tempfile
import multiprocessing as mp

import numpy as np
from rdkit import Chem

from standardiser import standardise

import chem.sdfileutils as sdfu
import chem.compute_md as computeMD
import chem.convert_3d as convert3D

import util.utils as utils

class Idata:

    def __init__ (self, parameters, input_source):

        self.parameters = parameters      # control object defining the processing
        self.dest_path = '.'              # path for temp files (fallback default)

        self.results = {
            'manifest':[],
            'meta':{'main':[],
                    'endpoint':self.parameters['endpoint'],
                    'version':self.parameters['version'],
                   }
            }    # create empty context index ('manifest')
        

        if ('ext_input' in parameters) and (parameters['ext_input']):
            self.idata = input_source
            self.ifile = None
            randomName = 'flame-'+utils.id_generator()
            self.dest_path = os.path.join(tempfile.gettempdir(),randomName) 

        else:
            self.idate = None
            self.ifile = input_source          
            self.dest_path = os.path.dirname(self.ifile) 


    def extractAnotations (self, ifile):
        """  

        Extracts molecule names, biological anotations and experimental values from an SDFile.

        Anotations must be added using method utils.add_result, so they are also inserted into the results manifest
        
        """

        try:
            suppl = Chem.SDMolSupplier(ifile)
        except:
            self.results['error'] = 'unable to open '+ifile+' input file'
            return

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

        utils.add_result (self.results, obj_nam, 'obj_nam', 'Mol name', 'label', 'objs', 'Name of the molecule, as present in the input file')
        utils.add_result (self.results, obj_sml, 'SMILES', 'SMILES', 'decoration', 'objs', 'Structure of the molecule in SMILES format')

        if not utils.is_empty(obj_bio):
            utils.add_result (self.results, np.array(obj_bio, dtype=np.float64), 'ymatrix', 'Activity', 'decoration', 'objs', 'Biological anotation to be predicted by the model')
        if not utils.is_empty(obj_exp):
            utils.add_result (self.results, np.array(obj_exp, dtype=np.float64), 'experim', 'Experim.', 'decoration', 'objs', 'Experimental anotation present in the input file')
        
        return

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

    def save (self):
        """ 
        Saves the results in serialized form, together with the MD5 signature of the control class and the input file
        """

        if 'ext_input' in self.parameters and self.parameters['ext_input']:
            return

        md5_parameters = self.parameters['md5']
        md5_input = utils.md5sum(self.ifile)  # run md5 in self.ifile

        try:
            with open (self.dest_path+'/data.pkl', 'wb') as fo:

                pickle.dump (md5_parameters, fo)
                pickle.dump (md5_input, fo)
                
                pickle.dump (self.results,fo)
                
        except :
            pass

    def load (self):
        """ 
        Loads the results in serialized form, together with the MD5 signature of the control class and the input file
        """

        if 'ext_input' in self.parameters and self.parameters['ext_input']:
            return False

        try:
            with open (self.dest_path+'/data.pkl', 'rb') as fi:
                md5_parameters = pickle.load(fi)
                if md5_parameters != self.parameters['md5']:
                    return False

                md5_input = pickle.load(fi)
                if md5_input != utils.md5sum(self.ifile):
                    return False

                self.results = pickle.load(fi)

                # these values are added programatically and therefore not
                # checked by the md5
                self.results['meta']['endpoint']=self.parameters['endpoint']
                self.results['meta']['version']=self.parameters['version']

        except :
            return False

        print ('*** recycling ***')

        return True

    def workflow (self, ifile):
        """      

        Executes in sequence methods required to generate MD, starting from a single molecular file

        input : ifile, a molecular file in SDFile format
        output: results is a numpy bidimensional array containing MD     

        """

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

        self.extractAnotations (self.ifile)
        if 'error' in self.results:
            return

        # TODO: Generalize mixing multiple CPUs and object-wise processing
        # Split input file in chuncks of
        #     no objectwise and no multicpu: 1
        #     no objectwise and c cpus     : c
        #     objectwise and no multicpu   : n
        #     objectwise and c cpus        : n/c
        #
        # in every case, first split and then send each chunk to a 
        # common conveniece function what decides if raise threads or not
        # and consolidates results if neccesary
        #
        # guarantee 100% that the number of objects is the same and that 
        # no uncontroled "holes" are shown
        #
        # objectwise:
        # prune the serie of "holes" and create "warnings" in self.results
        # reporting processing errors
        #
        # serieswise:
        # if the number of objects disagree create "error" in self.results 
        # and exit
        #
        # auto:
        # start serieswise and in error repeat objectwise
        

        nobj = len(self.results['obj_nam'])

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
            self.results['error'] = 'error in run molecule workflow'
            return

        if not self.parameters['verbose_error']:
            stderr_fd.close()                     # close the RDKit log
            os.dup2(stderr_save, stderr_fileno)   # restore old syserr

        self.results['xmatrix'] = results[0]
        utils.add_result (self.results, results[0], 'xmatrix', 'X matrix', 'method', 'vars', 'Molecular descriptors')

        if len(results)>1 :
            utils.add_result (self.results, results[1], 'var_nam', 'Var names', 'method', 'vars', 'Names of the X variables')

        # return success, results
        return


    def _run_data (self):
        """
        version of Run for data input (CSV tabular format)
        """

        self.results ['error'] = 'importing data is not implemented yet'

        return results


    def _run_ext_data (self):
        """
        version of Run for inter-process input (calling another model to obtain input)
        """

        # idata is a list of JSON from 1-n sources
        # the data usable for input must be listed in the ['meta']['main'] key

        # use first JSON to load common info like obj_nam, etc         
        obj_common=['label', 'decoration']

        # load object identifiers and decorators
        first_results = json.loads(self.idata[0])
        first_manifest = first_results['manifest']

        for item in first_manifest:
            if item['type'] in obj_common:
                item_key = item['key']
                self.results[item_key]=first_results[item_key]
                self.results['manifest'].append(item)

        # extract usable data from every source and add to 'combo' np.array
        combo_results = None
        combo_confidence = None
        var_nam = []
        conf_nam = []
        
        for ijson in self.idata:
            i_result = json.loads(ijson)
            i_manifest = i_result['manifest']
            i_meta = i_result['meta']

            for item in i_manifest:
                if item['type']=='result':
                    item_key = item['key']
                    if combo_results is None:  # for first element just copy
                        combo_results = np.array(i_result[item_key], dtype=np.float64)
                    else: # append laterally
                        combo_results = np.c_[combo_results, np.array(i_result[item_key], dtype=np.float64)]

                    var_nam.append(item_key+':'+i_meta['endpoint']+':'+str(i_meta['version']))

                if item['type']=='confidence':
                    item_key = item['key']
                    if combo_confidence is None:  # for first element just copy
                        combo_confidence = np.array(i_result[item_key], dtype=np.float64)
                    else: # append laterally
                        combo_confidence = np.c_[combo_confidence, np.array(i_result[item_key], dtype=np.float64)]

                    conf_nam.append(item_key+':'+i_meta['endpoint']+':'+str(i_meta['version']))

        utils.add_result (self.results, combo_results, 'xmatrix', 'X matrix', 'results', 'objs', 'Combined output from external sources')

        utils.add_result (self.results, combo_confidence, 'confidence', 'Confidence', 'confidence', 'objs', 'Combined confidence from external sources')

        utils.add_result (self.results, var_nam, 'var_nam', 'Var. names', 'method', 'vars', 'Variable names from external sources')

        utils.add_result (self.results, conf_nam, 'conf_nam', 'Conf. names', 'method', 'vars', 'Confidence indexes from external sources')


        return 


    def run (self):
        """         
        Process input file to obtain metadata (size, type, number of objects, name of objects, etc.) as well
        as for generating MD
            
        The results are saved in a MD5 stamped pickle, to avoid recomputing model input from the same input
        file
        
        This methods supports multiprocessing, splitting original files in a chunck per CPU        
        """

        # check for the presence of a valid pickle file
        if self.load():
            return self.results

        input_type = self.parameters['input_type'] 

        # processing for molecular input (for now an SDFile)
        if (input_type== 'molecule'):
            self._run_molecule()

        # processing for non-molecular input (not implemented)
        elif (input_type == 'data'):
            self._run_data()

        # processing for external data 
        elif (input_type == 'ext_data'):
            self._run_ext_data()

        else:
            self.results['error']='unknown input data format'

        # save in a pickle file stamped with MD5 hash of file and control
        if not 'error' in self.results:
            self.save ()

        return self.results
