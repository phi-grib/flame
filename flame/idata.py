#! -*- coding: utf-8 -*-

# Description    Flame Idata class
#
# Authors:       Manuel Pastor (manuel.pastor@upf.edu)
#
# Copyright 2018 Manuel Pastor
#
# This file is part of Flame
#
# Flame is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation version 3.
#
# Flame is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Flame. If not, see <http://www.gnu.org/licenses/>.

import os
import sys
import pickle
import shutil
import tempfile
# import multiprocessing as mp
from joblib import Parallel, delayed
import pathlib

import numpy as np
from rdkit import Chem

from standardiser import standardise

import flame.chem.sdfileutils as sdfutils
import flame.chem.compute_md as computeMD
import flame.chem.convert_3d as convert3D

from flame.util import utils, get_logger, supress_log

LOG = get_logger(__name__)


class Idata:

    def __init__(self, parameters, conveyor, input_source: str):
        """
        Input data class to standarize mol inputs

        Parameters
        ----------
        parameters: dict
            dict with model parameters

        conveyor: class
            main class to store the workflow results

        input_source: str
            SDF file with the molecules to use as training or predict        

        """

        # parameters and conveyor should have been initialized by the
        # parent class calling idata
        self.param = parameters
        self.conveyor = conveyor

        # self.format can inform if we are running in ghost mode
        # as part of an ensemble (low ensemble models)
        self.format = self.param.getVal('output_format')

        # path for temp files (fallback default)
        self.dest_path = '.'

        # add metainformation
        self.conveyor.addMeta('endpoint',self.param.getVal('endpoint'))
        self.conveyor.addMeta('version',self.param.getVal('version'))
        self.conveyor.addMeta('quantitative',self.param.getVal('quantitative'))
        
        input_type = self.param.getVal('input_type')
        self.conveyor.addMeta('input_type',input_type)


        # in case of top ensemble models...
        if input_type == 'model_ensemble':
            self.idata = input_source
            self.ifile = None
            randomName = 'flame-'+utils.id_generator()
            self.dest_path = os.path.join(tempfile.gettempdir(), randomName)

            #analyze first result to get the name of the input file
            ifile = 'ensemble input'
            try:
                ifile = input_source[0].getMeta('input_file')
            except:
                pass

            self.conveyor.addMeta('input_file',ifile)

        else:
            self.idata = None
            self.ifile = input_source
            self.dest_path = os.path.dirname(self.ifile)

            self.conveyor.addMeta('input_file',self.ifile)


    def captureStdError (self, status):
        '''
        This function captures errors sent by RDKit to stderror and adds them
        to a file called errorRDKit.log

        It is not active unless the parameter 'verbose error' is set to False
        '''

        if self.param.getVal('verbose_error'):
            return

        if status:
            # When running from a ipython notebooks fileno() breaks
            # Handle the exception notifying the issue.
            LOG.disabled = True
            try:
                self.stderr_fileno = sys.stderr.fileno()  # saves current syserr
                self.stderr_save = os.dup(self.stderr_fileno)

                # open a specific RDKit log file
                with open('errorRDKit.log', 'w') as self.stderr_fd:
                    os.dup2(self.stderr_fd.fileno(), self.stderr_fileno)
            except:
                pass
        else:                
            # Handle the exception due to undefined variable when
            # previous catch fails to define syserr
            LOG.disabled = False
            try:
                # close the RDKit log
                self.stderr_fd.close()  
                # restore old syserr                   
                os.dup2(self.stderr_save, self.stderr_fileno)   
            except:
                pass

    def extractInformation(self, ifile):
        '''
        Extracts molecule names, biological anotations and experimental values
        from an SDFile.

        All this information is added to the results using method utils.add_result,
        so they are also inserted into the results manifest.
        '''

        # Initiate a RDKit SDFile iterator to process the molecules one by one
        try:
            suppl = Chem.SDMolSupplier(ifile, sanitize=True)
            LOG.debug(f'mol supplier created from {ifile}')
        except Exception as e:
            LOG.debug('Unable to create mol supplier with the exception: '
                      f'{e}')
            self.conveyor.setError(f'unable to open {ifile}. {e}')
            return

        #Raise error if SDF is empty
        if len(suppl) == 0:
            LOG.debug(f'Input file {ifile} is empty')
            self.conveyor.setError(f'Input file {ifile} is empty')
            return

        # Initate lists which will contain the extracted values
        obj_nam = []
        obj_id  = []
        obj_bio = []
        obj_exp = []
        obj_cmp = []
        obj_sml = []
        success_list = []
        obj_num = 0

        # Iterate for every molecule inside the SDFile
        for mol in suppl:

            # Do not try to process molecules not recognised by RDKit.
            # They will be removed at the pre-normalization step, which is
            # compulsory for every molecule
            if mol is None:
                LOG.error(f'(@extractInformaton) Unable to process molecule #{obj_num+1}'
                          f' in file {ifile}')
                continue

            # extract the molecule name, using a sdfileutils algorithm 
            name = sdfutils.getName(
                mol, count=obj_num, field=self.param.getVal('SDFile_name'))

            # extracts molecule ID value, if any.
            idv = ''    
            if self.param.getVal('SDFile_id') is not None:
                if isinstance (self.param.getVal('SDFile_id'),str):
                    idv = sdfutils.getStr(mol, self.param.getVal('SDFile_id'))

            # extracts complementary information, if any.
            cmp = ''    
            if self.param.getVal('SDFile_complementary') is not None:
                if isinstance (self.param.getVal('SDFile_complementary'),str):
                    cmp = sdfutils.getStr(mol, self.param.getVal('SDFile_complementary'))

            # extracts biological information (activity) which is used as dependent variable
            # for the model training and is provided as a prediction for new compounds
            bio = None
            if self.param.getVal('SDFile_activity') is not None:
                bio = sdfutils.getVal(mol, self.param.getVal('SDFile_activity'))
            
            # extracts experimental information, if any.
            # note that experimental information is used only in prediction, as a value
            # which overrides any model predicted value
            exp = None    
            if self.param.getVal('SDFile_experimental') is not None:
                if isinstance (self.param.getVal('SDFile_experimental'),str):
                    exp = sdfutils.getVal(mol, self.param.getVal('SDFile_experimental'))

            # generates a SMILES
            sml = None
            try:
                sml = Chem.MolToSmiles(mol)
            except Exception as e:
                LOG.error('while converting mol to smiles'
                          f' an exception has ocurred: {e}')

            # assigns the information extracted from the SDFile to the corresponding lists
            obj_nam.append(name)
            obj_sml.append(sml)
            obj_id.append(idv)
            obj_cmp.append(cmp)
            obj_bio.append(bio)
            obj_exp.append(exp)

            success_list.append(True)
            obj_num += 1

        # # Insert in metadata the name of the input file
        # self.conveyor.addMeta('input_file',ifile)

        # Insert the values as lists in 'results' using an utility function

        self.conveyor.addVal(obj_num, 'obj_num', 'Num mol',
                         'method', 'single',
                         'Number of molecules present in the input file')

        self.conveyor.addVal(obj_nam, 'obj_nam', 'Mol name',
                         'label', 'objs',
                         'Name of the molecule, as present in the input file')

        self.conveyor.addVal(obj_sml, 'SMILES', 'SMILES',
                         'smiles', 'objs',
                         'Structure of the molecule in SMILES format')

        if not utils.is_string_empty(obj_id):
            self.conveyor.addVal(obj_id, 'obj_id', 'Mol id',
                            'label', 'objs',
                            'ID of the molecule, as present in the input file')

        if not utils.is_string_empty(obj_cmp):
            self.conveyor.addVal(obj_cmp, 'complementary', 'Complem.',
                            'method', 'objs',
                            'Complementary anotation present in the input file')

        if not utils.is_empty(obj_bio):
            self.conveyor.addVal(np.array(obj_bio, dtype=np.float64),
                             'ymatrix', 'Activity',
                             'decoration', 'objs',
                             'Biological anotation to be predicted by the model')

        if not utils.is_empty(obj_exp):
            self.conveyor.addVal(np.array(obj_exp, dtype=np.float64),
                             'experim', 'Experim.',
                             'method', 'objs',
                             'Experimental anotation present in the input file')

        LOG.debug(f'processed {obj_num} molecules'
                  f' from a supplier of {len(suppl)} without issues')
        
        return success_list

    def normalize(self, ifile, method):
        '''
        Generates a simplified SDFile with MolBlock and an internal ID for
        further processing

        Note that this method is applied to every molecule and that it removes
        mol blocks in the input SDFile not able to generate a valid mol

        Also, when defined in control, applies chemical standardization
        protocols, like the one provided by Francis Atkinson (EBI),
        accessible from:

            https://github.com/flatkinson/standardiser

        Returns a tuple containing the result of the method and (if True)
        the name of the output molecule and an error message otherwyse

        '''
        
        success_list = [True for i in range(sdfutils.count_mols(ifile))]
        
        if not method :
            method = ''

        LOG.info(f'Normalizing structures with method: {method}')
        try:
            suppl = Chem.SDMolSupplier(ifile)
            LOG.debug(f'mol supplier created from {ifile}')
        except Exception as e:
            LOG.error('Unable to create mol supplier with the exception: '
                      f'{e}')
            return False, 'Error at processing input file for standardizing structures'

        # Raise error if SDF is empty
        if len(suppl) == 0:
            LOG.debug(f'Input file {ifile} is empty')
            self.conveyor.setError(f'Input file {ifile} is empty')
            return

        filename, fileext = os.path.splitext(ifile)
        ofile = filename + '_std' + fileext
        LOG.debug(f'writing standarized molecules to {ofile}')
        with open(ofile, 'w') as fo:
            mcount = 0
            # merror = 0
            for m in suppl:

                # molecule not recognised by RDKit
                if m is None:
                    LOG.error('Unable to process molecule'
                              f' #{mcount+1} in {ifile}')
                    continue

                name = sdfutils.getName(m, count=mcount,
                                    field=self.param.getVal('SDFile_name'))

                parent = None

                if 'standardize' in method:
                    try:

                        parent = standardise.run(Chem.MolToMolBlock(m))

                    except standardise.StandardiseException as e:

                        if e.name == "no_non_salt":
                            # very commong warning, use parent mol and proceed
                            LOG.debug(f'"No non salt error" found. Skiped standardize for mol'
                                    f' #{mcount} {name}')
                            parent = Chem.MolToMolBlock(m)
                        else:
                            # serious issue, no parent was generated, use original mol
                            if (parent is None):
                                LOG.error(f'Critical standardize exception: {e}'
                                        f' when processing mol #{mcount} {name}. Skipping normalization')
                                parent = Chem.MolToMolBlock(m)
                            # minor isse, parent was generated, show a warning and proceed
                            else:
                                LOG.info(f'Standardize exception: {e}'
                                        f' when processing mol #{mcount} {name}. Normalization applied')
                        #return False, e.name

                    except Exception as e:
                        # this error means an execution error running standardizer
                        # the molecule is discarded and therefore the list of molecules must be updated 
                        LOG.error(f'Critical standardize execution exception {e}'
                                    f' when processing mol #{mcount} {name}. Discarding molecule')
                        success_list[mcount]=False
                        mcount += 1
                        continue

                elif 'chEMBL' in method:
                    # Get allowed penalty score from parameters
                    score = self.param.getDict('normalize_settings')['score']

                    from chembl_structure_pipeline import standardizer as embl
                    from chembl_structure_pipeline import checker
                    
                    try:
                        parent = embl.standardize_molblock(Chem.MolToMolBlock(m))
                        issues = checker.check_molblock(Chem.MolToMolBlock(m))
                        if len(issues) > 0:
                            if issues[0][0] > score:
                                success_list[mcount]=False
                                mcount += 1
                                continue

                    except Exception as e:
                        # this error means an execution error running standardizer
                        # the molecule is discarded and therefore the list of molecules must be updated 
                        LOG.error(f'Critical standardize execution exception {e}'
                                    f' when processing mol #{mcount} {name}. Discarding molecule')
                        success_list[mcount]=False
                        mcount += 1
                        continue

                else:
                    try:
                        parent = Chem.MolToMolBlock(m)
                    except Exception as e:
                        # this error means an severe error when processing the molecule
                        # the molecule is discarded and therefore the list of molecules must be updated 
                        LOG.error(f'Critical molecule processing exception {e}'
                                    f' when processing mol #{mcount} {name}. Discarding molecule')
                        success_list[mcount]=False
                        mcount += 1
                        continue

                # in any case, write parent plus internal ID (flameID)
                fo.write(parent)

                # *** discarded method to control errors ****
                # flameID = 'fl%0.10d' % mcount
                # fo.write('>  <flameID>\n'+flameID+'\n\n')

                mcount += 1

                # terminator
                fo.write('$$$$\n')

        return success_list, ofile


    def ionize(self, ifile, method):
        '''
        Adjust the ionization status of the molecular structure,
        using a given pH.
        '''
        
        success_list = [True for i in range(sdfutils.count_mols(ifile))]
    
        if not method:
            return success_list, ifile

        else:
            LOG.debug ('ionize called, but no method implemented so far')
            # methods here

        return success_list, ifile

    def convert3D(self, ifile, method):
        '''
        Assigns 3D structures to the molecular structures provided as input.
        '''

        success_list = [True for i in range(sdfutils.count_mols(ifile))]

        if not method:
            return success_list, ifile
        
        if method == 'ETKDG':
            success_list, ofile = convert3D._ETKDG(ifile)
        else:
            LOG.warning(f'Value of parameter "convert3D_method" not recognized: {method}. No 3D conversion applied')
            ofile = ifile

        return success_list, ofile

    def computeMD_custom(self, ifile):
        '''
        Empty method for computing molecular descriptors.

        ifile is a molecular file in SDFile format.

        output is boolean and a dictionary of:
            - matrix: xmatrix (nparray np.float64)
            - names:  list of variable names (str)
            - success_arr: list of booleans indicating if the computation succeeded for each molecule
            - fingerprint_index: list of booleans indicating if the md is a binary (fingerprint) or a float  

        example:    return True, combined
        '''
        return False, 'not implemented'

    def computeMD(self, ifile: str, methods: list):
        '''
        Uses the molecular structures for computing an array
        of values (int or float).

        input is the name of a molecule or a series of molecules and a label
        of the methods 
        
        output is boolean and a dictionary of:
            - matrix: xmatrix (nparray np.float64)
            - names:  list of variable names (str)
            - success_arr: list of booleans indicating if the computation succeeded for each molecule
            - fingerprint_index: list of booleans indicating if the md is a binary (fingerprint) or a float  

        example:    return True, combined

        '''

        LOG.info(f'Computing molecular descriptors with methods: {methods}')
        
        # Load descriptor settings

        md_settings = self.param.getDict('MD_settings')

        registered_methods = dict([('RDKit_properties', computeMD._RDKit_properties),
                                   ('morganFP', computeMD._RDKit_morganFPS),
                                   ('substructureFP', computeMD._RDKit_patternFPS),
                                   ('rdkFP', computeMD._RDKit_rdkFPS),
                                   ('RDKit_md', computeMD._RDKit_descriptors),
                                   ('custom', self.computeMD_custom)])

        fingerprint_list = ['rdkFP','morganFP','substructreFP']  # update with any other fingerprint method

        # check if input methods are members of registered methods
        if not all(m in registered_methods for m in methods):
            # find the non member methods
            no_recog_meth = [m for m in methods if m not in registered_methods]

            if len(no_recog_meth) == len(methods):
                # then no md method is correct... so error
                return False, f'Methods {no_recog_meth} not recognized. No valid method found.'

            # remove bad methods
            methods = [m for m in methods if m not in no_recog_meth]

        is_empty = True
        shape = []

        # sort methods to avoid non-reproducible results when blocks are combined in diverse order
        methods.sort()

        combined = {}
        for method in methods:
            # success, results = registered_methods[method](ifile)
            success, results = registered_methods[method](ifile, **md_settings)

            if not success:  # if computing returns False in status
                return success, results

            is_fingerprint = method in fingerprint_list
            nobj, nvarx = np.shape(results['matrix'])

            if is_empty:  # first md computed, just copy

                combined['matrix'] = results['matrix']  # np.array of values
                combined['names'] = results['names']  # list of variable names
                combined['fingerprint_index'] = [is_fingerprint for i in range(nvarx)]
                combined['success_arr'] = results['success_arr'] # list of true/false

                shape = np.shape(combined['matrix'])

                is_empty = False

            else:  # append laterally
                ishape = np.shape(results['matrix'])

                # for 2D arrays, shape[0] is the number of objects
                if (len(ishape) > 1) and ishape[0] != shape[0]:

                    # combination
                    LOG.warning(f'Number of objects processed by {method} '
                              'does not match those computed by other methods and will be skipped')

                    self.conveyor.setWarning(f'Number of objects processed by {method} '
                              'does not match those computed by other methods and will be skipped')          
                    continue

                combined['matrix'] = np.hstack((combined['matrix'], results['matrix']))
                combined['names'].extend(results['names'])
                combined['fingerprint_index'] += [is_fingerprint for i in range(nvarx)]

                # combine sucess results into one list with AND
                # All results must be True to get True
                # scc stands for success
                # new_sc = [scc and results['success_arr'][i]for i, scc in enumerate(combined['success_arr'])]
                # combined['success_arr'] = new_sc

                for i, sci in enumerate(results['success_arr']):
                    if not sci:
                        combined['success_arr'][i] = False

            
        # delete all objects for which success is not true but 
        # IN REVERSE order, so the index if the lines to remove
        # is not affected by the removal
        
        for i, scc in reversed(list(enumerate(combined['success_arr']))):
            if not scc:
                combined['matrix'] = np.delete(combined['matrix'],i,axis=0) 

        return True, combined


    def consolidate(self, results_tuple, nobj):
        '''
        Mix the results obtained by multiple CPUs into a single result file.
        '''
        LOG.info('Concatenating results from '
                 f'{len(nobj)} jobs with shapes {nobj}')

        first = True
        combined = {}
        shape = []

        for iresults in results_tuple:

            # iresults_tuple is a tupla of Boolean (iresults[0])
            #  and results (iresults[1])
            if iresults[0] == False:
                return False, iresults[1]

            # internal is a tupla of 4 elements
            #  (xmatrix, var_nam, success_list, fingerprint_index)
            internal = iresults[1]

            # copy X matrix to imatrix because it is referenced a lot
            ixmatrix = internal['matrix']

            # check that the type of ixmatrix is correct (np.ndarray)
            if not isinstance(ixmatrix, np.ndarray):
                LOG.error('Results type in consolidate must be `np.ndarray`.'
                          f' Found: {type(ixmatrix)}')
                return False, "unknown results type in consolidate"

            if first: # only for the first element of the loop
                combined['matrix'] = ixmatrix
                combined['names'] = internal['names']
                combined['success_arr'] = internal['success_arr']
                combined['fingerprint_index'] = internal['fingerprint_index']

                shape = np.shape(ixmatrix)
                first = False

            else:
                ishape = np.shape(ixmatrix)

                if len(shape) > 1 and len(ishape) > 1:
                    # for bidimensional arrays, num_var is shape[1]
                    if shape[1] != ishape[1]:
                        LOG.error('Impossible to concat arrays'
                                  f'with shape {shape} and {ishape}')

                        return False, "inconsistent number of variables"
                else:
                    # for vectors obtained with a single object, numvar is shape[0]
                    if shape[0] != ishape[0]:
                        LOG.error('Impossible to concat arrays'
                                  f' with shape {shape} and {ishape}')

                        return False, "inconsistent number of variables"

                combined['matrix'] = np.vstack((combined['matrix'], ixmatrix))
                combined['success_arr'] += internal['success_arr']

        return True, combined

    def save(self):
        '''
        Saves the results in serialized form, together with the MD5 signature
        of the control class and the input file.
        '''
        #######################################################
        # uncomment to avoid saving results
        # print ('*** save commented for debugging ***')
        # return
        #######################################################

        # if this is the top model or an ensemble, exit
        if self.param.getVal('input_type') == 'model_ensemble':
            return

        # if this is a low model or an ensemble, exit
        if 'ghost' in self.format:
            return 

        md5_parameters = self.param.getVal('md5')
        md5_input = utils.md5sum(self.ifile)  # run md5 in self.ifile

        try:
            with open(os.path.join(self.dest_path, 'data.pkl'), 'wb') as fo:

                pickle.dump(md5_parameters, fo)
                pickle.dump(md5_input, fo)

                self.conveyor.save (fo)

        except Exception as e:
            LOG.error(f"Can't serialize descriptors because of exception: {e}")


    def load(self):
        '''
        Loads the results in serialized form, together with the MD5 signature
        of the control class and the input file.
        '''

        # return False

        # if this is the top model or an ensemble, exit
        if self.param.getVal('input_type') == 'model_ensemble':
            return False

        # if this is a low model or an ensemble, exit
        if 'ghost' in self.format:
            return 

        try:
            # debug option when you want to compute MD allways
            # return False
            picklfile = os.path.join(self.dest_path, 'data.pkl')
            if not os.path.isfile(picklfile):
                return False

            with open(picklfile, 'rb') as fi:
                # check that MD5 hash of the relevant parameters is the same that the stored 
                # hash value in the pickle
                md5_parameters = pickle.load(fi)
                if md5_parameters != self.param.getVal('md5'):
                    return False

                # check that MD5 hash of the input file is the same that the stored 
                # hash value in the pickle
                md5_input = pickle.load(fi)
                # print (self.ifile, md5_input, utils.md5sum(self.ifile))
                if md5_input != utils.md5sum(self.ifile):
                    return False

                # preserve original origin tag and modelID, save it
                origin = self.conveyor.getOrigin()
                modelID = self.conveyor.getMeta('modelID')
                # warning = self.conveyor.getWarningMessage()

                success, message = self.conveyor.load(fi)
                
                if not success:
                    LOG.error(f'Failed to load pickle file with error: "{message}"')
                    return False

                # presenve original origin tag, apply it
                self.conveyor.setOrigin(origin)
                self.conveyor.addMeta('modelID', modelID)
                # if warning is not None:
                #     self.conveyor.setWarningMessage(warning)

        except Exception as e:
            self.conveyor.setError('Error loading pickle with exception: {}'.format(e))
            LOG.error('Error loading pickle with exception: {}'.format(e))
            return False

        LOG.info(f'Recycling data from {picklfile}')

        return True

    #@supress_log(logger=LOG)
    def workflow_objects(self, input_file):
        '''
        Executes in sequence methods required to generate MD,
        starting from a single molecular file.

        input : ifile, a molecular file in SDFile format
        output: results is a numpy bidimensional array containing MD
        '''

        success_list = []

        # split in single molecule pieces
        num_mol = sdfutils.count_mols(input_file)
        success, results = sdfutils.split_SDFile(input_file, num_mol)

        if not success:
            return success, results

        file_list = results[0]
        file_size = results[1]

        # check if any of the molecules is empty
        for fsize in file_size:
            success_list.append(fsize == 1)

        combined = {}
        first_mol = True

        for i, ifile in enumerate(file_list):

            if not success_list[i]:   # molecule was empty, do not process
                LOG.error(f'Molecule {i+1} in {ifile} is empty, skiping...')
                continue

            success, results = self.workflow_series(ifile)

            # since the workflow was run for a single molecule, results[2] is ignored, because it must match
            # the value in success
            success_list[i] = success

            if not success:           # failed in the workflow
                LOG.error(f'Workflow failed for molecule #{str(i+1)}'
                          f' in file {input_file}')
                continue

            if first_mol:  # we extract common features from first molecule only
                combined['matrix']  = results['matrix']
                combined['names']   = results['names'] # variable names
                combined['fingerprint_index'] = results['fingerprint_index'] # fingerprint index
                first_mol = False
                
                shape = np.shape(combined['matrix'])
            else:  # append laterally
                ishape = np.shape(results['matrix'])

                # for 2D arrays, shape[0] is the number of objects
                if (len(ishape) > 1) and ishape[0] != shape[0]:
                    LOG.warning(f'MD length for molecule #{str(i+1)} in file'
                                f' {input_file} does not match the MD length of the first molecule')
                    success_list[i] = False
                    continue

                combined['matrix'] = np.vstack((combined['matrix'], results['matrix']))

        combined['success_arr'] = success_list
        return True, combined


    def updateMolIndex (self, mol_index, success_list):

        # if success list is True for all elements
        if len(success_list) == sum(i for i in success_list):
            return True, mol_index

        # update mol_index
        j = 0
        for i in range(len(mol_index)):
            if not mol_index[i]:
                continue
            if not success_list[j]:
                mol_index[i]=False
            j=j+1

        # if no molecules left return False
        if sum(i for i in mol_index) == 0:
            return False, 'no molecules left'

        return True, mol_index

    def workflow_series(self, input_file):
        '''
        Executes in sequence methods required to generate MD,
        starting from a single molecular file

        input : ifile, a molecular file in SDFile format
        output is boolean and a dictionary of:
            - matrix: xmatrix (nparray np.float64)
            - names:  list of variable names (str)
            - success_arr: list of booleans indicating if the computation succeeded for each molecule
            - fingerprint_index: list of booleans indicating if the md is a binary (fingerprint) or a float  

        example:    return True, combined

        '''

        mol_index = [True for i in range(sdfutils.count_mols(input_file))]

        ###
        # 1. normalize
        ###
        success_list, output_normalize_file = self.normalize(
            input_file, self.param.getVal('normalize_method'))
        success, mol_index = self.updateMolIndex(mol_index, success_list)

        if not success:
            return False, 'failed to normalize '+input_file

        ###
        # 2. ionize
        ###
        success_list, output_ionize_file = self.ionize(
            output_normalize_file, self.param.getVal('ionize_method'))
        success, mol_index = self.updateMolIndex(mol_index, success_list)

        if not success:
            return False, 'failed to ionize '+input_file

        ###
        # 3. convert3D
        ###
        success_list, output_convert3D_file = self.convert3D(
            output_ionize_file, self.param.getVal('convert3D_method'))
        success, mol_index = self.updateMolIndex(mol_index, success_list)

        if not success:
            return False, 'failed to convert 3D '+input_file

        ###
        # 4. compute MD
        ###
        success, results = self.computeMD(output_convert3D_file, self.param.getVal('computeMD_method'))

        if not success:
            return False, results

        # update success results using updateMolIndex function
        success, results['success_arr'] = self.updateMolIndex(mol_index, results['success_arr'])

        return success, results

    def ammend_objects(self, inform, workflow) -> None:
        '''
        The arguments inform and workflow are lists of booleans describing
        when the objects were successfully informed (inform)
        or completed the workflow.

        This functions is called only when a disagreement if found, revealing
        that any object failed to be processed, and that the xmatrix will
        have less rows than expected.

        The function ammends all keys describing objects,
        removing those appearing as "false" in workflow and not in inform.
        '''

        # list objects to remove
        remove_index = []
        warning_list = []
        obj_num = 0
        obj_nam = self.conveyor.getVal('obj_nam')

        for i in range(len(workflow)):
            if inform[i] and not workflow[i]:
                remove_index.append(i)
                warning_list.append(obj_nam[i])
            if inform[i] and workflow[i]:
                obj_num += 1

        LOG.debug('(@ammend_objects) going to remove these'
                  'indexes from manifest: {}'.format(remove_index))
        self.conveyor.setVal('obj_num', obj_num)

        objkeys = self.conveyor.objectKeys()
        for ikey in objkeys: 
            ilist = self.conveyor.getVal(ikey)

            # keys are experim or ymatrix are numpy arrays
            # if 'numpy.ndarray' in str(type(ilist)):
            if isinstance(ilist, np.ndarray):
                ilist = np.delete(ilist, remove_index)
            # other keys are regular list
            else:
                for i in sorted(remove_index, reverse=True):
                    del ilist[i]

            self.conveyor.setVal(ikey, ilist)

        message = f'Failed to process {len(warning_list)} molecules : {str(warning_list)}'
        self.conveyor.setWarning(message)

        LOG.warning(message)
        LOG.warning('Will show results for the rest of the series...')

        return

    def _run_molecule(self):
        '''
        version of Run for molecular input

        '''

        # extract useful information from file

        success_inform = self.extractInformation(self.ifile)
        if self.conveyor.getError():
            return

        nobj = self.conveyor.getVal('obj_num')
        ncpu = min(nobj, self.param.getVal('numCPUs'))

        # copy the input file to a temp file which will be cleaned at the end
        temp_path = tempfile.mkdtemp()
        shutil.copy(self.ifile, temp_path)
        lfile = os.path.join(temp_path, os.path.basename(self.ifile))

        # Execute the workflow in 1 or n CPUs
        
        if ncpu > 1:
            LOG.debug('Entering molecule workflow for {} cpus'.format(ncpu))
            success, results = sdfutils.split_SDFile(lfile, ncpu)

            if not success:
                self.conveyor.setError('Unable to split input molecule')
                return

            split_files_names = results[0]
            split_files_sizes = results[1]

            # pool = mp.Pool(ncpu)
            # if self.param.getVal('mol_batch') == 'series':
            #     results_tuple = pool.map(self.workflow_series, split_files_names)
            # else:
            #     results_tuple = pool.map(self.workflow_objects, split_files_names)

            if self.param.getVal('mol_batch') == 'series':
                results_tuple = Parallel(n_jobs=ncpu)(delayed(self.workflow_series)(split_files_names[i]) for i in range(ncpu))
            else:
                results_tuple = Parallel(n_jobs=ncpu)(delayed(self.workflow_objects)(split_files_names[i]) for i in range(ncpu))

            success, results = self.consolidate(results_tuple, split_files_sizes)

        else:

            if self.param.getVal('mol_batch') == 'series':
                success, results = self.workflow_series(lfile)
            else:
                success, results = self.workflow_objects(lfile)

        # series processing (1 or n CPUs) can produce a success == False if
        # any of the series/pieces contains an error. Abort the processing...
        if not success:
            self.conveyor.setError('error in workflow processing')
            return

        # check if any molecule failed to complete the workflow and then
        # ammend object annotations in self.conveyor
        success_workflow = results['success_arr']

        if len(success_inform) != len(success_workflow):

            LOG.error('shape mismatch of informed and workflow results:'
                      f' ({len(success_inform), len(success_workflow)})'
                      ' This is because some molecules failed during'
                      ' the standarization or descriptors computations.')

            self.conveyor.setError('number of molecules informed'
                                   ' and processed does not match')
            return

        # Check if molecules not informed succeded
        # to be complete MD generation.
        # This should never happen, because they
        # do not pass the normalization step

        for i, (inform, workflow) in enumerate(zip(success_inform,
                                                   success_workflow)):
            if workflow and not inform:

                LOG.critical(f'Molecule #{i} is `None` in Rdkit'
                             ' but appears to be processed. This means that'
                             ' there is a serious workflow issue and the '
                             ' molecule should be cured or eliminated.')

                self.conveyor.setError('Unknown error processing input file.'
                                       ' Probably the format is wrong or not supported')
                return

        # check if a molecule informed did not
        # succeed to complete MD generation
        for i, j in zip(success_inform, success_workflow):
            if i and not j:
                self.ammend_objects(success_inform, success_workflow)
                break

        # remove the temp directory with all the temp files inside
        shutil.rmtree(temp_path)

        #TODO: optional sanitization step, to check if the X matrix contains extreme values or variables
        # with unreasonable variances

        self.conveyor.addVal(results['matrix'], 'xmatrix', 'X matrix',
            'method', 'vars', 'Molecular descriptors')

        self.conveyor.addVal(results['names'], 'var_nam',
            'Var names', 'method', 'vars', 'Names of the X variables')

        self.conveyor.addVal(results['fingerprint_index'], 'fingerprint_index',
            'fingerprint index', 'method', 'vars', 'Index true for fingerprint variables')

        return

    def _run_data(self):
        '''
        version of Run for data input (TSV tabular format)
        '''
        if not os.path.isfile(self.ifile):
            self.conveyor.setError(f'{self.ifile} not found')
            return

        #  Reading TSV by hand
        with open(self.ifile, 'r') as fi:

            var_nam = []
            obj_nam = []
            smiles = []
            ymatrix = []

            hasObjNames = self.param.getVal('TSV_objnames')
            activity_param = self.param.getVal('TSV_activity')

            iSMILES = -1
            iActivity = -1

            for index, line in enumerate(fi):

                # FIRST LINE: read var names and generate a mask to speed up
                # the reading of MDs and populate names, SMILES and y's
                if index == 0:
                    var_nam = line.strip().split('\t')

                    # create mask
                    mask = np.ones(len(var_nam), dtype=np.int32)

                    # blind first column (names)
                    if hasObjNames:
                        mask[0] = 0
                    
                    # blind SMILES column 
                    if 'SMILES' in var_nam:
                        iSMILES = var_nam.index('SMILES')
                        mask[iSMILES] = 0

                    # blind activity column 
                    if activity_param in var_nam:
                        iActivity = var_nam.index(activity_param)
                        mask[iActivity] = 0

                    # assign names to X variables (md's)
                    md_nam = [ x for x, z in zip(var_nam, mask) if z==1 ]     

                # REST OF LINES: apply the mask and collect names, SMILES and y's
                # from predefined possitions
                else:
                    value_list = line.strip().split('\t')

                    try:
                        if hasObjNames:
                            obj_nam.append(value_list[0])

                        if iSMILES != -1:
                            smiles.append(value_list[iSMILES])

                        if iActivity != -1:
                            ymatrix.append(np.float(value_list[iActivity]))

                        # extract only the variables assumed to be md
                        masked = [ x for x, z in zip(value_list, mask) if z==1 ]
                        value_array = np.array(masked, dtype=np.float64)

                        if index == 1:  # for the fist row, just copy the value list to the xmatrix
                            xmatrix = value_array
                        else:
                            xmatrix = np.vstack((xmatrix, value_array))
                    except Exception as e:
                        self.conveyor.setError(f'Error in line {index+1}: '+str(e))
                        return
                        
        obj_num = index - 1  # the first line are variable names 
        LOG.debug('loaded TSV with shape {} '.format(xmatrix.shape))
        LOG.debug('creating ymatrix from column {}'.format(activity_param))
        
        if iActivity != -1:
            self.conveyor.addVal( np.array(ymatrix), 'ymatrix', 'Activity', 'decoration',
                             'objs', 'Biological anotation to be predicted by the model')

        self.conveyor.addVal( obj_num, 'obj_num', 'Num mol', 'method',
                         'single', 'Number of molecules present in the input file')

        self.conveyor.addVal( xmatrix, 'xmatrix',
                         'X matrix', 'method', 'vars', 'Molecular descriptors')

        self.conveyor.addVal( md_nam, 'var_nam', 'Var names',
                             'method', 'vars', 'Names of the X variables')

        if not hasObjNames:
            for i in range(obj_num):
                obj_nam.append('obj%.10f' % i)

        self.conveyor.addVal( obj_nam, 'obj_nam', 'Mol name', 'label',
                         'objs', 'Name of the molecule, as present in the input file')

        if len(smiles) > 0:
            self.conveyor.addVal( smiles, 'SMILES', 'SMILES',
                             'smiles', 'objs', 'Structure of the molecule in SMILES format')

        return

    def _run_model_ensemble(self):
        '''
        version of Run for inter-process input
        (calling another model to obtain input)
        '''

        self.param.setVal('computeMD_method', [])
        
        # idata is a list of conveyor from n sources
        # the data usable for input must be listed in the ['meta']['main'] key

        # get input file name from conveyor, as defined in the constructor
        ifile = self.conveyor.getMeta('input_file')


        # call extractInformation to obtain names, activities, smiles, id, etc.
        success_inform = self.extractInformation(ifile)

        if not success_inform or self.conveyor.getError():
            return

        # obj_common = ['label', 'decoration', 'smiles']
        # for item in first_manifest:
        #     if item['type'] in obj_common:  # for elements of type label or decoration
        #         self.conveyor.addVal(first_results[item['key']], 
        #                              item['key'], 
        #                              item['label'], 
        #                              item['type'],
        #                              item['dimension'],
        #                              item['description']
        #                             )

        # extract usable data from every source and add to 'combo' np.array
        combined_md = None
        combined_ci = None
        combined_md_names = []
        combined_ci_names = []
        combined_confidence = []

        num_obj = self.conveyor.getVal ('obj_num')
        num_conformal = 0

        for i_result in self.idata:

            # predictions
            i_md = i_result.getVal('values')

            i_md_size = len(i_md)
            if i_md_size != num_obj:
                self.conveyor.setError('the number of results produced by the first model is inconsistent with the number of molecules recognized in the input file')
                return

            # re-scale quantitatives as -1 for negative and 0 for uncertains
            # this facilitates the use of cualitative input by methods like RF, PLSDA, SVM without
            # discarding objects containing a few uncertain values
            if not i_result.getMeta('quantitative'):
                for i in range(i_md_size):
                    ival = i_md[i]
                    if ival == 0:
                        i_md[i] = -1
                    elif ival == -1:
                        i_md[i] = 0
            
            if combined_md is None:  # for first element just copy
                combined_md = np.array(i_md, dtype=np.float64)
            else:
                combined_md = np.c_[combined_md, np.array(i_md, dtype=np.float64)]

            combined_md_names.append('values'+':'+i_result.getMeta('endpoint')+':'+str(i_result.getMeta('version')))
            combined_confidence.append(i_result.getVal('confidence'))

            # confidence values and names

            # we don't know if the low level models are qualitative or quantitative or a mixture of both
            # simply check the presence of the CI keys and add whatever is present

            i_low = i_result.getVal('lower_limit')
            i_up  = i_result.getVal('upper_limit')
            if i_up is not None and i_low is not None:

                num_conformal += 1

                if combined_ci is None:  # for first element just copy
                    combined_ci = np.array(i_low, dtype=np.float64)
                    combined_ci = np.column_stack((combined_ci, i_up))
                else:  # append laterally
                    combined_ci = np.column_stack((combined_ci, i_low))
                    combined_ci = np.column_stack((combined_ci, i_up))

                combined_ci_names.append(
                    'lower_limit'+':'+i_result.getMeta('endpoint')+':'+str(i_result.getMeta('version')))
                combined_ci_names.append(
                    'upper_limit'+':'+i_result.getMeta('endpoint')+':'+str(i_result.getMeta('version')))
                # confidence values and names

            i_c0 = i_result.getVal('c0')
            i_c1 = i_result.getVal('c1')
            if i_c0 is not None and i_c1 is not None:

                num_conformal += 1

                if combined_ci is None:  # for first element just copy
                    combined_ci = np.array(i_c0, dtype=np.float64)
                    combined_ci = np.column_stack((combined_ci, i_c1))
                else:  # append laterally
                    combined_ci = np.column_stack((combined_ci, i_c0))
                    combined_ci = np.column_stack((combined_ci, i_c1))

                combined_ci_names.append(
                    'c0'+':'+i_result.getMeta('endpoint')+':'+str(i_result.getMeta('version')))
                combined_ci_names.append(
                    'c1'+':'+i_result.getMeta('endpoint')+':'+str(i_result.getMeta('version')))

        self.conveyor.addVal( combined_md, 'xmatrix', 'X matrix',
                         'results', 'vars', 'Combined output from external sources')

        self.conveyor.addVal( combined_md_names, 'var_nam', 'Var. names',
                         'method', 'vars', 'Variable names from external sources')

        if num_conformal == len (self.idata):  # add this info only if ALL inputs are conformal
            if combined_ci is not None:
                self.conveyor.addVal( combined_ci, 'ensemble_ci', 'Ensemble Confidence',
                                'method', 'objs', 'Combined confidence from external sources')

            if len(combined_ci_names) > 0:
                self.conveyor.addVal( combined_ci_names, 'ensemble_ci_names', 'Ensemble Conf. names',
                                'method', 'vars', 'Confidence indexes from external sources')

            if len(combined_confidence) > 0:
                self.conveyor.addVal( combined_confidence, 'ensemble_confidence', 'Ensemble Confidence',
                                'method', 'vars', 'Confidence from external sources')
        
        return

    def run(self):
        '''
        Process input file to obtain metadata (size, type, number of objects,
        name of objects, etc.) as well as for generating MD.

        The results are saved in a MD5 stamped pickle, to avoid recomputing
        model input from the same input file.

        This methods supports multiprocessing, splitting original files in a
        chunck per CPU        
        '''

        input_type = self.param.getVal('input_type')
        LOG.info('Running with input type: {}'.format(input_type))

        if input_type != 'model_ensemble':

            # if the input file is not found return
            if not os.path.isfile(self.ifile):
                self.conveyor.setError(f'input data file {self.ifile} not found')
                return 

            # check for the presence of a valid pickle file
            if self.load():

                #print (self.conveyor.data)

                return 

        # processing for molecular input (for now an SDFile)
        if input_type == 'molecule':

            self.captureStdError(True)
            self._run_molecule()
            self.captureStdError(False)
                
        # processing for non-molecular input
        elif input_type == 'data':
            self._run_data()

        # processing for external data
        elif input_type == 'model_ensemble':
            self._run_model_ensemble()

        else:
            LOG.debug('Unknown input data format')
            self.conveyor.setError('Unknown input data format')

        # save in a pickle file stamped with MD5 hash of file and control
        if not self.conveyor.getError():
            self.save()

        return 
