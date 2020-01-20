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
import json
import tempfile
import multiprocessing as mp
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
        
        input_source: str
            SDF file with the molecules to use as training or predict        

        Returns
        -------
        # TODO: clear what this class returns

        """
        # control object defining the processing
        self.param = parameters
        self.conveyor = conveyor

        # path for temp files (fallback default)
        self.dest_path = '.'

        self.conveyor.addMeta('endpoint',self.param.getVal('endpoint'))
        self.conveyor.addMeta('version',self.param.getVal('version'))

        if self.param.getVal('input_type') == 'model_ensemble':
            LOG.debug('model_ensemble input type')
            self.idata = input_source
            self.ifile = None
            randomName = 'flame-'+utils.id_generator()
            self.dest_path = os.path.join(tempfile.gettempdir(), randomName)
        else:
            self.idate = None
            self.ifile = input_source
            self.dest_path = os.path.dirname(self.ifile)


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
            suppl = Chem.SDMolSupplier(ifile)
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
        obj_bio = []
        obj_exp = []
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
            obj_bio.append(bio)
            obj_exp.append(exp)
            obj_sml.append(sml)

            success_list.append(True)
            obj_num += 1

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

        LOG.info('Starting normalization...')
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

                else:
                    #LOG.info(f'Skipping normalization.')
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
        
        if method is 'ETKDG':
            success_list, ofile = convert3D._ETKDG(ifile)
        else:
            ofile = ifile

        return success_list, ofile

    def computeMD_custom(self, ifile):
        '''
        Empty method for computing molecular descriptors.

        ifile is a molecular file in SDFile format.

        returns a boolean anda a tupla of two elements:
        [0] xmatrix (nparray np.float64)
        [1] list of variable names (str)
        [2] list of booleans indicating if the computation succeeded for each molecule

        example:    return True, (xmatrix, md_nam, success_list)
        '''
        raise NotImplementedError
        #return False, 'not implemented'

    def computeMD(self, ifile: str, methods: list) -> (bool, (np.ndarray, list, list)):
        '''
        Uses the molecular structures for computing an array
        of values (int or float).

        input is the name of a molecule or a series of molecules and a label
        of the methods output is boolean anda a tupla of two elements:
        [0] xmatrix (nparray np.float64)
        [1] list of variable names (str)

        FIXIT
        '''
        LOG.info(f'Computing molecular descriptors with methods {methods}...')
        
        # Load descriptor settings

        md_settings = self.param.getDict('MD_settings')

        registered_methods = dict([('RDKit_properties', computeMD._RDKit_properties),
                                   ('morganFP', computeMD._RDKit_morganFPS),
                                   ('RDKit_md', computeMD._RDKit_descriptors),
                                #    ('padel', computeMD._padel_descriptors),
                                #    ('mordred', computeMD._mordred_descriptors),
                                   ('custom', self.computeMD_custom)])

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

        for method in methods:
            # success, results = registered_methods[method](ifile)
            success, results = registered_methods[method](ifile, **md_settings)

            if not success:  # if computing returns False in status
                return success, results

            if is_empty:  # first md computed, just copy

                combined_md = results['matrix']  # np.array of values
                combined_nm = results['names']  # list of variable names
                combined_sc = results['success_arr'] # list of true/false

                shape = np.shape(combined_md)

                is_empty = False

            else:  # append laterally
                ishape = np.shape(results['matrix'])

                # for 2D arrays, shape[0] is the number of objects
                if (len(ishape) > 1) and ishape[0] != shape[0]:

                    LOG.error(f'Number of objects processed by {method}'
                              'does not match those computed by other methods')
                    continue

                combined_md = np.hstack((combined_md, results['matrix']))
                combined_nm.extend(results['names'])

                # combine sucess results into one list with AND
                # All results must be True to get True
                # scc stands for success
                new_sc = [scc and results['success_arr'][i]
                          for i, scc in enumerate(combined_sc)]
                          
                combined_sc = new_sc

        return True, (combined_md, combined_nm, combined_sc)

   
    @staticmethod
    def _filter_matrix(matrix: np.ndarray, succes_list: list):
        """Filters matrix via boolean mask.

        The boolean mask is the logical AND combination of all the masks in
        `succes_list`.
        This way we get rid of molecules with NaNs or that have failed during
        supplier reading in any descriptor computation.

        Parameters
        ----------

        matrix: np.ndarray
            descriptors matrix that's going to be filtered

        succes_list: list
            list of array masks that will be used to filter 
            the descriptors matrix

        Returns
        -------

        np.ndarray
            Filtered matrix with the elements that have
            only `True` in succes_list arrays

        list
            the resultant succes list. `all()` combination of
            `succes_list` (param) arrays. The length must be same
            as the number of molecules.

        """
        # using all bcause of arbitrary list length
        filter_mask = np.all(succes_list, axis=0)
        n_filtered_mols = len(filter_mask) - sum(filter_mask)
        LOG.info(f'removed {n_filtered_mols} molecules of {len(filter_mask)}'
                 ' because of malformation or problems computing descriptors')

        if matrix.shape[0] != len(filter_mask):
            raise ValueError('Matrix and filter mask do not have the'
                             ' same shape on filter axis')

        filtered_matrix = matrix[filter_mask, :]
        return filtered_matrix, filter_mask.tolist()

    @staticmethod
    def _concat_descriptors_matrix(matrices: list) -> np.ndarray:
        """ Concatenates horizontaly an arbritary number of matrices.

        Used to concat multiple descriptors results into a one array.

        Parameters
        ----------

        matrices: list
            list of matrices (np.ndarrays) to concat horizontally

        Returns
        -------

        np.ndarray
            concatenated matrix of input matrices
        """
        # type check input
        if not all(isinstance(m, np.ndarray) for m in matrices):
            raise TypeError('input matrices must be numpy arrays')

        try:
            xmatrix = np.concatenate(matrices, axis=1)

            LOG.debug('concatenated matrices with shapes: '
                      f'{[m.shape for m in matrices]} into a'
                      f' matrix with shape {xmatrix.shape}')

        except ValueError as e:
            LOG.critical('Cannot concatenate matrix with different shapes: '
                         f'{[m.shape[0] for m in matrices]}')
            raise ValueError('Cannot concatenate matrix with different shapes: '
                             f'{[m.shape[0] for m in matrices]}')
        return xmatrix

    def consolidate(self, results, nobj):
        '''
        Mix the results obtained by multiple CPUs into a single result file.
        '''
        LOG.info('Concatenating results from'
                 f'{len(nobj)} jobs with shapes {nobj}')

        first = True
        xmatrix = None
        var_nam = None

        for iresults in results:

            # iresults is a tupla of Boolean (iresults[0])
            #  and results (iresults[1])
            if iresults[0] == False:
                return False, iresults[1]

            # internal is a tupla of 3 elements
            #  (xmatrix, var_nam, success_list)
            internal = iresults[1]
            ixmatrix = internal[0]

            # check that the type of ixmatrix is correct (np.ndarray)
            if not isinstance(ixmatrix, np.ndarray):
                LOG.error('Results type in consolidate must be `np.ndarray`.'
                          f' Found: {type(ixmatrix)}')
                return False, "unknown results type in consolidate"


            if first: # only for the first element of the loop
                xmatrix = ixmatrix
                var_nam = internal[1]
                success_list = internal[2]

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

                xmatrix = np.vstack((xmatrix, ixmatrix))
                success_list += internal[2]

        return True, (xmatrix, var_nam, success_list)

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

        if self.param.getVal('input_type') == 'model_ensemble':
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

        if self.param.getVal('input_type') == 'model_ensemble':
            return False

        try:
            picklfile = os.path.join(self.dest_path, 'data.pkl')
            if not os.path.isfile(picklfile):
                return False

            with open(picklfile, 'rb') as fi:
                md5_parameters = pickle.load(fi)
                if md5_parameters != self.param.getVal('md5'):
                    return False

                md5_input = pickle.load(fi)
                if md5_input != utils.md5sum(self.ifile):
                    return False

                success, message = self.conveyor.load(fi)
                if not success:
                    LOG.error(f'Failed to load pickle file with error: "{message}"')
                    return False

        except Exception as e:
            self.conveyor.setError('Error loading pickle with exception: {}'.format(e))
            LOG.error('Error loading pickle with exception: {}'.format(e))
            return False

        LOG.info(f'Recycling data from {picklfile}')

        return True

    @supress_log(logger=LOG)
    def workflow_objects(self, input_file):
        '''
        Executes in sequence methods required to generate MD,
        starting from a single molecular file.

        input : ifile, a molecular file in SDFile format
        output: results is a numpy bidimensional array containing MD
        '''

        success_list = []
        md_results = []
        va_results = []

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

            if first_mol:  # first molecule
                md_results = results[0]
                va_results = results[1]
                num_var = len(md_results)
                first_mol = False
            else:
                if len(results[0]) != num_var:
                    LOG.warning(f'MD length for molecule #{str(i+1)} in file'
                                f' {input_file} does not match the MD length'
                                'of the first molecule')
                    success_list[i] = False
                    continue

                md_results = np.vstack((md_results, results[0]))

        #print (success_list)

        return True, (md_results, va_results, success_list)

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
        output: results contains the following  lists
                results[0] a numpy bidimensional array containing MD
                results[1] a list of strings containing the names of the MD vars
                results[2] a list of booleans indicating for which objects the 
                           MD computations succeeded    

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
        success, results = self.computeMD(
            output_convert3D_file, self.param.getVal('computeMD_method'))

        if not success:
            return False, results

        x = results [0]
        xnames = results [1]
        success_list = results [2]

        success, mol_index = self.updateMolIndex(mol_index, success_list)
        
        return success, (x, xnames, mol_index)

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

        message = 'Failed to process ' + \
            str(len(warning_list))+' molecules : '+str(warning_list)
        message += '\nWill show results for the rest of the series...'
        message += '\nCheck the error.log file for further details'

        self.conveyor.setWarning(message)

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

            pool = mp.Pool(ncpu)

            if self.param.getVal('mol_batch') == 'series':
                results = pool.map(self.workflow_series, split_files_names)
            else:
                results = pool.map(self.workflow_objects, split_files_names)

            success, results = self.consolidate(results, split_files_sizes)

        else:

            if self.param.getVal('mol_batch') == 'series':
                success, results = self.workflow_series(lfile)
            else:
                success, results = self.workflow_objects(lfile)

        # series processing (1 or n CPUs) can produce a success == False if
        # any of the series/pieces contains an error. Abort the processing...
        if not success:
            self.conveyor.setError(results)

        # check if any molecule failed to complete the workflow and then
        # ammend object annotations in self.conveyor

        success_workflow = results[2]

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

                self.conveyor.setError('Unknown error processing input file. Probably the format is wrong or not supported')
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

        self.conveyor.addVal(results[0], 'xmatrix', 'X matrix',
            'method', 'vars', 'Molecular descriptors')

        self.conveyor.addVal(results[1], 'var_nam',
            'Var names', 'method', 'vars', 'Names of the X variables')

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

            for index, line in enumerate(fi):
                # we asume that the first row contains var names
                if index == 0 and self.param.getVal('TSV_varnames'):
                    var_nam = line.strip().split('\t')
                    var_nam = var_nam[1:]
                else:
                    value_list = line.strip().split('\t')

                    if self.param.getVal('TSV_objnames'):
                        # we asume that the first column contains object names
                        obj_nam.append(value_list[0])
                        value_list = value_list[1:]

                    if 'SMILES' in var_nam:
                        col = var_nam.index('SMILES')
                        smiles.append(value_list[col])
                        del value_list[col]

                    value_array = np.array(value_list, dtype=np.float64)
                    if index == 1:  # for the fist row, just copy the value list to the xmatrix
                        xmatrix = value_array
                    else:
                        xmatrix = np.vstack((xmatrix, value_array))

        obj_num = index
        LOG.debug('loaded TSV with shape {} '.format(xmatrix.shape))
        if self.param.getVal('TSV_varnames'):
            obj_num -= 1  # what?

        # extract any named as "TSV_activity" as the ymatrix
        activity_param = self.param.getVal('TSV_activity')
        LOG.debug('creating ymatrix from column {}'.format(activity_param))
        if activity_param in var_nam:
            col = var_nam.index(activity_param)  # Something is failing here: + 1 needed
            ymatrix = xmatrix[:, col]
            xmatrix = np.delete(xmatrix, col, 1)
            self.conveyor.addVal( ymatrix, 'ymatrix', 'Activity', 'decoration',
                             'objs', 'Biological anotation to be predicted by the model')

        self.conveyor.addVal( obj_num, 'obj_num', 'Num mol', 'method',
                         'single', 'Number of molecules present in the input file')

        #TODO: optional sanitization step, to check if the X matrix contains extreme values or variables
        # with unreasonable variances
        
        self.conveyor.addVal( xmatrix, 'xmatrix',
                         'X matrix', 'method', 'vars', 'Molecular descriptors')

        if self.param.getVal('TSV_varnames'):
            self.conveyor.addVal( var_nam, 'var_nam', 'Var names',
                             'method', 'vars', 'Names of the X variables')

        if not self.param.getVal('TSV_objnames'):
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

        # idata is a list conveyor (in JSON format) from n sources
        # the data usable for input must be listed in the ['meta']['main'] key

        # use first JSON to load common info like obj_nam, etc
        obj_common = ['label', 'decoration', 'smiles']
        
        first_results = json.loads(self.idata[0])
        first_manifest = first_results['manifest']

        for item in first_manifest:
            if item['type'] in obj_common:  # for elements of type label or decoration
                self.conveyor.addVal(first_results[item['key']], 
                                     item['key'], 
                                     item['label'], 
                                     item['type'],
                                     item['dimension'],
                                     item['description']
                                    )

        # extract usable data from every source and add to 'combo' np.array
        combined_md = None
        combined_cf = None
        combined_md_names = []
        combined_cf_names = []

        for ijson in self.idata:
            i_result = json.loads(ijson)
            i_manifest = i_result['manifest']
            i_meta = i_result['meta']

            for item in i_manifest:

                # predictions
                if item['type'] == 'result':
                    item_key = item['key']

                    if combined_md is None:  # for first element just copy
                        combined_md = np.array(
                            i_result[item_key], dtype=np.float64)
                        num_obj = len(i_result[item_key])
                    else:  # append laterally
                        if len(i_result[item_key]) != num_obj:
                            self.conveyor.setError('incompatible size of results obtained from external sources')
                            return

                        combined_md = np.c_[combined_md, np.array(
                            i_result[item_key], dtype=np.float64)]

                    combined_md_names.append(
                        item_key+':'+i_meta['endpoint']+':'+str(i_meta['version']))

                # confidence indexes 
                elif item['type'] == 'confidence':
                    item_key = item['key']
                    if combined_cf is None:  # for first element just copy
                        combined_cf = np.array(
                            i_result[item_key], dtype=np.float64)
                    else:  # append laterally
                        # combined_cf = np.c_[combined_cf, np.array(
                        #     i_result[item_key], dtype=np.float64)]
                        combined_cf = np.column_stack((combined_cf, np.array(
                            i_result[item_key], dtype=np.float64)))
                    combined_cf_names.append(
                        item_key+':'+i_meta['endpoint']+':'+str(i_meta['version']))

        self.conveyor.addVal( num_obj, 'obj_num', 'Num mol', 
                         'method', 'single', 'Number of molecules present in the input file')

        self.conveyor.addVal( combined_md, 'xmatrix', 'X matrix',
                         'results', 'vars', 'Combined output from external sources')

        self.conveyor.addVal( combined_md_names, 'var_nam', 'Var. names',
                         'method', 'vars', 'Variable names from external sources')

        if combined_cf is not None:
            self.conveyor.addVal( combined_cf, 'ensemble_confidence', 'Ensemble Confidence',
                            'method', 'objs', 'Combined confidence from external sources')

        if len(combined_cf_names) > 0:
            self.conveyor.addVal( combined_cf_names, 'ensemble_confidence_names', 'Ensemble Conf. names',
                            'method', 'vars', 'Confidence indexes from external sources')

        # print ('combined_md', combined_md)
        # print ('combined_md_names', combined_md_names)
        # print ('ensemble_confidence', combined_cf)
        # print ('ensemble_confidence_names', combined_cf_names)

        #print (self.conveyor.getJSON())

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
