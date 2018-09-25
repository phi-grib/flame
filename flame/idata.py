#! -*- coding: utf-8 -*-

# Description    Flame Idata class
##
# Authors:       Manuel Pastor (manuel.pastor@upf.edu)
##
# Copyright 2018 Manuel Pastor
##
# This file is part of Flame
##
# Flame is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation version 3.
##
# Flame is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
##
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

import flame.chem.sdfileutils as sdfu
import flame.chem.compute_md as computeMD
import flame.chem.convert_3d as convert3D

from flame.util import utils


class Idata:

    def __init__(self, parameters, input_source):

        self.parameters = parameters      # control object defining the processing
        # path for temp files (fallback default)
        self.dest_path = '.'

        self.results = {
            'manifest': [],
            'meta': {'main': [],
                     'endpoint': self.parameters['endpoint'],
                     'version': self.parameters['version'],
                     }
        }    # create empty context index ('manifest')

        if ('ext_input' in parameters) and (parameters['ext_input']):
            self.idata = input_source
            self.ifile = None
            randomName = 'flame-'+utils.id_generator()
            self.dest_path = os.path.join(tempfile.gettempdir(), randomName)

        else:
            self.idate = None
            self.ifile = input_source
            self.dest_path = os.path.dirname(self.ifile)

    def extractInformation(self, ifile):
        '''
        Extracts molecule names, biological anotations and experimental values
        from an SDFile.

        Anotations must be added using method utils.add_result,
        so they are also inserted into the results manifest.
        '''

        try:
            suppl = Chem.SDMolSupplier(ifile)
        except Exception as e:
            self.results['error'] = f'unable to open {ifile}. {e}'
            return

        obj_nam = []
        obj_bio = []
        obj_exp = []
        obj_sml = []
        obj_num = 0
        success_list = []

        for mol in suppl:

            # Do not even try to process molecules not recognised by RDKit.
            # They will be removed at the normalization step
            if mol is None:
                print('ERROR: (@extractInformaton) Unable to process molecule #', str(
                    obj_num+1), 'in file ' + ifile)
                success_list.append(False)
                continue

            name = sdfu.getName(
                mol, count=obj_num, field=self.parameters['SDFile_name'], suppl=suppl)

            activity_num = None
            exp = None

            if mol.HasProp(self.parameters['SDFile_activity']):
                activity_str = mol.GetProp(self.parameters['SDFile_activity'])
                try:
                    activity_num = float(activity_str)
                except:
                    activity_num = None

            if mol.HasProp(self.parameters['SDFile_experimental']):
                exp = mol.GetProp(self.parameters['SDFile_experimental'])

            # generate a SMILES
            try:
                sml = Chem.MolToSmiles(mol)
            except:
                sml = None

            obj_nam.append(name)
            obj_bio.append(activity_num)
            obj_exp.append(exp)
            obj_sml.append(sml)

            success_list.append(True)
            obj_num += 1

        utils.add_result(self.results, obj_num, 'obj_num', 'Num mol', 'method',
                         'single', 'Number of molecules present in the input file')
        utils.add_result(self.results, obj_nam, 'obj_nam', 'Mol name', 'label',
                         'objs', 'Name of the molecule, as present in the input file')
        utils.add_result(self.results, obj_sml, 'SMILES', 'SMILES',
                         'smiles', 'objs', 'Structure of the molecule in SMILES format')

        if not utils.is_empty(obj_bio):
            utils.add_result(self.results, np.array(obj_bio, dtype=np.float64),
                             'ymatrix', 'Activity',
                             'decoration', 'objs',
                             'Biological anotation to be predicted by the model')

        if not utils.is_empty(obj_exp):
            utils.add_result(self.results, np.array(obj_exp, dtype=np.float64),
                             'experim', 'Experim.',
                             'decoration', 'objs',
                             'Experimental anotation present in the input file')

        return success_list

    def normalize(self, ifile, method):
        '''
        Generates a simplified SDFile with MolBlock and an internal ID for
        further processing.

        Also, when defined in control, applies chemical standardization
        protocols, like the one provided by Francis Atkinson (EBI),
        accessible from:

            https://github.com/flatkinson/standardiser

        Returns a tuple containing the result of the method and (if True)
        the name of the output molecule and an error message otherwyse

        '''

        if not method:
            return True, ifile

        try:
            suppl = Chem.SDMolSupplier(ifile)
        except:
            return False, 'Error at processing input file for standardizing structures'

        success = True
        filename, fileext = os.path.splitext(ifile)
        ofile = filename + '_std' + fileext

        with open(ofile, 'w') as fo:
            mcount = 0
            # merror = 0
            for m in suppl:

                # molecule not recognised by RDKit
                if m is None:
                    print('ERROR: (@normalize) Unable to process molecule #',
                        str(mcount+1), 'in file ' + ifile)

                    continue

                # if standardize
                if 'standardize' in method:
                    try:
                        parent = standardise.run(Chem.MolToMolBlock(m))
                    except standardise.StandardiseException as e:
                        if e.name == "no_non_salt":
                            parent = Chem.MolToMolBlock(m)
                        else:
                            return False, e.name
                    except:
                        return False, "Unknown standardiser error"

                else:
                    print('ERROR: (@normalize) method ' +
                          method+' not recognized')
                    parent = Chem.MolToMolBlock(m)

                # in any case, write parent plus internal ID (flameID)
                fo.write(parent)

                flameID = 'fl%0.10d' % mcount
                fo.write('>  <flameID>\n'+flameID+'\n\n')

                mcount += 1

                # terminator
                fo.write('$$$$\n')

        return success, ofile

    def ionize(self, ifile, method):
        '''
        Adjust the ionization status of the molecular structure,
        using a given pH.
        '''

        if not method:
            return True, ifile

        success = False
        results = 'not ionized'

        # methods here

        results = 'ionization method not recognised'

        return success, results

    def convert3D(self, ifile, method):
        '''
        Assigns 3D structures to the molecular structures provided as input.
        '''

        if not method:
            return True, ifile

        success = False
        results = 'not converted to 3D'

        if 'ETKDG' in method:
            success, results = convert3D._ETKDG(ifile)

        return success, results

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

        return False, 'not implemented'

    def computeMD(self, ifile, method):
        '''
        Uses the molecular structures for computing an array
        of values (int or float).

        input is the name of a molecule or a series of molecules and a label
        of the method output is boolean anda a tupla of two elements:
        [0] xmatrix (nparray np.float64)
        [1] list of variable names (str)

        '''

        combined_md = None
        combined_nm = None
        combined_sc = None

        is_empty = True

        registered_methods = [('RDKit_properties', computeMD._RDKit_properties),
                              ('RDKit_md', computeMD._RDKit_descriptors),
                              ('padel', computeMD._padel_descriptors),
                              ('custom', self.computeMD_custom)]

        for imethod in registered_methods:
            if imethod[0] in method:

                success, results = imethod[1](ifile)

                if not success:
                    return success, results

                if is_empty:  # first md computed, just copy

                    combined_md = results[0]  # np.array of values
                    combined_nm = results[1]  # list of variable names
                    combined_sc = results[2]  # list of true/false

                    shape = np.shape(combined_md)

                    is_empty = False

                else:  # append laterally

                    ishape = np.shape(results[0])

                    if (len(ishape) > 1):
                        # for 2D arrays, shape[0] is the number of objects
                        if ishape[0] != shape[0]:
                            print('ERROR: number of objects processed by md method "' +
                                  imethod[0]+'" does not match those computed by other methods')
                            continue

                    combined_md = np.hstack((combined_md, results[0]))
                    combined_nm.extend(results[1])

                    new_sc = []
                    for i in range(len(combined_sc)):
                        new_sc.append(combined_sc and results[2][i])

                    combined_sc = new_sc

        return True, (combined_md, combined_nm, combined_sc)

    def consolidate(self, results, nobj):
        ''' 
        Mix the results obtained by multiple CPUs into a single result file.
        '''

        first = True
        xmatrix = None
        var_nam = None

        for iresults in results:

            # iresults is a tupla of Boolean (iresults[0]) and results (iresults[1])
            if iresults[0] == False:
                return False, iresults[1]

            # internal is a tupla of 3 elements (xmatrix, var_nam, success_list)
            internal = iresults[1]
            ixmatrix = internal[0]

            if type(ixmatrix).__module__ != np.__name__:
                return False, "unknown results type in consolidate"

            if first:
                xmatrix = ixmatrix
                var_nam = internal[1]
                success_list = internal[2]

                shape = np.shape(ixmatrix)
                first = False
            else:
                ishape = np.shape(ixmatrix)
                # for bidimensional arrays, num_var is shape[1]
                if len(shape) > 1 and len(ishape) > 1:
                    if shape[1] != ishape[1]:
                        return False, "inconsistent number of variables"
                else:
                    # for vectors obtained with a single object, numvar is shape[0]
                    if shape[0] != ishape[0]:
                        return False, "inconsistent number of variables"

                xmatrix = np.vstack((xmatrix, ixmatrix))
                success_list += internal[2]

        return True, (xmatrix, var_nam, success_list)

    def save(self):
        ''' 
        Saves the results in serialized form, together with the MD5 signature
        of the control class and the input file.
        '''

        # uncomment to avoid saving results
        # print ('*** save commented for debugging ***')
        # return
        ##

        if 'ext_input' in self.parameters and self.parameters['ext_input']:
            return

        md5_parameters = self.parameters['md5']
        md5_input = utils.md5sum(self.ifile)  # run md5 in self.ifile

        try:
            with open(os.path.join(self.dest_path, 'data.pkl'), 'wb') as fo:

                pickle.dump(md5_parameters, fo)
                pickle.dump(md5_input, fo)

                pickle.dump(self.results, fo)

        except:
            pass

    def load(self):
        ''' 
        Loads the results in serialized form, together with the MD5 signature
        of the control class and the input file.
        '''

        if 'ext_input' in self.parameters and self.parameters['ext_input']:
            return False

        try:
            with open(os.path.join(self.dest_path, 'data.pkl'), 'rb') as fi:
                md5_parameters = pickle.load(fi)
                if md5_parameters != self.parameters['md5']:
                    return False

                md5_input = pickle.load(fi)
                if md5_input != utils.md5sum(self.ifile):
                    return False

                self.results = pickle.load(fi)

                # these values are added programatically and therefore not
                # checked by the md5
                self.results['meta']['endpoint'] = self.parameters['endpoint']
                self.results['meta']['version'] = self.parameters['version']

        except:
            return False

        print('>>> recycling data >>>', os.path.join(self.dest_path, 'data.pkl'))

        return True

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
        num_mol = sdfu.count_mols(input_file)
        success, results = sdfu.split_SDFile(input_file, num_mol)

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

                print('ERROR: (@workflow_objects) Unable to process molecule #', str(
                    i+1), 'in file ' + ifile)

                continue

            success, results = self.workflow_series(ifile)

            # since the workflow was run for a single molecule, results[2] is ignored, because it must match
            # the value in success
            success_list[i] = success

            if not success:           # failed in the workflow
                print('ERROR: (@workflow_objects) Workflow failed for molecule #', str(
                    i+1), 'in file ' + input_file)
                continue

            if first_mol:  # first molecule
                md_results = results[0]
                va_results = results[1]
                num_var = len(md_results)
                first_mol = False
            else:
                if len(results[0]) != num_var:
                    print('ERROR: (@workflow_objects) MD length for molecule #', str(
                        i+1), 'in file ' + input_file + 'does not match the MD length of the first molecule')
                    success_list[i] = False
                    continue

                md_results = np.vstack((md_results, results[0]))

        #print (success_list)

        return True, (md_results, va_results, success_list)

    def workflow_series(self, input_file):
        '''      
        Executes in sequence methods required to generate MD,
        starting from a single molecular file

        input : ifile, a molecular file in SDFile format
        output: results contains two lists
                results[0] a numpy bidimensional array containing MD
                results[1] a list of strings containing the names of the MD vars
                results[2] a list of booleans indicating for which objects the MD computations succeeded    

        '''

        success, results = self.normalize(
            input_file, self.parameters['normalize_method'])
        if success:
            success, results = self.ionize(
                results, self.parameters['ionize_method'])
            if success:
                success, results = self.convert3D(
                    results, self.parameters['convert3D_method'])
                if success:
                    success, results = self.computeMD(
                        results, self.parameters['computeMD_method'])

        return success, results

    def ammend_objects(self, inform, workflow):
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
        for i in range(len(workflow)):
            if inform[i] and not workflow[i]:
                remove_index.append(i)
                warning_list.append(self.results['obj_nam'][i])
            if inform[i] and workflow[i]:
                obj_num += 1

        if 'obj_num' in self.results:
            self.results['obj_num'] = obj_num

        manifest = self.results['manifest']
        for element in manifest:
            if element['dimension'] == 'objs':

                ikey = element['key']
                ilist = self.results[ikey]

                # keys are experim or ymatrix are numpy arrays
                if 'numpy.ndarray' in str(type(ilist)):
                    self.results[ikey] = np.delete(ilist, remove_index)
                # other keys are regular list
                else:
                    for i in sorted(remove_index, reverse=True):
                        del ilist[i]

        message = 'Failed to process ' + \
            str(len(warning_list))+' molecules : '+str(warning_list)
        message += '\nWill show results for the rest of the series...'
        message += '\nCheck the error.log file for further details'

        self.results['warning'] = message

        return

    def _run_molecule(self):
        '''
        version of Run for molecular input

        '''

        # extract useful information from file

        success_inform = self.extractInformation(self.ifile)
        if 'error' in self.results:
            return

        nobj = self.results['obj_num']
        ncpu = min(nobj, self.parameters['numCPUs'])

        # copy the input file to a temp file which will be cleaned at the end
        temp_path = tempfile.mkdtemp()
        shutil.copy(self.ifile, temp_path)
        lfile = os.path.join(temp_path, os.path.basename(self.ifile))

        # Execute the workflow in 1 or n CPUs
        if ncpu > 1:

            success, results = sdfu.split_SDFile(lfile, ncpu)

            if not success:
                self.results['error'] = 'unable to split input molecule'
                return

            split_files_names = results[0]
            split_files_sizes = results[1]

            pool = mp.Pool(ncpu)

            if self.parameters['mol_batch'] == 'series':
                results = pool.map(self.workflow_series, split_files_names)
            else:
                results = pool.map(self.workflow_objects, split_files_names)

            success, results = self.consolidate(results, split_files_sizes)

        else:

            if self.parameters['mol_batch'] == 'series':
                success, results = self.workflow_series(lfile)
            else:
                success, results = self.workflow_objects(lfile)

        # series processing (1 or n CPUs) can produce a success == False if
        # any of the series/pieces contains an error. Abort the processing...
        if not success:
            self.results['error'] = results

        # check if any molecule failed to complete the workflow and then
        # ammend object annotations in self.results

        success_workflow = results[2]

        if len(success_inform) != len(success_workflow):
            self.results['error'] = 'number of molecules informed and processed does not match'
            return

        # Check if molecules not informed succeded
        # to be complete MD generation.
        # This should never happen, because they
        # do not pass the normalization step
        for i, j in zip(success_inform, success_workflow):
            if j and not i:
                self.results['error'] = 'unknown error in molecule inform'
                return

        # check if a molecule informed did not 
        # succeed to complete MD generation
        for i, j in zip(success_inform, success_workflow):
            if i and not j:
                self.ammend_objects(success_inform, success_workflow)
                break

        # remove the temp directory with all the temp files inside
        shutil.rmtree(temp_path)

        utils.add_result(
            self.results, results[0], 'xmatrix', 'X matrix',
            'method', 'vars', 'Molecular descriptors')

        utils.add_result(
            self.results, results[1], 'var_nam',
            'Var names', 'method', 'vars', 'Names of the X variables')

        return

    def _run_data(self):
        '''
        version of Run for data input (TSV tabular format)
        '''

        if not os.path.isfile(self.ifile):
            self.results['error'] = 'unable to open file '+self.ifile
            return

        with open(self.ifile, 'r') as fi:
            index = 0
            var_nam = []
            obj_nam = []
            smiles = []

            for line in fi:
                # we asume that the first row contains var names
                if index == 0 and self.parameters['TSV_varnames']:
                    var_nam = line.strip().split('\t')
                    var_nam = var_nam[1:]
                else:
                    value_list = line.strip().split('\t')

                    if self.parameters['TSV_objnames']:
                        # we asume that the first column contains object names
                        obj_nam.append(value_list[0])
                        value_list = value_list[1:]

                    if 'SMILES' in var_nam:
                        col = var_nam.index('SMILES')
                        smiles.append(value_list[col])
                        del value_list[col]

                    if index == 1:  # for the fist row, just copy the value list to the xmatrix
                        xmatrix = np.array(value_list, dtype=np.float64)
                    else:
                        xmatrix = np.vstack(
                            (xmatrix, np.array(value_list, dtype=np.float64)))
                index += 1

        obj_num = index
        if self.parameters['TSV_varnames']:
            obj_num -= 1

        # extract any named as "TSV_activity" as the ymatrix
        if self.parameters['TSV_activity'] in var_nam:
            col = var_nam.index(self.parameters['TSV_activity'])
            ymatrix = xmatrix[:, col]
            xmatrix = np.delete(xmatrix, col, 1)
            utils.add_result(self.results, ymatrix, 'ymatrix', 'Activity', 'decoration',
                             'objs', 'Biological anotation to be predicted by the model')

        utils.add_result(self.results, obj_num, 'obj_num', 'Num mol', 'method',
                         'single', 'Number of molecules present in the input file')
        utils.add_result(self.results, xmatrix, 'xmatrix',
                         'X matrix', 'method', 'vars', 'Molecular descriptors')

        if self.parameters['TSV_varnames']:
            utils.add_result(self.results, var_nam, 'var_nam', 'Var names',
                             'method', 'vars', 'Names of the X variables')

        if not self.parameters['TSV_objnames']:
            for i in range(obj_num):
                obj_nam.append('obj%.10f' % i)

        utils.add_result(self.results, obj_nam, 'obj_nam', 'Mol name', 'label',
                         'objs', 'Name of the molecule, as present in the input file')

        if len(smiles) > 0:
            utils.add_result(self.results, smiles, 'SMILES', 'SMILES',
                             'smiles', 'objs', 'Structure of the molecule in SMILES format')

        return

    def _run_ext_data(self):
        '''
        version of Run for inter-process input (calling another model to obtain input)
        '''

        # idata is a list of JSON from 1-n sources
        # the data usable for input must be listed in the ['meta']['main'] key

        # use first JSON to load common info like obj_nam, etc
        obj_common = ['label', 'decoration']

        # load object identifiers and decorators
        first_results = json.loads(self.idata[0])
        first_manifest = first_results['manifest']

        for item in first_manifest:
            if item['type'] in obj_common:
                item_key = item['key']
                self.results[item_key] = first_results[item_key]
                self.results['manifest'].append(item)

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
                if item['type'] == 'result':
                    item_key = item['key']

                    if combined_md is None:  # for first element just copy
                        combined_md = np.array(
                            i_result[item_key], dtype=np.float64)
                        num_obj = len(i_result[item_key])
                    else:  # append laterally
                        if len(i_result[item_key]) != num_obj:
                            self.results['error'] = 'incompatible size of results obtained from external sources'
                            return

                        combined_md = np.c_[combined_md, np.array(
                            i_result[item_key], dtype=np.float64)]

                    combined_md_names.append(
                        item_key+':'+i_meta['endpoint']+':'+str(i_meta['version']))

                if item['type'] == 'confidence':
                    item_key = item['key']
                    if combined_cf is None:  # for first element just copy
                        combined_cf = np.array(
                            i_result[item_key], dtype=np.float64)
                    else:  # append laterally
                        combined_cf = np.c_[combined_cf, np.array(
                            i_result[item_key], dtype=np.float64)]

                    combined_cf_names.append(
                        item_key+':'+i_meta['endpoint']+':'+str(i_meta['version']))

        utils.add_result(self.results, combined_md, 'xmatrix', 'X matrix',
                         'results', 'objs', 'Combined output from external sources')
        utils.add_result(self.results, combined_cf, 'confidence', 'Confidence',
                         'confidence', 'objs', 'Combined confidence from external sources')
        utils.add_result(self.results, combined_md_names, 'var_nam', 'Var. names',
                         'method', 'vars', 'Variable names from external sources')
        utils.add_result(self.results, combined_cf_names, 'conf_nam', 'Conf. names',
                         'method', 'vars', 'Confidence indexes from external sources')

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

        # check for the presence of a valid pickle file
        if self.load():
            return self.results

        if self.parameters['input_type'] == 'ext_data':
            input_type = 'ext_data'
        else:
            suffix = pathlib.Path(self.ifile).suffix
            if suffix == '.tsv':
                input_type = 'data'
            elif suffix == '.sdf':
                input_type = 'molecule'
            else:
                input_type = self.parameters['input_type']

        # processing for molecular input (for now an SDFile)
        if (input_type == 'molecule'):

            # trick to avoid RDKit dumping warnings to the console
            if not self.parameters['verbose_error']:
                stderr_fileno = sys.stderr.fileno()  # saves current syserr
                stderr_save = os.dup(stderr_fileno)
                # open a specific RDKit log file
                stderr_fd = open('errorRDKit.log', 'w')
                os.dup2(stderr_fd.fileno(), stderr_fileno)

            self._run_molecule()

            if not self.parameters['verbose_error']:
                stderr_fd.close()                     # close the RDKit log
                os.dup2(stderr_save, stderr_fileno)   # restore old syserr

        # processing for non-molecular input (not implemented)
        elif (input_type == 'data'):
            self._run_data()

        # processing for external data
        elif (input_type == 'ext_data'):
            self._run_ext_data()

        else:
            self.results['error'] = 'unknown input data format'

        # save in a pickle file stamped with MD5 hash of file and control
        if 'error' not in self.results:
            self.save()

        return self.results
