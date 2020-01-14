#! -*- coding: utf-8 -*-

# Description    Flame compute molecular descriptor functions
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
import shutil
import tempfile
import requests
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import Descriptors
from rdkit.Chem import DataStructs
from rdkit.ML.Descriptors import MoleculeDescriptors
from flame.util import get_logger

LOG = get_logger(__name__)


def clean_extra_xrows (xmatrix, num_obj, est_obj):
    for i in range (num_obj, est_obj):
        xmatrix = np.delete(xmatrix,num_obj,axis=0)

    return xmatrix


def _RDKit_morganFPS(ifile, **kwargs) -> (bool, (np.ndarray, list, list)):
    ''' 
    Morgan circular FP using RDkit output is a boolean and
    a tupla with the xmatrix and the variable names
    '''
    try:
        suppl = Chem.SDMolSupplier(ifile)
    except Exception as e:
        LOG.error(f'Unable to create supplier with exception {e}')
        return False, 'unable to create supplier'

    morgan_radius = kwargs['morgan_radius']
    morgan_features = kwargs['morgan_features']

    LOG.info(f'computing MorganFP fingerprint... with r={morgan_radius}')

    # get from here num of properties

    success_list = []
    est_obj = len(suppl)
    xmatrix = np.zeros((est_obj, 2048), dtype=np.int8)

    try:
        num_obj = 0
        for mol in suppl:
            if mol is None:
                LOG.error(
                    f'Unable to process molecule #{num_obj+1} in {ifile}')
                success_list.append(False)
                continue

            fp = AllChem.GetMorganFingerprintAsBitVect(mol,
                            morgan_radius,  
                            useFeatures=morgan_features)

            #xvector = np.empty((1, 2048), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(fp,xmatrix[num_obj])

            # if np.isnan(xvector).any():
            #     success_list.append(False)
            #     continue

            # xmatrix.append(xvector)
            success_list.append(True)
            num_obj += 1

    except Exception as e:
        LOG.error(f'Failed computing RDKit Morgan Fingerprints for molecule #{num_obj+1} in {ifile}'
                  f' with exception: {e}')
        return False, 'Failed computing RDKit Morgan Fingerprints for molecule' + str(num_obj+1) + 'in file ' + ifile

    if num_obj < est_obj:
        clean_extra_xrows(xmatrix, num_obj, est_obj)

    LOG.debug(f'computed RDKit Morgan Fingerprints matrix with shape {np.shape(xmatrix)}')
    if num_obj == 0:
        return False, 'Unable to compute RDKit Morgan Fingerprints for molecule '+ifile

    results = {
        'matrix': xmatrix,
        'names' : [],
        'success_arr': success_list
    }
    return True, results


def _padel_descriptors(ifile):
    ''' 
    computes Padel molecular descriptors calling an external web service for
    the file provided as argument

    output is a boolean and a tupla with the xmatrix and the variable names

    '''

    # TODO: this cannot be hardcoded! maybe read from the component registry?
    uri = "http://localhost:5000/padel/api/v0.1/calc/json"

    tmpdir = os.path.abspath(tempfile.mkdtemp(dir=os.path.dirname(ifile)))
    shutil.copy(ifile, tmpdir)

    payload = {
        '-2d': '',
        '-dir': tmpdir
    }

    try:
        req = requests.post(uri, json=payload)
        if req.status_code != 200:
            return False, 'ERROR: failed to contact padel service with code: '+str(req.status_code)
    except:
        return False, 'ERROR: failed to contact padel service'

    # DEBUG only
    print('padel service results : ', req.json())

    results = req.json()

    if not results['success']:
        return False, 'padel service returned error condition'

    ofile = os.path.join(tmpdir, results['filename'])

    if not os.path.isfile(ofile):
        return False, 'padel service returned no file'

    with open(ofile, 'r') as of:
        index = 0
        var_nam = []
        success_list = []
        xmatrix = []

        for line in of:

            if index == 0:  # we asume that the first row contains var names
                var_nam = line.strip().split(',')
                var_nam = var_nam[1:]

            else:

                value_list = line.strip().split(',')

                try:
                    nvalue_list = [float(x) for x in value_list[1:]]
                except:
                    success_list.append(False)
                    print(
                        'ERROR (@_padel_descriptors) in Padel results parsing for object '+str(index))
                    continue

                md = np.array(nvalue_list, dtype=np.float64)

                # md = np.nan_to_num(md)
                # detected a rare bug producing extremely large PaDel
                # descriptors (>1.0e300), leading to overflows
                # apply a conservative top cutoff of 1.0e10
                # md [ md > 1.0e10 ] = 1.0e10

                if index == 1:  # copy the value list to the xmatrix
                    xmatrix = md
                else:
                    xmatrix = np.vstack((xmatrix, md))

                success_list.append(True)

            index += 1

    shutil.rmtree(tmpdir)

    # if no object was processed with success (index==1) return False
    # this is common when series are processed object-wise

    return (index > 1), (xmatrix, var_nam, success_list)


def _RDKit_descriptors(ifile, **kwargs) -> (bool, (np.ndarray, list, list)):
    '''
    computes RDKit descriptors for the file provided as argument

    output is a boolean and a tupla with the xmatrix and the variable names
    '''
    try:
        suppl = Chem.SDMolSupplier(ifile)
    except Exception as e:
        LOG.error(f'Unable to create supplier with exception {e}')
        return False, 'Unable to compute RDKit MD'

    LOG.info('Computing RDKit descriptors...')

    black_list = []
    if 'rdkit_black_list' in kwargs:
        black_list = kwargs['rdkit_black_list']

    # colecciona lista de descriptores moleculares
    nms = []
    for md_id in Descriptors._descList:
        if md_id[0] in black_list:
            print ('skipping MD:', md_id[0])
            continue
        nms.append(md_id[0])

    #nms = [x[0] for x in Descriptors._descList]

    md = MoleculeDescriptors.MolecularDescriptorCalculator(nms)

    # list of MD computation success/failure for every object
    success_list = []
    
    # allocate a np.matrix for storing the X matrix
    # the number of rows is an estimation and will be checked
    # and corrected at the end
    est_obj = len(suppl)
    xmatrix = np.zeros((est_obj,len(nms)))

    try:
        num_obj = 0
        for mol in suppl:
            if mol is None:
                LOG.error('Unable to process molecule'
                          f'#{num_obj+1} in {ifile}')
                success_list.append(False)
                continue

            mdi = md.CalcDescriptors(mol)

            if np.isnan(mdi).any():
                success_list.append(False)
                continue               

            xmatrix[num_obj]=mdi
            success_list.append(True)
            num_obj += 1

    except:  # if any mol fails the whole try except will break
        return False, 'Failed computing RDKit descriptors for molecule' + str(num_obj+1) + 'in file ' + ifile

    if num_obj < est_obj:
        clean_extra_xrows(xmatrix, num_obj, est_obj)

        # # if some molecules failed to compute we will clean xmatrix by 
        # # removing extra rows
        # # for this we will call the first extra row (xmatrix[num_obj])
        # # est_obj-num_obj times
        # for i in range (num_obj,est_obj):
        #     xmatrix = np.delete(xmatrix,num_obj,axis=0)
        #     print ('deleted ', i, np.shape(xmatrix))

    LOG.debug(f'computed RDKit descriptors matrix with shape {np.shape(xmatrix)}')
    if num_obj == 0:
        return False, 'Unable to compute RDKit descriptors for molecule '+ifile
    
    results = {
        'matrix': xmatrix,
        'names': nms,
        'success_arr': success_list
    }

    return True, results


def _RDKit_properties(ifile, **kwargs) -> (bool, (np.ndarray, list, list)):
    ''' 
    computes RDKit properties for the file provided as argument

    output is a boolean and a tupla with the xmatrix and the variable names
    '''
    try:
        suppl = Chem.SDMolSupplier(ifile)
    except Exception as e:
        LOG.error(f'Unable to create supplier with exception {e}')
        return False, 'unable to create supplier'

    LOG.info('computing RDKit properties...')

    properties = rdMolDescriptors.Properties()

    # get from here num of properties
    md_name = [prop_name for prop_name in properties.GetPropertyNames()]

    success_list = []
    est_obj = len(suppl)
    xmatrix = np.zeros((est_obj,len(md_name)))

    try:
        num_obj = 0
        for mol in suppl:
            if mol is None:
                LOG.error(
                    f'Unable to process molecule #{num_obj+1} in {ifile}')
                success_list.append(False)
                continue

            descriptors = properties.ComputeProperties(mol)

            if np.isnan(descriptors).any():
                success_list.append(False)
                continue
            
            xmatrix [num_obj] = descriptors
            
            # xmatrix.append(descriptors)
            # if num_obj == 0:
            #     xmatrix = descriptors
            #     LOG.debug(f'first descriptor vector computed')
            # else:
            #     xmatrix = np.vstack((xmatrix, descriptors))

            success_list.append(True)
            num_obj += 1

    except Exception as e:
        LOG.error(f'Failed computing RDKit properties for molecule #{num_obj+1} in {ifile}'
                  f' with exception: {e}')
        return False, 'Failed computing RDKit properties for molecule' + str(num_obj+1) + 'in file ' + ifile

    if num_obj < est_obj:
        clean_extra_xrows(xmatrix, num_obj, est_obj)

    LOG.debug(f'computed RDKit properties matrix with shape {np.shape(xmatrix)}')
    if num_obj == 0:
        return False, 'Unable to compute RDKit properties for molecule '+ifile

    results = {
        'matrix': xmatrix,
        'names': md_name,
        'success_arr': success_list
    }
    return True, results
