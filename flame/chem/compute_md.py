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
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors

from flame.util import get_logger

LOG = get_logger(__name__)

def _RDKit_properties(ifile) -> (bool, (np.array, list, list)):
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

    md_name = []
    success_list = []
    xmatrix = []

    for prop_name in properties.GetPropertyNames():
        md_name.append(prop_name)

    try:
        num_obj = 0
        for mol in suppl:
            if mol is None:
                LOG.error(f'Unable to process molecule #{num_obj+1} in {ifile}')
                success_list.append(False)
                continue
            
            # xmatrix [num_obj] = properties.ComputeProperties(mol)
            if num_obj == 0:
                descriptors = properties.ComputeProperties(mol)
                # what is going on here??
                if np.isnan(xmatrix).any():  
                    success_list.append(False)
                    continue
                else:
                    xmatrix = descriptors
            else:
                descriptors = properties.ComputeProperties(mol)
                if np.isnan(descriptors).any():
                    success_list.append(False)
                    continue
                xmatrix = np.vstack(
                    (xmatrix, descriptors))

            success_list.append(True)
            num_obj += 1

    LOG.debug(f'Descriptors matrix (xmatrix) created with shape {xmatrix.shape}')
    except Exception as e:
        LOG.error(f'Failed computing RDKit properties for molecule #{num_obj+1} in {ifile}'
                  f' with exception: {e}')
        return False, 'Failed computing RDKit properties for molecule' + str(num_obj+1) + 'in file ' + ifile

    if num_obj == 0:
        return False, 'Unable to compute RDKit properties for molecule '+ifile

    return True, (xmatrix, md_name, success_list)


def _RDKit_descriptors(ifile) -> (bool, (np.array, list, list)):
    ''' 
    computes RDKit descriptors for the file provided as argument

    output is a boolean and a tupla with the xmatrix and the variable names
    '''
    try:
        suppl = Chem.SDMolSupplier(ifile)
    except:
        LOG.error(f'Unable to create supplier with exception {e}')
        return False, 'Unable to compute RDKit MD'

    LOG.info('Computing RDKit descriptors...')
    #what is this??
    nms = [x[0] for x in Descriptors._descList]

    md = MoleculeDescriptors.MolecularDescriptorCalculator(nms)
    success_list = []
    xmatrix = []

    try:
        num_obj = 0
        for mol in suppl:
            if mol is None:
                LOG.error(f'Unable to process molecule #{num_obj+1} in {ifile}')
                print('ERROR: (@_RDKit_descriptors) Unable to process molecule #', str(
                    num_obj+1), 'in file ' + ifile)
                success_list.append(False)
                continue

            if num_obj == 0:
                xmatrix = md.CalcDescriptors(mol)
                if np.isnan(xmatrix).any():
                    success_list.append(False)
                    continue
            else:
                descriptors = md.CalcDescriptors(mol)
                if np.isnan(descriptors).any():
                    success_list.append(False)
                    continue
                xmatrix = np.vstack((xmatrix, descriptors))

            success_list.append(True)
            num_obj += 1

    except:
        return False, 'Failed computing RDKit descriptors for molecule' + str(num_obj+1) + 'in file ' + ifile
    
    if num_obj == 0:    
        return False, 'Unable to compute RDKit properties for molecule '+ifile

    if num_obj == 0:
        return False, 'Unable to compute RDKit descriptors for molecule '+ifile

    return True, (xmatrix, nms, success_list)


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
