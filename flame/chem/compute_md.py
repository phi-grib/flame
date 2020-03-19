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


def _mordred_descriptors(ifile, **kwargs):
    ''' 
    mordred descriptors. output is a boolean and
    a tupla with the xmatrix and the variable names
    '''
    # Import pandas library to compute/filter descriptors
    import pandas as pd
    import pathlib
    from mordred import Calculator, descriptors

    try:
        suppl = Chem.SDMolSupplier(ifile)
    except Exception as e:
        LOG.error(f'Unable to create supplier with exception {e}')
        return False, 'unable to create supplier'

    # Whether or not compute 3D descriptors
    compute3D = kwargs['mordred_3D']
    
    # Read black list for 2D or 3D descriptors
    wkd = pathlib.Path(os.path.dirname(os.path.abspath(__file__)))
    black_list = []
    if not compute3D:
        try:
            black_list = pd.read_csv(str(wkd) + '/mordred_discarded_1percent2D.csv',
                                     sep="\t")
            black_list = black_list['Name'].to_list()
        except Exception as e:
            print(e)
    else:
        try:
            black_list = pd.read_csv(str(wkd) + '/mordred3D_discarded_1percent.csv',
                                     sep="\t")
            black_list = black_list['Name'].to_list()
        except Exception as e:
            print(e)

    LOG.info(f'computing mordred descriptors... with ignore_3D option = { not compute3D}')

     # list of MD computation success/failure for every object
    success_list = []
    calc = Calculator(descriptors, ignore_3D=(not compute3D))
    # Get descriptors names
    nms = [str(d) for d in calc.descriptors]
    est_obj = len(suppl)
    xmatrix = np.zeros((est_obj, len(calc.descriptors)))

    try:
        num_obj = 0
        for mol in suppl:
            if mol is None:
                LOG.error('Unable to process molecule'
                          f'#{num_obj+1} in {ifile}')
                success_list.append(False)
                continue

            mdi = np.asarray(list(calc(mol)), dtype=np.float)
            xmatrix[num_obj] = mdi
            success_list.append(True)
            num_obj += 1
    except:  # if any mol fails the whole try except will break
        return False, 'Failed computing mordred descriptors for molecule' + str(num_obj+1) + 'in file ' + ifile
   
    # Create a DataFrame and remove black list descritors 
    frame = pd.DataFrame(xmatrix, columns=nms)
    frame = frame.drop(columns=black_list, axis=0)
    xmatrix = frame.values
    nms = frame.columns.to_list()
    success_list2 = []
    matrix_f = []
    # Now check nan values in rows 
    for row in xmatrix:
        if np.isnan(row).any():
            success_list2.append(False)
        else:
            matrix_f.append(row)
            success_list2.append(True)
    xmatrix = np.asarray(matrix_f)
    success_l1 = np.asarray(success_list)
    success_l2 = np.asarray(success_list2)
    # Update success list
    success_list = list(success_l1 & success_l2) 
    if num_obj < est_obj:
        # if some molecules failed to compute we will clean xmatrix by 
        # removing extra rows
        for i in range (num_obj, est_obj):
            xmatrix = np.delete(xmatrix,num_obj,axis=0)

    LOG.info(f'computed mordred descriptors matrix with shape {np.shape(xmatrix)}')
    if num_obj == 0:
        return False, 'Unable to compute mordred descriptors for molecule '+ifile
    
    results = {
        'matrix': xmatrix,
        'names': nms,
        'success_arr': success_list
    }

    return True, results


def _RDKit_morganFPS(ifile, **kwargs):
    ''' 
    Morgan circular FP using RDkit output is a boolean and
    a tupla with the xmatrix and the variable names
    '''
    try:
        suppl = Chem.SDMolSupplier(ifile)
    except Exception as e:
        LOG.error(f'Unable to create supplier with exception {e}')
        return False, 'unable to create supplier'

    # defaults
    morgan_radius = 2
    morgan_nbits = 2048
    morgan_features = True

    # read arguments
    if 'morgan_radius' in kwargs:
        morgan_radius = kwargs['morgan_radius']

    if 'morgan_features' in kwargs:
        morgan_features = kwargs['morgan_features']

    if 'morgan_nbits' in kwargs:
        morgan_nbits = kwargs['morgan_nbits']

    LOG.info(f'computing RDKit Morgan fingerprint... with radius {morgan_radius}, size {morgan_nbits} and features {morgan_features}')

    # get from here num of properties

    success_list = []
    est_obj = len(suppl)
    xmatrix = np.zeros((est_obj, morgan_nbits), dtype=np.int8)

    try:
        num_obj = 0
        for mol in suppl:
            if mol is None:
                LOG.error(
                    f'Unable to process molecule #{num_obj+1} in {ifile}')
                success_list.append(False)
                continue

            if mol.GetNumHeavyAtoms() == 0:
                LOG.error('Empty molecule'
                          f'#{num_obj+1} in {ifile}')
                success_list.append(False)
                continue

            fp = AllChem.GetMorganFingerprintAsBitVect(mol,
                            morgan_radius,  
                            nBits=morgan_nbits,
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
        # if some molecules failed to compute we will clean xmatrix by 
        # removing extra rows
        for i in range (num_obj, est_obj):
            xmatrix = np.delete(xmatrix,num_obj,axis=0)

    LOG.debug(f'computed RDKit Morgan Fingerprints matrix with shape {np.shape(xmatrix)}')
    if num_obj == 0:
        return False, 'Unable to compute RDKit Morgan Fingerprints for molecule '+ifile

    results = {
        'matrix': xmatrix,
        'names' : [],
        'success_arr': success_list
    }
    return True, results


def _RDKit_descriptors(ifile, **kwargs):
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
            LOG.info (f'Skipping molecular descriptors: {md_id[0]}')
            continue
        nms.append(md_id[0])

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

            if mol.GetNumHeavyAtoms() == 0:
                LOG.error('Empty molecule'
                          f'#{num_obj+1} in {ifile}')
                success_list.append(False)
                continue

            mdi = md.CalcDescriptors(mol)

            if np.isnan(mdi).any() or np.isinf(mdi).any():
                success_list.append(False)
                continue               

            xmatrix[num_obj]=mdi
            success_list.append(True)
            num_obj += 1

    except:  # if any mol fails the whole try except will break
        return False, 'Failed computing RDKit descriptors for molecule' + str(num_obj+1) + 'in file ' + ifile

    if num_obj < est_obj:
        # if some molecules failed to compute we will clean xmatrix by 
        # removing extra rows
        for i in range (num_obj,est_obj):
            xmatrix = np.delete(xmatrix,num_obj,axis=0)

    LOG.debug(f'computed RDKit descriptors matrix with shape {np.shape(xmatrix)}')
    if num_obj == 0:
        return False, 'Unable to compute RDKit descriptors for molecule '+ifile
    
    results = {
        'matrix': xmatrix,
        'names': nms,
        'success_arr': success_list
    }

    return True, results


def _RDKit_properties(ifile, **kwargs):
    ''' 
    computes RDKit properties for the file provided as argument

    output is a boolean and a tupla with the xmatrix and the variable names
    '''
    try:
        suppl = Chem.SDMolSupplier(ifile)
    except Exception as e:
        LOG.error(f'Unable to create supplier with exception {e}')
        return False, 'unable to create supplier'

    LOG.info('Computing RDKit properties...')

    properties = rdMolDescriptors.Properties()

    # get from here num of properties
    md_name = [prop_name for prop_name in properties.GetPropertyNames()]

    #print (md_name)

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

            if mol.GetNumHeavyAtoms() == 0:
                LOG.error('Empty molecule'
                          f'#{num_obj+1} in {ifile}')
                success_list.append(False)
                continue

            descriptors = properties.ComputeProperties(mol)

            if np.isnan(descriptors).any() or np.isinf(descriptors).any():
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
        # if some molecules failed to compute we will clean xmatrix by 
        # removing extra rows
        for i in range (num_obj,est_obj):
            xmatrix = np.delete(xmatrix,num_obj,axis=0)

    LOG.debug(f'computed RDKit properties matrix with shape {np.shape(xmatrix)}')
    if num_obj == 0:
        return False, 'Unable to compute RDKit properties for molecule '+ifile

    results = {
        'matrix': xmatrix,
        'names': md_name,
        'success_arr': success_list
    }
    return True, results
