#! -*- coding: utf-8 -*-

##    Description    Flame compute molecular descriptor functions
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
import shutil
import tempfile
import requests
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors


def _RDKit_properties (ifile):
    ''' 
    computes RDKit properties for the file provided as argument
    
    output is a boolean and a tupla with the xmatrix and the variable names

    '''
    try:
        suppl=Chem.SDMolSupplier(ifile)
    except:
        return False, 'unable to compute RDKit properties'

    properties = rdMolDescriptors.Properties()

    md_nam = []

    for nam in properties.GetPropertyNames():
        md_nam.append(nam)

    try:
        num_obj = 0
        for mol in suppl: 
            if mol is None:
                print ('ERROR: (@_RDKit_properties) Unable to process molecule #',str(num_obj+1), 'in file '+ ifile)
                continue      
            #xmatrix [num_obj] = properties.ComputeProperties(mol)
            if num_obj == 0:
                xmatrix = properties.ComputeProperties(mol)
            else:
                xmatrix = np.vstack ((xmatrix,properties.ComputeProperties(mol)))

            # ##### DEBUG
            # if properties.ComputeProperties(mol)[0]>400.0:
            #     print ('**** simulated error for DEBUG in compute_md.py ****')
            #     return False, 'Failed compute RDKit properties' 
            ##### REMOVE!!!!
            num_obj += 1 

    except:
        return False, 'Failed computing RDKit properties for molecule' + str(num_obj+1) + 'in file '+ ifile

    if num_obj == 0:
        return False, 'Unable to compute RDKit properties for molecule '+ifile

    return True, (xmatrix, md_nam)


def _RDKit_descriptors (ifile):
    ''' 
    computes RDKit descriptors for the file provided as argument
    
    output is a boolean and a tupla with the xmatrix and the variable names
    
    '''
    try:
        suppl=Chem.SDMolSupplier(ifile)
    except:
        return False, 'Unable to compute RDKit MD'

    nms=[x[0] for x in Descriptors._descList]

    md = MoleculeDescriptors.MolecularDescriptorCalculator(nms)

    try:
        num_obj = 0
        for mol in suppl:
            if mol is None:
                print ('ERROR: (@_RDKit_descriptors) Unable to process molecule #',str(num_obj+1), 'in file '+ ifile)
                continue      
            
            if num_obj == 0:
                xmatrix = md.CalcDescriptors(mol)
            else:
                xmatrix = np.vstack ((xmatrix,md.CalcDescriptors(mol)))

            num_obj += 1 
    
    except:
        return False, 'Failed computing RDKit descriptors for molecule' + str(num_obj+1) + 'in file '+ ifile

    return True, (xmatrix, nms)


def _padel_descriptors (ifile):
    ''' 
    computes Padel molecular descriptors calling an external web service for the file provided as argument
    
    output is a boolean and a tupla with the xmatrix and the variable names
    
    '''

    # TODO: this cannot be hardcoded! maybe read from the component registry?
    uri = "http://localhost:5000/padel/api/v0.1/calc/json"

    tmpdir = os.path.abspath(tempfile.mkdtemp(dir='./'))
    shutil.copy(ifile,tmpdir)

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

    ## DEBUG only
    print('padel service results : ', req.json())

    results = req.json()

    if not results['success']:
        return False, 'padel service returned error condition'

    ofile = os.path.join(tmpdir,results['filename'])

    if not os.path.isfile(ofile):
        return False, 'padel service returned no file'

    with open(ofile,'r') as of:
        index = 0
        var_nam = []
        xmatrix = None

        for line in of:

            if index==0 :  # we asume that the first row contains var names
                var_nam = line.strip().split(',')
                var_nam = var_nam[1:]
            else:

                value_list = line.strip().split(',')
                errors = False

                try:
                    nvalue_list = [float(x) for x in value_list[1:] ]
                except:
                    errors = True

                # value_list = value_list[1:]
                # nvalue_list = []
                # for i in range(len(value_list)):
                #     try:
                #         v = float(value_list[i])
                #     except:
                #         print ('error in object: ',index,' md: ', i)
                #         errors = True
                #         v = 99.999
                #     nvalue_list.append(v)
                
                ## TODO: send back a list of True/False indicating if the xmatrix contains
                ## MDs for all the molecues. As it is now, the size of the object information
                ## and the xmatrix might disagree

                if errors:
                    return False, 'ERROR in Padel results parsing for object '+str(index+1)
                    
                md = np.array(nvalue_list, dtype=np.float64)

                # md = np.nan_to_num(md)
                # # detected a rare bug producing extremely large PaDel descriptors (>1.0e300), leading to overflows
                # # apply a conservative top cutoff of 1.0e10
                # md [ md > 1.0e10 ] = 1.0e10

                if index==1:  # for the fist row, just copy the value list to the xmatrix
                    xmatrix = md
                else:
                    xmatrix = np.vstack((xmatrix, md))

            index+=1

    shutil.rmtree (tmpdir)

    return True, (xmatrix, var_nam)