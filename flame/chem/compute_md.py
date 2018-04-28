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

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors

def _RDKit_properties (ifile):
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