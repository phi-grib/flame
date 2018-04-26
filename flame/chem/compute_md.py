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

    #print (len(md_nam), md_nam)

    xmatrix = np.zeros ((len(suppl),len(md_nam)),dtype=np.float64)

    try:
        for i,mol in enumerate(suppl): 
            xmatrix [i] = properties.ComputeProperties(mol)

            ##### DEBUG 
            if xmatrix[i][0]>400.0:
                print ('DEBUG****')
                return False, 'error in compute properties' 

    except:
        return False, 'unable to compute RDKit properties'

    return True, (xmatrix, md_nam)

def _RDKit_descriptors (ifile):
    try:
        suppl=Chem.SDMolSupplier(ifile)
    except:
        return False, 'unable to compute RDKit MD'

    nms=[x[0] for x in Descriptors._descList]

    md = MoleculeDescriptors.MolecularDescriptorCalculator(nms)

    #print(len(nms), nms)

    xmatrix = np.zeros ((len(suppl),len(nms)),dtype=np.float64)

    for i,mol in enumerate(suppl):      
        xmatrix [i] = md.CalcDescriptors(mol) 

    return True, (xmatrix, nms)