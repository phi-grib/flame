# -*- coding: utf-8 -*-

##    Description    eTOXlab model template
##                   
##    Authors:       Manuel Pastor (manuel.pastor@upf.edu) 
##
##    Copyright 2013 Manuel Pastor
##
##    This file is part of eTOXlab.
##
##    eTOXlab is free software: you can redistribute it and/or modify
##    it under the terms of the GNU General Public License as published by
##    the Free Software Foundation version 3.
##
##    eTOXlab is distributed in the hope that it will be useful,
##    but WITHOUT ANY WARRANTY; without even the implied warranty of
##    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
##    GNU General Public License for more details.
##
##    You should have received a copy of the GNU General Public License
##    along with eTOXlab.  If not, see <http://www.gnu.org/licenses/>.

from rdkit.Chem import Descriptors
from rdkit import Chem


def computeLogP (mol, logpLabel = ''):

    # open the file
    try:
        suppl = Chem.SDMolSupplier(mol)
    except:
        return (False, 'unable to open molfile')
    
    mi = suppl.next()

    if mi is None:
        return (False, 'wrong input format')


    lp = []

    # check if the file contains this property embeeded. Note that
    # the logpLabel must be defined first as metadata
    if logpLabel != '':
        if mi.HasProp (logpLabel):
            try:
                lp.append ( float ( mi.GetProp(logpLabel) ) )
            except:
                pass
            
            if len(lp) > 0 :
                return (True, lp)

    # Compute LogP using RDKit
    try:
        lp.append(Descriptors.MolLogP(mi))
    except:
        return (False, 'unable to compute RDKit logP')

    if len(lp) == 0:
        return (False,'error in logP computation')
    else:
        return (True, lp)  
