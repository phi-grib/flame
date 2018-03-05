#! -*- coding: utf-8 -*-

##    Description    SDFile tools class
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
            
import os, math
from rdkit import Chem

def count_mols (ifile):
    suppl = Chem.SDMolSupplier(ifile)
    
    return len(suppl)

def split_SDFile (ifile, numCPUs):
    """     
    Splits the input SDfile in num_chunks SDfiles files of balanced size. Every file is named "filename_0.sdf", "filename_1.sdf", ...
    
    Argument:
        ifile       : input SDfile
        numCPUs     : number of available CPUs

    Output:
        list of split file names
    """

    # Count number of molecules in input file
    suppl = Chem.SDMolSupplier(ifile)
    num_mols = len(suppl)

    if num_mols == 0:
        raise Exception('No molecule found in file: '+ifile)

    if numCPUs > 1 :
        # Get number of molecules per split file
        print ('multi CPU')
        chunk_size = math.ceil(num_mols / numCPUs)

        filename, fileext = os.path.splitext(ifile)
        temp_files = []
        num_mols_in_temp_files = []
        mol_i = 0  # General counter of molecule within the input file
        for chunk_i in range(numCPUs-1):
            # Generate all the split files except the last one
            # The last file will not contain 'chunk_size' molecules,
            # but could contain less. Avoid checking end of fileby handling 
            # the last file separately
            chunk_name = '{}_{}{}'.format(filename, chunk_i, fileext)
            temp_files.append(chunk_name)
            num_mols_in_temp_files.append(chunk_size)
            with open (chunk_name, 'w') as fo:
                for i in range(chunk_size):
                    buffer = suppl.GetItemText(mol_i)
                    fo.write(buffer)
                    mol_i += 1

        # Now put the remaining molecules into the last split file
        chunk_i += 1
        chunk_name = '{}_{}{}'.format(filename, chunk_i, fileext)
        temp_files.append(chunk_name)
        num_mols_in_last_file = 0
        with open (chunk_name, 'w') as fo:
            for mol_i in range(mol_i, num_mols):
                buffer = suppl.GetItemText(mol_i)
                fo.write(buffer)
                num_mols_in_last_file += 1
        num_mols_in_temp_files.append(chunk_size)

    else:
        # If only one CPU, the output will be only the original file
        print ('single CPU')
        temp_files = [ifile]
        num_mols_in_temp_files = [num_mols]

    return (num_mols_in_temp_files, temp_files)

def getNameFromEmpty(suppl, count=1, field=None):

    molText = suppl.GetItemText(count)
    name = ''
    if field is not None:
        fieldName = '> <%s>' %field
        found = False
        for line in molText.split('\n'):
            if line.rstrip() == fieldName:
                found = True
                continue
            if found:
                name = line.rstrip()
                break
    else:
        name = molText.split('\n')[0].rstrip()
        
    if name == '':
        name = 'mol%0.10d'%(count)

    if ' ' in name:
        name = name.replace(' ','_')

    return name

def getName(mol, count=1, field=None, suppl= None):

    if not mol and suppl:
        # The molecule object is empty but it comes from an 
        # SD file and the suppl is provided
        name = getNameFromEmpty(suppl, count, field)
    else:
        name = ''

        if field and mol.HasProp (field):
            name = mol.GetProp(field)
        else:
            name = mol.GetProp('_Name')
            
        if name == '':
            name = 'mol%0.10d'%count

        if ' ' in name:
            name = name.replace(' ','_')

    return name