#! -*- coding: utf-8 -*-

# Description    SDFile tools class
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
from rdkit import Chem


def count_mols(ifile):
    return len(Chem.SDMolSupplier(ifile))


def split_SDFile(ifile, num_chunks):
    """
    Splits the input SDfile in pieces SDfiles files of balanced size.
    Every file is named "filename_0.sdf", "filename_1.sdf", ...

    Argument:
        ifile       : input SDfile
        num_chunks  : number of pieces

    Output:
        list of split file names
        list of number of molecules within each file
    """

    # Count number of molecules in input file
    suppl = Chem.SDMolSupplier(ifile)

    num_mols = len(suppl)

    # Inital checking for early return
    if num_mols == 0:
        return False, 'No molecule found in file: '+ifile

    if num_chunks < 2:
        # If only one CPU, the output will be only the original file
        return True, ([ifile], [num_mols])

    # Get number of molecules per chunks
    chunk_size = num_mols // num_chunks

    filename, fileext = os.path.splitext(ifile)
    temp_files_name = []
    temp_files_size = []

    chunk_i = 0  # chunk counter

    # Chunk i initialization
    chunk_mol_i = 0  # counter of molecules within each chunk
    chunk_name = '{}_{}{}'.format(filename, chunk_i, fileext)
    # force \n to avoid using CR-LF in Windows
    fo = open(chunk_name, 'w', newline='\n')
    temp_files_name.append(chunk_name)

    for mi in range(num_mols):

        if (mi//chunk_size) > chunk_i and chunk_i < (num_chunks - 1):
            # Terminate chunk i
            fo.close()
            temp_files_size.append(chunk_mol_i)

            chunk_i += 1

            # Chunk i initialization
            chunk_mol_i = 0  # counter of molecules within each chunk
            chunk_name = '{}_{}{}'.format(filename, chunk_i, fileext)
            # force \n to avoid using CR-LF in Windows
            fo = open(chunk_name, 'w', newline='\n')
            temp_files_name.append(chunk_name)

        # write the mol content in the output file
        fo.write(suppl.GetItemText(mi))
        chunk_mol_i += 1

    # Terminate chunk i
    fo.close()
    temp_files_size.append(chunk_mol_i)

    return True, (temp_files_name, temp_files_size)


def getNameFromEmpty(suppl, count=1, field=None):

    molText = suppl.GetItemText(count)
    name = ''
    if field is not None:
        fieldName = '> <%s>' % field
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
        name = 'mol%0.10d' % (count)

    if ' ' in name:
        name = name.replace(' ', '_')

    return name


def getName(mol, count=1, field=None, suppl=None):

    if not mol and suppl:
        # The molecule object is empty but it comes from an
        # SD file and the suppl is provided
        name = getNameFromEmpty(suppl, count, field)
    else:
        name = ''

        candidates = []
        if field:
            if isinstance(field, list):
                candidates = field
            elif isinstance(field, str):
                candidates = [field]

        candidates.append('_Name')

        for iname in candidates:
            if mol.HasProp(iname):
                name = mol.GetProp(iname)
                break

        if name == '':
            name = 'mol%0.10d' % count

        if ' ' in name:
            name = name.replace(' ', '_')

    return name
