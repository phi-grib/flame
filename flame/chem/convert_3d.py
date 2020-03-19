#! -*- coding: utf-8 -*-

# Description    Flame compute molecular 3D coordinates functions
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
from rdkit.Chem import AllChem

from flame.util import get_logger

LOG = get_logger(__name__)

def _ETKDG(ifile) -> (bool, str):
    """ Assigns 3D structures to the molecular structures provided as input.
    """

    LOG.info('Converting to ETKDG 3D structures')

    try:
        suppl = Chem.SDMolSupplier(ifile)
    except Exception as e:
        LOG.error(f'Unable to create supplier with exception {e}')
        return False, f'Unable to create supplier with exception {e}'

    success_list = []

    filename, fileext = os.path.splitext(ifile)
    ofile = filename + '_3d' + fileext
    LOG.debug(f'3D stucture ouput file is: {ofile}')

    with open(ofile, 'w') as fo:

        mcount = 0  # used only for the error msg
        for mol in suppl:
            if mol is None:
                LOG.debug('Supplier failed to read'
                            f' molecule #{mcount+1} in {ifile}')
                success_list.append(False)
                continue

            try:
                mol3 = Chem.AddHs(mol)
            except Exception as e:
                LOG.error(f'Failed to add H molecule #{mcount+1} in {ifile}'
                            f' with error {e}')
                success_list.append(False)
                mcount += 1
                continue

            try:
                AllChem.EmbedMolecule(mol3, AllChem.ETKDG())
            except Exception as e:
                LOG.error('Failed to generate 3D structures using'
                            f' ETKDG method for molecule #{mcount+1} in {ifile}'
                            f' with error {e}')
                success_list.append(False)
                mcount += 1
                continue

            fo.write(Chem.MolToMolBlock(mol3))
            fo.write('\n$$$$\n')  # end of mol
            
            success_list.append(True)
            mcount += 1

    return success_list, ofile
